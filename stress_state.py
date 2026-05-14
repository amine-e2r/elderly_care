"""
stress_state.py
===============
Production-grade realtime post-processing layer for the stress classifier.

Pipeline (per 5 s model tick):
    p_raw → quality gate → median spike filter → EMA smoothing
          → stress-load accumulator → state machine → notification dispatcher

Decouples noisy per-window probabilities from user-facing alerts:
  • short spikes are absorbed by the median + EMA
  • hysteresis (asymmetric thresholds) prevents ping-pong
  • dwell timers require persistence before any state change
  • cooldown / quiet hours / per-hour cap suppress notification spam
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median


class StressState(str, Enum):
    CALM     = "CALM"
    ELEVATED = "ELEVATED"
    STRESS   = "STRESS"
    RECOVERY = "RECOVERY"


@dataclass
class StressConfig:
    # cadence
    dt: float = 5.0                  # seconds between model outputs

    # quality gate
    min_rr: int = 8
    max_vm_std: float = 200.0
    bad_activities: tuple = ("ACTIVE",)
    qc_freeze_after_sec: float = 30.0

    # smoothing
    tau_fast: float = 45.0           # decision EMA time constant
    tau_slow: float = 120.0          # display EMA time constant
    median_window: int = 3

    # hysteresis thresholds (on smoothed probability)
    enter_thr: float = 0.70
    confirm_thr: float = 0.65
    exit_thr: float = 0.45

    # dwell (seconds of consecutive eligibility)
    dwell_enter: float = 60.0        # CALM -> ELEVATED
    dwell_confirm: float = 90.0      # ELEVATED -> STRESS
    dwell_exit_stress: float = 120.0 # STRESS -> RECOVERY
    dwell_exit_elevated: float = 60.0
    dwell_recovery: float = 60.0     # RECOVERY -> CALM

    # stress load (0..100)
    load_max: float = 100.0
    load_gain_up: float = 1.0
    load_gain_down: float = 0.7
    load_dead_lo: float = 0.45
    load_dead_hi: float = 0.65
    load_enter_elevated: float = 40.0
    load_enter_stress: float = 70.0
    load_exit_elevated: float = 15.0
    load_exit_stress: float = 25.0
    load_exit_recovery: float = 10.0
    reflare_load: float = 60.0

    # notifications
    cooldown_sec: float = 15 * 60
    per_hour_cap: int = 2
    per_day_cap: int = 6
    quiet_hours: tuple | None = (22, 7)   # (start_h, end_h) or None to disable

    # seed
    init_p: float = 0.25             # population prior for EMAs at startup


@dataclass
class StressOutput:
    ts: float
    state: StressState
    p_raw: float
    p_smooth: float                  # τ_fast EMA — used for decisions
    stress_score: int                # 0..100 from τ_slow EMA — for UI
    load: float                      # 0..100 stress budget
    qc_pass: bool
    notify: bool                     # dispatcher decided to fire
    notify_kind: str = ""            # "STRESS_ONSET" | ""
    reason: str = ""


class StressStateManager:
    def __init__(self, cfg: StressConfig | None = None):
        self.cfg = cfg or StressConfig()
        c = self.cfg

        self._p_smooth      = c.init_p
        self._p_smooth_slow = c.init_p
        self._alpha_fast = 1 - math.exp(-c.dt / c.tau_fast)
        self._alpha_slow = 1 - math.exp(-c.dt / c.tau_slow)

        self._median_buf: deque = deque(maxlen=c.median_window)
        self._load: float = 0.0

        self._state: StressState = StressState.CALM
        self._state_since: float = time.time()

        self._dwell: dict[str, float] = {
            "enter": 0.0, "confirm": 0.0,
            "exit_stress": 0.0, "exit_elevated": 0.0, "recovery": 0.0,
        }

        self._last_good_ts: float | None = None

        self._notif_history: deque = deque()
        self._last_notif: dict[str, float] = {}

    # ── public API ────────────────────────────────────────────────
    def update(self, *,
               p_raw: float,
               n_rr: int,
               activity: str,
               vm_std: float,
               now: float | None = None) -> StressOutput:
        now = now if now is not None else time.time()
        c = self.cfg

        # 1. QC
        qc_pass = (
            n_rr >= c.min_rr
            and activity not in c.bad_activities
            and vm_std < c.max_vm_std
            and isinstance(p_raw, (int, float))
            and math.isfinite(float(p_raw))
        )

        if not qc_pass:
            stale = (self._last_good_ts is not None
                     and now - self._last_good_ts > c.qc_freeze_after_sec)
            if stale:
                for k in self._dwell:
                    self._dwell[k] = 0.0
            return StressOutput(
                ts=now, state=self._state,
                p_raw=float(p_raw) if isinstance(p_raw, (int, float)) else 0.0,
                p_smooth=self._p_smooth,
                stress_score=int(round(self._p_smooth_slow * 100)),
                load=self._load, qc_pass=False, notify=False,
                reason="qc_fail",
            )
        self._last_good_ts = now

        # 2. median spike filter
        self._median_buf.append(float(p_raw))
        p_med = median(self._median_buf)

        # 3. EMAs
        self._p_smooth      += self._alpha_fast * (p_med - self._p_smooth)
        self._p_smooth_slow += self._alpha_slow * (p_med - self._p_smooth_slow)
        ps = self._p_smooth

        # 4. stress load (leaky integrator)
        if ps >= c.load_dead_hi:
            self._load += c.load_gain_up * (ps - c.load_dead_hi) / (1 - c.load_dead_hi) * c.dt
        elif ps <= c.load_dead_lo:
            self._load -= c.load_gain_down * (c.load_dead_lo - ps) / c.load_dead_lo * c.dt
        self._load = max(0.0, min(c.load_max, self._load))

        # 5. dwell accumulators (consecutive-eligible time only)
        self._tick_dwell("enter",         ps >= c.enter_thr)
        self._tick_dwell("confirm",       ps >= c.confirm_thr)
        self._tick_dwell("exit_stress",   ps <= c.exit_thr)
        self._tick_dwell("exit_elevated", ps <= c.exit_thr)
        self._tick_dwell("recovery",      self._load <= c.load_exit_recovery)

        # 6. state transitions
        notify, notif_kind, reason = False, "", ""
        new_state = self._state

        if self._state == StressState.CALM:
            if (self._dwell["enter"] >= c.dwell_enter
                    and self._load >= c.load_enter_elevated):
                new_state = StressState.ELEVATED
                reason = f"calm→elevated (p={ps:.2f}, load={self._load:.0f})"

        elif self._state == StressState.ELEVATED:
            if (self._load >= c.load_enter_stress
                    and self._dwell["confirm"] >= c.dwell_confirm):
                new_state = StressState.STRESS
                reason = f"stress onset (p={ps:.2f}, load={self._load:.0f})"
                if self._allow_notify(now, "STRESS_ONSET"):
                    notify, notif_kind = True, "STRESS_ONSET"
                    self._record_notif(now, "STRESS_ONSET")
            elif (self._dwell["exit_elevated"] >= c.dwell_exit_elevated
                    and self._load <= c.load_exit_elevated):
                new_state = StressState.CALM
                reason = "elevated→calm"

        elif self._state == StressState.STRESS:
            if (self._dwell["exit_stress"] >= c.dwell_exit_stress
                    and self._load <= c.load_exit_stress):
                new_state = StressState.RECOVERY
                reason = "stress→recovery"

        elif self._state == StressState.RECOVERY:
            if (self._load >= c.reflare_load
                    and self._dwell["confirm"] >= c.dwell_enter):
                new_state = StressState.STRESS
                reason = "re-flare (no new notif)"
            elif self._dwell["recovery"] >= c.dwell_recovery:
                new_state = StressState.CALM
                reason = "recovered"

        if new_state != self._state:
            self._state = new_state
            self._state_since = now
            # reset the "entering" dwell counters so a new state must build
            # its own entry evidence; leave exit counters alone
            self._dwell["enter"] = 0.0
            self._dwell["confirm"] = 0.0

        return StressOutput(
            ts=now, state=self._state, p_raw=float(p_raw), p_smooth=ps,
            stress_score=int(round(self._p_smooth_slow * 100)),
            load=self._load, qc_pass=True, notify=notify,
            notify_kind=notif_kind, reason=reason,
        )

    # ── helpers ───────────────────────────────────────────────────
    def _tick_dwell(self, key: str, eligible: bool):
        self._dwell[key] = self._dwell[key] + self.cfg.dt if eligible else 0.0

    def _allow_notify(self, now: float, kind: str) -> bool:
        c = self.cfg
        if self._in_quiet_hours(now):
            return False
        if now - self._last_notif.get(kind, 0.0) < c.cooldown_sec:
            return False
        while self._notif_history and now - self._notif_history[0][0] > 86400:
            self._notif_history.popleft()
        h = sum(1 for t, k in self._notif_history if k == kind and now - t < 3600)
        d = sum(1 for t, k in self._notif_history if k == kind)
        return h < c.per_hour_cap and d < c.per_day_cap

    def _record_notif(self, now: float, kind: str):
        self._last_notif[kind] = now
        self._notif_history.append((now, kind))

    def _in_quiet_hours(self, now: float) -> bool:
        if not self.cfg.quiet_hours:
            return False
        s, e = self.cfg.quiet_hours
        h = time.localtime(now).tm_hour
        return (s <= h or h < e) if s > e else (s <= h < e)
