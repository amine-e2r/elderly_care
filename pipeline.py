"""
pipeline.py
===========
Polar H10 realtime pipeline: thu thập ECG (130Hz) + ACC (25Hz) + HR/RR,
trích xuất feature theo cửa sổ và chạy 2 model đã train:
  • random_forest_model.pkl   → STRESS (HRV 5 features / 10 RR)
  • isolation_forest_model.pkl → ANOMALY (HRV 5 + morphological 5 / 10 RR)

Khác bản cũ:
  • Sửa HRV: thêm LF/HF (5 feature) — bản cũ thiếu nên model stress lỗi shape.
  • Sửa StandardScaler: training scaler không được lưu cùng model, nên ta
    FIT scaler trên N cửa sổ đầu của session (baseline của chính người dùng),
    rồi dùng scaler đó cho các cửa sổ sau. Đây là "anomaly so với baseline
    của người dùng", phù hợp giám sát người cao tuổi.
  • Thêm ACC stream + phân loại hoạt động (REST/SLOW_WALK/WALK/ACTIVE)
    + phát hiện ngã (impact + free-fall + bất động) — lấy nguyên logic
    đã được kiểm chứng trong ecg_acc_collector.py.
  • Bandpass 5–40Hz trước khi tìm R-peak (giống script_stress.py).
  • Output đa file CSV trong polar_data/.
"""

import asyncio
import csv
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.preprocessing import StandardScaler

from bleak import BleakClient, BleakScanner

# ════════════════════════════════════════════════════════════════
#  CẤU HÌNH BLE / SAMPLING
# ════════════════════════════════════════════════════════════════
POLAR_ADDRESS = "24:AC:AC:13:EB:86"   # đặt None để auto-scan theo tên
POLAR_DEVICE_NAME = "Polar H10"

HR_UUID     = "00002a37-0000-1000-8000-00805f9b34fb"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA    = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

FS_ECG = 130
FS_ACC = 25

PMD_START_OP = 0x02
PMD_STOP_OP  = 0x03
TYPE_ECG     = 0x00
TYPE_ACC     = 0x02
SETTING_SAMPLE_RATE = 0x00
SETTING_RESOLUTION  = 0x01
SETTING_RANGE       = 0x02


def _pmd_setting(setting_type: int, value: int) -> list[int]:
    return [setting_type, 0x01, value & 0xFF, (value >> 8) & 0xFF]


def build_pmd_start(measurement_type: int,
                    settings: list[tuple[int, int]]) -> bytearray:
    cmd = [PMD_START_OP, measurement_type]
    for st, v in settings:
        cmd.extend(_pmd_setting(st, v))
    return bytearray(cmd)


ECG_START = build_pmd_start(TYPE_ECG, [
    (SETTING_SAMPLE_RATE, FS_ECG),
    (SETTING_RESOLUTION, 14),
])
ACC_START = build_pmd_start(TYPE_ACC, [
    (SETTING_SAMPLE_RATE, FS_ACC),
    (SETTING_RESOLUTION, 16),
    (SETTING_RANGE, 8),
])
ECG_STOP = bytearray([PMD_STOP_OP, TYPE_ECG])
ACC_STOP = bytearray([PMD_STOP_OP, TYPE_ACC])

# ════════════════════════════════════════════════════════════════
#  CỬA SỔ ANALYSIS
# ════════════════════════════════════════════════════════════════
STEP_SEC        = 5      # mỗi 5s đánh giá một lần
WIN_SEC         = 30     # cửa sổ HRV
BUFFER_KEEP_SEC = 600    # giữ 10 phút raw trong RAM
RR_WINDOW       = 10     # 10 RR intervals — KHỚP với training (script_stress.py & script_anomaly.py)

# Bandpass cho ECG trước khi tìm R-peak (giống script_stress.py)
BP_LOW, BP_HIGH = 5, 40

# ════════════════════════════════════════════════════════════════
#  ACTIVITY + FALL (lấy nguyên từ ecg_acc_collector.py — đã kiểm chứng)
# ════════════════════════════════════════════════════════════════
ACTIVITY_HR_THRESHOLDS = {
    "REST"      : (45,  90, 100, 130),
    "SLOW_WALK" : (48, 105, 115, 140),
    "WALK"      : (50, 120, 130, 150),
    "ACTIVE"    : (55, 135, 145, 160),
    "UNKNOWN"   : (45, 100, 110, 135),
}
ACC_VM_STD_THRESHOLDS = {20: "REST", 80: "SLOW_WALK", 300: "WALK"}

FALL_IMPACT_THRESHOLD_MG    = 2000
FALL_FREE_FALL_THRESHOLD_MG = 500
FALL_POST_STILL_STD_MG      = 120
FALL_PROX_S                 = 1.0
FALL_ALERT_COOLDOWN_S       = 15
FALL_STRONG_IMPACT_MG       = 2800

# ════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ════════════════════════════════════════════════════════════════
try:
    stress_model = joblib.load("random_forest_model.pkl")
    print("✓ Loaded random_forest_model.pkl (STRESS)")
except Exception as e:
    stress_model = None
    print(f"⚠ Không load được random_forest_model.pkl: {e}")

try:
    anomaly_model = joblib.load("isolation_forest_model.pkl")
    print("✓ Loaded isolation_forest_model.pkl (ANOMALY)")
except Exception as e:
    anomaly_model = None
    print(f"⚠ Không load được isolation_forest_model.pkl: {e}")

try:
    _anomaly_scaler = joblib.load("anomaly_scaler.pkl")
    print("✓ Loaded anomaly_scaler.pkl (dùng scaler gốc MIT-BIH)")
except Exception:
    _anomaly_scaler = None
    print("⚠ anomaly_scaler.pkl chưa có — sẽ fit scaler trên baseline người dùng")

# Thứ tự feature PHẢI khớp với lúc train.
STRESS_FEATURES  = ["RR_mean", "SDNN", "RMSSD", "pNN50", "LF/HF"]
ANOMALY_FEATURES = STRESS_FEATURES + [
    "mean_amplitude", "std_amplitude", "mean_energy", "std_energy", "mean_qrs_slope"
]

# Scaler runtime: training scaler không được persist cùng model nên
# ta fit lại trên dữ liệu baseline của chính người đeo (warm-up).
ANOMALY_WARMUP_WINDOWS = 20
_anomaly_feature_history: deque = deque(maxlen=200)
_anomaly_scaler: StandardScaler | None = None

# ════════════════════════════════════════════════════════════════
#  BUFFERS
# ════════════════════════════════════════════════════════════════
ecg_buf: list = []   # (ts_unix, ecg_uV)
acc_buf: list = []   # (ts_unix, x, y, z, vm)
rr_buf : list = []   # (ts_unix, rr_ms)
hr_buf : list = []   # (ts_unix, hr_bpm)

buf_lock = threading.Lock()
is_running = True

# ════════════════════════════════════════════════════════════════
#  CSV SETUP
# ════════════════════════════════════════════════════════════════
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path("polar_data"); out_dir.mkdir(exist_ok=True)

ecg_path     = out_dir / f"ecg_{session_id}.csv"
acc_path     = out_dir / f"acc_{session_id}.csv"
predict_path = out_dir / f"predictions_{session_id}.csv"
alert_path   = out_dir / f"alerts_{session_id}.csv"

f_ecg = open(ecg_path,     "w", newline="")
f_acc = open(acc_path,     "w", newline="")
f_pred = open(predict_path, "w", newline="")
f_alert = open(alert_path,  "w", newline="")

w_ecg = csv.writer(f_ecg)
w_acc = csv.writer(f_acc)

PRED_COLS = [
    "timestamp", "n_rr", "hr_bpm",
    "RR_mean", "SDNN", "RMSSD", "pNN50", "LF/HF",
    "mean_amplitude", "std_amplitude", "mean_energy", "std_energy", "mean_qrs_slope",
    "activity", "vm_std",
    "stress_pred", "stress_proba",
    "anomaly_pred", "anomaly_score",
    "fall_candidate", "fall_strong", "fall_peak_vm",
    "hr_context_alert",
]
w_pred = csv.DictWriter(f_pred, fieldnames=PRED_COLS)
w_pred.writeheader()

w_alert = csv.DictWriter(f_alert, fieldnames=[
    "timestamp", "level", "type", "reason", "hr_bpm", "activity"
])
w_alert.writeheader()

w_ecg.writerow(["timestamp_unix", "timestamp_polar_ns", "sample_index", "ecg_uv"])
w_acc.writerow(["timestamp_unix", "timestamp_polar_ns", "sample_index",
                "x_mg", "y_mg", "z_mg", "vm_mg"])
for f in (f_ecg, f_acc, f_pred, f_alert):
    f.flush()


# ════════════════════════════════════════════════════════════════
#  PARSERS
# ════════════════════════════════════════════════════════════════
def parse_ecg_packet(data: bytes) -> tuple[int, list[int]]:
    if len(data) < 10 or (data[0] & 0x3F) != TYPE_ECG:
        return 0, []
    ts_ns = struct.unpack_from("<Q", data, 1)[0]
    samples = []
    offset = 10
    while offset + 2 < len(data):
        b0, b1, b2 = data[offset], data[offset+1], data[offset+2]
        raw = b0 | (b1 << 8) | (b2 << 16)
        if raw & 0x800000:
            raw -= 0x1000000
        samples.append(raw)
        offset += 3
    return ts_ns, samples


def parse_acc_packet(data: bytes) -> tuple[int, list[tuple]]:
    if len(data) < 10 or (data[0] & 0x3F) != TYPE_ACC:
        return 0, []
    ts_ns = struct.unpack_from("<Q", data, 1)[0]
    frame_type = data[9] & 0x7F
    is_compressed = bool(data[9] & 0x80)
    if is_compressed or frame_type not in (0, 1, 2):
        return ts_ns, []
    samples = []
    offset = 10
    sample_size = {0: 1, 1: 2, 2: 3}[frame_type]
    sample_bytes = sample_size * 3
    while offset + sample_bytes <= len(data):
        if sample_size == 1:
            x, y, z = struct.unpack_from("<bbb", data, offset)
        elif sample_size == 2:
            x, y, z = struct.unpack_from("<hhh", data, offset)
        else:
            x = int.from_bytes(data[offset:offset+3],   "little", signed=True)
            y = int.from_bytes(data[offset+3:offset+6], "little", signed=True)
            z = int.from_bytes(data[offset+6:offset+9], "little", signed=True)
        vm = float(np.sqrt(x*x + y*y + z*z))
        samples.append((float(x), float(y), float(z), vm))
        offset += sample_bytes
    return ts_ns, samples


def parse_hr_gatt(data: bytes) -> dict:
    if len(data) < 2:
        return {"hr": 0, "rr": []}
    flags = data[0]; offset = 1
    if flags & 0x01:
        if len(data) < offset + 2:
            return {"hr": 0, "rr": []}
        hr = int.from_bytes(data[offset:offset+2], "little"); offset += 2
    else:
        hr = data[offset]; offset += 1
    if flags & 0x08:
        offset += 2
    rr = []
    if flags & 0x10:
        while offset + 1 < len(data):
            rr.append(round(int.from_bytes(data[offset:offset+2], "little") * 1000 / 1024, 1))
            offset += 2
    return {"hr": hr, "rr": rr}


# ════════════════════════════════════════════════════════════════
#  BLE CALLBACKS
# ════════════════════════════════════════════════════════════════
_pkt_idx_ecg = 0
_pkt_idx_acc = 0
_last_prune  = 0.0


def _drop_old(buf: list, cutoff: float) -> None:
    idx = 0
    n = len(buf)
    while idx < n and buf[idx][0] < cutoff:
        idx += 1
    if idx:
        del buf[:idx]


def prune_runtime_buffers(now: float) -> None:
    cutoff = now - BUFFER_KEEP_SEC
    for buf in (ecg_buf, acc_buf, rr_buf, hr_buf):
        _drop_old(buf, cutoff)


def pmd_callback(sender, data: bytearray):
    global _pkt_idx_ecg, _pkt_idx_acc, _last_prune
    raw = bytes(data)
    if not raw:
        return
    now = time.time()
    data_type = raw[0] & 0x3F

    if data_type == TYPE_ECG:
        ts_ns, samples = parse_ecg_packet(raw)
        if not samples:
            return
        dt_ns = int(1e9 / FS_ECG)
        n = len(samples)
        with buf_lock:
            for i, s in enumerate(samples):
                t_unix = now - (n - 1 - i) / FS_ECG
                ecg_buf.append((t_unix, s))
                w_ecg.writerow([f"{t_unix:.6f}", ts_ns + i * dt_ns, _pkt_idx_ecg, s])
            _pkt_idx_ecg += n
            if now - _last_prune > 5:
                prune_runtime_buffers(now)
                _last_prune = now
        f_ecg.flush()

    elif data_type == TYPE_ACC:
        ts_ns, samples = parse_acc_packet(raw)
        if not samples:
            return
        dt_ns = int(1e9 / FS_ACC)
        n = len(samples)
        with buf_lock:
            for i, (x, y, z, vm) in enumerate(samples):
                t_unix = now - (n - 1 - i) / FS_ACC
                acc_buf.append((t_unix, x, y, z, vm))
                w_acc.writerow([f"{t_unix:.6f}", ts_ns + i * dt_ns,
                                _pkt_idx_acc, x, y, z, round(vm, 2)])
            _pkt_idx_acc += n
            if now - _last_prune > 5:
                prune_runtime_buffers(now)
                _last_prune = now
        f_acc.flush()


def pmd_control_callback(sender, data: bytearray):
    raw = bytes(data)
    if len(raw) >= 4 and raw[0] == 0xF0:
        op   = raw[1]
        meas = raw[2] & 0x3F
        err  = raw[3]
        op_name   = {PMD_START_OP: "START", PMD_STOP_OP: "STOP"}.get(op, f"0x{op:02x}")
        meas_name = {TYPE_ECG: "ECG", TYPE_ACC: "ACC"}.get(meas, f"0x{meas:02x}")
        status    = "SUCCESS" if err == 0 else f"ERROR 0x{err:02x}"
        print(f"  PMD response: {op_name} {meas_name} → {status}")


def hr_callback(sender, data: bytearray):
    global _last_prune
    parsed = parse_hr_gatt(bytes(data))
    now = time.time()
    with buf_lock:
        hr_buf.append((now, parsed["hr"]))
        for rr in parsed["rr"]:
            rr_buf.append((now, rr))
        if now - _last_prune > 5:
            prune_runtime_buffers(now)
            _last_prune = now


# ════════════════════════════════════════════════════════════════
#  PREPROCESS + FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════
def bandpass_filter(x: np.ndarray, fs: int = FS_ECG) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(2, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    return filtfilt(b, a, x)


def detect_r_peaks(ecg_filtered: np.ndarray, fs: int = FS_ECG) -> np.ndarray:
    if len(ecg_filtered) < fs:
        return np.array([], dtype=int)
    prom = max(0.15, 0.5 * float(np.std(ecg_filtered)))
    peaks, _ = find_peaks(
        ecg_filtered,
        distance=int(0.3 * fs),
        prominence=prom,
    )
    return peaks


def hrv_features_from_rr(rr_ms: np.ndarray) -> dict:
    """Khớp ĐÚNG extract_hrv_features() trong script_stress.py / script_anomaly.py."""
    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) < 2:
        return {k: 0.0 for k in STRESS_FEATURES}

    rr_mean = float(np.mean(rr_ms))
    sdnn    = float(np.std(rr_ms))
    diff_rr = np.diff(rr_ms)
    rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
    pnn50   = float(100 * np.sum(np.abs(diff_rr) > 50) / max(len(diff_rr), 1))

    # LF/HF qua FFT — đúng như script_stress.py
    fft_result = np.fft.fft(rr_ms)
    freqs = np.fft.fftfreq(len(rr_ms), d=rr_mean / 1000.0)
    power = np.abs(fft_result) ** 2
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    lf = float(np.sum(power[lf_mask]))
    hf = float(np.sum(power[hf_mask]))
    lf_hf = lf / hf if hf > 0 else 0.0

    return {
        "RR_mean": rr_mean,
        "SDNN"   : sdnn,
        "RMSSD"  : rmssd,
        "pNN50"  : pnn50,
        "LF/HF"  : lf_hf,
    }


def morpho_features_from_beats(beats: list[np.ndarray]) -> dict:
    """Khớp ĐÚNG extract_morphological_features() trong script_anomaly.py."""
    amps, means, energies, slopes = [], [], [], []
    for beat in beats:
        if len(beat) < 2:
            continue
        amps.append(float(np.max(beat) - np.min(beat)))
        means.append(float(np.mean(beat)))
        energies.append(float(np.sum(beat ** 2)))
        slopes.append(float(np.max(np.diff(beat))))

    if not amps:
        return {k: 0.0 for k in ANOMALY_FEATURES[5:]}

    return {
        "mean_amplitude": float(np.mean(amps)),
        "std_amplitude" : float(np.std(amps)),
        "mean_energy"   : float(np.mean(energies)),
        "std_energy"    : float(np.std(energies)),
        "mean_qrs_slope": float(np.mean(slopes)),
    }


def extract_beats_around_peaks(ecg: np.ndarray, peaks: np.ndarray,
                               fs: int = FS_ECG) -> list[np.ndarray]:
    """Cắt ±0.15s quanh mỗi R-peak — khớp với hrv_windows() khi train."""
    half = int(0.15 * fs)
    beats = []
    for p in peaks:
        a = max(0, p - half)
        b = min(len(ecg), p + half)
        if b - a > 1:
            beats.append(ecg[a:b])
    return beats


# ════════════════════════════════════════════════════════════════
#  ACTIVITY + FALL DETECTION
# ════════════════════════════════════════════════════════════════
def classify_activity(vm_std: float) -> str:
    for threshold, label in sorted(ACC_VM_STD_THRESHOLDS.items()):
        if vm_std < threshold:
            return label
    return "ACTIVE"


def detect_fall(vm_array: np.ndarray) -> dict:
    """Quét window cho chữ ký free-fall → impact → bất động."""
    result = {
        "impact_spike": False, "free_fall": False, "post_still": False,
        "fall_candidate": False, "strong_fall": False,
        "peak_vm": 0.0, "min_before_1s": 0.0, "std_after_5s": 0.0,
        "fall_offset_s": -1.0,
    }
    n = len(vm_array)
    if n < FS_ACC * 3:
        return result

    peaks, _ = find_peaks(
        vm_array,
        height=FALL_IMPACT_THRESHOLD_MG,
        distance=max(1, int(FS_ACC * 0.6)),
    )
    if len(peaks) == 0:
        return result

    prox = max(1, int(FS_ACC * FALL_PROX_S))
    after_n   = int(FS_ACC * 3)
    min_after = int(FS_ACC * 2)

    best_score = -1
    best = None
    for pi in peaks:
        before = vm_array[max(0, pi - prox): pi]
        if len(before) == 0:
            continue
        min_before = float(np.min(before))
        free_fall = min_before < FALL_FREE_FALL_THRESHOLD_MG

        after = vm_array[pi + 1: pi + 1 + after_n]
        post_still = False
        std_after = 0.0
        if len(after) >= min_after:
            std_after = float(np.std(after))
            post_still = std_after < FALL_POST_STILL_STD_MG

        score = (2 if free_fall else 0) + (1 if post_still else 0)
        if score > best_score or (
            score == best_score and (best is None or vm_array[pi] > vm_array[best[0]])
        ):
            best_score = score
            best = (int(pi), free_fall, post_still, min_before, std_after)

    if best is None:
        return result
    pi, free_fall, post_still, min_before, std_after = best
    peak_vm = float(vm_array[pi])
    result.update({
        "impact_spike"  : True,
        "peak_vm"       : round(peak_vm, 2),
        "free_fall"     : bool(free_fall),
        "post_still"    : bool(post_still),
        "min_before_1s" : round(min_before, 2),
        "std_after_5s"  : round(std_after, 2),
        "fall_offset_s" : round(pi / FS_ACC, 2),
        "fall_candidate": bool(free_fall),
        "strong_fall"   : bool(free_fall and (post_still or peak_vm > FALL_STRONG_IMPACT_MG)),
    })
    return result


def context_hr_alert(hr_mean: float, activity: str) -> tuple[str, str]:
    lo, hi_w, hi_a, hi_c = ACTIVITY_HR_THRESHOLDS.get(activity,
                                                     ACTIVITY_HR_THRESHOLDS["UNKNOWN"])
    if hr_mean >= hi_c:
        return "CRITICAL", f"HR={hr_mean:.0f} nguy hiểm khi {activity} (>{hi_c})"
    if hr_mean >= hi_a:
        return "ALERT",    f"HR={hr_mean:.0f} cao bất thường khi {activity} (>{hi_a})"
    if hr_mean >= hi_w:
        return "WATCH",    f"HR={hr_mean:.0f} hơi cao khi {activity} (>{hi_w})"
    if hr_mean < lo:
        return "ALERT",    f"HR={hr_mean:.0f} thấp bất thường khi {activity} (<{lo})"
    return "SAFE", ""


# ════════════════════════════════════════════════════════════════
#  ALERT FIRING
# ════════════════════════════════════════════════════════════════
_alert_count = 0
_last_fall_alert_ts = 0.0
_last_fall_event_ts = 0.0


def _fire_alert(level: str, alert_type: str, reason: str,
                hr_bpm: float = 0, activity: str = ""):
    global _alert_count
    _alert_count += 1
    color = {
        "CRITICAL": "\033[1;97;41m",
        "ALERT"   : "\033[1;91m",
        "WATCH"   : "\033[1;93m",
        "SAFE"    : "\033[1;92m",
    }.get(level, "\033[0m")
    reset = "\033[0m"
    bar = "█" * 60

    if level in ("CRITICAL", "ALERT"):
        sys.stderr.write(
            f"\n{color}{bar}{reset}\n"
            f"{color}  🚨 ALERT #{_alert_count} [{level}] {alert_type}{reset}\n"
            f"  {reason}\n"
            f"{color}{bar}{reset}\n\a"
        )
        sys.stderr.flush()

    w_alert.writerow({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level"    : level,
        "type"     : alert_type,
        "reason"   : reason,
        "hr_bpm"   : hr_bpm,
        "activity" : activity,
    })
    f_alert.flush()


# ════════════════════════════════════════════════════════════════
#  PREDICTION PIPELINE — chạy mỗi STEP_SEC
# ════════════════════════════════════════════════════════════════
last_row: dict = {}


def analyze_window() -> dict:
    """Lấy WIN_SEC giây gần nhất → bandpass → tìm R-peak → 10 RR cuối → predict."""
    global _anomaly_scaler, _last_fall_alert_ts, _last_fall_event_ts

    now = time.time()
    cutoff = now - WIN_SEC

    with buf_lock:
        ecg_window = np.array([s for ts, s in ecg_buf if ts >= cutoff], dtype=float)
        acc_vm     = np.array([vm for ts, _, _, _, vm in acc_buf if ts >= cutoff])
        hr_latest  = hr_buf[-1][1] if hr_buf else 0

    row = {k: "" for k in PRED_COLS}
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["hr_bpm"]    = hr_latest

    # ── ACC / activity / fall ────────────────────────────────
    activity = "UNKNOWN"
    fall = {"fall_candidate": False, "strong_fall": False, "peak_vm": 0.0,
            "free_fall": False, "post_still": False, "fall_offset_s": -1.0,
            "min_before_1s": 0.0, "std_after_5s": 0.0}
    vm_std = 0.0
    if len(acc_vm) >= 5:
        vm_std   = float(np.std(acc_vm))
        activity = classify_activity(vm_std)
        fall     = detect_fall(acc_vm)
    row["activity"]        = activity
    row["vm_std"]          = round(vm_std, 2)
    row["fall_candidate"]  = int(fall["fall_candidate"])
    row["fall_strong"]     = int(fall["strong_fall"])
    row["fall_peak_vm"]    = fall["peak_vm"]

    # Fire fall alert (cooldown + dedup giữa các overlapping windows)
    if fall["fall_candidate"]:
        event_ts    = (now - WIN_SEC) + max(0.0, fall["fall_offset_s"])
        same_event  = abs(event_ts - _last_fall_event_ts) <= 3.0
        in_cooldown = (now - _last_fall_alert_ts) < FALL_ALERT_COOLDOWN_S
        if not (same_event or in_cooldown):
            _last_fall_alert_ts = now
            _last_fall_event_ts = event_ts
            level  = "CRITICAL" if fall["strong_fall"] else "ALERT"
            reason = (f"Impact {fall['peak_vm']:.0f} mG | "
                      f"free-fall {fall['min_before_1s']:.0f} mG | "
                      f"std sau {fall['std_after_5s']:.0f} mG")
            _fire_alert(level, "FALL", reason, hr_latest, activity)

    # ── ECG → RR → features ──────────────────────────────────
    n_rr = 0
    if len(ecg_window) >= FS_ECG * 5:        # cần ≥5s ECG
        try:
            ecg_filt = bandpass_filter(ecg_window)
        except Exception:
            ecg_filt = ecg_window
        peaks = detect_r_peaks(ecg_filt)
        if len(peaks) >= RR_WINDOW + 1:
            rr_ms = np.diff(peaks) * 1000.0 / FS_ECG
            # Lấy RR_WINDOW cuối cùng — khớp với window kích thước 10 lúc train
            rr_last = rr_ms[-RR_WINDOW:]
            peak_last = peaks[-(RR_WINDOW + 1):]   # +1 để có beat đầu

            n_rr = len(rr_last)
            row["n_rr"] = n_rr

            hrv = hrv_features_from_rr(rr_last)
            beats = extract_beats_around_peaks(ecg_filt, peak_last)
            morpho = morpho_features_from_beats(beats)

            for k, v in {**hrv, **morpho}.items():
                row[k] = round(v, 4)

            # ── STRESS ────────────────────────────────────
            if stress_model is not None:
                try:
                    x_stress = np.array([[hrv[k] for k in STRESS_FEATURES]])
                    pred = int(stress_model.predict(x_stress)[0])
                    proba = ""
                    if hasattr(stress_model, "predict_proba"):
                        proba = round(float(stress_model.predict_proba(x_stress)[0, 1]), 3)
                    row["stress_pred"]  = pred
                    row["stress_proba"] = proba
                    if pred == 1:
                        _fire_alert("ALERT", "STRESS",
                                    f"Random Forest gắn cờ STRESS (proba={proba})",
                                    hr_latest, activity)
                except Exception as e:
                    row["stress_pred"] = f"err:{e}"

            # ── ANOMALY ───────────────────────────────────
            if anomaly_model is not None:
                try:
                    feat_vec = np.array([hrv[k] if k in hrv else morpho[k]
                                         for k in ANOMALY_FEATURES])
                    _anomaly_feature_history.append(feat_vec)

                    # Dùng scaler gốc MIT-BIH nếu có, hoặc fit trên baseline người dùng.
                    if _anomaly_scaler is None:
                        if len(_anomaly_feature_history) >= ANOMALY_WARMUP_WINDOWS:
                            if len(_anomaly_feature_history) % 30 == 0 or \
                                    len(_anomaly_feature_history) == ANOMALY_WARMUP_WINDOWS:
                                _anomaly_scaler = StandardScaler().fit(
                                    np.array(_anomaly_feature_history)
                                )

                    if _anomaly_scaler is not None:
                        x_scaled = _anomaly_scaler.transform(feat_vec.reshape(1, -1))
                        pred = int(anomaly_model.predict(x_scaled)[0])  # 1 normal / -1 anomaly
                        score = float(anomaly_model.decision_function(x_scaled)[0])
                        is_anomaly = (pred == -1)
                        row["anomaly_pred"]  = 1 if is_anomaly else 0
                        row["anomaly_score"] = round(score, 4)
                        if is_anomaly:
                            _fire_alert("ALERT", "ANOMALY",
                                        f"Isolation Forest gắn cờ bất thường ECG (score={score:.3f})",
                                        hr_latest, activity)
                    else:
                        row["anomaly_pred"]  = "warmup"
                        row["anomaly_score"] = ""
                except Exception as e:
                    row["anomaly_pred"] = f"err:{e}"

            # ── HR context alert ──────────────────────────
            hr_mean_bpm = 60000.0 / hrv["RR_mean"] if hrv["RR_mean"] > 0 else hr_latest
            level, reason = context_hr_alert(hr_mean_bpm, activity)
            row["hr_context_alert"] = level
            if level in ("ALERT", "CRITICAL"):
                _fire_alert(level, "HR_CONTEXT", reason, hr_mean_bpm, activity)

    w_pred.writerow(row)
    f_pred.flush()
    return row


# ════════════════════════════════════════════════════════════════
#  DISPLAY LOOP
# ════════════════════════════════════════════════════════════════
ALERT_COLORS = {
    "SAFE"    : "\033[92m",
    "WATCH"   : "\033[93m",
    "ALERT"   : "\033[91m",
    "CRITICAL": "\033[95m",
}
RESET = "\033[0m"


def _count_rows(p):
    try:
        with open(p) as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def display_loop():
    start = time.time()
    while is_running:
        time.sleep(2)
        now = time.time()
        elapsed = int(now - start)

        with buf_lock:
            n_ecg = len(ecg_buf)
            n_acc = len(acc_buf)
            n_rr  = len(rr_buf)
            hr_v  = hr_buf[-1][1] if hr_buf else 0
            cut10 = now - 10
            vm_10s = np.array([vm for ts, _, _, _, vm in acc_buf if ts >= cut10])

        vm_std   = float(np.std(vm_10s)) if len(vm_10s) > 2 else 0
        activity = classify_activity(vm_std) if vm_std > 0 else "UNKNOWN"
        alert_l  = last_row.get("hr_context_alert", "SAFE") or "SAFE"
        alert_c  = ALERT_COLORS.get(alert_l, RESET)

        stress_p = last_row.get("stress_pred", "—")
        if stress_p == 1:
            stress_str = f"\033[1;91mSTRESS\033[0m"
        elif stress_p == 0:
            stress_str = f"\033[92mNormal\033[0m"
        else:
            stress_str = str(stress_p)

        anom_p = last_row.get("anomaly_pred", "—")
        if anom_p == 1:
            anom_str = f"\033[1;91mANOMALY\033[0m"
        elif anom_p == 0:
            anom_str = f"\033[92mNormal\033[0m"
        elif anom_p == "warmup":
            warmup_n = len(_anomaly_feature_history)
            anom_str = f"\033[93mwarming up ({warmup_n}/{ANOMALY_WARMUP_WINDOWS})\033[0m"
        else:
            anom_str = str(anom_p)

        print("\033[2J\033[H", end="")
        print("═" * 64)
        print("  POLAR H10  —  PIPELINE (ECG + ACC + STRESS + ANOMALY)")
        print(f"  Session: {session_id}")
        print("═" * 64)
        print(f"  ⏱  Runtime    : {elapsed//60:02d}:{elapsed%60:02d}")
        print(f"  ❤  HR         : {hr_v} BPM")
        print(f"  🏃 Activity   : {activity}  (VM_std={vm_std:.0f} mG)")
        print(f"  ⚡ HR alert   : {alert_c}{alert_l}{RESET}")
        print()
        print(f"  🧠 Stress     : {stress_str}    proba={last_row.get('stress_proba','—')}")
        print(f"  🩺 Anomaly    : {anom_str}    score={last_row.get('anomaly_score','—')}")
        print()
        rr_mean = last_row.get("RR_mean", "—")
        rmssd   = last_row.get("RMSSD",   "—")
        lf_hf   = last_row.get("LF/HF",   "—")
        print(f"  📊 HRV (last) : RR_mean={rr_mean}  RMSSD={rmssd}  LF/HF={lf_hf}")
        print()
        print(f"  📡 ECG samples: {n_ecg:>8,}  ({n_ecg/FS_ECG:.0f}s)")
        print(f"  📡 ACC samples: {n_acc:>8,}  ({n_acc/FS_ACC:.0f}s)")
        print(f"  💓 RR (GATT)  : {n_rr:>8,}")
        print()
        n_pred  = _count_rows(predict_path)
        n_alert = _count_rows(alert_path)
        print(f"  📄 Predictions: {n_pred} rows  →  {predict_path.name}")
        print(f"  🚨 Alerts     : {n_alert} rows  →  {alert_path.name}")
        print("═" * 64)
        print("  [Ctrl+C] dừng và lưu")
        print("═" * 64)


def extractor_loop():
    global last_row
    while is_running:
        time.sleep(STEP_SEC)
        try:
            last_row = analyze_window()
        except Exception as e:
            print(f"[Extractor] {e}")


# ════════════════════════════════════════════════════════════════
#  BLE
# ════════════════════════════════════════════════════════════════
async def find_polar_h10(timeout: int = 20) -> str:
    print(f"Scanning for {POLAR_DEVICE_NAME}...")
    def _filter(dev, adv):
        name = (dev.name or "") + " " + (getattr(adv, "local_name", "") or "")
        return POLAR_DEVICE_NAME in name
    dev = await BleakScanner.find_device_by_filter(_filter, timeout=timeout)
    if dev is None:
        raise RuntimeError(f"Không tìm thấy {POLAR_DEVICE_NAME}")
    print(f"  ✓ Found {dev.address}")
    return dev.address


async def run_ble():
    global is_running
    while is_running:
        try:
            address = POLAR_ADDRESS or await find_polar_h10()
            print(f"Kết nối {address}...")
            async with BleakClient(address, timeout=20) as client:
                print("✓ Đã kết nối!\n")

                try:
                    await client.start_notify(HR_UUID, hr_callback)
                    print("  ✓ HR notify: ACTIVE")
                except Exception as e:
                    print(f"  ⚠ HR notify skipped: {e}")

                try:
                    await client.start_notify(PMD_CONTROL, pmd_control_callback)
                    print("  ✓ PMD control notify: ACTIVE")
                except Exception as e:
                    print(f"  ⚠ PMD control notify skipped: {e}")

                await client.start_notify(PMD_DATA, pmd_callback)
                print("  ✓ PMD data notify: ACTIVE")

                for stop_cmd, name in ((ECG_STOP, "ECG"), (ACC_STOP, "ACC")):
                    try:
                        await client.write_gatt_char(PMD_CONTROL, stop_cmd, response=True)
                        await asyncio.sleep(0.2)
                    except Exception:
                        pass

                await client.write_gatt_char(PMD_CONTROL, ECG_START, response=True)
                await asyncio.sleep(0.5)
                print("  ✓ ECG 130 Hz START")

                await client.write_gatt_char(PMD_CONTROL, ACC_START, response=True)
                await asyncio.sleep(0.5)
                print("  ✓ ACC 25 Hz START")

                print("\n  📊 Đang thu thập + phân tích realtime...\n")
                while is_running and client.is_connected:
                    await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠  Lỗi BLE: {e}. Reconnect sau 5s...")
            await asyncio.sleep(5)


def ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_ble())
    finally:
        loop.close()


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 64)
    print("  POLAR H10 — REALTIME PIPELINE (Stress + Anomaly + Fall)")
    print("═" * 64)
    print(f"  Output dir   : {out_dir}/")
    print(f"  ECG raw      : {ecg_path.name}")
    print(f"  ACC raw      : {acc_path.name}")
    print(f"  Predictions  : {predict_path.name}")
    print(f"  Alerts       : {alert_path.name}")
    print(f"  Models       : stress={'yes' if stress_model else 'NO'} | "
          f"anomaly={'yes' if anomaly_model else 'NO'}")
    print(f"  Warmup       : {ANOMALY_WARMUP_WINDOWS} windows (~{ANOMALY_WARMUP_WINDOWS*STEP_SEC}s) "
          "trước khi bật anomaly")
    print("\n  Bắt đầu sau 3s...\n")
    time.sleep(3)

    threads = [
        threading.Thread(target=ble_thread,     daemon=True, name="BLE"),
        threading.Thread(target=extractor_loop, daemon=True, name="Extractor"),
        threading.Thread(target=display_loop,   daemon=True, name="Display"),
    ]
    for t in threads:
        t.start()

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nĐang dừng và lưu...")
        is_running = False
        time.sleep(2)
    finally:
        try:
            analyze_window()
        except Exception:
            pass
        for f in (f_ecg, f_acc, f_pred, f_alert):
            try:
                f.close()
            except Exception:
                pass
        print("\n" + "═" * 64)
        print("  ĐÃ LƯU XONG")
        print("═" * 64)
        for p in (ecg_path, acc_path, predict_path, alert_path):
            n = _count_rows(p)
            print(f"  {p.name}  ({n} dòng dữ liệu)")
        print("═" * 64)
