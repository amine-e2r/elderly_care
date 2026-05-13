"""
ecg_acc_collector.py
====================
Thu thập ĐỒNG THỜI ECG (130Hz) + ACC (25Hz) từ Polar H10.

Mục đích:
  Kết hợp gia tốc (ACC) với nhịp tim (HR/ECG) để:
  1. Phân loại trạng thái hoạt động (REST/WALK/ACTIVE)
  2. Áp dụng ngưỡng HR ĐỘNG theo hoạt động
  3. Loại trừ HR tăng sinh lý do vận động (không báo nhầm)
  4. Phát hiện ngã (ACC spike + bất động)

Output:
  polar_data/
    ecg_YYYYMMDD_HHMMSS.csv          ← ECG thô (µV, timestamp Polar)
    acc_YYYYMMDD_HHMMSS.csv          ← ACC thô (mG, x/y/z/vm)
    windows_YYYYMMDD_HHMMSS.csv      ← Features 30s (HR + HRV + ACC + label)
    realtime_log_YYYYMMDD_HHMMSS.csv ← Log alert realtime

Sửa lỗi so với only_ecg.py:
  - ECG parse: offset đúng (10, không phải 4), little-endian, đơn vị µV
  - ACC parse: 6 bytes/sample (int16 LE × 3 trục), đơn vị mG
  - Timestamp đúng: lấy từ Polar packet header, tính offset từng sample
  - Dual stream: gửi 2 lệnh START riêng biệt (ECG + ACC)
"""

import asyncio
import struct
import time
import csv
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from scipy.signal import butter, filtfilt, welch, find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠  scipy chưa cài: pip install scipy")
    def find_peaks(x, height=None, distance=None):
        x = np.asarray(x)
        idx = []
        for i in range(1, len(x) - 1):
            if x[i] >= x[i-1] and x[i] >= x[i+1]:
                if height is None or x[i] >= height:
                    if not idx or (i - idx[-1]) >= (distance or 1):
                        idx.append(i)
        return np.array(idx, dtype=int), {}

from bleak import BleakClient

# ════════════════════════════════════════════════════════════════
#  CẤU HÌNH
# ════════════════════════════════════════════════════════════════
POLAR_ADDRESS = "24:AC:AC:13:EB:86"   # ← thay MAC của bạn

# UUIDs (Polar Open SDK)
HR_UUID      = "00002a37-0000-1000-8000-00805f9b34fb"
PMD_CONTROL  = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA     = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

FS_ECG = 130   # Hz
FS_ACC = 25    # Hz

# Polar PMD control command layout:
#   [REQUEST_START=0x02, measurement_type, setting_type, value_count, value_le...]
# Setting ids follow the official Polar BLE SDK:
#   SAMPLE_RATE=0, RESOLUTION=1, RANGE=2
PMD_START = 0x02
TYPE_ECG = 0x00
TYPE_ACC = 0x02
SETTING_SAMPLE_RATE = 0x00
SETTING_RESOLUTION = 0x01
SETTING_RANGE = 0x02


def _pmd_setting(setting_type: int, value: int) -> list[int]:
    return [setting_type, 0x01, value & 0xFF, (value >> 8) & 0xFF]


def build_pmd_start(measurement_type: int,
                    settings: list[tuple[int, int]]) -> bytearray:
    cmd = [PMD_START, measurement_type]
    for setting_type, value in settings:
        cmd.extend(_pmd_setting(setting_type, value))
    return bytearray(cmd)


# Start ECG: 130 Hz, 14-bit. This is the H10 command variant that was
# already proven to stream ECG in this project.
ECG_START = build_pmd_start(TYPE_ECG, [
    (SETTING_SAMPLE_RATE, FS_ECG),
    (SETTING_RESOLUTION, 14),
])

# Start ACC: 25 Hz, 16-bit, +/-8G.
# Té ngã trên ngực thường tạo impact 3-6 G; range ±2G sẽ bị clip ở 2000 mG/trục
# (đã quan sát trong log: x,y,z chạm ±2000 → VM bị giới hạn ~3464 mG).
ACC_START = build_pmd_start(TYPE_ACC, [
    (SETTING_SAMPLE_RATE, FS_ACC),
    (SETTING_RESOLUTION, 16),
    (SETTING_RANGE, 8),
])

ECG_STOP = bytearray([0x03, TYPE_ECG])
ACC_STOP = bytearray([0x03, TYPE_ACC])

# Window feature extraction
WIN_SEC   = 30    # giây mỗi window
STEP_SEC  = 5     # overlap để không bỏ lỡ impact té ngã ngắn
BUFFER_KEEP_SEC = 600  # giữ 10 phút gần nhất trong RAM; raw data vẫn lưu CSV

# ════════════════════════════════════════════════════════════════
#  NGƯỠNG HR THEO HOẠT ĐỘNG (người cao tuổi ≥60t)
# ════════════════════════════════════════════════════════════════
ACTIVITY_HR_THRESHOLDS = {
    #              (low_alert, high_watch, high_alert, high_critical)
    "REST"      : (45,  90,  100, 130),
    "SLOW_WALK" : (48, 105,  115, 140),
    "WALK"      : (50, 120,  130, 150),
    "ACTIVE"    : (55, 135,  145, 160),
    "UNKNOWN"   : (45, 100,  110, 135),
}

# Ngưỡng ACC phân loại hoạt động từ VM standard deviation
ACC_VM_STD_THRESHOLDS = {
    # (vm_std < X) → activity
    20  : "REST",
    80  : "SLOW_WALK",
    300 : "WALK",
    # > 300 → ACTIVE
}

# Ngưỡng phát hiện ngã. ACC bao gồm trọng lực ⇒ VM nghỉ ≈ 1000 mG.
#   IMPACT  : đỉnh VM tuyệt đối phải > 2500 mG để loại trừ bước chân/stomp.
#             Chest-worn IMU với range ±8G có thể đo tới ~8000 mG.
#   FREE_FALL: VM rơi < 500 mG trong 0.2–0.6 s ngay trước impact.
#   STILL   : std VM 3 s sau impact < 80 mG (người bất động chỉ còn 1 G ổn định).
#   FALL_PROX_S: free-fall và impact phải cách nhau ≤ thời gian này.
FALL_IMPACT_THRESHOLD_MG     = 2000
FALL_FREE_FALL_THRESHOLD_MG  = 500
FALL_POST_STILL_STD_MG       = 120
FALL_PROX_S                  = 1.0   # free-fall–impact tối đa cách 1.0 s
FALL_ALERT_COOLDOWN_S        = 15    # không lặp alert cùng cú ngã trong 15 s
FALL_STRONG_IMPACT_MG        = 2800  # ngưỡng để xếp CRITICAL

# ════════════════════════════════════════════════════════════════
#  GLOBAL BUFFERS
# ════════════════════════════════════════════════════════════════
# (timestamp_unix, value)
ecg_buf  : list = []   # ECG µV
acc_buf  : list = []   # (ts, x, y, z, vm) trong mG
rr_buf   : list = []   # (ts, rr_ms)
hr_buf   : list = []   # (ts, hr_bpm)

buf_lock = threading.Lock()
is_running = True

# ════════════════════════════════════════════════════════════════
#  CSV SETUP
# ════════════════════════════════════════════════════════════════
session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir     = Path("polar_data"); out_dir.mkdir(exist_ok=True)

ecg_path    = out_dir / f"ecg_{session_id}.csv"
acc_path    = out_dir / f"acc_{session_id}.csv"
win_path    = out_dir / f"windows_{session_id}.csv"
log_path    = out_dir / f"realtime_log_{session_id}.csv"

f_ecg = open(ecg_path, "w", newline="")
f_acc = open(acc_path, "w", newline="")
f_win = open(win_path, "w", newline="")
f_log = open(log_path, "w", newline="")

w_ecg = csv.writer(f_ecg)
w_acc = csv.writer(f_acc)

# ── Window features header ─────────────────────────────────────
WIN_COLS = [
    "timestamp", "window_sec", "n_rr", "n_acc",
    # HR
    "hr_mean", "hr_std", "hr_min", "hr_max",
    # HRV time-domain
    "rr_mean", "rr_std", "rmssd", "pnn50",
    "rr_cv", "rr_iqr", "rr_skew", "rr_kurt",
    "sd1", "sd2", "sd12",
    "rr_entropy", "rr_long_ratio", "rr_short_ratio",
    "sdrr", "rmssd_sdnn_ratio",
    # HRV freq-domain
    "lf_power", "hf_power", "lf_hf",
    # ACC features
    "acc_vm_mean", "acc_vm_std", "acc_vm_max", "acc_vm_range",
    "acc_x_std", "acc_y_std", "acc_z_std",
    "acc_energy",
    # Activity & anomaly
    "activity_state",
    "fall_spike",      # 1 nếu có impact spike > ngưỡng
    "free_fall",       # 1 nếu có pha gần rơi tự do trước impact
    "post_still",      # 1 nếu bất động sau spike
    "fall_candidate",  # 1 nếu impact + (free_fall hoặc post_still)
    "strong_fall",     # 1 nếu impact + free_fall + post_still
    "fall_peak_vm",
    "fall_min_before_1s",
    "fall_std_after_5s",
    # Context-aware HR alert
    "hr_context_alert",     # SAFE/WATCH/ALERT/CRITICAL theo activity
    "hr_context_threshold", # ngưỡng HR đã áp dụng
    # Label (người gán thủ công khi phân tích sau)
    "label_activity",   # để trống, gán sau
    "label_anomaly",    # để trống, gán sau
]

w_win = csv.DictWriter(f_win, fieldnames=WIN_COLS)
w_log = csv.DictWriter(f_log, fieldnames=[
    "timestamp", "hr_bpm", "activity", "alert_level",
    "alert_reason", "vm_std", "rmssd"
])

# Ghi headers
w_ecg.writerow(["timestamp_unix", "timestamp_polar_ns", "sample_index", "ecg_uv"])
w_acc.writerow(["timestamp_unix", "timestamp_polar_ns", "sample_index",
                "x_mg", "y_mg", "z_mg", "vm_mg"])
w_win.writeheader()
w_log.writeheader()

for f in [f_ecg, f_acc, f_win, f_log]:
    f.flush()


# ════════════════════════════════════════════════════════════════
#  PARSERS
# ════════════════════════════════════════════════════════════════
def parse_ecg_packet(data: bytes) -> tuple[int, list[int]]:
    """
    Parse ECG PMD packet. Trả về (timestamp_polar_ns, [sample_uV]).

    Cấu trúc:
      Byte 0      : measurement type (0x00 = ECG)
      Bytes 1–8   : timestamp uint64 LE (nanoseconds)
      Byte 9      : frame type (0x00)
      Byte 10+    : ECG samples, 3 bytes/sample (int24 LE signed)
    """
    if len(data) < 10 or (data[0] & 0x3F) != TYPE_ECG:
        return 0, []

    ts_ns = struct.unpack_from("<Q", data, 1)[0]   # little-endian uint64

    samples = []
    offset  = 10
    while offset + 2 < len(data):
        b0, b1, b2 = data[offset], data[offset+1], data[offset+2]
        raw = b0 | (b1 << 8) | (b2 << 16)
        if raw & 0x800000:         # two's complement sign extension
            raw -= 0x1000000
        samples.append(raw)        # đơn vị µV
        offset += 3

    return ts_ns, samples


def parse_acc_packet(data: bytes) -> tuple[int, list[tuple]]:
    """
    Parse ACC PMD packet. Trả về (timestamp_polar_ns, [(x,y,z,vm)]).

    Cấu trúc:
      Byte 0      : measurement type (0x02 = ACC)
      Bytes 1–8   : timestamp uint64 LE (nanoseconds)
      Byte 9      : frame type (0x01 for int16)
      Byte 10+    : ACC samples, 6 bytes/sample:
                    2 bytes X (int16 LE) + 2Y + 2Z
                    Đơn vị: mG (milligravity)
    """
    if len(data) < 10 or (data[0] & 0x3F) != TYPE_ACC:
        return 0, []

    ts_ns = struct.unpack_from("<Q", data, 1)[0]
    frame_type = data[9] & 0x7F
    is_compressed = bool(data[9] & 0x80)
    if is_compressed or frame_type not in (0, 1, 2):
        print(f"[ACC Parse] Unsupported ACC frame type: 0x{data[9]:02x}")
        return ts_ns, []

    samples = []
    offset  = 10
    sample_size = {0: 1, 1: 2, 2: 3}[frame_type]
    sample_bytes = sample_size * 3
    while offset + sample_bytes <= len(data):
        if sample_size == 1:
            x, y, z = struct.unpack_from("<bbb", data, offset)
        elif sample_size == 2:
            x, y, z = struct.unpack_from("<hhh", data, offset)
        else:
            x = int.from_bytes(data[offset:offset+3], "little", signed=True)
            y = int.from_bytes(data[offset+3:offset+6], "little", signed=True)
            z = int.from_bytes(data[offset+6:offset+9], "little", signed=True)
        vm = float(np.sqrt(x*x + y*y + z*z))
        samples.append((float(x), float(y), float(z), vm))
        offset += sample_size * 3

    return ts_ns, samples


def parse_hr_gatt(data: bytes) -> dict:
    """Parse GATT 0x2A37 Heart Rate Measurement."""
    if len(data) < 2:
        return {"hr": 0, "rr": []}

    flags  = data[0]; offset = 1
    if flags & 0x01 and len(data) < offset + 2:
        return {"hr": 0, "rr": []}

    hr = (int.from_bytes(data[offset:offset+2],"little"),offset:=offset+2)[0] \
         if flags & 0x01 \
         else (data[offset], offset:=offset+1)[0]
    if flags & 0x08: offset += 2
    rr = []
    if flags & 0x10:
        while offset + 1 < len(data):
            rr.append(round(int.from_bytes(data[offset:offset+2],"little")*1000/1024,1))
            offset += 2
    return {"hr": hr, "rr": rr}


# ════════════════════════════════════════════════════════════════
#  BLE CALLBACKS
# ════════════════════════════════════════════════════════════════
_pkt_idx_ecg = 0
_pkt_idx_acc = 0
_last_prune = 0.0


def _drop_old_samples(buf: list, cutoff: float) -> None:
    idx = 0
    n = len(buf)
    while idx < n and buf[idx][0] < cutoff:
        idx += 1
    if idx:
        del buf[:idx]


def prune_runtime_buffers(now: float) -> None:
    cutoff = now - BUFFER_KEEP_SEC
    for buf in (ecg_buf, acc_buf, rr_buf, hr_buf):
        _drop_old_samples(buf, cutoff)

def pmd_callback(sender, data: bytearray):
    """Router: phân loại ECG hay ACC và xử lý."""
    global _pkt_idx_ecg, _pkt_idx_acc, _last_prune
    raw = bytes(data)
    if not raw:
        return
    now = time.time()
    data_type = raw[0] & 0x3F

    if data_type == TYPE_ECG:      # ECG packet
        ts_ns, samples = parse_ecg_packet(raw)
        if not samples: return
        dt_ns = int(1e9 / FS_ECG)  # khoảng cách ns giữa mỗi sample
        n = len(samples)
        # Sample cuối packet là sample mới nhất ≈ now; sample đầu được lấy
        # trước đó (n-1)/FS giây. Cần "lùi" về quá khứ.
        with buf_lock:
            for i, s in enumerate(samples):
                t_unix = now - (n - 1 - i) / FS_ECG
                ecg_buf.append((t_unix, s))
                w_ecg.writerow([f"{t_unix:.6f}", ts_ns + i*dt_ns,
                                 _pkt_idx_ecg, s])
            _pkt_idx_ecg += n
            if now - _last_prune > 5:
                prune_runtime_buffers(now)
                _last_prune = now
        f_ecg.flush()

    elif data_type == TYPE_ACC:    # ACC packet
        ts_ns, samples = parse_acc_packet(raw)
        if not samples: return
        dt_ns = int(1e9 / FS_ACC)
        n = len(samples)
        with buf_lock:
            for i, (x, y, z, vm) in enumerate(samples):
                t_unix = now - (n - 1 - i) / FS_ACC
                acc_buf.append((t_unix, x, y, z, vm))
                w_acc.writerow([f"{t_unix:.6f}", ts_ns + i*dt_ns,
                                 _pkt_idx_acc, x, y, z, round(vm, 2)])
            _pkt_idx_acc += n
            if now - _last_prune > 5:
                prune_runtime_buffers(now)
                _last_prune = now
        f_acc.flush()


def pmd_control_callback(sender, data: bytearray):
    raw = bytes(data)
    if len(raw) >= 4 and raw[0] == 0xF0:
        command = raw[1]
        measurement = raw[2] & 0x3F
        status = raw[3]
        name = {TYPE_ECG: "ECG", TYPE_ACC: "ACC"}.get(measurement, f"0x{measurement:02x}")
        status_name = "SUCCESS" if status == 0 else f"ERROR({status})"
        print(f"  PMD response: command=0x{command:02x}, type={name}, {status_name}")
    else:
        print(f"  PMD control: {raw.hex()}")


def hr_callback(sender, data: bytearray):
    global _last_prune
    parsed = parse_hr_gatt(bytes(data))
    now    = time.time()
    with buf_lock:
        hr_buf.append((now, parsed["hr"]))
        for rr in parsed["rr"]:
            rr_buf.append((now, rr))
        if now - _last_prune > 5:
            prune_runtime_buffers(now)
            _last_prune = now


# ════════════════════════════════════════════════════════════════
#  ACTIVITY CLASSIFICATION
# ════════════════════════════════════════════════════════════════
def classify_activity(vm_std: float) -> str:
    """Phân loại hoạt động từ VM standard deviation."""
    for threshold, label in sorted(ACC_VM_STD_THRESHOLDS.items()):
        if vm_std < threshold:
            return label
    return "ACTIVE"


def detect_fall(vm_array: np.ndarray) -> dict:
    """
    Quét toàn bộ window cho chữ ký free-fall → impact → bất động.

    Khác bản trước:
      • Không dùng argmax (chỉ thấy 1 đỉnh / window — sẽ bỏ sót nhiều cú ngã
        chồng nhau và lỗi khi đỉnh argmax là bước chân chứ không phải impact).
      • Quét MỌI peak vượt FALL_IMPACT_THRESHOLD_MG bằng find_peaks.
      • Free-fall phải xảy ra trong FALL_PROX_S giây ngay trước peak
        (không phải bất kỳ chỗ nào trong cửa sổ 30 s).
      • Trả về peak có "điểm tin cậy" cao nhất (free_fall + post_still).
    """
    result = {
        "impact_spike": False,
        "free_fall": False,
        "post_still": False,
        "fall_candidate": False,
        "strong_fall": False,
        "peak_vm": 0.0,
        "min_before_1s": 0.0,
        "std_after_5s": 0.0,
        "n_impacts": 0,
        "fall_offset_s": -1.0,   # vị trí peak tính từ đầu window, để dedup
    }

    n = len(vm_array)
    if n < FS_ACC * 3:           # cần tối thiểu 3 s dữ liệu
        return result

    # Yêu cầu các peak cách nhau >= 0.6 s để không nhặt cùng một cú impact
    # nhiều lần (impact thật là một burst rất ngắn).
    peaks, _ = find_peaks(
        vm_array,
        height=FALL_IMPACT_THRESHOLD_MG,
        distance=max(1, int(FS_ACC * 0.6)),
    )
    result["n_impacts"] = int(len(peaks))
    if len(peaks) == 0:
        return result

    prox = max(1, int(FS_ACC * FALL_PROX_S))
    after_n = int(FS_ACC * 3)     # 3 s bất động là đủ để khẳng định ngã
    min_after = int(FS_ACC * 2)   # cần ít nhất 2 s sau peak để tin std

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

        # Điểm tin cậy: ưu tiên cú nào có cả free-fall và bất động.
        score = (2 if free_fall else 0) + (1 if post_still else 0)
        # Tie-break bằng peak cao nhất.
        if score > best_score or (
            score == best_score and (best is None or vm_array[pi] > vm_array[best[0]])
        ):
            best_score = score
            best = (int(pi), free_fall, post_still, min_before, std_after)

    if best is None:
        return result

    pi, free_fall, post_still, min_before, std_after = best
    peak_vm = float(vm_array[pi])
    result["impact_spike"]   = True
    result["peak_vm"]        = round(peak_vm, 2)
    result["free_fall"]      = bool(free_fall)
    result["post_still"]     = bool(post_still)
    result["min_before_1s"]  = round(min_before, 2)
    result["std_after_5s"]   = round(std_after, 2)
    result["fall_offset_s"]  = round(pi / FS_ACC, 2)
    # Cốt lõi: free-fall ngay trước impact là chữ ký rất hiếm khi đi bộ/stomp.
    # Walking peak (2-3 G) KHÔNG đi kèm pha rơi tự do trước đó.
    # Không yêu cầu post_still vì người ngã có thể còn cử động/giẫy.
    result["fall_candidate"] = bool(free_fall)
    result["strong_fall"]    = bool(
        free_fall and (post_still or peak_vm > FALL_STRONG_IMPACT_MG)
    )
    return result


# ════════════════════════════════════════════════════════════════
#  CONTEXT-AWARE HR ALERT
# ════════════════════════════════════════════════════════════════
def context_hr_alert(hr_mean: float, activity: str) -> tuple[str, str]:
    """
    Trả về (alert_level, reason) theo activity context.

    Đây là logic cốt lõi để loại trừ HR tăng sinh lý:
    - Cùng HR 115 BPM:
        REST   → ALERT (bình thường HR nghỉ <90)
        WALK   → SAFE  (bình thường khi đi bộ ≤130)
    """
    lo, hi_w, hi_a, hi_c = ACTIVITY_HR_THRESHOLDS.get(
        activity, ACTIVITY_HR_THRESHOLDS["UNKNOWN"]
    )

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
#  HRV FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════
def _skew(x): m=np.mean(x); s=np.std(x)+1e-9; return float(np.mean(((x-m)/s)**3))
def _kurt(x): m=np.mean(x); s=np.std(x)+1e-9; return float(np.mean(((x-m)/s)**4)-3)

def extract_hrv(rr_ms: np.ndarray) -> dict:
    rr     = rr_ms[(rr_ms>300)&(rr_ms<2000)]
    if len(rr) < 5: return {}
    rr_d   = np.diff(rr); n_d = max(len(rr_d),1)
    hr     = 60000.0/rr; mu = np.mean(rr); s = np.std(rr)
    rmssd  = float(np.sqrt(np.mean(rr_d**2))) if len(rr_d) else 0
    sd1    = float(np.std(rr_d)/np.sqrt(2)) if len(rr_d) else 0
    sd2_sq = max(0, 2*np.var(rr)-np.var(rr_d)/2) if len(rr_d) else 0
    sd2    = float(np.sqrt(sd2_sq))
    hist,_ = np.histogram(rr, bins=10)
    p      = (hist[hist>0]/hist.sum()).astype(float)

    feats = {
        "hr_mean" : round(float(np.mean(hr)),2),
        "hr_std"  : round(float(np.std(hr)),2),
        "hr_min"  : round(float(np.min(hr)),2),
        "hr_max"  : round(float(np.max(hr)),2),
        "rr_mean" : round(float(mu),2),
        "rr_std"  : round(float(s),2),
        "rmssd"   : round(rmssd,2),
        "pnn50"   : round(float(np.sum(np.abs(rr_d)>50)/n_d*100),2),
        "rr_cv"   : round(float(s/(mu+1e-9)),6),
        "rr_iqr"  : round(float(np.percentile(rr,75)-np.percentile(rr,25)),2),
        "rr_skew" : round(_skew(rr),4),
        "rr_kurt" : round(_kurt(rr),4),
        "sd1"     : round(sd1,2),
        "sd2"     : round(sd2,2),
        "sd12"    : round(sd1/(sd2+1e-9),4),
        "rr_entropy"      : round(float(-np.sum(p*np.log2(p+1e-12))),4),
        "rr_long_ratio"   : round(float(np.sum(rr>mu+s)/len(rr)),4),
        "rr_short_ratio"  : round(float(np.sum(rr<mu-s)/len(rr)),4),
        "sdrr"            : round(float(np.std(rr_d)) if len(rr_d) else 0,2),
        "rmssd_sdnn_ratio": round(rmssd/(s+1e-9),4),
        "rr_p10"          : round(float(np.percentile(rr,10)),2),
        "rr_p90"          : round(float(np.percentile(rr,90)),2),
        "lf_power":0.0, "hf_power":0.0, "lf_hf":0.0,
    }

    if SCIPY_OK and len(rr) >= 20:
        try:
            t  = np.cumsum(rr)/1000.0
            te = np.arange(t[0],t[-1],0.25)
            ri = np.interp(te,t,rr)
            f,p= welch(ri,fs=4.0,nperseg=min(len(ri),256))
            bp = lambda lo,hi: float(np.trapz(p[(f>=lo)&(f<=hi)],f[(f>=lo)&(f<=hi)]))
            lf = bp(0.04,0.15); hf = bp(0.15,0.40)
            feats.update({"lf_power":round(lf,2),
                          "hf_power":round(hf,2),
                          "lf_hf"   :round(lf/(hf+1e-9),4)})
        except: pass

    return feats


# ════════════════════════════════════════════════════════════════
#  FALL ALERT STATE — dùng để dedup giữa các overlapping windows
# ════════════════════════════════════════════════════════════════
_last_fall_alert_ts: float = 0.0     # unix ts cú alert gần nhất
_last_fall_event_ts: float = 0.0     # ts của cú té đã alert (đầu peak)
_fall_alert_count: int = 0


def _fire_fall_alert(level: str, reason: str, peak_vm: float) -> None:
    """Bật chuông + in banner đỏ ra stderr để người trực thấy."""
    import sys
    global _fall_alert_count
    _fall_alert_count += 1
    bar = "█" * 60
    msg = (
        f"\n\033[1;97;41m{bar}\033[0m\n"
        f"\033[1;97;41m  🚨 FALL ALERT #{_fall_alert_count} — {level}  "
        f"peak={peak_vm:.0f} mG  \033[0m\n"
        f"\033[1;91m  {reason}\033[0m\n"
        f"\033[1;97;41m{bar}\033[0m\n"
        "\a"   # BEL — system beep
    )
    sys.stderr.write(msg)
    sys.stderr.flush()


# ════════════════════════════════════════════════════════════════
#  WINDOW EXTRACTION — chạy mỗi STEP_SEC giây
# ════════════════════════════════════════════════════════════════
def extract_window():
    """
    Trích xuất 1 window đặc trưng từ dữ liệu WIN_SEC giây gần nhất.
    Kết hợp HRV (từ RR) + ACC features + context-aware alert.
    """
    now    = time.time()
    cutoff = now - WIN_SEC

    with buf_lock:
        rr_w  = np.array([rr for ts,rr in rr_buf  if ts>=cutoff])
        acc_w = np.array([vm for ts,_,_,_,vm in acc_buf if ts>=cutoff])
        acc_xyz = [(x,y,z) for ts,x,y,z,vm in acc_buf if ts>=cutoff]

    row = {k: "" for k in WIN_COLS}
    row["timestamp"]  = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    row["window_sec"] = WIN_SEC
    row["n_rr"]       = len(rr_w)
    row["n_acc"]      = len(acc_w)

    # ── HRV features ──────────────────────────────────────────
    if len(rr_w) >= 5:
        hrv = extract_hrv(rr_w)
        for k,v in hrv.items():
            if k in row: row[k] = v
        hr_mean = hrv.get("hr_mean", 0)
    else:
        hr_mean = 0

    # ── ACC features ──────────────────────────────────────────
    if len(acc_w) >= 5:
        vm_std  = float(np.std(acc_w))
        vm_mean = float(np.mean(acc_w))
        vm_max  = float(np.max(acc_w))

        row["acc_vm_mean"]  = round(vm_mean, 2)
        row["acc_vm_std"]   = round(vm_std, 2)
        row["acc_vm_max"]   = round(vm_max, 2)
        row["acc_vm_range"] = round(vm_max - float(np.min(acc_w)), 2)
        row["acc_energy"]   = round(float(np.mean(acc_w**2)), 2)

        if acc_xyz:
            xs = np.array([x for x,y,z in acc_xyz])
            ys = np.array([y for x,y,z in acc_xyz])
            zs = np.array([z for x,y,z in acc_xyz])
            row["acc_x_std"] = round(float(np.std(xs)), 2)
            row["acc_y_std"] = round(float(np.std(ys)), 2)
            row["acc_z_std"] = round(float(np.std(zs)), 2)

        activity = classify_activity(vm_std)
        fall = detect_fall(acc_w)

        row["activity_state"]       = activity
        row["fall_spike"]           = int(fall["impact_spike"])
        row["free_fall"]            = int(fall["free_fall"])
        row["post_still"]           = int(fall["post_still"])
        row["fall_candidate"]       = int(fall["fall_candidate"])
        row["strong_fall"]          = int(fall["strong_fall"])
        row["fall_peak_vm"]         = fall["peak_vm"]
        row["fall_min_before_1s"]   = fall["min_before_1s"]
        row["fall_std_after_5s"]    = fall["std_after_5s"]

        # ── Context-aware HR alert ─────────────────────────
        alert, reason = ("SAFE", "")
        if hr_mean > 0:
            alert, reason = context_hr_alert(hr_mean, activity)

        # Override theo mức tin cậy của phát hiện ngã.
        # Yêu cầu ĐỦ HAI dấu hiệu (free-fall + bất động) cho cảnh báo,
        # tránh false positive khi đi bộ/đứng yên thở mạnh.
        global _last_fall_alert_ts, _last_fall_event_ts
        fall_alert_level = None
        if fall["strong_fall"]:
            fall_alert_level = "CRITICAL"
            reason = (f"Phát hiện ngã mạnh (impact {fall['peak_vm']:.0f} mG "
                      f"+ free-fall {fall['min_before_1s']:.0f} mG "
                      f"+ bất động std={fall['std_after_5s']:.0f} mG)")
        elif fall["fall_candidate"]:
            fall_alert_level = "ALERT"
            reason = (f"Nghi ngờ ngã (impact {fall['peak_vm']:.0f} mG, "
                      f"min trước {fall['min_before_1s']:.0f} mG, "
                      f"std sau {fall['std_after_5s']:.0f} mG)")

        if fall_alert_level:
            # Quy đổi offset của peak trong window thành ts tuyệt đối, dùng để
            # biết hai window có đang nói về cùng một cú té không. Cú ngã đi
            # qua tối đa 6 cửa sổ liên tiếp (WIN_SEC/STEP_SEC) — chỉ alert 1 lần.
            event_ts = (now - WIN_SEC) + max(0.0, fall["fall_offset_s"])
            # Cùng cú té xuất hiện trong nhiều cửa sổ chồng nhưng peak_ts gần
            # giống nhau (±2-3 s). Cooldown thì chặn các cú té khác quá sát.
            same_event = abs(event_ts - _last_fall_event_ts) <= 3.0
            in_cooldown = (now - _last_fall_alert_ts) < FALL_ALERT_COOLDOWN_S
            alert = fall_alert_level
            if not (same_event or in_cooldown):
                _last_fall_alert_ts = now
                _last_fall_event_ts = event_ts
                _fire_fall_alert(fall_alert_level, reason, fall["peak_vm"])

        row["hr_context_alert"]     = alert
        row["hr_context_threshold"] = ACTIVITY_HR_THRESHOLDS.get(
            activity, ACTIVITY_HR_THRESHOLDS["UNKNOWN"]
        )[2] if hr_mean > 0 else ""

        # Ghi log realtime khi có HR context hoặc nghi ngờ ngã.
        if hr_mean > 0 or fall["fall_candidate"]:
            rmssd_val = hrv.get("rmssd", 0) if len(rr_w) >= 5 else 0
            w_log.writerow({
                "timestamp"   : row["timestamp"],
                "hr_bpm"      : hr_mean,
                "activity"    : activity,
                "alert_level" : alert,
                "alert_reason": reason,
                "vm_std"      : round(vm_std, 2),
                "rmssd"       : rmssd_val,
            })
            f_log.flush()
    else:
        row["activity_state"]       = "UNKNOWN"
        row["hr_context_alert"]     = "SAFE"
        row["hr_context_threshold"] = ""

    w_win.writerow(row)
    f_win.flush()
    return row


# ════════════════════════════════════════════════════════════════
#  DISPLAY REALTIME
# ════════════════════════════════════════════════════════════════
ALERT_COLORS = {
    "SAFE"    : "\033[92m",   # xanh
    "WATCH"   : "\033[93m",   # vàng
    "ALERT"   : "\033[91m",   # đỏ
    "CRITICAL": "\033[95m",   # tím đỏ
}
RESET = "\033[0m"

def display_loop():
    global last_row
    start = time.time()
    while is_running:
        time.sleep(2)
        now     = time.time()
        elapsed = int(now - start)

        with buf_lock:
            n_ecg = len(ecg_buf)
            n_acc = len(acc_buf)
            n_rr  = len(rr_buf)
            hr_v  = hr_buf[-1][1] if hr_buf else 0

            # ACC 10s gần nhất
            cut10  = now - 10
            vm_10s = np.array([vm for ts,_,_,_,vm in acc_buf if ts>=cut10])

        vm_std   = float(np.std(vm_10s)) if len(vm_10s)>2 else 0
        activity = classify_activity(vm_std)
        alert_c  = ALERT_COLORS.get(last_row.get("hr_context_alert","SAFE"), RESET)
        alert_l  = last_row.get("hr_context_alert","SAFE")
        rmssd_v  = last_row.get("rmssd", "--")

        print("\033[2J\033[H", end="")
        print("═"*62)
        print("  POLAR H10  —  ECG + ACC COLLECTOR")
        print(f"  Session: {session_id}")
        print("═"*62)
        print(f"  ⏱  Runtime     : {elapsed//60:02d}:{elapsed%60:02d}")
        print(f"  ❤  HR          : {hr_v} BPM")
        print(f"  🏃 Activity    : {activity}  (VM_std={vm_std:.0f} mG)")
        print(f"  ⚡ Alert       : {alert_c}{alert_l}{RESET}")
        if not hr_buf:
            print("     HR characteristic chưa hoạt động; vẫn thu ECG/ACC")
        elif last_row.get("hr_context_alert") == "SAFE":
            print(f"     HR {hr_v} BPM trong ngưỡng bình thường khi {activity}")
        else:
            threshold = last_row.get("hr_context_threshold", "")
            if threshold:
                print(f"     {threshold} BPM ngưỡng cho {activity}")
            else:
                print(f"     Theo dõi cảnh báo theo ACC/HR context")
        print()
        print(f"  📡 ECG samples : {n_ecg:>8,}  ({n_ecg/FS_ECG:.0f}s)")
        print(f"  📡 ACC samples : {n_acc:>8,}  ({n_acc/FS_ACC:.0f}s)")
        print(f"  💓 RR samples  : {n_rr:>8,}")
        print(f"  📊 RMSSD 30s   : {rmssd_v} ms")
        print()

        # VM trend bar
        if len(vm_10s) > 0:
            vm_norm = min(int(np.mean(vm_10s)/50), 30)
            bar = "▓"*vm_norm + "░"*(30-vm_norm)
            print(f"  ACC VM : [{bar}] {np.mean(vm_10s):.0f} mG")

        win_count = _count_rows(win_path)
        print(f"\n  Windows lưu   : {win_count}")
        print(f"  📄 {win_path.name}")
        print("═"*62)
        print("  [Ctrl+C] dừng và lưu")
        print("═"*62)


def _count_rows(p):
    try:
        with open(p) as f: return max(0, sum(1 for _ in f)-1)
    except: return 0


# ════════════════════════════════════════════════════════════════
#  EXTRACTION SCHEDULER
# ════════════════════════════════════════════════════════════════
def extractor_loop():
    global last_row
    last_row = {}
    while is_running:
        time.sleep(STEP_SEC)
        try:
            last_row = extract_window()
        except Exception as e:
            print(f"[Extractor] {e}")


# Biến dùng giữa display và extractor
last_row = {}


# ════════════════════════════════════════════════════════════════
#  BLE ASYNC
# ════════════════════════════════════════════════════════════════
async def run_ble():
    global is_running
    while is_running:
        try:
            print(f"Kết nối tới {POLAR_ADDRESS}...")
            async with BleakClient(POLAR_ADDRESS, timeout=20) as client:
                print("✓ Đã kết nối!\n")

                # HR + RR is optional for this collector. Some Windows BLE
                # sessions expose PMD but not the standard HR characteristic.
                try:
                    await client.start_notify(HR_UUID, hr_callback)
                    print("  ✓ HR + RR: ACTIVE")
                except Exception as e:
                    print(f"  ⚠ HR + RR skipped: {e}")

                # Start PMD data notification before START commands so the first
                # ECG/ACC packets are not missed. PMD_CONTROL notify is optional
                # on some Windows BLE stacks, so failure there must not block data.
                try:
                    await client.start_notify(PMD_CONTROL, pmd_control_callback)
                    print("  ✓ PMD control notify: ACTIVE")
                except Exception as e:
                    print(f"  ⚠ PMD control notify skipped: {e}")
                await client.start_notify(PMD_DATA, pmd_callback)
                print("  ✓ PMD data notify: ACTIVE")

                # Clear stale streams left active by previous runs.
                for stop_cmd, name in ((ECG_STOP, "ECG"), (ACC_STOP, "ACC")):
                    try:
                        await client.write_gatt_char(PMD_CONTROL, stop_cmd, response=True)
                        await asyncio.sleep(0.2)
                        print(f"  ✓ {name}: STOP previous stream")
                    except Exception:
                        pass

                # START ECG
                await client.write_gatt_char(PMD_CONTROL, ECG_START, response=True)
                await asyncio.sleep(0.5)
                print("  ✓ ECG 130Hz: ACTIVE")

                # START ACC
                await client.write_gatt_char(PMD_CONTROL, ACC_START, response=True)
                await asyncio.sleep(0.5)
                print("  ✓ ACC 25Hz: ACTIVE")

                print("\n  📊 Đang thu thập ECG + ACC đồng thời...\n")

                while is_running and client.is_connected:
                    await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠  Lỗi: {e}. Thử lại sau 5 giây...")
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
    print("═"*62)
    print("  POLAR H10 — ECG + ACC DUAL STREAM COLLECTOR")
    print("═"*62)
    print(f"\nOutput: {out_dir}/")
    print(f"  ECG thô      : {ecg_path.name}")
    print(f"  ACC thô      : {acc_path.name}")
    print(f"  Windows feat : {win_path.name}")
    print(f"  Realtime log : {log_path.name}")
    print(f"\nFeatures mỗi window 30s:")
    print(f"  HRV (24) + ACC (8) + Context alert (2) = 34 features")
    print(f"\nBắt đầu sau 3 giây...\n")
    time.sleep(3)

    threads = [
        threading.Thread(target=ble_thread,     daemon=True, name="BLE"),
        threading.Thread(target=extractor_loop, daemon=True, name="Extractor"),
        threading.Thread(target=display_loop,   daemon=True, name="Display"),
    ]
    for t in threads: t.start()

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nĐang lưu và kết thúc...")
        is_running = False
        time.sleep(2)
    finally:
        # Trích xuất window cuối
        try: extract_window()
        except: pass

        for f in [f_ecg, f_acc, f_win, f_log]:
            try: f.close()
            except: pass

        print("\n" + "═"*62)
        print("  ĐÃ LƯU XONG")
        print("═"*62)
        for p in [ecg_path, acc_path, win_path, log_path]:
            n = _count_rows(p)
            print(f"  {p.name}  ({n} dòng dữ liệu)")
        print("═"*62)
