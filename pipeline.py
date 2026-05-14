import numpy as np
import joblib
import scipy.signal as signal

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import asyncio
from datetime import datetime

from bleak import BleakScanner, BleakClient


# Standard and proprietary Polar UUIDs
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
POLAR_DEVICE_NAME = "Polar H10"  # Device name used during BLE scanning

# Polar PMD constants — same layout used by the working ecg_acc_collector.
PMD_START_OP = 0x02
PMD_STOP_OP  = 0x03
TYPE_ECG     = 0x00
TYPE_ACC     = 0x02
SETTING_SAMPLE_RATE = 0x00
SETTING_RESOLUTION  = 0x01

FS_ECG = 130  # Hz

def _pmd_setting(setting_type: int, value: int) -> list[int]:
    return [setting_type, 0x01, value & 0xFF, (value >> 8) & 0xFF]

def build_pmd_start(measurement_type: int, settings: list[tuple[int, int]]) -> bytearray:
    cmd = [PMD_START_OP, measurement_type]
    for st, v in settings:
        cmd.extend(_pmd_setting(st, v))
    return bytearray(cmd)

# ECG: 130 Hz, 14-bit. H10 transmits each sample as 3-byte signed LE.
ECG_START_CMD = build_pmd_start(TYPE_ECG, [
    (SETTING_SAMPLE_RATE, FS_ECG),
    (SETTING_RESOLUTION, 14),
])
ECG_STOP_CMD = bytearray([PMD_STOP_OP, TYPE_ECG])
ACC_STOP_CMD = bytearray([PMD_STOP_OP, TYPE_ACC])

def parse_ecg_packet(data: bytes) -> list[int]:
    """Decode a PMD ECG notification into a list of µV samples.

    Frame layout:
      byte 0     : measurement type (low 6 bits = 0x00 for ECG)
      bytes 1-8  : timestamp uint64 LE
      byte 9     : frame type
      byte 10+   : samples, 3 bytes/sample (int24 LE signed)
    """
    if len(data) < 13 or (data[0] & 0x3F) != TYPE_ECG:
        return []
    samples = []
    offset = 10
    while offset + 2 < len(data):
        samples.append(int.from_bytes(data[offset:offset+3], "little", signed=True))
        offset += 3
    return samples

async def find_polar_h10(timeout=20):
    # Automatically scan for a Polar H10 device by advertised BLE name.
    print(f"Scanning for {POLAR_DEVICE_NAME} for up to {timeout} seconds...")

    def polar_filter(device, adv):
        if device.name and POLAR_DEVICE_NAME in device.name:
            return True
        if adv and hasattr(adv, 'local_name') and adv.local_name and POLAR_DEVICE_NAME in adv.local_name:
            return True
        return False

    device = await BleakScanner.find_device_by_filter(polar_filter, timeout=timeout)
    if device is None:
        raise Exception(f"Could not find {POLAR_DEVICE_NAME} during BLE scan")

    print(f"Found {POLAR_DEVICE_NAME}: {device.address}")
    return device.address




# Import the models
try:
    random_forest_model = joblib.load("random_forest_model.pkl")
    print("Loaded random forest model for stress detection")
except FileNotFoundError:
    print("Warning: random_forest_model.pkl not found. Stress detection will be disabled.")
    random_forest_model = None

try:
    isolation_forest_model = joblib.load("isolation_forest_model.pkl")
    print("Loaded isolation forest model for anomaly detection")
except FileNotFoundError:
    print("Warning: isolation_forest_model.pkl not found. Anomaly detection will be disabled.")
    isolation_forest_model = None

# RR intervals extractor | Input : ECG signal | Output : RR intervals
class RRIntervalsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        X: (n_samples,) ECG signal brut
        """
        peaks, _ = signal.find_peaks(X, distance=self.sample_rate*0.3)
        rr_intervals = np.diff(peaks) / self.sample_rate * 1000  # Convert to milliseconds
        return rr_intervals.reshape(1, -1) # (1, n_rr)

# RR Window sliding | Input : RR intervals | Output : Windows of RR intervals
class RRWindowing(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size=None):
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X,):
        """
        X: (1, n_rr)
        """
        rr = X.flatten()
        windows = []
        for i in range(0, len(rr) - self.window_size + 1, self.step_size):
            windows.append(rr[i:i + self.window_size])
        
        return np.array(windows) # (n_windows, window_size) each line is a window of RR intervals

# Beats extractor | Input : ECG signal | Output : Beats (segments of ECG around each R peak)
class BeatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,sample_rate):
        self.sample_rate = sample_rate
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        X: (n_samples,) ECG signal brut
        """
        peaks, _ = signal.find_peaks(X, distance=self.sample_rate*0.3)
        target_len = int( (0.2 + 0.4) * self.sample_rate)
        
        beats = []
        for peak in peaks:
            start = max(0, peak - int(0.2 * self.sample_rate))  # 200ms before
            end = min(len(X), peak + int(0.4 * self.sample_rate))  # 400ms after
            
            if start >= 0 and end <= len(X):
                beats.append(X[start:end])
        
        return np.array(beats) # (n_beats, n_samples/target_len) each line is a beat segment

#Beats Window sliding | Input : Beats | Output : Windows of beats
class BeatsWindowing(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size=None):
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: (n_beats, n_samples) each line is a beat segment
        """
        windows = []
        for i in range(0, X.shape[0] - self.window_size + 1, self.step_size):
            windows.append(X[i:i + self.window_size])
        return np.array(windows) # (n_windows, window_size, n_samples)

# HRV extractor | Input : RR intervals windows | Output : HRV features
class HRVFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self    

    def transform(self, X):
        """
        X: (n_windows, window_size) each line is a window of RR intervals
        """
        features = []
        for rr_window in X:
            rr_mean = np.mean(rr_window)
            sdnn = np.std(rr_window)
            rmssd = np.sqrt(np.mean(np.diff(rr_window) ** 2))
            pnn50 = 100 * np.sum(np.abs(np.diff(rr_window)) > 50) / len(rr_window)
            features.append([rr_mean, sdnn, rmssd, pnn50])

        return np.array(features)

# Morphological features extractor | Input : Beats windows| Output : Morphological features
class MorphologicalFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self    

    def transform(self, X):
        """
        X: (n_windows, window_size, n_samples) each line is a window of beat segments
        """
        features = []
        for beat_window in X:
            window_features = []
            for beat in beat_window:
                beat_amplitude = np.max(beat) - np.min(beat)
                beat_mean = np.mean(beat)
                beat_energy = np.sum(beat ** 2)
                qrs_slope = np.max(np.diff(beat))
                
                window_features.append([beat_amplitude, beat_mean, beat_energy, qrs_slope])
            
            window_features = np.mean(window_features, axis=0)
            features.append(window_features)
        return np.array(features) # (n_windows, 4) each line is the mean morphological features of the beats in the window

# Union of HRV and morphological features | Input : RR intervals windows and Beats windows | Output : Concatenation of HRV and morphological
class ECGFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self,sample_rate, window_size, step_size = None):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size
        
        self.rr_extractor = RRIntervalsExtractor(sample_rate)
        self.beats_extractor = BeatsExtractor(sample_rate)

        self.rr_window = RRWindowing(window_size, step_size)
        self.beats_window = BeatsWindowing(window_size, step_size)

        self.hrv = HRVFeaturesExtractor()
        self.morph = MorphologicalFeaturesExtractor()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # RR branch
        rr = self.rr_extractor.transform(X)
        rr_windows = self.rr_window.transform(rr)

        # Beats branch
        beats = self.beats_extractor.transform(X)
        beat_windows = self.beats_window.transform(beats)
        
        # Alignment
        min_len = min(len(rr_windows), len(beat_windows))

        rr_windows = rr_windows[:min_len]
        beat_windows = beat_windows[:min_len]
        
        # Feature extraction
        hrv_features = self.hrv.transform(rr_windows)
        morph_features = self.morph.transform(beat_windows)
        
        return np.hstack((hrv_features, morph_features))

# Pipelines creation
sample_rate    = 130
window_size    = 30   # 30 RR intervals ≈ 30 secondes
step_size      = 15   # analyse toutes les 15 secondes
WINDOW_SAMPLES = sample_rate * 30  # 3900 samples = 30 secondes d'ECG


ecg_buffer = []   # raw ECG signal
rr_buffer  = []   # RR intervals

if random_forest_model is not None:
    stress_pipeline = Pipeline([
        ("rr_extractor", RRIntervalsExtractor(sample_rate=sample_rate)),
        ("windowing", RRWindowing(window_size=window_size, step_size=step_size)),
        ("hrv_extractor", HRVFeaturesExtractor()),
        ("random_forest", random_forest_model)
    ])
    print("Stress detection pipeline created")
else:
    print("Stress detection pipeline not created due to missing model")

try:
    scaler_anomaly = joblib.load("scaler_anomaly.pkl")
    print("Loaded scaler for anomaly detection")
except FileNotFoundError:
    print("Warning: scaler_anomaly.pkl not found.")
    scaler_anomaly = None
    
if isolation_forest_model is not None:
    anomaly_pipeline = Pipeline([
        ("features", ECGFeatureUnion(sample_rate=sample_rate, window_size=window_size, step_size=step_size)),
        ("standard_scaler", scaler_anomaly),
        ("isolation_forest", isolation_forest_model)
    ])
else:
    print("Anomaly detection pipeline not created due to missing model")



# Real-time analysis
def analyze_window():
    ecg = np.array(ecg_buffer)

    if len(ecg) < WINDOW_SAMPLES:
        return   # not enough data

    ts = datetime.now().strftime("%H:%M:%S")
    results = []

    try:
        if random_forest_model:
            stress = stress_pipeline.predict(ecg)
            label  = "STRESS" if 1 in stress else "Normal"
            results.append(f"Stress: {label}")
    except Exception as e:
        results.append(f"Stress: error ({e})")

    try:
        if isolation_forest_model:
            anomaly = anomaly_pipeline.predict(ecg)
            # Isolation Forest returns -1 for anomaly
            label   = "ANOMALIE" if -1 in anomaly else "Normal"
            results.append(f"Anomalie: {label}")
    except Exception as e:
        results.append(f"Anomaly: error ({e})")

    print(f"\n[{ts}] ── Analyse ──")
    for r in results:
        print(f"  {r}")
    print(f"  ECG samples : {len(ecg)} | RR intervals : {len(rr_buffer)}")


async def stream(address):
    print(f"Connecting to {address}...")
    async with BleakClient(address, timeout=30.0) as client:
        # Lister tous les services et caractéristiques disponibles
        print("\n── Services disponibles sur le H10 ──")
        for service in client.services:
            print(f"  Service: {service.uuid}")
            for char in service.characteristics:
                print(f"    Caractéristique: {char.uuid} | Propriétés: {char.properties}")
        print("─────────────────────────────────────\n")

        state = {
            "since_last": 0,    # samples accumulated since the last progress line
            "hr_seen": False,
            "ecg_seen": False,
        }

        def hr_callback(sender, data):
            if not state["hr_seen"]:
                state["hr_seen"] = True
                print(f"First HR packet received ({len(data)} bytes)")
            flag   = data[0]
            has_rr = (flag & 0x10) != 0
            if has_rr:
                offset = 3 if (flag & 0x01) else 2
                while offset + 1 < len(data):
                    rr_ms = int.from_bytes(data[offset:offset+2], "little") / 1024.0 * 1000.0
                    rr_buffer.append(rr_ms)
                    offset += 2

        def ecg_callback(sender, data):
            raw = bytes(data)
            samples = parse_ecg_packet(raw)
            if not samples:
                return
            if not state["ecg_seen"]:
                state["ecg_seen"] = True
                print(f"First ECG packet received "
                      f"({len(raw)} bytes, {len(samples)} samples, "
                      f"head={raw[:14].hex()})")

            ecg_buffer.extend(samples)
            state["since_last"] += len(samples)

            # Every ~1 s of incoming data, show a live progress line.
            if state["since_last"] >= sample_rate:
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}] ECG samples: {len(ecg_buffer):>5} / {WINDOW_SAMPLES}"
                    f" | RR: {len(rr_buffer)}",
                    flush=True,
                )
                state["since_last"] = 0

            # Analyze as soon as we have enough data
            if len(ecg_buffer) >= WINDOW_SAMPLES:
                analyze_window()
                # Keep only the latest samples (sliding window)
                del ecg_buffer[:step_size * sample_rate]

        def pmd_control_callback(sender, data):
            # Polar PMD response: [0xF0, op, meas_type, error, ...settings]
            # Mask measurement_type with 0x3F because the high bits carry flags.
            raw = bytes(data)
            hex_str = raw.hex()
            if len(raw) >= 4 and raw[0] == 0xF0:
                op   = raw[1]
                meas = raw[2] & 0x3F
                err  = raw[3]
                op_name = {PMD_START_OP: "START", PMD_STOP_OP: "STOP"}.get(op, f"0x{op:02x}")
                meas_name = {TYPE_ECG: "ECG", TYPE_ACC: "ACC"}.get(meas, f"0x{meas:02x}")
                status = "SUCCESS" if err == 0 else f"ERROR 0x{err:02x}"
                print(f"PMD response: {op_name} {meas_name} → {status}  ({hex_str})")
            else:
                print(f"PMD control: {hex_str}")

        # Subscribe to all notifications BEFORE writing PMD commands —
        # this is the order proven to work in ecg_acc_collector.py.
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
        await client.start_notify(PMD_DATA, ecg_callback)
        print("  ✓ PMD data notify: ACTIVE")

        # Clear any stale streams left by previous runs. Stop both ECG and
        # ACC because either one being stuck blocks new START commands with
        # error 0x06 (INVALID_STATE). Errors here are non-fatal.
        for stop_cmd, name in ((ECG_STOP_CMD, "ECG"), (ACC_STOP_CMD, "ACC")):
            try:
                await client.write_gatt_char(PMD_CONTROL, stop_cmd, response=True)
                await asyncio.sleep(0.2)
            except Exception:
                pass

        # START ECG
        try:
            await client.write_gatt_char(PMD_CONTROL, ECG_START_CMD, response=True)
            await asyncio.sleep(0.5)
            print("  ✓ ECG 130 Hz START sent")
        except Exception as e:
            print(f"  ⚠ START write failed: {e}")

        print("Streaming in progress — Ctrl+C to stop\n")
        elapsed = 0
        warned_5s = False
        retried = False
        while True:
            await asyncio.sleep(1)
            elapsed += 1

            # 5 s: nothing yet — likely strap / skin contact issue.
            if not state["ecg_seen"] and elapsed >= 5 and not warned_5s:
                print(
                    "⚠ No ECG packets after 5 s. Check that:\n"
                    "   • the H10 pod is snapped onto a moistened chest strap,\n"
                    "   • the strap is worn (electrodes touching skin),\n"
                    "   • no other app (Polar Flow / Polar Beat) is connected\n"
                    "     to the H10 at the same time."
                )
                warned_5s = True

            # 10 s: still nothing — retry START once in case the first
            # write was rejected because the H10 was mid-state-transition.
            if not state["ecg_seen"] and elapsed >= 10 and not retried:
                retried = True
                print("Retrying START_MEASUREMENT...")
                try:
                    await client.write_gatt_char(
                        PMD_CONTROL, ECG_STOP_CMD, response=True
                    )
                    await asyncio.sleep(1.0)
                    await client.write_gatt_char(
                        PMD_CONTROL, ECG_START_CMD, response=True
                    )
                except Exception as e:
                    print(f"Retry failed: {e}")

            # 20 s: give up — ask user to power-cycle the H10.
            if not state["ecg_seen"] and elapsed >= 20:
                print(
                    "⚠ Still no ECG after 20 s. Power-cycle the H10:\n"
                    "   take the pod off the strap (or the strap off the body)\n"
                    "   for ~30 s, then put it back on and rerun."
                )
                break
            


# ── Main ──
# Main
async def main():
    while True:
        try:
            address = await find_polar_h10()
            await stream(address)
        except KeyboardInterrupt:
            print("\nStopping the pipeline")
            break
        except Exception as e:
            print(f"Error: {e} — reconnecting in 10s...")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
