import numpy as np
import joblib
import scipy.signal as signal

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import asyncio
import time
from bleak import BleakScanner, BleakClient
import sys
import os

# Collect data via Bluetooth
collected_rr = []
collected_ecg = []

# Standard and proprietary Polar UUIDs
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
POLAR_DEVICE_NAME = "Polar H10"  # Device name used during BLE scanning

# Command to start ECG (130Hz, 24-bit)
ECG_START_CMD = bytearray([0x02, 0x01, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

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

async def collect_bluetooth_data(address, duration_seconds=30):
    collected_rr.clear()
    collected_ecg.clear()

    print(f"Connecting to device {address}...")
    async with BleakClient(address, timeout=30.0) as client:
        if not client.is_connected:
            raise Exception("Unable to connect to Polar H10")

        # --- RR Intervals + Heart Rate Callback ---
        def hr_callback(sender, data):
            flag = data[0]
            # RR Intervals are present if bit 4 is set
            has_rr = (flag & 0x10) != 0
            if has_rr:
                # RR starts after flag (1 byte) and BPM (1 or 2 bytes)
                offset = 2 if not (flag & 0x01) else 3
                while offset + 1 < len(data):
                    # Value in 1/1024 seconds, converted to ms
                    rr_raw = int.from_bytes(data[offset:offset+2], "little")
                    rr_ms = (rr_raw / 1024.0) * 1000.0
                    collected_rr.append(rr_ms)
                    offset += 2

        # --- Raw ECG Callback ---
        def ecg_callback(sender, data):
            # H10 sends packets with a 10-byte header
            # ECG data starts at index 10. Each sample is 3 bytes (24-bit).
            for i in range(10, len(data) - 2, 3):
                # 24-bit signed conversion
                sample = int.from_bytes(data[i:i+3], byteorder='little', signed=True)
                collected_ecg.append(sample)

        # Start notifications
        print("Starting data streams...")
        await client.start_notify(HR_UUID, hr_callback)
        
        # Write to PMD_CONTROL to request ECG
        await client.write_gatt_char(PMD_CONTROL, ECG_START_CMD, response=True)
        await client.start_notify(PMD_DATA, ecg_callback)

        print(f"Collecting for {duration_seconds} seconds...")
        await asyncio.sleep(duration_seconds)

        # Stop streams
        await client.stop_notify(PMD_DATA)
        await client.stop_notify(HR_UUID)
        
    print("Collection completed.")
    return np.array(collected_ecg), np.array(collected_rr)




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
sample_rate = 130
window_size = 10
step_size = 5

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

if isolation_forest_model is not None:
    anomaly_pipeline = Pipeline([
        ("features", ECGFeatureUnion(sample_rate=sample_rate, window_size=window_size, step_size=step_size)),
        ("standard_scaler", StandardScaler()),
        ("isolation_forest", isolation_forest_model)
    ])
else:
    print("Anomaly detection pipeline not created due to missing model")


def run_pipeline(ecg_signal):
    # Stress
    stress_pred = stress_pipeline.predict(ecg_signal)
    # Anomalie
    anomaly_pred = anomaly_pipeline.predict(ecg_signal)
    return stress_pred, anomaly_pred

async def main():
    # Main continuous loop for data collection and prediction.
    # First discover the Polar H10 address automatically, then collect ECG data repeatedly.
    try:
        device_address = await find_polar_h10(timeout=20)
        while True:
            ecg_signal, _ = await collect_bluetooth_data(device_address, duration_seconds=30)
            stress, anomaly = run_pipeline(ecg_signal)
            print(f"Stress: {stress}, Anomaly: {anomaly}")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping pipeline")
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    # Entry point: run the async main loop.
    asyncio.run(main())
