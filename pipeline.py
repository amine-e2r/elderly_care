import numpy as np
import joblib
import scipy.signal as signal

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Import the models
random_forest_model = joblib.load("random_forest_model.pkl")
isolation_forest_model = joblib.load("isolation_forest_model.pkl")

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
        beats = []
        for peak in peaks:
            start = max(0, peak - int(0.2 * self.sample_rate))  # 200ms before
            end = min(len(X), peak + int(0.4 * self.sample_rate))  # 400ms after
            beats.append(X[start:end])
        return np.array(beats) # (n_beats, n_samples) each line is a beat segment

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

        return features

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
            for beat in beat_window:
                beat_amplitude = np.max(beat) - np.min(beat)
                beat_mean = np.mean(beat)
                beat_energy = np.sum(beat ** 2)
                qrs_slope = np.max(np.diff(beat))
                features.append([beat_amplitude, beat_mean, beat_energy, qrs_slope])
        return features

# Union of HRV and morphological features | Input : RR intervals windows and Beats windows | Output : Concatenation of HRV and morphological
class FeatureUnion(BaseEstimator, TransformerMixin):
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

sample_rate = 130
window_size = 10
step_size = 5

stress_pipeline = Pipeline([
    ("rr_extractor", RRIntervalsExtractor(sample_rate=sample_rate)),
    ("windowing", RRWindowing(window_size=window_size, step_size=step_size)),
    ("hrv_extractor", HRVFeaturesExtractor()),
    ("random_forest", random_forest_model)
])

anomaly_pipeline = Pipeline([
    ("features", FeatureUnion(sample_rate=sample_rate, window_size=window_size, step_size=step_size)),
    ("standard_scaler", StandardScaler()),
    ("isolation_forest", isolation_forest_model)
])


