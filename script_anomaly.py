import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib


#------------------------------! ANOMALY DETECTION !------------------------------

# ---- PREPROCESSING ----

mitbih_train = pd.read_csv('dataset/mitbih/mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('dataset/mitbih/mitbih_test.csv', header=None)
ptbdb_abnormal = pd.read_csv('dataset/mitbih/ptbdb_abnormal.csv', header=None)
ptbdb_normal = pd.read_csv('dataset/mitbih/ptbdb_normal.csv', header=None)


# We combine the MIT-BIH datasets and separate normal and abnormal samples
mitbih_all = pd.concat([mitbih_train, mitbih_test], ignore_index=True)

mitbih_normal = mitbih_all[mitbih_all[187] == 0].reset_index(drop=True)
mitbih_abnormal = mitbih_all[mitbih_all[187] != 0].reset_index(drop=True)

normal_merged = pd.concat([mitbih_normal, ptbdb_normal], ignore_index=True)
abnormal_merged = pd.concat([mitbih_abnormal, ptbdb_abnormal], ignore_index=True)

# ---- FEATURE EXTRACTION ----

# We use windows of 10 intervals
window_size = 10
sampling_rate_mitbih = 125

def get_rr_ms(df, label_col=187, fs=125):
    all_rr = []
    all_peaks = []
    all_signals = []
    ecg_matrix = df.drop(columns=[label_col]).to_numpy(dtype=float)

    for row in ecg_matrix:
        peaks, _ = scipy.signal.find_peaks(row, distance=int(0.3 * fs))
        if len(peaks) > 1:
            rr = np.diff(peaks) * 1000 / fs
            all_rr.append(rr)
            all_peaks.append(peaks)
            all_signals.append(row)

    rr_ms = np.concatenate(all_rr) if all_rr else np.array([])
    # Signal et peaks concaténés pour les features morphologiques
    ecg_signal = np.concatenate(all_signals) if all_signals else np.array([])
    peaks_concat = np.concatenate(all_peaks) if all_peaks else np.array([], dtype=int)

    return ecg_signal, peaks_concat, rr_ms

# HRV features based on RR intervals (mean, SDNN, RMSSD, pNN50, LF/HF ratio)

def extract_hrv_features(rr_intervals_df):
    rr_ms = np.array(rr_intervals_df)
    
      
    rr_mean = np.mean(rr_ms)
    
    sdnn = np.std(rr_ms)
    
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    pnn50 = 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
    
    fft_result = np.fft.fft(rr_ms)
    freqs = np.fft.fftfreq(len(rr_ms), d=np.mean(rr_ms) / 1000)
    power = np.abs(fft_result) ** 2
    
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    
    lf_power = np.sum(power[lf_mask])
    hf_power = np.sum(power[hf_mask])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    return {
        "RR_mean": rr_mean,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "LF/HF": lf_hf_ratio,
    }


# Morphological features for each beat (amplitude mean, std, skewness, kurtosis, energy, QRS slope)

def extract_morphological_features(beats):
    """
    beats: array 2D (n_beats, n_samples) - chaque ligne est un battement.
    """
    features_list = []

    for beat in beats:
        if len(beat) < 2:
            continue
        diff_beat = np.diff(beat)
        features_list.append({
            "beat_amplitude" : np.max(beat) - np.min(beat),
            "beat_mean"      : np.mean(beat),
            "beat_energy"    : np.sum(beat ** 2),
            "qrs_slope"      : np.max(diff_beat),
        })

    if not features_list:
        return {}

    df_beats = pd.DataFrame(features_list)
    
    return {
        "mean_amplitude" : df_beats["beat_amplitude"].mean(),
        "std_amplitude"  : df_beats["beat_amplitude"].std(),
        "mean_energy"    : df_beats["beat_energy"].mean(),
        "std_energy"     : df_beats["beat_energy"].std(),
        "mean_qrs_slope" : df_beats["qrs_slope"].mean(),
    }


# Combine HRV and morphological features with windows of 10 beats
def hrv_windows(ecg_signal, peaks, rr_ms, dataset_name, target_label, win=30, step=30, fs=125):
    rows = []
    if len(rr_ms) < win:
        return pd.DataFrame()

    for start_idx in range(0, len(rr_ms) - win + 1, step):
        end_idx   = start_idx + win
        rr_window = rr_ms[start_idx:end_idx]
        pk_window = peaks[start_idx:end_idx]

        hrv_feats  = extract_hrv_features(rr_window)
        half_win = int(0.15 * fs)
        beats = [
            ecg_signal[max(0, p - half_win) : min(len(ecg_signal), p + half_win)]
            for p in pk_window
            if min(len(ecg_signal), p + half_win) - max(0, p - half_win) > 1
        ]
        morph_feats = extract_morphological_features(beats)

        row = {**hrv_feats, **morph_feats,
               "label": target_label, "dataset": dataset_name,
               "start_idx": start_idx, "end_idx": end_idx}
        rows.append(row)

    return pd.DataFrame(rows)

normal_ecg, normal_peaks, normal_rr_ms = get_rr_ms(normal_merged, fs=sampling_rate_mitbih)
abnormal_ecg, abnormal_peaks, abnormal_rr_ms = get_rr_ms(abnormal_merged, fs=sampling_rate_mitbih)

normal_windowed_hrv = hrv_windows(normal_ecg, normal_peaks, normal_rr_ms, "normal_merged", 0, win=window_size, step=window_size, fs=sampling_rate_mitbih)
abnormal_windowed_hrv = hrv_windows(abnormal_ecg, abnormal_peaks, abnormal_rr_ms, "abnormal_merged", 1, win=window_size, step=window_size, fs=sampling_rate_mitbih)

# Creating test and train set

train_frac = 0.8
random_state = 42

normal_df = normal_windowed_hrv.copy().reset_index(drop=True)
abnormal_df = abnormal_windowed_hrv.copy().reset_index(drop=True)

normal_train = normal_df.sample(frac=train_frac, random_state=random_state)
normal_test = normal_df.drop(normal_train.index).reset_index(drop=True)

abnormal_test = abnormal_df.sample(n=min(100, len(abnormal_df)), random_state=random_state)

train = normal_train.copy()
test = pd.concat([normal_test, abnormal_test], ignore_index=True).sample(frac=1.0, random_state=random_state) # Concatenate and shuffle

# ---- ISOLATION FOREST ----

feature_cols = ["RR_mean", "SDNN", "RMSSD", "pNN50", "LF/HF",
                "mean_amplitude", "std_amplitude", "mean_energy",
                "std_energy", "mean_qrs_slope"]

# We prepare the feature matrices and labels for training and testing
X_train = train[feature_cols]
X_test = test[feature_cols]
y_test = test["label"]

# Scale the features for better performance of Isolation Forest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

isolation_forest = IsolationForest(n_estimators=100, 
                                   contamination=0.05, 
                                   max_features=len(feature_cols),
                                   random_state=42)

isolation_forest.fit(X_train_scaled)
y_pred = isolation_forest.predict(X_test_scaled)
y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary labels (1 for anomaly, 0 for normal)

# Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision (reflects the reliability of the detected anomalies): {precision:.4f}")
print(f"Recall (ability of the model to correctly detect anomalies,):    {recall:.4f}")
print(f"F1 Score (harmonic mean of precision and recall): {2 * (precision * recall) / (precision + recall + 1e-10):.4f}")

cm_if = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_if,
    annot=True,
    fmt="d", 
    cmap="Blues",
    xticklabels=["Normal", "Abnormal"],
    yticklabels=["Normal", "Abnormal"],
)
plt.title("Confusion Matrix - Isolation Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# save the model and scaler
joblib.dump(isolation_forest, "isolation_forest_model.pkl")
joblib.dump(scaler, "anomaly_scaler.pkl")