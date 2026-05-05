import os
import numpy as np
import pickle
import sklearn
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from copy import deepcopy
from sklearn.model_selection import GroupKFold

#------------------------------! STRESS DETECTION !------------------------------

# ----PREPROCESSING----

#We load the data from the WESAD dataset
DATASET_PATH = "dataset/WESAD/"

patients_data = {}

for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)

    if os.path.isdir(subject_path):
        file_path = os.path.join(subject_path, f"{subject}.pkl")

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            ecg = data['signal']['chest']['ECG'].flatten()
            labels = data['label']

            df = pd.DataFrame({
                "ECG": ecg,
                "label": labels
            })

            patients_data[subject] = df

# We clean the data by removing the labels that are not relevant for our analysis (0, 5, 6, 7)
labels_to_remove = [0, 5, 6, 7]

for subject in patients_data:
    df = patients_data[subject]
    
    df_clean = df[~df["label"].isin(labels_to_remove)]
    
    df_clean = df_clean.reset_index(drop=True)
    
    patients_data[subject] = df_clean

# We extract the RR intervals from the ECG signal for each patient
sampling_rate_wesad = 700  # WESAD ECG sampling frequency in Hz (samplesper second)
patients_rr_intervals = {}

for subject, patient_df in patients_data.items():
    ecg_signal = patient_df["ECG"].to_numpy()
    labels_signal = patient_df["label"].to_numpy()

    peaks, _ = scipy.signal.find_peaks(
        ecg_signal,
        distance=int(0.3 * sampling_rate_wesad),
        prominence=max(0.15, 0.5 * np.std(ecg_signal)),
    )

    rr_interval_ms = np.diff(peaks) * 1000 / sampling_rate_wesad
    rr_labels = labels_signal[(peaks[:-1] + peaks[1:]) // 2]

    patients_rr_intervals[subject] = pd.DataFrame(
        {
            "rr_interval_ms": rr_interval_ms,
            "label": rr_labels,
        }   
    )

# ---- FEATURE EXTRACTION ----

def extract_hrv_features(rr_intervals_df):
    rr_ms = np.array(rr_intervals_df["rr_interval_ms"])
    
      
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


# We use windows of 10 intervals

window_size = 10  # 10 heartbeats
window_rows = []

for subject, rr_df in patients_rr_intervals.items():
    rr_values = rr_df["rr_interval_ms"].to_numpy()
    label_values = rr_df["label"].to_numpy()

    if len(rr_values) < window_size:
        continue

    for start_idx in range(0, len(rr_values) - window_size + 1, window_size):
        end_idx = start_idx + window_size
        rr_window = rr_values[start_idx:end_idx]
        label_window = label_values[start_idx:end_idx]

        # Majority label in the window
        majority_label = pd.Series(label_window).value_counts().idxmax()

        features = extract_hrv_features(pd.DataFrame({"rr_interval_ms": rr_window}))
        features["label"] = majority_label
        features["subject"] = subject
        features["start_idx"] = start_idx
        features["end_idx"] = end_idx

        window_rows.append(features)

wesad_window_df = pd.DataFrame(window_rows)

wesad_window_df.to_csv("dataset/wesad_hrv_dataset_made_10.csv", index=False)

# ---- RANDOM FOREST ----

df_hrv = pd.read_csv("dataset/wesad_hrv_dataset_made_10.csv")

df_hrv = df_hrv[df_hrv["label"].isin([1, 2, 3, 4])].copy()
df_hrv["stress"] = (df_hrv["label"] == 2).astype(int)
df_hrv = df_hrv.dropna().reset_index(drop=True)

feature_cols = [c for c in df_hrv.columns if c not in ["label", "subject", "stress", "start_idx", "end_idx"]]
X = df_hrv[feature_cols]
y = df_hrv["stress"]
groups = df_hrv["subject"]

n_splits = min(6, groups.nunique())
if n_splits < 2:
    raise ValueError("Need at least 2 different subjects for GroupKFold.")

gkf = GroupKFold(n_splits=n_splits)

fold_scores = []
best_score = -np.inf
best_forest = None
best_fold = None

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    forest = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    forest.fit(X_train, y_train)
    fold_pred = forest.predict(X_test)
    fold_score = accuracy_score(y_test, fold_pred)

    fold_scores.append(fold_score)

    if fold_score > best_score:
        best_score = fold_score
        best_forest = deepcopy(forest)
        best_fold = fold_idx

print("Dataset shape:", df_hrv.shape)
print("Features:", feature_cols)
print("Number of subjects:", groups.nunique())
print("GroupKFold splits:", n_splits)
print("Accuracy per fold:", np.array(fold_scores))
print("Mean accuracy:", np.mean(fold_scores))
print("Std accuracy:", np.std(fold_scores))
print("Best fold:", best_fold)
print("Best fold accuracy:", best_score)
print("Best forest kept in variable: best_forest")

stress_classifier = best_forest

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
                "std_energy", "mean_qrs_slope", "mean_beat_std",]

# We prepare the feature matrices and labels for training and testing
X_train = train[feature_cols]
X_test = test[feature_cols]
y_test = test["label"]

# Scale the features for better performance of Isolation Forest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

isolation_forest = IsolationForest(n_estimators=100, 
                                   contamination=0.1, 
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







