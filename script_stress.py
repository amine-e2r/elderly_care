import os
import scipy
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from copy import deepcopy


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