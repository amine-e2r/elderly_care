import os
import scipy
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from copy import deepcopy
import os
os.makedirs("dataset/wesad_modified", exist_ok=True)


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
    
# We apply a bandpass filter to the ECG signal to remove noise and artifacts

def bandpass_filter(X, fs):
    low,high = 5, 40 
    nyq = 0.5 * fs
    b, a = butter(2, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, X) 

# We extract the RR intervals from the ECG signal for each patient
sampling_rate_wesad = 700  # WESAD ECG sampling frequency in Hz (samplesper second)
patients_rr_intervals = {}

for subject, patient_df in patients_data.items():
    ecg_signal = patient_df["ECG"].to_numpy()
    labels_signal = patient_df["label"].to_numpy()

    # Apply bandpass filter
    ecg_signal = bandpass_filter(ecg_signal, sampling_rate_wesad)

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

window_size = 30  # 30 heartbeats
window_rows = []

for subject, rr_df in patients_rr_intervals.items():
    rr_values = rr_df["rr_interval_ms"].to_numpy()
    label_values = rr_df["label"].to_numpy()

    if len(rr_values) < window_size:
        continue

    for start_idx in range(0, len(rr_values) - window_size + 1, 15):

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
wesad_window_df["label"] = (wesad_window_df["label"] == 2).astype(int)

# We load collected data
COLLECTED_DATA_PATH = "dataset/collected/cleaned/"
sampling_rate_collected = 130


collected_data = {}
collected_rr_intervals = {}

# Load each CSV file from collected/cleaned
for filename in os.listdir(COLLECTED_DATA_PATH):
    if not filename.lower().endswith(".csv"):
        continue
    
    file_path = os.path.join(COLLECTED_DATA_PATH, filename)
    df = pd.read_csv(file_path)
    
    # Last two columns are subject and stress (added by script_clean_raw_data)
    if df.shape[1] >= 2:
        subject = df["subject"].iloc[0]  # subject column
        stress = df["stress"].iloc[0]  # stress column
        # All columns except last two are ECG signal
        ecg_signal = df.iloc[:, :-2].values.flatten()
    else:
        # Fallback: entire row is ECG
        subject = filename
        stress = 1 if "stress" in filename.lower() else 0
        ecg_signal = df.iloc[:, :].values.flatten()
    
    ecg_signal = bandpass_filter(ecg_signal, sampling_rate_collected)
    # Store in dictionary by filename
    collected_data[filename] = {
        "ECG": ecg_signal,
        "subject": subject,
        "stress": stress
    }
    
    # Calculate RR intervals
    peaks, _ = scipy.signal.find_peaks(ecg_signal,
        distance=int(0.3 * sampling_rate_collected),
        prominence=max(0.15, 0.5 * np.std(ecg_signal)),
    )
    
    if len(peaks) > 1:
        rr_interval_ms = np.diff(peaks) * 1000 / sampling_rate_collected
        collected_rr_intervals[filename] = pd.DataFrame({
            "rr_interval_ms": rr_interval_ms,
            "subject": subject,
            "stress": stress
        })

# Extract HRV features from collected data
collected_window_rows = []

for filename, rr_df in collected_rr_intervals.items():
    if rr_df.empty or len(rr_df) < window_size:
        continue
    
    rr_values = rr_df["rr_interval_ms"].to_numpy()
    subject = rr_df["subject"].iloc[0]
    stress = rr_df["stress"].iloc[0]
    
    for start_idx in range(0, len(rr_values) - window_size + 1, 15):

        end_idx = start_idx + window_size
        rr_window = rr_values[start_idx:end_idx]
        
        features = extract_hrv_features(pd.DataFrame({"rr_interval_ms": rr_window}))
        features["label"] = stress
        features["subject"] = subject
        features["start_idx"] = start_idx
        features["end_idx"] = end_idx
        
        collected_window_rows.append(features)

collected_window_df = pd.DataFrame(collected_window_rows) if collected_window_rows else pd.DataFrame()

print(f"WESAD data shape: {wesad_window_df.shape}")
print(f"Collected data shape: {collected_window_df.shape}")

# We combine WESAD and collected data
combined_df = pd.concat([wesad_window_df, collected_window_df], ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

combined_df.to_csv("dataset/wesad_modified/combined_dataset.csv", index=False)


# ---- PREPARE DATA FOR CLASSIFICATION ----
df_hrv = pd.read_csv("dataset/wesad_modified/combined_dataset.csv")

df_hrv = df_hrv.dropna().reset_index(drop=True)

# ---- RANDOM FOREST ----

feature_cols = [c for c in df_hrv.columns if c not in ["label", "subject", "stress", "start_idx", "end_idx"]]
X = df_hrv[feature_cols]
y = df_hrv["label"]
groups = df_hrv["subject"]

print(y.unique(), y.value_counts())

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

# Test best_forest on the full prepared dataset
y_pred_all = best_forest.predict(X)

print("Accuracy on full dataset:", accuracy_score(y, y_pred_all))
print(classification_report(y, y_pred_all))

cm = confusion_matrix(y, y_pred_all)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-stress", "Stress"],
    yticklabels=["Non-stress", "Stress"],
)
plt.title("Confusion Matrix - best_forest (full dataset)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# save the model
joblib.dump(best_forest, "random_forest_model.pkl")