import pandas as pd
import os

#------------------------------! DATA CLEANING !------------------------------

# We load the raw data collected (dataset/collected/raw)
RAW_DATA_PATH = "dataset/collected/raw/"
cleaned_data_path = "dataset/collected/cleaned/"
merged_data_path = "dataset/collected/merged/"

os.makedirs(cleaned_data_path, exist_ok=True)
os.makedirs(merged_data_path, exist_ok=True)

all_data = []

for filename in os.listdir(RAW_DATA_PATH):
    if filename.endswith(".csv"):
        file_path = os.path.join(RAW_DATA_PATH, filename)
        df = pd.read_csv(file_path)
        columns_to_remove = ["timestamp", "timestamp_unix", "timestamp_ns_polar"]
        df_clean = df.drop(columns=columns_to_remove, errors='ignore')
        
        # Extract subject name from filename (e.g., "amine_records_calm.csv" -> "amine")
        subject_name = filename.split("_")[0]
        
        # Default: mark as stressed if filename contains "stress"
        df_clean["stress"] = 1 if "stress" in filename.lower() else 0
        
        # Special case: chinh_record_1.csv - mark lines 161073..276936 as stressed
        if filename.lower() == "chinh_record_1.csv":
            df_clean["stress"] = 0  # start with all non-stressed
            start_idx = 161073
            end_idx = 276936
            if len(df_clean) > start_idx:
                end_loc = min(end_idx, len(df_clean))
                df_clean.loc[start_idx:end_loc, "stress"] = 1
        
        df_clean["subject"] = subject_name
        
        # We save the cleaned data to a new directory (dataset/collected/cleaned)
        cleaned_file_path = os.path.join(cleaned_data_path, filename)
        df_clean.to_csv(cleaned_file_path, index=False)

        all_data.append(df_clean)


