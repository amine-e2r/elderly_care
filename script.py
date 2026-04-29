import pandas as pd
import matplotlib.pyplot as plt

abnormal = pd.read_csv('dataset/mitbih/ptbdb_abnormal.csv', header=None)
normal = pd.read_csv('dataset/mitbih/ptbdb_normal.csv', header=None)


def extract_ecg_signal(row_values):
    # PTBDB files often store the class label in the last column.
    if row_values.size > 1 and row_values[-1] in (0, 1):
        return row_values[:-1]
    return row_values


normal_signal = extract_ecg_signal(normal.iloc[0].to_numpy(dtype=float))
abnormal_signal = extract_ecg_signal(abnormal.iloc[0].to_numpy(dtype=float))

plt.figure(figsize=(12, 5))
plt.plot(normal_signal, label='Normal ECG', linewidth=1.4)
plt.plot(abnormal_signal, label='Abnormal ECG', linewidth=1.4)
plt.title('One Normal vs One Abnormal ECG')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()