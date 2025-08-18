import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
CSV_PATH = os.path.join('ECLSS', 'data', 'telemetry_logs', 'normal_run.csv')
FEATURES = ['ppO2', 'ppCO2', 'humidity', 'ppO21', 'ppCO21', 'humidity1']  # Add more features as needed
SEQ_LENGTH = 10  # Number of timesteps per sequence
TEST_SIZE = 0.2  # Fraction of data for test set

# --- Load Data ---
df = pd.read_csv(CSV_PATH)

# --- Select Features ---
data = df[FEATURES].values

# --- Handle Missing Data ---
# (Simple forward fill, then back fill, then drop any remaining NaNs)
df[FEATURES] = df[FEATURES].fillna(method='ffill').fillna(method='bfill')
data = df[FEATURES].dropna().values

# --- Normalize Features ---
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- Create Sequences ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

X, y = create_sequences(data_scaled, SEQ_LENGTH)

# --- Split Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False  # No shuffle to preserve time order
)

# --- Save Processed Data ---
np.savez(os.path.join('data', 'lstm_train_data.npz'), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# --- Save Scaler for Later Use ---
import joblib
scaler_path = os.path.join('data', 'lstm_scaler.save')
joblib.dump(scaler, scaler_path)

print(f"Data prepared and saved. Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
print(f"Scaler saved to {scaler_path}") 