import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# --- Configuration ---
DATA_PATH = os.path.join('data', 'lstm_train_data.npz')
MODEL_PATH = os.path.join('data', 'lstm_model2.h5')
EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 5  # For early stopping

# --- Load Data ---
data = np.load(DATA_PATH)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

seq_length = X_train.shape[1]
num_features = X_train.shape[2]

print(f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

# --- Build LSTM Model ---
model = Sequential([
    LSTM(64, input_shape=(seq_length, num_features)),
    Dense(num_features)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Train Model ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)]
)

# --- Save Model ---
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- Plot Training History ---
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('LSTM Training History')
plt.tight_layout()
plt.savefig(os.path.join('data', 'lstm_training_history.png'))
plt.show()

# --- Evaluate Model ---
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)
print(f"Test MSE: {mse:.6f}")

# --- Plot Predictions vs Actuals for all features ---
feature_names = ['ppO2', 'ppCO2', 'humidity', 'ppO21', 'ppCO21', 'humidity1']

# Create a large figure with subplots for all features
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, feature_name in enumerate(feature_names):
    axes[i].plot(y_test[:, i], label='Actual', alpha=0.7, linewidth=1)
    axes[i].plot(y_pred[:, i], label='Predicted', alpha=0.7, linewidth=1)
    axes[i].set_xlabel('Sample')
    axes[i].set_ylabel(f'{feature_name}')
    axes[i].set_title(f'LSTM Prediction vs Actual - {feature_name}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('data', 'lstm_pred_vs_actual_all_features.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Create correlation plots for each feature ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature_name in enumerate(feature_names):
    axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=20)
    axes[i].plot([y_test[:, i].min(), y_test[:, i].max()], 
                 [y_test[:, i].min(), y_test[:, i].max()], 'r--', linewidth=2)
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
    axes[i].set_title(f'Correlation Plot - {feature_name}')
    axes[i].grid(True, alpha=0.3)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                 transform=axes[i].transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join('data', 'lstm_correlation_plots_all_features.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Create error distribution plots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature_name in enumerate(feature_names):
    errors = y_pred[:, i] - y_test[:, i]
    axes[i].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_xlabel('Prediction Error')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Error Distribution - {feature_name}')
    axes[i].grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    axes[i].text(0.05, 0.95, f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}', 
                 transform=axes[i].transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join('data', 'lstm_error_distributions_all_features.png'), dpi=300, bbox_inches='tight')
plt.show() 