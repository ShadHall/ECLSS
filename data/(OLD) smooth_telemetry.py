import pandas as pd
import os

# Paths
input_csv = os.path.join('ECLSS', 'data', 'telemetry_logs', 'normal_run.csv')
output_csv = os.path.join('ECLSS', 'data', 'telemetry_logs', 'smoothed_normal_run.csv')

# Smoothing window size
WINDOW = 10

# Read the CSV
df = pd.read_csv(input_csv)

# Columns to smooth (all except timestamp and step)
numeric_cols = [col for col in df.columns if col not in ['timestamp', 'step']]

# Apply moving average smoothing
df_smooth = df.copy()
df_smooth[numeric_cols] = df[numeric_cols].rolling(window=WINDOW, min_periods=1, center=True).mean()

# Write to new CSV
df_smooth.to_csv(output_csv, index=False)

print(f"Smoothed data written to: {output_csv}") 