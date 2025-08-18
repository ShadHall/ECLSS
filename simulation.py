import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import json
import time
from datetime import datetime
from noise import PPCO2TrueValue, PPCO2Sensor, PPO2TrueValue, PPO2Sensor, HumidityTrueValue, HumiditySensor
import csv
from tensorflow.keras.models import load_model
import joblib

# Simulation settings
real_time_mode = True  # Set to False to run as fast as possible
simulation_speed = 1.0  # 1.0 = real-time (1 second per step), 2.0 = 2x faster, etc.


# File path to the JSON file
json_file_path = 'jsonfile/sim_data.json'

# URL to post to
# url = 'https://daphne-at-lab.selva-research.com/api/at/receiveHeraFeed'
url = 'http://localhost:8002/api/at/receiveHeraFeed'


PARAMETER_INFO1 = {
    "ppO2": {"DisplayName": "Cabin_ppO2", "Id": 43, "ParameterGroup": "L1", "NominalValue": 163.81,
             "UpperCautionLimit": 175.0, "UpperWarningLimit": 185.0, "LowerCautionLimit": 155.0,
             "LowerWarningLimit": 145.0, "Divisor": 100, "Name": "ppO2", "Unit":"mmHg"},
    "ppCO2": {"DisplayName": "Cabin_ppCO2", "Id": 44, "ParameterGroup": "L1", "NominalValue": 0.4,
              "UpperCautionLimit": 4.5, "UpperWarningLimit": 6.0, "LowerCautionLimit": -1.0,
              "LowerWarningLimit": -2.0, "Divisor": 100, "Name": "ppCO2", "Unit":"mmHg"},
    "humidity": {"DisplayName": "Humidity", "Id": 45, "ParameterGroup": "L1", "NominalValue": 52,
              "UpperCautionLimit": 61, "UpperWarningLimit": 70, "LowerCautionLimit": 50,
              "LowerWarningLimit": 40, "Divisor": 1, "Name": "Humidity", "Unit":"L"},
    "ppO21": {"DisplayName": "Cabin_ppO2", "Id": 46, "ParameterGroup": "L2", "NominalValue": 163.81,
             "UpperCautionLimit": 175.0, "UpperWarningLimit": 185.0, "LowerCautionLimit": 155.0,
             "LowerWarningLimit": 145.0, "Divisor": 100, "Name": "ppO2", "Unit":"mmHg"},
    "ppCO21": {"DisplayName": "Cabin_ppCO2", "Id": 47, "ParameterGroup": "L2", "NominalValue": 0.4,
              "UpperCautionLimit": 4.5, "UpperWarningLimit": 6.0, "LowerCautionLimit": -1.0,
              "LowerWarningLimit": -2.0, "Divisor": 100, "Name": "ppCO2", "Unit":"mmHg"},
    "humidity1": {"DisplayName": "Humidity", "Id": 48, "ParameterGroup": "L2", "NominalValue": 52,
              "UpperCautionLimit": 61, "UpperWarningLimit": 70, "LowerCautionLimit": 50,
              "LowerWarningLimit": 40, "Divisor": 1, "Name": "Humidity", "Unit":"L"},
    "H2O": {"DisplayName": "H2O", "Id": 49, "ParameterGroup": "Crew", "NominalValue": 3.6,
              "UpperCautionLimit": 4.5, "UpperWarningLimit": 5.0, "LowerCautionLimit": 2.5,
              "LowerWarningLimit": 2.0, "Divisor": 1, "Name": "H2O", "Unit":"L"},
    "CabinTemperature": {"DisplayName": "Cabin Temperature", "Id": 49, "ParameterGroup": "L1", "NominalValue": 72.3,
              "UpperCautionLimit": 79.0, "UpperWarningLimit": 87.7, "LowerCautionLimit": 68,
              "LowerWarningLimit": 64, "Divisor": 1, "Name": "Cabin Temperature", "Unit":"L"},
    "CabinTemperature1": {"DisplayName": "Cabin Temperature", "Id": 49, "ParameterGroup": "L2", "NominalValue": 72.3,
              "UpperCautionLimit": 79.0, "UpperWarningLimit": 87.7, "LowerCautionLimit": 68,
              "LowerWarningLimit": 64, "Divisor": 1, "Name": "Cabin Temperature", "Unit":"L"},
     "SOXIE": {"DisplayName": "SOXIE Stack Temp", "Id": 49, "ParameterGroup": "", "NominalValue": 1481,
              "UpperCautionLimit": 1629, "UpperWarningLimit": 1670, "LowerCautionLimit": 1333,
              "LowerWarningLimit": 1292, "Divisor": 1, "Name": "SOXIE Stack Temp", "Unit":"L"},
    "MOXIE": {"DisplayName": "MOXIE Compressor Temp", "Id": 49, "ParameterGroup": "", "NominalValue": 160,
              "UpperCautionLimit": 170, "UpperWarningLimit": 180, "LowerCautionLimit": 140,
              "LowerWarningLimit": 120, "Divisor": 1, "Name": "MOXIE Compressor Temp", "Unit":"L"},
    
    
    # "temperature": {"DisplayName": "Cabin_Temperature", "Id": 46, "ParameterGroup": "L1", "NominalValue": 100,
    #           "UpperCautionLimit": 79.0, "UpperWarningLimit": 87.7, "LowerCautionLimit": 68.0,
    #           "LowerWarningLimit": 64.0, "Divisor": 1, "Name": "Cabin Temperature", "Unit":"F"},
    # "temperature": {"DisplayName": "Cabin_Temperature", "Id": 46, "ParameterGroup": "L1", "NominalValue": 100,
    #           "UpperCautionLimit": 79.0, "UpperWarningLimit": 87.7, "LowerCautionLimit": 68.0,
    #           "LowerWarningLimit": 64.0, "Divisor": 1, "Name": "Cabin Temperature", "Unit":"F"},
}

# 

PARAMETER_INFO = {
    "ppO2": PARAMETER_INFO1["ppO2"],
    "ppCO2": PARAMETER_INFO1["ppCO2"],
    "ppO21": PARAMETER_INFO1["ppO21"],
    "ppCO21": PARAMETER_INFO1["ppCO21"],
    # "H2O": PARAMETER_INFO1["H2O"],
    "humidity": PARAMETER_INFO1["humidity"],
    "humidity1": PARAMETER_INFO1["humidity1"],
    # "CabinTemperature": PARAMETER_INFO1["CabinTemperature"],
    # "CabinTemperature1": PARAMETER_INFO1["CabinTemperature1"],
    # "SOXIE": PARAMETER_INFO1["SOXIE"],
    # "MOXIE": PARAMETER_INFO1["MOXIE"],
}

cabin = {
    "ppO2": 165, 
    "ppCO2": 0.5, 
    "humidity": 52, 
    # "CabinTemperature": 80.0, 
     "ppO21": 165, 
    "ppCO21": 0.5, 
    "humidity1": 52, 
    # "CabinTemperature1": 80.0, 
    # "H2O": 3.5,
    # "SOXIE": 1480,
    # "MOXIE": 172,
}

# Initialize the true value model (shared between sensors)
true_co2_model = PPCO2TrueValue()
true_o2_model = PPO2TrueValue()
true_humidity_model = HumidityTrueValue()
humidity_sensor_model = HumiditySensor()

sensor_parameters = {
    "ppO2": {
        'true_value': 165,  # Base ppO2 value in mmHg
        'temp': 80,            # Temperature in Fahrenheit
        'humidity': 52,        # Humidity in %RH
        'noise_model': PPO2Sensor()
    },
    "ppCO2": {
        'true_value': 0.5,  # Base ppCO2 value in mmHg
        'temp': 80,         # Temperature in Fahrenheit
        'noise_model': PPCO2Sensor()  # Initialize with sensor model
    },
    "humidity": {
        'true_value': 52,
        'temp': 80,
        'noise_model': HumiditySensor()
    },
    "ppO21": {
        'true_value': 165,  # Base ppO2 value in mmHg
        'temp': 80,            # Temperature in Fahrenheit
        'humidity': 52,        # Humidity in %RH
        'noise_model': PPO2Sensor()
    },
    "ppCO21": {
        'true_value': 0.5,  # Base ppCO2 value in mmHg
        'temp': 80,         # Temperature in Fahrenheit
        'noise_model': PPCO2Sensor()  # Initialize with sensor model
    },
    "humidity1": {
        'true_value': 52,
        'temp': 80,
        'noise_model': HumiditySensor()
    },
}

def noise_model(sensor_parameters, co2_level=None, o2_level=None, humidity=None):
    if 'noise_model' in sensor_parameters:
        # For supported features, use the sensor model with fixed time step
        if isinstance(sensor_parameters['noise_model'], PPCO2Sensor):
            return sensor_parameters['noise_model'].generate(
                time_step=1.0,
                temperature=sensor_parameters['temp'],
                co2_level=co2_level if co2_level is not None else sensor_parameters['true_value']
            )
        elif isinstance(sensor_parameters['noise_model'], PPO2Sensor):
            return sensor_parameters['noise_model'].generate(
                time_step=1.0,
                temperature=sensor_parameters['temp'],
                ppO2_level=o2_level if o2_level is not None else sensor_parameters['true_value'],
                humidity=humidity if humidity is not None else sensor_parameters.get('humidity', 50.0)
            )
        elif isinstance(sensor_parameters['noise_model'], HumiditySensor):
            return sensor_parameters['noise_model'].generate(
                humidity if humidity is not None else sensor_parameters['true_value']
            )
    else:
        # For other features, use the old noise model
        return np.random.normal(0, 0.5)  # fallback, should not be used

# Simulate over time
time_steps = 10000
counter = [0]
def simulate_step(cabin):
    """
    Simulates one time step in the ECLSS system, applying noise and subsystem dynamics.
    """
    if counter[0] > 10:
        # Get the true CO2 value with variations
        true_co2 = true_co2_model.generate(time_step=1.0, base_co2=sensor_parameters["ppCO2"]["true_value"])
        # Get the true O2 value with variations
        true_o2 = true_o2_model.generate(time_step=1.0, base_o2=sensor_parameters["ppO2"]["true_value"])
        # Get the true humidity value with variations
        true_humidity = true_humidity_model.generate(time_step=1.0)

        temp = sensor_parameters["ppO2"].get("temp", 80)

        measured_humidity = noise_model(sensor_parameters["humidity"], humidity=true_humidity)
        measured_humidity1 = noise_model(sensor_parameters["humidity1"], humidity=true_humidity)

        cabin = {
            "ppO2": true_o2 + noise_model(sensor_parameters["ppO2"], o2_level=true_o2, humidity=true_humidity),
            "ppCO2": true_co2 + noise_model(sensor_parameters["ppCO2"], co2_level=true_co2),
            "humidity": measured_humidity,
            "ppO21": true_o2 + noise_model(sensor_parameters["ppO21"], o2_level=true_o2, humidity=true_humidity),
            "ppCO21": true_co2 + noise_model(sensor_parameters["ppCO21"], co2_level=true_co2),
            "humidity1": measured_humidity1
        }
        return cabin

    counter[0] += 1
    return cabin

def post_json_to_url():
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)  # Load JSON data from the file
            
        print(f"Sending request to: {url}")

        # Make the POST request
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=json_data, headers=headers, timeout=5)
        # response = requests.post(url, json=json_data)

        # Print the response status and data
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.status_code)
            # print(f"Response content: {response.text}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print("An error occurred:", e)


def create_parameter_entry(param_name, cabin):
    """
    Creates a parameter entry for the JSON data structure.
    Args:
        param_name (str): The name of the parameter (e.g., "ppO2", "water").
        cabin (dict): The dictionary containing the current cabin state.
    Returns:
        dict: A dictionary representing the parameter entry for the JSON structure.
    """
    print(cabin)
    param_info = PARAMETER_INFO[param_name]
    current_value = cabin[param_name]

    current_value_display = current_value
    unit = param_info["Unit"]

    parameter_entry = {
        "SimulatedParameter": True,
        "DisplayName": param_info["DisplayName"],
        "DisplayValue": f"{current_value_display} {unit}",
        "Id": param_info["Id"],
        "Name": param_info["Name"],
        "ParameterGroup": param_info["ParameterGroup"],
        "NominalValue": param_info["NominalValue"],
        "UpperCautionLimit": param_info["UpperCautionLimit"],
        "UpperWarningLimit": param_info["UpperWarningLimit"],
        "LowerCautionLimit": param_info["LowerCautionLimit"],
        "LowerWarningLimit": param_info["LowerWarningLimit"],
        "Divisor": param_info["Divisor"],
        "Unit": unit,
        "SimValue": current_value_display,
        "noise": 0.01,
        "currentValue": current_value_display,
        "CurrentValue": current_value_display,
        "simulationValue": current_value_display,
        "Status": {
            "LowerWarning": current_value < param_info["LowerWarningLimit"],
            "LowerCaution": current_value < param_info["LowerCautionLimit"],
            "Nominal": param_info["LowerCautionLimit"] <= current_value <= param_info["UpperCautionLimit"],
            "UpperCaution": current_value > param_info["UpperCautionLimit"],
            "UpperWarning": current_value > param_info["UpperWarningLimit"],
            "UnderLimit": current_value < param_info["LowerWarningLimit"],
            "OverLimit": current_value > param_info["UpperWarningLimit"],
            "Caution": current_value < param_info["LowerCautionLimit"] or current_value > param_info["UpperCautionLimit"],
            "Warning": current_value < param_info["LowerWarningLimit"] or current_value > param_info["UpperWarningLimit"]
        },
        "HasDuplicationError": False,
        "DuplicationError": None
    }
    return parameter_entry

# Function to save JSON data with a unique filename
def create_json(anomaly_detected=False):
    """Save simulation telemetry data with a specific structured format."""
    folder_path = os.path.join(os.getcwd(), "jsonfile")
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"sim_data.json"
    file_path = os.path.join(folder_path, file_name)


    parameters_list = [create_parameter_entry(param_name, cabin) for param_name in PARAMETER_INFO]

    habitat_status = {
        "habitatStatus": {
            "Parameters": parameters_list,
            "MasterStatus": {
                "Caution": any(p["Status"]["Caution"] for p in parameters_list),
                "Warning": any(p["Status"]["Warning"] for p in parameters_list)
            },
            "HardwareList": [],
            "SimulationList": [],
            "Timestamp": datetime.now().strftime("MD %j %H:%M:%S"),  # Mission day format
            "AnomalyDetected": anomaly_detected
        }
    }

    # Save the JSON file
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(habitat_status, json_file, indent=4)

    #print(f"JSON file saved at: {file_path}")

    post_json_to_url()

# --- Logging Setup ---
log_file = os.path.join('ECLSS', 'data', 'telemetry_logs', 'normal_run.csv')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

def log_normal_telemetry(step, cabin):
    fieldnames = ['timestamp', 'step'] + list(cabin.keys())
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'timestamp': datetime.now().isoformat(), 'step': step}
        row.update(cabin)
        writer.writerow(row)

# --- LSTM Anomaly Detection Setup ---
LSTM_MODEL_PATH = os.path.join('data', 'lstm_model.h5')
SCALER_PATH = os.path.join('data', 'lstm_scaler.save')
FEATURES = ['ppO2', 'ppCO2', 'humidity','ppO21', 'ppCO21', 'humidity1']  # Must match training
SEQ_LENGTH = 10  # Must match training

# Load model and scaler
try:
    lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    print("LSTM model and scaler loaded for anomaly detection.")
except Exception as e:
    print(f"Warning: Could not load LSTM model or scaler: {e}")
    lstm_model = None
    scaler = None

recent_data = []  # Sliding window buffer for anomaly detection
# --- FINAL ERROR BOUNDS ---
error_bounds = [0.8660, 0.2, 2.3007, 0.8660, 0.2, 2.3007]

# For error statistics collection
all_errors = []

anomaly_threshold = 2.0  # (no longer used, but kept for reference)

# --- Hysteresis counters per feature ---
HYST_MAX = 10
HYST_THRESHOLD = 5
hysteresis_counters = {feat: 0 for feat in FEATURES}
hysteresis_alerted = {feat: False for feat in FEATURES}

# Main simulation loop with controlled or unrestricted time steps
for t in range(time_steps):
    start_time = time.time()  # Record start time of the timestep

    # subsystems = check_limits_and_control(cabin, subsystems)
    cabin = simulate_step(cabin)

    # --- LSTM Anomaly Detection ---
    current_features = [cabin[feat] for feat in FEATURES]
    recent_data.append(current_features)
    if len(recent_data) > SEQ_LENGTH:
        recent_data.pop(0)

    anomaly_detected = False
    if lstm_model is not None and scaler is not None and len(recent_data) == SEQ_LENGTH:
        # Convert to numpy array
        recent_np = np.array(recent_data)
        # Scale
        try:
            recent_scaled = scaler.transform(recent_np)
        except Exception as e:
            print(f"Scaler transform error at step {t}: {e}")
            recent_scaled = recent_np  # fallback, but not recommended

        # Prepare input for LSTM
        input_seq = np.expand_dims(recent_scaled, axis=0)  # shape: (1, SEQ_LENGTH, num_features)

        # Predict next step
        try:
            pred = lstm_model.predict(input_seq, verbose=0)[0]  # shape: (num_features,)
        except Exception as e:
            print(f"Prediction error at step {t}: {e}")
            pred = np.zeros_like(recent_scaled[-1])

        # Calculate error
        error = np.abs(pred - recent_scaled[-1])

        # Collect errors for statistics
        all_errors.append(error)

        # Update hysteresis counters per feature
        for i, (feat, bound) in enumerate(zip(FEATURES, error_bounds)):
            if error[i] > bound:
                hysteresis_counters[feat] = min(HYST_MAX, hysteresis_counters[feat] + 1)
            else:
                hysteresis_counters[feat] = max(0, hysteresis_counters[feat] - 1)

        # Trigger anomaly print on threshold crossing and set anomaly flag
        for feat in FEATURES:
            if hysteresis_counters[feat] >= HYST_THRESHOLD and not hysteresis_alerted[feat]:
                print(f"[Anomaly Detected] Step {t}: Feature '{feat}' counter reached {hysteresis_counters[feat]}")
                hysteresis_alerted[feat] = True
            if hysteresis_counters[feat] < HYST_THRESHOLD:
                hysteresis_alerted[feat] = False

        anomaly_detected = any(count >= HYST_THRESHOLD for count in hysteresis_counters.values())

    # Display error statistics at step 1000
    #if t == 999 and all_errors:
        #import numpy as np
        #all_errors_np = np.array(all_errors)
        #means = np.mean(all_errors_np, axis=0)
        #stds = np.std(all_errors_np, axis=0)
        #thresholds = means + 3 * stds
        #print("\n--- LSTM Prediction Error Statistics at Step 1000 ---")
        #for i, feat in enumerate(FEATURES):
        #    print(f"{feat}: mean={means[i]:.4f}, std={stds[i]:.4f}, suggested threshold={thresholds[i]:.4f}")
        #print("-----------------------------------------------------\n")
        # Optionally clear errors to avoid memory issues
        #all_errors.clear()

    # Save telemetry data with anomaly flag
    create_json(anomaly_detected=anomaly_detected)
    
    # Log normal telemetry to CSV
    log_normal_telemetry(t, cabin)

    # Print per-timestep counters
    counters_str = ", ".join(f"{feat}:{hysteresis_counters[feat]}" for feat in FEATURES)
    print(f"Step {t}: Counters [{counters_str}]")
   
    if real_time_mode:
        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / simulation_speed) - elapsed_time)  # Ensure non-negative sleep time
        time.sleep(sleep_time)  # Pause before next timestep


    

