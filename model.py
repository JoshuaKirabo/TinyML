import smbus2 as smbus
import time
import numpy as np
from scipy import fftpack
from joblib import load
import pandas as pd

# Constants
window_size = 8  # Matches the window size used during training
overlap = 4      # Matches the overlap used during training

# Initialize I2C bus
bus = smbus.SMBus(1)  # Usually bus 1 on newer Raspberry Pi models

# MPU6050 addresses and registers
DEVICE_ADDRESS = 0x68  # MPU6050 device address
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

# MPU6050 configuration for reading
bus.write_byte_data(DEVICE_ADDRESS, PWR_MGMT_1, 0)  # Wake the sensor up

# Load the trained model
model = load('gesture_recognition_model.joblib')

# Function to extract features from data
def extract_features(data):
    features = []
    for axis in ['x', 'y', 'z']:
        axis_data = np.array([d[axis] for d in data])
        features.append(np.mean(axis_data))
        features.append(np.std(axis_data))
    return features[:5]  # Adjust this line to only include the necessary features


# Function to predict gesture from features
def predict_gesture(features):
    # Ensure these names match exactly those used during model training
    feature_names = ['mean_x', 'std_x', 'max_x', 'min_x', 'fft_peak_x']
    features_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(features_df)
    return prediction[0]

# Real-time data acquisition and prediction loop
window_data = []

while True:
    # Read acceleration data from MPU6050
    accel_data = []
    for i in range(3):  # Reading X, Y, Z axes
        high = bus.read_byte_data(DEVICE_ADDRESS, ACCEL_XOUT_H + i * 2)
        low = bus.read_byte_data(DEVICE_ADDRESS, ACCEL_XOUT_H + i * 2 + 1)
        value = (high << 8) | low
        if value > 32768:
            value -= 65536  # Convert to signed value if necessary
        accel_data.append(value)

    # Append new data to the sliding window
    window_data.append({'x': accel_data[0], 'y': accel_data[1], 'z': accel_data[2]})

    # Predict gesture when enough data is collected
    if len(window_data) >= window_size:
        features = extract_features(window_data)
        gesture = predict_gesture(features)
        print(f"Predicted gesture: {gesture}")

        # Move the window with overlap
        window_data = window_data[overlap:]

    time.sleep(0.01)  # Small delay to prevent excessive CPU usage
