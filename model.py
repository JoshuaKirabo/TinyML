import smbus
import time
import numpy as np
from scipy import fftpack
from joblib import load

# Define the window size and overlap for real-time prediction
window_size = 8  # Must match the window size used during training
overlap = 4      # Must match the overlap used during training

# Initialize I2C bus
bus = smbus.SMBus(1)  # Use the appropriate bus number (usually 1 on newer Raspberry Pi models)

# MPU6050 addresses and registers
DEVICE_ADDRESS = 0x68  # MPU6050 device address
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

# Configure MPU6050
bus.write_byte_data(DEVICE_ADDRESS, PWR_MGMT_1, 0)

# Load the trained model
model = load('gesture_recognition_model.joblib')

# Function to preprocess data and extract features
def extract_features(data):
    features = {}
    for axis in ['x', 'y', 'z']:
        axis_data = np.array([d[axis] for d in data])
        features[f'{axis}_mean'] = np.mean(axis_data)
        features[f'{axis}_std'] = np.std(axis_data)
        features[f'{axis}_max'] = np.max(axis_data)
        features[f'{axis}_min'] = np.min(axis_data)
        features[f'{axis}_fft_peak'] = np.max(np.abs(fftpack.fft(axis_data)))
    return np.array([features[key] for key in sorted(features.keys())])

# Function to predict gesture from features
def predict_gesture(features):
    # Reshape into the format expected by the model (1, num_features)
    prediction = model.predict([features])
    return prediction[0]

# Real-time prediction loop
window_data = []

while True:
    # Read acceleration data
    accel_data = []
    for i in range(3):  # Read X, Y, Z axes
        high = bus.read_byte_data(DEVICE_ADDRESS, ACCEL_XOUT_H + i * 2)
        low = bus.read_byte_data(DEVICE_ADDRESS, ACCEL_XOUT_H + i * 2 + 1)
        value = (high << 8) | low
        if value > 32768:
            value -= 65536  # Correct signed value
        accel_data.append(value)

    # Append to window data
    window_data.append({'x': accel_data[0], 'y': accel_data[1], 'z': accel_data[2]})

    # Once we have enough data for one window, predict the gesture
    if len(window_data) == window_size:
        features = extract_features(window_data)
        gesture = predict_gesture(features)
        print(f"Predicted gesture: {gesture}")

        # Move the window with overlap
        window_data = window_data[window_size - overlap:]

    time.sleep(0.01)  # Adjust the sleep time as needed
