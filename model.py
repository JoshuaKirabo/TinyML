import time
import numpy as np
from scipy import fftpack
from joblib import load
from mpu6050 import mpu6050  # Assuming using mpu6050 accelerometer

# Define the window size and overlap for real-time prediction
window_size = 8  # Must match the window size used during training
overlap = 4      # Must match the overlap used during training

# Initialize the accelerometer
sensor = mpu6050(0x68)

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
    # Read accelerometer data
    accel_data = sensor.get_accel_data()
    window_data.append(accel_data)

    # Once we have enough data for one window, predict the gesture
    if len(window_data) == window_size:
        features = extract_features(window_data)
        gesture = predict_gesture(features)
        print(f"Predicted gesture: {gesture}")

        # Move the window with overlap
        window_data = window_data[window_size - overlap:]

    time.sleep(0.01)  # Adjust the sleep time as needed
