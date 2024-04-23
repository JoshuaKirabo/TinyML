import os
import pandas as pd
import numpy as np
from scipy import fftpack
from sklearn.preprocessing import StandardScaler

# Define paths and gesture labels
base_path = 'Gestures'
output_path = 'CleanData'
gestures = ['bellingham', 'josh_celebration', 'josh_thumbs_up', 'sip_tea']
window_size = 8  # Window size for segmenting data
overlap = 4      # Overlap between windows

# Column names for the CSV files
column_names = ['Timestamp', 'Accel X', 'Accel Y', 'Accel Z']

# Create output directory if it does not exist
os.makedirs(output_path, exist_ok=True)

# Process each gesture folder
for gesture in gestures:
    gesture_path = os.path.join(base_path, gesture)
    
    # Collect and concatenate data for calculating global statistics
    all_data = []
    for filename in os.listdir(gesture_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(gesture_path, filename)
            # Read the file with no header and specify column names
            data = pd.read_csv(file_path, header=None, names=column_names)
            all_data.append(data)
    
    # Concatenate all data for this gesture to calculate mean and std
    gesture_data = pd.concat(all_data)
    scaler = StandardScaler()
    scaler.fit(gesture_data[['Accel X', 'Accel Y', 'Accel Z']])

    # Re-process each file for standardization and feature extraction
    for filename in os.listdir(gesture_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(gesture_path, filename)
            data = pd.read_csv(file_path, header=None, names=column_names)

            # Standardize data
            data[['Accel X', 'Accel Y', 'Accel Z']] = scaler.transform(data[['Accel X', 'Accel Y', 'Accel Z']])

            # Initialize a DataFrame for features
            features_df = pd.DataFrame()

            # Windowing and feature extraction
            start = 0
            while start + window_size <= len(data):
                end = start + window_size
                window = data.iloc[start:end]

                # Extract features
                features = {
                'mean_x': window['Accel X'].mean(),
                'std_x': window['Accel X'].std(),
                'max_x': window['Accel X'].max(),
                'min_x': window['Accel X'].min(),
                'fft_peak_x': np.max(np.abs(fftpack.fft(window['Accel X'].to_numpy()))),  # Convert to numpy array here
                'gesture': gesture  # Label each window with the gesture type
            }

                # Append features to DataFrame
                features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)

                # Move the window with overlap
                start += window_size - overlap

            # Save the extracted features to a new CSV file in the CleanData directory
            output_file_path = os.path.join(output_path, 'features_' + filename)
            features_df.to_csv(output_file_path, index=False)
