# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:12:32 2024

@author: najdi.boubker
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

# Load the data
file_path = 'Train_data_H.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)
data = data.drop(data.index[[0, 2803, 3672]])

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Calculate RMS for each row
def calculate_rms(data):
    return np.sqrt(np.mean(np.square(data), axis=1))

# Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized_features, scaler

# Fit Gaussian Process Regression model
def fit_gpr(x, y):
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x.reshape(-1, 1), y)
    return gp

# Smooth the health indicator using exponential moving average
def smooth_health_indicator(health_indicator, alpha=0.1):
    ema = [health_indicator[0]]
    for point in health_indicator[1:]:
        ema.append(alpha * point + (1 - alpha) * ema[-1])
    return np.array(ema)

# Normalize the health indicator to ensure it starts at 0 and ends at 1
def scale_health_indicator(health_indicator):
    health_indicator -= health_indicator[0]  # Make the start 0
    health_indicator /= health_indicator[-1]  # Make the end 1
    return health_indicator

# Process each bearing separately
start_idx = 0
all_health_indicators = []
for i, length in enumerate(bearing_lengths):
    # Extract the data for the current bearing
    end_idx = start_idx + length
    bearing_data = data.iloc[start_idx:end_idx, :].values
    
    # Calculate RMS
    rms = calculate_rms(bearing_data)
    
    # Normalize the RMS data
    normalized_rms, scaler = normalize_features(rms)
    
    # Fit GPR model to the normalized RMS data
    x_data = np.arange(len(normalized_rms))
    gpr_model = fit_gpr(x_data, normalized_rms)
    fitted_rms, _ = gpr_model.predict(x_data.reshape(-1, 1), return_std=True)
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_rms)
    
    # Normalize the health indicator to ensure it starts at 0 and ends at 1
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(normalized_rms, label='Normalized RMS')
    plt.plot(fitted_rms, label='Fitted GPR RMS', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'GPR Trend Labeling for Bearing {i}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value / Health Indicator')
    plt.show()
    
    # Update the starting index for the next bearing
    start_idx = end_idx

# Convert health indicators to DataFrame
health_indicators_df = pd.DataFrame(all_health_indicators, columns=['Health Indicator'])

# Save health indicators to a CSV file
health_indicators_df.to_csv('bearing_health_indicators.csv', index=False)









import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

# Load the data
file_path = 'Train_data_H.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)
data = data.drop(data.index[[0, 2803, 3672]])

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Calculate RMS for each row
def calculate_rms(data):
    return np.sqrt(np.mean(np.square(data), axis=1))

# Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized_features, scaler

# Fit Gaussian Process Regression model with RBF kernel
def fit_gpr(x, y):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x.reshape(-1, 1), y)
    return gp

# Smooth the health indicator using exponential moving average
def smooth_health_indicator(health_indicator, alpha=0.1):
    ema = [health_indicator[0]]
    for point in health_indicator[1:]:
        ema.append(alpha * point + (1 - alpha) * ema[-1])
    return np.array(ema)

# Normalize the health indicator to ensure it starts at 0 and ends at 1 and is positive
def scale_health_indicator(health_indicator):
    health_indicator -= health_indicator.min()  # Shift to make positive
    health_indicator /= health_indicator.max()  # Scale to range [0, 1]
    return health_indicator

# Process each bearing separately
start_idx = 0
all_health_indicators = []
for i, length in enumerate(bearing_lengths):
    # Extract the data for the current bearing
    end_idx = start_idx + length
    bearing_data = data.iloc[start_idx:end_idx, :].values
    
    # Calculate RMS
    rms = calculate_rms(bearing_data)
    
    # Normalize the RMS data
    normalized_rms, scaler = normalize_features(rms)
    
    # Fit GPR model to the normalized RMS data
    x_data = np.arange(len(normalized_rms))
    gpr_model = fit_gpr(x_data, normalized_rms)
    fitted_rms, _ = gpr_model.predict(x_data.reshape(-1, 1), return_std=True)
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_rms)
    
    # Normalize the health indicator to ensure it starts at 0, ends at 1, and is positive
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(normalized_rms, label='Normalized RMS')
    plt.plot(fitted_rms, label='Fitted GPR RMS', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'GPR Trend Labeling for Bearing {i}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value / Health Indicator')
    plt.show()
    
    # Update the starting index for the next bearing
    start_idx = end_idx

# Convert health indicators to DataFrame
health_indicators_df = pd.DataFrame(all_health_indicators, columns=['Health Indicator'])

# Save health indicators to a CSV file
health_indicators_df.to_csv('bearing_health_indicators.csv', index=False)




import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

# Load the data
file_path = 'Train_data_H.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)
data = data.drop(data.index[[0, 2803, 3672]])

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Calculate RMS for each row
def calculate_rms(data):
    return np.sqrt(np.mean(np.square(data), axis=1))

# Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized_features, scaler

# Fit Gaussian Process Regression model with RBF kernel
def fit_gpr(x, y):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x.reshape(-1, 1), y)
    return gp

# Smooth the health indicator using exponential moving average
def smooth_health_indicator(health_indicator, alpha=0.1):
    ema = [health_indicator[0]]
    for point in health_indicator[1:]:
        ema.append(alpha * point + (1 - alpha) * ema[-1])
    return np.array(ema)

# Normalize the health indicator to ensure it starts at 0 and ends at 1
def scale_health_indicator(health_indicator):
    health_indicator -= health_indicator.min()  # Shift to make positive
    health_indicator /= health_indicator.max()  # Scale to range [0, 1]
    return health_indicator

# Process each bearing separately
start_idx = 0
all_health_indicators = []
for i, length in enumerate(bearing_lengths):
    # Extract the data for the current bearing
    end_idx = start_idx + length
    bearing_data = data.iloc[start_idx:end_idx, :].values
    
    # Calculate RMS
    rms = calculate_rms(bearing_data)
    
    # Normalize the RMS data
    normalized_rms, scaler = normalize_features(rms)
    
    # Fit GPR model to the normalized RMS data
    x_data = np.arange(len(normalized_rms))
    gpr_model = fit_gpr(x_data, normalized_rms)
    fitted_rms, _ = gpr_model.predict(x_data.reshape(-1, 1), return_std=True)
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_rms)
    
    # Normalize the health indicator to ensure it starts at 0, ends at 1, and is positive
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(normalized_rms, label='Normalized RMS')
    plt.plot(fitted_rms, label='Fitted GPR RMS', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'GPR Trend Labeling for Bearing {i}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value / Health Indicator')
    plt.show()
    
    # Update the starting index for the next bearing
    start_idx = end_idx

# Convert health indicators to DataFrame
health_indicators_df = pd.DataFrame(all_health_indicators, columns=['Health Indicator'])

# Save health indicators to a CSV file
health_indicators_df.to_csv('bearing_health_indicators.csv', index=False)













