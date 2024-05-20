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



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
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

# Exponential function to fit
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

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
    
    # Fit exponential model to the RMS data
    x_data = np.arange(len(normalized_rms))
    initial_guess = [1, 0.001, 0]  # Initial guess for the exponential model parameters
    try:
        popt, _ = curve_fit(exponential_func, x_data, normalized_rms, p0=initial_guess, maxfev=5000)
    except RuntimeError:
        print(f"Optimal parameters not found for Bearing {i}.")
        start_idx = end_idx
        continue
    
    fitted_rms = exponential_func(x_data, *popt)
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_rms)
    
    # Normalize the health indicator to ensure it starts at 0, ends at 1, and is positive
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(normalized_rms, label='Normalized RMS')
    plt.plot(fitted_rms, label='Fitted Exponential RMS', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'Exponential Trend Labeling for Bearing {i}')
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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the data
file_path = 'Train_data_H.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)
data = data.drop(data.index[[0, 2803, 3672]])

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Calculate features for each row
def calculate_features(data):
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    kurtosis = np.mean(((data - np.mean(data, axis=1, keepdims=True))**4), axis=1) / (np.var(data, axis=1)**2)
    skewness = np.mean(((data - np.mean(data, axis=1, keepdims=True))**3), axis=1) / (np.std(data, axis=1)**3)
    crest_factor = np.max(np.abs(data), axis=1) / rms
    peak_to_peak = np.ptp(data, axis=1)
    return rms, kurtosis, skewness, crest_factor, peak_to_peak

# Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized_features, scaler

# Exponential function to fit
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

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
    
    # Calculate features
    rms, kurtosis, skewness, crest_factor, peak_to_peak = calculate_features(bearing_data)
    
    # Normalize the features
    normalized_rms, _ = normalize_features(rms)
    normalized_kurtosis, _ = normalize_features(kurtosis)
    normalized_skewness, _ = normalize_features(skewness)
    normalized_crest_factor, _ = normalize_features(crest_factor)
    normalized_peak_to_peak, _ = normalize_features(peak_to_peak)
    
    # Combine features into a composite health indicator
    composite_health_indicator = (normalized_rms + normalized_kurtosis + normalized_skewness + normalized_crest_factor + normalized_peak_to_peak) / 5
    
    # Fit exponential model to the composite health indicator
    x_data = np.arange(len(composite_health_indicator))
    initial_guess = [1, 0.001, 0]  # Initial guess for the exponential model parameters
    try:
        popt, _ = curve_fit(exponential_func, x_data, composite_health_indicator, p0=initial_guess, maxfev=5000)
    except RuntimeError:
        print(f"Optimal parameters not found for Bearing {i}.")
        start_idx = end_idx
        continue
    
    fitted_indicator = exponential_func(x_data, *popt)
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_indicator)
    
    # Normalize the health indicator to ensure it starts at 0, ends at 1, and is positive
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(composite_health_indicator, label='Composite Health Indicator')
    plt.plot(fitted_indicator, label='Fitted Exponential Composite Indicator', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'Exponential Trend Labeling for Bearing {i}')
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
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

# Load the data
file_path = 'Train_data_H.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)
data = data.drop(data.index[[0, 2803, 3672]])

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Calculate features for each row
def calculate_features(data):
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    kurtosis = np.mean(((data - np.mean(data, axis=1, keepdims=True))**4), axis=1) / (np.var(data, axis=1)**2)
    skewness = np.mean(((data - np.mean(data, axis=1, keepdims=True))**3), axis=1) / (np.std(data, axis=1)**3)
    crest_factor = np.max(np.abs(data), axis=1) / rms
    peak_to_peak = np.ptp(data, axis=1)
    return rms, kurtosis, skewness, crest_factor, peak_to_peak

# Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized_features, scaler

# Fit cubic polynomial regression model
def fit_polynomial(x, y, degree=1):
    p = Polynomial.fit(x, y, degree)
    return p

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
    
    # Calculate features
    rms, kurtosis, skewness, crest_factor, peak_to_peak = calculate_features(bearing_data)
    
    # Normalize the features
    normalized_rms, _ = normalize_features(rms)
    normalized_kurtosis, _ = normalize_features(kurtosis)
    normalized_skewness, _ = normalize_features(skewness)
    normalized_crest_factor, _ = normalize_features(crest_factor)
    normalized_peak_to_peak, _ = normalize_features(peak_to_peak)
    
    # Combine features into a composite health indicator
    composite_health_indicator = (normalized_rms + normalized_kurtosis + normalized_skewness + normalized_crest_factor + normalized_peak_to_peak) / 5
    
    # Fit cubic polynomial regression model to the composite health indicator
    x_data = np.arange(len(composite_health_indicator))
    poly_model = fit_polynomial(x_data, composite_health_indicator, degree=3)
    fitted_indicator = poly_model(x_data)
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_indicator)
    
    # Normalize the health indicator to ensure it starts at 0, ends at 1, and is positive
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(composite_health_indicator, label='Composite Health Indicator')
    plt.plot(fitted_indicator, label='Fitted Polynomial Indicator', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'Polynomial Trend Labeling for Bearing {i}')
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
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Load the data
file_path = 'Train_data_H.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)
data = data.drop(data.index[[0, 2803,3684, 3672]])

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Calculate features for each row
def calculate_features(data):
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    kurtosis = np.mean(((data - np.mean(data, axis=1, keepdims=True))**4), axis=1) / (np.var(data, axis=1)**2)
    skewness = np.mean(((data - np.mean(data, axis=1, keepdims=True))**3), axis=1) / (np.std(data, axis=1)**3)
    crest_factor = np.max(np.abs(data), axis=1) / rms
    peak_to_peak = np.ptp(data, axis=1)
    return rms, kurtosis, skewness, crest_factor, peak_to_peak

# Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized_features, scaler

# Fit spline model
def fit_spline(x, y):
    spline = UnivariateSpline(x, y, s=0.5)  # s is a smoothing factor
    return spline

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
    
    # Calculate features
    rms, kurtosis, skewness, crest_factor, peak_to_peak = calculate_features(bearing_data)
    
    # Normalize the features
    normalized_rms, _ = normalize_features(rms)
    normalized_kurtosis, _ = normalize_features(kurtosis)
    normalized_skewness, _ = normalize_features(skewness)
    normalized_crest_factor, _ = normalize_features(crest_factor)
    normalized_peak_to_peak, _ = normalize_features(peak_to_peak)
    
    # Combine features into a composite health indicator
    composite_health_indicator = (normalized_rms + normalized_kurtosis + normalized_skewness + normalized_crest_factor + normalized_peak_to_peak) / 5
    
    # Fit spline model to the composite health indicator
    x_data = np.arange(len(composite_health_indicator))
    spline_model = fit_spline(x_data, composite_health_indicator)
    fitted_indicator = spline_model(x_data)
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_indicator)
    
    # Normalize the health indicator to ensure it starts at 0, ends at 1, and is positive
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(composite_health_indicator, label='Composite Health Indicator')
    plt.plot(fitted_indicator, label='Fitted Spline Indicator', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'Spline Trend Labeling for Bearing {i}')
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
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Load the data
file_path = 'Train_data_H.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)
data = data.drop(data.index[[0, 2803,3684, 3672]])

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Calculate features for each row
def calculate_features(data):
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    kurtosis = np.mean(((data - np.mean(data, axis=1, keepdims=True))**4), axis=1) / (np.var(data, axis=1)**2)
    skewness = np.mean(((data - np.mean(data, axis=1, keepdims=True))**3), axis=1) / (np.std(data, axis=1)**3)
    crest_factor = np.max(np.abs(data), axis=1) / rms
    peak_to_peak = np.ptp(data, axis=1)
    return rms, kurtosis, skewness, crest_factor, peak_to_peak

# Normalize features
def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized_features, scaler

# Fit spline model
def fit_spline(x, y):
    spline = UnivariateSpline(x, y, s=0.5)  # s is a smoothing factor
    return spline

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
    
    # Calculate features
    rms, kurtosis, skewness, crest_factor, peak_to_peak = calculate_features(bearing_data)
    
    # Normalize the features
    normalized_rms, _ = normalize_features(rms)
    normalized_kurtosis, _ = normalize_features(kurtosis)
    normalized_skewness, _ = normalize_features(skewness)
    normalized_crest_factor, _ = normalize_features(crest_factor)
    normalized_peak_to_peak, _ = normalize_features(peak_to_peak)
    
    # Combine features into a composite health indicator
    # composite_health_indicator = (normalized_rms + normalized_kurtosis + normalized_skewness + normalized_crest_factor + normalized_peak_to_peak) / 5
    composite_health_indicator = normalized_rms
    # Fit spline model to the composite health indicator
    x_data = np.arange(len(composite_health_indicator))
    spline_model = fit_spline(x_data, composite_health_indicator)
    fitted_indicator = spline_model(x_data)
    
    # Ensure the health indicator starts at 0 and ends at 1
    fitted_indicator[0] = 0
    fitted_indicator[-1] = 1
    
    # Smooth the health indicator
    smoothed_health_indicator = smooth_health_indicator(fitted_indicator)
    
    # Normalize the health indicator to ensure it starts at 0, ends at 1, and is positive
    scaled_health_indicator = scale_health_indicator(smoothed_health_indicator)
    
    all_health_indicators.extend(scaled_health_indicator)
    
    # Plot the data for the current bearing
    plt.figure(figsize=(12, 6))
    plt.plot(composite_health_indicator, label='Composite Health Indicator')
    plt.plot(fitted_indicator, label='Fitted Spline Indicator', linestyle='--')
    plt.plot(scaled_health_indicator, label='Scaled Health Indicator', linestyle='-.')
    plt.legend()
    plt.title(f'Spline Trend Labeling for Bearing {i}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value / Health Indicator')
    plt.show()
    
    # Update the starting index for the next bearing
    start_idx = end_idx

# Convert health indicators to DataFrame
health_indicators_df = pd.DataFrame(all_health_indicators, columns=['Health Indicator'])

# Save health indicators to a CSV file
health_indicators_df.to_csv('bearing_health_indicators.csv', index=False)


