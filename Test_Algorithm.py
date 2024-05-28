#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:41:03 2024

@author: mac
"""

from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.visuals import imshow, plot
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up
from ssqueezepy.utils import logscale_transition_idx



# from tensorflow.keras.mixed_precision import set_global_policy
# set_global_policy('float32')

import tensorflow as tf
from tensorflow.keras import layers, models

from config import set_seed
set_seed()

# Load the concatenated data
data = pd.read_csv('Test_Data.csv')

# Load the file counts
files_count_df = pd.read_csv('files_count.csv')

# Function to get the index range for a specific bearing
def get_bearing_data_range(bearing_folder, files_count_df):
    start_idx = 0
    for index, row in files_count_df.iterrows():
        if row['Folder'] == bearing_folder:
            file_count = row['File_Count']
            end_idx = start_idx + file_count
            return start_idx, end_idx
        start_idx += row['File_Count']
    raise ValueError(f"Bearing folder {bearing_folder} not found in files_count_df")

# Select the specific bearing folder
specific_bearing_folder = "Bearing3_3"  # Change this to the specific bearing you want to process
start_idx, end_idx = get_bearing_data_range(specific_bearing_folder, files_count_df)

# Extract the specific bearing data
bearing_data = data.iloc[start_idx:end_idx].reset_index(drop=True)

# Wavelet transform parameters

wavelet = 'morlet'
# choose padding scheme for CWT (doesn't affect scales selection)
padtype = 'reflect'

# one of: 'log', 'log-piecewise', 'linear'
# 'log-piecewise' lowers low-frequency redundancy; see
# https://github.com/OverLordGoldDragon/ssqueezepy/issues/29#issuecomment-778526900
scaletype = 'linear'
# one of: 'minimal', 'maximal', 'naive' (not recommended)
preset = 'maximal'
# number of voices (wavelets per octave); more = more scales
nv = 32
# downsampling factor for higher scales (used only if `scaletype='log-piecewise'`)
downsample = 4
# show this many of lowest-frequency wavelets
show_last = 20
N1 = len(data.iloc[0])

# `cwt` uses `p2up`'d N internally
M = p2up(N1)[0]
wavelet = Wavelet(wavelet, N=M)

x = data.iloc[50]
x = np.array(x)

min_scale, max_scale = cwt_scalebounds(wavelet, N=len(x), preset=None)

scales = make_scales(N1, min_scale, max_scale, nv=nv, scaletype=scaletype,
                      wavelet=wavelet, downsample=downsample)


# q = ssq_cwt1(data.iloc[0],wavelet,scales,padtype,n_components)

def evenly_distribute_elements(array, n):
    """
    Select n elements from array ensuring the first element is the first from the list,
    the last element is the last from the list, and the elements are evenly distributed.
    """
    length = len(array)
    if n > length:
        raise ValueError("n cannot be greater than the length of the array")

    # Calculate the interval
    step = (length - 1) / (n - 1)

    # Select the indices
    indices = [int(round(i * step)) for i in range(n)]
    
    # Ensure the first and last elements are correctly assigned
    indices[0] = 0
    indices[-1] = length - 1

    # Select elements from the array
    selected_elements = array[indices]

    return selected_elements

# # Example usage
n = 128
scales = evenly_distribute_elements(scales, n)
# # print("Selected elements:", scales)

# def select_linear_numbers(min_value, max_value, n):
#     """
#     Select n linearly spaced numbers between min_value and max_value.
#     """
#     return np.linspace(min_value, max_value, n)

# # Example usage

# n = 128
# linear_numbers = select_linear_numbers(min_scale, max_scale, n)
# print("Linearly spaced numbers:", linear_numbers)

# # n_components = len(scales)



n_components = len(scales)


tnew_data = []

# Apply the algorithm to each row of the specific bearing data
for i in range(bearing_data.shape[0]):
    x = bearing_data.iloc[i].values
    x = np.array(x)
    cwt_result = ssq_cwt1(x, wavelet, scales, padtype, n_components)
    tnew_data.append(cwt_result)
    print(f"Processed row {i+1}/{bearing_data.shape[0]} for {specific_bearing_folder}")

# Resize and format the processed data
tresData = []
for i in range(len(tnew_data)):
    m = tnew_data[i]
    m = np.resize(m, (128, 128, 1))
    tresData.append(m)

data2 = np.array(tresData)
print(data2.shape)

# Clean up
del tnew_data
del tresData

# np.save("{}.npy".format(specific_bearing_folder), data2)


y_train_pred = Regressor.predict(data2)

plt.figure(figsize=(12, 6))
# plt.plot(actual, label='Actual Health Indicator')
plt.plot(y_train_pred, label='Predicted Health Indicator', linestyle='--')
plt.title(f'{specific_bearing_folder} - Actual vs Predicted Health Indicator')
plt.ylabel('Health Indicator')
plt.xlabel('Sample Index')
plt.legend()
plt.show()




y_train_pred = Regressor.predict(data2)

plt.figure(figsize=(12, 6))
# plt.scatter(range(len(actual)), actual, label='Actual Health Indicator', alpha=0.6)
plt.scatter(range(len(y_train_pred)), y_train_pred, label='Predicted Health Indicator', alpha=0.6)
plt.title(f'{specific_bearing_folder} - Actual vs Predicted Health Indicator')
plt.ylabel('Health Indicator')
plt.xlabel('Sample Index')
plt.legend()
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Generate synthetic RMS data
np.random.seed(0)
rms_data = np.concatenate([np.random.normal(1, 0.1, 100), 
                           np.random.normal(1.5, 0.1, 100), 
                           np.random.normal(1, 0.1, 100)])

# Apply CUSUM
def calculate_cusum(data, target=0, drift=0.01):
    cusum_pos = np.zeros_like(data)
    cusum_neg = np.zeros_like(data)
    for i in range(1, len(data)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - target - drift)
        cusum_neg[i] = min(0, cusum_neg[i-1] + data[i] - target + drift)
    return cusum_pos, cusum_neg

target_value = np.mean(rms_data[:100])
cusum_pos, cusum_neg = calculate_cusum(rms_data, target=target_value)

# Plotting the RMS data and CUSUM results
plt.figure(figsize=(12, 6))
plt.plot(rms_data, label='RMS Data')
plt.plot(cusum_pos, label='CUSUM Positive', linestyle='--')
plt.plot(cusum_neg, label='CUSUM Negative', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.title('RMS Data with CUSUM Detection')
plt.xlabel('Time')
plt.ylabel('RMS Value / CUSUM')
plt.legend()
plt.show()







