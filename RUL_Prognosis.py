#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:16:26 2024

@author: mac
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# import utils
import pywt
import os,glob
import random
import numpy.linalg as linalg
import seaborn as sns
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential,Model
# from keras.layers.merge import concatenate
from keras.layers import Concatenate
from keras.layers import Conv2D,LSTM, Reshape
from keras.layers import MaxPooling2D
from keras.layers import Flatten,BatchNormalization
from keras.layers import Dense,Dropout,Activation
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import backend
from sklearn import metrics
# import cv2
from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.visuals import imshow, plot
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up
from ssqueezepy.utils import logscale_transition_idx
# import time_series_augmentation.utils.augmentation as aug
# from time_series_augmentation.utils.augmentation import scaling
# from Bearing_Data_Train import Bearing_Data_Train
from cwt_rp import ssq_cwt1 
# import cwt_rp
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('float32')

from config import set_seed
set_seed()

# import faulthandler
# faulthandler.enable() 

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

# tf.device('GPU: 0')

tf.config.set_visible_devices([], 'GPU')
    
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options
    
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# data = pd.read_csv('Train_data_H.csv')
# data = data.reset_index(drop=True)
# data = data.drop(data.index[[0, 2803, 3684, 3672]])
data = pd.read_csv('..\\Train_data_H.csv',index_col=[0])
data = data.reset_index(drop=True)
data = data.drop(data.index[[0, 2803, 3672]])
# cnn_data = pd.DataFrame(cnn_data)
# data = data.drop(columns = ['Unnamed: 0'])
print('cnn data done')

data = pd.read_csv('Train_data_H.csv',index_col=[0])
data = data.reset_index(drop=True)
data = data.iloc[:,:128]
# data = data.drop(data.index[[0, 2803, 3684, 3672]])
# data = np.delete(data,3686)


# data = pd.read_csv('Train_data_H.csv',index_col=[0])
# data = data.reset_index(drop=True)

n = 2560
signal = []
for i in range(len(data)):
    idx = random.randint(0,n)
    if idx>=(n-128):
        sample = data.iloc[i][idx-128:idx]
    else: 
        sample = data.iloc[i][idx:idx+128]
    signal.append(np.array(sample))
    
signal = np.concatenate([signal],axis=0)
# signal = signal.reshape(signal.shape[0]*signal.shape[1],1)

data = pd.DataFrame(signal)


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


new_data = []

for i in range(data.shape[0]):
    new_data.append(ssq_cwt1(data.iloc[i],wavelet,scales,padtype,n_components))
    print(i)


resData = []
for i in range(len(new_data)):
    m = new_data[i]
    # m = crop_square(abs(Tx), 128 )
    m = np.resize(m,(128,128,1))
    resData.append(m)
    
    
# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.imshow(np.abs(resData[0]))
# plt.title('Synchrosqueezing Wavelet Transform')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar(label='Magnitude')
# plt.show()


# resData = []
# for i in range(len(new_data)):
#     m = new_data[i]
#     # m = crop_square(abs(Tx), 128 )
#     m = np.resize(m,(246,246,1))
#     resData.append(m)


# for i in range(len(data1)):
#     m = data1[i]
#     m = np.resize(m,(128,128,1))
    
data1=np.array(resData)
data1.shape



del resData
del new_data
# =============================================================================
# Split Data into train and test set
# =============================================================================
all_files = pd.read_csv('all_files.csv',index_col=[0])
all_files = all_files.reset_index(drop=True)
allfiles=np.array(all_files.T)


y0 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][0]))
y1 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][1]))
y2 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][2]))
y3 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][3]))
y4 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][4]))
y5 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][5]))

label  = [y0,y1,y2,y3,y4,y5]
labels =np.array(pd.concat(label,ignore_index=True))


# labels = np.array(all_health_indicators)
# labels = np.array(pd.read_csv('Segmented_bearing_health_indicators.csv'))
# labels = np.array(pd.read_csv('Segmented_bearing_health_indicators.csv'))
labels = np.array(pd.read_csv('Linear_bearing_health_indicators.csv'))



X_train, X_test, y_train, y_test = train_test_split(data1, np.array(labels), test_size=0.2, random_state=0)
import keras.backend as K

def rmse(y_true, y_pred):
	return K.sqrt(k.mean(k.square(y_pred - y_true)))

# =============================================================================
# Build CNN model
# =============================================================================
from keras import ops

def rmse_loss_fn(y_true, y_pred):
    """
    Custom RMSE loss function.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: RMSE value.
    """
    squared_difference = ops.square(y_true - y_pred)
    mean_squared_difference = ops.mean(squared_difference, axis=-1)  # Note the `axis=-1`
    return ops.sqrt(mean_squared_difference)


# Initialize the model
Regressor = Sequential()

# Step 1 - Convolution
Regressor.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
# Regressor.add(Conv2D(32, (3, 3), input_shape=(246, 246, 1), activation='relu'))
Regressor.add(MaxPooling2D(pool_size=(2, 2)))
Regressor.add(Conv2D(64, (3, 3), activation='relu'))
Regressor.add(MaxPooling2D(pool_size=(2, 2)))
Regressor.add(Conv2D(128, (3, 3), activation='relu'))
Regressor.add(MaxPooling2D(pool_size=(2, 2)))
Regressor.add(Conv2D(256, (3, 3), activation='relu'))
Regressor.add(MaxPooling2D(pool_size=(2, 2)))

# Prepare for LSTM
# The output shape after the last MaxPooling layer is (batch_size, 8, 8, 256)
Regressor.add(Reshape((6, 6 * 256)))  # Flatten the spatial dimensions for LSTM

# # Add LSTM layer
Regressor.add(LSTM(256))  # You can adjust the number of LSTM units based on your needs

# Step 2 - Neural Network configuration
Regressor.add(Flatten())
Regressor.add(Dense(units=2560, activation='relu'))
Regressor.add(Dropout(0.3))
Regressor.add(Dense(units=768, activation='relu'))
Regressor.add(Dropout(0.1))
Regressor.add(Dense(1, activation='sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate=6.8e-05)

# Step 3 - Compiling the model
#Regressor.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
# Regressor = Sequential()

Regressor.compile(loss=rmse_loss_fn, optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])


epochs = 60
batch_size = 256
history = Regressor.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)
# history = Regressor.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),shuffle = True)

import h5py
from keras.models import load_model

Regressor.save('..\\model1.h5')


def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

import tensorflow as tf

# Load the model in the SavedModel format
Regressor = tf.keras.models.load_model("model1.h5", custom_objects={'rmse': rmse})

Regressor = load_model('model1.h5',custom_objects={'rmse':rmse})



import tf2onnx

# Convert the Keras model to ONNX
model_proto, _ = tf2onnx.convert.from_keras(Regressor, output_path="model.onnx")

y_pred0 = Regressor.predict(data1[:2802])
y_pred1 = Regressor.predict(data1[2802:3674])
y_pred2 = Regressor.predict(data1[3674:4584])
y_pred3 = Regressor.predict(data1[4584:5381])
y_pred4 = Regressor.predict(data1[5381:5896])
y_pred5 = Regressor.predict(data1[5896:7533])

y_predt = Regressor.predict(data1)


sns.scatterplot(y=pred.ravel(),x=np.arange(len(pred)))

result = np.concatenate([100*abs(y_pred2),100*y2],axis=1)

sns.set(color_codes=True)
fig, ax = plt.subplots(figsize=(15, 6))
ax = sns.regplot(x=np.arange(len(pred)),y=pred,data=np.array(tdata))
ax.set(xlabel='time',ylabel='CWTCNN-HI')
pd.DataFrame(y_predt).to_csv(r'/Users/bakrnajdi/Desktop/stage/Test_wavelets/Predictive_maintenance_proj/Rul_Project/ttpred3_3.csv')
pd.DataFrame(y_pred5).to_csv(r'/Users/bakrnajdi/Desktop/stage/Test_wavelets/Predictive_maintenance_proj/Rul_Project/y_pred5.csv')
m=y_pred1

#================Training Data Visualization======================================== 
#Visualize Results
fig, ax = plt.subplots(figsize=(15, 6))
 # Plot training data.
sns.scatterplot(x=10 *np.arange(len(y_pred0)).ravel(), y=y_pred0.ravel(), label='training data', ax=ax,color='k',s=7);
sns.scatterplot(x=10 *np.arange(len(y_pred1)).ravel(), y=y_pred1.ravel(),  ax=ax,color='r',s=7);
sns.scatterplot(x=10 *np.arange(len(y_pred2)).ravel(), y=y_pred2.ravel(),  ax=ax,color='b',s=7);
sns.scatterplot(x=10 *np.arange(len(y_pred3)).ravel(), y=y_pred3.ravel(),  ax=ax,color='g',s=7);
sns.scatterplot(x=10 *np.arange(len(y_pred4)).ravel(), y=y_pred4.ravel(), ax=ax,color='c',s=7);
sns.scatterplot(x=10 *np.arange(len(y_pred5)).ravel(), y=y_pred5.ravel(), ax=ax,color='y',s=7);
 
 # Plot prediction. sns.scatterplot(x=10 *np.arange(len(y_pred1)).ravel(), y=m.ravel(), label='training data', ax=ax,color='k',s=5);
 
sns.lineplot(x=10 *np.arange(len(y_pred0)).ravel(), y=np.linspace(0,1,len(y_pred0)), color='k', label='Actual HI')
sns.lineplot(x=10 *np.arange(len(y_pred1)).ravel(), y=np.linspace(0,1,len(y_pred1)), color='k')
sns.lineplot(x=10 *np.arange(len(y_pred2)).ravel(), y=np.linspace(0,1,len(y_pred2)), color='k')
sns.lineplot(x=10 *np.arange(len(y_pred3)).ravel(), y=np.linspace(0,1,len(y_pred3)), color='k')
sns.lineplot(x=10 *np.arange(len(y_pred4)).ravel(), y=np.linspace(0,1,len(y_pred4)), color='k')
sns.lineplot(x=10 *np.arange(len(y_pred5)).ravel(), y=np.linspace(0,1,len(y_pred5)), color='k')
 
ax.set(title='Training Data')
ax.legend(loc='upper left');
ax.set(xlabel='Time (s)', ylabel='Health Indicator')




Regressor.save("model1.h5")

Regressor = load_model('model1.h5')
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

# Make predictions
y_pred = Regressor.predict(data1)

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_pred, label='Actual')
plt.plot(labels, label='Predicted')
plt.title('Actual vs Predicted Values')
plt.ylabel('Health Indicator')
plt.xlabel('Sample Index')
plt.legend()
plt.show()


# Assuming y_test and y_pred are your actual and predicted values

# Make predictions
y_pred = Regressor.predict(X_test)

# Calculate MSE
mse = np.mean((y_test - y_pred)**2)
print(f'MSE: {mse}')

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Values')
plt.ylabel('Health Indicator')
plt.xlabel('Sample Index')
plt.legend()
plt.show()





# Make predictions on the training data
y_train_pred = Regressor.predict(data1)

# Define the lengths of the bearings
bearing_lengths = [2802, 870, 911, 797, 515, 1637]

# Plot actual vs predicted values for each bearing
start_idx = 0
for i, length in enumerate(bearing_lengths):
    end_idx = start_idx + length
    actual = labels[start_idx:end_idx]
    predicted = y_train_pred[start_idx:end_idx]
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Health Indicator')
    plt.plot(predicted, label='Predicted Health Indicator', linestyle='--')
    plt.title(f'Bearing {i} - Actual vs Predicted Health Indicator')
    plt.ylabel('Health Indicator')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.show()
    
    start_idx = end_idx




import os
import pandas as pd

# Define the root directory containing the bearing folders
root_dir = "/Users/mac/Desktop/PhD/Prognosis_RUL/phm-ieee-2012-data-challenge-dataset-master/Test_set"  # Change this to your directory path

# List of test bearing folders
test_bearing_folders = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3"
]

# Initialize a list to hold the data
data_list = []

# Dictionary to hold the number of files processed in each folder
files_count = {}

# Process each test bearing folder
for folder in test_bearing_folders:
    folder_path = os.path.join(root_dir, folder)
    count = 0  # Initialize the count of files for this folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('acc') and file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file
            df = pd.read_csv(file_path, header=None)
            # Extract the 5th column (index 4) with 2560 observations
            x_vibrational_signal = df.iloc[:, 4].values
            # Append to the data list
            data_list.append(x_vibrational_signal)
            count += 1  # Increment the count
    # Save the count for this folder
    files_count[folder] = count

# Convert the list to a DataFrame
final_df = pd.DataFrame(data_list)

# Save the concatenated DataFrame to a new CSV file
final_df.to_csv("Test_Data.csv", index=False)

# Save the file counts to a new CSV file
files_count_df = pd.DataFrame(list(files_count.items()), columns=['Folder', 'File_Count'])
files_count_df.to_csv("files_count.csv", index=False)

# Display the files count
print(files_count_df)








