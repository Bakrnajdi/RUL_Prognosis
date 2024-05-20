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

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import random

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# import faulthandler
# faulthandler.enable() 

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

tf.device('GPU: 0')

tf.config.set_visible_devices([], 'GPU')
    
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options
    
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


data = pd.read_csv('Train_data_H.csv',index_col=[0])
data = data.reset_index(drop=True)
data = data.drop(data.index[[3686]])
# cnn_data = pd.DataFrame(cnn_data)

print('cnn data done')


# data = np.delete(data,3686)



wavelet = 'morlet'
# choose padding scheme for CWT (doesn't affect scales selection)
padtype = 'reflect'

# one of: 'log', 'log-piecewise', 'linear'
# 'log-piecewise' lowers low-frequency redundancy; see
# https://github.com/OverLordGoldDragon/ssqueezepy/issues/29#issuecomment-778526900
scaletype = 'log-piecewise'
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
min_scale, max_scale = cwt_scalebounds(wavelet, N=len(x), preset=preset)
scales = make_scales(N1, min_scale, max_scale, nv=nv, scaletype=scaletype,
                      wavelet=wavelet, downsample=downsample)



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

# for i in range(len(data1)):
#     m = data1[i]
#     m = np.resize(m,(128,128,1))
    
data1=np.array(resData)
data1.shape



del resData
# =============================================================================
# Split Data into train and test set
# =============================================================================
all_files = pd.read_csv('all_files.csv',index_col=[0])
all_files = all_files.reset_index(drop=True)
allfiles=np.array(all_files.T)


y0 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][0]))
y1 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][1]))
y2 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][2]-1))
y3 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][3]))
y4 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][4]))
y5 = pd.DataFrame(np.linspace(0.0,1.0,allfiles[0][5]))

label  = [y0,y1,y2,y3,y4,y5]
labels =np.array(pd.concat(label,ignore_index=True))


X_train, X_test, y_train, y_test = train_test_split(data1, np.array(labels), test_size=0.2, random_state=0)

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

# =============================================================================
# Build CNN model
# =============================================================================

# Initialize the model
Regressor = Sequential()

# Step 1 - Convolution
Regressor.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
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

optimizer = keras.optimizers.Adam(lr=6.8e-04)

# Step 3 - Compiling the model
#Regressor.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
# Regressor = Sequential()

Regressor.compile(loss=rmse, optimizer=optimizer, metrics=['mse'])


epochs = 20
batch_size = 256
# history = Regressor.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)
history = Regressor.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),shuffle = True)







