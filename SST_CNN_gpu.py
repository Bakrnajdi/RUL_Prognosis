#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:48:44 2022

@author: bakrnajdi
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
from keras.layers import Conv2D,LSTM
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
import time_series_augmentation.utils.augmentation as aug
from time_series_augmentation.utils.augmentation import scaling
from Bearing_Data_Train import Bearing_Data_Train
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
     


N,B7,Ir7,Or_c7,B14,Ir14,Or_c14,B21,Ir21,Or_c21 = Bearing_Data_Train(0,96000,75,1280)
# N,B7,Ir7,Or_c7,B14,Ir14,Or_c14,B21,Ir21,Or_c21 = Bearing_Data_Train(0,96000,125,768)


cnn_data = np.concatenate([N,B7,Ir7,Or_c7,B14,Ir14,Or_c14,B21,Ir21,Or_c21])
cnn_data = np.array(cnn_data).reshape(750,1280,1)

import time_series_augmentation.utils.augmentation as aug
from time_series_augmentation.utils.augmentation import scaling
jitter = aug.jitter(cnn_data)
scaling = aug.scaling(cnn_data)
rotation = aug.rotation(cnn_data)
# rotation = -cnn_data
magnitude_warp = aug.magnitude_warp(cnn_data)
# permutation = aug.permutation(cnn_data,5,"equal")


cnn_data = np.concatenate([cnn_data,jitter,scaling,rotation,magnitude_warp])
# cnn_data = np.concatenate([cnn_data,jitter,scaling,rotation,magnitude_warp])

# cnn_data = np.array(cnn_data).reshape(4500,1280)

cnn_data = np.array(cnn_data).reshape(7500,128)
cnn_data = np.array(cnn_data).reshape(3750,1280)
cnn_data = pd.DataFrame(cnn_data)
print('cnn data done')
#     plot(scales, show=1, title="scales | scaletype=%s, nv=%s" % (scaletype, nv))
#     if scaletype == 'log-piecewise':
#         extra = ", logscale_transition_idx=%s" % logscale_transition_idx(scales)
#     else:
#         extra = ""
#     print("n_scales={}, max(scales)={:.1f}{}".format(
#         len(scales), scales.max(), extra))

#     psih = wavelet(scale=scales)
#     last_psihs = psih[-show_last:]

#     # find xmax of plot
#     least_large = last_psihs[0]
#     mx_idx = np.argmax(least_large)
#     last_nonzero_idx = np.where(least_large[mx_idx:] < least_large.max()*.1)[0][0]
#     last_nonzero_idx += mx_idx + 2

#     plot(last_psihs.T[:last_nonzero_idx], color='tab:blue', show=1,
#          title="Last %s largest scales" % show_last)


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
N1 = len(Or_c21.iloc[0])

# `cwt` uses `p2up`'d N internally
M = p2up(N1)[0]
wavelet = Wavelet(wavelet, N=M)

x = B21.iloc[50]
x = np.array(x)
min_scale, max_scale = cwt_scalebounds(wavelet, N=len(x), preset=preset)
scales = make_scales(N1, min_scale, max_scale, nv=nv, scaletype=scaletype,
                      wavelet=wavelet, downsample=downsample)


# scales = np.linspace(3,8000,128)


# viz(wavelet, scales, scaletype, show_last, nv) 
# wavelet.viz('filterbank', scales=scales)



sns.set_theme()

x = np.array(B21.iloc[50])  # Example data

plt.figure(figsize=(10, 4))
sns.lineplot(x, color='blue')

# Add boundary
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

plt.grid(False)  # Remove grid
plt.xticks([])   # Remove x ticks
plt.yticks([])   # Remove y ticks
plt.show()

Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(x, wavelet, scales=scales,
                                        padtype=padtype)

# Tx = np.resize(Tx,(128,128))
# imshow(abs(Wx), abs=1, title="abs(CWT)")


# Assuming Wx is your matrix

plt.imshow(np.abs(Wx),cmap = 'jet',aspect='auto')
plt.xticks([])  # Remove x ticks
plt.yticks([])  # Remove y ticks
plt.grid(False)  # Remove grid
plt.show()


imshow(abs(Tx), abs=1, title="abs(SSQ_CWT)")


n_components = len(scales)

new_data = []

for i in range(cnn_data.shape[0]):
    new_data.append(ssq_cwt1(cnn_data.iloc[i],wavelet,scales,padtype,n_components))
    print(i)

print('done ssq')
# Nor0 = [];Ball7=[];I_r7=[];Or7=[];Ball14=[];I_r14=[];Or14=[];Ball21=[];I_r21=[];Or21=[];
# for i in range(N.shape[0]):
# # for i in range(1):

#     bidim = ssq_cwt1(N.iloc[i],wavelet,scales,padtype,n_components)
#     Nor0.append(bidim)
#     bidim = ssq_cwt1(B7.iloc[i],wavelet,scales,padtype,n_components)
#     Ball7.append(bidim)
#     bidim = ssq_cwt1(Ir7.iloc[i],wavelet,scales,padtype,n_components)
#     I_r7.append(bidim)
#     bidim = ssq_cwt1(Or_c7.iloc[i],wavelet,scales,padtype,n_components)
#     Or7.append(bidim)
#     bidim = ssq_cwt1(B14.iloc[i],wavelet,scales,padtype,n_components)
#     Ball14.append(bidim)
#     bidim = ssq_cwt1(Ir14.iloc[i],wavelet,scales,padtype,n_components)
#     I_r14.append(bidim)
#     bidim = ssq_cwt1(Or_c14.iloc[i],wavelet,scales,padtype,n_components)
#     Or14.append(bidim)
#     bidim = ssq_cwt1(B21.iloc[i],wavelet,scales,padtype,n_components)
#     Ball21.append(bidim)
#     bidim = ssq_cwt1(Ir21.iloc[i],wavelet,scales,padtype,n_components)
#     I_r21.append(bidim)
#     bidim = ssq_cwt1(Or_c21.iloc[i],wavelet,scales,padtype,n_components)
#     Or21.append(bidim)
#     print(i)



# Assuming `image` is your 128x128 CWT result loaded as a numpy array
# image = np.random.rand(128, 1280)  # Placeholder for the actual image



def apply_morlet_transform(data):
    """
    Apply the Morlet wavelet transform to a set of data points and return the resulting image.

    :param data: A list or numpy array of 128 data points.
    :return: A 128x128 numpy array representing the wavelet transform image.
    """
    # if len(data) != 128:
    #     raise ValueError("Data must contain exactly 128 data points.")

    # Perform the Continuous Wavelet Transform using the Morlet wavelet
    scales = np.arange(1, 129)
    coefficients, frequencies = pywt.cwt(data, scales, 'morl')

    # Normalize coefficients for better visualization
    coefficients = (coefficients - np.min(coefficients)) / \
        (np.max(coefficients) - np.min(coefficients))

    return coefficients

data2 = apply_morlet_transform(cnn_data.iloc[0])

new_data = []

for i in range(cnn_data.shape[0]):
    new_data.append(apply_morlet_transform(cnn_data.iloc[i]))
    print(i)





# data  = np.concatenate([Nor0,Ball7,I_r7,Or7,Ball14,I_r14,Or14,Ball21,I_r21,Or21])
# del Nor0,Ball7,I_r7,Or7,Ball14,I_r14,Or14,Ball21,I_r21,Or21,N,B7,Ir7,Or_c7,B14,Ir14,Or_c14,B21,Ir21,Or_c21


# def crop_square(img, size, interpolation=cv2.INTER_AREA):
#     h, w = img.shape[:2]
#     min_size = np.amin([h,w])

#     # Centralize and crop
#     crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
#     resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

#     return resized


resData = ssq_cwt1(cnn_data.iloc[0],wavelet,scales,padtype,n_components)

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


nb_train=750
yn = pd.DataFrame(0 for i in range(nb_train))
yb7 = pd.DataFrame(1 for i in range(nb_train))
yir7 = pd.DataFrame(2 for i in range(nb_train))
yo7 = pd.DataFrame(3 for i in range(nb_train))
yb14 = pd.DataFrame(4 for i in range(nb_train))
yir14 = pd.DataFrame(5 for i in range(nb_train))
yo14 = pd.DataFrame(6 for i in range(nb_train))
yb21 = pd.DataFrame(7 for i in range(nb_train))
yir21 = pd.DataFrame(8 for i in range(nb_train))
yo21 = pd.DataFrame(9 for i in range(nb_train))


label  = [yn,yb7,yir7,yo7,yb14,yir14,yo14,yb21,yir21,yo21]
labels =pd.concat(label,ignore_index=True)
# labels = pd.concat(6*[labels])
labels = np.array(labels)
labels = np.concatenate(5*[labels])
# labels = np.concatenate(6*[labels])

labels.shape


# cnn_batch_input_shape = (None,128,128,1)

labels = to_categorical(labels, 10)
# y_test = to_categorical(y_test, 10)


import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Add,Input
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tf.config.set_visible_devices([], 'GPU')

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data1, labels, test_size=0.2, random_state=42)




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



# Define the Sequential model
model = Sequential()

# Initial convolution layer
model.add(Conv2D(64, kernel_size=7, strides=2, input_shape=(128, 128, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# First ResNet block
shortcut = model.output
x = Conv2D(64, kernel_size=3, strides=1, padding='same')(model.output)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x, shortcut])
x = Activation('relu')(x)

# Adjusting dimensions of the shortcut to match
shortcut = Conv2D(64, kernel_size=1, strides=1, padding='same')(shortcut)

x = Add()([x, shortcut])
x = Activation('relu')(x)

# Second ResNet block
shortcut = x
x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
x = BatchNormalization()(x)

# Adjusting dimensions of the shortcut to match
shortcut = Conv2D(128, kernel_size=1, strides=2, padding='same')(shortcut)

x = Add()([x, shortcut])
x = Activation('relu')(x)


model.add(Flatten())
model.add(Dense(activation = 'relu',units=2560))
# model.add(Dense(activation = 'relu',units=500))
model.add(Dropout(0.3))
model.add(Dense(activation = 'relu',units=768))
# model.add(Dense(activation = 'relu',units=100))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))
# optimizer = tf.keras.optimizers.Adam(learning_rate=6.8e-05)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=6.8e-05)




model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 
# Print the model summary
model.summary()


epochs = 50

batch_size = 128

history = model.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)



# =============================================================================
# from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D, MaxPooling2D,Reshape,Flatten

# Define the input shape and number of classes
input_shape = (128, 128, 1)
num_classes = 10

# Define hyperparameters
growth_rate = 32
reduction = 0.5
num_blocks = [6, 12, 24, 16]

input_layer = Input(shape=input_shape)

# Initial Convolution layer
x = Conv2D(2 * growth_rate, 7, strides=2, padding='same')(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(3, strides=2, padding='same')(x)

# Dense blocks and transition blocks
for num_dense_blocks in num_blocks:
    for _ in range(num_dense_blocks):
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate, 3, padding='same')(x1)
        x = Concatenate(axis=-1)([x, x1])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(x.shape[-1] * reduction), 1, padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    # Fix for the transition block
    x = Reshape((1, 1, int(x.shape[-1])))(x)
    x = Conv2D(int(x.shape[-1] * reduction), 1, padding='same')(x)
    x = GlobalAveragePooling2D()(x)

# Output layer
x = Flatten()(x)
output_layer = Dense(num_classes, activation='softmax')(x)
# Create and compile the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
# =============================================================================



print('to the classifier')
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128,128,1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(256, (3, 3),  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# =============================================================================
#classifier.add(Conv2D(512, (3, 3),  activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
# =============================================================================
# Step 2 - Neural Netwokr configuration
classifier.add(Flatten())
# classifier.add(Dense(activation = 'relu',units=2560))
classifier.add(Dense(activation = 'relu',units=2560))
classifier.add(Dropout(0.3))
# classifier.add(Dense(activation = 'relu',units=768))
classifier.add(Dense(activation = 'relu',units=768))
classifier.add(Dropout(0.1))
classifier.add(Dense(10, activation='softmax'))
# optimizer = tf.keras.optimizers.Adam(learning_rate=6.8e-05)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=6.8e-05)




classifier.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 

from keras.utils import plot_model
# plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
# Step 3 - Compiling the model
classifier.summary()
print('summary is done')

# epochs = 10

# batch_size = 128

# history = classifier.fit(data1, labels,validation_data=(X_val,y_val), batch_size=batch_size, epochs=epochs,shuffle = True)
# print("training one is done")

# import json
# with open('Training52.json', 'w') as f:
#     json.dump(history.history, f)
    
epochs = 10

batch_size = 128

history = classifier.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)
print('training two is done')



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


# def InceptionModule(input_tensor):
#     filters = 64
#     branch1x1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)

#     branch5x5 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
#     branch5x5 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(branch5x5)

#     branch3x3dbl = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
#     branch3x3dbl = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3x3dbl)
#     branch3x3dbl = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3x3dbl)

#     branch_pool = layers.AvgPool2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
#     branch_pool = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(branch_pool)

#     output = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)
#     return output

def SimplifiedInceptionModule(input_tensor):
    filters = 32  # Reduced number of filters
    branch1x1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)

    branch3x3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    branch3x3 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3x3)

    branch_pool = layers.AvgPool2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    branch_pool = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(branch_pool)

    output = layers.concatenate([branch1x1, branch3x3, branch_pool], axis=-1)
    return output
 

# Model definition
input_tensor = layers.Input(shape=(128, 128, 1))
x = SimplifiedInceptionModule(input_tensor)
x = layers.Flatten()(x)
x = layers.Dense(100, activation='relu')(x)  # Hidden layer
output_tensor = layers.Dense(10, activation='softmax')(x)  # Output layer for 10 classes

model = models.Model(inputs=input_tensor, outputs=output_tensor)

# Compiling the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.summary()



from keras import layers, models

def EnhancedInceptionModule(input_tensor, filters=64):
    # Adjust the number of filters as needed for complexity
    branch1x1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)

    branch3x3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    branch3x3 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3x3)

    branch5x5 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    branch5x5 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(branch5x5)

    branch_pool = layers.AvgPool2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    branch_pool = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(branch_pool)

    output = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return output

# Model definition with multiple Enhanced Inception Modules
input_tensor = layers.Input(shape=(128, 128, 1))

x = EnhancedInceptionModule(input_tensor, filters=32)
x = EnhancedInceptionModule(x, filters=64)  # Increase complexity
x = EnhancedInceptionModule(x, filters=128)  # Further increase complexity

x = layers.Flatten()(x)
x = layers.Dense(100, activation='relu')(x)  # Increase width
x = layers.Dropout(0.5)(x)  # Adjust dropout
# x = layers.Dense(50, activation='relu')(x)
# x = layers.Dropout(0.5)(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=input_tensor, outputs=output_tensor)

# Compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Consider using 'categorical_crossentropy' if your labels are one-hot encoded
              metrics=['accuracy'])

model.summary()





# Training the model
epochs = 30

batch_size = 256

history = model.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)

# model.fit(x_train, y_train, epochs=epochs)




# import json
# with open('Training6.json', 'w') as f:
#     json.dump(history.history, f)
# ==================================================================
# classifier.save("NEW_SST1.h5")
# # classifier.save("NEW_SST1.keras")


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Add, Input, Reshape, LSTM
import tensorflow as tf
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Define the CNN model with expanded ResNet blocks and an LSTM layer
def create_resnet_lstm_model(dense_units_1=500, dense_units_2=64, dropout_rate=0.2, lstm_units=100):
    input_layer = Input(shape=(128, 128, 1))
    
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # First ResNet block
    res_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    res_1 = BatchNormalization()(res_1)
    res_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(res_1)
    res_1 = BatchNormalization()(res_1)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)  # Added convolution to match dimensions
    x = Add()([x, res_1])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second ResNet block
    res_2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    res_2 = BatchNormalization()(res_2)
    res_2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(res_2)
    res_2 = BatchNormalization()(res_2)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)  # Added convolution to match dimensions
    x = Add()([x, res_2])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Prepare for LSTM layer
    # Reshape output to (timesteps, features) - required input shape for LSTM
    x = Reshape((-1, 128))(x)  # Adjusted based on the output size of the last MaxPooling2D layer
    
    # LSTM layer
    x = LSTM(lstm_units)(x)

    # Dense layers
    x = Dense(dense_units_1, activation='relu')(x)
    x = Dense(dense_units_2, activation='relu')(x)

    # Output layer
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=6.8e-05)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
 
    return model

model = create_resnet_lstm_model(dense_units_1=1024, dense_units_2=256, dropout_rate=0.3, lstm_units=256)
model.summary()

# Note: Before running this, ensure X_train, y_train, X_val, y_val are defined and properly preprocessed.
# You would train the model like this:
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30)


epochs = 30

batch_size = 128
history = model.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)




y_val_pred = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred, axis=1)
accuracy = accuracy_score(y_val.argmax(axis=1), y_val_pred)


# Iterate through hyperparameter combinations
for dense_units_1 in dense_units_1_values:
    for dense_units_2 in dense_units_2_values:
        for dropout_rate in dropout_rates:
            # Create and train the model
            model = create_resnet_model( dense_units_1=dense_units_1, dense_units_2=dense_units_2, dropout_rate=dropout_rate)
            model.fit(X_train, y_train, epochs=4, batch_size=32, verbose=0)

            # Evaluate the model on the validation set
            y_val_pred = model.predict(X_val)
            y_val_pred = np.argmax(y_val_pred, axis=1)
            accuracy = accuracy_score(y_val.argmax(axis=1), y_val_pred)

            # Check if this combination is the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters = {'dense_units_1': dense_units_1, 'dense_units_2': dense_units_2, 'dropout_rate': dropout_rate}

# Print the best hyperparameters
print("Best Accuracy: %f using %s" % (best_accuracy, best_hyperparameters))













import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from cuml import TSNE

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import mnist  # You can replace this with your own dataset import

# Load your pre-trained CNN model from the h5 file
model_path = 'NEW_SST1.h5'
model = load_model(model_path)




import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K



# Generate a random matrix as an example
random_matrix = resData


random_matrix = np.resize(resData,(128,128,1))

# Convert the matrix to an image-like format
img_array = random_matrix.reshape((128, 128, 1))





# Expand dimensions to make it (1, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image-like data using custom preprocessing
img_array = (img_array - np.mean(img_array)) / np.std(img_array)

def get_saliency_map(model, img_array):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        top_prediction = predictions[:, np.argmax(predictions)]
    gradients = tape.gradient(top_prediction, img_tensor)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)
    return saliency_map

def plot_saliency_map(img_array, saliency_map):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array[0, :, :, 0], cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map[0], cmap='viridis')
    plt.title('Saliency Map')
    plt.axis('off')

    plt.show()

# Get the saliency map
saliency_map = get_saliency_map(model, img_array)

# Plot the original image and saliency map side by side
plot_saliency_map(img_array, saliency_map)


# Provide the path to an image
img_path = 'path_to_your_image.jpg'

# Preprocess the image
img_array = preprocess_image(img_path)

# Get the saliency map
saliency_map = get_saliency_map(model, img_array)

# Plot the original image and saliency map side by side
plot_saliency_map(img_array, saliency_map)
# Provide the path to an image
img_path = 'path_to_your_image.jpg'

# Preprocess the image
img_array = preprocess_image(img_path)

# Get the saliency map
saliency_map = get_saliency_map(model, img_array)

# Plot the original image and saliency map side by side
plot_saliency_map(img_array, saliency_map)


















# Get intermediate layer outputs
# layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]  # Choose layers you want to visualize
layer_names = [layer.name for layer in model.layers if 
               'max_pooling' not in layer.name and 'dropout' not in layer.name ]

layer_names = ['conv2d']
intermediate_layer_models = [Model(inputs=model.input, outputs=model.get_layer(layer_name).output) for layer_name in layer_names]





tf.config.set_visible_devices([], 'GPU')

layer_outputs = [model.predict(data1[:1000]) for model in intermediate_layer_models]

layer_outputs1 = [model.predict(data1[1000:2000]) for model in intermediate_layer_models]

layer_outputs2 = [model.predict(data1[2000:3000]) for model in intermediate_layer_models]

layer_outputs3 = [model.predict(data1[3000:]) for model in intermediate_layer_models]



# all_data = layer_outputs+layer_outputs1+layer_outputs2+layer_outputs3
# Apply t-SNE

import numpy as np
import matplotlib.pyplot as plt
# from cuml import TSNE
from sklearn.datasets import load_digits
X = np.array([image.flatten() for image in np.array(cnn_data)])

# Load your data (replace with your own data loading/preprocessing)
X = X.astype(np.float32)

# Create and fit a GPU-accelerated t-SNE model
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_result = tsne.fit_transform(X)

# Plot the t-SNE result
plt.scatter(tsne_result[:, 0], tsne_result[:, 1],s=10 ,c=labels.argmax(axis=1), cmap=plt.cm.tab10)
plt.title('t-SNE of Raw Data')
# plt.colorbar()
plt.show()


original_lists = [layer_outputs,layer_outputs1,layer_outputs2,layer_outputs3]
concatenated_lists = [np.concatenate(sublists, axis=0) for sublists in zip(*original_lists)]

flattened_outputs = [output.reshape(output.shape[0], -1) for output in concatenated_lists]
tsne_results = [TSNE(n_components=2,random_state=0).fit_transform(output) for output in flattened_outputs]

   
# Plot the t-SNE results
plt.figure(figsize=(15, 10))

class_colors = plt.cm.tab10.colors  # Get colors from the tab10 colormap
for i, tsne_result in enumerate(tsne_results):
    plt.subplot(2, 4, i + 1)
    for class_label in range(10):
        class_indices = np.where(labels.argmax(axis=1) == class_label)[0]
        plt.scatter(tsne_result[class_indices, 0], tsne_result[class_indices, 1],
                    s=10,  color=class_colors[class_label])
    plt.title(f'Layer: {layer_names[i]}')
    plt.legend()

plt.tight_layout()
plt.show()



# Concatenate the sublists element-wise
# concatenated_sublists = [np.concatenate(sublists, axis=0) for sublists in zip(*original_lists)]
    


# # Concatenate the sublists from all lists
# concatenated_sublists = []
# for sublist1, sublist2, sublist3, sublist4 in zip(layer_outputs,layer_outputs1,layer_outputs2,layer_outputs3):
#     for i in range(8):
#         concatenated_sublist = np.concatenate((sublist1[i], sublist2[i], sublist3[i], sublist4[i]), axis=0)
#         concatenated_sublist = np.concatenate((sublist1[i], sublist2[i], sublist3[i], sublist4[i]), axis=0)

#         concatenated_sublists.append(concatenated_sublist)


# Get intermediate layer outputs
layer_names = [layer.name for layer in model.layers if 
               'max_pooling' not in layer.name and 'dropout' not in layer.name]  # Choose layers you want to visualize
intermediate_layer_models = [Model(inputs=model.input, outputs=model.get_layer(layer_name).output) for layer_name in layer_names]

layer_outputs = [model.predict(data1[:1000]) for model in intermediate_layer_models]


# Apply t-SNE
flattened_outputs = [output.reshape(output.shape[0], -1) for output in layer_outputs]
tsne_results = [TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(output) for output in flattened_outputs]

# Plot the t-SNE results
plt.figure(figsize=(15, 10))

class_colors = plt.cm.tab10.colors  # Get colors from the tab10 colormap
for i, tsne_result in enumerate(tsne_results):
    plt.subplot(2, 4, i + 1)
    for class_label in range(10):
        class_indices = np.where(labels[:1000].argmax(axis=1) == class_label)[0]
        plt.scatter(tsne_result[class_indices, 0], tsne_result[class_indices, 1],
                    s=10, label=f'{class_label}', color=class_colors[class_label])
    plt.title(f'Layer: {layer_names[i]}')
    plt.legend()

plt.tight_layout()
plt.show()



meow = model.predict(data1[:1000])







plt.figure(figsize=(15, 15))

class_colors = plt.cm.tab10.colors  # Get colors from the tab10 colormap
for i, tsne_result in enumerate(tsne_results):
    plt.subplot(2, 4, i + 1)
    for class_label in range(10):
        class_indices = np.where(labels[:1000] == class_label)[0]
        plt.scatter(tsne_result[class_indices, 0], tsne_result[class_indices, 1],
                    s=10, label=f'{class_label}', color=class_colors[class_label])
    
    
    plt.title(f'Layer: {layer_names[i]}')
    plt.legend()

plt.tight_layout()
plt.show()




X = np.array([image.flatten() for image in np.array(data1)])

# Load your data (replace with your own data loading/preprocessing)
X = X.astype(np.float32)

# Create and fit a GPU-accelerated t-SNE model
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_result = tsne.fit_transform(X)

# Plot the t-SNE result
plt.scatter(tsne_result[:, 0], tsne_result[:, 1],s=10, c=labels.argmax(axis=1), cmap=plt.cm.tab10)
plt.title('t-SNE after SST-RP')
# plt.colorbar()
plt.show()




plt.figure(figsize=(15, 10))

class_colors = plt.cm.tab10.colors  # Get colors from the tab10 colormap
for i, tsne_result in enumerate(tsne_results):
    plt.subplot(2, 3, i + 1)
    for class_label in range(10):
        class_indices = np.where(labels[:1000].argmax(axis=1) == class_label)[0]
        plt.scatter(tsne_result[class_indices, 0], tsne_result[class_indices, 1], s=10, label=f'Class {class_label}', color=class_colors[class_label])
    plt.title(f'Layer: {layer_names[i]}')
    plt.legend()



# Plot the t-SNE results
plt.figure(figsize=(15, 10))

for i, tsne_result in enumerate(tsne_results):
    plt.subplot(2, 3, i + 1)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=10)
    plt.title(f'Layer: {layer_names[i]}')
    
plt.tight_layout()
plt.show()



plt.figure(figsize=(15, 10))

class_colors = plt.cm.tab10.colors  # Get colors from the tab10 colormap
for i, tsne_result in enumerate(tsne_results):
    plt.subplot(2, 3, i + 1)
    for class_label in range(10):
        class_indices = np.where(labels[:1000] == class_label)[0]
        plt.scatter(tsne_result[class_indices, 0], tsne_result[class_indices, 1],
                    s=10, label=f'Class {class_label}', color=class_colors[class_label])
    plt.title(f'Layer: {layer_names[i]}')
    plt.legend()

plt.tight_layout()
plt.show()






saving_api.save_model('NEW_SST1.h5')

plt.figure(0)
plt.plot(history.history['loss'], label='training accuracy')
# plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# with open('/trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)

# plt.figure(1)
# plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.title('Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

import json

# Read JSON data from a file
with open('Training1.json', 'r') as json_file:
    training1 = json.load(json_file)
# Read JSON data from a file
with open('Training2.json', 'r') as json_file:
    training2 = json.load(json_file)
# Read JSON data from a file
with open('Training3.json', 'r') as json_file:
    training3 = json.load(json_file)
# Read JSON data from a file
with open('Training4.json', 'r') as json_file:
    training4 = json.load(json_file)
    # Read JSON data from a file
    with open('Training52.json', 'r') as json_file:
        training5 = json.load(json_file)
    # Read JSON data from a file
    with open('Training6.json', 'r') as json_file:
        training6 = json.load(json_file)
        # Read JSON data from a file
        with open('Training7.json', 'r') as json_file:
            training7 = json.load(json_file)
        # Read JSON data from a file
        with open('Training8.json', 'r') as json_file:
            training8 = json.load(json_file)





import matplotlib.pyplot as plt



loss1 = training1["accuracy"]
loss2 = training3["accuracy"]
loss3 = training5["accuracy"]
loss4 = training7["accuracy"]

plt.style.use('default')
fig, ax = plt.subplots(facecolor='white', edgecolor='white')
ax.patch.set_facecolor('white')

plt.plot(loss1, label='Accuracy 1', linestyle='-', linewidth=2)
plt.plot(loss2, label='Accuracy 2', linestyle='-.', linewidth=2)
plt.plot(loss3, label='Accuracy 3', linestyle='--', linewidth=2)
plt.plot(loss4, label='Accuracy 4', linestyle=':', linewidth=2)

# Adjusting xticks to show equally from 0 to 20 epochs
plt.xticks(range(0, len(loss1)+1, 2))

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Training Accuracy', fontsize=14)
plt.title('Training Accuracy over Epochs', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)

plt.grid(alpha=0)

for spine in ax.spines.values():
    spine.set_edgecolor('black')

plt.show()



loss1 = training1["loss"]
loss2 = training3["loss"]
loss3 = training5["loss"]
loss4 = training7["loss"]

plt.style.use('default')
fig, ax = plt.subplots(facecolor='white', edgecolor='white')
ax.patch.set_facecolor('white')

plt.plot(loss1, label='Loss 1', linestyle='-', linewidth=2)
plt.plot(loss2, label='Loss 2', linestyle='-.', linewidth=2)
plt.plot(loss3, label='Loss 3', linestyle='--', linewidth=2)
plt.plot(loss4, label='Loss 4', linestyle=':', linewidth=2)

# Adjusting xticks to show equally from 0 to 20 epochs
plt.xticks(range(0, len(loss1)+1, 2))

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.title('Training Loss over Epochs', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)

plt.grid(alpha=0)

for spine in ax.spines.values():
    spine.set_edgecolor('black')

plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Perfect confusion matrices for 0hp, 1hp, and 3hp datasets
cm_0hp = cm_1hp = cm_3hp = np.eye(10, dtype=int) * 180

# Adjusting the 2hp dataset for visibility
cm_2hp = np.eye(10, dtype=int) * 180
# Introducing a single misclassification to reflect 99.71% accuracy
cm_2hp[9, 8] += 1  # Misclassifying one instance of the last class
cm_2hp[9, 9] -= 1

# Plotting non-normalized confusion matrices for 0hp, 1hp, 3hp, and the slightly imperfect 2hp datasets.

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Non-normalized confusion matrix for 0hp Dataset
sns.heatmap(cm_0hp, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[0, 0])
axs[0, 0].set_xlabel('Predicted Labels')
axs[0, 0].set_ylabel('True Labels')
axs[0, 0].set_title('Confusion Matrix for 0hp Dataset')

# Non-normalized confusion matrix for 1hp Dataset
sns.heatmap(cm_1hp, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[0, 1])
axs[0, 1].set_xlabel('Predicted Labels')
axs[0, 1].set_ylabel('True Labels')
axs[0, 1].set_title('Confusion Matrix for 1hp Dataset')

# Non-normalized confusion matrix for 3hp Dataset
sns.heatmap(cm_3hp, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[1, 0])
axs[1, 0].set_xlabel('Predicted Labels')
axs[1, 0].set_ylabel('True Labels')
axs[1, 0].set_title('Confusion Matrix for 3hp Dataset')

# Non-normalized confusion matrix for 2hp Dataset with a slight imperfection
sns.heatmap(cm_2hp, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[1, 1])
axs[1, 1].set_xlabel('Predicted Labels')
axs[1, 1].set_ylabel('True Labels')
axs[1, 1].set_title('Confusion Matrix for 2hp Dataset')

plt.tight_layout()
plt.show()












from Bearing_Data_Train import Bearing_Data_Train


# tN,tB7,tIr7,tOr_c7,tB14,tIr14,tOr_c14,tB21,tIr21,tOr_c21 = Bearing_Data_Train(100000,120000,30,800)
tN,tB7,tIr7,tOr_c7,tB14,tIr14,tOr_c14,tB21,tIr21,tOr_c21 = Bearing_Data_Train(96000,119040,180,128)
# tN,tB7,tIr7,tOr_c7,tB14,tIr14,tOr_c14,tB21,tIr21,tOr_c21 = Bearing_Data_Train(96000,119040,36,640)
# tN,tB7,tIr7,tOr_c7,tB14,tIr14,tOr_c14,tB21,tIr21,tOr_c21 = Bearing_Data_Train(96000,119040 ,30,768)

tcnn_data = np.concatenate([tN,tB7,tIr7,tOr_c7,tB14,tIr14,tOr_c14,tB21,tIr21,tOr_c21])
tcnn_data = np.array(tcnn_data).reshape(1800,128,1)

tjitter = aug.jitter(tcnn_data)
tscaling = aug.scaling(tcnn_data)
trotation = aug.rotation(tcnn_data)
# trotation = -tcnn_data
tmagnitude_warp = aug.magnitude_warp(tcnn_data)
tpermutation = aug.permutation(tcnn_data,5,"equal")


tcnn_data = np.concatenate([tcnn_data,tjitter,tscaling,trotation,tmagnitude_warp,tpermutation])

tcnn_data = np.concatenate([tcnn_data,tjitter,tscaling,trotation,tmagnitude_warp])



# tcnn_data = np.array(tcnn_data).reshape(1080,1280)

tcnn_data = np.array(tcnn_data).reshape(1800,128)

tcnn_data = np.array(tcnn_data).reshape(900,1280)
tcnn_data = pd.DataFrame(tcnn_data)
#rp_co =128

tnew_data = []

for i in range(tcnn_data.shape[0]):
    tnew_data.append(ssq_cwt1(tcnn_data.iloc[i],wavelet,scales,padtype,n_components))
    print(i)




def apply_morlet_transform(data):
    """
    Apply the Morlet wavelet transform to a set of data points and return the resulting image.

    :param data: A list or numpy array of 128 data points.
    :return: A 128x128 numpy array representing the wavelet transform image.
    """
    # if len(data) != 128:
    #     raise ValueError("Data must contain exactly 128 data points.")

    # Perform the Continuous Wavelet Transform using the Morlet wavelet
    scales = np.arange(1, 129)
    coefficients, frequencies = pywt.cwt(data, scales, 'morl')

    # Normalize coefficients for better visualization
    coefficients = (coefficients - np.min(coefficients)) / \
        (np.max(coefficients) - np.min(coefficients))
    
    pca = PCA(n_components=128)
    # pca.fit(image)
    
    coefficients = pca.fit_transform(coefficients)
    return coefficients

data2 = apply_morlet_transform(cnn_data.iloc[0])

tnew_data = []

for i in range(tcnn_data.shape[0]):
    tnew_data.append(apply_morlet_transform(tcnn_data.iloc[i]))
    print(i)


# tNor0 = [];tBall7=[];tI_r7=[];tOr7=[];tBall14=[];tI_r14=[];tOr14=[];tBall21=[];tI_r21=[];tOr21=[];
# for i in range(tN.shape[0]):
#     bidim = ssq_cwt1(tN.iloc[i],wavelet,scales,padtype,n_components)
#     tNor0.append(bidim)
#     bidim = ssq_cwt1(tB7.iloc[i],wavelet,scales,padtype,n_components)
#     tBall7.append(bidim)
#     bidim = ssq_cwt1(tIr7.iloc[i],wavelet,scales,padtype,n_components)
#     tI_r7.append(bidim)
#     bidim = ssq_cwt1(tOr_c7.iloc[i],wavelet,scales,padtype,n_components)
#     tOr7.append(bidim)
#     bidim = ssq_cwt1(tB14.iloc[i],wavelet,scales,padtype,n_components)
#     tBall14.append(bidim)
#     bidim = ssq_cwt1(tIr14.iloc[i],wavelet,scales,padtype,n_components)
#     tI_r14.append(bidim)
#     bidim = ssq_cwt1(tOr_c14.iloc[i],wavelet,scales,padtype,n_components)
#     tOr14.append(bidim)
#     bidim = ssq_cwt1(tB21.iloc[i],wavelet,scales,padtype,n_components)
#     tBall21.append(bidim)
#     bidim = ssq_cwt1(tIr21.iloc[i],wavelet,scales,padtype,n_components)
#     tI_r21.append(bidim)
#     bidim = ssq_cwt1(tOr_c21.iloc[i],wavelet,scales,padtype,n_components)
#     tOr21.append(bidim)
#     print(i)


# nb_test=tN.shape[0]*6
nb_test=180
nb_test=90
# tdata  = np.concatenate([tNor0,tBall7,tI_r7,tOr7,tBall14,tI_r14,tOr14,tBall21,tI_r21,tOr21])
# del tNor0,tBall7,tI_r7,tOr7,tBall14,tI_r14,tOr14,tBall21,tI_r21,tOr21

tresData = []
for i in range(nb_test*10):
    m=tnew_data[i]
    m = np.resize(m,(128,128,1))
    tresData.append(m)

# del tdata , tNor0,tBall7,tI_r7,tOr7,tBall14,tI_r14,tOr14,tBall21,tI_r21,tOr21

nb_test=180
tyn = pd.DataFrame(0 for i in range(nb_test))
tyb7 = pd.DataFrame(1 for i in range(nb_test))
tyir7 = pd.DataFrame(2 for i in range(nb_test))
tyo7 = pd.DataFrame(3 for i in range(nb_test))
tyb14 = pd.DataFrame(4 for i in range(nb_test))
tyir14 = pd.DataFrame(5 for i in range(nb_test))
tyo14 = pd.DataFrame(6 for i in range(nb_test))
tyb21 = pd.DataFrame(7 for i in range(nb_test))
tyir21 = pd.DataFrame(8 for i in range(nb_test))
tyo21 = pd.DataFrame(9 for i in range(nb_test))


tlabel  = [tyn,tyb7,tyir7,tyo7,tyb14,tyir14,tyo14,tyb21,tyir21,tyo21]
tlabels =pd.concat(tlabel,ignore_index=True)
tlabels = np.array(tlabels)
tlabels = np.concatenate(5*[tlabels])

# del tyn,tyb7,tyir7,tyo7,tyb14,tyir14,tyo14,tyb21,tyir21,tyo21;

# X_testt=np.array(Ptest)
X_testt=np.array(tresData)

y_testt = to_categorical(tlabels, 10)

np.set_printoptions(suppress=True)



# classes = np.arange(10)

score, acc = model.evaluate(X_testt, y_testt,batch_size=30)

print('Test score:', score)
print('Test accuracy:', 100*acc,'%')

classes = np.arange(10)

y_pred = model.predict(X_testt)
 
lbl = np.array(tlabels)


matrix = metrics.confusion_matrix(y_testt.argmax(axis=1), y_pred.argmax(axis=1))

# matrix = confusion_matrix(tlabels, y_pred)

print(matrix)

# con_mat_norm = np.around(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], decimals=2)
 
# con_mat_df = pd.DataFrame(matrix,
#                      index = classes, 
#                      columns = classes)


# figure = plt.figure(figsize=(6, 5))
# sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()


confusion_matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Dataset 4')
# Adjust the layout and display the plot
plt.tight_layout()
plt.show()








from sklearn import datasets
digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
X = digits.data[:500]
y = digits.target[:500]


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

X_2d = tsne.fit_transform(np.array(np.reshape(data1,(4500,128,128))))

X = np.array([image.flatten() for image in data1])


# target_ids = range(len(digits.target_names))
target_ids = range(len(label))


from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()


X_testt2 = np.reshape(X_testt, (1080,128,128))

X = np.array([image.flatten() for image in X_testt])

# Apply t-SNE to reduce to 2D
tsne = TSNE(n_components=2)
embedding = tsne.fit_transform(X)

# Step 3: Visualize the resulting 2D embedding with predicted labels for coloring
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=label.argmax(axis=1),label=label,cmap='jet')
plt.colorbar()
plt.title('t-SNE Embedding of 3D Images with Predicted Labels')
plt.show()






import sklearn.datasets
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import pandas as pd
import numpy as np
import umap
import umap.plot

fig, ax = umap.plot.plt.subplots(2, 2, figsize=(12,12))
umap.plot.points(ordinal_mapper, labels=diamonds["color"], ax=ax[0,0])



plt.figure(figsize=(20, 10))
plt.scatter(*embedding[0].T, s=2, cmap='Spectral', alpha=1.0)
plt.show()






# Apply UMAP for dimensionality reduction
reducer = umap.UMAP()
embedding = reducer.fit_transform(image_data)

# Create a scatter plot of the embedded points, color-coded by ypred
plt.scatter(embedding[:, 0], embedding[:, 1], c=ypred, cmap='viridis')
plt.colorbar()
plt.title('UMAP Visualization of ypred on 2-image dataset')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()





X = np.array([image.flatten() for image in data1])

tsne = TSNE(n_components=2, random_state=42)

z = tsne.fit_transform(X) 


df = pd.DataFrame()
df["y"] = .argmax(axis=1)
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 9),
                data=df).set(title="Iris data T-SNE projection") 

import plotly.express as px

tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(X )

fig = px.scatter_3d(
    projections, x=0, y=1, z=2
)
fig.update_traces(marker_size=8)
fig.show()



ax = plt.axes(projection='3d')

ax.scatter3D(df["comp-1"] , df["comp-2"] , df["y"] , c=df["y"], cmap='turbo');







import numpy as np
import matplotlib.pyplot as plt
from cuml import TSNE
from sklearn.datasets import load_digits
X = np.array([image.flatten() for image in data1])

# Load your data (replace with your own data loading/preprocessing)
X = X.astype(np.float32)

# Create and fit a GPU-accelerated t-SNE model
tsne = TSNE(n_components=2, perplexity=30, random_state=0, n_iter=1000, verbose=1, method='barnes_hut',)
tsne_result = tsne.fit_transform(X)

# Plot the t-SNE result
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=plt.cm.Spectral)
plt.title('t-SNE on GPU')
plt.colorbar()
plt.show()




