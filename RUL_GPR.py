#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:49:27 2020

@author: bakrnajdi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os,glob
import random
import numpy.linalg as linalg
import seaborn as sns
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,BatchNormalization
from keras.layers import Dense,Dropout,Activation
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import backend
from sklearn import metrics
# =============================================================================
# DATA PREPARATION AND WRANGLING
# =============================================================================

train_path = '/Users/bakrnajdi/Desktop/stage/Datasets/ieee-phm-2012-data-challenge-dataset/Learning_set/Bearing{i}_{j}'
test_path = '/Users/bakrnajdi/Desktop/stage/Datasets/ieee-phm-2012-data-challenge-dataset/Test_set/Bearing{i}_{j}'


data,all_files =Train_data1(train_path)
tdata,all_filest = Test_data(test_path)

all_filest = all_filest + all_filest2
pd.DataFrame(data).to_csv(r'/Users/bakrnajdi/Desktop/stage/Test_wavelets/Predictive_maintenance_proj/Rul_Project/Train_data_H.csv')
data1 = pd.read_csv('Train_data_H.csv',index_col=[0])
path = test_path
datapaths = []
for i in range(2):
    for j in range(2,7):
        path1 = path.format(i=str(i+1),j =str(j+1))
        datapaths.append(path1)
        print(j)
path1 = path.format(i=str(3),j =str(3))
datapaths.append(path1)

allfilest = []
for i in range(len(datapaths)):
    bearing = glob.glob(datapaths[i] + "/*.csv")
    allfilest.append(bearing)
# =============================================================================
# m = 2
# n =2
# datapathst = []
# path1t = '/Users/bakrnajdi/Desktop/stage/Datasets/ieee-phm-2012-data-challenge-dataset/Test_set/Bearing{m}_{n}'.format(m=str(m+1),n =str(n+1))
# datapathst.append(path1t)
# #path1 = path.format(i=str(3),j =str(3))
# #datapaths.append(path1)
# 
# allfilest = []
# for i in range(len(datapathst)):
#     bearing = glob.glob(datapathst[i] + "/*.csv")
#     allfilest.append(bearing)
#     
# data = []
#     
# for i in allfilest:
#     for filename in i:
#         df = pd.read_csv(filename, index_col=None, header=None)
#         data.append(df)
# Dataa = pd.DataFrame()
# 
# for i in range(len(data)):
#     Dataa = pd.concat([Dataa,data[i][4]],axis=1)
#     print(i)
# dataa = Dataa.T
# del Dataa
# 
allfiles=[]
for i in range(6):
    a= len(all_files[i])
    allfiles.append(a)
rp_co =128
scales = 129
wavelet = 'morl'
data = pd.read_csv('Bearing1_3.csv',index_col=[0]).iloc[:,1128:1256]

Data = pd.read_csv('Train_data_H.csv',index_col=[0])
Data = Data.reset_index(drop=True)
Data = Data.drop(Data.index[[3686]])

n = 2560
signal = []
for i in range(len(Data)):
    idx = random.randint(0,n)
    if idx>=(n-128):
        sample = Data.iloc[i][idx-128:idx]
    else: 
        sample = Data.iloc[i][idx:idx+128]
    signal.append(np.array(sample))
    
signal = np.concatenate([signal],axis=0)
signal = signal.reshape(signal.shape[0]*signal.shape[1],1)
sns.set("white")
sns.set_style("white")# Generate data

x = np.linspace(0,28030,len(signal))
y = signal.ravel()

# setup figures
fig = plt.figure(figsize=((12,5)))

plt.plot(x, y,'mediumblue')
plt.xlabel('time(s)')
plt.ylabel('Vibration Amplitude(g)')
plt.title('Bearing1_1 life time', fontsize=20)
plt.grid(False)
plt.axis([-1000, 30000, -30, 30]) 
plt.show()



rp_co=128
scales=128
wavelet = 'ben'
f_min =0.05
f_max = 200
data = CWTANN_HIM1(Data,scales,wavelet,rp_co,f_min,f_max)

data = cwt(Data.iloc[2801],0.01,7, f_min, f_max,nf=scales,wl='ben')
data,freqs = pywt.cwt(Data.iloc[2801],np.arange(1,129),'morl')

plt.imshow(abs(m), cmap='jet', aspect='auto') 
plt.colorbar()
plt.xlabel('time')
plt.ylabel('scale')
plt.title('0%')
# data = CWTANN_HIM1(data,scales,wavelet,rp_co)
Ptrain = alg11(data,20)

pd.DataFrame(allfiles).to_csv(r'/Users/bakrnajdi/Desktop/stage/Test_wavelets/Predictive_maintenance_proj/Rul_Project/all_files.csv')
all_files = pd.read_csv('all_files.csv',index_col=[0])
# =============================================================================

data = pd.read_csv('Train_data_H.csv',index_col=[0]).iloc[:,1128:1256]
tdata= pd.read_csv('Test_data_H.csv',index_col=[0]).iloc[:,128:256]
tdata = tdata.iloc[:,128:256]

data = pd.read_csv('Train_data_H.csv',index_col=[0])
bidim = cwt_rp(data.iloc[2802],129,'morl',128)

plt.imshow(bidim)
from matplotlib.ticker import NullFormatter

plt.subplot(151)
plt.imshow(abs(cwt_rp(data.iloc[20],129,'morl',128))**2, cmap='rainbow', aspect='auto') 
plt.colorbar()
plt.xlabel('time')
plt.ylabel('scale')
plt.title('0%')

plt.subplot(152)
plt.imshow(abs(cwt_rp(data.iloc[750],129,'morl',128))**2, cmap='rainbow', aspect='auto') 
plt.colorbar()
plt.xlabel('time')
plt.ylabel('scale')
plt.title('25 %')

plt.subplot(153)
plt.imshow(abs(cwt_rp(data.iloc[1400],129,'morl',128))**2, cmap='rainbow', aspect='auto') 
plt.colorbar()
plt.xlabel('time')
plt.ylabel('scale')
plt.title('50%')

plt.subplot(154)
plt.imshow(abs(cwt_rp(data.iloc[2100],129,'morl',128))**2, cmap='rainbow', aspect='auto') 
plt.colorbar()
plt.xlabel('time')
plt.ylabel('scale')
plt.title('75%')

plt.subplot(155)
plt.imshow(abs(cwt_rp(data.iloc[2802],129,'morl',128))**2, cmap='rainbow', aspect='auto') 
plt.colorbar()
plt.xlabel('time')
plt.ylabel('scale')
plt.title('100%')

plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=1.0, bottom=0.5, left=0.10, right=2.5, hspace=0.75,
                    wspace=0.3)
plt.show()

sns.heatmap(abs(bidim),cmap='rainbow', annot=True)
# =============================================================================
# Wavelet Transform of signals
# =============================================================================

rp_co =128
scales = 129
wavelet = 'morl'

Data,tData = CWTANN_HIM(data,tdata[:len(all_filest[0])],scales,wavelet,rp_co)
data = np.delete(data,3686)

resData = []
for i in range(len(data)):
    m=data[i]
    m = np.resize(m,(128,128,1))
    resData.append(m)
del data


data1=np.array(resData)
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
del data1

data  = data1
del data1

# =============================================================================
# Build ANN_model
# =============================================================================


Regressor = Sequential()

#network.add(Dense(units=4096, activation='relu'))
Regressor.add(Dense(units=512, activation='relu'))

Regressor.add(Dense(units=128, activation='relu'))
Regressor.add(Dense(units=64, activation='relu'))
#network.add(Dense(units=32, activation='relu'))
Regressor.add(Dense(units=10, activation='relu'))
#keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
Regressor.add(Dense(units=1, activation='sigmoid'))
optimizer = keras.optimizers.Adam(lr=6.8e-4)

Regressor.compile(loss='mse', optimizer=optimizer, metrics=['mse']) 
#callbacks = [TerminateOnBaseline(monitor='accuracy', baseline=1.0)]
#callbacks = [TerminateOnBaseline(monitor='val_accuracy', baseline=1.0)]
#callbacks = [TerminateOnBaseline(monitor1='accuracy',monitor2='val_accuracy',baseline=1.0)]          

# stopping_criterions =[
#     keras.callbacks.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience = 1000),
#     keras.callbacks.callbacks.EarlyStopping(monitor='val_accuracy', baseline=1.0, patience =0)

# ]
history = Regressor.fit(X_train, y_train, 
            epochs=50,  batch_size=5, 
            validation_data=(X_test,y_test  )) 
import keras as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
# =============================================================================
# Build CNN model
# =============================================================================
Regressor = Sequential()

# Step 1 - Convolution
Regressor.add(Conv2D(32, (3, 3), input_shape = (128,128,1), activation = 'relu'))
#Regressor.add(BatchNormalization())
Regressor.add(MaxPooling2D(pool_size = (2, 2)))
Regressor.add(Conv2D(64, (3, 3), activation = 'relu'))
#Regressor.add(BatchNormalization())
Regressor.add(MaxPooling2D(pool_size = (2, 2)))
Regressor.add(Conv2D(128, (3, 3), activation = 'relu'))
#Regressor.add(BatchNormalization())
Regressor.add(MaxPooling2D(pool_size = (2, 2)))
Regressor.add(Conv2D(256, (3, 3),  activation = 'relu'))
#Regressor.add(BatchNormalization())
Regressor.add(MaxPooling2D(pool_size = (2, 2)))
# =============================================================================
# classifier.add(Conv2D(512, (3, 3),  activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# =============================================================================
# Step 2 - Neural Netwokr configuration
Regressor.add(Flatten())
Regressor.add(Dense(activation='relu',units=2560))
#Regressor.add(Activation('relu'))
#Regressor.add(BatchNormalization())
Regressor.add(Dropout(0.3))
Regressor.add(Dense(activation='relu',units=768))
#Regressor.add(BatchNormalization())
#Regressor.add(Activation('relu'))
Regressor.add(Dropout(0.1))
#Regressor.add(BatchNormalization())
Regressor.add(Dense(1, activation='sigmoid'))
#optimizer = keras.optimizers.Adam(lr=6.8e-05)
optimizer = keras.optimizers.Adam(lr=6.8e-05)

# Step 3 - Compiling the model
#Regressor.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
Regressor.compile(loss=rmse, optimizer=optimizer, metrics=['mse'])

epochs = 10
batch_size = 128
#history = Regressor.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_test,y_test),shuffle = True)
history = Regressor.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)

# ===========================================

Regressor.save("model1.h5")
Regressor.save("model11.h5")

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))


Regressor = load_model('model4.h5',custom_objects={'rmse':rmse})

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

X_train = np.concatenate([X_train,X_test])
labels = np.concatenate([y_train,y_test])

# =============================================================================
# Save the model
# =============================================================================
Regressor.save("rul_0.0029.h5")
Regressor = load_model('all_rul_0.0012.h5')

# =============================================================================
# Build CNN model
# =============================================================================
Regressor = Sequential()

# Step 1 - Convolution
Regressor.add(Conv2D(32, (3, 3), input_shape = (128,128,1),use_bias=False))
Regressor.add(BatchNormalization(axis=-1, center=True, scale=False))
Regressor.add(Activation('relu'))
Regressor.add(MaxPooling2D(pool_size = (2, 2)))

Regressor.add(Conv2D(64, (3, 3),use_bias=False))
Regressor.add(BatchNormalization(axis=-1, center=True, scale=False))
Regressor.add(Activation('relu'))
Regressor.add(MaxPooling2D(pool_size = (2, 2)))

Regressor.add(Conv2D(128, (3, 3),use_bias=False))
Regressor.add(BatchNormalization(axis=-1, center=True, scale=False))
Regressor.add(Activation('relu'))
Regressor.add(MaxPooling2D(pool_size = (2, 2)))

Regressor.add(Conv2D(256, (3, 3),use_bias=False))
Regressor.add(BatchNormalization(axis=-1, center=True, scale=False))
Regressor.add(Activation('relu'))
Regressor.add(MaxPooling2D(pool_size = (2, 2)))
# =============================================================================
# classifier.add(Conv2D(512, (3, 3),  activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# =============================================================================
# Step 2 - Neural Netwokr configuration
Regressor.add(Flatten())
Regressor.add(Dense(units=2560))
Regressor.add(BatchNormalization(axis=-1, center=True, scale=False))
Regressor.add(Activation('relu'))
#Regressor.add(Dropout(0.3))
Regressor.add(Dense(units=768))
Regressor.add(BatchNormalization(axis=-1, center=True, scale=False))
#Regressor.add(Activation('relu'))
#Regressor.add(Dropout(0.1))
#Regressor.add(BatchNormalization())
Regressor.add(Dense(1, activation='sigmoid'))
#optimizer = keras.optimizers.Adam(lr=6.8e-05)
optimizer = keras.optimizers.Adam(lr=6.8e-05)

# Step 3 - Compiling the model
#Regressor.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
Regressor.compile(loss= 'mse', optimizer=optimizer, metrics=['mse'])

epochs = 5
batch_size = 256
history = Regressor.fit(data1, labels, batch_size=batch_size, epochs=epochs,shuffle = True)
#history = Regressor.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),shuffle = True)


Regressor.save("model1.h5")

Regressor = load_model('model1.h5')
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))


# =============================================================================
# Save the model
# =============================================================================
Regressor.save("rul_0.0029.h5")
Regressor = load_model('all_rul_0.0012.h5')



plt.plot(Regressor.history['loss'], label='val_loss')
plt.plot(Regressor.history['val_loss'], label='val loss')
plt.title('MSE')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()


# =============================================================================
# Evaluation 
# =============================================================================
Regressor.summary()
test = pd.read_csv('Bearing2_3.csv',index_col=[0])
pd.DataFrame(y_predt).to_csv(r'/Users/bakrnajdi/Desktop/stage/Test_wavelets/Predictive_maintenance_proj/Rul_Project/ttpred3_3.csv')

rp_co=128
scales=128
wavelet = 'ben'
f_min =0.05
f_max = 200
tdata = CWTANN_HIM1(test,scales,wavelet,rp_co,f_min,f_max)



# data = CWTANN_HIM1(data,scales,wavelet,rp_co)
Ptrain = alg11(tdata,20)

pred = Regressor.predict(np.array(tresData))
pd.DataFrame(pred).to_csv(r'/Users/bakrnajdi/Desktop/stage/Test_wavelets/Predictive_maintenance_proj/Rul_Project/HI_2_3.csv')

tresData = []
for i in range(len(tdata)):
    m=tdata[i]
    m = np.resize(m,(128,128,1))
    tresData.append(m)
del Data

ty0 = pd.DataFrame(np.linspace(0.0,1.0,len(allfilest[0])))
ty1 = pd.DataFrame(np.linspace(0.0,1.0,len(all_filest[1])))
ty2 = pd.DataFrame(np.linspace(0.0,1.0,len(allfilest[2])))


tlabel  = ty0
tlabels =np.array(pd.concat(tlabel,ignore_index=True))
tdata1=np.array(tresData)
del tresData

# Predict
pred = Regressor.predict(data1[5897:7534])
# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y5))
print("Final score (RMSE): {}".format(score))

result = np.concatenate([100*abs(pred),100*y0],axis=1)

score, acc = classifier.evaluate(tdata1, ty0,
                            batch_size=32)

y_pred0 = Regressor.predict(data1[:2803])
y_pred1 = Regressor.predict(data1[2803:3674])
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
================Training Data Visualization========================================
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
# =============================================================================
# Building the Gaussian Process Regression Model
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared , DotProduct,ConstantKernel


y = np.array(pd.read_csv('HI_3_3.csv',index_col=[0])).ravel()
#y= np.array(y).ravel()
sns.scatterplot(y=y.ravel(),x=np.arange(len(y)))

#y= np.array(y_predt).ravel()
X = np.arange(len(y)).reshape(-1,1)


y = np.delete(y, np.where(y>0.2))
#Kernel with parameters given in GPML book
k1 = 70.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0)  \
    * ExpSineSquared(length_scale=1.3, periodicity=5.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4


gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0.,
                              optimizer='fmin_l_bfgs_b', normalize_y=True)
gp.fit(X, y)

print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))
print(gp.score(X, y))


 
# Kernel with optimized parameters
#k1 = 66.0**2 * RBF(length_scale=67.0) + 0.316 **2*DotProduct(sigma_0=3)**2  # long term smooth rising trend
k1 = 66.0**2 * RBF(length_scale=10.0) + 0.316 **2*DotProduct(sigma_0=1) # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0)  \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0,
                      periodicity_bounds="fixed")  # seasonal component

# medium term irregularities
k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)

# k4 = 0.18**2 * RBF(length_scale=0.134) \
#     + WhiteKernel(noise_level=0.361,
#                   noise_level_bounds=(1e-3, np.inf))  # noise terms
k4 = WhiteKernel(noise_level=0.361,
                  noise_level_bounds=(1e-3, np.inf))

kernel = k1 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              normalize_y=True)
gp.fit(X, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))
print(gp.score(X, y))


#X_ = np.linspace(X.min(), X.max() + 1000, 1000)[:, np.newaxis]
#Xstar = np.linspace(X.min(), X.max() + 1500, 1000)[:, np.newaxis]
X_ = np.arange(X.max() + 600)[:, np.newaxis]

#X_ = np.arange(X.max())[:, np.newaxis]

y_pred, y_std = gp.predict(X_, return_std=True)

pd.DataFrame(y_pred).to_csv(r'/Users/bakrnajdi/Desktop/stage/Test_wavelets/Predictive_maintenance_proj/Rul_Project/GPHI_3_3.csv')


ypred1 = signal.resample(y_pred, 2000)

# load the model from disk
filename = 'clustering_model.sav'
kmp = pickle.load(open(filename, 'rb'))

a =   ypred1.reshape(1,2000,1)

clustern =  kmp.predict(a)

treeshold = cluster(clustern)

#treeshold = 1.0
#Visualize Results
fig, ax = plt.subplots(figsize=(10, 6))
# Plot training data.
sns.scatterplot(x=10 * X.ravel(), y=y, label='training data', ax=ax,color='k',s=5);

# Plot "true" linear fit.

# Plot corridor. 
ax.fill_between(
    x=10 *X_[:,0].ravel(), 
    y1=y_pred- 2 *y_std, 
    y2=y_pred+ 2 *y_std, 
    color='black',
    alpha=0.2, 
    label='95 % Confidence Interval'
)
# Plot prediction. 
sns.lineplot(x=10 * X_[len(X):].ravel(), y=y_pred[len(y):], color='green', label='Predictions')
sns.lineplot(x=10 *X_[:len(X)].ravel(), y=y_pred[:len(y)], color='red', label='Estimations')
ax.axhline(treeshold, ls='--',label='treeshold',color='black')
idx = 10 * np.argwhere(np.diff(np.sign(y_pred - treeshold)))[-1].flatten() 


ax.axvline(idx, ls='--',label='Failure time')
ax.set(title='Bearing3_3')
ax.legend(loc='upper left')
ax.set(xlabel='Time (s)', ylabel='Health Indicator')
#y1 = pd.read_csv('ypred0.csv',index_col=[0])



rul = idx - 10 * len(y)

print(rul[0],'s')










for i in X_train:
    
    sns.lineplot(y=i.ravel(),x=np.arange(len(i)))
    












