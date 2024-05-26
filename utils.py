#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:17:51 2020

@author: bakrnajdi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os,glob
import numpy.linalg as linalg
import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from scipy import signal
import pickle

def Train_data(path):

    datapaths = []
    for i in range(3):
        for j in range(2):
            path1 = path.format(i=str(i+1),j =str(j+1))
            datapaths.append(path1)
    
    allfiles = []
    for i in range(len(datapaths)):
        bearing = glob.glob(datapaths[i] + "/*.csv")
        allfiles.append(bearing)
        
    data = []
    
    for i in allfiles:
        for filename in i:
            df = pd.read_csv(filename, index_col=None, header=None)
            data.append(df)
    
    Dataa = pd.DataFrame()

    for i in range(len(data)):
        Dataa = pd.concat([Dataa,data[i][4]],axis=1)
    
    dataa = Dataa.T
    return dataa,allfiles


def Test_data(path):

    datapaths = []
    for i in range(2):
        for j in range(2,7):
            path1 = path.format(i=str(i+1),j =str(j+1))
            datapaths.append(path1)
            print(j)
    path1 = path.format(i=str(3),j =str(3))
    datapaths.append(path1)
    
    allfiles = []
    for i in range(len(datapaths)):
        bearing = glob.glob(datapaths[i] + "/*.csv")
        allfiles.append(bearing)
        
    data = []
    
    for i in allfiles:
        for filename in i:
            df = pd.read_csv(filename, index_col=None, header=None)
            data.append(df)

    Dataa = pd.DataFrame()

    for i in range(len(data)):
        Dataa = pd.concat([Dataa,data[i][4]],axis=1)

    dataa = Dataa.T
    return dataa,allfiles

def Train_data1(path):

    datapaths = []
    for i in range(3):
        for j in range(2):
            path1 = path.format(i=str(i+1),j =str(j+1))
            datapaths.append(path1)
    
    allfiles = []
    for i in range(len(datapaths)):
        bearing = glob.glob(datapaths[i] + "/*.csv")
        allfiles.append(bearing)
        
    data = []
    
    for i in allfiles:
        for filename in i:
            df = pd.read_csv(filename, index_col=None, header=None)
            data.append(df)
    
    Dataa = pd.DataFrame()

    for i in range(len(data)):
        Dataa = pd.concat([Dataa,data[i][4]],axis=1)
    
    dataa = Dataa.T
    return dataa,allfiles


def Test_data1(path):

    datapaths = []
    for i in range(1):
        for j in range(2,5):
            path1 = path.format(i=str(i+1),j =str(j+1))
            datapaths.append(path1)
    #path1 = path.format(i=str(3),j =str(3))
    #datapaths.append(path1)
    
    allfiles = []
    for i in range(len(datapaths)):
        bearing = glob.glob(datapaths[i] + "/*.csv")
        allfiles.append(bearing)
        
    data = []
    
    for i in allfiles:
        for filename in i:
            df = pd.read_csv(filename, index_col=None, header=None)
            data.append(df)
    
    Dataa = pd.DataFrame()

    for i in range(len(data)):
        Dataa = pd.concat([Dataa,data[i][4]],axis=1)
    
    dataa = Dataa.T
    return dataa,allfiles

def pp_pca(A,m):

    C = np.cov(np.transpose(A))
    n = C.shape[0]
    Ev, V = linalg.eigh(C)
    idx = Ev.argsort()[::-1]   
    Ev = Ev[idx]
    
    V = V[:,idx]
    
    P = V[:,idx[n-m:n]];
    
    return P,Ev
# =============================================================================
# 
# def rp(bidim ,m ):
#     np.random.seed(0)
#     Rpm = np.random.random(m)
#     q = bidim @ Rpm
#     return q
# 
# =============================================================================
from sklearn.random_projection import GaussianRandomProjection

def rp(bidim,n_components):
    rng = np.random.RandomState(0)
    transformer = GaussianRandomProjection(random_state=rng,n_components=n_components)
    X_new = transformer.fit_transform(bidim)
    return X_new


def cwt_rp(signal,scales1,wavelet,n_components):
    if wavelet == 'morl':
        scales = np.arange(1,scales1)
        coef, freqs=pywt.cwt(signal,scales,wavelet)
        bidim = abs(coef)
#        bidim  = rp(bidim ,n_components)
#        X = MinMaxScaler(feature_range=(0,1))
#        X.fit(bidim)
#        bidim=X.transform(bidim)

    elif wavelet == 'mexh':
        scales = np.arange(1,scales1)
        coef, freqs=pywt.cwt(signal,scales,wavelet)
        bidim = abs(coef)**2
        bidim  = rp(bidim ,n_components)
    elif wavelet == 'ben':
        scales = np.arange(1,scales1)
        coef, freqs=pywt.cwt(signal,scales,wavelet)
        bidim = abs(coef)**2
        bidim  = rp(bidim ,n_components)
    
    return bidim



def cwt_rp_ob(signal,scales,wavelet,n_components,f_max,f_min):
    if wavelet == 'morlet':
        coef = cwt(signal, 0.01,5, f_min, f_max,nf =scales,wl=wavelet)
        bidim = coef
        bidim  = rp(bidim ,n_components)

    elif wavelet == 'mexh':
        coef = cwt(signal, 0.01,5, f_min, f_max,nf =scales,wl=wavelet)
        bidim = abs(coef)**2
        bidim  = rp(bidim ,n_components)
        
    elif wavelet == 'ben':
        coef = cwt(signal, 0.01,7, f_min, f_max,nf=scales,wl=wavelet)
        bidim = abs(coef)
        bidim  = rp(bidim ,n_components)
    
    return abs(bidim)



def alg1(tr , tes ,n_features ):
    

    Ptrain = 0
    Ptest = 0
    n_train,d1,d2 = tr.shape
    n_test,d1,d2 = tes.shape
    n=n_train+n_test
    
    Xx=np.concatenate([tr,tes])
    
    del tr,tes
    
    X = []
    for i in range(n):
        a = np.divide(Xx[i],(np.linalg.norm(Xx[i])))
        X.append(a)
        print(i)
    X = np.array(X)
    
    nca=128
    Y= np.zeros((n,128))
    for i in range(n):
        for j in range(nca):
            a = max(X[i][j,:])
            print(i)
            Y[i][j] = a
    
    del X
    Z =Y
    del Y
    Zz = Z[:n_train]
    P,Ev = pp_pca (Zz,n_features)
    
    Ptrain = np.dot(Z[:n_train],P)
    Ptest = np.dot(Z[n_train:n],P)
    
    return Ptrain,Ptest

def alg11(tr ,n_features ):
    

    Ptrain = 0
    n,d1,d2 = tr.shape
    
    Xx=tr
    
    del tr
    
    X = []
    for i in range(n):
        a = np.divide(Xx[i],(np.linalg.norm(Xx[i])))
        X.append(a)
        print(i)
    X = np.array(X)
    
    nca=128
    Y= np.zeros((n,128))
    for i in range(n):
        for j in range(nca):
            a = max(X[i][j,:])
            print(i)
            Y[i][j] = a
    
    del X
    Z =Y
    del Y
    P,Ev = pp_pca (Z,n_features)
    
    Ptrain = np.dot(Z,P)
    
    return Ptrain

def CWTANN_HI(train,test,scales,wavelet,rp_co,n_features):
    
    Data = []
    for i in range(len(train)):
        Data.append(cwt_rp(train.iloc[i],scales,wavelet,rp_co))
        print(i)
    tData = []
    for i in range(len(test)):
        tData.append(cwt_rp(test.iloc[i],scales,wavelet,rp_co))
        print(i)

    Data = np.concatenate([Data])
    tData = np.concatenate([tData])
    
    
    Ptrain,Ptest = alg1(Data,tData)
    
    return Ptrain,Ptest

def CWTANN_HIM(train,test,scales,wavelet,rp_co):
    
    Data = []
    for i in range(len(train)):
        Data.append(cwt_rp(train.iloc[i],scales,wavelet,rp_co))
        print(i)
    tData = []
    for i in range(len(test)):
        tData.append(cwt_rp(test.iloc[i],scales,wavelet,rp_co))
        print(i)

    Data = np.concatenate([Data])
    tData = np.concatenate([tData])
    
    
    
    return Data,tData

def CWTANN_HIM1(train,scales,wavelet,rp_co,f_max,f_min):
    
    Data = []
    for i in range(len(train)):
        Data.append(cwt_rp_ob(train.iloc[i],scales,wavelet,rp_co,f_max,f_min))
        print(i)

    #Data = np.concatenate([Data])
    
    
    
    return Data

def CWTANN_HIM1(train,scales,wavelet,rp_co,f_max,f_min):
    
    Data = []
    for i in range(len(train)):
        Data.append(abs(cwt(train[i], 0.01,7, f_min, f_max,nf =scales,wl=wavelet))**2)
        print(i)
    
    #Data = np.concatenate([Data])
    
    
    
    return Data
       

def cluster(clustern):
    if clustern == 0:
        treeshold = 0.83
    elif clustern ==1:
        treeshold = 0.86
    elif clustern ==2:
        treeshold = 1.18
    elif clustern ==3:
        treeshold = 0.4
    elif clustern ==4:
        treeshold = 0.93
        
    return treeshold




