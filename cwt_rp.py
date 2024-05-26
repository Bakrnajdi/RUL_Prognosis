#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:22:16 2020

@author: bakrnajdi
"""
from sklearn.random_projection import GaussianRandomProjection
# import numpy as np
# import pywt
from sklearn.preprocessing import MinMaxScaler

# import obspy
# from obspy.imaging.cm import obspy_sequential
# from obspy.signal.tf_misfit import cwt

import numpy as np
from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.visuals import imshow, plot
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up
from ssqueezepy.utils import logscale_transition_idx

# import random
from config import set_seed,SEED
set_seed()

# random.seed(10)
# print(random.random()) 
# =============================================================================
# def cwt_rp(signal,scales1,wavelet,n_components):
#     if wavelet == 'morl':
#         scales = np.arange(1,scales1)
#         coef, freqs=pywt.cwt(signal,scales,wavelet)
#         bidim = abs(coef)
#         bidim  = rp(bidim ,n_components)
# #        X = MinMaxScaler(feature_range=(0,1))
# #        X.fit(bidim)
# #        bidim=X.transform(bidim)
#     elif wavelet == 'mexh':
#         scales = np.arange(1,scales1)
#         coef, freqs=pywt.cwt(signal,scales,wavelet)
#         bidim = abs(coef)**2
#         bidim  = rp(bidim ,n_components)
#     elif wavelet == 'ben':
#         scales = np.arange(1,scales1)
#         coef, freqs=pywt.cwt(signal,scales,wavelet)
#         bidim = abs(coef)**2
#         bidim  = rp(bidim ,n_components)
    
#     return bidim
# =============================================================================
def ssq_cwt1(signal,wavelet,scales,padtype,n_components):
    # Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(np.array(signal), wavelet, scales=scales,padtype=padtype)
    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(np.array(signal), wavelet, scales=scales)
    bidim = abs(Tx)
    bidim  = rp(bidim ,n_components)
    # X = MinMaxScaler(feature_range=(0,1))
    # X.fit(bidim)
    # bidim=X.transform(bidim)
    return abs(bidim)

def rp(bidim,n_components, seed=SEED):
    rng = np.random.RandomState(0)
    transformer=GaussianRandomProjection(random_state=seed,n_components=n_components)
    X_new = transformer.fit_transform(bidim)
    return X_new


# def cwt_rp(signal,scales,wavelet,n_components,f_max,f_min):
#     if wavelet == 'morlet':
#         coef = cwt(signal, 0.01,5, f_min, f_max,nf =scales,wl=wavelet)
#         bidim = abs(coef)**2
#         bidim  = rp(bidim ,n_components)
        # X = MinMaxScaler(feature_range=(0,1))
        # X.fit(bidim)
        # bidim=X.transform(bidim)
#     elif wavelet == 'mexh':
#         coef = cwt(signal, 0.01,5, f_min, f_max,nf =scales,wl=wavelet)
#         bidim = abs(coef)**2
#         bidim  = rp(bidim ,n_components)
        
#     elif wavelet == 'ben':
#         coef = cwt(signal, 0.01,7, f_min, f_max,nf=scales,wl=wavelet)
#         bidim = abs(coef)**2
#         bidim  = rp(bidim ,n_components)
    
#     return bidim