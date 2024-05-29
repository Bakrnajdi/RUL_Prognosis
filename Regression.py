#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:37:07 2024

@author: mac
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

bearings = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3"
]


def rmse_loss_fn(y_true, y_pred):
    """
    Custom RMSE loss function.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: RMSE value.
    """
    squared_difference = ops.square(y_true - y_pred)
    mean_squared_difference = ops.mean(squared_difference, axis=-1)  # Note the `axis=-1`
    return ops.abs(ops.sqrt(mean_squared_difference))

bearing_data = {bearing: np.load(f'Results/{bearing}.npy') for bearing in bearings}


# Regressor = load_model('model1.h5',custom_objects={'rmse_loss_fn':rmse_loss_fn})


hi_predictions = {bearing: Regressor.predict(bearing_data[bearing]) for bearing in bearings}

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel

def train_and_predict_gpr(bearing, hi_data):
    X = np.arange(len(hi_data)).reshape(-1, 1)
    y = hi_data
    
    # Define the complex kernel
    k1 = 66.0**2 * RBF(length_scale=10.0) + 0.316**2 * DotProduct(sigma_0=1)  # long term smooth rising trend
    k2 = 2.4**2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0, periodicity_bounds="fixed")  # seasonal component
    k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)  # medium term irregularities
    k4 = WhiteKernel(noise_level=0.361, noise_level_bounds=(1e-3, np.inf))  # noise terms
    kernel = k1 + k3
    
    # Fit the GPR model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)
    gp.fit(X, y)
    
    print(f"\nLearned kernel for {bearing}: {gp.kernel_}")
    print(f"Log-marginal-likelihood for {bearing}: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")
    
    # Forecasting 2000 points after the last value of y
    X_forecast = np.arange(len(hi_data), len(hi_data) + 2000).reshape(-1, 1)
    X_ = np.vstack((X, X_forecast))
    y_pred, y_std = gp.predict(X_, return_std=True)
    return y_pred, y_std, X, X_forecast





import matplotlib.pyplot as plt
import seaborn as sns

def plot_gpr(bearing, X, y, X_, y_pred, y_std, threshold=1.0):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Flatten the arrays for seaborn plotting
    X_flat = 10 * X.ravel()
    y_flat = y.ravel()
    X_forecast_flat = 10 * X_[:, 0].ravel()
    y_pred_flat = y_pred.ravel()
    y_std_flat = y_std.ravel()
    
    # Plot training data
    sns.scatterplot(x=X_flat, y=y_flat, label='Training data', ax=ax, color='k', s=5)
    
    # Plot prediction interval
    ax.fill_between(
        x=10 * np.concatenate((X[:, 0], X_forecast[:, 0])), 
        y1=y_pred - 2 * y_std, 
        y2=y_pred + 2 * y_std, 
        color='black',
        alpha=0.2, 
        label='95% Confidence Interval'
    )
    
    # Plot predictions and estimations
    sns.lineplot(x=10 * X_forecast[:, 0], y=y_pred[len(y):], color='green', label='Predictions')
    sns.lineplot(x=10 * X[:, 0], y=y_pred[:len(y)], color='red', label='Estimations')
    
    # Plot threshold and failure time
    ax.axhline(threshold, ls='--', label='Threshold', color='black')
    failure_time = None
    if np.any(y_pred >= threshold):
        idx = np.argwhere(y_pred >= threshold)[0][0]
        failure_time = idx - len(y)
        ax.axvline(10 * idx, ls='--', label=f'Failure time: {failure_time}', color='blue')
    
    # Set titles and labels
    ax.set(title=f'Bearing {bearing}', xlabel='Time (s)', ylabel='Health Indicator')
    ax.legend(loc='upper left')
    plt.show()
    
    return failure_time



bearing = "Bearing3_3"
hi_data = hi_predictions[bearing]
y_pred, y_std, X, X_forecast = train_and_predict_gpr(bearing, hi_data)
X_ = np.vstack((X, X_forecast))

failure_time = plot_gpr(bearing, X, hi_data, X_, y_pred, y_std)
if failure_time is not None:
    print(f"The Remaining Useful Life (RUL) for {bearing} is {failure_time} time units.")
else:
    print(f"The Remaining Useful Life (RUL) for {bearing} could not be determined as the threshold was not reached.")





with open('Results/bearing_results.json', 'w') as f:
    json.dump(results, f)
