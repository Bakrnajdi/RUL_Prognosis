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

bearing_data = {bearing: np.load(f'{bearing}.npy') for bearing in bearings}


# Regressor = load_model('model1.h5',custom_objects={'rmse_loss_fn':rmse_loss_fn})


hi_predictions = {bearing: Regressor.predict(bearing_data[bearing]) for bearing in bearings}

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

gpr_models = {bearing: GaussianProcessRegressor() for bearing in bearings}
rul_forecasts = {}
y_stds = {}

for bearing in bearings:
    hi_data = hi_predictions[bearing]
    X = np.arange(len(hi_data)).reshape(-1, 1)
    y = hi_data
    
    # Fit the GPR model
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
    gpr_models[bearing] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    gpr_models[bearing].fit(X, y)
    
    # Forecasting 2000 points after the last value of y
    X_forecast = np.arange(len(hi_data), len(hi_data) + 2000).reshape(-1, 1)
    X_ = np.vstack((X, X_forecast))
    y_pred, y_std = gpr_models[bearing].predict(X_, return_std=True)
    rul_forecasts[bearing] = y_pred
    y_stds[bearing] = y_std

import matplotlib.pyplot as plt
import seaborn as sns

def plot_gpr(bearing, X, y, X_, y_pred, y_std, threshold=1.0):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Flatten the arrays for seaborn plotting
    X_flat = 10 * X.ravel()
    y_flat = y.ravel()
    X_forecast_flat = 10 * X_forecast.ravel()
    y_pred_flat = y_pred.ravel()
    y_std_flat = y_std.ravel()
    
    # Plot training data
    sns.scatterplot(x=X_flat, y=y_flat, label='Training data', ax=ax, color='k', s=5)
    
    # Plot prediction interval
    ax.fill_between(
        x=10 * X_[:, 0].ravel(), 
        y1=y_pred - 2 * y_std, 
        y2=y_pred + 2 * y_std, 
        color='black',
        alpha=0.2, 
        label='95% Confidence Interval'
    )
    
    # Plot predictions and estimations
    sns.lineplot(x=10 * X_[len(X):].ravel(), y=y_pred[len(y):], color='green', label='Predictions')
    sns.lineplot(x=10 * X_[:len(X)].ravel(), y=y_pred[:len(y)], color='red', label='Estimations')
    
    # Plot threshold and failure time
    ax.axhline(threshold, ls='--', label='Threshold', color='black')
    if np.any(y_pred >= threshold):
        idx = 10 * np.argwhere(np.diff(np.sign(y_pred - threshold)))[-1].flatten()
        ax.axvline(idx, ls='--', label='Failure time')
    
    # Set titles and labels
    ax.set(title=f'Bearing {bearing}', xlabel='Time (s)', ylabel='Health Indicator')
    ax.legend(loc='upper left')
    plt.show()

# Example plotting for a specific bearing
bearing = "Bearing3_3"
X = np.arange(len(hi_predictions[bearing])).reshape(-1, 1)
y = hi_predictions[bearing]
X_forecast = np.arange(len(hi_predictions[bearing]), len(hi_predictions[bearing]) + 2000).reshape(-1, 1)
X_ = np.vstack((X, X_forecast))
y_pred = rul_forecasts[bearing]
y_std = y_stds[bearing]

plot_gpr(bearing, X, y, X_, y_pred, y_std)


with open('bearing_results.json', 'w') as f:
    json.dump(results, f)
