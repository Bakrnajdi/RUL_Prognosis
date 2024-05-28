#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:18:16 2024

@author: mac
"""


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, Reshape
from keras import ops
import keras
import keras_tuner

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

def build_model(hp):
    model = Sequential()

    # Convolutional Layers
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Optional LSTM Layer
    if hp.Boolean("include_lstm"):
        model.add(Reshape((6, 6 * 256)))  # Reshape for LSTM
        model.add(LSTM(hp.Int('lstm_units', min_value=128, max_value=512, step=128)))

    # Flatten for dense layers
    model.add(Flatten())

    # Determine configuration
    config = hp.Choice('hidden_layer_configuration', values=['config1', 'config2'])
    if config == 'config1':
        print("Training with configuration 1: 2500 and 760 neurons")
        model.add(Dense(units=2500, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=760, activation='relu'))
    else:
        print("Training with configuration 2: 1000 and 100 neurons")
        model.add(Dense(units=1000, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=100, activation='relu'))

    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=6.8e-05),
        loss=rmse_loss_fn,
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    return model


label_files = [
    'Linear_bearing_health_indicators.csv',
    'Exponential_bearing_health_indicators.csv',
    'Segmented_bearing_health_indicators.csv',
    'Spline_Fit_bearing_health_indicators.csv'
]

for file in label_files:
    print(f"Running grid search for: {file}")
    labels = np.array(pd.read_csv(file))

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective='val_root_mean_squared_error',
        max_trials=1,  # Adjust number of trials based on your computation limits
        executions_per_trial=1,
        directory='/Users/mac/Desktop/PhD/Prognosis_RUL/RUL_Prognosis',
        project_name=f'CNN_LSTM_RUL_{file[:-4]}'
    )

    # data1 = np.random.random((1000, 128, 128, 1))  # Example data; replace with actual
    tuner.search(data1, labels, epochs=3, validation_split=0.2, batch_size=256)

    # Optionally, print or save the best model's performance
    best_model = tuner.get_best_models(num_models=1)[0]
    print(f"Best model for {file} has been saved.")





