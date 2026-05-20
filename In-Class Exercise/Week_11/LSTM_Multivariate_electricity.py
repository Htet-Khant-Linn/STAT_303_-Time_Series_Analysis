# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:49:23 2026

@author: Si Thu Aung
"""

# %% Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Load the Dataset

df = pd.read_csv("Electricity_Consumption.csv")
df.dropna(inplace=True)
df_feaures = df.iloc[:, 1:4]

# %% Feature Correlation
import seaborn as sns

sns.heatmap(df_feaures.corr(), annot=True)

# %% Split the Features

training_set = df.iloc[:8712, 1:4].values
test_set = df.iloc[8712:, 1:4].values

# %% Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)

test_set_scaled = test_set_scaled[:, 0:2]

# %% Create Sequences
X_train = []
y_train = []
window_size = 24

for i in range(window_size, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-window_size:i, 0:3])
    y_train.append(training_set_scaled[i, 2])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3))

# %% Model Development

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

Model = Sequential()

# First layer and input layer
Model.add(LSTM(units=60, return_sequences=True, input_shape=(X_train.shape[1],3)))
Model.add(Dropout(0.2))

# Second layer
Model.add(LSTM(units=60, return_sequences=True))
Model.add(Dropout(0.2))

# Third layer
Model.add(LSTM(units=60, return_sequences=True))
Model.add(Dropout(0.2))

# Fourth layer
Model.add(LSTM(units=60))
Model.add(Dropout(0.2))

# Output layer
Model.add(Dense(units=1))

Model.compile(optimizer='adam', loss='mean_squared_error')

Model.fit(X_train, y_train, epochs=10, batch_size=32)

plt.plot(range(len(Model.history.history['loss'])), Model.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()


# %% Feature Prediction
prediction_test = []

Batch_one = training_set_scaled[-24:]
Batch_new = Batch_one.reshape((1, 24, 3))

for i in range(48):
    First_pred = Model.predict(Batch_new)[0]
    
    prediction_test.append(First_pred)
    
    New_var = test_set_scaled[i, :]
    
    New_var = New_var.reshape(1,2)
    
    New_test = np.insert(New_var, 2, [First_pred], axis=1)
    
    New_test = New_test.reshape(1, 1, 3)
    
    Batch_new = np.append(Batch_new[:, 1:, :], New_test, axis=1)
    
prediction_test = np.array(prediction_test)

SI = MinMaxScaler(feature_range=(0,1))

y_scale = training_set[:, 2:3]

SI.fit_transform(y_scale)

preditions = SI.inverse_transform(prediction_test)

real_values = test_set[:, 2]

plt.plot(real_values, color='red', label='Actual Values')
plt.plot(preditions, color='blue', label='Predicted Values')
plt.title('Electricity Comsumption Prediction')
plt.xlabel('Time (hr)')
plt.ylabel('Electricity Demand (mW)')
plt.legend()
plt.show()
    
# %% Performance Measure

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RMSE = math.sqrt(mean_squared_error(real_values, preditions))
MAE = mean_absolute_error(real_values, preditions)
R_squared = r2_score(real_values, preditions)

print(f"RMSE: {RMSE:.4f}")
print(f"MAE: {MAE:.4f}")
print(f"R_squared: {R_squared:.4f}")


def mean_abosolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_pred)) * 100

MAPE = mean_abosolute_percentage_error(real_values, preditions)
print(f"MAPE: {MAPE:.2f}%")

# On average, our prediction are about 15.92% aways from the true values. 






























