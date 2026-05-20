# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:04:07 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Solar_Data_Set.csv")

df.dropna(inplace=True)

one_day = df.iloc[0:48, 1].values
plt.plot(one_day)

training_set = df.iloc[:8712, 1:2].values
test_set = df.iloc[8712:, 1:2].values

# pip install scikit-learn
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1)) # scaled data to a fixed range(0,1)

training_set_scaled = sc.fit_transform(training_set)

test_set_scaled = sc.transform(test_set)

X_train = []
y_train = []
window_size = 24

for i in range(len(training_set_scaled) - window_size):
    X_train.append(training_set_scaled[i : i+window_size])
    y_train.append(training_set_scaled[i+window_size])

X_train  = np.array(X_train)
y_train = np.array(y_train)

# Developing LSTM Model

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

Model_P = Sequential()

# First layer and input layer
Model_P.add(LSTM(units=60, return_sequences=True, input_shape=(X_train.shape[1],1))) # (samples, window_size=24, 1)
Model_P.add(Dropout(0.2))

# Second layer
Model_P.add(LSTM(units=60, return_sequences=True))
Model_P.add(Dropout(0.2))

# Third layer
Model_P.add(LSTM(units=60, return_sequences=True))
Model_P.add(Dropout(0.2))

# Fourth layer
Model_P.add(LSTM(units=60))
Model_P.add(Dropout(0.2))

# Output layer
Model_P.add(Dense(units=1))

Model_P.compile(optimizer='adam', loss ='mean_squared_error')

Model_P.fit(X_train, y_train, epochs=10, batch_size=32)

plt.plot(range(len(Model_P.history.history['loss'])), Model_P.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()

prediction_test = []

Batch_one = training_set_scaled[-window_size:]
Batch_new = Batch_one.reshape((1, window_size, 1)) # (samples, 24, 1)

for i in range(48):
    First_pred = Model_P.predict(Batch_new)[0]
    
    prediction_test.append(First_pred)
    
    Batch_new = np.append(Batch_new[:, 1:, :], [[First_pred]], axis=1)
    
prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)

plt.plot(test_set, color='red', label='Actual Values')
plt.plot(predictions, color='blue', label='Predicted Values')
plt.title('LSTM Prediction')
plt.xlabel('Time (hr)')
plt.ylabel('Solar')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RMSE = math.sqrt(mean_squared_error(test_set, predictions))
R_squared = r2_score(test_set, predictions)
MAE = mean_absolute_error(test_set, predictions)

print(f"RMSE : {RMSE}")
print(f"R_squared : {R_squared}")
print(f"MAE : {MAE}")













