# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:33:23 2026

@author: Si Thu Aung
"""
# .py or .ipynb 
# %% Project: Feature Engineering + RNN/LSTM/GRU
# Multivariate multi-output time series prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# %% 1. Load Data


file_path = "synthetic_ml_metrics_datetime_2026.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# %% 2. Target Columns


target_cols = ["accuracy", "loss", "precision", "recall", "f1_score"]

# %% 3. Feature Engineering


# There are 8 events/day, choose window size = 8

window_size = 8

def mad(x):
    return np.mean(np.abs(x - np.mean(x)))

for col in target_cols:
    df[f"{col}_roll_mean"] = df[col].rolling(window=window_size).mean()
    df[f"{col}_roll_median"] = df[col].rolling(window=window_size).median()
    df[f"{col}_roll_std"] = df[col].rolling(window=window_size).std()
    df[f"{col}_roll_mad"] = df[col].rolling(window=window_size).apply(mad, raw=True)


extra_cols = [
    "new_data_arrived",
    "retraining_on",
    "retraining_complete",
    "event_order"]

feature_cols = target_cols + extra_cols + [
    c for c in df.columns if "roll_" in c]

df = df.dropna().reset_index(drop=True)

# %% 4. Preparing X and y

X_raw = df[feature_cols].copy()
y_raw = df[target_cols].shift(-1)

X_raw = X_raw.iloc[:-1]
y_raw = y_raw.iloc[:-1]

# %% 5. Train/val/test (time-based)

"""
Train -> 70%
Val -> 15%
Test -> 15%
"""

n = len(X_raw)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_raw = X_raw.iloc[:train_end]
X_val_raw = X_raw.iloc[train_end:val_end]
X_test_raw = X_raw.iloc[val_end:]

y_train_raw = y_raw.iloc[:train_end]
y_val_raw = y_raw.iloc[train_end:val_end]
y_test_raw = y_raw.iloc[val_end:]

# %% 6. Scale features and targets


x_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))


X_train_scaled = x_scaler.fit_transform(X_train_raw)
X_val_scaled = x_scaler.transform(X_val_raw)
x_test_scaled = x_scaler.transform(X_test_raw)

y_train_scaled = y_scaler.fit_transform(y_train_raw)
y_val_scaled = y_scaler.transform(y_val_raw)
y_test_scaled = y_scaler.transform(y_test_raw)

# %% 7. Create sequences

"""
Use past 2 days -> predict Next event

"""

seq_length = 16 # 16 timesteps = 2 days because 8 events/day

def create_sequences(X, y, seq_length=16):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, seq_length)
X_test, y_test = create_sequences(x_test_scaled, y_test_scaled)

print("X_train shape", X_train.shape)
print("y train shape", y_train.shape)



# %% 8. Build models

def build_rnn_model(model_type, input_shape, output_dim):
    
    model = Sequential()
    
    # First layer and input layer
    if model_type == "SimpleRNN":
        model.add(SimpleRNN(64, activation='tanh', input_shape=input_shape))
    elif model_type == "LSTM":
        model.add(LSTM(64, activation='tanh', input_shape=input_shape))
    elif model_type == "GRU":
        model.add(GRU(64, activation="tanh", input_shape=input_shape))
    else:
        raise ValueError("Model type must be SimpleRNN, LSTM or GRU.")
    model.add(Dropout(0.2))
    
    # Second layer
    model.add(Dense(32, activation="relu"))
    
    # Output layer
    model.add(Dense(output_dim))
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    return model

# %% 9. Train and Evaluate

# ===========================================================================
def evaluate_model(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test, y_scaler):
    
    # Early Stopping
    early_stop = EarlyStopping(
                                monitor='val_loss',
                                patience = 10,
                                restore_best_weights=True)
    
    # Model Training
    history = model.fit(
                        X_train, y_train,
                        validation_data = (X_val, y_val),
                        epochs = 100,
                        batch_size = 32,
                        callbacks = [early_stop],
                        verbose = 1)
    
    # Prediction
    y_pred_scaled = model.predict(X_test)
    
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)
    
    result = {}
    result["Model"] = model_name
    result["MAE"] = mean_absolute_error(y_true, y_pred)
    result["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    result["R2"] = r2_score(y_true, y_pred)
    
    return history, y_true, y_pred, result

input_shape = (X_train.shape[1], X_train.shape[2]) # 16, 29
output_dim = y_train.shape[1] # 5

# ============================================================================
models = {
    "SimpleRNN": build_rnn_model("SimpleRNN", input_shape, output_dim),
    "LSTM": build_rnn_model("LSTM", input_shape, output_dim),
    "GRU": build_rnn_model("GRU", input_shape, output_dim)
    }

# ============================================================================

all_result = []
histories = {}
predictions = {}

# ============================================================================
for name, model in models.items():
    history, y_true, y_pred, result = evaluate_model(
                                            name, model, 
                                            X_train, y_train, 
                                            X_val, y_val, 
                                            X_test, y_test, 
                                            y_scaler)
    all_result.append(result)
    histories[name] = history
    predictions[name] = (y_true, y_pred) 
    
# %% 10. Compare Results

result_df = pd.DataFrame(all_result)

result_df.set_index("Model", inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# MAE Plot
result_df["MAE"].plot(kind='bar', ax=axes[0])
axes[0].set_title("MAE Comparison")
axes[0].set_ylabel("MAE")

# RMSE Plot
result_df["RMSE"].plot(kind='bar', ax=axes[1])
axes[1].set_title("RMSE Comparison")
axes[1].set_ylabel("RMSE")

# R2 Plot
result_df["R2"].plot(kind='bar', ax=axes[2])
axes[2].set_title("R2 Comparison")
axes[2].set_ylabel("R2")

plt.show()
    
# %% 11. Plot Training History

plt.figure(figsize=(10,6))

for name, history in histories.items():
    plt.plot(history.history["val_loss"], label=f"{name} Val Loss")

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
    
# %% 12. Plot Prediction three
# target_cols = ["accuracy", "loss", "precision", "recall", "f1_score"]


metric_index = 0 # accuracy
metric_name = target_cols[metric_index]

for name in predictions:
    y_true, y_pred = predictions[name]
    plt.figure(figsize=(10,6))
    plt.plot(y_true[:100, metric_index], label="True", color="blue", linewidth=1)
    plt.plot(y_pred[:100, metric_index], label=f"{name} Pred", color="red", linewidth=1)
    plt.title(f"Prediction for {metric_name}")
    plt.xlabel("test Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
  
#

metric_index = 1 # loss
metric_name = target_cols[metric_index]

for name in predictions:
    y_true, y_pred = predictions[name]
    plt.figure(figsize=(10,6))
    plt.plot(y_true[:100, metric_index], label="True", color="blue", linewidth=1)
    plt.plot(y_pred[:100, metric_index], label=f"{name} Pred", color="red", linewidth=1)
    plt.title(f"Prediction for {metric_name}")
    plt.xlabel("test Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
    

metric_index = 2 # precision
metric_name = target_cols[metric_index]

for name in predictions:
    y_true, y_pred = predictions[name]
    plt.figure(figsize=(10,6))
    plt.plot(y_true[:100, metric_index], label="True", color="blue", linewidth=1)
    plt.plot(y_pred[:100, metric_index], label=f"{name} Pred", color="red", linewidth=1)
    plt.title(f"Prediction for {metric_name}")
    plt.xlabel("test Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


# %%





















