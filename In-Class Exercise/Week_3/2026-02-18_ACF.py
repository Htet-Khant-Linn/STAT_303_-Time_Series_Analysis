# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 23:18:56 2026

@author: Si Thu Aung
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

num_timesteps = 300  # Length of time series we want
np.random.seed(0)  # Ensures we generate the same random numbers every time

y = np.random.normal(loc=0, scale=1, size=num_timesteps)# loc = mean is 0 and scale = 1 std = 1
ts = pd.date_range(start="2026-01-01", periods=num_timesteps, freq="D")

df = pd.DataFrame(data={"y": y}, index=ts)
df.head()

# Plot the time series
df.plot(figsize=[10, 5])
plt.title("white noise")
plt.ylabel("y")
plt.xlabel("Time")
plt.grid()
plt.tight_layout()

# Create a copy
df_ = df.copy()

# Compute the lag of the target
lag = 1
df_[f"y_lag_{lag}"] = df_["y"].shift(periods=lag)
df_.head()

# Compute the numerator and denominator in the ACF formula.
y_mean = df_["y"].mean()  # Compute mean of y of the dataset.
# Numerator of the ACF formula.
numerator = ((df_["y"] - y_mean) * (df_[f"y_lag_{lag}"] - y_mean)).sum()
# Denominator of the ACF formula.
denominator = ((df_["y"] - y_mean) ** 2).sum()
# Compute the autocorrelation.
r = numerator / denominator
print(r)


# Compute the autocorrelation for multiple lags
r = {}
for lag in range(0, 30):
    df_ = df.copy()
    df_["y_lag"] = df_["y"].shift(lag)
    y_mean = df_["y"].mean()
    numerator = ((df_["y"] - y_mean) * (df_["y_lag"] - y_mean)).sum()
    denominator = ((df_["y"] - y_mean) ** 2).sum()
    r[lag] = numerator / denominator

acf_ = pd.Series(r)

# ACF plot using stem
plt.figure(figsize=[12, 5])
plt.stem(acf_)
plt.title("Autcorrelation of white noise")
plt.ylabel("Autocorrelation")
plt.xlabel("Lag")
plt.tight_layout()

# ACF Plot using Library
plot_acf(
    x=df_["y"],
    lags=50,
    alpha=0.05,
    auto_ylims=True
);
plt.title("Autocorrelation of white noise")
plt.ylabel("Autocorrelation")
plt.xlabel("Lag")
