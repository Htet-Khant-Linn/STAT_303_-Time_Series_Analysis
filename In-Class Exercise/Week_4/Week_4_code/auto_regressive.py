# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:54:13 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# White noise
num_timesteps = 300  # Length of time series we want
np.random.seed(0)  # Ensures we generate the same random numbers every time

y = np.random.normal(loc=0, scale=1, size=num_timesteps)
ts = pd.date_range(start="2026-01-01", periods=num_timesteps, freq="D")

df_white_noise = pd.DataFrame(data={"y": y}, index=ts)

# Plot the time series
ax = df_white_noise.plot(figsize=[10, 5])
ax.set_title("white noise")
ax.set_ylabel("y")
ax.set_xlabel("Time")

fig, ax = plt.subplots(figsize=[10, 5])
plot_acf(
    x=df_white_noise["y"],
    lags=50,
    ax=ax,
    alpha=0.05,
    auto_ylims=True
);
ax.set_title("Autocorrelation of white noise")
ax.set_ylabel("Autocorrelation")
ax.set_xlabel("Lag")

fig, ax = plt.subplots(figsize=[10, 5])
plot_pacf(
    x=df_white_noise["y"],
    method="ywmle",
    lags=50,
    ax=ax,
    alpha=0.05,
    auto_ylims=True

);
ax.set_title("Parial autocorrelation of white noise")
ax.set_ylabel("Partial autcorrelation")
ax.set_xlabel("Lag")
plt.tight_layout()


# Generate AR(1) process
num_timesteps = 300
phi_1 = 0.9 # larger the phi value, larger the correlation value
ts = pd.date_range(start="2026-01-01", periods=num_timesteps, freq="D")

y = np.zeros(num_timesteps)

for t in range(1, num_timesteps):
    noise = np.random.normal()
    y[t] = phi_1 * y[t - 1] + noise
    
df_ar1 = pd.DataFrame(data={"y":y}, index=ts)


# Plot AR(1) process
ax = df_ar1.plot(figsize=[10, 5])
ax.set_title("AR(1) process")
ax.set_ylabel("y")
ax.set_xlabel("Time")
    
fig, ax = plt.subplots(figsize=[10,5])
plot_acf(
    x = df_ar1,
    lags= 36,
    ax = ax,
    alpha=0.05,
    auto_ylims=True
    );
ax.set_title("Autocorrelation of AR(1) process")
ax.set_ylabel("Autocorrelation")
ax.set_xlabel("Lag")
    
fig,ax = plt.subplots(figsize=[10,5])
plot_pacf(
    x=df_ar1,
    method= "ywmle",
    lags = 50,
    ax = ax,
    alpha=0.05,
    auto_ylims=True
    );
ax.set_title("Parial autocorrelation of AR(1) process")
ax.set_ylabel("Partial autcorrelation")
ax.set_xlabel("Lag")
plt.tight_layout()

# Generate AR(2) process
phi_1 = 0.7
phi_2 = 0.3
y = np.zeros(num_timesteps)

for t in range(1, num_timesteps):
    noise = np.random.normal()
    y[t] = phi_1 * y[t - 1] + phi_2 * y[t - 2] + noise


df_ar2 = pd.DataFrame(data={"y":y}, index=ts)

































