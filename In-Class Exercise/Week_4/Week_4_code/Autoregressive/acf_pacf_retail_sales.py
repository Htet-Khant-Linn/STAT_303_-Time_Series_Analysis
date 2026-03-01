# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:22:14 2026

@author: Si Thu Aung
"""

# 1. Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 2. Load the csv file
df = pd.read_csv(
    "retail_sales.csv",
    parse_dates=["ds"],
    index_col=["ds"],
)

# 3. Plot the time series data
ax = df.plot(y="y", marker=".", figsize=[10, 5])
ax.set_title("Retail sales")
ax.set_ylabel("Sales")
ax.set_xlabel("Time")

# 4. Plot acf using the time series data
fig, ax = plt.subplots(figsize=[10, 5])
plot_acf(
    x=df["y"],
    lags=36,
    ax=ax,
    alpha=0.05,
    auto_ylims=True
)
ax.set_title("Autocorrelation of retail sales")
ax.set_ylabel("Autcorrelation")
ax.set_xlabel("Lag")
plt.tight_layout()

# 5. Plot pacf using the time series data
fig, ax = plt.subplots(figsize=[10, 5])
plot_pacf(
    x=df["y"],
    method='ywmle', # Recommended method in Statsmodels notes
    lags=36,
    ax=ax,
    alpha=0.05,
    auto_ylims=True
)
ax.set_title("Partial autocorrelation of retail sales")
ax.set_ylabel("Partial autcorrelation")
ax.set_xlabel("Lag")
plt.tight_layout()

# 6. Extract the trend from time series data using MA (t = 12) 2 * 12-MA
window_size = 12
df["trend"] = (
    df.rolling(window = window_size)
    .mean()
    .rolling(window=2)
    .mean()
    .shift(-window_size//2)
    )


# 6. Detrend the data
df["y_detrended"] = df["y"] - df["trend"]

# 7. Plot the detrend data
df["y_detrended"].plot(figsize=(10,5))


# 8. **Plot again acf and pacf
df_clean = df["y_detrended"].dropna()

fig, ax = plt.subplots(figsize=[10, 5])
plot_acf(
    x=df_clean,
    lags=36,
    ax=ax,
    alpha=0.05,
    auto_ylims=True
)
ax.set_title("Autocorrelation of retail sales (Detrended)")
ax.set_ylabel("Autcorrelation")
ax.set_xlabel("Lag")
plt.tight_layout()

fig, ax = plt.subplots(figsize=[10, 5])
plot_pacf(
    x=df_clean,
    method='ywmle', # Recommended method in Statsmodels notes
    lags=36,
    ax=ax,
    alpha=0.05,
    auto_ylims=True
)
ax.set_title("Partial autocorrelation of retail sales (Detrended)")
ax.set_ylabel("Partial autcorrelation")
ax.set_xlabel("Lag")
plt.tight_layout()




































