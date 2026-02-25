# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 23:34:32 2026

@author: Si Thu Aung
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf

num_timesteps = 1000  # Length of time series we want
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

# ACF Plot using Library
plot_pacf(
    x=df_["y"],
    lags=50,
    alpha=0.05,
    auto_ylims=True
);
plt.title("Parial autocorrelation of white noise")
plt.ylabel("Partial autcorrelation")
plt.xlabel("Lag")


# Load retail sales dataset with the artificially added outliers
df = pd.read_csv(
    "retail_sales.csv",
    parse_dates=["ds"],
    index_col=["ds"],
)

df.plot(y="y", marker=".", figsize=[10, 5])
plt.title("Retail sales")
plt.ylabel("Sales")
plt.xlabel("Time")

# before
plot_pacf(
    x=df["y"],
    method='ywmle', # Recommended method in Statsmodels notes
    lags=36,
    alpha=0.05,
    auto_ylims=True
)
plt.title("Partial autocorrelation of retail sales")
plt.ylabel("Partial autcorrelation")
plt.xlabel("Lag")
plt.tight_layout()

