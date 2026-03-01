# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:41:07 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv(
    "retail_sales.csv",
    parse_dates=['ds'],
    index_col=['ds']
    )

# Classical decomposition
window_size = 12
df["trend"] = (
    df.rolling(window=window_size)
    .mean()
    .rolling(window=2)
    .mean()
    .shift(-window_size//2)
    )

df["y_detrended"] = df["y"] - df["trend"]

df["month"] = df.index.month
df.head()

seasonality = df.groupby("month").mean()["y_detrended"]
seasonality.name = "seasonality"

fig, ax = plt.subplots(figsize=[12,5])
seasonality.plot(y="seasonality", ax=ax, marker=".")
ax.set_xlabel("Month")
ax.set_ylabel("Seasonality")
ax.set_title("Average of de-trended time series for each month")

df = df.merge(right=seasonality, left_on="month", right_index= True)
df = df.sort_index()

df["residual"] = df["y"] - df["trend"] - df["seasonality"]

fig, ax = plt.subplots(nrows=4, figsize=[12,12], sharex=True)

df["y"].plot(ax=ax[0], legend="y")
ax[0].set_ylabel("y")

df["trend"].plot(ax=ax[1], legend="trend")
ax[1].set_ylabel("trend")

df["seasonality"].plot(ax=ax[2], legend="seasonality")
ax[2].set_ylabel("seasonality")

df["residual"].plot(ax=ax[3], legend="residual")
ax[3].set_ylabel("residual")

# Seasonal decomposition using moving average
res = seasonal_decompose(x=df["y"], model="additive", period=12)
res.trend.head(10)
res.seasonal.head(10)
plt.rc("figure", figsize=(10,10))
plt.rc("font", size=5)
res.plot();















