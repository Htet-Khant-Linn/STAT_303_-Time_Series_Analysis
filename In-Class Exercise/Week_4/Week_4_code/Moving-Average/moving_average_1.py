# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:09:52 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "retail_sales.csv",
    parse_dates=['ds'],
    index_col=['ds']
    )

df.head()

df.plot(y='y', marker='.', figsize=[10, 5])
plt.title("Retail Sales")
plt.ylabel("Sales")
plt.xlabel("Time")

# Moving Average - Order of 3
window_size = 3
ma_3 = df.rolling(
    window = window_size,
    center=True
    ).mean()

ma_3.rename(columns={"y":"3-MA"}, inplace=True)

ma_3.head()

# Plot the result
fig, ax = plt.subplots(figsize=[12,5])

df.plot(ax=ax, marker='.')
ma_3.plot(ax=ax, color='r', alpha=0.75)

ax.set_title("Retail Sales with Moving Average")
ax.set_xlabel("Time")
ax.set_ylabel("Retail Sales")

# Moving Average - Order of 4
df_ = df.copy()
df_["4_ma"] = df_.rolling(window=4).mean()
df_["2x4_ma"] = df_["4_ma"].rolling(window=2).mean()
df_["result"] = df_["2x4_ma"].shift(-2)

# Moving Average - Order of 12 or One year period
window_size_12 = 12 
ma_2_12 = (
    df.rolling(window=window_size_12)
    .mean()
    .rolling(window=2)
    .mean()
    .shift(-window_size_12//2)
    )

ma_2_12.rename(columns={"y": "ma_2_12"}, inplace=True)

fig, ax = plt.subplots(figsize=[12,5])

df.plot(ax=ax, marker='.')
ma_2_12.plot(ax=ax, color='r', alpha=0.75)
ax.set_title("Retail Sales with Moving Average (Window size = 12)")
ax.set_xlabel("Time")
ax.set_ylabel("Retail Sales")

























