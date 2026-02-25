# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 20:23:30 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('stationary_vs_nonstationary_data.csv')

df['date'] = pd.to_datetime(df['date'])

rolling_mean = df['non_stationary'].rolling(30).mean()

# Create the plot
plt.figure(figsize=(14, 5))

# Stationary
plt.subplot(1,2,1)
plt.plot(df['date'], df['stationary'], alpha=0.7)
plt.axhline(y=df['stationary'].mean(), color='red', linestyle='--')
plt.title('Weakly Stationary\n(Constant Mean & Variance)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

# Non-stationary
plt.subplot(1,2,2)
plt.plot(df['date'], df['non_stationary'], alpha=0.7, color='orange')
plt.plot(df['date'], rolling_mean, color='red', linestyle='--', linewidth=2)
plt.title('Non-Stationary\n(Changing Mean & Variance)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()