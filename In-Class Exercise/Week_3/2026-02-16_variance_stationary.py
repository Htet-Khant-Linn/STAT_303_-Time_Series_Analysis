# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 20:12:31 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('variance_stationary_data.csv')

df['date'] = pd.to_datetime(df['date'])

# Calculate overall mean
mean_temp = df['temperature'].mean()
variance_temp = df['temperature'].var()


# Create the plot
plt.figure(figsize=(12, 5))

# Plot temperature
plt.plot(df['date'], df['temperature'], alpha=0.7, linewidth=0.8, label='Temperature')
plt.axhline(y=mean_temp, color='red', linestyle='--', 
            label=f'Mean = {mean_temp:.1f}°C')
std_dev = np.sqrt(variance_temp)
plt.fill_between(df['date'], 
                 mean_temp - std_dev,
                 mean_temp + std_dev,
                 alpha=0.2, color='gray', label='±1 Std Dev')

plt.title('Variance Stationary Time Series: Indoor Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
