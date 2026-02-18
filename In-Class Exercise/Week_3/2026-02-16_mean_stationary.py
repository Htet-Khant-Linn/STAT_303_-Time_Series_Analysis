# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 19:42:58 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mean_stationary_data.csv')

df['date'] = pd.to_datetime(df['date'])

# Calculate overall mean
overall_mean = df['customers'].mean()


# Create the plot
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['customers'], color='blue', alpha=0.7, linewidth=0.8)
plt.axhline(y=overall_mean, color='red', linestyle='--', 
            label=f'Overall Mean = {overall_mean:.1f}')
plt.title('Mean Stationary Time Series: Daily Coffee Shop Customers')
plt.xlabel('Date')
plt.ylabel('Number of Customers')
plt.legend()
plt.grid(True, linewidth=2, alpha=0.3)
plt.tight_layout()

