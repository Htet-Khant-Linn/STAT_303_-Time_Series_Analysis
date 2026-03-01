# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:52:46 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv(
    "retail_sales.csv",
    parse_dates = ["ds"],
    index_col = ["ds"]
    )


df_feature = df.copy()

# lag = 1

lag = 12

df_feature["y_lag_12"] = df_feature.shift(periods = lag)
df_feature.head()

df_feature = df_feature.dropna()

df_feature.to_csv("time_series_with_lag_feature.csv", index=True)
