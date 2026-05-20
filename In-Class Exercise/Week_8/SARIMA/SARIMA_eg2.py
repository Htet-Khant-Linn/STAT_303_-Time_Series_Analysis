# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:41:23 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.ModResidDiag import plot_ResidDiag


h02 = pd.read_csv('h02July1991June2008.csv', header = 0)
dates = pd.date_range(start='1991-07', periods=len(h02), freq='M')
h02.index = dates
h02 = pd.Series(h02['h02'])

h02.plot()

Dh02 = sm.tsa.statespace.tools.diff(h02, k_diff=0, 
                                    k_seasonal_diff=1, 
                                    seasonal_periods=12)

Dh02.plot()

acf_pacf_fig(Dh02, both=True, lag=36)

sm.tsa.kpss(Dh02, regression='c', nlags='auto')

sarima0520 = sm.tsa.SARIMAX(h02, order=(0,0,5), 
                            seasonal_order=(2,1,0, 12))

sarimaMod0520 = sarima0520.fit()
print(sarimaMod0520.summary())

# d + Ds
# d = 0
# D = 1
# s = 12
# 12
resid0520 = sarimaMod0520.resid[12:]

plot_ResidDiag(resid0520, noestimatedcoef=7, nolags=48, lag = 36)


sarima0520 = sm.tsa.SARIMAX(h02, order=(0, 0, [1, 1, 1, 0, 1]), 
                            seasonal_order=(2,1,0, 12))

sarimaMod0520 = sarima0520.fit()
print(sarimaMod0520.summary())

resid0520 = sarimaMod0520.resid[12:]
plot_ResidDiag(resid0520, noestimatedcoef=6, nolags=48, lag = 36)




















