# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:04:38 2026

@author: Si Thu Aung
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.ModResidDiag import plot_ResidDiag

x = pd.read_csv('gdpquarterlychina1992.1-2017.4.csv', header=0)
dates = pd.date_range(start='1992', periods=len(x), freq='QE')
x.index = dates
x = pd.Series(x['GDP'])

x.plot()

lx = np.log(x)

dDlx = sm.tsa.statespace.tools.diff(lx, k_diff=1, 
                                    k_seasonal_diff=1, 
                                    seasonal_periods=4)

dDlx.plot(marker='o', ms=3)

acf_pacf_fig(dDlx, both=True, lag=44)

sm.tsa.kpss(dDlx, regression='c', nlags='auto')

sarima0200 = sm.tsa.SARIMAX(lx, order=(0, 1, 2), seasonal_order=(0, 1, 0, 4))

sarimaRes0200 = sarima0200.fit()

print(sarimaRes0200.summary())

# d + Ds
# d = 1
# D = 1
# s = 4
# 5

resid0200 = sarimaRes0200.resid[5:]

plot_ResidDiag(resid0200, noestimatedcoef=2, nolags=28, lag=32)


# %%

import warnings
warnings.filterwarnings("ignore")


d = 1 
D = 1 
s = 4

results = []

for p in range(0,3):
    for q in range(0,3):
        for P in range(0,3):
            for Q in range(0,3):
                model = sm.tsa.SARIMAX(lx,
                                       order=(p, d, q),
                                       seasonal_order=(P, D, Q, s),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
                fit = model.fit(disp=False)
                
                results.append({
                    "order": (p, d, q),
                    "seasonal_order": (P, D, Q, s),
                    "aic": fit.aic,
                    "bic": fit.bic,
                    "hqic": fit.hqic
                    })
                
                
results_df = pd.DataFrame(results)

best_aic = results_df.loc[results_df["aic"].idxmin()]
best_bic = results_df.loc[results_df["bic"].idxmin()]
best_hqic = results_df.loc[results_df["hqic"].idxmin()]

print("Best model by AIC", best_aic)
print("Best model by BIC", best_bic)
print("Best model by HQIC", best_hqic)




















