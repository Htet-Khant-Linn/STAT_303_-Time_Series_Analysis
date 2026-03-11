# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:39:07 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from statsmodels.tsa.arima.model import ARIMA
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from scipy import stats

temp = pd.read_csv("Global mean surface air temp changes 1880-1985.csv", header=None)
dates = pd.date_range('1880-12', periods=len(temp), freq='A-DEC')
temp.index = dates
temps = pd.Series(temp[0])

plt.plot(temp, color='b')
plt.show()

dtemp = temp.diff(1)
dtemp = dtemp.dropna()

plt.plot(dtemp, color='b')
plt.show()

acf_pacf_fig(dtemp, both=True, lag=20)

inf = sm.tsa.arma_order_select_ic(dtemp, max_ar=7, max_ma=7, ic=['aic', 'bic', 'hqic'], trend='c')

inf.aic_min_order

inf.bic_min_order

inf.hqic_min_order

arima11 = ARIMA(dtemp, order=(1,0,1), trend='c').fit()
print(arima11.summary())

# arima111 = ARIMA(temps, order=(1,1,1), trend='t').fit()
# print(arima111)

# arima111_n = ARIMA(dtemp, order=(1,1,1), trend='n').fit()
# print(arima111_n.summary())
# resid111_n = arima111_n.resid
# plot_LB_pvalue(resid111_n, noestimatedcoef=1, nolags=20)

resid11 = arima11.resid

plot_LB_pvalue(resid11, noestimatedcoef=1, nolags=20)


# %% Prediction

pred = arima11.predict(start='1960-12', end='1990-12')
predframe = pd.concat([dtemp['1960-12':], pred], axis=1)
predframe.columns = ['y', 'predicted_mean']
predframe.plot()
plt.show()
























