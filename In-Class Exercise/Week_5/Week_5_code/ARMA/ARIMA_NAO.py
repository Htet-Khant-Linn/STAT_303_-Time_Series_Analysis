# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:50:05 2026

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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

nao = pd.read_csv("nao.csv", header=0)
timeindex = pd.date_range('1950-01', periods=len(nao), freq='M')
nao.index = timeindex

naots = nao['index']

ax = nao.plot(y='index', figsize=[10,5])
ax.set_title("NAO Index")
ax.set_ylabel("Year")
ax.set_xlabel("Index")

acf_pacf_fig(naots, both=True, lag=20)

# %% Stationary Test

sm.tsa.stattools.kpss(naots, regression='c', nlags=20)

# %% AR1 and MA1

ar1 = ARIMA(naots, order=(1,0,0), trend='c').fit()
print(ar1.summary())

# AIC                           2358.013
# BIC                           2372.181
# HQIC                          2363.446

ma1 = ARIMA(naots, order=(0,0,1), trend='c').fit()
print(ma1.summary())

# AIC                           2360.737
# BIC                           2374.905
# HQIC                          2366.170

resid1 = ar1.resid

plot_LB_pvalue(resid1, noestimatedcoef=1, nolags=20)

# %% Prediction using AR1

pred = ar1.get_prediction(start='2010-04', end='2019-12')
predicts = pred.predicted_mean

predframe = pd.concat([naots['2010-04-30':], predicts], axis=1)
predframe.columns = ['y', 'predicted_mean']

predframe.plot()
plt.show()





















