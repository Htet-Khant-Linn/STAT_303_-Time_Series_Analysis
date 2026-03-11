# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:23:08 2026

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


ar = np.array([1, -0.8, 0.6])
ma = np.array([1, 0.7, 0.4])

arrima_process = sm.tsa.ArmaProcess(ar, ma)

# Check Stationary 
arrima_process.isstationary

# Check Invertibility
arrima_process.isinvertible


np.random.seed(0)

y = arma_generate_sample(ar=ar, ma=ma, nsample=500)
y = pd.DataFrame(data={"y":y})
y = y['y']

y.plot()

acf_pacf_fig(y, both=True, lag=20)

results = []
potential_models = [(1,0), (0,1), (1,1), (2,0), (0,2), (2,2)]
for p, q in potential_models:
    model = ARIMA(y, order=(p, 0, q))
    fitted = model.fit()
    aic = fitted.aic
    bic = fitted.bic
    
    print(f"ARMA({p}, {q}): AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")
    
    
arma22 = ARIMA(y, order=(2,0,2), trend='c').fit()
print(arma22.summary())

resid22 = arma22.resid

acf_pacf_fig(resid22, both=True, lag=20)
plt.show()

plot_LB_pvalue(resid22, noestimatedcoef=1, nolags=20)
plt.show()

stats.normaltest(resid22)

# %% Prediction

pred = arma22.get_prediction(start=450, end=509)
predicts = pred.predicted_mean

predframe = pd.concat([y[450:], predicts], axis=1)
predframe.columns = ['y', 'predicted_mean']

predframe.plot()
plt.show()






















