# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:03:34 2026

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

ar = np.array([1, -0.6])
ma = np.array([1, 0.4])

arma_process = sm.tsa.ArmaProcess(ar, ma)

# Test Stationary
arma_process.isstationary

np.random.seed(0)
y = arma_generate_sample(ar = ar, ma = ma, nsample=500)
y = pd.DataFrame(data={"y":y})
y = y['y']

y.plot()

acf_pacf_fig(y, both=True, lag=20)

# %% Choose p and q for ARMA model
results = []

potential_models = [(1,0), (0,1), (1,1), (2,0)]

for p, q in potential_models:
    model = ARIMA(y, order=(p, 0, q))
    fitted = model.fit()
    aic = fitted.aic
    bic = fitted.bic
    print(f"ARMA({p}, {q}): AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")
    
    results.append({
        'p' :p,
        'q' : q,
        'aic': aic,
        'bic': bic
        })
    
results_df = pd.DataFrame(results)

# %% ARMA(1,1) -> ARIMA(1,0,1)

arma11 = ARIMA(y, order=(1,0,1), trend='c').fit()
print(arma11.summary())

resid11 = arma11.resid

plot_LB_pvalue(resid11, noestimatedcoef=1, nolags=20)

# %% Prediction

pred = arma11.get_prediction(start=450, end=509)
predicts = pred.predicted_mean

predframe = pd.concat([y[450:], predicts], axis=1)
predframe.columns = ['y', 'predicted_mean']

predframe.plot()
plt.show()




















































