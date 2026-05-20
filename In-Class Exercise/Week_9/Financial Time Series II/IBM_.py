# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:40:01 2026

@author: Si Thu Aung
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from arch import arch_model
from statsmodels.graphics.api import qqplot


ibm = pd.read_csv('ibmlogret.csv', header=0)
logret = ibm['logreturn']
logret.index = ibm['date']

logret.plot()

sm.tsa.kpss(logret, regression='c', nlags='auto')

acf_pacf_fig(logret, both=True, lag=36)

plot_LB_pvalue(logret, noestimatedcoef=0, nolags=36)

# ARMA model

inf = sm.tsa.arma_order_select_ic(logret, max_ar=2, max_ma=2,
                                  ic=['aic', 'bic', 'hqic'], trend='c')

inf.aic_min_order

inf.bic_min_order

inf.hqic_min_order

arma = ARIMA(logret, order=(0,0,1), trend='c').fit()
print(arma.summary())

armaresid = arma.resid

plot_LB_pvalue(armaresid, noestimatedcoef=1, nolags=36)

plot_LB_pvalue(armaresid**2, noestimatedcoef=0, nolags=36)


# GARCH Model

garch = arch_model(armaresid, p=1, q=1, mean='Zero').fit(disp='off')
print(garch.summary())

egarch = arch_model(armaresid, p=0, o=1, q=1, mean='Zero', 
                    vol='EGARCH').fit(disp='off')
print(egarch.summary())

egarchresid = egarch.std_resid

plot_LB_pvalue(egarchresid, noestimatedcoef=0, nolags=36)

plot_LB_pvalue(egarchresid**2, noestimatedcoef=0, nolags=36)

qqplot(egarchresid, line='q', fit=True)
plt.show()


















