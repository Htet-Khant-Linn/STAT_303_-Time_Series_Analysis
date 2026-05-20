# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:57:59 2026

@author: Si Thu Aung
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from arch import arch_model
from statsmodels.graphics.api import qqplot

ret = pd.read_csv('SP500dailyreturns.csv', header=None)
ret.columns = ['returns']
ret = pd.Series(ret['returns'])

ret.plot()
# plt.savefig()

acf_pacf_fig(ret, both=True, lag=40)

plot_LB_pvalue(ret, noestimatedcoef=0, nolags=40)

sm.tsa.kpss(ret, regression='c', nlags='auto')

# ARMA Nodel

arma152 = sm.tsa.SARIMAX(ret, order=(15,0,2), trend='c').fit()
print(arma152.summary())

# L8, L10, L14 = 0
arma122 = sm.tsa.SARIMAX(ret, order= ([1, 1, 1, 1, 1, 1, 1, 0, 1 , 0, 1, 1, 1, 0, 1], 0, 2),
                         trend='c').fit()
print(arma122.summary())

xresid = arma122.resid

acf_pacf_fig(xresid, both=True, lag=30)

plot_LB_pvalue(xresid, noestimatedcoef=14, nolags=30)

acf_pacf_fig(xresid**2, lag=30)

# GARCH Model

garch = arch_model(xresid, p = 2, q=2, mean='Zero')
garchmod = garch.fit(disp='off')
print(garchmod.summary())

garchresid = garchmod.std_resid

acf_pacf_fig(garchresid, both=True, lag=30)

plot_LB_pvalue(garchresid, noestimatedcoef=0, nolags=30)

acf_pacf_fig(garchresid**2, both=True, lag=30)

plot_LB_pvalue(garchresid**2, noestimatedcoef=0, nolags=30)

qqplot(garchresid, line='q', fit=True)
plt.show()





















