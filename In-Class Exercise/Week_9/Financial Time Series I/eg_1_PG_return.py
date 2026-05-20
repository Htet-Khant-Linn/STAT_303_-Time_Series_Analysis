# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:57:29 2026

@author: Si Thu Aung
"""

# pip install arch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PythonTsa.plot_acf_pacf import acf_pacf_fig
import statsmodels.api as sm
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from scipy.stats import norm
from arch import arch_model
from statsmodels.graphics.api import qqplot
from scipy import stats

pgret = pd.read_csv('monthly returns of PG.csv', header=0)

pgret = pgret['RET']
dates = pd.date_range('1961-01', periods=len(pgret), freq='M')
pgret.index = dates

pgret = 100 * pgret

sm.tsa.kpss(pgret, regression='c', nlags='auto')

plot_LB_pvalue(pgret, noestimatedcoef=0, nolags=36)
# plt.savefig(args, kwargs)

acf_pacf_fig(pgret**2, lag=48)

plot_LB_pvalue(pgret**2, noestimatedcoef=0, nolags=36)


fig = plt.figure()
ax = fig.add_subplot(111)
hfig = ax.hist(pgret, bins=40, density=True, label='Histogram')
kde = sm.nonparametric.KDEUnivariate(pgret)
kde.fit()
ax.plot(kde.support, kde.density, label='KDE')
smean = np.mean(pgret)
scal = np.std(pgret, ddof=1)
normden = norm.pdf(kde.support, loc=smean, scale=scal)
ax.plot(kde.support, normden, label='Normal density')
ax.legend(loc='best')

# plt.savefig(args, kwargs)


















