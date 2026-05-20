# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:19:09 2026

@author: Si Thu Aung
"""

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

dax = pd.read_csv('DAX.csv', header = 0)
dax.rename(columns = {'Adj Close' : 'index'}, inplace = True)
dax['logreturns'] = np.log(dax['index']/dax['index'].shift(1))
dax = dax.dropna()
logret = dax['logreturns']
logret.index = dax['Date']


fig = plt.figure()
dax['index'].plot(ax = fig.add_subplot(211))
plt.ylabel("Dax daily index")
plt.xticks([])
logret.plot(ax = fig.add_subplot(212))
plt.ylabel("Daily log return")
plt.xticks(rotation=15)
# plt.savefig(args, kwargs)

plot_LB_pvalue(logret, noestimatedcoef=0, nolags=48)





















