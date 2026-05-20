# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:56:20 2026

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

pgret = pd.read_csv('monthly returns of PG.csv', header=0)

pgret = pgret['RET']
dates = pd.date_range('1961-01', periods=len(pgret), freq='M')
pgret.index = dates

pgret = 100 * pgret

archmodel = arch_model(pgret).fit()
print(archmodel.summary())

archresid = archmodel.std_resid

plot_LB_pvalue(archresid, noestimatedcoef=0, nolags=36)

qqplot(archresid, line='q', fit=True)

# Student T
garch_T = arch_model(pgret, p=1, q=1, dist='StudentsT')
res = garch_T.fit()
print(res.summary())
archresidT = res.std_resid

plot_LB_pvalue(archresidT, noestimatedcoef=0, nolags=36)

qqplot(archresidT, stats.t, line='q', fit=True)







