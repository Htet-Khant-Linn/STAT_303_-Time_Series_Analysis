# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:47:36 2026

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

sim_mod = arch_model(None, p=1, q=1)
params = [0, 0.1, 0.2, 0.6]
sim_data = sim_mod.simulate(params, nobs=1000)

simdata = sim_data['data']
simdata.plot()

acf_pacf_fig(simdata, both=False, lag=36)

plot_LB_pvalue(simdata, noestimatedcoef=0, nolags=36)

acf_pacf_fig(simdata**2, both=False, lag=36)

fig = plt.figure()
ax = fig.add_subplot(111)
hfig = ax.hist(simdata, bins=40, density=True, label='Histogram')
kde = sm.nonparametric.KDEUnivariate(simdata)
kde.fit()
ax.plot(kde.support, kde.density, label='KDE')
smean = np.mean(simdata)
scal = np.std(simdata, ddof=1)
normden = norm.pdf(kde.support, loc=smean, scale=scal)
ax.plot(kde.support, normden, label='Normal density')
ax.legend(loc='best')

