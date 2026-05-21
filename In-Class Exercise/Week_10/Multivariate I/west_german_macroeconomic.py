# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:54:23 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from PythonTsa.plot_multi_ACF import multi_ACFfig
from PythonTsa.plot_multi_Q_pvalue import MultiQpvalue_plot

gEco = pd.read_csv('EconGermany.dat', header=0, sep='\s+')

dates = pd.date_range('1960-03', periods=len(gEco), freq='Q')
gEco.index = dates

gEco = gEco[['inv.', 'inc.', 'cons.']]
gEco.plot()

dlge = np.log(gEco).diff(1).dropna()
dlge.plot()
# plt.savefig()

dlge.tail(4)
dlgem = dlge['1960-06-30':'1981-12-31']
dlgem.tail(4)

multi_ACFfig(dlgem, nlags=10);

dlgemMod = VAR(dlgem)

# FPE - final prediction error - 
# how good our model is at predicting future values.
print(dlgemMod.select_order(maxlags=4))

dlgemRes = dlgemMod.fit(maxlags=2, ic=None, trend='c')
print(dlgemRes.summary())

dlgemRes.is_stable()

resid = dlgemRes.resid
multi_ACFfig(resid, nlags=10);

q, p = MultiQpvalue_plot(resid, p=2, q=0, noestimatedcoef=18, nolags=24)

coefmat = dlgemRes.coefs
coefmat

sigma_u = dlgemRes.sigma_u
sigma_u

dlgem = dlgem.values

type(dlgem)

fore_interval = dlgemRes.forecast_interval(dlgem, steps=4)

point, lower, upper = dlgemRes.forecast_interval(dlgem, steps=4)

point

lower

upper

dlgemRes.plot_forecast(steps=4);

g1 = dlgemRes.test_causality(caused='cons.', causing='inc.',
                             kind='f', signif=0.05)
print(g1)

g2 = dlgemRes.test_causality(caused='inc.', causing='cons.',
                             kind='f', signif=0.05)
print(g2)

irf = dlgemRes.irf(periods=10)
irf.plot();

irf.plot_cum_effects();







