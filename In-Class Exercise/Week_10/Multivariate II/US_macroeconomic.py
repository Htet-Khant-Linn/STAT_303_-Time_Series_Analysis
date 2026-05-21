# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:31:50 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VARMAX
import statsmodels.api as sm
from PythonTsa.plot_multi_ACF import multi_ACFfig
from PythonTsa.plot_multi_Q_pvalue import MultiQpvalue_plot

mdata = sm.datasets.macrodata.load_pandas().data

mdata = mdata[['realgdp', 'realcons', 'realinv']]

dates = pd.date_range('1959-01', periods=len(mdata), freq='Q')

mdata.index = dates

mdata.plot()

dldata = np.log(mdata).diff(1).dropna()

fig = plt.figure()
dldata['realgdp'].plot(ax=fig.add_subplot(311))
plt.title("Differenced log of real GDP")

dldata['realcons'].plot(ax=fig.add_subplot(312))
plt.title("Differenced log of real consume")

dldata['realinv'].plot(ax=fig.add_subplot(313))
plt.title("Differenced log of real invest")
plt.tight_layout()
plt.show()

dldata.tail(3)

myd = dldata['1959-06-30':'2008-12-31']

myd.tail(3)

multi_ACFfig(myd, nlags=10)
plt.show()

myd_mod = VAR(myd)
print(myd_mod.select_order(maxlags=11))

# Select p=3 for VAR(3) model - VAR(1)
myd_VAR3 = VARMAX(myd, order=(3,0), enforce_stationarity=True)
modfit_VAR3 = myd_VAR3.fit()
print(modfit_VAR3.summary())

param = myd_VAR3.param_names

resid = modfit_VAR3.resid
multi_ACFfig(resid, nlags=10)
plt.show()

qs, pv = MultiQpvalue_plot(resid, p=3, q=0, noestimatedcoef=27, nolags=24)
plt.show()


param = modfit_VAR3.param_names
mydmodf = VARMAX(myd, order=(3,0), enforce_stationarity=False)

with mydmodf.fix_params({param[0]:0.0, param[5]:0.0,
                          param[6]:0.0, param[8]:0.0,
                          param[9]:0.0, param[11]:0.0,
                          param[12]:0.0, param[14]:0.0,
                          param[15]:0.0, param[17]:0.0,
                          param[24]:0.0, param[26]:0.0,
                          param[27]:0.0, param[28]:0.0,
                          param[29]:0.0}): modff = mydmodf.fit(method='bfgs')

print(modff.summary())

residf = modff.resid
multi_ACFfig(residf, nlags=10)
plt.show()

qs, pv = MultiQpvalue_plot(residf, p = 3, q = 0, noestimatedcoef=13, nolags=24)
plt.show()

fore = modff.predict(end='2009-09-30')
realgdpFitgdp = pd.DataFrame({'realgdp':dldata['realgdp'], 'fittedgdp':fore['realgdp']})
realconsFitcons = pd.DataFrame({'realcons':dldata['realgdp'], 'fittedcons':fore['realcons']})
realinvFitinv = pd.DataFrame({'realinv':dldata['realinv'], 'fittedinv':fore['realinv']})

fig = plt.figure()
realgdpFitgdp.plot(style=['-', '--'], ax=fig.add_subplot(311))
realconsFitcons.plot(style=['-', '--'], ax=fig.add_subplot(312))
realinvFitinv.plot(style=['-', '--'], ax=fig.add_subplot(313))
plt.tight_layout()
plt.show()




















