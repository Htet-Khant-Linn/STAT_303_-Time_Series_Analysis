# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:51:14 2026

@author: Si Thu Aung
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import month_plot
from PythonTsa.plot_acf_pacf import acf_pacf_fig


x = pd.read_csv('USFemalesAged20+Job1948-81.csv', header=None)
dates = pd.date_range(start='1948-01', periods=len(x), freq='ME')
x.index = dates
x = pd.Series(x[0])

month_plot(x)
plt.savefig('Seasonal Plot.png', dpi=300, bbox_inches='tight')

plt.figure()
x.plot()
plt.savefig('Time Serie Plot.png', dpi=300, bbox_inches='tight')

plt.figure()
x['1961-01':'1965-12'].plot(marker='.')
plt.savefig('Time Serie Plot(1961-1965).png', dpi=300, bbox_inches='tight')

dDx = sm.tsa.statespace.tools.diff(x, k_diff=1, k_seasonal_diff=1, seasonal_periods=12)

dDx.plot()
plt.savefig('Time Serie Plot(After differencing).png', dpi=300, bbox_inches='tight')

acf_pacf_fig(dDx, both=True, lag=36)
plt.savefig('ACF and PACF(After differencing).png', dpi=300, bbox_inches='tight')















