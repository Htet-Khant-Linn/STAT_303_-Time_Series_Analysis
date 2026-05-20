# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:38:48 2026

@author: Si Thu Aung
"""

import numpy as np
import pandas as pd
from PythonTsa.True_acf import Tacf_pacf_fig
from PythonTsa.plot_acf_pacf import acf_pacf_fig
import matplotlib.pyplot as plt
import statsmodels.api as sm

ar1 = np.array([1, 0, 0, 0, -0.36])

Tacf_pacf_fig(ar=ar1, ma=[1], both=True, lag=20)
plt.savefig('pure_SAR.png', dpi=300, bbox_inches='tight')

ma1 = np.array([1, 0, 0, 0, 0.46])
Tacf_pacf_fig(ar=[1], ma=ma1, both=True, lag=20)
plt.savefig('pure_SMA.png', dpi=300, bbox_inches='tight')

Tacf_pacf_fig(ar=ar1, ma=ma1, both=True, lag=20)
plt.savefig('pure_SARMA.png', dpi=300, bbox_inches='tight')

#
phi=np.r_[0.2]
theta=np.r_[0.5]
Phi = np.r_[0.3]
Theta = np.r_[0.4]
sigma2 = 4.0
params = np.r_[phi, theta, Phi, Theta, sigma2]

sarma_sim = sm.tsa.SARIMAX([0], order=(1,0,1), seasonal_order=(1,0,1,4)).simulate(params=params, nsimulations=1000)
sarma_sim_series = pd.Series(sarma_sim)

sarma_sim_series.plot()

acf_pacf_fig(sarma_sim_series, both=True, lag=24)
plt.savefig('SARMA_1_1_4.png', dpi=300, bbox_inches='tight')










