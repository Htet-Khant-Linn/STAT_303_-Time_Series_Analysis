# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:29:46 2026

@author: Si Thu Aung
"""

import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

window_size = 3

X_seq = []
y_seq = []

for i in range(len(X) - window_size):
    X_seq.append(X[i: i+window_size])
    y_seq.append(X[i+window_size])

X_seq_arr = np.array(X_seq)
y_seq_arr = np.array(y_seq)

X_seq_slide = np.lib.stride_tricks.sliding_window_view(X, window_shape=window_size)
y_seq_slide = X[window_size:]
    