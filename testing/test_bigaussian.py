# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:03:58 2024

@author: yanni
"""
import numpy as np
import matplotlib.pyplot as plt

def bigaussian(x: np.ndarray, x_c, y0, H, sigma_l, sigma_r):
    """Evaluate bigaussian for mass vector based on parameters."""
    x_l = x[x <= x_c]
    x_r = x[x > x_c]
    y_l =  y0 + H * np.exp(-1/2 * ((x_l - x_c) / sigma_l) ** 2)
    y_r =  y0 + H * np.exp(-1/2 * ((x_r - x_c) / sigma_r) ** 2)
    return np.hstack([y_l, y_r])

x_c = 5
y0=0
H=5
sigma_l=1
sigma_r=2
A = np.sqrt(np.pi / 2) * (sigma_l  + sigma_r)
H = 1 / A

x = np.linspace(0, 15, 1000)

y = bigaussian(x, x_c, y0, H, sigma_l, sigma_r)


A_meas = np.trapz(y, x=x)
print(A_meas)

plt.plot(x, y)
plt.grid('on')
plt.show()