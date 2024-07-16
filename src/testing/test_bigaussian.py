# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:03:58 2024

@author: yanni
"""
import numpy as np
import matplotlib.pyplot as plt

def bigaussian(x: np.ndarray, x_c, H, sigma_l, sigma_r):
    """Evaluate bigaussian for mass vector based on parameters."""
    x_l = x[x <= x_c]
    x_r = x[x > x_c]
    y_l =  H * np.exp(-1/2 * ((x_l - x_c) / sigma_l) ** 2)
    y_r =  H * np.exp(-1/2 * ((x_r - x_c) / sigma_r) ** 2)
    return np.hstack([y_l, y_r])

# %% normaliation such that area stays the same
def H_from_area(area, sigma_l, sigma_r):
    # \int_{-infty}^{infty} H \exp(- (x - x_c)^2 / (2 sigma)^2)dx 
    #   = sqrt(2 pi) H sigma
    # => A = H sqrt(pi / 2) (sigma_l + sigma_r)
    # <=> H = sqrt(2 / pi) * A* 1 / (sigma_l + sigma_r)
    return np.sqrt(2 / np.pi) * area / (sigma_l + sigma_r)

H = 10
alpha = np.sqrt(2)
sigma_l = 5
sigma_r = 1
x_c = 0
x = np.linspace(-100, 100, 10000)
y1 = bigaussian(x, x_c, H, sigma_l, sigma_r)
y_ = bigaussian(x, x_c, alpha, sigma_l, sigma_r)
area_th = H * (sigma_l + sigma_r) / 2 * np.sqrt(2 * np.pi)
area_int = np.trapz(y1 * y_, x=x)
H_rec = H_from_area(area_int, sigma_l, sigma_r)
y_rec = bigaussian(x, x_c, H_rec, sigma_l, sigma_r)
print(area_int, area_th)

plt.plot(x, y1, label='signal')
plt.plot(x, y_, label='kernel')
# plt.plot(x, y_ * y1, label='result')
plt.plot(x, y_rec,'-.', label='reconstructed', color='red')
plt.legend()
plt.show()