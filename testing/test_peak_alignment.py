import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math import factorial
import numpy as np
import matplotlib.pyplot as plt

# from exporting.from_mcf.rtms_communicator import Spectra

def gaussian(x: np.ndarray, x_c, H, sigma):
    return H * np.exp(-1/2 * ((x - x_c) / sigma) ** 2)


x = np.linspace(-3, 5, 1_000)

sigma1 = 1
sigma2 = 2
g1 = gaussian(x, x_c = 1, H = 1, sigma=sigma1)
# 2 std away
g2 = gaussian(x, x_c = 1 + sigma1 + sigma2, H = 2, sigma=sigma2)
s = g1 + g2

plt.plot(x, g1)
plt.plot(x, g2)
plt.plot(x, s)
plt.show()