"""
Following A Practical Guide to Wavelet Analysis.

Notation:
    x_n: time series, n = 0, ..., N - 1
    delta_t: time step of time series, time series must be evenly spaced
    N: number of points in time series
    psi0: wavel function, depends on
    eta: nondimensional time
    omega0: nondimensional frequency




"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from timeSeries.cProxy import UK37

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt


def psi_0(eta, omega0):
    """
    An example is the Morlet wavelet, consisting of a plane wave modulated 
    by a Gaussian
    """
    const = np.pi ** (-1 / 4)
    osz = np.exp(1j * omega0 * eta)
    dec = np.exp(- eta ** 2 / 2)
    return const * osz * dec


def psi(N, dt, s, omega0):
    # normalixation constant such that psi has unit energy
    const = (dt / s) ** 2 
    eta = (np.arange(N) - N / 2) * dt / s
    return const * psi_0(eta, omega0)



def cmor_wiki(t, sigma):
    c_sigma = 1 / np.sqrt(1 + np.exp(-sigma ** 2) - 2 * np.exp(-3 / 4 * sigma ** 2))
    kappa_sigma = np.exp(sigma ** 2)
    
    const = c_sigma * np.pi ** (-1 / 4)
    dec = np.exp(- t ** 2 / 2)
    osz = np.exp(1j * sigma * t)
    return const * dec * (osz - kappa_sigma)


def cmor_pywt(t, bandwidth, center_frequency):
    const = 1 / np.sqrt(np.pi * bandwidth)
    osz = np.exp(2j * np.pi * center_frequency * t)
    dec = np.exp(-t ** 2 / bandwidth)
    return const * osz * dec


def W(s, x_n, psi, dt, omega):
    """
    continuous wavelet transform of a discrete sequence x_n
    is defined as the convolution of x_n with a
    scaled and translated version of psi_0(eta)
    
    n: point
    s: scale
    x_n: time series
    psi: wavelet function
    dt: sample interval
    """
    psi_values = np.conj(psi(len(x_n), dt, s, omega))
    print(s, psi_values.shape, x_n.shape)
    
    return np.convolve(x_n, psi_values, 'same')
    


def calc_mtrx(x_n, s_vec, psi, dt, omega):
    N = len(x_n)
    mtrx = np.zeros((len(s_vec), N), dtype=complex)
    
    for i, s in enumerate(s_vec):
        mtrx[i, :] = W(s, x_n, psi, dt, omega)
            
    return mtrx


# ts = UK37(path_file=r'C:/Users/Yannick Zander/Promotion/Cariaco 2024/Long Time Series 490-510/UK_490-510.pickle')

# # resample to even intervals
# target = 'ratio'
# sample_interval = .5  # yrs

# mask = (ts.feature_table.seed != 0) & (~ts.feature_table[target].isna())
# df: pd.DataFrame = ts.feature_table.loc[mask, [target, 'age']].copy()
# df['age_rounded'] = sample_interval * np.around(df.age / sample_interval)
# # average duplicates
# df = df.groupby('age', as_index=False).mean()
# # add missing values
# ages: np.ndarray[float] = np.arange(
#     df.age_rounded.min(),
#     df.age_rounded.max() + sample_interval,
#     sample_interval
# )
# ages_center: float = (ages.max() - ages.min()) / 2 + ages.min()

# intensities: np.ndarray[float] = np.interp(ages, df['age_rounded'], df[target])

# #%% 
# plt.figure()
# plt.scatter(ts.feature_table.age, ts.feature_table[target])
# plt.plot(df.age, df.ratio, label='uk proxy')
# plt.plot(ages, intensities, label='resampled')
# plt.legend()
# plt.show()

N = 64
ages = np.arange(N)

intensities = np.hstack([np.sin(ages[: N // 2] * 2 * np.pi / (N / 4)), np.sin(ages[N // 2:] * 2 * np.pi / (N / 8))])

plt.figure()
plt.plot(ages, intensities)
plt.show()

s_vec = [2 ** i for i in range(8)]
dt = 1
omega = 2 * np.pi

mtrx = calc_mtrx(x_n=intensities, s_vec=s_vec, psi=psi, dt=dt, omega=omega)
mtrx_py, freqs = pywt.cwt(data=intensities, scales=s_vec, wavelet='cmor2-1', sampling_period=dt)

plt.figure()
plt.pcolormesh(ages, s_vec, np.abs(mtrx))
plt.show()

# %%
plt.figure()
plt.pcolormesh(ages, s_vec, np.abs(mtrx_py))
plt.show()

# %%
# s = 1
# plt.plot(ages, np.imag(psi(N, dt, s, omega)))
# plt.plot(ages, np.imag(cmor_pywt(np.arange(N) - N / 2, 2, 1)))

