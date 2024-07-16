# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def xcorr(a, b, maxlags=1):
    """
    Calculate crosscorrelation for a and b within a certain lag window.

    Parameters
    ----------
    a : Iterable
        DESCRIPTION.
    b : Iterable
        DESCRIPTION.
    maxlags : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    lags : TYPE
        DESCRIPTION.
    corrs : TYPE
        DESCRIPTION.

    """
    assert len(a) == len(b), 'inputs must have same length'
    N = len(b)

    b = np.pad(b.copy(), maxlags)    
    lags = np.arange(-maxlags, maxlags+1, dtype=int)
    corrs = np.array([
        np.correlate(
            a, 
            b[maxlags + lag:maxlags + lag + N], 
            mode='valid'
        )[0] for lag in lags
    ])
    return lags, corrs

def find_offset(a, b, mz, maxdifference: float):
    assert len(np.unique(np.diff(mz))) == 1, \
        'masses must be equally spaced, interpolate if necessary'
    # convert difference to lag
    dmz = mz[1] - mz[0]
    maxlags = int(maxdifference / dmz) + 1
    lags, corrs = xcorr(a, b, maxlags)
    idx = np.argmax(corrs)
    lag = lags[idx]
    offset = dmz * lag
    return offset

maxlags = 10

a = np.array([0, 0, 1, 2, 3, 2, 1, 0])
b = np.array([0, 1, 2, 3, 2, 1, 0, 0])
mz = np.arange(0, len(a))

lags, corrs = xcorr(a, b, maxlags=maxlags)
plt.plot(lags, corrs)

offset = find_offset(a, b, mz, maxdifference=3)
print(f'b lags a by {offset}')
