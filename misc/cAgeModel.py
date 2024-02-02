from res.directory_paths import file_480_to_510, file_510_to_540
from res.constants import YD_transition

import numpy as np
import pandas as pd

import warnings
import matplotlib.pyplot as plt


d = 'Corrected Depth (m)'
d_cm = 'Corrected Depth (cm)'
d_mm = 'Corrected Depth (mm)'
a = 'Age (a b2k)'

data_480_to_510 = pd.read_csv(
    file_480_to_510, sep='\t',
    names=[d_mm, a],
    index_col=False)
data_510_to_540 = pd.read_csv(
    file_510_to_540, sep='\t',
    names=[d_mm, a],
    index_col=False)

data_480_to_510[d_mm] += 4_800  # mm
data_510_to_540[d_mm] += 5_100  # mm

# combine
data = pd.concat([data_480_to_510, data_510_to_540])

# add m column
data[d] = data[d_mm] / 1000
# add cm column
data[d_cm] = data[d] * 100

xp = data[d]
yp = data[a]


def depth_to_age(depth: float) -> float:
    """
    Return the corresponding core age (a b2k) for a given depth (m).

    Parameters
    ----------
    depth : float
        depth in core in m.

    Raises
    ------
    ValueError
        If depths in age model are not monotonically increasing.

    Returns
    -------
    float
        interpolated age in a b2k.

    """
    if not np.all(np.diff(xp) > 0):
        warnings.warn('Depths not always strictly increasing!')
    if not np.all(np.diff(xp) >= 0):
        raise ValueError('Depths not always increasing!')
    # lineraly interpolate between values
    return np.interp(depth, xp, yp)


if __name__ == '__main__':
    #     plt.plot(data[d], data[a])
    #     # calculate the slope
    coef = np.polyfit(data[d], data[a], 1)
    poly1d_fn = np.poly1d(coef)
#     plt.plot(data[d], poly1d_fn(data[d]), '--k')
#     plt.xlabel('depth in m')
#     plt.ylabel('age in yrs b2k')
#     plt.show()

    # depth of interest
    d_core = (4.9, 5.20)
    data_core = data.loc[(data[d] >= d_core[0]) & (data[d] <= d_core[1])]

    plt.plot(data_core[d], data_core[a], color='black', label='depth to age model')
    # calculate the slope
    coef_core = np.polyfit(data_core[d], data_core[a], 1)
    poly1d_fn_core = np.poly1d(coef_core)
    # plt.plot(data_core[d], poly1d_fn_core(data_core[d]), '--k')
    # age = coef[0] * depth + coef[1]
    # [yrs] = [yrs / m] * [m] + [yrs]

    plt.xlim(d_core)
    plt.ylim((np.min(data_core[a]), np.max(data_core[a])))

    plt.hlines(
        YD_transition, d_core[0], d_core[1],
        color='red', linestyles=':', label='YD transition'
    )
    slices = np.arange(d_core[0], d_core[1], 5e-2)[1:]
    plt.vlines(
        slices, np.min(data_core[a]), np.max(data_core[a]),
        color='grey', label='section boundaries'
    )

    plt.xlabel(d)
    plt.ylabel(a)
    plt.legend()

    plt.show()

    print('max depth difference (micrometer):', np.diff(data[d]).max() * 1e6)
    print('min depth difference (micrometer):', np.diff(data[d]).min() * 1e6)
    print('sed rate (mm/yr):', 1e3 / coef[0])
    print('max depth difference core (micrometer):',
          np.diff(data_core[d]).max() * 1e6)
    print('min depth difference core (micrometer):',
          np.diff(data_core[d]).min() * 1e6)
    print('sed rate core (mm/yr):', 1e3 / coef_core[0])
