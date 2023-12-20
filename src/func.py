import numpy as np


def _find_closest_peak(mz, tol, mzs, intensities, snrs, min_int=10000, min_snr=0):
    mzs = np.array(mzs)
    intensities = np.array(intensities)
    snrs = np.array(snrs)
    intensities = intensities[np.where(np.abs(mzs - mz) < tol)]
    snrs = snrs[np.where(np.abs(mzs - mz) < tol)]
    mzs = mzs[np.where(np.abs(mzs - mz) < tol)]
    if len(mzs) > 0 and max(intensities) > min_int and max(snrs) > min_snr:
        mz = mzs[np.argmax(intensities)]
        snr = snrs[np.argmax(intensities)]
        intensity = intensities[np.argmax(intensities)]
    else:
        mz = np.nan
        intensity = np.nan
        snr = np.nan
    return mz, intensity, snr


def _mz_to_f(mz, A, B):
    """
    convert m/z to frequency
    :param mz: m/z
    :param A: calibration constant 1
    :param B: calibration constant 2
    :return: frequency
    """
    return (A + np.sqrt(A ** 2 + 4 * B * mz)) / (2 * mz)
