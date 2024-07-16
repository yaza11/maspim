import numpy as np


def _find_closest_peak(mz, tol, mzs, intensities, snrs, min_int=10000, min_snr=0):
    """
    find the closest peak to the target m/z value
    :param mz:
    :param tol:
    :param mzs:
    :param intensities:
    :param snrs:
    :param min_int:
    :param min_snr:
    :return:
    """

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


def threshold_otsu(image, nbins=256):
    # Stripped-down function from alexandrovteam/pyImagingMSpec
    assert image.min() != image.max()
    hist, bin_edges = np.histogram(image.flat, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hist = hist.astype(float)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def estimateThreshold(fwhms, mzs):
    coeff = fwhms / mzs ** 2  # fwhm is proportional to mz^2 for FTICR
    thr = threshold_otsu(coeff)  # find a point to split the histogram
    if float((coeff > thr).sum()) / len(coeff) < 0.4:
        thr = 0  # there's nothing to fix, apparently
    return thr



