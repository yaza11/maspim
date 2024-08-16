"""
a fit transformer function for mass calibration
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from maspim.exporting.from_sqlite.func import _find_closest_peak, _mz_to_f


class Calibration(BaseEstimator, TransformerMixin):
    """
    a fit transformer function for mass calibration, which is used to calibrate the mass of the spectra
    to (a) target mass(es). The usage is the same as sklearn's fit transformer functions.

    The calibration is done by finding the closest peak to the target mass, and then calculate the
    calibration distance by the difference between the target mass and the closest peak, and then
    add the calibration distance to the frequency of the spectrum, and then convert the frequency
    back to m/z.

    the conversion between m/z and frequency is done by the following formula:
    m/z = A / f + B / f^2
    where A and B are constants, and f is the frequency

    the calibration distance is calculated by the following formula:
    cal_dist_f = f_target - f_closest_peak
    where f_target is the frequency of the target mass, and f_closest_peak is the frequency of the
    closest peak to the target mass, and cal_dist_f is the calibration distance in frequency, if more than one target
    mass is provided, the calibration distance is the average of the calibration distances of all target masses (TODO: is this the best way to do it?)

    code example:
    .. code-block:: python
        cal = Calibration(target_mz=1000, A, B, tol=0.01, min_int=10000, min_snr=0)
        cal.fit(mzs, weights=intensities, snrs=snrs)
        mzs = cal.transform(mzs)

    """

    def __init__(self, target_mz: list or float, A: float, B: float, tol=0.01, min_int=10000, min_snr=0):
        """
        :param target_mz: the m/z value to be calibrated against, either a list of m/z values or a single m/z value
        :param A: the A constant in the formula m/z = A / f + B / f^2
        :param B: the B constant in the formula m/z = A / f + B / f^2
        :param tol: tolerance width for finding the closest peak, in Da
        :param min_int: minimum intensity for the closest peak, if the closest peak does not meet the minimum
        intensity, the calibration distance will be set to NaN
        :param min_snr: minimum signal to noise ratio for the closest peak, if the closest peak does not meet the
        minimum signal to noise ratio, the calibration distance will be set to NaN
        """
        self._X = None
        self._fX = None
        self.cal_dist_f = None
        self.cal_dist_mz = None
        self.A = A
        self.B = B
        self.target_mz = target_mz
        self.tol = tol
        self.min_int = min_int
        self.min_snr = min_snr

    def _reset(self):
        """
        reset the calibration distance
        :return:
        """
        self.cal_dist_f = None
        self.cal_dist_mz = None

    def fit(self, X, y=None, weights=None, snrs=None):
        """
        compute the calibration distance for each spectrum for later use
        :param X: the m/z values of the spectra
        :param y:
        :param weights: the intensity weights for each peak
        :param snrs: the signal to noise ratio for each peak
        :return:
        """
        self._reset()
        return self.partial_fit(X, y, weights, snrs)

    def partial_fit(self, X, y=None, weights=None, snrs_weight=None):
        """
        compute the calibration distance for each spectrum for later use
        :param X: the m/z values of the spectra
        :param y:
        :param snrs_weight: the signal to noise ratio for each peak
        :param weights: the intensity weights for each peak
        :return:
        """
        # convert the m/z values to frequency
        self._fX = _mz_to_f(X, self.A, self.B)

        self._X = X.copy()

        if weights is None:
            raise ValueError('weights cannot be None')
        if snrs_weight is None:
            raise ValueError('snrs cannot be None')

        self.cal_dist_f = np.zeros(X.shape[0])
        self.cal_dist_mz = np.zeros(X.shape[0])
        for num in range(X.shape[0]):
            mzs = X[num]
            intensities = weights[num]
            snrs = snrs_weight[num]
            # if target_mz is a list, find the average calibration distance
            if isinstance(self.target_mz, list):
                cal_dist_f = []
                for _target_mz in self.target_mz:
                    mz, _, _ = _find_closest_peak(_target_mz, self.tol, mzs, intensities, snrs, min_int=self.min_int,
                                                  min_snr=self.min_snr)
                    if not np.isnan(mz):
                        cal_dist_f.append(_mz_to_f(_target_mz, self.A, self.B) - _mz_to_f(mz, self.A, self.B))
                self.cal_dist_f[num] = np.mean(cal_dist_f)
            elif isinstance(self.target_mz, float):
                # find the highest peak within the tolerance
                mz, _, _ = _find_closest_peak(self.target_mz, self.tol, mzs, intensities, snrs, min_int=self.min_int,
                                              min_snr=self.min_snr)
                self.cal_dist_mz[num] = self.target_mz - mz
                self.cal_dist_f[num] = _mz_to_f(self.target_mz, self.A, self.B) - _mz_to_f(mz, self.A, self.B)

            self._fX[num] = self._fX[num] + self.cal_dist_f[num]
            self._X[num] = self.A / self._fX[num] + self.B / self._fX[num] ** 2
        return self

    @property
    def calibrated_ratio(self):
        """
        return the percentage of spectra that have been calibrated
        :return: the number of spectra that have been calibrated
        """
        return np.count_nonzero(~np.isnan(self.cal_dist_f)) / self.cal_dist_f.shape[0]

    def transform(self, X, y=None):
        return self._X


if __name__ == "__main__":
    raise NotImplementedError('this file is not meant to be executed')
