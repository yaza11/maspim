"""
a fit transformer function for mass calibration
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.func import _find_closest_peak, _mz_to_f
from src.parser import parse_sqlite


class Calibration(BaseEstimator, TransformerMixin):
    def __init__(self, target_mz: list or float, A: float, B: float, tol=0.01, min_int=10000, min_snr=0):
        """
        m/z = A/f + B/f^2
        :param target_mz: the m/z value to be calibrated against
        :param tol: tolerance width for finding the closest peak
        :param min_int: minimum intensity for the closest peak
        """
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
        :param X:
        :param y:
        :param weights: the intensity weights for each peak
        :return:
        """
        self._reset()
        return self.partial_fit(X, y, weights, snrs)

    def partial_fit(self, X, y=None, weights=None, snrs_weight=None):
        """
        compute the calibration distance for each spectrum for later use
        :param X:
        :param y:
        :return:
        """
        # conver the m/z values to frequency
        self._fX = _mz_to_f(X, self.A, self.B)
        self._X = X.copy()

        if weights is None:
            raise ValueError('weights cannot be None')
        self.cal_dist_f = np.zeros(X.shape[0])
        self.cal_dist_mz = np.zeros(X.shape[0])
        for num in range(X.shape[0]):
            if np.floor(num / 1000) == num / 1000:
                print(num, ',', end="\r", flush=True)
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

    def transform(self, X, y=None):
        return self._X


if __name__ == "__main__":
    raise NotImplementedError('this file is not meant to be executed')