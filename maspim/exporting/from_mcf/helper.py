"""
Helper functions for Spectra class.
"""
import numpy as np
import pandas as pd

from typing import Iterable, Callable, Any, Self
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from maspim.util.convenience import check_attr

import logging

logger = logging.getLogger(__name__)


def split_spot(spotname: str) -> tuple[int, int, int]:
    """
    Extract region, x, and y from spot name

    >>> split_spot('R00X121Y085')
    (0, 121, 85)
    """
    r, xy = spotname.lstrip('R').split('X')
    x , y = xy.split('Y')
    return int(r), int(x), int(y)


def get_mzs_for_limits(
        limits: Iterable[float | int],
        delta_mz: float | int
) -> np.ndarray[float]:
    """
    Get vector of equally spaced mz values, guaranteed to include the limits.

    This function rounds the lower limit to the closest lower multiple of delta_mz and
    the upper limit to the next higher multiple of delta_mz. Therefor, the min and max
    values of the returned vector might span a slightly larger interval.

    Parameters
    ----------
    limits : Iterable[float | int]
        The lower and upper limit of the mz values.
    delta_mz : float | int
        Spacing of mz values in the returned vector

    Returns
    -------
    mzs: np.ndarray[float]
        The equally spaced mz values in the interval.
    """
    assert hasattr(limits, "__iter__") and (len(limits) == 2), \
        "Limits must be an iterable with two entries."
    assert isinstance(float(delta_mz), float), "delta_mz must be a number."

    smallest_mz: float = int(limits[0] / delta_mz) * delta_mz
    # round to next biggest multiple of delta_mz
    biggest_mz: float = (int(limits[1] / delta_mz) + 1) * delta_mz
    # equally spaced
    mzs: np.ndarray[float] = np.arange(
        smallest_mz, biggest_mz + delta_mz, delta_mz
    )

    return mzs


def local_max_2D(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return values of local maxima in a 2D matrix along specified axis."""
    assert array.ndim == 2, 'array must be 2D'
    assert axis in (0, 1, -1), 'axis must be 0, 1 or -1.'

    # transpose
    t: bool = axis != 0

    padded = np.pad(array.T if t else array, ((1, 1), (0, 0)), constant_values=np.inf)

    is_local_max = (padded[1:-1, :] > padded[:-2, :]) & (padded[1:-1, :] > padded[2:, :])

    if t:
        return is_local_max * array.T
    return is_local_max * array


def local_max_fast(padded: np.ndarray) -> np.ndarray:
    """Same as local_max_2D but boundary conditions are ignored and can't transpose"""
    is_local_max = (padded[1:-1, :] > padded[:-2, :]) & (padded[1:-1, :] > padded[2:, :])

    return is_local_max * padded[1:-1, :]


class Spots:
    """
    Container for spots.

    Convert rtms spot to python.
    """

    def __init__(self, rspots):
        """
        Initialize the spots object.

        Parameters
        ----------
        rspots : object | Iterable[Iterable[int] | Iterable[str] | Iterable[str]]
            The spots to convert. Either R-object or 3-Tuple-like of indices, names and times.
        """
        self.idxs: np.ndarray[int] = np.array(rspots['index'])
        self.names: np.ndarray[str] = np.array(rspots['SpotNumber'])
        self.times: np.ndarray[str] = np.array(rspots['Timestamp'])


class Spectrum:
    """
    Spectrum with mzs and intensities.

    Adds functionality for converting an R-object to numpy as well as resampling and plotting.
    """

    def __init__(
            self,
            pyrtms_spectrum: np.ndarray[float] | tuple[np.ndarray, np.ndarray],
            limits: Iterable[float] | None = None
    ) -> None:
        """
        Convert rtms spectrum to python.

        Parameters
        ----------
        pyrtms_spectrum : object | Iterable[Iterable[float]]
            The rtms object to convert or a 2-Tuple like object with the mz and intensity values.
        limits : Iterable[float] | None, optional
            mz limits to crop the spectrum as a tuple with upper and lower bound.
            The default is None and will not crop the spectrum.

        Returns
        -------
        None.

        """
        if isinstance(pyrtms_spectrum, np.ndarray):
            self.mzs: np.ndarray[float] = pyrtms_spectrum[:, 0]
            self.intensities: np.ndarray[float] = pyrtms_spectrum[:, 1]
        else:
            self.mzs: np.ndarray[float] = pyrtms_spectrum[0]
            self.intensities: np.ndarray[float] = pyrtms_spectrum[1]
        assert len(self.mzs) == len(self.intensities), \
            (f'Length of mzs and intensities should be the same' +
             f' but are {len(self.mzs)} and {len(self.intensities)}.')
        assert (hasattr(limits, "__iter__") and (len(limits) == 2)) or (limits is None), \
            f"Limits must be an iterable with two entries. You provided {limits}"

        if limits is not None:  # if limits provided, crop spectrum to interval
            mask: np.ndarray[bool] = (self.mzs >= limits[0]) & (self.mzs <= limits[1])
            self.mzs: np.ndarray[float] = self.mzs[mask]
            self.intensities: np.ndarray[float] = self.intensities[mask]

    def plot(
            self,
            *args,
            limits: Iterable[float] | None = None,
            hold: bool = False,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
            **kwargs
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """
        Plots the intensities against m/z values.

        Parameters
        ----------
        args: list[Any]
            Arguments to pass to matplotlib's plot function.
            By default, set's fmt='+-'
        limits: Iterable[float], optional
            mz limits of the plot. None defaults to entire range.
        hold: bool, optional
            Whether to return the fig and ax instead of plotting. The default is False.
        fig: plt.Figure, optional
            Figure object to modify
        ax: plt.Axes, optional
            Axes object to modify
        kwargs: dict[str, Any]
            Optional keyword arguments passed on to matplotlib's plot function.
        """
        if len(args) == 0:
            args = ['+-']
        if limits is None:
            mask = np.ones_like(self.mzs, dtype=bool)
        else:
            assert limits[0] < limits[
                1], 'left bound has to be smaller, e.g. limits=(600, 2000)'
            mask = (self.mzs >= limits[0]) & (self.mzs <= limits[1])
        mzs = self.mzs[mask]
        intensities = self.intensities[mask]
        if fig is None:
            fig, ax = plt.subplots()
        ax.plot(mzs, intensities, *args, **kwargs)
        ax.set_xlabel('m/z in Da')
        ax.set_ylabel('Intensities')

        if hold:
            return fig, ax
        plt.show()

    def resample(
            self,
            delta_mz: float | Iterable[float] = 1e-4,
            check_intervals: bool = False
    ) -> None:
        """
        Resample mzs and intensities to regular intervals.

        Provide either the equally spaced mz values at which to resample or
        the distance for regular intervals resampling.

        Parameters
        ----------
        delta_mz : float | Iterable[float], optional
            Scalar or Array-like. The distances between mz values. If a scalar is provided,
            sample intervals will be equal. The default is to use equal spaced sampling with
            distances of 0.1 mDa
        check_intervals: bool, optional
            In case an array-like object is provided for the delta_mz, setting this option to
            True will check that the spacings are equal. For performance reasons this option is
            disabled by default.

        Notes
        -----
        This function acts in place. If you wish to obtain a copy, use
        >>> spec_new = spec.copy().resample()
        """
        if not hasattr(delta_mz, '__iter__'):  # scalar
            # create mzs spaced apart with specified precision
            # round to next smallest multiple of delta_mz
            smallest_mz = int(self.mzs.min() / delta_mz) * delta_mz
            # round to next biggest multiple of delta_mz
            biggest_mz = (int(self.mzs.max() / delta_mz) + 1) * delta_mz
            # equally spaced
            mzs_ip = np.arange(smallest_mz, biggest_mz + delta_mz, delta_mz)
        else:
            dmzs = np.diff(delta_mz)
            if check_intervals:
                assert np.allclose(dmzs[1:], dmzs[0]), \
                    'passed delta_mz must either be float or list of equally spaced mzs'
            mzs_ip = delta_mz
        # already same mzs, nothing todo
        if (len(self.mzs) == len(mzs_ip)) and np.allclose(self.mzs, mzs_ip):
            return
        # interpolate to regular spaced mz values
        ints_ip = np.interp(mzs_ip, self.mzs, self.intensities)
        # overwrite objects mz vals and intensities
        self.mzs = mzs_ip
        self.intensities = ints_ip

    def to_pandas(self) -> pd.DataFrame:
        """Return mass and intensity as pandas dataframe."""
        df: pd.DataFrame = pd.DataFrame({'mz': self.mzs, 'intensity': self.intensities})
        return df

    def copy(self) -> Self:
        """Return copy of object."""
        rspectrum: tuple[np.ndarray[float], np.ndarray[float]] = \
            [self.mzs.copy(), self.intensities.copy()]
        new_spectrum: Self = Spectrum(rspectrum)
        return new_spectrum


def apply_calibration(
        spectrum: Spectrum,
        poly_coeffs: np.ndarray[float],
        inplace=True
) -> Spectrum:
    """
    Apply the calibration function to a spectrum.

    Parameters
    ----------
    spectrum : np.ndarray[float]
        The intensities of a spectrum. It is assumed that the spectrum has the
        same sampling as the object.
    poly_coeffs : np.ndarray[float]
        Coefficients of the polynomial with the first coefficient corresponding
        to the highest order term.
    inplace : bool, optional
        If True, calibrates the provided spectrum, otherwise returns a new instance.

    Returns
    -------
    spectrum : Spectrum
        The calibrated spectrum.
    """
    f: Callable = np.poly1d(poly_coeffs)

    if not inplace:
        spectrum = spectrum.copy()

    # apply the transformation to the mzs of the spectrum
    spectrum.mzs = spectrum.mzs + f(spectrum.mzs)  # do not use += here since mzs may be referenced, so we want to create a copy here
    return spectrum


class ReaderBaseClass:
    """
    Base class for reader classes.

    Implements shared methods for hdf5Handler and ReadBrukerMCF, such as
    - calibrate_spectrum
    - get_spectrum_resampled_intensities
    """

    mzs: Iterable[float] | None = None

    def get_spectrum(self, *args, **kwargs) -> Any:
        """This method is implemented by the child classes."""
        raise NotImplementedError()

    def get_spectrum_resampled_intensities(
            self,
            *args,
            poly_coeffs: np.ndarray[float] | None = None,
            **kwargs
    ) -> np.ndarray[float]:
        """
        Return the intensities of a resampled (and calibrated) spectrum.

        Parameters
        ----------
        index: int
            The 1-based index of the spectrum.
        poly_coeffs: np.ndarray[float], optional
            Coefficients of the calibration polynomial. The default is None
            and will not apply a calibration.

        Returns
        -------
        spectrum_intensities: np.ndarray[float]
            The intensities of the resampled (and possibly calibrated) spectrum.
        """
        assert check_attr(self, 'mzs'), 'need mz values to resample'

        spectrum: Spectrum = self.get_spectrum(*args, poly_coeffs=poly_coeffs, **kwargs)
        spectrum.resample(self.mzs)
        return spectrum.intensities


def find_polycalibration_spectrum(
        mzs: np.ndarray[float],
        intensities: np.ndarray[float],
        calibrants_mz: Iterable[float],
        search_range: float,
        calib_snr_threshold: float,
        noise_level: np.ndarray,
        min_height: float,
        nearest: bool,
        max_degree: int
) -> tuple[np.ndarray[float], int, np.ndarray[bool]]:
    """Find the calibration function for a single spectrum."""
    # pick peaks
    if calib_snr_threshold > 0:  # only set peaks above the SNR threshold
        peaks: np.ndarray[int] = find_peaks(
            intensities, height=noise_level * calib_snr_threshold
        )[0]
    else:
        peaks: np.ndarray[int] = find_peaks(intensities, height=min_height)[0]
    peaks_mzs: np.ndarray[float] = mzs[peaks]
    peaks_intensities: np.ndarray[float] = intensities[peaks]

    # find valid peaks for each calibrant
    closest_peak_mzs: list[float] = []
    closest_calibrant_mzs: list[float] = []
    calibrator_presences = np.ones(len(calibrants_mz), dtype=bool)
    for jt, calibrant in enumerate(calibrants_mz):
        distances: np.ndarray[float] = np.abs(calibrant - peaks_mzs)  # theory - actual
        if not np.any(distances < search_range):  # no peak with required SNR found inside range
            calibrator_presences[jt] = False
            continue
        # select the highest peak within the search_range
        if nearest:
            closest_peak_mzs.append(peaks_mzs[np.argmin(distances)])
        else:
            peaks_mzs_within_range: np.ndarray[float] = peaks_mzs[distances < search_range]
            peaks_intensities_within_range: np.ndarray[float] = peaks_intensities[distances < search_range]
            closest_peak_mzs.append(peaks_mzs_within_range[np.argmax(peaks_intensities_within_range)])
        closest_calibrant_mzs.append(calibrant)

    # search the coefficients of the polynomial
    # need degree + 1 points for nth degree fit
    n_calibrants = len(closest_peak_mzs)
    degree: int = min([max_degree, n_calibrants - 1])
    if degree < 0:  # no calibrants found
        return np.array([0]), 0, calibrator_presences

    # forbid degree>=2 if the calibrants are far away from the
    # beginning and end of the spectrum, set 5Da for now.
    if degree > 1:
        assert abs(min(calibrants_mz) - min(peaks_mzs)) <= 5, \
            'calibrants are too far away from the beginning of the spectrum'
        assert abs(max(calibrants_mz) - max(peaks_mzs)) <= 5, \
            'calibrants are too far away from the end of the spectrum'

    # polynomial coefficients
    # theory - actual
    yvals = [t - a for t, a in zip(closest_calibrant_mzs, closest_peak_mzs)]
    p: np.ndarray[float] = np.polyfit(x=closest_peak_mzs, y=yvals, deg=degree)
    n_coeffs: int = degree + 1  # number of coefficients in polynomial
    # fill coeff matrix
    return p, n_coeffs, calibrator_presences
