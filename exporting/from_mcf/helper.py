"""
Helper functions for Spectra class.
"""
import os
import platform

import numpy as np
import pandas as pd

from subprocess import Popen, PIPE
from typing import Iterable, Callable, Any, Self

from matplotlib import pyplot as plt


def get_r_home():
    """Find the folder of R installation on the system."""
    os_name = platform.system()
    r_home = None

    if os_name == 'Windows':
        r_base_path = r"C:\Program Files\R"
        if os.path.exists(r_base_path):
            versions = sorted([d for d in os.listdir(r_base_path) if os.path.isdir(os.path.join(r_base_path, d))],
                              reverse=True)
            if versions:
                r_home = os.path.join(r_base_path, versions[0])
    elif os_name == 'Linux':
        if os.path.exists('/usr/local/lib/R'):
            r_home = '/usr/local/lib/R'
        elif os.path.exists('/usr/lib/R'):
            r_home = '/usr/lib/R'
    elif os_name == 'Darwin':  # macOS
        r_home = '/Library/Frameworks/R.framework/Resources'

    # Adjusting subprocess call to handle the R workspace image directive
    if not r_home or not os.path.exists(r_home):
        try:
            process = Popen(['R', 'RHOME'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            r_home_detected = stdout.decode('utf-8').split("\n")[0].strip()
            if r_home_detected and os.path.exists(r_home_detected):
                r_home = r_home_detected
        except Exception as e:
            print(f"An error occurred while detecting R_HOME: {str(e)}")

    if r_home and os.path.exists(r_home):
        return r_home
    else:
        raise EnvironmentError("R_HOME could not be determined. Please set it manually or ensure R is installed.")


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
        self.idxs: np.ndarray[int] = np.array(rspots[0])
        self.names: np.ndarray[str] = np.array(rspots[1])
        self.times: np.ndarray[str] = np.array(rspots[2])


class Spectrum:
    """
    Spectrum with mzs and intensities.

    Adds functionality for converting an R-object to numpy as well as resampling and plotting.
    """
    def __init__(
            self,
            rspectrum, limits: Iterable[float] | None = None
    ) -> None:
        """
        Convert rtms spectrum to python.

        Parameters
        ----------
        rspectrum : object | Iterable[Iterable[float]]
            The rtms object to convert or a 2-Tuple like object with the mz and intensity values.
        limits : Iterable[float] | None, optional
            mz limits to crop the spectrum as a tuple with upper and lower bound.
            The default is None and will not crop the spectrum.

        Returns
        -------
        None.

        """
        self.mzs: np.ndarray[float] = np.array(rspectrum[0]).astype(float)
        self.intensities: np.ndarray[float] = np.array(rspectrum[1]).astype(float)
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
        rspectrum: tuple[np.ndarray[float], np.ndarray[float]] =\
            [self.mzs.copy(), self.intensities.copy()]
        new_spectrum: Self = Spectrum(rspectrum)
        return new_spectrum


class ReaderBaseClass:
    """
    Base class for reader classes.

    Implements shared methods for hdf5Handler and ReadBrukerMCF, such as
    - calibrate_spectrum
    - get_spectrum_resampled_intensities
    """
    @staticmethod
    def calibrate_spectrum(
            spectrum: Spectrum,
            poly_coeffs: np.ndarray[float],
            inplace=True
    ) -> Spectrum:
        """
        Apply the calibration function to a spectrum.

        Parameters
        ----------
        spectrum : np.ndarray[float]
            The intensities of a spectrum. It is assumed that the spectrum has the same sampling as
            the object.
        poly_coeffs : np.ndarray[float]
            Coefficients of the polynomial with the first coefficient corresponding to the highest order term.
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
        spectrum.mzs += f(spectrum.mzs)
        return spectrum

    def get_spectrum(self, *args, **kwargs) -> Any:
        """This method is implemented by the child classes."""
        raise NotImplementedError()

    def get_spectrum_resampled_intensities(
            self,
            index: int,
            poly_coeffs: np.ndarray[float] | None = None
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
        spectrum: Spectrum = self.get_spectrum(index, poly_coeffs=poly_coeffs)
        spectrum.resample(self.mzs)
        return spectrum.intensities

