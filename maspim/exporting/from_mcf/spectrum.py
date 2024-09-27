"""
This module implements the Spectra class which is the intersection
between disk-files for MSI measurements and a feature table.

In this module is also a class, that allows to handle multiple depth sections at the same time.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import psutil
import logging

import scipy.signal
from tqdm import tqdm
from typing import Iterable, Self, Any, Callable
from scipy.signal import find_peaks, correlate, correlation_lags, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter, median_filter

from maspim.data.combine_feature_tables import combine_feature_tables
from maspim.exporting.from_mcf.rtms_communicator import ReadBrukerMCF, Spectrum
from maspim.exporting.sqlite_mcf_communicator.hdf import hdf5Handler
from maspim.exporting.sqlite_mcf_communicator.sql_to_mcf import get_sql_files
from maspim.exporting.from_mcf.helper import get_mzs_for_limits, local_max_2D, local_max_fast
from maspim.res.calibrants import get_calibrants
from maspim.util import Convenience
from maspim.util.convenience import check_attr
from maspim.project.file_helpers import ImagingInfoXML, get_rxy, get_spots

logger = logging.getLogger(__name__)


def gaussian(x: np.ndarray, x_c: float, H: float, sigma: float) -> np.ndarray:
    """
    Gaussian kernel function.

    Parameters
    ----------
    x : np.ndarray
        The values at which to evaluate the function
    x_c : np.ndarray
        Center of the kernel in same units as x.
    H : int | float
        Amplitude of the kernel.
    sigma : int | float
        Standard deviation defining the width of the kernel.

    Returns
    -------
    y: np.ndarray
        The function values at the provided x values.
    """
    return H * np.exp(-1 / 2 * ((x - x_c) / sigma) ** 2)


def bigaussian(
        x: np.ndarray, x_c: float, H: float, sigma_l: float, sigma_r: float
) -> np.ndarray:
    """
    Evaluate bigaussian for mass vector based on parameters.

    Parameters
    ----------
    x : np.ndarray
        mass vector.
    x_c : float
        mass at center of peak
    H : float
        Amplitude of peak.
    sigma_l : float
        left-side standard deviation.
    sigma_r : float
        right-side standard deviation.

    Returns
    -------
    y: np.ndarray
        Intensities of bigaussian at given x.

    """
    x_l = x[x <= x_c]
    x_r = x[x > x_c]
    y_l = H * np.exp(-1 / 2 * ((x_l - x_c) / sigma_l) ** 2)
    y_r = H * np.exp(-1 / 2 * ((x_r - x_c) / sigma_r) ** 2)
    return np.hstack([y_l, y_r])


class Spectra(Convenience):
    """
    Container for multiple Spectrum objects and binning.

    This class is used to access Bruker .d folders with MSI data through the R rtms package.
    It is recommended to create an ReadBrukerMCF reader once, read all the spectra in and store
    a hdf5handler, as this is much faster, if the data is read in again, which is usually required.

    Example Usage
    -------------
    >>> from maspim import ReadBrukerMCF
    >>> from maspim import Spectra
    Create a reader
    >>> reader = ReadBrukerMCF('path/to/your/d_folder.d')
    Providing the reader is only necessary the first time an object is instanciated for a folder,
    afterward a reader will be initiated from the stored hdf5 file automatically, when used through
    the get_project class (recommended). This guide only discusses steps when used on its own. For
    more info, take a look at the hdf5Handler and get_project classes.
    >>> spec = Spectra(reader=reader)
    In that case initialization looks like this:
    >>> spec = Spectra(path='path/to/your/d_folder.d', initiate=False)
    It is possible to include only certain spectra by specifying the indices parameter, setting custom
    mass limits or changing the mass interval with which spectra are stored in the hdf5 file, but
    usually this is not necassary.

    The next step is to add all spectra together (this requires reading every spectrum
    in and may take a while):
    >>> spec.add_all_spectra(reader=reader)
    It may be desirable to align spectra before adding them up. This functionality is
    provided
    >>> spec.add_all_spectra_aligned(reader=reader)
    although it was found not necessary for the data tested on and takes longer.

    The result can be viewed using the plotting function
    >>> spec.plot_summed()

    The next step is to subtract the baseline. Here a minimum filter is used. It is crucial to
    find the right window size: If it is too small, peaks will lose intensity because the minimum
    filter climbs up the flanks of peaks. If it is too large, the noise level will not be removed
    enterily. By default, the window_size is being estimated from the data by taking the broadest peak
    at a relative height of .8 as the window size. The result can be checked by setting the plts
    keyword to True.
    >>> spec.subtract_baseline(plts=True)
    
    Optionally, it is possible to perform lock mass calibration, e.g. with Pyropheophorbide a:
    >>> spec.require_calibration_functions(calibrants_mz=[557.25231],reader=reader,calib_snr_threshold=2)
    >>> spec.add_all_spectra(reader=reader)
    Although not necessary, it is adviced to do this step after subtracting the
    baseline in order to have access to the SNR level. Afterward the calibration will be used
    automatically for all steps.

    Then, peaks are searched in the summed spectrum. By default, peaks with a prominence above
    0.1 (relative to median intensity) are considered.
    >>> spec.set_peaks()

    Instead, it is also possible to define a few target compounds. By default, the method takes the
    nearest peaks and estimates the tolerance from the summed spectrum.
    >>> targets = [505.2354, 504.2451]  # arbitrary example, make sure target compounds are within limits
    >>> spec.set_targets(targets=targets)
    If targets are set, the workflow continues with binning the spectra.

    It is possible to filter out peaks below a certain signal-to-noise-ratio threshold or sidepeaks:
    >>> spec.filter_peaks(peaks_snr_threshold=2, remove_sidepeaks=True, plts=True)

    The next step is to set kernels
    >>> spec.set_kernels()
    It is possible to plot this as well, but due to the large number of kernels this will usually
    not be a pleasant experience and without the keyword only the summed intensity of the kernels
    will be plotted
    >>> spec.plot_summed(plt_kernels=True)

    Then, spectra are binned using the kernels (this step will also take a while because every spectrum has
    to be read in again, but this time the hdf5 file can be used if the get_project class was used, which is a lot faster)
    >>> spec.bin_spectra()

    Finally, a feature table is created from the binned spectra
    >>> spec.set_feature_table()
    """
    _indices: np.ndarray[int] | None = None
    _limits: np.ndarray[float] | None = None
    _mzs: np.ndarray[float] | None = None
    _delta_mz: float | None = None
    _intensities: np.ndarray[float] | None = None
    _tic: np.ndarray[float] | None = None

    _noise_level: np.ndarray[float] | None = None
    _noise_level_subtracted: bool = False
    _noise_level_parameters: dict[str, float] | None = None

    _calibration_parameters: np.ndarray[float] | None = None
    _calibration_settings: np.ndarray[float] | None = None

    _peaks: np.ndarray[int] | None = None
    _peak_properties: np.ndarray[float] | None = None
    _peak_setting_parameters: dict[str, Any] | None = None
    _peaks_SNR: np.ndarray[float] | None = None
    _peaks_is_side_peak: np.ndarray[bool] | None = None

    _kernel_params: np.ndarray[float] | None = None
    _kernel_shape: str | None = None

    _binning_by: str | None = None
    _line_spectra: np.ndarray[float] | None = None
    _feature_table: pd.DataFrame | None = None

    _losses: np.ndarray[float] | None = None

    _save_in_d_folder: bool = True
    _save_attrs: set[str] = {
        '_delta_mz',
        '_mzs',
        '_intensities',
        '_tic',
        '_indices',
        '_limits',
        '_peaks',
        '_peak_properties',
        '_peak_setting_parameters',
        '_peaks_SNR',
        '_peaks_is_side_peak',
        '_kernel_params',
        '_kernel_shape',
        '_line_spectra',
        '_feature_table',
        '_losses',
        '_binning_by',
        '_noise_level',
        '_noise_level_parameters',
        '_noise_level_subtracted',
        '_calibration_parameters',
        '_calibration_settings'
    }

    def __init__(
            self,
            *,
            reader: ReadBrukerMCF | hdf5Handler | None = None,
            limits: tuple[float, float] | None = None,
            delta_mz: float = 1e-4,
            indices: Iterable[int] | None = None,
            initiate: bool = True,
            path_d_folder: str | None = None
    ) -> None:
        """
        Initiate the object.

        Either pass a reader or load it from the specified d_folder.

        Parameters
        ----------
        reader : ReadBrukerMCF | hdf5Handler | None, optional
            Reader to obtain metadata and spectra from disk. The default is None.
        limits : tuple[float], optional
            mz range of spectra. The default is None. This defaults to setting
            the limits with parameters from the QTOF.
        delta_mz : float, optional
            Used to resample the spectra when combining to the summed
            spectrum. The default is 1e-4.
        indices : Iterable, optional
            Indices of spectra to be summed. This can be used to only sum up a
            subset of spectra. The default is None and results in summing up 
            all spectra.
        path_d_folder : str | None, optional
            folder with data. The default is str | None. Only necessary if reader
            is not passed.

        Returns
        -------
        None.

        """
        initiate = self._set_files(reader, path_d_folder, initiate)
        self._delta_mz: float = delta_mz
        if initiate:
            self._initiate(reader, indices, limits)
        else:
            if indices is not None:
                self._indices: np.ndarray[int] = np.array(indices)
            if limits is not None:
                self._limits: tuple[float, float] = limits

    def _from_reader(self, reader: ReadBrukerMCF | hdf5Handler) -> None:
        """
        Inherit indices and limits from reader if self does not have them.

        This method also attempts to set the mz values from the limits.
        """
        def try_inherit(attr: str) -> None:
            """Inherit attribute if self does not but reader does have it"""
            private_attr = f'_{attr}'
            if check_attr(self, private_attr):
                return
            if not check_attr(reader, attr):
                return

            logger.info(f'inherited {attr} from reader')
            setattr(self, private_attr, getattr(reader, attr))

        try_inherit('indices')
        try_inherit('limits')

        # set mzs
        if not check_attr(self, '_mzs') and check_attr(self, '_limits'):
            self._mzs = get_mzs_for_limits(self._limits, self._delta_mz)

    def _set_files(
            self,
            reader: ReadBrukerMCF | hdf5Handler | None,
            path_d_folder: str | None,
            initiate: bool,
    ) -> bool:
        assert (reader is not None) or (path_d_folder is not None), \
            'Either pass a reader or the corresponding d-folder'
        if reader is not None:
            assert isinstance(reader, ReadBrukerMCF | hdf5Handler), \
                'Reader must be an instance of ReadBrukerMCF or hdf5Handler'
        if path_d_folder is not None:
            assert isinstance(path_d_folder, str), \
                'Path d_folder must be an instance of str'
        assert initiate in (True, False),\
            'Initiate must be True or False'

        if (reader is None) and initiate:
            logger.warning('cannot initiate without a reader')
            initiate = False

        if path_d_folder is not None:
            path_folder, d_folder = os.path.split(path_d_folder)
        elif reader is not None:
            path_folder, d_folder = os.path.split(reader.path_d_folder)
        else:
            raise NotImplementedError("Internal error, please report.")
        self.path_folder = path_folder
        self.d_folder = d_folder

        return initiate

    @property
    def path_d_folder(self) -> str:
        return os.path.join(self.path_folder, self.d_folder)

    def _initiate(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            indices: Iterable[int] | None,
            limits: tuple[float, float] | None
    ) -> None:
        """
        Set limits and masses based on metadata from the reader.

        Parameters
        ----------
        reader: ReadBrukerMCF | hdf5Handler
            Reader from which to pull the metadata.
        indices: Iterable[int] | None
            Indices to be used. None defaults to using all of them.
        limits: tuple[float, float] | None
            mz limits of the spectra. None defaults to getting the limits from the QTOF
            metadata.

        Notes
        -----
        This method defines the following attribute(s):
        mzs: np.ndarray[float]
            Evenly spaced mz values within the limits (limits are guaranteed to be included)
        intensities: np.ndarray[float]
            Container for summed up intensities with the same shape as mzs.
        limits : tuple[float, float]
            mz limits of the spectra.
        indices : np.ndarray[int]
            Indices of the spectra to use.
        """
        type_reader: type = type(reader)
        is_rtms: bool = type_reader is ReadBrukerMCF

        assert is_rtms or (type_reader is hdf5Handler), \
            (f"Reader must be either a hdf5Handler or ReadBrukerMCF instance. "
             f"You provided {type_reader}")
        assert (indices is None) or (len(indices) > 0), \
            f"indices must either be None or of non-zero length. You provided {indices}."
        assert (limits is None) or ((len(limits) == 2) and limits[0] < limits[1]), \
            (f'limits must either be None or contain an upper and lower bound '
             f'with upper != lower, you provided {limits}.')

        if indices is None:
            if not check_attr(reader, 'indices'):
                reader.create_indices()
            indices = reader.indices
        self._indices = np.array(indices)
        if limits is None:
            if (not check_attr(reader, 'metaData')) and is_rtms:
                reader.set_meta_data()
            if not check_attr(reader, 'limits'):
                reader.set_casi_window()
            limits = reader.limits
        self._limits = limits

        if is_rtms:
            self._mzs = get_mzs_for_limits(self._limits, self._delta_mz)
            reader.set_mzs(self._mzs)
        else:
            self._mzs = reader.mzs
        self._intensities = np.zeros_like(self._mzs)

    def _pre_save(self):
        # only save line spectra, if both exist
        if (
                check_attr(self, '_feature_table')
                and check_attr(self, '_line_spectra')
        ):
            self._save_attrs.remove('_feature_table')

    def _post_save(self):
        self._save_attrs.add('_feature_table')

    @property
    def indices(self) -> np.ndarray[int]:
        assert check_attr(self, '_indices')
        return self._indices

    @property
    def intensities(self) -> np.ndarray[float]:
        assert check_attr(self, '_intensities')
        return self._intensities

    @property
    def mzs(self) -> np.ndarray[float]:
        assert check_attr(self, '_mzs')
        return self._mzs

    @property
    def limits(self):
        assert check_attr(self, '_limits')
        return self._limits

    @property
    def delta_mz(self) -> float:
        assert check_attr(self, '_delta_mz')
        return self._delta_mz

    @property
    def _n_spectra(self) -> int:
        return len(self.indices)

    @property
    def _n_peaks(self) -> int:
        assert check_attr(self, '_peaks')
        return len(self._peaks)

    def reset_binning(self) -> None:
        """Reset the binnend intensities"""
        self._binning_by: str | None = None
        self._line_spectra: np.ndarray[float] | None = None
        self._feature_table: pd.DataFrame | None = None

    def reset_kernels(self) -> None:
        """Reset the kernels and everything downstream"""
        self._kernel_params: np.ndarray[float] | None = None
        self._kernel_shape: str | None = None

        self.reset_binning()

    def reset_peaks(self) -> None:
        """Reset peaks and everything downstream"""
        self._peaks: np.ndarray[int] | None = None
        self._peak_properties: np.ndarray[float] | None = None
        self._peak_setting_parameters: dict[str, Any] | None = None
        self._peaks_SNR: np.ndarray[float] | None = None
        self._peaks_is_side_peak: np.ndarray[bool] | None = None

        self.reset_kernels()

    def reset_noise_level(self):
        self._noise_level: np.ndarray[float] | None = None
        self._noise_level_subtracted: bool = False

        self.reset_peaks()

    def reset_intensities(self) -> None:
        """Reset intensities and everything downstream"""
        self._intensities = np.zeros_like(self.mzs)
        self._tic = np.zeros(self._n_spectra)

        self.reset_noise_level()

    def add_spectrum(self, spectrum: np.ndarray[float]) -> None:
        """Add passed spectrum values to summed spectrum."""
        self._intensities += spectrum

    def get_spectrum(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            index: int | str,
            only_intensity: bool,
            **kwargs: Any
    ) -> np.ndarray[float] | Spectrum:
        """
        Function to obtain a spectrum or its intensities from a reader.
        
        This method is useful to manage the calibration and whether a Spectrum
        object or just its intensities should be returned. The returned spectrum
        is guaranteed to have the same sample points as the spectra instance.

        Parameters
        ----------
        reader : ReadBrukerMCF | hdf5Handler
            Reader object from which to obtain the spectrum.
        index : int | str
            Index of the spectrum to obtain (1-based).
        only_intensity : bool
            Whether to return only the intensity of the spectrum object.
        **kwargs : Any
            Additional keyword arguments to be passed on to reader.get_spectrum.
            Will be ignored if only_intensities is set to True.

        Returns
        -------
        spectrum : np.ndarray[float] | Spectrum
            Either a spectrum object or its intensities calibrated and resampled
            accordingly.

        """
        # Ensure index is int
        index: int = int(index)

        # Determine whether to use calibration functions on spectra
        calibrate: bool = check_attr(self, '_calibration_parameters')

        if calibrate:
            array_idx: int = self.spectrum_idx2array_idx(index)
            poly_coeffs: np.ndarray = self._calibration_parameters[array_idx, :]
        else:
            poly_coeffs: None = None

        if only_intensity:
            spectrum: np.ndarray[float] = \
                reader.get_spectrum_resampled_intensities(
                    index=index, 
                    poly_coeffs=poly_coeffs
            )
        else:
            spectrum: Spectrum = reader.get_spectrum(
                index=index, 
                poly_coeffs=poly_coeffs, 
                **kwargs
            )
        return spectrum

    def add_all_spectra(
            self, 
            reader: ReadBrukerMCF | hdf5Handler
    ) -> None:
        """
        Add up all spectra found in the mcf file.

        Parameters
        ----------
        reader : ReadBrukerMCF | hdf5Handler
            Reader from which to obtain the spectra.
        """
        # inherit indices and limits if not set before
        self._from_reader(reader)
        assert check_attr(self, '_indices'), \
            ('No indices were set and also none were found in the reader. '
             'Make sure to initialize the reader correctly')
        assert check_attr(self, '_mzs'), \
            ('instance does not have mz values and it was also not possible to '
             'get the limits from the reader. Please make sure to initialize '
             'either the instance with limits or the reader')

        if not check_attr(reader, 'mzs'):
            reader.set_mzs(self.mzs)

        self.reset_intensities()

        # iterate over all spectra
        for i, index in tqdm(
                enumerate(self.indices),
                desc='Adding spectra',
                smoothing=50/self._n_spectra,
                total=self._n_spectra
        ):
            spectrum: np.ndarray[float] = self.get_spectrum(
                reader=reader, index=index, only_intensity=True
            )
            self.add_spectrum(spectrum)
            self._tic[i] = np.trapz(spectrum, dx=self.delta_mz)

        # due to floating point precision?
        self._intensities[self._intensities < 0] = 0

        logger.info('done adding up spectra')

    def add_all_spectra_aligned(
            self,
            reader: ReadBrukerMCF | hdf5Handler
    ) -> None:
        """
        Add all spectra together but insert an alignment step based on the
        cross-correlation of the current sum and spectrum.
        
        Parameters
        ----------
        reader : ReadBrukerMCF | hdf5Handler
            Reader from which to obtain the spectra.
        """
        self.reset_intensities()

        # inherit indices and limits if not set before
        self._from_reader(reader)
        assert check_attr(self, '_indices'), \
            ('No indices were set and also none were found in the reader. '
             'Make sure to initialize the reader correctly')

        for it, index in tqdm(
                enumerate(self.indices),
                desc='Adding aligned spectra',
                smoothing=50/self._n_spectra,
                total=self._n_spectra
        ):
            spectrum: np.ndarray[float] = self.get_spectrum(
                reader=reader, index=index, only_intensity=False
            )
            if it > 0:
                shift: float = self.get_mass_shift(spectrum)  # shift in Da
                # shift according to number of spectra
                weight: float = 1 / (it + 1)
                self._mzs += shift * weight
                spectrum.mzs -= shift * (1 - weight)
            self.add_spectrum(spectrum.intensities)
            self._tic[it] = np.trapz(spectrum, dx=self.delta_mz)

        self._intensities[self._intensities < 0] = 0

    def set_noise_level(
            self,
            window_size: int | float | None = None,
            plts: bool = False,
            **_
    ):
        """
        Estimate and remove the noise level from the summed intensities.

        The noise level is stored as an attribute called 'noise_level'.
        It is crucial to find the right window size: If it is too small,
        peaks will lose intensity because the minimum filter climbs up the
        flanks of peaks. If it is too large, the noise level will not be removed
        enterily. By default, the window_size is being estimated from the data
        by taking the broadest peak at a relative height of .8 as the window size.
        The result can be checked by setting the plts keyword to True. The
        result of the minimum filter is passed on to a median filter of the
        same window size to smooth the output.

        Parameters
        ----------
        window_size : float | int, optional
            The window size of the minimum filter as number of sample points or
            in Da.
            None defaults to estimating the window size from the peak widths.
            A window_size of 0 will subtract the smallest intensity.
            A window size in the interval (0, 1) is assumed to be in Da.
        plts: bool, optional
            The default is False. If True, will plot the summed intensities
            before and after baseline removal.

        Notes
        -----
        This method defines the following attribute(s):
        noise_level: np.ndarray
            The noise level of each spectrum. This assumes that each spectrum
            has the same noise level. It is defined as the estimated bsae_line
            divided by the number of spectra.

        """

        def estimate_peaks_width() -> int:
            """Estimate the minimum filter size from the peak widths."""
            prominence = .1 * np.median(self.intensities)
            peaks, peak_props = find_peaks(self.intensities, prominence=prominence, width=3)
            widths, *_ = peak_widths(self.intensities, peaks=peaks, rel_height=.8)
            return int(np.max(widths))

        assert check_attr(self, 'intensities', True), \
            'call add_all_spectra before setting the noise level'

        n_spectra: int = self._n_spectra

        if window_size is None:
            # estimate peak width from FWHM
            window_size: int = estimate_peaks_width()
            logger.info(
                'estimated window size for baseline subtraction is ' +
                f'{self._delta_mz * window_size * 1e3:.1f} mDa'
            )
        elif window_size == 0:  # subtract minimum
            base_lvl: float = self.intensities.min()
            self._intensities -= base_lvl
            self._noise_level: np.ndarray[float] = np.full_like(
                self.intensities, base_lvl / n_spectra
            )
            return
        # convert Da to number of sample points
        elif (window_size < 1) and isinstance(window_size, float):
            dmz: float = self.mzs[1] - self.mzs[0]
            window_size: int = round(window_size / dmz)
        ys_min: np.ndarray[float] = minimum_filter(self.intensities, size=window_size)
        # run median filter on that
        ys_min: np.ndarray[float] = median_filter(ys_min, size=window_size)
        # store for SNR estimation
        self._noise_level: np.ndarray[float] = ys_min / n_spectra
        self._noise_level_parameters: dict[str, float] = dict(
            window_size=window_size
        )

        if plts:
            plt.figure()
            plt.plot(self.mzs, self.intensities, label='signal')
            plt.plot(self.mzs, self.intensities - ys_min, label='baseline corrected')
            plt.plot(self.mzs, ys_min, label='baseline')
            plt.xlabel('m/z in Da')
            plt.ylabel('Intensity')
            plt.legend()
            plt.show()

    def require_noise_level(self, overwrite=False, **kwargs) -> np.ndarray[float]:
        if overwrite:
            self.reset_noise_level()

        if not check_attr(self, '_noise_level'):
            self.set_noise_level(**kwargs)

        return self._noise_level

    @property
    def noise_level(self) -> np.ndarray[float]:
        return self.require_noise_level()

    def subtract_baseline(
            self,
            **kwargs
    ) -> None:
        assert not self._noise_level_subtracted, 'Cannot subtract baseline again'
        ys_min = self.require_noise_level(**kwargs)
        self._intensities -= ys_min * self._n_spectra
        self._noise_level_subtracted = True

        self._intensities[self._intensities < 0] = 0

    def get_mass_shift(
            self: Self,
            other: Spectrum,
            max_mass_offset: float | None = 1e-3,
            plts: bool = False
    ) -> float:
        """
        Calculate cross-correlation for self and other and return the maximum.

        Parameters
        ----------
        other: Spectrum,
            The spectrum with which to calculate the x-correlation.
        max_mass_offset : float | None, optional
            The maximal allowed mass difference between the two spectra in 
            Da. The default is 1 mDa. None will not restrict the search
            space.
        plts: bool, optional.
            Whether to plot the result of the cross-correlation. The default is False.

        Returns
        -------
        mass_offset: float
            The mass offset between the two spectra.

        """
        diffs: np.ndarray[float] = np.diff(self.mzs)

        # make sure other has the same sampling as this
        other.resample(self.mzs)
        a: np.ndarray = self.intensities
        b: np.ndarray = other.intensities
        n_mzs: int = len(b)

        lags: np.ndarray[float] = correlation_lags(n_mzs, n_mzs, mode='full')
        masses: np.ndarray[float] = diffs[0] * lags
        corrs: np.ndarray[float] = correlate(a, b, mode='full')
        if max_mass_offset is not None:
            mask: np.ndarray[bool] = np.abs(masses) <= max_mass_offset
        else:
            mask: np.ndarray[bool] = np.ones_like(corrs, dtype=bool)

        # set values outside allowed lag range to 0
        corrs[~mask] = 0
        idx: int = np.argmax(corrs)  # largest xcorr
        if (idx == 0) or (idx == len(corrs) - 1):  # idx at bound, set to 0
            idx: int = np.argwhere(lags == 0)[0][0]
        lag: float = lags[idx]
        mass_offset: float = masses[idx]
        if plts:
            plt.figure()
            plt.plot(masses[mask] * 1e3, corrs[mask])
            plt.plot(mass_offset * 1e3, corrs[idx], 'ro')
            plt.xlabel('m/z in mDa')
            plt.ylabel('Correlation')
            plt.title(f'{lag=}, mass shift = {mass_offset*1e3:.1f} mDa')
            plt.show()
        return mass_offset

    def set_peaks(self, prominence: float = .1, width=3, **kwargs):
        """
        Find peaks in summed spectrum using scipy's find_peaks function.

        Parameters
        ----------
        prominence : float, optional
            Required prominence for peaks. The default is 0.1. This defaults
            to 10 % of the maximum intensity. If the prominence is smaller than 1, 
            it will be interpreted to be relative to the median, otherwise as
            the absolute value.
        width : int, optional
            Minimum number of points between peaks. The default is 3.
        **kwargs : dict
            Additional kwargs for find_peaks.

        Sets peaks and properties

        """
        self.reset_peaks()

        if prominence < 1:
            med: float = np.median(self.intensities)
            prominence *= med

        # pop out valid kwargs
        valid_kwargs: list[str] = 'height threshold distance wlen rel_height plateau_size'.split()
        kwargs_peaks: dict[str, Any] = {k: v for k, v in kwargs.items() if k in valid_kwargs}

        self._peaks, self._peak_properties = find_peaks(
            self.intensities, prominence=prominence, width=width, **kwargs_peaks
        )

        # save parameters to dict for later reference
        self._peak_setting_parameters: dict[str, Any] = kwargs_peaks
        self._peak_setting_parameters['prominence'] = prominence
        self._peak_setting_parameters['width'] = width

    def require_peaks(self, overwrite=False, **kwargs) -> np.ndarray[int]:
        if overwrite or (not check_attr(self, '_peaks')):
            self.set_peaks(**kwargs)
        return self._peaks

    def _check_calibration_file_exists(self) -> bool:
        """
        Look for Calibrator.ami file in d folder.

        Lock mass calibration always creates this file, so this can be used as an indicator.
        """
        # calibration always creates an ami file
        return os.path.exists(
            os.path.join(
                self.path_d_folder, 'Calibrator.ami'
            )
        )

    def set_calibration_functions(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            calibrants_mz: Iterable[float] = None,
            search_range: float = 5e-3,
            calib_snr_threshold: float = 4,
            max_degree: int = 1,
            method: str = 'polynomial',
            min_height: float | int = 10_000,
            nearest: bool = False,
            **_
    ):
        """
        Calibrate spectra using calibrants by fitting a polynomial of degree
        max_degree or less.

        This algorithm matches the cloesest peak fulfilling the criteria (search
        range and calib_snr_threshold or min_height) to the theoretical masses.
        A polynomial of at most degree max_degree is fitted to the differences
        from the closest peak to theoretical masses. If not enough peaks are found,
        the degree of the polynomial will be lowered. If no peak is found, the
        spectrum will not be calibrated.

        Parameters
        ----------
        calibrants_mz : float | Iterable[float] | None, optional
            Exact mass(es) of the calibrants in Da. If not provided, will use
            the calibrant list from Wörmer et al., 2019 (Towards multiproxy,
            ultra-high resolution molecular stratigraphy: Enabling laser-induced
            mass spectrometry imaging of diverse molecular biomarkers in sediments,
            Appendix)
        search_range : float, optional
            Range in which to look for peaks in Da. The default is 5 mDa.
            This will look in a range of +/- 5 mDa around the theoretical mass.
        calib_snr_threshold : float, optional
            Minimal prominence required for a peak to be considered (see
            prominence in set_peaks). By default, an SNR of 4 is used. If a value
            of 0 is provided, the min_height condition is applied instead.
        max_degree: int, optional
            Maximum degree of the polynomial used to describe the fit. If the
            number of matched peaks is greater than the required number of
            points, the best fit is used.
        method: str, optional
            The type of calibration function to use. 'polynomial' will fit a
            polynomial of max_degree or number of found calibrants to the
            spectrum.
        min_height: float | int, optional
            Minimum intensity required. The default is 10_000. Only used, if
            calib_snr_threshold is not provided.
        nearest: bool, optional
            If True, will only take the nearest peak to the calibrant. The default
            is False, which will take the highest peak within the search range.

        Notes
        -----
        This function sets the attribute(s):
        calibration_parameters: np.ndarray,
            Array holding the coefficients of the calibration functions where
            each row corresponds to the coefficients of a calibration function.

        This function tries to emulate the calibration performed in
        DataAnalysis 5.0. This is the description from the handbook:
        -- The spectrum is searched for signals which (1) are within the given
        search range (m/z) of the expected lock mass(es) and which (2) exceed
        the given intensity threshold. The expected lock mass(es), search range
        (m/z) and intensity threshold are specified in the corresponding method.
        -- If at least one lock mass peak is found in the spectrum one of current
        calibration coefficients is corrected such as it is needed to adapt the
        current m/z value of the lock mass signal exactly to the theoretical
        m/z value of the respective lock mass. The new correlation coefficient
        is then used to recalibrate the respective spectrum.
        -- If the spectrum does not have at least one lock mass peak (above the
        given intensity threshold) the current calibration coefficient is kept
        for that spectrum.

        To me, it is not clear what the calibration/ correlation coefficient is
        supposed to mean, but a linear fit seems reasonable as higher degree
        polynomials may result in unreasonably large shifts outside the found
        peak range.
        """
        def calib_spec(_spectrum: np.ndarray[float]) -> int | tuple[np.ndarray[float], int]:
            """Find the calibration function for a single spectrum."""
            # pick peaks
            if calib_snr_threshold > 0:  # only set peaks above the SNR threshold
                peaks: np.ndarray[int] = find_peaks(
                    _spectrum, height=self._noise_level * calib_snr_threshold
                )[0]
            else:
                peaks: np.ndarray[int] = find_peaks(_spectrum, height=min_height)[0]
            peaks_mzs: np.ndarray[float] = self.mzs[peaks]
            peaks_intensities: np.ndarray[float] = self.intensities[peaks]

            # find valid peaks for each calibrant
            closest_peak_mzs: list[float] = []
            closest_calibrant_mzs: list[float] = []
            for jt, calibrant in enumerate(calibrants_mz):
                distances: np.ndarray[float] = np.abs(calibrant - peaks_mzs)  # theory - actual
                if not np.any(distances < search_range):  # no peak with required SNR found inside range
                    logger.debug(
                        f'found no peak above noise level for {calibrant=} and {index=}'
                    )
                    calibrator_presences[it, jt] = False
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
            if n_calibrants == 0:  # no calibrant found, keep identity
                logger.debug(f'found no calibrant for {index=}')
                return -1
            degree: int = min([max_degree, n_calibrants - 1])

            # forbid degree>=2 if the calibrants are far away from the beginning and end of the spectrum, set 5Da for now.
            if degree > 1:
                assert(abs(min(calibrants_mz) - min(peaks_mzs)) <= 5), \
                    'calibrants are too far away from the beginning of the spectrum'
                assert(abs(max(calibrants_mz) - max(peaks_mzs)) <= 5), \
                    'calibrants are too far away from the end of the spectrum'

            # polynomial coefficients
            # theory - actual
            yvals = [t - a for t, a in zip(closest_calibrant_mzs, closest_peak_mzs)]
            p: np.ndarray[float] = np.polyfit(x=closest_peak_mzs, y=yvals, deg=degree)
            n_coeffs: int = degree + 1  # number of coefficients in polynomial
            # fill coeff matrix
            return p, n_coeffs

        if self._check_calibration_file_exists():
            logger.warning(
                'Found calibration file. This suggests that Lock Mass calibration '
                'has been performed already.'
            )

        assert isinstance(reader, ReadBrukerMCF | hdf5Handler), \
            f'reader must be a ReadBrukerMCF or hdf5Handler instance, not {type(reader)}'
        assert check_attr(reader, 'indices'), 'call reader.set_indices()'
        assert check_attr(reader, 'mzs') and np.allclose(reader.mzs, self.mzs), \
            ('Make sure the mzs of the reader match that of the spectra object '
             '(consider calling reader.set_mzs(spec.mzs))')
        if calib_snr_threshold > 0:
            assert check_attr(self, 'noise_level'), \
                ('This step has to be performed after subtracting the baseline '
                 'to have access to the noise_level, unless you set the '
                 'calib_snr_threshold to 0.')
        if calibrants_mz is None:
            calibrants_mz = get_calibrants(self.limits)

        n_spectra: int = self._n_spectra  # number spectra

        # first column corresponds to highest degree
        # last column to constant
        # each row corresponds to a spectrum
        # ensure transformation has at least 2 parameters
        n_calibration_parameters: int = max_degree + 1

        calibration_parameters: np.ndarray[float] = np.zeros(
            (n_spectra, n_calibration_parameters),
            dtype=float
        )
        calibrator_presences: np.ndarray[bool] = np.ones(
            (n_spectra, len(calibrants_mz)),
            dtype=bool
        )

        # iterate over all spectra
        for it, index in tqdm(
                enumerate(self.indices),
                total=n_spectra,
                desc='Finding calibration parameters',
                smoothing=50 / n_spectra
        ):
            spectrum: np.ndarray[float] = \
                reader.get_spectrum_resampled_intensities(int(index))
            ret = calib_spec(spectrum)
            if ret == -1:
                continue
            p, n_coeffs = ret
            calibration_parameters[it, n_calibration_parameters - n_coeffs:] = p

        calibrant_matches: dict[float, float] = dict(zip(
            calibrants_mz,
            calibrator_presences.mean(axis=0)
        ))

        logger.info('done calibrating spectra, found calibrants in the following abundances:')
        logger.info('\n'.join([f'{k} : {v:.0%}' for k, v in calibrant_matches.items()]))

        self._calibration_parameters = calibration_parameters

        self._calibration_settings: dict[str, Any] = {
            'calibrants': calibrants_mz,
            'search_range': search_range,
            'calib_snr_threshold': calib_snr_threshold,
            'max_degree': max_degree,
            'presences calibrants': calibrant_matches
        }

    def require_calibration_functions(
            self,
            *args,
            overwrite: bool = False,
            **kwargs
    ) -> np.ndarray[float]:
        """
        Try to return existing calibration parameters.

        Parameters
        ----------
        overwrite: bool, optional
            If this is set to True, previous calibrations will be overwritten
        """
        if overwrite or (not check_attr(
                self, '_calibration_parameters', True)
        ):
            self.set_calibration_functions(*args, **kwargs)

        return self._calibration_parameters

    def set_side_peaks(
            self, 
            max_relative_height: float = .1,
            max_distance: float = .001,
    ) -> None:
        """
        This method defines a list of peaks that are likely artifacts from the
        Fourier transformation.

        Windowing introduces spectral leakage, which is especially pronounced
        around high isolated peaks. This method goes through the peaks list
        and marks peaks with low relative height to their neighbour within a
        small region as side-peaks.

        Parameters
        ----------
        max_relative_height : float, optional
            Relative height below which peaks are considered artifacts. The
            default is 0.1.
        max_distance : float, optional
            Region in which to look for artifacts. The default is 1 mDa.

        Notes
        -----
        This method defines the following attribute(s):
        peaks_is_side_peak : np.ndarray[bool]
            An array marking sidepeaks as True, otherwise as False.
        """
        assert check_attr(self, '_peaks'), 'call set_peaks first'

        def eval_peak_valid(pc: int, ps: int) -> bool:
            """
            Evaluate if the side peak (ps) of a given peak (pc) is valid 
            (meaning likely not a sidepeak).
            """
            # get corresponding mz and intensity values
            pc_mz, pc_int = self.mzs[pc], self.intensities[pc]
            ps_mz, ps_int = self.mzs[ps], self.intensities[ps]
            # check if inside influence zone, otherwise can't be sidepeak
            if abs(pc_mz - ps_mz) > max_distance:
                return True
            # check if below max_relative_height, otherwise can't be sidepeak
            elif ps_int / pc_int > max_relative_height:
                return True
            return False

        # sort peaks by intensity (highest first)
        N_peaks: int = len(self._peaks)
        peak_order: np.ndarray[int] = np.argsort(self.intensities[self._peaks])[::-1]
        # peaks that have to be evaluated
        to_do: np.ndarray[bool] = np.full(N_peaks, True)
        valids: np.ndarray[bool] = np.full(N_peaks, True)
        for peak_idx in peak_order:
            if not to_do[peak_idx]:
                continue
            # mz idx corresponding to peak
            peak: int = self._peaks[peak_idx]
            # check if peaks left and right of it fulfill conditions
            # if they are detected as sidepeaks, they no longer have to be taken into consideration
            # valids will be changed accordingly
            if ((peak_idx_l := peak_idx - 1) >= 0) and to_do[peak_idx_l]:
                peak_l: int = self._peaks[peak_idx_l]
                valid: bool = eval_peak_valid(peak, peak_l)
                valids[peak_idx_l] = valid
                if not valid:
                    to_do[peak_idx_l] = False
            if ((peak_idx_r := peak_idx + 1) < N_peaks) and to_do[peak_idx_r]:
                peak_r: int = self._peaks[peak_idx_r]
                valid: bool = eval_peak_valid(peak, peak_r)
                valids[peak_idx_r] = valid
                if not valid:
                    to_do[peak_idx_r] = False
            to_do[peak_idx] = False

        self._peaks_is_side_peak: np.ndarray[bool] = ~valids

    def require_side_peaks(self, **kwargs) -> np.ndarray[bool]:
        if not check_attr(self, '_peaks_is_side_peak'):
            self.set_side_peaks(**kwargs)

        return self._peaks_is_side_peak

    def _set_peaks_SNR(self) -> None:
        """
        Set the SNRs of peaks based on the noise level.

        Notes
        -----
        This method defines the following attribute(s):
        peaks_SNR: np.ndarray[float]
            The noise level for each mz value.
        """
        assert check_attr(self, 'noise_level'), 'call subtract_baseline'
        N_spec: int = len(self.indices)
        av_intensities: np.ndarray[float] = self.intensities[self._peaks] / N_spec
        self._peaks_SNR: np.ndarray[float] = \
            av_intensities / self.noise_level[self._peaks]

    @property
    def peaks_SNR(self) -> np.ndarray[float]:
        if not check_attr(self, '_peaks_SNR'):
            self._set_peaks_SNR()
        return self._peaks_SNR

    def _get_SNR_table(self) -> np.ndarray[float]:
        """
        Return SNR values for each spectrum as array where each row corresponds
        to a spectrum. 
        
        Noise level is assumed to be the same for each spectrum.
        This function scales the intensities by the estimated noise level, 
        thereby assigning SNR values to all peaks in each spectrum.
        """
        assert check_attr(self, '_noise_level'), \
            'call set_noise_level first'
        
        heights: np.ndarray[float] = self.get_heights()
        # noise level is assumed to be the same for each spectrum
        # get noise levels at centers of each peak
        noise_levels: np.ndarray[float] = np.array([
            self.noise_level[
                np.argmin(
                    np.abs(self.mzs - mz_c)
                )
            ]
            for mz_c in self.kernel_params[:, 0]
        ])

        SNRs: np.ndarray[float] = heights / noise_levels

        return SNRs

    def filter_peaks(
            self,
            whitelist: Iterable[int] | None = None,
            peaks_snr_threshold: float = 0,
            remove_sidepeaks: bool = False,
            thr_sigma: Iterable[float] | None = None,
            plts: bool = False,
            **kwargs_sidepeaks: Any
    ) -> None:
        """
        Eliminate peaks not fulfilling the criteria from the peak list.

        Parameters
        ----------
        whitelist : Iterable[int] | None,
            Peaks (indices refer to mz values array in _peaks) that shall not be
            removed. If this is provided, no other filtering will be done.
        peaks_snr_threshold : float, optional
            Minimum SNR required for keeping peaks. The default is to not
            remove any peaks based on SNR. Recommended value range: 0 to 10.
        remove_sidepeaks : bool, optional
            Remove peaks likely introduced by spectral leakage (see set_side_peaks).
            The default is not to remove side peaks.
        thr_sigma: tuple[float, float], optional
            Realistic bounds for peak widths. Recommended: thr_sigma=(2e-3, 5e-3).
            This only keeps kernels whose sigma is between 2 and 5 mDa.
        plts: bool, optional,
            Whether to plot the removed peaks.
        kwargs_sidepeaks: Any
            Additional kwargs passed on to set_side_peaks.
        """
        # skip filtering of valid peaks if a list of peaks to keep is provided
        skip_filtering: bool = whitelist is not None

        if skip_filtering and (
                (peaks_snr_threshold != 0)
                or remove_sidepeaks
                or (peaks_snr_threshold > 0)
                or (thr_sigma is not None)
        ):
            logger.warning(
                'A whitelist was provided as well as other criteria but ' +
                'filtering will not be executed if a whitelist is provided. ' +
                'If this is what you want, consider performing two filtering ' +
                'steps instead.'
            )

        n_peaks: int = len(self._peaks)
        peaks_valid: np.ndarray[bool] = np.full(n_peaks, True, dtype=bool)

        if skip_filtering:  # exclude all peaks not in whitelist
            whitelisted = np.array([peak in whitelist
                                    for peak in self._peaks], dtype=bool)
            peaks_valid &= whitelisted
            logging.info(f'keeping {whitelisted.sum()} whitelisted '
                         f'out of {n_peaks} peaks')
        if (not skip_filtering) and (peaks_snr_threshold > 0):
            # set peaks below SNR threshold to False
            snred = self.peaks_SNR > peaks_snr_threshold
            peaks_valid &= snred
            logging.info(f'keeping {snred.sum()} out of {n_peaks} peaks with SNR '
                         f'above {peaks_snr_threshold}')
        if (not skip_filtering) and remove_sidepeaks:  # set sidepeaks to False
            side_peaks = self.require_side_peaks(**kwargs_sidepeaks)
            peaks_valid &= ~side_peaks
            logging.info(f'Removing {side_peaks.sum()} peaks identified as side peaks')
        if (not skip_filtering) and (thr_sigma is not None):
            if not check_attr(self, '_kernel_params'):
                logger.warning(f'Kernel parameters not set, peak filtering '
                               f'with {thr_sigma=} has no effect.')
            else:
                assert (
                        hasattr(thr_sigma, '__iter__')
                        and (len(thr_sigma) == 2)
                        and (thr_sigma[0] < thr_sigma[1])
                ), (f'thr_sigma must be a 2-tuple with the first value being '
                    f'strictly less than the second one, you provided {thr_sigma=}')
                sigma_filtered: np.ndarray[bool] = (
                        (self._kernel_params[:, -1] > thr_sigma[0])
                        & (self._kernel_params[:, -1] < thr_sigma[1])
                )
                peaks_valid &= sigma_filtered
                logger.info(f'Keeping {sigma_filtered.sum()} out of {n_peaks} '
                            f'with sigma between {thr_sigma[0] * 1e3:.1f} and '
                            f'{thr_sigma[1] * 1e3:.1f} mDa')

        if plts:  # keep a copy of the original peaks
            peaks = self._peaks.copy()

        # filter out invalid peaks
        self._peaks: np.ndarray[int] = self._peaks[peaks_valid]
        peak_props = {}
        for key, val in self._peak_properties.items():
            peak_props[key] = val[peaks_valid]
        self._peak_properties = peak_props
        # add flag
        self._peak_setting_parameters['modified'] = {
            'whitelist': whitelist,
            'calib_snr_threshold': peaks_snr_threshold,
            'remove_sidepeaks': remove_sidepeaks
        }

        if check_attr(self, '_kernel_params'):
            self._kernel_params = self._kernel_params[peaks_valid, :]

        if plts:
            idxs_removed: np.ndarray[int] = peaks[~peaks_valid]
            idxs_valid: np.ndarray[int] = peaks[peaks_valid]
            plt.figure()
            plt.plot(self.mzs, self.intensities)
            plt.scatter(
                self.mzs[idxs_removed],
                self.intensities[idxs_removed],
                label='removed peaks',
                c='r'
            )
            plt.scatter(
                self.mzs[idxs_valid],
                self.intensities[idxs_valid],
                label='valid peak',
                c='g'
            )
            plt.xlabel('m/z in Da')
            plt.ylabel('Intensities')
            plt.legend()
            plt.show()

    def _gaussian_from_peak(self, peak_idx: int) -> tuple[float, ...]:
        """
        Obtain the parameters for a gaussian from peak properties and intensities.

        Parameters
        ----------
        peak_idx : int
            The n-th peak.

        Returns
        -------
        tuple with
        mz_c : float
            The center of the gaussian.
        H : float
            The height of the gaussian.
        sigma : flaot
            The standard deviation of the gaussian.
        """
        assert check_attr(self, '_peaks'), 'call set_peaks first'
        mz_idx: float = self._peaks[peak_idx]  # mz index of center

        H: float = self.intensities[mz_idx]  # corresponding height
        # width of peak at half maximum
        FWHM_l: float = self.mzs[
            (self._peak_properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r: float = self.mzs[
            (self._peak_properties["right_ips"][peak_idx] + .5).astype(int)
        ]
        mz_c: float = (FWHM_l + FWHM_r) / 2
        # convert FWHM to standard deviation
        sigma_l: float = -(FWHM_l - mz_c) / (2 * np.log(2))
        sigma_r: float = (FWHM_r - mz_c) / (2 * np.log(2))
        sigma: float = (sigma_l + sigma_r) / 2
        return mz_c, H, sigma

    def _bigaussian_from_peak(self, peak_idx: int) -> tuple[float, ...]:
        """
        Obtain the parameters for a gaussian from peak properties and intensities.

        Parameters
        ----------
        peak_idx : int
            The n-th peak.

        Returns
        -------
        tuple with
        mz_c : float
            The center of the gaussian.
        H : float
            The height of the gaussian.
        sigma_l : flaot
            The left-sided standard deviation of the gaussian.
        sigma_r : flaot
            The right-sided standard deviation of the gaussian.
        """
        assert check_attr(self, '_peaks'), 'call set_peaks first'
        mz_idx: int = self._peaks[peak_idx]  # mz index of of center
        mz_c: float = self.mzs[mz_idx]  # center of gaussian
        # height at center of peak - prominence
        H: float = self.intensities[mz_idx]  # corresponding height
        # width of peak at half maximum
        FWHM_l: float = self.mzs[
            (self._peak_properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r: float = self.mzs[
            (self._peak_properties["right_ips"][peak_idx] + .5).astype(int)
        ]
        # convert FWHM to standard deviation
        sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
        sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))
        return mz_c, H, sigma_l, sigma_r

    def _kernel_fit_from_peak(
            self,
            peak_idx: int,
            sigma_max: float = 5e-3,
            suppress_warnings: bool = False,
            **_
    ) -> np.ndarray | None:
        """
        Fine-tune kernel parameters for a peak with the shape of a (bi)gaussian.

        This method fine-tunes kernel parameters making use of scipy's curve_fit
        function. If the kernel parameters. If all kernel parameters are 0, the function exists prematurely.

        Parameters
        ----------
        peak_idx: int
            Index of the peak to tune (refers to the array of self.peaks)
        sigma_max: float, optional
            The maximum allowed value for sigma. Defaults to 5 mDa. Larger
            peaks are likely coelutions.
        suppress_warnings: bool, optional
            If this is set to True, will not throw a warning if parameters
            can not be determined.
        """
        assert check_attr(self, '_peaks'), 'call set_peaks first'
        assert check_attr(self, '_kernel_params'), \
            'call set_kernel_params first'

        # width of peak at half maximum
        idx_l: np.ndarray[int] = np.around(
            self._peak_properties["left_ips"][peak_idx]
        ).astype(int)
        idx_r: np.ndarray[int] = np.around(
            self._peak_properties["right_ips"][peak_idx]
        ).astype(int)
        mask: slice = slice(idx_l, idx_r)

        if not np.any(self._kernel_params[peak_idx, :]):
            return None

        mz_c, H, sigma, *sigma_r = self._kernel_params[peak_idx, :]
        if (sigma > sigma_max) and not suppress_warnings:
            logger.warning(
                f'sigma of kernel ({sigma * 1e3:.1f} mDa) with index {peak_idx} '
                f'is bigger than max ({sigma_max * 1e3:.1f} mDa), halfing sigma.'
            )

            # take smaller window to hopefully climb up peak
            *_, l, r = peak_widths(
                self.intensities, [self._peaks[peak_idx]], rel_height=.9
            )
            mask: slice = slice(round(l[0]), round(r[0]))

            sigma /= 2

        bounds_l: list[float] = [
            mz_c - sigma / 4,
            H * .8,
            sigma * .8
        ]
        bounds_r: list[float] = [
            mz_c + sigma / 4,
            H * 1.2,
            min(sigma * 1.2, sigma_max)
        ]

        if len(sigma_r) > 0:  # bigaussian kernel shape
            bounds_l.append(sigma_r[0] * .8)
            bounds_r.append(sigma_r[0] * 1.2)
        try:
            params, *_ = curve_fit(
                f=self._kernel_func,
                xdata=self.mzs[mask],
                ydata=self.intensities[mask],
                p0=self._kernel_params[peak_idx, :],
                bounds=(bounds_l, bounds_r)
            )
        except ValueError as e:
            if not suppress_warnings:
                logger.warning(
                    f'encountered a value error while trying to find parameters '
                    f'for peak with index {peak_idx}: \n {e} \n'
                    f'This can happen for double peaks.'
                )
            return

        return params

    @property
    def _kernel_func(
            self
    ) -> (Callable[[np.ndarray, float, float, float, float], np.ndarray] |
          Callable[[np.ndarray, float, float, float], np.ndarray]):
        """Return either gaussian or bigaussian function."""
        assert check_attr(self, '_kernel_shape')
        if self._kernel_shape == 'bigaussian':
            return bigaussian
        elif self._kernel_shape == 'gaussian':
            return gaussian

    @property
    def _kernel_func_from_peak(self) -> Callable[[int], tuple[float, ...]]:
        """Return either gaussian or bigaussian function"""
        if self._kernel_shape == 'bigaussian':
            return self._bigaussian_from_peak
        elif self._kernel_shape == 'gaussian':
            return self._gaussian_from_peak

    def set_kernels(
            self,
            use_bigaussian: bool = False,
            fine_tune: bool = True,
            discard_invalid: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Based on the peak properties, find (bi)gaussian parameters to
        approximate spectrum.

        Creates kernel_params where cols correspond to peaks and rows different
        properties. Properties are: m/z, intensity H, sigma (left, sigma right)

        Parameters
        ----------
        use_bigaussian : bool, optional
            Whether to use bigaussian or gaussian kernels (testing recommends
            using gaussian kernels). The default is False (so using gaussian
            kernels).
        fine_tune : bool, optional
            If this is set to False, kernel parameters will only be estimated
            from the height and width of the peak. This can be inaccurate for
            noisy peaks. It is recommeded to set this parameter to True in
            which case an optimizer will be used to find the peak shape on all
            points within a few standard deviations. The default is True.
        discard_invalid: bool, optional
            This controls what happens with kernels for which the fine-tuning
            failed. By default, those peaks will be discarded but it may be
            desirable to keep those peaks. In that case set discard_invalid to
            False. This will leave the kernel parameters from the rough
            estimate.
        kwargs: Any
            Additional keyword arguments for _kernel_fit_from_peak

        Notes
        -----
        This method defines the following attribute(s):
        kernel_shape : str
            Flag defining whether gaussian or bigaussian kernels are used.
        kernel_params : np.ndarray[float]
            Array storing parameters describing the kernels. Each row
            corresponds to a kernel. Depending on the kernel shape, columns are
            either mz_c, H, sigma (for gaussian) or mz_c, H, sigma_l, sigma_r
            (for bigaussian).

        """
        assert check_attr(self, '_peaks'), 'call set peaks first'

        y: np.ndarray[float] = self.intensities.copy()

        if use_bigaussian:
            self._kernel_shape: str = 'bigaussian'
            self._kernel_params: np.ndarray[float] = np.zeros((len(self._peaks), 4))
        else:
            self._kernel_shape: str = 'gaussian'
            self._kernel_params: np.ndarray[float] = np.zeros((len(self._peaks), 3))

        # start with heighest peak, work down
        idxs_peaks: np.ndarray[int] = np.argsort(
            self._peak_properties['prominences']
        )[::-1]
        mask_valid: np.ndarray[bool] = np.ones(len(self._peaks), dtype=bool)
        for idx in tqdm(
                idxs_peaks,
                desc='setting peak parameters',
                total=len(idxs_peaks),
                smoothing=50/len(idxs_peaks)
        ):
            params: tuple = self._kernel_func_from_peak(idx)
            if self._kernel_shape == 'bigaussian':
                mz_c, H, sigma_l, sigma_r = params
            else:
                mz_c, H, sigma_l = params
                sigma_r = 1  # so that next condition doesn't fail
            if (H <= 0) or (sigma_l <= 0) or (sigma_r <= 0):
                mask_valid[idx] = False
                continue
            else:
                self._kernel_params[idx, :] = params
            if fine_tune:
                params: np.ndarray = self._kernel_fit_from_peak(idx, **kwargs)
                if params is not None:
                    self._kernel_params[idx, :] = params
                else:
                    mask_valid[idx] = False
                    continue
            self._intensities -= self._kernel_func(
                self.mzs, *self._kernel_params[idx, :]
            )
        # restore intensities
        self._intensities: np.ndarray[float] = y

        # delete invalid peaks
        if discard_invalid:
            self.filter_peaks(whitelist=self._peaks[mask_valid])

    def require_kernels(self, **kwargs) -> np.ndarray[float]:
        if not check_attr(self, '_kernel_params', True):
            self.set_kernels(**kwargs)
        return self._kernel_params

    @property
    def kernel_params(self):
        return self.require_kernels()

    def _find_tolerances(
            self,
            targets: np.ndarray[float],
            tolerances: Iterable[float] | float | None
    ) -> np.ndarray[float]:
        if tolerances is None:
            assert check_attr(self, 'intensities', True), \
                'estimating the tolerance requires that the spectra have been summed up'

        if tolerances is None:
            logger.info('estimating tolerance from summed spectrum ...')
            self.require_peaks()
            self.require_kernels()
            tolerances: float = np.median(self.kernel_params[:, -1])
            logger.info(f'found tolerance of {tolerances * 1e3:.1f} mDa')

        tolerances_a: np.ndarray[float] = np.array(tolerances)

        if tolerances_a.shape in ((1,), ()):
            tolerances_a: np.ndarray[float] = np.full_like(targets, tolerances)
        assert tolerances_a.shape == targets.shape, \
            ('If widths is not a scalar, widths and targets must have the same '
             'number of elements.')

        return tolerances_a

    def set_targets(
            self,
            targets: Iterable[float],
            tolerances: Iterable[float] | float | None = None,
            method_peak_center: str = 'theory',
            method: str = 'max',
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Set the peaks based on a number of target compounds (mass given in Da).
        Peaks are searched within a mass window defined by tolerances, such that the
        assigned mass is within target +/- tolerance. If no tolerance is given,
        it is estimated from the kernel widths.

        This function sets the kernel_params and then requires using the
        bin_spectra function.

        Parameters
        ----------
        targets: Iterable[float]
            The m/z values of interest in Da
        tolerances: Iterable[float] | float
            Tolerance(s) for deviation of peaks from theoretical masses.
        method_peak_center: str, optional
            How target mzs are set. Options are
            - theo: theoretical m/z masses
            - closest: will use found peaks and pick the closest one, if it is
              inside the tolerance
        method: str, optional
            Method used to bin intensities. Will be provided to bin_spectra.
            Options are
            - max (default)
            - height
            - area
        plts: bool, optional
            Whether to plot the target kernels on top of the summed spectrum.
        kwargs: Any,
            Keyword arguments provided to bin_spectra and require_peaks.
        """
        assert method_peak_center in ["theory", "closest"], \
            (f'available options for method_peak_center are "theo" and '
             f'"closest, you provided {method_peak_center}')

        if plts:
            assert not np.all(self.intensities == 0), \
                ('if you want to plot the target kernels, add up the '
                 'intensities first')

        targets: np.ndarray[float] = np.array(targets)

        tolerances = self._find_tolerances(targets, tolerances)

        # set peaks and kernels artificially from the target mzs and tolerances
        if method_peak_center == 'theory':
            self.reset_peaks()
            self._peaks = [np.argmin(np.abs(self.mzs - target)) for target in targets]
        elif method_peak_center == 'closest':
            self.require_peaks(**kwargs)
            whitelist = []
            peak_mzs: np.ndarray = self.mzs[self._peaks]
            valid_idcs = []
            for i, (target, tolerance) in enumerate(zip(targets, tolerances)):
                # idx of mz closest to
                dists: np.ndarray[float] = np.abs(target - peak_mzs)
                dist: float = np.min(dists)
                idx_peak: int = self._peaks[np.argmin(dists)]
                if dist > tolerance:
                    logger.warning(
                        f'did not find peak for {target} within {tolerance=}'
                    )
                    continue
                whitelist.append(idx_peak)
                valid_idcs.append(i)
            self.filter_peaks(whitelist=whitelist)
            # update targets and tolerances to those found
            targets = [targets[i] for i in valid_idcs]
            tolerances = [tolerances[i] for i in valid_idcs]
            print(whitelist, targets, tolerances)
        else:
            raise NotImplementedError('internal error')

        n_peaks: int = len(targets)  # number of peaks equal to targets
        n_spectra: int = self._n_spectra
        self._line_spectra: np.ndarray[float] = np.zeros((n_spectra, n_peaks))

        self._peak_setting_parameters = {
            'method': method_peak_center,
            'targets': np.array(targets),
            'tolerance': tolerances
        }
        # set kernels based on tolerance
        self._kernel_shape = 'gaussian'
        self._kernel_params = np.zeros((n_peaks, 3))
        self._kernel_params[:, 0] = targets  # center
        self._kernel_params[:, 1] = self.intensities[self._peaks]
        self._kernel_params[:, 2] = tolerances

        if plts:
            ys = np.zeros_like(self.mzs)
            mask = np.zeros_like(self.mzs, dtype=bool)
            plt.figure()
            plt.plot(self.mzs, self.intensities, label='original')
            for i in range(len(self._peaks)):
                y = self._kernel_func(self.mzs, *self.kernel_params[i, :])
                mask_kernel = (np.abs(self.kernel_params[i, 0] - self.mzs)
                               <= (20 * self.kernel_params[i, -1]))

                mask |= mask_kernel

                ys += self._kernel_func(self.mzs, *self.kernel_params[i, :])

                plt.plot(self.mzs[mask_kernel], y[mask_kernel])
            plt.vlines(
                targets,
                ymin=0,
                ymax=self.intensities.max(),
                colors='k',
                linestyles='--',
                label='targets'
            )
            plt.xlabel('m/z in Da')
            plt.ylabel('Intensity')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(ys[mask], label='kernels')
            plt.plot(self.intensities[mask], label='intensities')
            plt.xlabel('broken m/z line')
            plt.xticks([])
            plt.ylabel('Intensity')
            plt.legend()
            plt.show()

        self.bin_spectra(method=method, **kwargs)

    def _get_kernels(self, norm_mode: str = 'area') -> np.ndarray[float]:
        """
        Return matrix in which each column corresponds to the intensities
        of a kernel.

        The area is calculated from kernel parameters.

        Parameters
        ----------
        norm_mode: str, optional
            Factor by which to scale the intensities.
            Options are 'area' and 'height'. Default is area.
        """
        assert check_attr(self, '_kernel_params'), \
            'call set_kernels() first'

        n_peaks: int = self.kernel_params.shape[0]  # number of identified peaks

        if norm_mode == 'area':
            H: float = np.sqrt(2)  # normalization constant
        elif norm_mode == 'height':
            H: float = 1.
        else:
            raise KeyError('norm_mode must be one of "area", "height", "prob"')

        kernels: np.ndarray[float] = np.zeros((n_peaks, len(self.mzs)))
        # TODO: make use of _kernel_func
        if self._kernel_shape == 'bigaussian':
            for idx_peak in range(n_peaks):
                # x_c, H, sigma_l, sigma_r
                sigma_l: float = self.kernel_params[idx_peak, 2]
                sigma_r: float = self.kernel_params[idx_peak, 3]
                kernels[idx_peak] = bigaussian(
                    self.mzs,
                    x_c=self.kernel_params[idx_peak, 0],
                    H=H,  # normalized kernels
                    sigma_l=sigma_l,
                    sigma_r=sigma_r
                )
        elif self._kernel_shape == 'gaussian':
            for idx_peak in range(n_peaks):
                # x_c, H, sigma
                sigma: float = self.kernel_params[idx_peak, 2]
                kernels[idx_peak] = gaussian(
                    self.mzs,
                    x_c=self.kernel_params[idx_peak, 0],
                    H=H,  # normalized kernels
                    sigma=sigma
                )
        return kernels.T

    def bin_spectra(
            self,
            reader: ReadBrukerMCF | hdf5Handler | None = None,
            profile_spectra: np.ndarray[float] | None = None,
            method: str = 'height',
            **_
    ) -> None:
        """
        For each spectrum find overlap between kernels and signal.

        Parameters
        ----------
        reader: ReadBrukerMCF, optional
            Reader to get the spectra. The default is None. Either reader or
            profile_spectra must be provided.
        profile_spectra: np.ndarray[float], optional
            Resampled spectra as 2D matrix. The default is None. Either reader
            or profile_spectra must be provided.
        method: str, optional
            Method for calculating the intensities of target compounds in each
            spectrum:
            -- 'height' takes the height of spectra at target masses
            -- 'area' calculates the overlap between the kernels
            -- 'max' takes the highest intensity within the given tolerance.

        Notes
        -----
        This method defines the following attribute(s):
        _line_spectra : np.ndarray[float]
            The centroided spectra.
        """
        def _bin_spectrum_area(_spectrum: np.ndarray[float], _idx: int) -> None:
            """
            Find intensities of compound based on kernels as the overlap.

            Parameters
            ----------
            _spectrum: np.ndarray[float]
                Intensities of spectrum for which to find integrated peak intensities.
            _idx: int
                Index of spectrum

            Notes
            -----
            weight is the integrated weighted signal
            ideally this would take the integral but since mzs are equally
            spaced, we can use the sum (scaled accordingly), so instead of
            >>> line_spectrum[idx_peak] = np.trapz(weighted_signal, x=self.mzs)
            take
            >>> line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz

            and instead of summing over peaks we can write this as matrix
            multiplication

            equivalent to
            line_spectrum = np.zeros(n_peaks)
            >>> for idx_peak in range(n_peaks):
            >>>     weighted_signal = spectrum.intensities * bigaussians[idx_peak, :]
            >>>     line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
            """
            self._line_spectra[_idx, :] = (_spectrum @ kernels) * dmz

        def _bin_spectrum_height(
                _spectrum: np.ndarray[float], _idx: int
        ) -> None:
            """
            Find intensities of compound based on kernels as the height of
            spectra at kernel centroids.

            Parameters
            ----------
            _spectrum: np.ndarray[float]
                Intensities of spectrum for which to find peak intensities.
            _idx: int
                Index of spectrum

            """
            # pick values of profile spectrum at kernel maxima
            self._line_spectra[_idx, :] = _spectrum[idxs_mzs_c]

        def _bin_spectrum_max(_spectrum: np.ndarray[float], _idx_spectrum: int) -> None:
            # 2D matrix with intensities windowed to kernels: each column is the
            # product of a kernel with the spectrum

            # vectorized
            # vals = kernels[1:-1, :] * _spectrum[:, None]
            # maxs = local_max_fast(vals)
            # # of the local maxima, take the biggest
            # self._line_spectra[_idx_spectrum, :] = maxs.max(axis=0)

            # loop
            # faster than vectorized in this case because we don't have to
            # modify the big kernels matrix
            for peak_idx in range(kernels.shape[1]):
                vals = kernels[:, peak_idx] * _spectrum
                peaks, _ = scipy.signal.find_peaks(vals)
                if len(peaks) == 0:
                    continue
                self._line_spectra[_idx_spectrum, peak_idx] = vals[peaks].max()

        assert len(self._peaks) > 0, 'need at least one peak'
        assert check_attr(self, '_kernel_params'), \
            'calculate kernels with set_kernels'
        assert (profile_spectra is not None) or (reader is not None), \
            'provide either a reader or the profile spectra'
        assert method in ('area', 'height', 'max'), \
            f'method must be either "area" or "height" or "max", not {method}'

        indices_spectra: np.ndarray[int] = self.indices
        n_spectra: int = self._n_spectra  # number of spectra in mcf file
        n_peaks: int = self._n_peaks  # number of identified peaks
        self._line_spectra: np.ndarray[float] = np.zeros((n_spectra, n_peaks))  # result array

        self._binning_by: str = method

        if method == 'area':
            _bin_spectrum: Callable[[np.ndarray[float], int], None] = _bin_spectrum_area
            dmz: float = self.delta_mz
            # precompute (bi)gaussians
            kernels: np.ndarray[float] = self._get_kernels(norm_mode='area')
        elif method == 'height':
            _bin_spectrum: Callable[[np.ndarray[float], int], None] = _bin_spectrum_height
            # indices in mzs corresponding to peak centers
            mzs_c: np.ndarray[float] = self.kernel_params[:, 0]
            idxs_mzs_c: np.ndarray[int] = np.array([
                np.argmin(np.abs(mz_c - self.mzs)) for mz_c in mzs_c
            ])
        elif method == 'max':
            if self._kernel_shape != 'gaussian':
                raise NotImplementedError(
                    f'max only implemented for gaussian kernels, not {self._kernel_shape}'
                )
            _bin_spectrum: Callable[[np.ndarray[float], int], None] = _bin_spectrum_max
            # from taking 1 sigma interval
            # H is 1 since g(x = sigma) is required to be 1
            # from g(x = mu +/- sigma) it follows that g must be bigger than
            # exp(-1 / 2) inside the kernel window
            kernels: np.ndarray[float] = (
                    self._get_kernels(norm_mode='height')
                    > np.exp(-1 / 2)
            ).astype(float)
            # set values next to window boundary to inf to exclude from local max
            n_rows, n_columns = kernels.shape
            # stolen from first_nonzero and last_nonzero
            first_rows = kernels.argmax(axis=0)
            last_rows = kernels.shape[0] - np.flip(kernels, axis=0).argmax(axis=0) - 1
            # convert to 1D indices for easier access
            first_raveled = np.ravel_multi_index(
                np.array([first_rows - 1, range(n_columns)]),
                dims=(n_rows, n_columns)
            )
            last_raveled = np.ravel_multi_index(
                np.array([last_rows + 1, range(n_columns)]),
                dims=(n_rows, n_columns)
            )

            kernels.ravel()[first_raveled] = np.inf
            kernels.ravel()[last_raveled] = np.inf
            # pad for compatibility with shifting (for vectorized version)
            # kernels = np.pad(kernels, ((1, 1), (0, 0)), constant_values=np.inf)
        else:
            raise NotImplementedError()

        # iterate over spectra and bin according to kernels
        for it, idx_spectrum in tqdm(
                enumerate(indices_spectra),
                total=n_spectra,
                desc='binning spectra', smoothing=50 / n_spectra
        ):
            if reader is not None:
                spectrum: np.ndarray[float] = self.get_spectrum(
                    reader=reader, index=idx_spectrum, only_intensity=True
                )
            else:
                spectrum: np.ndarray[float] = profile_spectra[:, it]
            _bin_spectrum(spectrum, it)

    def _add_rxys_to_df(
            self,
            df: pd.DataFrame,
            reader: ReadBrukerMCF | ImagingInfoXML | hdf5Handler | pd.DataFrame | None = None,
            **_
    ) -> pd.DataFrame:
        """
        Add region and spot names to dataframe.

        This method obtains the R, x, y coordinates either from a ReadBrukerMCF or
        ImagingInfoXML object. Unless the ReadBrukerMCF object does not exist already,
        it is much faster to leave the reader object empty and let the method evoke a new
        ImagingInfoXML instance. Indices are matched to spot names automatically.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame on which to append the R, x, and y columns
        reader : ReadBrukerMCF | None, optional
            Reader from which to obtain the spot data. Defaults to evoking new
            ImagingInfoXML object.
        """
        if (reader is not None) and isinstance(reader, ReadBrukerMCF):
            reader.create_spots()
            names: np.ndarray[str] = reader.spots.names
            if not check_attr(reader, 'indices'):
                reader.create_indices()
            imaging_indices = reader.indices
        else:
            reader: pd.DataFrame = get_spots(
                path_d_folder=self.path_d_folder
            )
            names: np.ndarray[str] = reader.spotName

            if isinstance(reader, ImagingInfoXML | ReadBrukerMCF | hdf5Handler):
                imaging_indices = reader.indices
            else:
                assert isinstance(reader, pd.DataFrame)
                imaging_indices = reader.index

        RXYs = get_rxy(names)

        # search indices of spectra object in reader
        if self._n_spectra != len(imaging_indices):
            mask: np.ndarray[bool] = np.array(
                [np.argwhere(idx == imaging_indices)[0][0] for idx in self.indices]
            )
        else:
            mask: np.ndarray[bool] = np.ones_like(self.indices, dtype=bool)
        df['R'] = RXYs[mask, 0]
        df['x'] = RXYs[mask, 1]
        df['y'] = RXYs[mask, 2]

        return df

    def set_feature_table(
            self, integrate_area: bool = False, **kwargs
    ) -> pd.DataFrame:
        """
        Turn the line_spectra into the familiar df with R, x, y columns.

        Heights are taken from the kernel parameters, unless integrate_area is set to True and
        binning was done by area as well. Obtaining the areas from height-binned spectra is
        currently not implemented.

        Parameters
        ----------
        integrate_area: If True, this will return the area of each peak,
            otherwise it will return the height of each peak

        Notes
        -----
        This method defines the following attribute(s):
        feature_table : pd.DataFrame
            Feature table with the spot-wise and compound wise intensities
            together with the R, x, y columns.

        Returns
        -------
        feature_table : pd.DataFrame
            Feature table with the spot-wise and compound wise intensities
            together with the R, x, y columns.
        """
        assert check_attr(self, '_line_spectra'), \
            'create line spectra with bin_spectra'

        if integrate_area:
            assert self._binning_by == 'area', \
                ('currently writing a feature table with area is only possible '
                 'if binning was also done with peak integration.')
            data: np.ndarray[float] = self._line_spectra.copy()
        else:
            data: np.ndarray[float] = self.get_heights()

        df: pd.DataFrame = pd.DataFrame(
            data=data,
            columns=np.around(self.kernel_params[:, 0], 4).astype(str)
        )
        df.loc[:, 'tic_window'] = self._tic

        df: pd.DataFrame = self._add_rxys_to_df(df, **kwargs)
        # drop possible duplicates due to shift in optimizer
        df: pd.DataFrame = df.loc[:, ~df.columns.duplicated()].copy()

        self._feature_table: pd.DataFrame = df
        return self._feature_table

    def require_feature_table(self, **kwargs) -> pd.DataFrame:
        if not check_attr(self, '_feature_table'):
            self.set_feature_table(**kwargs)
        return self._feature_table

    @property
    def feature_table(self):
        return self.require_feature_table()

    def get_kernel_params_df(self) -> pd.DataFrame:
        """Turn the kernel parameters into a feature table."""
        assert check_attr(self, '_kernel_params'), 'call set_kernels'
        if self._kernel_shape == 'bigaussian':
            columns = ['mz', 'H', 'sigma_l', 'sigma_r']
        elif self._kernel_shape == 'gaussian':
            columns = ['mz', 'H', 'sigma']
        else:
            raise NotImplementedError()
        df: pd.DataFrame = pd.DataFrame(data=self.kernel_params, columns=columns)
        return df

    @staticmethod
    def H_from_area(
            area: float | int | np.ndarray[float | int],
            sigma_l: float | int | np.ndarray[float | int],
            sigma_r: float | int | np.ndarray[float | int] | None = None
    ) -> float | np.ndarray[float]:
        """
        Calculate the height of a (bi)gaussian from kernel parameters analytically.

        Parameters
        ----------
        area: float | int | np.ndarray[float | int]
            The area of the kernel.
        sigma_l: float | int | np.ndarray[float | int]
            The left standard deviation for bigaussian or standard deviation for gaussian.
        sigma_r: float | int | np.ndarray[float | int]
            The right standard deviation for bigaussian. The default is None.

        Returns
        -------
        The height(s) of (bi)gaussian(s).

        Notes
        -----
        Theory:
            \int_{-infty}^{infty} H \exp(- (x - x_c)^2 / (2 sigma)^2)dx
               = sqrt(2 pi) H sigma
            => A = H sqrt(pi / 2) (sigma_l + sigma_r)
            <=> H = sqrt(2 / pi) * A* 1 / (sigma_l + sigma_r)
        """
        if sigma_r is None:
            sigma_r = sigma_l
        return np.sqrt(2 / np.pi) * area / (sigma_l + sigma_r)

    def get_heights(self) -> np.ndarray[float]:
        """
        Calculate the peak heights corresponding to the estimated peaks.

        If binning was done by height, returns the line_spectra attribute.
        Otherwise, heights are calculated from areas.
        """
        assert check_attr(self, '_line_spectra'), 'call bin_spectra'
        assert check_attr(self, '_binning_by'), \
            'instance is corrupted, found _line_spectra, but not _binning_by'
        if self._binning_by in ('height', 'max'):
            return self._line_spectra.copy()
        elif self._binning_by == 'area':
            area: np.ndarray[float] = self._line_spectra
            sigma_l: np.ndarray[float] = self.kernel_params[:, 2]
            if self._kernel_shape == 'bigaussian':
                sigma_r: np.ndarray[float] = self.kernel_params[:, 3]
            else:
                sigma_r: None = None
            Hs: np.ndarray[float] = self.H_from_area(area, sigma_l, sigma_r)
            return Hs
        else:
            raise NotImplementedError(
                f'get_heights for {self._binning_by} not implemented'
            )

    def spectrum_idx2array_idx(
            self, spectrum_idx: int | Iterable[int]
    ) -> int | np.ndarray[int]:
        """
        Convert the 1-based spectrum index to 0-based array index.

        Parameters
        ----------
        spectrum_idx : int | Iterable[int]
            Spectrum index or indices to convert (scalar or 1D array-like).

        Return
        ------
        array_idx : int | np.ndarray[int]
            Indices in array that correspond to the spectrum indices.
        """
        if hasattr(spectrum_idx, '__iter__'):
            idxs: list[int] = [np.argwhere(self.indices == idx)[0][0] for idx in spectrum_idx]
            return np.array(idxs)
        else:
            return np.argwhere(self.indices == spectrum_idx)[0][0]

    def set_reconstruction_losses(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            spectrum_idxs: list[int] | None = None,
    ) -> None:
        """
        Obtain the loss of information for each spectrum from the binning.

        Peak areas are integrated based on the assumption that peaks are 
        (bi)gaussian shaped. These assumptions may not always be 
        true in which case the binning may result in information
        loss. This function calculates the difference between the original
        (processed) signals and the one described by the kernels and gives the
        loss in terms of the integrated difference divided by the area of the 
        original signal.

        Parameters
        ----------
        reader: ReadBrukerMCF | hdf5Handler
            Reader to use for getting spectra.
        spectrum_idxs: list[int] | None, optional
            Indices for which to construct the loss. Defaults to all.
        plts: bool, optional
            Whether to plot the reconstructed and original spectra.
        """
        if spectrum_idxs is None:
            spectrum_idxs: np.ndarray[int] = self.indices

        if not check_attr(self, 'losses'):
            self._losses: np.ndarray[float] = np.zeros((self._n_spectra, len(self.mzs)))
        # get sigmas
        sigma_ls: np.ndarray[float] = self.kernel_params[:, 2]
        if self._kernel_shape == 'bigaussian':
            sigma_rs: np.ndarray[float] = self.kernel_params[:, 3]
        else:
            sigma_rs: None = None
        # precompute kernel functions
        kernels: np.ndarray[float] = self._get_kernels(norm_mode='height')
        # loop over spectra
        for c, spectrum_idx in tqdm(
                enumerate(spectrum_idxs),
                total=self._n_spectra,
                desc='Setting losses'
        ):
            # get index in array corresponding to spectrum index
            array_idx: int = self.spectrum_idx2array_idx(spectrum_idx)
            spec: np.ndarray[float] = self.get_spectrum(
                reader=reader, index=spectrum_idx, only_intensity=True
            )

            Hs: np.ndarray[float] = self.H_from_area(
                self._line_spectra[array_idx, :],
                sigma_ls,
                sigma_rs
            )
            y_rec: np.ndarray[float] = kernels @ Hs
            tic = self._tic[c]
            self._losses[c, :] = np.abs(y_rec - spec) / tic if tic else 0

    def filter_line_spectra(
            self,
            binned_snr_threshold: float = 0,
            intensity_min: float = 0,
            success_rate_threshold: float = 0,
            **_: Any
    ) -> np.ndarray[bool]:
        """
        Set the intensities that fall below SNR or min intensity to zero
        in the line_spectra attribute
        and return array of changed pixels.

        Parameters
        ----------
        binned_snr_threshold : float, optional
            Set intensities below this SNR threshold to False.
        intensity_min: float, optional
            Set intensities below this absolute threshold to False.
        success_rate_threshold: float, optional
            Remove compounds with low success rates. Default is 0. For gentle
            filtering a value of .01 is recommended. This filtering step is
            performed after the intensity and snr threshold filtering.

        Returns
        -------
        mask : np.ndarray[bool]
            Mask object where values not meeting the criteria are set to False
        """
        # check if the baseline has been estimated, otherwise can't filter based
        # on noise leveles
        if binned_snr_threshold > 0:
            assert check_attr(self, '_noise_level'), \
                'Call set_noise_level first'
            mask_snr_too_low: np.ndarray[bool] = (self._get_SNR_table()
                                                  < binned_snr_threshold)
        else:
            mask_snr_too_low: np.ndarray[bool] = np.zeros(
                self._line_spectra.shape, dtype=bool
            )
        if intensity_min > 0:
            mask_intensity_too_low: np.ndarray[bool] = (self._line_spectra
                                                        < intensity_min)
        else:
            mask_intensity_too_low: np.ndarray[bool] = np.zeros(
                self._line_spectra.shape, dtype=bool
            )
        # set pixels falling below thresholds to 0
        mask = mask_snr_too_low | mask_intensity_too_low
        self._line_spectra[mask] = 0

        # check which compounds do not have enough successful spectra after
        # filtering
        if success_rate_threshold > 0:
            mask_successes = ((self._line_spectra > 0).mean(axis=0)
                              > success_rate_threshold)
            self._line_spectra = self._line_spectra[:, mask_successes]
            self.filter_peaks(whitelist=self._peaks[mask_successes].copy())

        # update feature table
        if check_attr(self, '_feature_table'):
            self.set_feature_table()

        return mask

    def reconstruct_all(self) -> np.ndarray[float]:
        """Calculate reconstructed profiles from line_spectra and peak properties."""
        kernels: np.ndarray[float] = self._get_kernels(norm_mode='height')
        line_spectra: np.ndarray[float] = self.get_heights()
        reconstructed: np.ndarray[float] = kernels @ line_spectra.T
        return reconstructed

    def combine_with(self, other: Self) -> Self:
        """
        Combine this spectra object with another one. 

        If spectra objects were created from different d_folders,
        the index functionality is lost.

        mz values are inherited from the first object,
        summed intensities are added,
        profiles are stacked,
        noise_levels are averaged,
        peaks and kernels are reevaluated, spectra rebinned.
        Here the kernel-type (bigaussian or gaussian) and whether peaks are integrated are
        inherited by the first object.

        Parameters
        ----------
        other: Self
            Another spectra object.

        Returns
        -------
        s_new : Self
            Spectra object with combined properties.
        """
        assert type(other) is type(self), '"+" is only defined for Spectra objects'
        assert check_attr(self, 'noise_level') and check_attr(other, 'noise_level'), \
            'make sure both objects have the baseline removed'
        assert check_attr(self, '_kernel_params') and check_attr(other, '_kernel_params')
        assert check_attr(self, '_binning_by') and check_attr(other, '_binning_by')

        # determine if spectra are from the same source folder
        is_same_measurement: bool = self.path_d_folder == other.path_d_folder
        if is_same_measurement:
            assert set(self.indices) & set(other.indices) == set(), \
                'spectra objects must not contain the same spectrum twice'
            path_d_folder: str = self.path_d_folder
            indices: np.ndarray[int] = np.hstack([self.indices, other.indices])
        else:
            logger.warning('this and other object are not from the same folder, '
                           'this will result in loss of functionality!')
            path_d_folder: str = os.path.commonpath([
                self.path_d_folder, other.path_d_folder
            ])
            indices: None = None

        s_new: Self = self.__class__(
            limits=self.limits,
            delta_mz=self.delta_mz,
            indices=indices,
            path_d_folder=path_d_folder,
            initiate=False
        )

        # combine spectra
        s_new.mzs = self.mzs.copy()
        s_new.intensities = np.interp(self.mzs, other.mzs, other.intensities) + self.intensities

        self_profiles: np.ndarray[float] = self.reconstruct_all()
        other_profiles: np.ndarray[float] = other.reconstruct_all()
        profiles: np.ndarray[float] = np.hstack([self_profiles, other_profiles])
        # find peaks, bin
        # summed: np.ndarray[float] = profiles.sum(axis=0)
        s_new.noise_level = (np.interp(self.mzs, other.mzs, other.noise_level) + self.noise_level) / 2
        kwargs_peak = self._peak_setting_parameters.copy()
        prominence: float = kwargs_peak.pop('prominence')
        prominence *= 2  # assume both summed spectra have roughly the same prominences
        s_new.set_peaks(prominence=prominence, **kwargs_peak)
        s_new.set_kernels(use_bigaussian=self._kernel_shape == 'bigaussian')
        s_new.bin_spectra(
            profile_spectra=profiles,
            integrate_peaks=self._binning_by == 'area'
        )
        return s_new

    def add_calibrated_spectra(self, reader: ReadBrukerMCF | hdf5Handler, **kwargs: Any):
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(**kwargs)
        self.require_calibration_functions(reader=reader, **kwargs)
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(**kwargs)

    def full(self, reader: ReadBrukerMCF | hdf5Handler, **kwargs: Any):
        """Perform all steps with the provided parameters."""
        self.add_calibrated_spectra(reader=reader, **kwargs)
        self.set_peaks(**kwargs)
        self.filter_peaks(**kwargs)
        self.set_kernels(**kwargs)
        self.bin_spectra(reader, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.set_feature_table(**kwargs)

    def full_targeted(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            targets: list[float],
            **kwargs
    ) -> None:
        """Perform all steps for targeted compounds with the provided parameters."""
        self.add_calibrated_spectra(reader=reader, **kwargs)
        # set target compounds
        self.set_targets(targets, reader=reader, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.set_feature_table(**kwargs)

    def plot_summed(
            self,
            plt_kernels: bool = False,
            plt_lines: bool = False,
            limits: tuple[float | int, float | int] | None = None,
            hold: bool = False,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None
    ) -> tuple[plt.Axes, plt.Axes] | None:
        """
        Plot the summed up intensities with synthetic spectrum estimated
        from kernel parameters, if determined.

        This method plots the summed intensities. If 'set_kernel_params' has
        been called already, the synthetic spectrum will be plotted as well,
        the loss denotes the area between the original and synthetic data.

        Parameters
        ----------
        plt_kernels: bool, optional
            The default is False. If kernel parameters have been determined,
            this option becomes available. It is generally not recommended to
            plot individual kernels and intended for debugging. Unless your
            mass window or number of kernels is fairly small, this will take a
            long time to plot.
        plt_lines: bool, optional
            If spectra have been binned, this option will plot vertical lines
            at the peak centers where there height corresponds to the summed
            intensity across all spectra.
        limits: tuple[float] | None, optional.
            By default, the entire mass range is plotted. With this parameter
            it can be decreased.
        """
        # calculate approximated signal by summing up kernels
        if plt_reconstructed := check_attr(self, '_kernel_params'):
            kernels = self._get_kernels(norm_mode='height')
            intensities_approx = (kernels * self.kernel_params[:, 1]).sum(axis=1)
            loss = np.sum(np.abs(self.intensities - intensities_approx)) \
                / np.sum(self.intensities)

        if fig is None:
            assert ax is None
            fig, ax = plt.subplots()

        if plt_kernels and plt_reconstructed:
            for i in range(len(self._peaks)):
                y: np.ndarray[float] = self._kernel_func(
                    self.mzs, *self.kernel_params[i, :]
                )
                # reduce to 10 std
                mask: np.ndarray[bool] = np.abs(
                    self.kernel_params[i, 0] - self.mzs
                ) <= 10 * self.kernel_params[i, -1]
                ax.plot(self.mzs[mask], y[mask])
        ax.plot(self.mzs, self.intensities, label='summed intensity')
        if plt_reconstructed:
            ax.plot(self.mzs, intensities_approx, label='estimated')
        if check_attr(self, '_binning_by') and plt_lines:
            ax.stem(self.kernel_params[:, 0], self.get_heights().sum(axis=0),
                     markerfmt='', linefmt='red')
        if limits is not None:
            ax.set_xlim(limits)
            mask: np.ndarray[bool] = (self.mzs >= limits[0]) & (self.mzs <= limits[1])
            ax.set_ylim((0, self.intensities[mask].max()))
        ax.legend()
        ax.set_xlabel(r'$m/z$ in Da')
        ax.set_ylabel('Intensity')
        if plt_reconstructed:
            ax.set_title(f'Reconstructed summed intensities (loss: {loss:.1f})')

        if not hold:
            plt.show()
        else:
            return fig, ax

    def plot_tic(self):
        fig, ax = plt.subplots()
        ax.plot(self.indices, self._tic)
        ax.set_xlabel('Index')
        ax.set_ylabel('TIC')
        plt.show()

    def plot_calibration_functions(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            indices: Iterable[int] | None = None,
            n_plot: int = 10
    ) -> None:
        assert check_attr(self, '_calibration_parameters'), \
            'Call set_calibrate_functions before plotting.'
        if indices is not None:
            n_plot = len(indices)
        assert (n_plot > 0), 'Number of spectra should be bigger than 0'

        if indices is None:
            indices: np.ndarray[int] = np.random.choice(
                self.indices, size=n_plot, replace=False
            )

        fig, axs = plt.subplots(nrows=n_plot, ncols=2, sharex='col')
        if n_plot == 1:
            axs = [axs]

        calibrants_mz: np.ndarray[float] = get_calibrants(self.limits)

        for idx, (ax_l, ax_r) in zip(indices, axs):
            # obtain spectra
            spec_o: Spectrum = reader.get_spectrum(index=idx, limits=self.limits)
            spec_c: Spectrum = self.get_spectrum(
                reader=reader, index=idx, only_intensity=False
            )

            array_idx: int = self.spectrum_idx2array_idx(idx)
            poly_coeffs: np.ndarray = self._calibration_parameters[array_idx, :]
            f: Callable = np.poly1d(poly_coeffs)
            cal_vals: np.ndarray[float] = f(self.mzs)
            ax_l.plot(self.mzs, self.noise_level)
            ax_l.plot(spec_o.mzs, spec_o.intensities)
            ax_l.plot(spec_c.mzs, spec_c.intensities)
            ax_l.vlines(calibrants_mz, 0, spec_o.intensities.max(), colors='k', linestyles='--')

            ax_r.plot(self.mzs, cal_vals * 1e3)
        ax_l.set_xlabel('m/z in Da')
        ax_r.set_xlabel('m/z in Da')
        ax_l.legend(['noise lvl', 'original', 'calibrated', 'calibrants', 'verification'])
        ax_r.legend(['shift in mDa'])

        fig.tight_layout()
        plt.show()

    def plot_losses(self, n_max_spectra=1000, n_max_masses=1000):
        # summed at top
        # TIC at left side
        # heat map in middle
        # spectra-wise loss at right
        assert check_attr(self, '_losses'), 'call set_reconstruction_losses() first'

        # Create figure and main axis for the heatmap
        fig, ax = plt.subplots(layout='compressed', figsize=(15, 12))

        # we can only plot a selection
        n_spectra = min([n_max_spectra, self._n_spectra])
        n_masses = min([n_max_masses, len(self.mzs)])
        every_spectra = round(self._n_spectra/n_spectra)
        every_masses = round(len(self.mzs) / n_masses)

        losses = self._losses[::every_spectra, ::every_masses]

        # Plot heatmap
        ax.imshow(
            np.log(losses),
            aspect='auto',
            cmap='viridis',
            extent=[
                self.mzs[0],
                self.mzs[-1],
                self.indices.max(),
                self.indices.min()
            ],
            origin='upper'
        )

        # Create additional axes for the functions
        top_ax = ax.inset_axes([0, 1, 1, 0.2])
        bottom_ax = ax.inset_axes([0, -0.2, 1, 0.2])
        left_ax = ax.inset_axes([-0.2, 0, 0.2, 1])
        right_ax = ax.inset_axes([1, 0, 0.2, 1])

        # Plot the functions on each axis
        # normalized to total intensity of spectrum
        spectra_wise_loss = np.trapz(self._losses, dx=self.delta_mz, axis=1)
        # average loss at masses, normalized to average intensity
        mass_wise_loss = np.nanmedian(self._losses, axis=0)

        top_ax.plot(self.mzs, self.intensities, color='C0')
        bottom_ax.plot(self.mzs, mass_wise_loss, color='C1')
        left_ax.plot(
            self._tic, self.indices, color='C0'
        )
        right_ax.plot(spectra_wise_loss, self.indices, color='C1')

        # Hide x and y ticks for the function plots
        top_ax.set_xticks([])
        top_ax.set_title('summed intensities')

        bottom_ax.set_title('mass-wise loss', y=-0.3)

        left_ax.set_title('spectra-wise TIC')
        left_ax.set_ylabel('index')

        right_ax.set_yticks([])
        right_ax.set_title('spectra-wise loss')

        # Adjust the limits to match the heatmap _extent
        top_ax.set_xlim(ax.get_xlim())
        bottom_ax.set_xlim(ax.get_xlim())
        left_ax.set_ylim(ax.get_ylim())
        right_ax.set_ylim(ax.get_ylim())

        # Show plot
        plt.show()

        # distribution
        plt.hist(
            self._losses[~np.isnan(self._losses)],
            bins=np.linspace(
                np.nanmin(self._losses),
                np.nanquantile(self._losses, .95),
                1000
            )
        )
        plt.title(f'Distribution of losses (median = {np.nanmedian(self._losses):.2f})')
        plt.xlabel('Normed loss')
        plt.ylabel('Count')
        plt.show()


class ClusteringManager:
    """
    Handle Spectra in clusters (e.g. regions of interest) for more precise
    peak-picking. Also, more RAM-light.

    Currently not tested, usage is discouraged.
    """
    def __init__(self, reader: ReadBrukerMCF, **kwargs_spectra):
        assert check_attr(reader, 'indices'), 'call create_indices'
        assert check_attr(reader, 'limits'), \
            'call set_casi_window or define mass limits'
        self.path_d_folder: str = reader.path_d_folder
        self.indices = reader.indices
        self.N_spectra = len(self.indices)
        self.limits = reader.limits
        # create dummy object
        dummy: Spectra = Spectra(
            reader=reader, limits=self.limits, **kwargs_spectra
        )
        self.ram_spectrum: int = dummy.mzs.nbytes + dummy.intensities.nbytes
        self.kwargs_spectra: dict = kwargs_spectra

    def set_clusters(
        self,
        ram_GB: float | None = None,
        min_chunk_size: int = 100,
        max_chunk_size: int | None = None,
        N_chunks: int = None,
        method: str = 'random'
    ) -> tuple[int, int]:
        if ram_GB is None:
            ram_free = psutil.virtual_memory().total * .8
        else:
            # convert to bytes
            ram_free: float = ram_GB * 1024 ** 3
        # number of spectra that can be hold in memory at one time
        max_chunk_size_cal: int = int(ram_free / self.ram_spectrum)
        if (max_chunk_size is not None) and (max_chunk_size > max_chunk_size_cal):
            logger.warning(
                'chunk size likely too large to be held in memory, this ' +
                'will result in significant performance drop, the maximum ' +
                'recommended chunk size with the specified sample rate is ' +
                f'{max_chunk_size_cal}'
            )
        elif max_chunk_size is None:
            max_chunk_size: int = max_chunk_size_cal
        if max_chunk_size_cal < min_chunk_size:
            logger.warning(
                f'the requested minimum chunk size {min_chunk_size} is bigger ' +
                f'than the recommended max {max_chunk_size_cal}'
            )

        N_chunks_cal: int = np.ceil(self.N_spectra / max_chunk_size).astype(int)
        if (N_chunks is not None) and (N_chunks < N_chunks_cal):
            logger.warning(
                'chunk size likely too large to be held in memory, this ' +
                'will result in significant performance drop, the maximum ' +
                'recommended chunk number with the specified sample rate is ' +
                f'{N_chunks_cal}'
            )
        elif N_chunks is None:
            N_chunks = N_chunks_cal
        self.N_chunks: int = N_chunks

        self._set_clusters(method)

    def read_sql(self) -> None:
        sql_tables = get_sql_files(self.path_d_folder)
        intensities = sql_tables['intensities']
        mzs = sql_tables['mzs']

    def _set_clusters(self, method):
        # get spectra info from sql
        # cluster into N chunks
        if method == 'random':
            indices_shuffled = self.indices.copy()
            np.random.shuffle(indices_shuffled)
            chunks: list[np.ndarray[int]] = np.array_split(
                indices_shuffled, self.N_chunks
            )
            self.clusters: dict[int, np.ndarray[int]] = dict(zip(
                range(self.N_chunks), chunks
            ))
        elif method == 'similarity':
            self.get_sql_similiarities()
        else:
            raise NotImplementedError()

    def plt_cluster_distribution(self):
        assert check_attr(self, 'clusters')
        info: ImagingInfoXML = ImagingInfoXML(path_d_folder=self.path_d_folder)
        names: np.ndarray[str] = info.spotName
        idxs: np.ndarray[int] = info.count
        xs = np.array([int(name.split('X')[1].split('Y')[0]) for name in names])
        ys = np.array([int(name.split('Y')[1]) for name in names])
        # map indices to colors
        cs = np.zeros_like(idxs)
        for i, c in self.clusters.items():
            cs[c - 1] = i

        df = pd.DataFrame({'x': xs, 'y': ys, 'c': cs}, index=idxs)
        img = df.pivot(index='y', columns='x', values='c')
        plt.figure()
        plt.imshow(img, interpolation='None')
        plt.xlabel('data x coordinate')
        plt.ylabel('data y coordinate')
        plt.show()

    def get_spectra(self, reader: ReadBrukerMCF, **kwargs: dict) -> Spectra:
        assert check_attr(self, 'clusters'), 'call set_clusters first'
        spec_main = Spectra(
            reader=reader, indices=self.clusters[0], **self.kwargs_spectra
        )
        spec_main.full(reader, **kwargs)
        for i, c in self.clusters.items():
            if i == 0:
                continue
            spec = Spectra(reader=reader, indices=c)
            spec.full(reader, **kwargs)
            spec_main.combine_with(spec)

        return spec_main


class MultiSectionSpectra(Spectra):
    """
    Class for handling processing of multiple spectra objects at the same time.

    Usage similar to spectra class. This allows to bin spectra in the same bins across
    different measurements, ensuring feature tables can be aligned.

    Example
    -------
    Imports
    >>> from maspim.exporting.from_mcf.spectrum import MultiSectionSpectra
    >>> from maspim.exporting.from_mcf.rtms_communicator import ReadBrukerMCF
    Initiate readers (could also be hdf5Handler's)
    >>> folders = ['path/to/d_folder1.d', 'path/to/d_folder2.d', ...]
    >>> readers = [ReadBrukerMCF(path_d_folder=folder) for folder in folders]
    Initiate MultiSectionSpectra
    >>> ms = MultiSectionSpectra(readers=readers)
    From here on same steps as in Spectra, except pass readers instead of reader.
    Note, however, that it is necessary to call
    >>> ms.distribute_peaks_and_kernels()
    To transmit the kernel parameters to the spectra children after calling
    set_kernels or set_targets!
    """
    def __init__(
            self, 
            readers: list[ReadBrukerMCF | hdf5Handler]
    ) -> None:
        """
        Initialization.

        Initializes all the spectra.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to create spectra objects

        """
        self.specs: list[Spectra | None] = [None] * len(readers)
        self._initiate(readers)

    def _initiate(self, readers: list[ReadBrukerMCF | hdf5Handler]) -> None:
        """Set mzs and indices"""
        assert len(readers) > 0, 'pass at least one reader'
        reader = readers[0]
        assert all([r.limits[0] == reader.limits[0] for r in readers]), \
            'readers must have the same limits'
        assert all([r.limits[1] == reader.limits[1] for r in readers]), \
            'readers must have the same limits'

        indices: list[np.ndarray[int]] = []
        offset = 0
        for i, reader in enumerate(readers):
            spec: Spectra = Spectra(reader=reader)
            self.specs[i] = spec.copy()
            idxs = spec.indices.copy()
            idxs += offset
            indices.append(idxs)
            offset = idxs[-1]
        self.mzs: np.ndarray[float] = spec.mzs.copy()
        self.delta_mz = spec.delta_mz
        self.intensities: np.ndarray[float] = np.zeros_like(self.mzs)
        self.indices = np.hstack(indices)

    def add_all_spectra(self, readers: list[ReadBrukerMCF | hdf5Handler]) -> None:
        """
        Iterate over spectra by calling the corresponding Spectra method.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.require_calibration_functions
        """
        for i, reader in enumerate(readers):
            self.specs[i].add_all_spectra(reader)
            self.intensities += self.specs[i].intensities

    def subtract_baseline(self, **kwargs) -> None:
        self.noise_level = np.zeros_like(self.intensities)
        for spec in self.specs:
            spec.subtract_baseline(**kwargs)
            self.intensities -= spec.noise_level * len(spec.indices)
            self.noise_level += spec.noise_level
        self.noise_level /= len(self.specs)

    def require_calibration_functions(
            self,
            *,
            readers: list[ReadBrukerMCF | hdf5Handler] | None = None,
            **kwargs
    ) -> None:
        """
        Iterate over spectra by calling the corresponding Spectra method.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.require_calibration_functions
        """
        for i, reader in enumerate(readers):
            self.specs[i].require_calibration_functions(reader=reader, **kwargs)

    def distribute_peaks_and_kernels(self):
        """
        Transmit the peak and kernel properties of this object to the Spectra children
        """
        assert check_attr(self, '_peaks'), 'call set_peaks() first'
        assert check_attr(self, '_kernel_params'), 'call set_kernel_params() first'
        for spec in self.specs:
            spec.peaks = self.peaks.copy()
            spec.peak_properties = self.peak_properties.copy()
            spec.peak_setting_parameters = self.peak_setting_parameters.copy()
            spec.peak_setting_parameters['notes'] = 'peak properties set in MultiSectionSpectra'
            spec.kernel_params = self.kernel_params.copy()
            spec.kernel_shape = self.kernel_shape
            if check_attr(self, 'noise_level'):
                spec.noise_level = self.noise_level

    def bin_spectra(self, readers: Iterable[ReadBrukerMCF | hdf5Handler], **kwargs):
        """
        Iterate over spectra by calling the corresponding Spectra method.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.require_calibration_functions
        """
        line_spectra = []
        for i, reader in enumerate(readers):
            self.specs[i].bin_spectra(reader, **kwargs)
            line_spectra.append(self.specs[i].line_spectra.copy())
        self.line_spectra: np.ndarray[float] = np.vstack(line_spectra)
        self.binning_by = self.specs[0].binning_by

    def set_feature_table(
            self,
            readers: Iterable[ReadBrukerMCF | hdf5Handler] | None = None,
            **kwargs: dict
    ) -> pd.DataFrame:
        """
        Iterate over spectra by calling the corresponding Spectra method.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.require_calibration_functions
        """
        if readers is None:
            readers = [None] * len(self.specs)

        fts = []
        for i, reader in enumerate(readers):
            self.specs[i].set_feature_table(reader=reader, **kwargs)
            fts.append(self.specs[i].feature_table)

        self._feature_table: pd.DataFrame = combine_feature_tables(fts)
        return self._feature_table

    def full(self, readers: list[ReadBrukerMCF | hdf5Handler], **kwargs: dict):
        """
        Perform all steps to process and bin the spectra.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.require_calibration_functions
        """
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(**kwargs)
        self.require_calibration_functions(readers=readers, **kwargs)
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(overwrite=True, **kwargs)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        self.distribute_peaks_and_kernels()
        self.bin_spectra(readers=readers, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.set_feature_table(readers=readers, **kwargs)

    def full_targeted(
            self,
            readers: list[ReadBrukerMCF | hdf5Handler],
            targets: list[float],
            **kwargs
    ):
        """
        Perform all steps to process and bin the spectra for targeted compounds.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        targets: list[float]
            Target mzs.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.require_calibration_functions
        """
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(**kwargs)
        self.require_calibration_functions(readers=readers, **kwargs)
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(overwrite=True, **kwargs)
        # set target compounds
        self.set_targets(targets, **kwargs, plts=True)
        self.distribute_peaks_and_kernels()
        self.bin_spectra(readers=readers, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.set_feature_table(readers=readers, **kwargs)
