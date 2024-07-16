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
import pickle
import psutil
import logging

from tqdm import tqdm
from typing import Iterable, Self, Any, Callable
from scipy.signal import find_peaks, correlate, correlation_lags, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter, median_filter

from msi_workflow.data.combine_feature_tables import combine_feature_tables
from msi_workflow.exporting.from_mcf.rtms_communicator import ReadBrukerMCF, Spectrum
from msi_workflow.exporting.sqlite_mcf_communicator.hdf import hdf5Handler
from msi_workflow.exporting.sqlite_mcf_communicator.sql_to_mcf import get_sql_files
from msi_workflow.exporting.from_mcf.helper import get_mzs_for_limits
from msi_workflow.res.calibrants import get_calibrants
from msi_workflow.util import Convinience
from msi_workflow.util.manage_obj_saves import class_to_attributes
from msi_workflow.project.file_helpers import ImagingInfoXML, get_rxy

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


class Spectra(Convinience):
    """
    Container for multiple Spectrum objects and binning.

    This class is used to access Bruker .d folders with MSI data through the R rtms package.
    It is recommended to create an ReadBrukerMCF reader once, read all the spectra in and store
    a hdf5handler, as this is much faster, if the data is read in again, which is usually required.

    Example Usage
    -------------
    >>> from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
    >>> from exporting.from_mcf.cSpectrum import Spectra
    Create a reader
    >>> reader = ReadBrukerMCF('path/to/your/d_folder.d')
    Providing the reader is only necessary the first time an object is instanciated for a folder,
    afterward a reader will be initiated from the stored hdf5file automatically, when used through
    the get_project class (recommended). This guide only discusses steps when used on its own. For
    more info, take a look at the hdf5handler and get_project classes.
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
    >>> spec.plt_summed()

    The next step is to subtract the baseline. Here a minimum filter is used. It is crucial to
    find the right window size: If it is too small, peaks will lose intensity because the minimum
    filter climbs up the flanks of peaks. If it is too large, the noise level will not be removed
    enterily. By default, the window_size is being estimated from the data by taking the broadest peak
    at a relative height of .8 as the window size. The result can be checked by setting the plts
    keyword to True.
    >>> spec.subtract_baseline(plts=True)
    
    Optionally, it is possible to perform lock mass calibration, e.g. with Pyropheophorbide a:
    >>> spec.set_calibrate_functions(calibrants_mz=[557.25231],reader=reader,SNR_threshold=2)
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
    >>> spec.filter_peaks(SNR_threshold=2, remove_sidepeaks=True, plts=True)

    The next step is to set kernels
    >>> spec.set_kernels()
    It is possible to plot this as well, but due to the large number of kernels this will usually
    not be a pleasant experience and without the keyword only the summed intensity of the kernels
    will be plotted
    >>> spec.plt_summed(plt_kernels=True)

    Then, spectra are binned using the kernels (this step will also take a while because every spectrum has
    to be read in again, but this time the hdf5 file can be used if the get_project class was used, which is a lot faster)
    >>> spec.bin_spectra()

    Finally, a feature table is created from the binned spectra
    >>> spec.binned_spectra_to_df()
    """

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
        reader : ReadBrukerMCF | hdf5handler | None, optional
            Reader to obtain metadata and spectra from disk. The default is None.
        limits : tuple[float], optional
            mz range of spectra. The default is None. This defaults to setting the limits with
            parameters from the QTOF.
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
        self.delta_mz = delta_mz
        if initiate:
            self._initiate(reader, indices, limits)
        else:
            self.indices: np.ndarray[int] = np.array(indices)
            self.limits: tuple[float, float] = limits

    def _set_files(
            self,
            reader: ReadBrukerMCF | hdf5Handler | None,
            path_d_folder: str | None,
            initiate: bool,
    ) -> bool:
        assert (reader is not None) or (path_d_folder is not None), \
            'Either pass a reader or load and the corresponding d-folder'
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
            f"Reader must be either a hdf5Handler or ReadBrukerMCF instance. You provided {type_reader}"
        assert (indices is None) or (len(indices) > 0), \
            f"indices must either be None or of non-zero length. You provided {indices}."
        assert (limits is None) or ((len(limits) == 2) and limits[0] < limits[1]), \
            (f'limits must either be None or contain an upper and lower bound with upper != lower, you' +
             f'provided {limits}.')

        if indices is None:
            if not hasattr(reader, 'indices'):
                reader.create_indices()
            indices = reader.indices
        self.indices = np.array(indices)
        if limits is None:
            if (not hasattr(reader, 'metaData')) and is_rtms:
                reader.set_meta_data()
            if not hasattr(reader, 'limits'):
                reader.set_QTOF_window()
            limits = reader.limits
        self.limits = limits

        if is_rtms:
            self.mzs = get_mzs_for_limits(self.limits, self.delta_mz)
            reader.set_mzs(self.mzs)
        else:
            self.mzs = reader.mzs
        self.intensities = np.zeros_like(self.mzs)

    def add_spectrum(self, spectrum: np.ndarray[float]):
        """Add passed spectrum values to summed spectrum."""
        # spectrum = spectrum.copy()
        self.intensities += spectrum

    def _get_spectrum(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            index: int | str,
            only_intensity: bool,
            **kwargs
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
        **kwargs : dict
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
        calibrate: bool = hasattr(self, 'calibration_parameters')

        if calibrate:
            array_idx: int = self.spectrum_idx2array_idx(index)
            poly_coeffs: np.ndarray = self.calibration_parameters[array_idx, :]
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

        N: int = len(self.indices)
        logger.info(f'adding up {N} spectra ...')

        self.intensities: np.ndarray[float] = np.zeros_like(self.mzs)

        if not hasattr(reader, 'mzs'):
            reader.set_mzs(self.mzs)


        # iterate over all spectra
        for index in tqdm(self.indices, desc='Adding spectra', smoothing=50/N):
            spectrum: np.ndarray[float] = self._get_spectrum(reader=reader, index=index, only_intensity=True)
            self.add_spectrum(spectrum)

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
        for it, index in enumerate(self.indices):
            spectrum: np.ndarray[float] = self._get_spectrum(reader=reader, index=index, only_intensity=True)
            if it > 0:
                shift: float = self.xcorr(spectrum)  # shift in Da
                # shift according to number of spectra
                weight: float = 1 / (it + 1)
                self.mzs += shift * weight
                spectrum.mzs -= shift * (1 - weight)
            self.add_spectrum(spectrum.intensities)

    def subtract_baseline(
            self,
            window_size: float | int | None = None,
            overwrite: bool = False,
            plts: bool = False,
            **_: dict
    ) -> None:
        """
        Estimate and remove the noise level from the summed intensities.

        The noise level is stored as an attribute called 'noise_level'.
        It is crucial to find the right window size: If it is too small, peaks will lose intensity because
        the minimum filter climbs up the flanks of peaks. If it is too large, the noise level will
        not be removed enterily. By default, the window_size is being estimated from the data by
        taking the broadest peak at a relative height of .8 as the window size. The result can
        be checked by setting the plts keyword to True. The result of the minimum filter is passed on
        to a median filter of the same window size to smooth the output.

        Parameters
        ----------
        window_size : float | int, optional
            The window size of the minimum filter as number of sample point. 
            None defaults to estimating the window size from the peak widths. 
            A window_size of 0 will subtract the smallest intensity.
            A window size in the interval (0, 1) is interpreted as the fraction 
            of the entire window size.
        plts: bool, optional
            The default is False. If True, will plot the summed intensities before and after baseline
            removal.

        Notes
        -----
        This method defines the following attribute(s):
        noise_level: np.ndarray
            The noise level of each spectrum. This assumes that each spectrum has the same noise level.
            It is defined as the estimated bsae_line divided by the number of spectra.

        """
        def estimate_peaks_width() -> int:
            """Estimate the minimum filter size from the peak widths."""
            prominence = .1 * np.median(self.intensities)
            peaks, peak_props = find_peaks(self.intensities, prominence=prominence, width=3)
            widths, *_ = peak_widths(self.intensities, peaks=peaks, rel_height=.8)
            return int(np.max(widths))
        
        assert np.any(self.intensities), \
            'call add_all_spectra before subtracting the baseline'

        if hasattr(self, 'noise_level') and (not overwrite):
            logger.warning('found a noise level, exiting method')
            return
        
        N: int = len(self.indices)

        if window_size is None:
            # estimate peak width from FWHM
            window_size: int = estimate_peaks_width()
            logger.info(
                'estimated window size for baseline subtraction is ' +
                f'{self.delta_mz * window_size * 1e3:.1f} mDa'
            )
        elif window_size == 0:  # subtract minimum
            base_lvl: float = self.intensities.min()
            self.intensities -= base_lvl
            self.noise_level: np.ndarray[float] = np.full_like(self.intensities, base_lvl / N)
            return
        # convert Da to number of sample points
        elif (window_size < 1) and isinstance(window_size, float):
            dmz: float = self.mzs[1] - self.mzs[0]
            window_size: int = int(window_size / dmz + .5)
        ys_min: np.ndarray[float] = minimum_filter(self.intensities, size=window_size)
        # run median filter on that
        ys_min:np.ndarray[float] = median_filter(ys_min, size=window_size)
        # store for SNR estimation
        self.noise_level: np.ndarray[float] = ys_min / N

        if plts:
            plt.figure()
            plt.plot(self.mzs, self.intensities, label='signal')
            plt.plot(self.mzs, self.intensities - ys_min, label='baseline corrected')
            plt.plot(self.mzs, ys_min, label='baseline')
            plt.xlabel('m/z in Da')
            plt.ylabel('Intensity')
            plt.legend()
            plt.show()
            
        self.intensities -= ys_min

    def xcorr(
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
        N: int = len(b)

        lags: np.ndarray[float] = correlation_lags(N, N, mode='full')
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

    def set_targets(
            self,
            targets: Iterable[float],
            tolerances: Iterable[float] | float | None = None,
            method: str = 'nearest_peak',
            plts: bool = False,
            **kwargs: dict
    ) -> None:
        """
        Set the peaks based on a number of target compounds (mass given in Da). 
        Peaks are searched within a mass window defined by tolerances, such that the
        assigned mass is within target +/- tolerance. If no tolerance is given,
        it is estimated from the kernel widths.

        This function sets the kernel_params and then requires using the bin_spectra function.

        Parameters
        ----------
        targets: Iterable[float]
            The m/z values of interest in Da
        tolerances: Iterable[float] | float
            Tolerance(s) for deviation of peaks from theoretical masses.
        method: str, optional
            Method for calculating the intensities of target compounds in each spectrum:
            -- 'nearest_peak' searches the closest peak in the summed up spectrum
                and estimates the kernel shape of that
            -- 'area_overlap' calculates the overlap between the kernels
                (calculated from the tolerance where the tolerance is assumed to be
                the standard deviation).
            -- 'highest' takes the highest intensity within the given tolerance.
                Currently not implemented.
        plts: bool, optional
            Whether to plot the target kernels on top of the summed spectrum.
        kwargs: dict
            Options passed on to set_peaks
        """
        # assertions
        assert method in (methods := ('nearest_peak', 'area_overlap', 'highest')), \
            f'methods must be in {methods}, not {method}'
        has_summed_spectra: bool = np.any(self.intensities.astype(bool))
        if method == 'nearest_peak':
            assert has_summed_spectra, \
                'nearest method requires summing up the spectra'
        if tolerances is None:
            assert has_summed_spectra, \
                'estimating the tolerance requires that the spectra have been summed up'
        if plts:
            assert not np.all(self.intensities == 0), \
                'if you want to plot the target kernels, add up the intensities first'

        if tolerances is None:
            logger.info('estimating tolerance from summed spectrum ...')
            self.set_peaks(**kwargs)
            self.set_kernels(**kwargs)
            tolerances: float = np.median(self.kernel_params[:, -1])
            logger.info(f'found tolerance of {tolerances*1e3:.1f} mDa')
        if isinstance(tolerances, float | int):
            tolerances: np.ndarray[float] = np.full_like(targets, tolerances)
        assert len(tolerances) == len(targets), \
            'If widths is not a scalar, widths and targets must have the same number of elements.'

        N_peaks: int = len(targets)  # number of peaks equal to targets
        N_spectra: int = len(self.indices)
        self.line_spectra: np.ndarray[float] = np.zeros((N_spectra, N_peaks))
        if method == 'area_overlap':
            self.peaks = [np.argmin(np.abs(self.mzs - target)) for target in targets]
            self.peak_setting_parameters['method'] = method
            self.peak_setting_parameters['targets'] = np.array(targets)
            self.peak_setting_parameters['tolerances'] = np.array(tolerances)
            self.kernel_params = np.zeros((N_peaks, 3 + (self.kernel_shape == 'bigaussian')))  # gaussian
            self.kernel_params[:, 0] = targets  # center
            self.kernel_params[:, 1] = self.intensities[self.peaks]
            self.kernel_params[:, 2] = tolerances
        elif method == 'nearest_peak':
            if not hasattr(self, 'peaks'):
                self.set_peaks(**kwargs)
            # filter peaks to keep only the closest one to each target
            idxs_keep: list[int] = []
            dists_keep: list[float] = []
            for idx, target in enumerate(targets):
                dists: np.ndarray[float] = np.abs(self.mzs[self.peaks] - target)
                if not np.any(dists < tolerances[idx]):
                    logger.warning(f'did not find peak for {target=}')
                    continue
                idx_keep: int = np.argmin(dists)
                idxs_keep.append(idx_keep)
                dists_keep.append(dists[idx_keep])
            peaks_keep = self.peaks[idxs_keep].copy()
            # delete irrelevant peaks
            self.filter_peaks(whitelist=peaks_keep)
            self.peak_setting_parameters['method'] = method
            self.peak_setting_parameters['targets'] = np.array(targets)
            self.peak_setting_parameters['tolerances'] = np.array(tolerances)
            self.peak_setting_parameters['distances'] = np.array(dists_keep)
            self.set_kernels()
        elif method == 'highest':
            # TODO: <--
            ...
            raise NotImplementedError

        if plts:
            plt.figure()
            plt.plot(self.mzs, self.intensities, label='original')
            for i in range(len(self.peaks)):
                y = self._kernel_func(self.mzs, *self.kernel_params[i, :])
                mask = np.abs(self.kernel_params[i, 0] - self.mzs) <= 10 * self.kernel_params[i, -1]
                plt.plot(self.mzs[mask], y[mask])
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
        if prominence < 1:
            med: float = np.median(self.intensities)
            prominence *= med

        self.peaks, self.peak_properties = find_peaks(
            self.intensities, prominence=prominence, width=width
        )

        # save parameters to dict for later reference
        self.peak_setting_parameters: dict[str, Any] = kwargs
        self.peak_setting_parameters['prominence'] = prominence
        self.peak_setting_parameters['width'] = width

    def _check_calibration_file_exists(self):
        """
        Look for Calibrator.ami file in d folder.

        Lock mass calibration always creates this file, so this can be used as an indicator.
        """
        # calibration always creates an ami file
        if os.path.exists(os.path.join(self.path_d_folder, 'Calibrator.ami')):
            return True
        return False

    def set_calibrate_functions(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            calibrants_mz: Iterable[float] = None,
            search_range: float = 5e-3,
            SNR_threshold: float = 4,
            max_degree: int = 1,
            min_height: float | int = 10_000,
            **_
    ) -> None:
        """
        Calibrate spectra using calibrants by fitting a polynomial of degree max_degree or less.

        This algorithm matches the cloesest peak fulfilling the criteria (search
        range and SNR_threshold or min_height) to the theoretical masses.
        A polynomial of at most degree max_degree is fitted to the differences
        from the closest peak to theoretical masses. If not enough peaks are found,
        the degree of the polynomial will be lowered. If no peak is found, the
        spectrum will not be calibrated.

        Parameters
        ----------
        calibrants_mz : float | Iterable[float] | None, optional
            Exact mass(es) of the calibrants in Da. If not provided, will use the calibrant list from
            Wörmer et al., 2019 (Towards multiproxy, ultra-high resolution molecular stratigraphy:
            Enabling laser-induced mass spectrometry imaging of diverse molecular biomarkers in sediments, Appendix)
        search_range : float, optional
            Range in which to look for peaks in Da. The default is 5 mDa. This will look in a range of
            +/- 5 mDa around the theoretical mass.
        SNR_threshold : float, optional
            Minimal prominence required for a peak to be considered (see prominence in set_peaks).
            By default, an SNR of 4 is used. If a value
            of 0 is provided, the min_height condition is applied instead.
        max_degree: int, optional
            Maximum degree of the polynomial used to describe the fit. If the number of matched
            peaks is greater than the required number of points, the best fit is used.
        min_height: float | int, optional
            Minimum intensity required. The default is 10_000. Only used, if SNR_threshold is
            not provided.

        Notes
        -----
        This function sets the attribute(s):
        calibration_parameters: np.ndarray,
            Array holding the coefficients of the calibration functions where each

        This function tries to emulate the calibration performed in DataAnalysis 5.0. This is the
        description from the handbook:
        -- The spectrum is searched for signals which (1) are within the given search range (m/z)
        of the expected lock mass(es) and which (2) exceed the given intensity threshold. The
        expected lock mass(es), search range (m/z) and intensity threshold are specified in the
        corresponding method.
        -- If at least one lock mass peak is found in the spectrum one of current calibration
        coefficients is corrected such as it is needed to adapt the current m/z value of the lock
        mass signal exactly to the theoretical m/z value of the respective lock mass. The new
        correlation coefficient is then used to recalibrate the respective spectrum.
        -- If the spectrum does not have at least one lock mass peak (above the given intensity
        threshold) the current calibration coefficient is kept for that spectrum.

        To me, it is not clear what the calibration/ correlation coefficient is supposed to mean,
        but a linear fit seems reasonable as higher degree polynomials may result in unreasonably large
        shifts outside the found peak range.
        """
        def calib_spec(spectrum: np.ndarray[float]) -> int | tuple[np.ndarray[float], int]:
            """Find the calibration function for a single spectrum."""
            # pick peaks
            if SNR_threshold > 0:
                peaks: np.ndarray[int] = find_peaks(spectrum, height=self.noise_level * SNR_threshold)[0]
            else:
                peaks: np.ndarray[int] = find_peaks(spectrum, height=min_height)[0]
            peaks_mzs: np.ndarray[float] = self.mzs[peaks]

            # find valid peaks for each calibrant
            closest_peak_mzs: list[float] = []
            closest_calibrant_mzs: list[float] = []
            for jt, calibrant in enumerate(calibrants_mz):
                distances: np.ndarray[float] = np.abs(calibrant - peaks_mzs)  # theory - actual
                if not np.any(distances < search_range):  # no peak with required SNR found inside range
                    logger.debug(f'found no peak above noise level for {calibrant=} and {index=}')
                    calibrator_presences[it, jt] = False
                    continue
                closest_peak_mzs.append(peaks_mzs[np.argmin(distances)])
                closest_calibrant_mzs.append(calibrant)

            # search the coefficients of the polynomial
            # need degree + 1 points for nth degree fit
            n_calibrants = len(closest_peak_mzs)
            if n_calibrants == 0:  # no calibrant found, keep identity
                logger.debug(f'found no calibrant for {index=}')
                return -1
            degree: int = min([max_degree, n_calibrants - 1])
            # polynomial coefficients
            # theory - actual
            yvals = [t - a for t, a in zip(closest_calibrant_mzs, closest_peak_mzs)]
            p: np.ndarray[float] = np.polyfit(x=closest_peak_mzs, y=yvals, deg=degree)
            n_coeffs: int = degree + 1  # number of coefficients in polynomial
            # fill coeff matrix
            return p, n_coeffs

        if self._check_calibration_file_exists():
            logger.warning(
                'Found calibration file. This suggests that Lock Mass calibration has been ' +
                'performed already. It is not recommended to do this step for already calibrated ' +
                'data.'
            )

        assert isinstance(reader, ReadBrukerMCF | hdf5Handler), \
            f'reader must be a ReadBrukerMCF or hdf5Handler instance, not {type(reader)}'
        assert hasattr(reader, 'indices'), 'call reader.set_indices()'
        assert hasattr(reader, 'mzs') and np.allclose(reader.mzs, self.mzs), \
            ('Make sure the mzs of the reader match that of the spectra object (consider calling' +
             ' reader.set_mzs(spec.mzs))')
        if SNR_threshold > 0:
            assert hasattr(self, 'noise_level'), \
                ('This step has to be performed after subtracting the baseline to have access ' +
                 'to the noise_level, unless you set the SNR_threshold to 0.')
        if calibrants_mz is None:
            calibrants_mz = get_calibrants(self.limits)

        N: int = len(self.indices)  # number spectra

        # first column corresponds to highest degree
        # last column to constant
        # each row corresponds to a spectrum
        # ensure transformation has at least 2 parameters
        n_calibration_parameters: int = max_degree + 1

        calibration_parameters: np.ndarray[float] = np.zeros(
            (N, n_calibration_parameters),
            dtype=float
        )
        calibrator_presences: np.ndarray[bool] = np.ones(
            (N, len(calibrants_mz)),
            dtype=bool
        )

        # iterate over all spectra
        for it, index in tqdm(
                enumerate(self.indices),
                total=N,
                desc='Finding calibration parameters',
                smoothing=50 / N
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

        self.calibration_parameters = calibration_parameters

        self.calibration_settings: dict[str, Any] = {
            'calibrants': calibrants_mz,
            'search_range': search_range,
            'SNR_threshold': SNR_threshold,
            'max_degree': max_degree,
            'presences calibrants': calibrant_matches
        }

    def plot_calibration_functions(
            self,
            reader: ReadBrukerMCF | hdf5Handler,
            indices: Iterable[int] | None = None,
            n: int = 10
    ) -> None:
        assert hasattr(self, 'calibration_parameters'), 'Call set_calibrate_functions before plotting.'
        if indices is not None:
            n = len(indices)
        assert (n > 0), 'Number of spectra should be bigger than 0'

        if indices is None:
            indices: np.ndarray[int] = np.random.choice(self.indices, size=n, replace=False)

        fig, axs = plt.subplots(nrows=n, ncols=2, sharex='col')
        if n == 1:
            axs = [axs]

        calibrants_mz: np.ndarray[float] = get_calibrants(self.limits)

        for idx, (ax_l, ax_r) in zip(indices, axs):
            # obtain spectra
            spec_o: Spectrum = reader.get_spectrum(index=idx, limits=self.limits)
            spec_c: Spectrum = self._get_spectrum(reader=reader, index=idx, only_intensity=False)

            array_idx: int = self.spectrum_idx2array_idx(idx)
            poly_coeffs: np.ndarray = self.calibration_parameters[array_idx, :]
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

        # fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        # ax = plt.gca()

        # plt.xlabel("common X")
        # ax.set_ylabel('Intensity')
        # ax_t.set_ylabel('correction (mDa)')

        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        fig.tight_layout()
        plt.show()


    def detect_side_peaks(
            self, 
            max_relative_height: float = .1,
            max_distance: float = .001,
    ) -> None:
        """
        This method defines a list of peaks that are likely artifacts from the Fourier transformation.

        Windowing introduces spectral leakage, which is especially pronounced around high isolated peaks.
        This method goes through the peaks list and marks peaks with low relative height to their neighbour
        within a small region as side-peaks.

        Parameters
        ----------
        max_relative_height : float, optional
            Relative height below which peaks are considered artifacts. The default is 0.1.
        max_distance : float, optional
            Region in which to look for artifacts. The default is 1 mDa.

        Notes
        -----
        This method defines the following attribute(s):
        peaks_is_side_peak : np.ndarray[bool]
            An array marking sidepeaks as True, otherwise as False.
        """
        assert hasattr(self, 'peaks'), 'call set_peaks first'

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
        N_peaks: int = len(self.peaks)
        peak_order: np.ndarray[int] = np.argsort(self.intensities[self.peaks])[::-1]
        # peaks that have to be evaluated
        to_do: np.ndarray[bool] = np.full(N_peaks, True)
        valids: np.ndarray[bool] = np.full(N_peaks, True)
        for peak_idx in peak_order:
            if not to_do[peak_idx]:
                continue
            # mz idx corresponding to peak
            peak: int = self.peaks[peak_idx]
            # check if peaks left and right of it fulfill conditions
            # if they are detected as sidepeaks, they no longer have to be taken into consideration
            # valids will be changed accordingly
            if ((peak_idx_l := peak_idx - 1) >= 0) and to_do[peak_idx_l]:
                peak_l: int = self.peaks[peak_idx_l]
                valid: bool = eval_peak_valid(peak, peak_l)
                valids[peak_idx_l] = valid
                if not valid:
                    to_do[peak_idx_l] = False
            if ((peak_idx_r := peak_idx + 1) < N_peaks) and to_do[peak_idx_r]:
                peak_r: int = self.peaks[peak_idx_r]
                valid: bool = eval_peak_valid(peak, peak_r)
                valids[peak_idx_r] = valid
                if not valid:
                    to_do[peak_idx_r] = False
            to_do[peak_idx] = False

        self.peaks_is_side_peak: np.ndarray[bool] = ~valids

    def _set_peaks_SNR(self) -> None:
        """
        Set the SNRs of peaks based on the noise level.

        Notes
        -----
        This method defines the following attribute(s):
        peaks_SNR: np.ndarray[float]
            The noise level for each mz value.
        """
        assert hasattr(self, 'noise_level'), 'call subtract_baseline'
        N_spec: int = len(self.indices)
        av_intensities: np.ndarray[float] = self.intensities[self.peaks] / N_spec
        self.peaks_SNR: np.ndarray[float] = \
            av_intensities / self.noise_level[self.peaks]

    def _get_SNR_table(self) -> np.ndarray[float]:
        """
        Return SNR values for each spectrum as array where each row corresponds
        to a spectrum. 
        
        Noise level is assumed to be the same for each spectrum.
        This function scales the intensities by the estimated noise level, 
        thereby assigning SNR values to all peaks in each spectrum.
        """
        assert hasattr(self, 'noise_level'), 'call subtract_baseline'
        
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
            SNR_threshold: float = 0,
            remove_sidepeaks: bool = False,
            plts=False,
            **kwargs_sidepeaks: dict
    ) -> None:
        """
        Eliminate peaks not fulfilling the criteria from the peak list.

        Parameters
        ----------
        whitelist : Iterable[int] | None,
            Peaks (idx corresponding to mz value in mzs) that shall not be removed. If this is provided,
            no other filtering will be done.
        SNR_threshold : float, optional
            Minimum SNR required for keeping peaks. The default is to not remove any peaks
            based on SNR.
        remove_sidepeaks : bool, optional
            Remove peaks likely introduced by spectral leakage (see detect_side_peaks).
            The default is not to remove side peaks
        plts: bool, optional,
            Whether to plot the removed peaks.
        kwargs_sidepeaks: dict
            Additional peaks passed on to detect_side_peaks.
        """
        if not hasattr(self, 'peaks_SNR') and SNR_threshold:
            self._set_peaks_SNR()
        if remove_sidepeaks and (not hasattr(self, 'peaks_is_side_peak')):
            self.detect_side_peaks(**kwargs_sidepeaks)

        # skip filtering of valid peaks if a list of peaks to keep is provided
        skip_filtering: bool = whitelist is not None

        if skip_filtering and (SNR_threshold != 0):
            logger.warning(
                'A whitelist was provided as well as an SNR threshold, but ' +
                'filtering will not be executed if a whitelist is provided. ' +
                'If this is what you want, consider performing two filtering ' +
                'steps instead.'
            )
        if skip_filtering and remove_sidepeaks:
            logger.warning(
                'A whitelist was provided and remove_sidepeaks was set to ' +
                'True, but filtering will not be executed if a whitelist ' +
                'is provided. If this is what you want, consider performing ' +
                'two filtering steps instead.'
            )

        N_peaks: int = len(self.peaks)
        peaks_valid: np.ndarray[bool] = np.full(N_peaks, True, dtype=bool)

        if skip_filtering:  # exclude all peaks not in whitelist
            peaks_valid &= np.array([peak in whitelist for peak in self.peaks], dtype=bool)
        if (not skip_filtering) and (SNR_threshold > 0):  # set peaks below SNR threshold to False
            peaks_valid &= self.peaks_SNR > SNR_threshold
        if (not skip_filtering) and remove_sidepeaks:  # set sidepeaks to False
            peaks_valid &= ~self.peaks_is_side_peak

        if plts:  # keep a copy of the original peaks
            peaks = self.peaks.copy()

        # filter out invalid peaks
        self.peaks: np.ndarray[int] = self.peaks[peaks_valid]
        peak_props = {}
        for key, val in self.peak_properties.items():
            peak_props[key] = val[peaks_valid]
        self.peak_properties = peak_props
        # add flag
        self.peak_setting_parameters['modified'] = {
            'whitelist': whitelist,
            'SNR_threshold': SNR_threshold,
            'remove_sidepeaks': remove_sidepeaks
        }

        if hasattr(self, 'kernel_params'):
            self.kernel_params = self.kernel_params[peaks_valid, :]

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
        assert hasattr(self, 'peaks'), 'call set_peaks first'
        mz_idx: float = self.peaks[peak_idx]  # mz index of of center

        H: float = self.intensities[mz_idx]  # corresponding height
        # width of peak at half maximum
        FWHM_l: float = self.mzs[
            (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r: float = self.mzs[
            (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
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
        assert hasattr(self, 'peaks'), 'call set_peaks first'
        mz_idx: int = self.peaks[peak_idx]  # mz index of of center
        mz_c: float = self.mzs[mz_idx]  # center of gaussian
        # height at center of peak - prominence
        H: float = self.intensities[mz_idx]  # corresponding height
        # width of peak at half maximum
        FWHM_l: float = self.mzs[
            (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r: float = self.mzs[
            (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
        ]
        # convert FWHM to standard deviation
        sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
        sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))
        return mz_c, H, sigma_l, sigma_r

    def _kernel_fit_from_peak(
            self, peak_idx: int, sigma_max: float = 5e-3, **ignore
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
        """
        assert hasattr(self, 'peaks'), 'call set_peaks first'
        assert hasattr(self, 'kernel_params'), 'call set_kernel_params first'

        if len(ignore) > 0:
            logger.info(f'unused kwargs in set_kernels: {ignore}')

        # width of peak at half maximum
        idx_l: np.ndarray[int] = (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        idx_r: np.ndarray[int] = (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
        mask: slice = slice(idx_l, idx_r)

        if not np.any(self.kernel_params[peak_idx, :]):
            return None

        mz_c, H, sigma, *sigma_r = self.kernel_params[peak_idx, :]
        if sigma_clippied := (sigma > sigma_max):
            logger.warning(
                f'sigma of kernel ({sigma * 1e3:.1f} mDa) with index {peak_idx} '
                f'is bigger than max ({sigma_max * 1e3:.1f} mDa), halfing sigma.'
            )

            ys = self.intensities[mask].copy()
            # take smaller window to hopefully climb up peak
            *_, l, r = peak_widths(self.intensities, [self.peaks[peak_idx]], rel_height=.9)
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
                p0=self.kernel_params[peak_idx, :],
                bounds=(bounds_l, bounds_r)
            )
        except ValueError as e:
            logger.warning(
                f'encountered a value error while trying to find parameters '
                f'for peak with index {peak_idx}: \n {e} \n'
                f'This can happen for double peaks.'
            )
            return None

        return params

    @property
    def _kernel_func(self) -> Callable[[tuple[np.ndarray, float, ...]], np.ndarray[float]]:
        """Return either gaussian or bigaussian function."""
        if self.kernel_shape == 'bigaussian':
            return bigaussian
        elif self.kernel_shape == 'gaussian':
            return gaussian

    @property
    def _kernel_func_from_peak(self) -> Callable[[int], tuple[float, ...]]:
        """Return either gaussian or bigaussian function"""
        if self.kernel_shape == 'bigaussian':
            return self._bigaussian_from_peak
        elif self.kernel_shape == 'gaussian':
            return self._gaussian_from_peak

    def set_kernels(
            self,
            use_bigaussian: bool = False,
            fine_tune: bool = True,
            **kwargs
    ) -> None:
        """
        Based on the peak properties, find (bi)gaussian parameters to approximate spectrum.

        Creates kernel_params where cols correspond to
        peaks and rows different properties. Properties are: m/z, intensity H, sigma (left, sigma right)

        Parameters
        ----------
        use_bigaussian : bool, optional
            Whether to use bigaussian or gaussian kernels (testing recommends using gaussian kernels).
            The default is False (so using gaussian kernels).
        fine_tune : bool, optional
            If this is set to False, kernel parameters will only be estimated from the height and width
            of the peak. This can be inaccurate for noisy peaks. It is recommeded to set this parameter
            to True in which case an optimizer will be used to find the peak shape on all points within
            a few standard deviations. The default is True.
        kwargs: dict
            Additional keyword arguments for _kernel_fit_from_peak

        Notes
        -----
        This method defines the following attribute(s):
        kernel_shape : str
            Flag defining whether gaussian or bigaussian kernels are used.
        kernel_params : np.ndarray[float]
            Array storing parameters describing the kernels. Each row corresponds to a kernel.
            Depending on the kernel shape, columns are either mz_c, H, sigma (for gaussian)
            or mz_c, H, sigma_l, sigma_r (for bigaussian).

        """
        assert hasattr(self, 'peaks'), 'call set peaks first'

        y: np.ndarray[float] = self.intensities.copy()

        if use_bigaussian:
            self.kernel_shape: str = 'bigaussian'
            self.kernel_params: np.ndarray[float] = np.zeros((len(self.peaks), 4))
        else:
            self.kernel_shape: str = 'gaussian'
            self.kernel_params: np.ndarray[float] = np.zeros((len(self.peaks), 3))

        # start with heighest peak, work down
        idxs_peaks: np.ndarray[int] = np.argsort(self.peak_properties['prominences'])[::-1]
        mask_valid: np.ndarray[bool] = np.ones(len(self.peaks), dtype=bool)
        for idx in idxs_peaks:
            params: tuple = self._kernel_func_from_peak(idx)
            if self.kernel_shape == 'bigaussian':
                mz_c, H, sigma_l, sigma_r = params
            else:
                mz_c, H, sigma_l = params
                sigma_r = 1  # so that next condition doesn't fail
            if (H <= 0) or (sigma_l <= 0) or (sigma_r <= 0):
                mask_valid[idx] = False
                continue
            else:
                self.kernel_params[idx, :] = params
            if fine_tune:
                params: np.ndarray = self._kernel_fit_from_peak(idx, **kwargs)
                if params is not None:
                    self.kernel_params[idx, :] = params
                else:
                    mask_valid[idx] = False
                    continue
            self.intensities -= self._kernel_func(
                self.mzs, *self.kernel_params[idx, :]
            )
        # restore intensities
        self.intensities: np.ndarray[float] = y

        # delete invalid peaks
        self.filter_peaks(whitelist=self.peaks[mask_valid])

    def plt_summed(
            self,
            plt_kernels: bool = False,
            plt_lines: bool = False,
            mz_limits: tuple[float] | None = None
    ) -> None:
        """
        Plot the summed up intensities with synthetic spectrum estimated
        from kernel parameters, if determined.

        This method plots the summed intensities. If 'set_kernel_params' has been called already,
        the synthetic spectrum will be plotted as well, the loss denotes the area between the
        original and synthetic data.

        Parameters
        ----------
        plt_kernels: bool, optional
            The default is False. If kernel parameters have been determined, this option becomes
            available. It is generally not recommended to plot individual kernels and intended for
            debugging. Unless your mass window or number of kernels is fairly small, this will take
            a long time to plot.
        plt_lines: bool, optional
            If spectra have been binned, this option will plot vertical lines at the peak centers where
            there height corresponds to the summed intensity across all spectra.
        mz_limits: tuple[float] | None, optional.
            By default, the entire mass range is plotted. With this parameter it can be decreased.
        """
        # calculate approximated signal by summing up kernels
        if plt_reconstructed := hasattr(self, 'kernel_params'):
            kernels = self._get_kernels(norm_mode='height')
            intensities_approx = (kernels * self.kernel_params[:, 1]).sum(axis=1)
            loss = np.sum(np.abs(self.intensities - intensities_approx)) \
                / np.sum(self.intensities)

        plt.figure()
        if plt_kernels and plt_reconstructed:
            for i in range(len(self.peaks)):
                y: np.ndarray[float] = self._kernel_func(self.mzs, *self.kernel_params[i, :])
                # reduce to 10 std
                mask: np.ndarray[bool] = np.abs(self.kernel_params[i, 0] - self.mzs) <= 10 * self.kernel_params[i, -1]
                plt.plot(self.mzs[mask], y[mask])
        plt.plot(self.mzs, self.intensities, label='summed intensity')
        if plt_reconstructed:
            plt.plot(self.mzs, intensities_approx, label='estimated')
        if hasattr(self, 'binning_by') and plt_lines:
            plt.stem(self.kernel_params[:, 0], self.get_heights().sum(axis=0),
                     markerfmt='', linefmt='red')
        if mz_limits is not None:
            plt.xlim(mz_limits)
            mask: np.ndarray[bool] = (self.mzs >= mz_limits[0]) & (self.mzs <= mz_limits[1])
            plt.ylim((0, self.intensities[mask].max()))
        plt.legend()
        plt.xlabel(r'$m/z$ in Da')
        plt.ylabel('Intensity')
        if plt_reconstructed:
            plt.title(f'Reconstructed summed intensities (loss: {loss:.1f})')
        plt.show()

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
        assert hasattr(self, 'kernel_params'), \
            'call set_kernels() first'

        N_peaks: int = self.kernel_params.shape[0]  # number of identified peaks


        if norm_mode == 'area':
            H: float = np.sqrt(2)  # normalization constant
        elif norm_mode == 'height':
            H: float = 1.
        else:
            raise KeyError('norm_mode must be one of "area", "height", "prob"')


        kernels: np.ndarray[float] = np.zeros((N_peaks, len(self.mzs)))
        # TODO: make use of _kernel_func
        if self.kernel_shape == 'bigaussian':
            for idx_peak in range(N_peaks):
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
        elif self.kernel_shape == 'gaussian':
            for idx_peak in range(N_peaks):
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
            reader: ReadBrukerMCF | None = None,
            profile_spectra: np.ndarray[float] | None = None,
            method: str = 'height',
            **_: dict
    ) -> None:
        """
        For each spectrum find overlap between kernels and signal.

        Parameters
        ----------
        reader: ReadBrukerMCF, optional
            Reader to get the spectra. The default is None. Either reader or profile_spectra must be
            provided.
        profile_spectra: np.ndarray[float], optional
            Resampled spectra as 2D matrix. The default is None. Either reader or profile_spectra must be
            provided.
        integrate_peaks: bool
            If this is set to True, estimate the intensity of each compound by
            assuming that the area under the kernel corresponds to the compound,
            this is valid if spectra are fairly similar. If this is set to False,
            the height of the signal at the center of kernel estimation is used.

        Notes
        -----
        This method defines the following attribute(s):
        line_spectra : np.ndarray[float]
            The centroided spectra.
        """
        def _bin_spectrum_area(spectrum: np.ndarray[float], idx: int) -> None:
            """
            Find intensities of compound based on kernels as the overlap.

            Parameters
            ----------
            spectrum: np.ndarray[float]
                Intensities of spectrum for which to find integrated peak intensities.
            idx: int
                Index of spectrum

            weight is the integrated weighted signal
            ideally this would take the integral but since mzs are equally
            spaced, we can use the sum (scaled accordingly), so instead of
            >>> line_spectrum[idx_peak] = np.trapz(weighted_signal, x=self.mzs)
            take
            >>> line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz

            and instead of summing over peaks we can write this as matrix
            multiplication

            equivalent to
            line_spectrum = np.zeros(N_peaks)
            >>> for idx_peak in range(N_peaks):
            >>>     weighted_signal = spectrum.intensities * bigaussians[idx_peak, :]
            >>>     line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
            """
            self.line_spectra[idx, :] = (spectrum @ kernels) * dmz

        def _bin_spectrum_height(
                spectrum: np.ndarray[float], idx: int
        ) -> None:
            """
            Find intensities of compound based on kernels as the height of
            spectra at kernel centroids.

            Parameters
            ----------
            spectrum: np.ndarray[float]
                Intensities of spectrum for which to find peak intensities.
            idx: int
                Index of spectrum

            """
            # pick values of profile spectrum at kernel maxima
            self.line_spectra[idx, :] = spectrum[idxs_mzs_c]

        def _bin_spectrum_max(spectrum: np.ndarray[float], idx: int) -> None:
            # 2D matrix with intensities windowed to kernels: each column is the
            # product of a kernel with the spectrum
            vals = kernels * spectrum[:, None]
            # the position of the highest value for each kernel
            idcs = np.argmax(vals, axis=0)
            # only accept value if it is not at the boundary of the kernel window
            mask_valid = np.array([
                (vals[idx - 1, i] > 0) & (vals[idx + 1, i] > 0)
                for i, idx in enumerate(idcs)
            ])
            self.line_spectra[idx, :] = vals.max(axis=0) * mask_valid

        assert hasattr(self, 'kernel_params'), 'calculate kernels with set_kernels'
        assert (profile_spectra is not None) or (reader is not None), \
            'provide either a reader or the profile spectra'
        assert method in ('area', 'height', 'max'), \
            'method must be either "area" or "height" or "max"'

        indices_spectra: np.ndarray[int] = self.indices
        N_spectra: int = len(indices_spectra)  # number of spectra in mcf file
        N_peaks: int = self.kernel_params.shape[0]  # number of identified peaks
        self.line_spectra: np.ndarray[float] = np.zeros((N_spectra, N_peaks))  # result array

        self.binning_by: str = method

        if method == 'area':
            _bin_spectrum: Callable[[np.ndarray[float], int], None] = _bin_spectrum_area
            dmz: float = self.mzs[1] - self.mzs[0]
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
            if self.kernel_shape != 'gaussian':
                raise NotImplementedError(
                    f'max only implemented for gaussian kernels, not {self.kernel_shape}'
                )
            _bin_spectrum: Callable[[np.ndarray[float], int], None] = _bin_spectrum_max
            kernels: np.ndarray[float] = self._get_kernels(norm_mode='height')
            # from taking 1 sigma interval
            kernels = (kernels > np.exp(-1)).astype(float)
        else:
            raise NotImplementedError()

        # iterate over spectra and bin according to kernels
        for it, idx_spectrum in tqdm(
                enumerate(indices_spectra),
                total=N_spectra,
                desc='binning spectra', smoothing=50 / N_spectra
        ):
            if reader is not None:
                spectrum: np.ndarray[float] = self._get_spectrum(
                    reader=reader, index=idx_spectrum, only_intensity=True
                )
            else:
                spectrum: np.ndarray[float] = profile_spectra[:, it]
            _bin_spectrum(spectrum, it)

        logger.info('done binning spectra')

    def _add_rxys_to_df(
            self,
            df: pd.DataFrame,
            reader: ReadBrukerMCF | None = None,
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
            Reader from which to obtain the spot data. Defaults to evoking new ImagingInfoXML
            object.
        """
        if (reader is not None) and (type(reader) is ReadBrukerMCF):
            reader.create_spots()
            names: np.ndarray[str] = reader.spots.names
        else:
            reader: ImagingInfoXML = ImagingInfoXML(
                path_d_folder=self.path_d_folder
            )
            names: np.ndarray[str] = reader.spotName

        RXYs = get_rxy(names)

        # search indices of spectra object in reader
        if len(self.indices) != len(reader.indices):
            mask: np.ndarray[bool] = np.array(
                [np.argwhere(idx == reader.indices)[0][0] for idx in self.indices]
            )
        else:
            mask: np.ndarray[bool] = np.ones_like(self.indices, dtype=bool)
        df['R'] = RXYs[mask, 0]
        df['x'] = RXYs[mask, 1]
        df['y'] = RXYs[mask, 2]

        return df

    def binned_spectra_to_df(
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
        kwargs: dict
            Keyword arguments. Allows providing a reader.

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
        if hasattr(self, 'feature_table'):
            return self.feature_table
        assert hasattr(self, 'line_spectra'), 'create line spectra with bin_spectra'

        if integrate_area:
            assert self.binning_by == 'area', \
                ('currently writing a feature table with area is only possible '
                 'if binning was also done with peak integration.')
            data: np.ndarray[float] = self.line_spectra.copy()
        else:
            data: np.ndarray[float] = self.get_heights()

        df: pd.DataFrame = pd.DataFrame(
            data=data,
            columns=np.around(self.kernel_params[:, 0], 4).astype(str)
        )

        df: pd.DataFrame = self._add_rxys_to_df(df, **kwargs)
        # drop possible duplicates due to shift in optimizer
        df: pd.DataFrame = df.loc[:, ~df.columns.duplicated()].copy()

        self.feature_table: pd.DataFrame = df
        return self.feature_table

    def get_kernel_params_df(self) -> pd.DataFrame:
        """Turn the kernel parameters into a feature table."""
        assert hasattr(self, 'kernel_params'), 'call set_kernels'
        if self.kernel_shape == 'bigaussian':
            columns = ['mz', 'H', 'sigma_l', 'sigma_r']
        elif self.kernel_shape == 'gaussian':
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
        assert hasattr(self, 'line_spectra'), 'call bin_spectra'
        if self.binning_by in ('height', 'max'):
            return self.line_spectra.copy()
        elif self.binning_by == 'area':
            area: np.ndarray[float] = self.line_spectra
            sigma_l: np.ndarray[float] = self.kernel_params[:, 2]
            if self.kernel_shape == 'bigaussian':
                sigma_r: np.ndarray[float] = self.kernel_params[:, 3]
            else:
                sigma_r: None = None
            Hs: np.ndarray[float] = self.H_from_area(area, sigma_l, sigma_r)
            return Hs
        else:
            raise NotImplementedError(
                f'get_heights for {self.binning_by} not implemented'
            )

    def spectrum_idx2array_idx(self, spectrum_idx: int | Iterable[int]) -> int | np.ndarray[int]:
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
            plts: bool = False
    ) -> None:
        """
        Obtain the loss of information for each spectrum from the binning.

        Peak areas are integrated based on the assumption that peaks are 
        (bi)gaussian shaped. These assumptions may not always be 
        true in which case the binning may result in significant information 
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

        if not hasattr(self, 'losses'):
            self.losses: np.ndarray[float] = np.zeros(len(self.indices))
        # get sigmas (same for all spectra)
        sigma_ls: np.ndarray[float]  = self.kernel_params[:, 2]
        if self.kernel_shape == 'bigaussian':
            sigma_rs: np.ndarray[float] = self.kernel_params[:, 3]
        else:
            sigma_rs: None = None
        # precompute kernel functions
        kernels: np.ndarray[float] = self._get_kernels(norm_mode='height')
        # loop over spectra
        for c, spectrum_idx in enumerate(spectrum_idxs):
            logger.info(f'setting loss for spectrum {c + 1} out of {len(spectrum_idxs)} ...')
            # get index in array corresponding to spectrum index
            array_idx: int = self.spectrum_idx2array_idx(spectrum_idx)
            spec: np.ndarray[float] = self._get_spectrum(
                reader=reader, index=spectrum_idx, only_intensity=True
            )

            Hs: np.ndarray[float] = self.H_from_area(
                self.line_spectra[array_idx, :],
                sigma_ls,
                sigma_rs
            )
            y_rec: np.ndarray[float] = kernels @ Hs
            loss: float = np.sum(np.abs(spec - y_rec)) / np.sum(spec)
            self.losses[c] = loss

            if plts:
                plt.figure()
                plt.plot(self.mzs, spec, label='original')
                plt.plot(self.mzs, y_rec, label='reconstructed')
                plt.legend()
                plt.title(f'Reconstruction loss: {loss:.3f}')
                plt.show()

    def filter_line_spectra(
            self, SNR_threshold: float = 0, intensity_min: float = 0, **_: dict
    ) -> np.ndarray[bool]:
        """
        Set the intensities that fall below SNR or min intensity to zero
        in the line_spectra attribute
        and return array of changed pixels.

        Parameters
        ----------
        SNR_threshold : float, optional
            Set intensities below this SNR threshold to False.
        intensity_min: float, optional
            Set intensities below this absolute threshold to False.

        Returns
        -------
        mask : np.ndarray[bool]
            Mask object where values not meeting the criteria are set to False
        """
        if SNR_threshold > 0:
            assert hasattr(self, 'noise_level'), 'Call subtract_baseline first'
            mask_snr_too_low: np.ndarray[bool] = self._get_SNR_table() < SNR_threshold
        else:
            mask_snr_too_low: np.ndarray[bool] = np.zeros(self.line_spectra.shape, dtype=bool)
        if intensity_min > 0:
            mask_intensity_too_low: np.ndarray[bool] = self.line_spectra < intensity_min
        else:
            mask_intensity_too_low: np.ndarray[bool] = np.zeros(
                self.line_spectra.shape, dtype=bool
            )

        mask = mask_snr_too_low | mask_intensity_too_low
        self.line_spectra[mask] = 0

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
        assert hasattr(self, 'noise_level') and hasattr(other, 'noise_level'), \
            'make sure both objects have the baseline removed'
        assert hasattr(self, 'kernel_params') and hasattr(other, 'kernel_params')
        assert hasattr(self, 'binning_by') and hasattr(other, 'binning_by')

        # determine if spectra are from the same source folder
        is_same_measurement: bool = self.path_d_folder == other.path_d_folder
        if is_same_measurement:
            assert set(self.indices) & set(other.indices) == set(), \
                'spectra objects must not contain the same spectrum twice'
            path_d_folder: str = self.path_d_folder
            indices: np.ndarray[int] = np.hstack([self.indices, other.indices])
        else:
            logger.warning('this and other object are not from the same folder, this will result \
in loss of functionality!')
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
        kwargs_peak = self.peak_setting_parameters.copy()
        prominence: float = kwargs_peak.pop('prominence')
        prominence *= 2  # assume both summed spectra have roughly the same prominences
        s_new.set_peaks(prominence=prominence, **kwargs_peak)
        s_new.set_kernels(use_bigaussian=self.kernel_shape == 'bigaussian')
        s_new.bin_spectra(
            profile_spectra=profiles,
            integrate_peaks=self.binning_by == 'area'
        )
        return s_new

    def full(self, reader: ReadBrukerMCF | hdf5Handler, **kwargs: dict):
        """Perform all steps with the provided parameters."""
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(**kwargs)
        self.set_calibrate_functions(reader=reader, **kwargs)
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(overwrite=True, **kwargs)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        self.bin_spectra(reader, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(**kwargs)

    def full_targeted(self, reader: ReadBrukerMCF | hdf5Handler, targets: list[float], **kwargs):
        """Perform all steps for targeted compounds with the provided parameters."""
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(**kwargs)
        self.set_calibrate_functions(reader=reader, **kwargs)
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(overwrite=True)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        # set target compounds
        self.set_targets(targets, plts=True)
        self.bin_spectra(reader=reader, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(**kwargs)

    def save(self):
        """Save object to d-folder."""
        folder, file = self._get_disc_folder_and_file()
        file_old: str = self.path_d_folder + '/' + 'spectra_object.pickle'
        if os.path.exists(os.path.join(folder, file_old)):
            logger.info('deleting legacy file')
            os.remove(os.path.join(folder, file_old))

        keep_attributes: set[str] = set(self.__dict__.keys()) & class_to_attributes(self)
        if hasattr(self, 'feature_table') and hasattr(self, 'line_spectra'):
            keep_attributes.remove('line_spectra')

        save_dict: dict[str, Any] = {key: self.__dict__[key] for key in keep_attributes}

        logger.info(f'saving image object with {self.__dict__.keys()} to {folder}')
        with open(file, 'wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        """Load object from d-folder."""
        folder, file = self._get_disc_folder_and_file()
        file_old: str = os.path.join(folder,'spectra_object.pickle')

        if not os.path.exists(file):
            logger.info(f'loading legacy file {file_old} from {folder}')
            assert os.path.exists(file_old), \
                f'found no saved spectra object in {folder}'
            file: str = file_old

        # for backwards compatibility, filter out attributes that are no longer
        # desired to load
        filter_attr = class_to_attributes(self)

        with open(file, 'rb') as f:
            obj: object | dict = pickle.load(f)
            if type(obj) is not dict:  # legacy
                obj: dict[str, Any] = obj.__dict__
            # filter out attributes that are not supposed to be saved
            load_attr: set[str] = filter_attr & set(obj.keys())
            # generate new dict, that only has the desired attributes
            obj_new: dict[str, Any] = {key: obj[key] for key in load_attr}
            # merge the objects dict with the disk dict, overwriting
            # instance attributes with saved once, if they both exist
            self.__dict__ |= obj_new

        if hasattr(self, 'feature_table'):
            self.line_spectra: np.ndarray[float] = self.feature_table.\
                drop(columns=['R', 'x', 'y']).\
                to_numpy()


class ClusteringManager:
    """
    Handle Spectra in clusters (e.g. regions of interest) for more precise  peak-picking. Also, more
    RAM-light.

    Currently not tested, usage is discouraged.
    """
    def __init__(self, reader: ReadBrukerMCF, **kwargs_spectra):
        assert hasattr(reader, 'indices'), 'call create_indices'
        assert hasattr(reader, 'limits'), \
            'call set_QTOF_window or define mass limits'
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
        assert hasattr(self, 'clusters')
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
        assert hasattr(self, 'clusters'), 'call set_clusters first'
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
    >>> from exporting.from_mcf.cSpectrum import MultiSectionSpectra
    >>> from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
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
            Keyword arguments passed on to Spectra.set_calibrate_functions
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

    def set_calibrate_functions(
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
            Keyword arguments passed on to Spectra.set_calibrate_functions
        """
        for i, reader in enumerate(readers):
            self.specs[i].set_calibrate_functions(reader=reader, **kwargs)

    def distribute_peaks_and_kernels(self):
        """
        Transmit the peak and kernel properties of this object to the Spectra children
        """
        assert hasattr(self, 'peaks'), 'call set_peaks() first'
        assert hasattr(self, 'kernel_params'), 'call set_kernel_params() first'
        for spec in self.specs:
            spec.peaks = self.peaks.copy()
            spec.peak_properties = self.peak_properties.copy()
            spec.peak_setting_parameters = self.peak_setting_parameters.copy()
            spec.peak_setting_parameters['notes'] = 'peak properties set in MultiSectionSpectra'
            spec.kernel_params = self.kernel_params.copy()
            spec.kernel_shape = self.kernel_shape
            if hasattr(self, 'noise_level'):
                spec.noise_level = self.noise_level

    def bin_spectra(self, readers: Iterable[ReadBrukerMCF | hdf5Handler], **kwargs):
        """
        Iterate over spectra by calling the corresponding Spectra method.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.set_calibrate_functions
        """
        line_spectra = []
        for i, reader in enumerate(readers):
            self.specs[i].bin_spectra(reader, **kwargs)
            line_spectra.append(self.specs[i].line_spectra.copy())
        self.line_spectra: np.ndarray[float] = np.vstack(line_spectra)
        self.binning_by = self.specs[0].binning_by

    def binned_spectra_to_df(
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
            Keyword arguments passed on to Spectra.set_calibrate_functions
        """
        if readers is None:
            readers = [None] * len(self.specs)

        fts = []
        for i, reader in enumerate(readers):
            self.specs[i].binned_spectra_to_df(reader=reader, **kwargs)
            fts.append(self.specs[i].feature_table)

        self.feature_table: pd.DataFrame = combine_feature_tables(fts)
        return self.feature_table

    def full(self, readers: list[ReadBrukerMCF | hdf5Handler], **kwargs: dict):
        """
        Perform all steps to process and bin the spectra.

        Parameters
        ----------
        readers : list[ReadBrukerMCF | hdf5Handler]
            List of readers from which to obtain the spectra.
        kwargs: dict[str, Any]
            Keyword arguments passed on to Spectra.set_calibrate_functions
        """
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(**kwargs)
        self.set_calibrate_functions(readers=readers, **kwargs)
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(overwrite=True, **kwargs)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        self.distribute_peaks_and_kernels()
        self.bin_spectra(readers=readers, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(readers=readers, **kwargs)

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
            Keyword arguments passed on to Spectra.set_calibrate_functions
        """
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(**kwargs)
        self.set_calibrate_functions(readers=readers, **kwargs)
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(overwrite=True, **kwargs)
        # set target compounds
        self.set_targets(targets, **kwargs, plts=True)
        self.distribute_peaks_and_kernels()
        self.bin_spectra(readers=readers, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(readers=readers, **kwargs)

    def save(self, path_file: str):
        """
        Save object to d-folder.

        Parameters
        ----------
        path_file: str
            Path and name of saved file.

        Example
        -------
        >>> obj.save(r'path/to/save/file.pickle')
        """
        if os.path.exists(path_file):
            logger.info('deleting legacy file')
            os.remove(path_file)

        keep_attributes: set[str] = set(self.__dict__.keys()) & class_to_attributes(self)
        if hasattr(self, 'feature_table') and hasattr(self, 'line_spectra'):
            keep_attributes.remove('line_spectra')

        save_dict: dict[str, Any] = {key: self.__dict__[key] for key in keep_attributes}

        logger.info(f'saving image object with {self.__dict__.keys()} to {os.path.dirname(path_file)}.')
        with open(path_file, 'wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path_file: str):
        """
        Load object from d-folder.

        Parameters
        ----------
        path_file: str
            Path and name of saved file.

        Example
        -------
        >>> from exporting.from_mcf.cSpectrum import MultiSectionSpectra
        >>> obj = MultiSectionSpectra([])
        >>> obj.load(r'path/to/save/file.pickle')
        """
        assert os.path.exists(path_file), \
            (f'found no saved spectra object in {os.path.dirname(path_file)} ' +
             f'with name {os.path.basename(path_file)}')

        # for backwards compatibility, filter out attributes that are no longer
        # desired to load
        filter_attr = class_to_attributes(self)

        with open(path_file, 'rb') as f:
            obj: object | dict = pickle.load(f)
            if type(obj) is not dict:  # legacy
                obj: dict[str, Any] = obj.__dict__
            # filter out attributes that are not supposed to be saved
            load_attr: set[str] = filter_attr & set(obj.keys())
            # generate new dict, that only has the desired attributes
            obj_new: dict[str, Any] = {key: obj[key] for key in load_attr}
            # merge the objects dict with the disk dict, overwriting
            # instance attributes with saved once, if they both exist
            self.__dict__ |= obj_new

        if hasattr(self, 'feature_table'):
            self.line_spectra: np.ndarray[float] = self.feature_table. \
                drop(columns=['R', 'x', 'y']). \
                to_numpy()
