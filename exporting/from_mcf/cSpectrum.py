from __future__ import annotations

from data.combine_feature_tables import combine_feature_tables
from exporting.from_mcf.rtms_communicator import ReadBrukerMCF, Spectrum
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler
from exporting.sqlite_mcf_communicator.sql_to_mcf import get_sql_files
from exporting.from_mcf.helper import get_mzs_for_limits
from util.manage_obj_saves import class_to_attributes
from data.file_helpers import ImagingInfoXML

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import time
import pickle
import psutil

from typing import Iterable
from scipy.signal import find_peaks, correlate, correlation_lags
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter


def gaussian(x: np.ndarray, x_c, H, sigma):
    return H * np.exp(-1 / 2 * ((x - x_c) / sigma) ** 2)


def bigaussian(x: np.ndarray, x_c, H, sigma_l, sigma_r):
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
    np.ndarray
        Intensities of bigaussian.

    """
    x_l = x[x <= x_c]
    x_r = x[x > x_c]
    y_l = H * np.exp(-1 / 2 * ((x_l - x_c) / sigma_l) ** 2)
    y_r = H * np.exp(-1 / 2 * ((x_r - x_c) / sigma_r) ** 2)
    return np.hstack([y_l, y_r])


class Spectra:
    """Container for multiple Spectrum objects and binning."""

    def __init__(
            self,
            reader: ReadBrukerMCF | hdf5Handler | None = None,
            limits: tuple[float, float] = None,
            delta_mz: float = 1e-4,
            indices: Iterable = None,
            initiate: bool = True,
            path_d_folder: str | None = None
    ):
        """
        Initiate the object.

        Either pass a reader or load it from the specified d_folder.

        Parameters
        ----------
        reader : ReadBrukerMCF | None, optional
            DESCRIPTION. The default is None.
        limits : tuple[float], optional
            DESCRIPTION. The default is None.
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
        assert (reader is not None) or (path_d_folder is not None), \
            'Either pass a reader or load and the corresponding d-folder'
        if (reader is None) and initiate:
            print('cannot initiate without a reader')
            initiate = False

        if path_d_folder is not None:
            self.path_d_folder = path_d_folder
        elif reader is not None:
            self.path_d_folder = reader.path_d_folder
        self.delta_mz = delta_mz
        if initiate:
            self._initiate(reader, indices, limits)
        else:
            self.indices: Iterable[int] = indices
            self.limits: tuple[float, float] = limits

    def _initiate(
            self,
            reader: ReadBrukerMCF,
            indices: Iterable[int],
            limits: tuple[float, float]
    ):
        """Set limits and masses based on metadataa from the reader."""
        is_rtms: bool = type(reader) == ReadBrukerMCF
        if indices is None:
            if not hasattr(reader, 'indices'):
                reader.create_indices()

            indices = reader.indices
        self.indices = indices
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

    def add_all_spectra(self, reader):
        """Add up all spectra found in the mcf file."""
        N = len(self.indices)
        print(f'adding up {N} spectra ...')

        if not hasattr(reader, 'mzs'):
            reader.set_mzs(self.mzs)

        time0 = time.time()
        print_interval = 10 ** (np.around(np.log10(N), 0) - 2)
        # iterate over all spectra
        for it, index in enumerate(self.indices):
            spectrum: np.ndarray[float] = \
                reader.get_spectrum_resampled_intensities(int(index))
            self.add_spectrum(spectrum)

            if it % print_interval == 0:
                time_now = time.time()
                time_elapsed = time_now - time0
                predict = time_elapsed * N / (it + 1)  # s
                left = predict - time_elapsed
                left_min, left_sec = divmod(left, 60)
                print(end='\x1b[2K')
                print(f'estimated time left: {str(int(left_min)) + " min" if left_min != 0 else ""} {left_sec:.1f} sec', end='\r')
        print('done adding up spectra')

    def add_all_spectra_aligned(self, reader):
        for it, index in enumerate(self.indices):
            spectrum: Spectrum = reader.get_spectrum(int(index))
            if it > 0:
                shift = self.xcorr(spectrum)
                # shift according to number of spectra
                weight = 1 / (it + 1)
                self.mzs += shift * weight
                spectrum.mzs -= shift * (1 - weight)
            self.add_spectrum(spectrum.intensities)

    def subtract_baseline(self, window_size=.05, plts=False, **ignore: dict):
        if window_size == 0:
            base_lvl: float = self.intensities.min()
            self.intensities -= base_lvl
            self.noise_level = np.full_like(self.intensities, base_lvl)
            return
        # convert Da to number of sample points
        elif (window_size < 1) and isinstance(window_size, float):
            dmz = self.mzs[1] - self.mzs[0]
            window_size = int(window_size / dmz + .5)
        ys_min = minimum_filter(self.intensities, size=window_size)
        # store for SNR estimation
        self.noise_level = ys_min / len(self.indices)

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

    def xcorr(self, other: Spectra, max_mass_offset: float = 1e-3, plts: bool = False) -> float:
        """
        Calculate crosscorrelation for self and other and return the maximum.

        Parameters
        ----------
        max_mass_offset : float | None, optional
            The maximal allowed mass difference between the two spectra in 
            Da. The default is 1 mDa. None will not restrict the search
            space.

        Returns
        -------
        mass_offset: float
            The mass offset between the two spectra.

        """
        diffs = np.diff(self.mzs)

        other.resample(self.mzs)
        a = self.intensities
        b = other.intensities
        N = len(b)

        lags = correlation_lags(N, N, mode='full')
        masses = diffs[0] * lags
        corrs = correlate(a, b, mode='full')
        if max_mass_offset is not None:
            mask = np.abs(masses) <= max_mass_offset
        else:
            mask = np.ones_like(corrs, dtype=bool)
        corrs[~mask] = 0
        idx = np.argmax(corrs)
        if (idx == 0) or (idx == len(corrs) - 1):
            idx = np.argwhere(lags == 0)[0][0]
        lag = lags[idx]
        mass_offset = masses[idx]
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
    ):
        """
        Set the peaks based on a number of target compounds (mass given in Da). 
        Peaks are searched within a mass window defined by tolerances, such that the
        assigned mass is within target +/ tolerance. If no tolerance is given, 
        it is estimated from the kernel widths.

        Method 'nearest_peak' searches the closest peak in the summed up spectrum
        and estimates the kernel shape of that

        Method 'area_overlap' calculates the overlap between the kernels 
        (calculated from the tolerance where the tolerance is assumed to be the standard deviation).

        Method 'highest' takes the highest intensity within the given tolerance.

        This function sets the kernel_params and then requires using the bin_spectra function.

        targets: Iterable[float]
            The m/z values of interest in Da
        tolerance: Iterable[float] | float
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
            print('estimating tolerance from summed spectrum ...')
            self.set_peaks()
            self.set_kernels
            tolerances = np.median(self.kernel_params[:, -1])
            print(f'found tolerance of {tolerances*1e3:.1f} mDa')
        if isinstance(tolerances, float | int):
            tolerances = np.ones_like(targets) * tolerances
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
            self.kernel_params = np.zeros((N_peaks, 3))  # gaussian
            self.kernel_params[:, 0] = targets  # center
            self.kernel_params[:, 1] = self.intensities[self.peaks]
            self.kernel_params[:, 2] = tolerances
        elif method == 'nearest_peak':
            self.set_peaks(**kwargs)
            # filter peaks to keep only the closest one to each target
            idxs_keep: list[int] = []
            dists_keep: list[float] = []
            for idx, target in enumerate(targets):
                dists: np.ndarray[float] = np.abs(self.mzs[self.peaks] - target)
                if not np.any(dists < tolerances[idx]):
                    print(f'did not find peak for {target=}')
                    continue
                idx_keep: int = np.argmin(dists)
                idxs_keep.append(idx_keep)
                dists_keep.append(dists[idx_keep])
            peaks_keep = self.peaks[idxs_keep]
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
                y = self.kernel_func(self.mzs, *self.kernel_params[i, :])
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
        self.peak_setting_parameters = kwargs
        self.peak_setting_parameters['prominence'] = prominence
        self.peak_setting_parameters['width'] = width

    def detect_side_peaks(
            self, max_relative_height: float = .1,
            max_distance: float = .001,
            plts: bool = False
    ) -> None:
        """
        This method defines a list of peaks that are likely 
        artifacts from the Fourier transformation.
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

    def set_peaks_SNR(self) -> None:
        """
        Set the SNRs of peaks based on the noise level.

        Defines
        -------
        peaks_SNR
        """
        assert hasattr(self, 'noise_level'), 'call subtract_baseline'
        N_spec: int = len(self.indices)
        av_intensities: np.ndarray[float] = self.intensities[self.peaks] / N_spec
        self.peaks_SNR: np.ndarray[float] = av_intensities / self.noise_level[self.peaks]

    def get_SNR_table(self, **kwargs) -> np.ndarray[float]:
        """Return the corresponding SNR values for the feature table."""
        heights = self.get_heights()
        assert hasattr(self, 'noise_level'), 'call subtract_baseline'

        # noise level is assumed to be the same for each spectrum
        # get noise levels at centers of each peak
        noise_levels = np.array([
            self.noise_level[
                np.argmin(
                    np.abs(self.mzs - mz_c)
                )
            ]
            for mz_c in self.kernel_params[:, 0]
        ])

        SNRs: np.ndarray = heights / noise_levels

        return SNRs

    def filter_peaks(
            self,
            whitelist: Iterable[int] | None = None,
            SNR_threshold: float = 0,
            remove_sidepeaks: bool = False,
            plts=False,
            **kwargs_sidepeaks: dict
    ) -> None:
        """Eliminate peaks not fulfilling the criteria from the peak list."""
        if not hasattr(self, 'peaks_SNR') and SNR_threshold:
            self.set_peaks_SNR()
        if remove_sidepeaks and (not hasattr(self, 'peaks_is_side_peak')):
            self.detect_side_peaks(**kwargs_sidepeaks)

        # skip filtering of valid peaks if a list of peaks to keep is provided
        skip_filtering: bool = whitelist is not None

        N_peaks: int = len(self.peaks)
        peaks_valid: np.ndarray[bool] = np.full(N_peaks, True, dtype=bool)

        if whitelist is not None:
            peaks_valid &= np.array([peak in whitelist for peak in self.peaks], dtype=bool)
        if (not skip_filtering) and (SNR_threshold > 0):
            peaks_valid &= self.peaks_SNR > SNR_threshold
        if not skip_filtering and remove_sidepeaks:
            peaks_valid &= ~self.peaks_is_side_peak

        if plts:
            peaks = self.peaks.copy()

        self.peaks = self.peaks[peaks_valid]
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

    def gaussian_from_peak(self, peak_idx):
        assert hasattr(self, 'peaks'), 'call set_peaks first'
        mz_idx = self.peaks[peak_idx]  # mz index of of center

        H = self.intensities[mz_idx]  # corresponding height
        # width of peak at half maximum
        FWHM_l = self.mzs[
            (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r = self.mzs[
            (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
        ]
        mz_c = (FWHM_l + FWHM_r) / 2
        # convert FWHM to standard deviation
        sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
        sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))
        sigma = (sigma_l + sigma_r) / 2
        return mz_c, H, sigma

    def kernel_fit_from_peak(self, peak_idx):
        """Find kernel parameters for a peak with the shape of a bigaussian."""
        assert hasattr(self, 'peaks'), 'call set_peaks first'
        mz_idx = self.peaks[peak_idx]  # mz index of of center

        # width of peak at half maximum
        idx_l = (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        idx_r = (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
        mask = slice(idx_l, idx_r)
        if hasattr(self, 'kernel_params') and np.any(self.kernel_params[peak_idx, :]):
            mz_c, H, sigma, *sigma_r = self.kernel_params[peak_idx, :]
            bounds_l = [
                mz_c - sigma / 4,
                H * .8,
                sigma * .8
            ]
            bounds_r = [
                mz_c + sigma / 4,
                H * 1.2,
                sigma * 1.2
            ]
            if len(sigma_r) > 0:
                bounds_l.append(sigma_r[0] * .8)
                bounds_r.append(sigma_r[0] * 1.2)
            params, _ = curve_fit(
                f=self.kernel_func,
                xdata=self.mzs[mask],
                ydata=self.intensities[mask],
                p0=self.kernel_params[peak_idx, :],
                bounds=(bounds_l, bounds_r)
            )
        else:
            return self.kernel_params[peak_idx, :]

        return params

    def bigaussian_from_peak(self, peak_idx: int):
        """Find kernel parameters for a peak with the shape of a bigaussian."""
        assert hasattr(self, 'peaks'), 'call set_peaks first'
        mz_idx = self.peaks[peak_idx]  # mz index of of center
        mz_c = self.mzs[mz_idx]  # center of gaussian
        # height at center of peak - prominence
        H = self.intensities[mz_idx]  # corresponding height
        # width of peak at half maximum
        FWHM_l = self.mzs[
            (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r = self.mzs[
            (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
        ]
        # convert FWHM to standard deviation
        sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
        sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))
        return mz_c, H, sigma_l, sigma_r

    @property
    def kernel_func(self):
        if self.kernel_shape == 'bigaussian':
            return bigaussian
        elif self.kernel_shape == 'gaussian':
            return gaussian

    @property
    def kernel_func_from_peak(self):
        if self.kernel_shape == 'bigaussian':
            return self.bigaussian_from_peak
        elif self.kernel_shape == 'gaussian':
            return self.gaussian_from_peak

    def set_kernels(self, use_bigaussian=False, fine_tune=True, **ignore):
        """
        Based on the peak properties, find bigaussian parameters to 
        approximate spectrum. Creates kernel_params where cols correspond to 
        peaks and rows different properties. Properties are: m/z, vertical 
        shift, intensity at max, sigma left, sigma right
        """
        assert hasattr(self, 'peaks'), 'call set peaks first'

        y = self.intensities.copy()

        if use_bigaussian:
            self.kernel_shape = 'bigaussian'
            self.kernel_params = np.zeros((len(self.peaks), 4))
        else:
            self.kernel_shape = 'gaussian'
            self.kernel_params = np.zeros((len(self.peaks), 3))

        # start with heighest peak, work down
        idxs_peaks = np.argsort(self.peak_properties['prominences'])[::-1]
        mask_valid = np.ones(len(self.peaks), dtype=bool)
        for idx in idxs_peaks:
            params = self.kernel_func_from_peak(idx)
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
                params = self.kernel_fit_from_peak(idx)
                if params is not None:
                    self.kernel_params[idx, :] = params
                else:
                    mask_valid[idx] = False
                    continue
            self.intensities -= self.kernel_func(
                self.mzs, *self.kernel_params[idx, :]
            )
        # restore intensities
        self.intensities = y

        # delete invalid peaks
        self.filter_peaks(whitelist=self.peaks[mask_valid])
        self.kernel_params = self.kernel_params[mask_valid, :]

    def plt_summed(
            self,
            plt_kernels: bool = False,
            plt_lines: bool = False,
            mz_limits: tuple[float] | None = None
    ):
        assert hasattr(self, 'kernel_params'), 'call set_kernels first'
        # calculate approximated signal by summing up kernels
        if hasattr(self, 'kernel_params'):
            kernels = self._get_kernels(norm_mode='height')
            intensities_approx = (kernels * self.kernel_params[:, 1]).sum(axis=1)
            loss = np.sum(np.abs(self.intensities - intensities_approx)) \
                / np.sum(self.intensities)

        plt.figure()
        if plt_kernels and hasattr(self, 'kernel_params'):
            for i in range(len(self.peaks)):
                y = self.kernel_func(self.mzs, *self.kernel_params[i, :])
                mask = np.abs(self.kernel_params[i, 0] - self.mzs) <= 10 * self.kernel_params[i, -1]
                plt.plot(self.mzs[mask], y[mask])
        plt.plot(self.mzs, self.intensities, label='summed intensity')
        if hasattr(self, 'kernel_params'):
            plt.plot(self.mzs, intensities_approx, label='estimated')
        if hasattr(self, 'binning_by') and plt_lines:
            plt.stem(self.kernel_params[:, 0], self.get_heights().sum(axis=0),
                     markerfmt='', linefmt='red')
        if mz_limits is not None:
            plt.xlim(mz_limits)
            mask = (self.mzs >= mz_limits[0]) & (self.mzs <= mz_limits[1])
            plt.ylim((0, self.intensities[mask].max()))
        plt.legend()
        plt.xlabel(r'$m/z$ in Da')
        plt.ylabel('Intensity')
        plt.title(f'Reconstructed summed intensities (loss: {loss:.1f})')
        plt.show()

    def _get_kernels(self, norm_mode: str = 'area') -> np.ndarray[float]:
        """
        Return matrix in which each column corresponds to the intensities
        of a kernel.
        """
        N_peaks = self.kernel_params.shape[0]  # number of identified peaks

        if norm_mode == 'area':
            H = np.sqrt(2)  # normalization constant
        elif norm_mode == 'height':
            H = 1
        else:
            raise KeyError('norm_mode must be one of "area", "height"')

        kernels = np.zeros((N_peaks, len(self.mzs)))
        if self.kernel_shape == 'bigaussian':
            for idx_peak in range(N_peaks):
                # x_c, H, sigma_l, sigma_r
                sigma_l = self.kernel_params[idx_peak, 2]
                sigma_r = self.kernel_params[idx_peak, 3]
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
                sigma = self.kernel_params[idx_peak, 2]
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
            integrate_peaks: bool = False,
            **_: dict
    ):
        """
        For each spectrum find overlap between kernels and signal.

        integrate_peaks: bool
            If this is set to True, estimate the intensity of each compound by
            assuming that the area under the kernel corresponds to the compound,
            this is valid if spectra are fairly similar. If this is set to False,
            the height of the signal at the center of kernel estimation is used.

        """
        def _bin_spectrum_area(
                spectrum: np.ndarray[float] | Spectrum, idx: int
        ) -> None:
            """Find intensities of compound based on kernels."""
            # weight is the integrated weighted signal
            # ideally this would take the integral but since mzs are equally
            # spaced, we can use the sum (scaled accordingly), so instead of
            # line_spectrum[idx_peak] = np.trapz(weighted_signal, x=self.mzs)
            # take
            # line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
            #
            # and instead of summing over peaks we can write this as matrix
            # multiplication
            #
            # equivalent to
            # line_spectrum = np.zeros(N_peaks)
            # for idx_peak in range(N_peaks):
            #     weighted_signal = spectrum.intensities * bigaussians[idx_peak, :]
            #     line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
            line_spectrum = (spectrum @ kernels) * dmz
            self.line_spectra[idx, :] = line_spectrum

        def _bin_spectrum_height(
                spectrum: np.ndarray[float] | Spectrum, idx: int
        ) -> None:
            line_spectrum = spectrum[idxs_mzs_c]
            self.line_spectra[idx, :] = line_spectrum

        assert hasattr(self, 'kernel_params'), 'calculate kernels with set_kernels'
        assert (profile_spectra is not None) or (reader is not None), \
            'provide either a reader or the profile spectra'

        indices_spectra = self.indices
        N_spectra = len(indices_spectra)  # number of spectra in mcf file
        N_peaks = self.kernel_params.shape[0]  # number of identified peaks
        self.line_spectra = np.zeros((N_spectra, N_peaks))  # result array

        if integrate_peaks:
            _bin_spectrum = _bin_spectrum_area
            self.binning_by = 'area'
            dmz = self.mzs[1] - self.mzs[0]
            # precompute bigaussians
            kernels = self._get_kernels(norm_mode='area')
        else:
            _bin_spectrum = _bin_spectrum_height
            self.binning_by = 'height'
            # indices in mzs corresponding to peak centers
            mzs_c = self.kernel_params[:, 0]
            idxs_mzs_c = np.array([
                np.argmin(np.abs(mz_c - self.mzs)) for mz_c in mzs_c
            ])

        # iterate over spectra and bin according to kernels
        print(f'binning {N_spectra} spectra into {N_peaks} bins ...')
        time0 = time.time()
        for it, idx_spectrum in enumerate(indices_spectra):
            if reader is not None:
                spectrum: np.ndarray[float] = \
                    reader.get_spectrum_resampled_intensities(int(idx_spectrum))
            else:
                spectrum: np.ndarray[float] = profile_spectra[:, it]
            _bin_spectrum(spectrum, it)
            if it % 10 ** (np.around(np.log10(N_spectra), 0) - 2) == 0:
                time_now = time.time()
                time_elapsed = time_now - time0
                predict = time_elapsed * N_spectra / (it + 1)  # s
                left = predict - time_elapsed
                left_min, left_sec = divmod(left, 60)
                print(end='\x1b[2K')
                print(
                    f'estimated time left: {str(int(left_min)) + " min" if left_min != 0 else ""} {left_sec:.1f} sec',
                    end='\r'
                )
        print('\n')
        print('done binning spectra')

    def _add_rxys_to_df(self, df: pd.DataFrame, reader: ReadBrukerMCF | None = None) -> pd.DataFrame:
        if (reader is not None) and (type(reader) is ReadBrukerMCF):
            reader.create_spots()
            names = reader.spots.names
        else:
            reader: ImagingInfoXML = ImagingInfoXML(
                path_d_folder=self.path_d_folder
            )
            names = reader.spotName

        # add R, x, y columns
        str_prefix = r'R(\d+)X'
        str_x = r'R\d+X(.*?)Y'
        str_y = r'Y(.*?)$'

        def rxy(name):
            r = int(re.findall(str_prefix, name)[0])
            x = int(re.findall(str_x, name)[0])
            y = int(re.findall(str_y, name)[0])
            return [r, x, y]

        RXYs = np.array([rxy(name) for name in names])

        # search indices of spectra object in reader
        print(self.indices.shape, self.indices)
        print(reader.indices.shape, reader.indices)
        if len(self.indices) != len(reader.indices):
            mask = np.array(
                [np.argwhere(idx == reader.indices)[0][0] for idx in self.indices]
            )
        else:
            mask = np.ones_like(self.indices, dtype=bool)
        df['R'] = RXYs[mask, 0]
        df['x'] = RXYs[mask, 1]
        df['y'] = RXYs[mask, 2]

        return df

    def binned_spectra_to_df(
            self, integrate_area: bool = False, **kwargs
    ) -> pd.DataFrame:
        """
        Turn the line_spectra into the familiar df with R, x, y columns.

        integrate_area: If True, this will return the area of each peak, 
            otherwise it will return the height of each peak
        """
        if hasattr(self, 'feature_table'):
            return self.feature_table
        assert hasattr(self, 'line_spectra'), 'create line spectra with bin_spectra'

        if integrate_area:
            data = self.line_spectra.copy()
        else:
            data = self.get_heights()

        df = pd.DataFrame(
            data=data,
            columns=np.around(self.kernel_params[:, 0], 4).astype(str)
        )

        df = self._add_rxys_to_df(df, **kwargs)

        self.feature_table = df
        return self.feature_table

    def get_kernel_params_df(self):
        assert hasattr(self, 'kernel_params'), 'call set_kernels'
        if self.kernel_shape == 'bigaussian':
            columns = ['mz', 'H', 'sigma_l', 'sigma_r']
        elif self.kernel_shape == 'gaussian':
            columns = ['mz', 'H', 'sigma']
        df = pd.DataFrame(data=self.kernel_params, columns=columns)
        return df

    @staticmethod
    def H_from_area(area, sigma_l, sigma_r=None):
        # \int_{-infty}^{infty} H \exp(- (x - x_c)^2 / (2 sigma)^2)dx
        #   = sqrt(2 pi) H sigma
        # => A = H sqrt(pi / 2) (sigma_l + sigma_r)
        # <=> H = sqrt(2 / pi) * A* 1 / (sigma_l + sigma_r)
        if sigma_r is None:
            sigma_r = sigma_l
        return np.sqrt(2 / np.pi) * area / (sigma_l + sigma_r)

    def get_heights(self):
        """Calculate the peak heights corresponding to the estimated peaks"""
        assert hasattr(self, 'line_spectra'), 'call bin_spectra'
        if self.binning_by == 'height':
            return self.line_spectra.copy()

        area = self.line_spectra
        sigma_l = self.kernel_params[:, 2]
        if self.kernel_shape == 'bigaussian':
            sigma_r = self.kernel_params[:, 3]
        else:
            sigma_r = None
        Hs = self.H_from_area(area, sigma_l, sigma_r)
        return Hs

    def spectrum_idx2array_idx(self, spectrum_idx: int | Iterable[int]):
        if isinstance(spectrum_idx, int | np.int8 | np.int16 | np.int32 | np.int64):
            return np.argwhere(self.indices == spectrum_idx)[0][0]
        else:
            idxs = [np.argwhere(self.indices == idx)[0][0] for idx in spectrum_idx]
            return np.array(idxs)

    def set_reconstruction_losses(
            self, reader: ReadBrukerMCF,
            spectrum_idxs: list[int] = None, plts=False
    ):
        """
        Obtain the loss of information for each spectrum from the binning.

        Peak areas are integrated based on the assumption that peaks are 
        (bi)gaussian shaped. These assumptions may not always be 
        true in which case the binning may result in significant information 
        loss. This function calculates the difference between the original 
        (processed) signals and the one described by the kernels and gives the
        loss in terms of the integrated difference divided by the area of the 
        original signal.
        """
        if spectrum_idxs is None:
            spectrum_idxs = self.indices

        if not hasattr(self, 'losses'):
            self.losses = np.zeros(len(self.indices))
        # get sigmas (same for all spectra)
        sigma_ls = self.kernel_params[:, 2]
        if self.kernel_shape == 'bigaussian':
            sigma_rs = self.kernel_params[:, 3]
        else:
            sigma_rs = None
        # precompute kernel functions
        kernels = self._get_kernels(norm_mode='height')
        # loop over spectra
        for c, spectrum_idx in enumerate(spectrum_idxs):
            print(f'setting loss for spectrum {c + 1} out of {len(spectrum_idxs)} ...')
            # get index in array corresponding to spectrum index
            array_idx = self.spectrum_idx2array_idx(spectrum_idx)
            spec: np.ndarray[float] = \
                reader.get_spectrum_resampled_intensities(spectrum_idx)

            Hs = self.H_from_area(
                self.line_spectra[array_idx, :],
                sigma_ls,
                sigma_rs
            )
            y_rec = kernels @ Hs
            loss = np.sum(np.abs(spec - y_rec)) / np.sum(spec)
            self.losses[c] = loss

            if plts:
                plt.figure()
                plt.plot(spec.mzs, spec, label='original')
                plt.plot(self.mzs, y_rec, label='reconstructed')
                plt.legend()
                plt.xlim(self.limits)
                plt.title(f'Reconstruction loss: {loss:.3f}')
                plt.show()

    def copy(self) -> Spectra:
        c = Spectra(path_d_folder=self.path_d_folder, initiate=False)
        c.__dict__ = self.__dict__.copy()
        return c

    def reconstruct_all(self) -> np.ndarray[float]:
        """Calcualte reconstruct profiles from line_spectra and peak properties."""
        kernels: np.ndarray[float] = self._get_kernels(norm_mode='height')
        line_spectra: np.ndarray[float] = self.get_heights()
        reconstructed: np.ndarray[float] = kernels @ line_spectra.T
        return reconstructed

    def combine_with(self, other: object) -> Spectra:
        """
        Combine this spectra object with another one. 
        If spectra objects were created from different d_folders, 
        the index functionality is lost. mz values are inherited from the first object, 
        summed intensities are combined, 
        """
        assert type(other) == Spectra, '"+" is only defined for Spectra objects'
        assert hasattr(self, 'noise_level') and hasattr(other, 'noise_level'), \
            'if you want to (re)bin, make sure both objects have the baseline removed'

        # determine if spectra are from the same source folder
        is_same_measurement: bool = self.path_d_folder == other.path_d_folder
        if is_same_measurement:
            assert set(self.indices) & set(other.indices) == set(), \
                'spectra objects must not contain the same spectrum twice'
            path_d_folder = self.path_d_folder
            indices: np.ndarray[int] = np.hstack([self.indices, other.indices])
        else:
            print('this and other object are not from the same folder, this will result \
in loss of functionality!')
            path_d_folder = os.path.commonpath([
                self.path_d_folder, other.path_d_folder
            ])
            indices = None

        s_new: Spectra = Spectra(
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

    def filter_line_spectra(
            self, SNR_threshold: float = 0, intensity_min: float = 0, **_: dict
    ) -> np.ndarray[bool]:
        """Set the intensities that fall below SNR or min intensity to zero and return array of changed pixels."""
        if SNR_threshold > 0:
            assert hasattr(self, 'noise_level'), 'Call subtract_baseline first'
            mask_snr_too_low: np.ndarray[bool] = self.get_SNR_table() < SNR_threshold
        else:
            mask_snr_too_low: np.ndarray[bool] = np.zeros(self.line_spectra.shape, dtype=bool)
        if intensity_min > 0:
            mask_intensity_too_low: np.ndarray[bool] = self.line_spectra < intensity_min
        else:
            mask_intensity_too_low: np.ndarray[bool] = np.zeros(self.line_spectra.shape, dtype=bool)

        mask = mask_snr_too_low | mask_intensity_too_low
        self.line_spectra[mask] = 0

        return mask

    def full(self, reader: ReadBrukerMCF | hdf5Handler, **kwargs: dict):
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(**kwargs)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        self.bin_spectra(reader, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(reader=reader, **kwargs)

    def full_targeted(self, reader: ReadBrukerMCF | hdf5Handler, targets: list[float], **kwargs):
        self.add_all_spectra(reader=reader)
        self.subtract_baseline(**kwargs)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        # set target compounds
        self.set_targets(targets, plts=True)
        self.bin_spectra(reader=reader, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(reader=reader, **kwargs)

    def save(self):
        """Save object to d-folder."""
        dict_backup = self.__dict__.copy()
        # dont save feature table AND line spectra
        if hasattr(self, 'feature_table') and hasattr(self, 'line_spectra'):
            self.__delattr__('line_spectra')
        keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
        existent_attributes = list(self.__dict__.keys())
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)

        file = self.path_d_folder + '/' + 'spectra_object.pickle'
        with open(file, 'wb') as inp:
            pickle.dump(self, inp, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup

    def load(self):
        """Load object from d-folder."""
        file = self.path_d_folder + '/' + 'spectra_object.pickle'
        with open(file, 'rb') as inp:
            self.__dict__ = pickle.load(inp).__dict__

        if hasattr(self, 'feature_table'):
            self.line_spectra = self.feature_table.\
                drop(columns=['R', 'x', 'y']).\
                to_numpy()


class ClusteringManager:
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
            print(
                'chunk size likely too large to be held in memory, this ' +
                'will result in significant performance drop, the maximum ' +
                'recommended chunk size with the specified sample rate is ' +
                f'{max_chunk_size_cal}'
            )
        elif max_chunk_size is None:
            max_chunk_size: int = max_chunk_size_cal
        if max_chunk_size_cal < min_chunk_size:
            print(
                f'the requested minimum chunk size {min_chunk_size} is bigger ' +
                f'than the recommended max {max_chunk_size_cal}'
            )

        N_chunks_cal: int = np.ceil(self.N_spectra / max_chunk_size).astype(int)
        if (N_chunks is not None) and (N_chunks < N_chunks_cal):
            print(
                'chunk size likely too large to be held in memory, this ' +
                'will result in significant performance drop, the maximum ' +
                'recommended chunk number with the specified sample rate is ' +
                f'{N_chunks_cal}'
            )
        elif N_chunks is None:
            N_chunks = N_chunks_cal
        self.N_chunks: int = N_chunks

        print(max_chunk_size_cal, N_chunks_cal)

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
        print(xs, ys, cs)
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
    def __init__(
            self, 
            readers: list[ReadBrukerMCF | hdf5Handler]):
        self.specs: list[Spectra | None] = [None] * len(readers)
        self._initiate(readers)

    def _initiate(self, readers: list[ReadBrukerMCF | hdf5Handler]) -> None:
        assert len(readers) > 0, 'pass at least one reader'
        reader = readers[0]
        assert all([r.limits[0] == reader.limits[0] for r in readers]), 'readers must have the same limits'
        assert all([r.limits[1] == reader.limits[1] for r in readers]), 'readers must have the same limits'
        indices = []
        offset = 0
        for i, reader in enumerate(readers):
            spec: Spectra = Spectra(reader=reader)
            self.specs[i] = spec.copy()
            idxs = spec.indices.copy()
            idxs += offset
            indices.append(idxs)
            offset = idxs[-1]
        self.mzs: np.ndarray[float] = spec.mzs.copy()
        self.intensities: np.ndarray[float] = np.zeros_like(self.mzs)
        self.indices = np.hstack(indices)

    def add_all_spectra(self, readers: list[ReadBrukerMCF | hdf5Handler]):
        for i, reader in enumerate(readers):
            self.specs[i].add_all_spectra(reader)
            self.intensities += self.specs[i].intensities

    def distribute_peaks_and_kernels(self):
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
        line_spectra = []
        for i, reader in enumerate(readers):
            self.specs[i].bin_spectra(reader, **kwargs)
            line_spectra.append(self.specs[i].line_spectra.copy())
        self.line_spectra: np.ndarray[float] = np.vstack(line_spectra)
        self.binning_by = self.specs[0].binning_by

    def binned_spectra_to_df(
            self,
            readers: Iterable[ReadBrukerMCF | hdf5Handler] | None = None,
            **kwargs
    ) -> pd.DataFrame:
        if readers is None:
            readers = [None] * len(self.specs)

        fts = []
        for i, reader in enumerate(readers):
            self.specs[i].binned_spectra_to_df(reader=reader)
            fts.append(self.specs[i].feature_table)

        self.feature_table: pd.DataFrame = combine_feature_tables(fts)
        return self.feature_table

    def full(self, readers: list[ReadBrukerMCF | hdf5Handler], **kwargs: dict):
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(**kwargs)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        self.distribute_peaks_and_kernels()
        self.bin_spectra(readers=readers, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(readers=readers, **kwargs)

    def full_targeted(self, readers: list[ReadBrukerMCF | hdf5Handler], targets: list[float], **kwargs):
        self.add_all_spectra(readers=readers)
        self.subtract_baseline(**kwargs)
        self.set_peaks(**kwargs)
        self.set_kernels(**kwargs)
        # set target compounds
        self.set_targets(targets, plts=True)
        self.distribute_peaks_and_kernels()
        self.bin_spectra(readers=readers, **kwargs)
        self.filter_line_spectra(**kwargs)
        self.binned_spectra_to_df(readers=readers, **kwargs)