"""This module allows the usage of functions from the R rtms package in python."""
import os
# specify the R installation folder here (required by rpy2 package)
R_HOME = r"C:\Program Files\R\R-4.3.2"  # your installation path here
os.environ["R_HOME"] = R_HOME  # adding R_HOME folder to environment parameters
os.environ["PATH"]   = R_HOME + ";" + os.environ["PATH"]  # and to system path 

from util.manage_obj_saves import class_to_attributes

from typing import Iterable 
import re

import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import pickle
from scipy.signal import find_peaks

from rpy2.robjects.packages import importr, data, isinstalled
from rpy2.robjects import pandas2ri

# install package if not found
if not isinstalled("rtms"):
    utils = importr("utils")
    utils.install_packages("rtms")

# import package
rtms = importr('rtms')


class Spots:
    def __init__(self, rspots):
        """Convert rtms spot to python."""
        self.idxs = np.array(rspots[0])
        self.names = np.array(rspots[1])
        self.times = np.array(rspots[2])
        
        
class Spectrum:
    def __init__(self, rspectrum, limits: tuple[float] | None = None):
        """
        Convert rtms spectrum to python.

        Parameters
        ----------
        rspectrum : rtms.Spectrum
            The rtms object to convet.
        limits : tuple[float] | None, optional
            mz limits to crop the spectrum as a tuple with upper and lower bound. 
            The default is None and will not crop the spectrum.

        Returns
        -------
        None.

        """
        self.mzs = np.array(rspectrum[0])
        self.intensities = np.array(rspectrum[1])
        if limits is not None:  # if limits provided, crop spectrum to interval
            mask = (self.mzs >= limits[0]) & (self.mzs <= limits[1])
            self.mzs = self.mzs[mask]
            self.intensities = self.intensities[mask]
            
    def plot(self, *args, limits=None, hold=False, **kwargs):
        if len(args) == 0:
            args = ['+-']
        if limits is None:
            mask = np.ones_like(self.mzs, dtype=bool)
        else:
            assert limits[0] < limits[1], 'left bound has to be smaller, e.g. limits=(600, 2000)'
            mask = (self.mzs >= limits[0]) & (self.mzs <= limits[1])
        mzs = self.mzs[mask]
        intensities = self.intensities[mask]
        fig, ax = plt.subplots()
        plt.plot(mzs, intensities, *args, **kwargs)
        plt.xlabel('m/z in Da')
        plt.ylabel('Intensities')
        
        if hold:
            return fig, ax
        plt.show()
        
    def resample(self, delta_mz: float | Iterable[float] = 1e-4):
        """
        Resample mzs and intensities to regular intervals.
        
        Provide either the equally spaced mz values at which to resample or 
        the distance for regular intervals resampling.
        """
        if type(delta_mz) in (float, int):
            # create mzs spaced apart with specified precision
            # round to next smallest multiple of delta_mz
            smallest_mz = int(self.mzs.min() / delta_mz) * delta_mz
            # round to next biggest multiple of delta_mz
            biggest_mz = (int(self.mzs.max() / delta_mz) + 1) * delta_mz
            # equally spaced
            mzs_ip = np.arange(smallest_mz, biggest_mz + delta_mz, delta_mz)
        else:
            dmzs = np.diff(delta_mz)
            assert np.allclose(dmzs[1:], dmzs[0]), \
                'passed delta_mz must either be float or list of equally spaced mzs'
            mzs_ip = delta_mz
        # interpolate to regular spaced mz values
        ints_ip = np.interp(mzs_ip, self.mzs, self.intensities)
        # overwrite objects mz vals and intensities
        self.mzs = mzs_ip
        self.intensities = ints_ip
        
    def to_pandas(self):
        """Return mass and intensity as pandas dataframe."""
        df = pd.DataFrame({'mz': self.mzs, 'intensity': self.intensities})
        return df
    
    def copy(self):
        """Return copy of object."""
        rspectrum = [self.mzs.copy(), self.intensities.copy()]
        new_spectrum = Spectrum(rspectrum)
        return new_spectrum
    
            
class ReadBrukerMCF:
    """Python version of rtms ReadBrukerMCF."""
    def __init__(self, path_d_folder: str):
        self.path_d_folder = path_d_folder
        
    def create_reader(self):
        """Create a new BrukerMCFReader object."""
        print('creating BrukerMCF reader ...')
        self.reader = rtms.newBrukerMCFReader(self.path_d_folder)
        print('done creating reader')
        
    def create_indices(self):
        """Create indices of spectra in mcf file."""
        assert hasattr(self, 'reader'), 'create a reader with create_reader first'
        self.indices = np.array(rtms.getBrukerMCFIndices(self.reader))
        
    def create_spots(self):
        """Create spots object with indices and names."""
        assert hasattr(self, 'reader'), 'create a reader with create_reader first'
        print('creating spots table ...')
        rspots = rtms.getBrukerMCFSpots(self.reader)
        self.spots = Spots(rspots)
        print('done creating spots table')
    
    def set_meta_data(self):
        """Fetch metadata for measurement from mcf file and turn into df."""
        # arbitrary index, metaData should be the same for all spectra
        metaData = rtms.getBrukerMCFAllMetadata(self.reader, index=1)
        self.metaData = pd.DataFrame({
            'Index': np.array(metaData[0]),
            'PermanentName': np.array(metaData[1]),
            'GroupName': np.array(metaData[2]),
            'DisplayName': np.array(metaData[3]),
            'Value': np.array(metaData[4])
        })
    
    def set_QTOF_window(self):
        """Set mass window limits from QTOF values in metadata."""
        assert hasattr(self, 'metaData'), 'call set_meta_data first'
        # find entries
        idx_center = self.metaData.PermanentName == 'Q1Mass'
        idx_size = self.metaData.PermanentName == 'Q1Res'
        # only keep value (not 1320.0 m/z)
        mass_center = float(self.metaData.loc[idx_center, 'Value'].iat[0].split()[0])
        mass_window_size = float(self.metaData.Value[idx_size].iat[0].split()[0])
        self.limits = (mass_center - mass_window_size / 2, mass_center + mass_window_size / 2)
        
    def get_spectrum(self, index : int, **kwargs) -> Spectrum:
        """Get spectrum in mcf file by index (R index, so 1-based)."""
        rspectrum = rtms.getSpectrum(self.reader, index)
        # convert to python
        spectrum = Spectrum(rspectrum, **kwargs)
        return spectrum
    
    def get_spectrum_by_spot(self, spot: str):
        """Get spectrum by spot-name (e.g. R00X102Y80)."""
        assert hasattr(self, 'spots'), 'create spots with create_spots first'
        # find corresponding index 
        # index in spots may be shifted or have missing values
        matches = np.argwhere(self.spots.names == spot)[0]  # index corresponding to name in the spots table 
        if len(matches) == 0:
            print(f'{spot} not a valid spot, check spots.names')
            return None
        idx_spot = matches[0]
        # the corresponding value
        # int is necessary because rpy2 is picky about types and argwhere
        # returns np.intc, not int
        idx_spectrum = int(self.spots.idxs[idx_spot])  
        print(idx_spot, idx_spectrum)
        spectrum = self.get_spectrum(idx_spectrum)
        return spectrum
    
    
class Spectra:
    """Container for multiple Spectrum objects and binning."""
    def __init__(
            self, 
            reader: ReadBrukerMCF | None = None,
            limits: tuple[float] = None, 
            delta_mz: float = 1e-4,
            indices: Iterable = None,
            load: bool = False,
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
        load : bool, optional
            Load the object from the d-folder. The default is False.
        path_d_folder : str | None, optional
            folder with data. The default is str | None. Only necessary if reader
            is not passed.

        Returns
        -------
        None.

        """
        assert reader or (load and path_d_folder), \
            'Either pass a reader or load and the corresponding d-folder'
        
        if load:
            self.path_d_folder = path_d_folder
            self.load()
        else:
            self.path_d_folder = reader.path_d_folder
            self.delta_mz = delta_mz
            self.initiate(reader, indices, limits)
        
    def set_masses(self):
        """Initiate masses and intensities summed spectrum."""
        # round to next smallest multiple of delta_mz
        smallest_mz = int(self.limits[0] / self.delta_mz) * self.delta_mz
        # round to next biggest multiple of delta_mz
        biggest_mz = (int(self.limits[1] / self.delta_mz) + 1) * self.delta_mz
        # equally spaced
        self.mzs =  np.arange(smallest_mz, biggest_mz + self.delta_mz, self.delta_mz)
        self.intensities = np.zeros_like(self.mzs)
    
    def initiate(self, reader, indices, limits):
        """Set limits and masses."""
        if indices is None:
            if not hasattr(reader, 'indices'):
                reader.create_indices()    
            indices = reader.indices
        self.indices = indices
        if limits is None:
            if not hasattr(reader, 'metaData'):
                reader.set_meta_data()
            reader.set_QTOF_window()
            limits = reader.limits
        self.limits = limits
        self.set_masses()
    
    def add_spectrum(self, spectrum: Spectrum):
        """Add passed spectrum values to summed spectrum."""
        # spectrum = spectrum.copy()
        # check if resampling is necessary
        dmzs = np.diff(spectrum.mzs)
        if not np.allclose(dmzs[1:], dmzs[0]) or dmzs[0] != self.delta_mz:
            spectrum.resample(self.mzs)

        self.intensities += spectrum.intensities
    
    def add_all_spectra(self, reader):
        """Add up all spectra found in the mcf file."""
        if not hasattr(reader, 'indices'):
            reader.create_indices()
        indices = reader.indices
        N = len(indices)
        print(f'adding up {N} spectra ...')
            
        time0 = time.time()
        # iterate over all spectra
        for it, index in enumerate(indices):
            spectrum = reader.get_spectrum(int(index)) 
            self.add_spectrum(spectrum)
            time_now = time.time()
            if it % 10 ** (np.around(np.log10(N), 0) - 2) == 0:
                time_elapsed = time_now - time0
                predict = time_elapsed * N / (it + 1)
                print(f'estimated time left: {(predict - time_elapsed):.1f} s')
        # subtract baseline
        self.intensities -= self.intensities.min()
        print('done adding up spectra')
        
    def set_peaks(self, prominence: float | None = None, width=3, **kwargs):
        """
        Find peaks in summed spectrum using scipy's find_peaks function.

        Parameters
        ----------
        prominence : float, optional
            Required prominence for peaks. The default is None. This defaults
            to 10 % of the median intensity
        width : int, optional
            Minimum number of points between peaks. The default is 3.
        **kwargs : dict
            Additional kwargs for find_peaks.

        Sets peaks and properties

        """
        if prominence is None:
            median = np.median(self.intensities)
            prominence = .1 * median
        
        self.peaks, self.peak_properties = find_peaks(
            self.intensities, prominence=prominence, width=width, **kwargs
        )
        
        # save parameters to dict for later reference
        self.peak_setting_parameters = kwargs
        self.peak_setting_parameters['prominence'] = prominence
        self.peak_setting_parameters['width'] = width
        
    def gaussian_from_peak(self, peak_idx):
        """Find kernel parameters for a peak with the shape of a bigaussian."""
        assert hasattr(self, 'peaks'), 'call set_peaks first' 
        mz_idx = self.peaks[peak_idx]  # mz index of of center 
        
        # height at center of peak - prominence
        I0 = self.intensities[mz_idx] - self.peak_properties['prominences'][peak_idx]
        H = self.intensities[mz_idx] - I0 # corresponding height
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
        return mz_c, I0, H, sigma_l, sigma_r
    
    def bigaussian_from_peak(self, peak_idx: int):
        """Find kernel parameters for a peak with the shape of a bigaussian."""
        assert hasattr(self, 'peaks'), 'call set_peaks first' 
        mz_idx = self.peaks[peak_idx]  # mz index of of center 
        mz_c = self.mzs[mz_idx] # center of gaussian
        # height at center of peak - prominence
        I0 = self.intensities[mz_idx] - self.peak_properties['prominences'][peak_idx]
        H = self.intensities[mz_idx] - I0 # corresponding height
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
        return mz_c, I0, H, sigma_l, sigma_r
    
    @staticmethod
    def gaussian(x: np.ndarray, x_c, y0, H, sigma):
        return y0 + H * np.exp(-1/2 * ((x - x_c) / sigma) ** 2)
    
    @staticmethod
    def bigaussian(x: np.ndarray, x_c, y0, H, sigma_l, sigma_r):
        """
        Evaluate bigaussian for mass vector based on parameters.

        Parameters
        ----------
        x : np.ndarray
            mass vector.
        x_c : float
            mass at center of peak
        y0 : float
            vertical offset of peak
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
        y_l =  y0 + H * np.exp(-1/2 * ((x_l - x_c) / sigma_l) ** 2)
        y_r =  y0 + H * np.exp(-1/2 * ((x_r - x_c) / sigma_r) ** 2)
        return np.hstack([y_l, y_r])
    
    @property
    def kernel_func(self):
        if self.kernel_shape == 'bigaussian':
            return self.bigaussian
        elif self.kernel_shape == 'gaussian':
            return self.gaussian
        
    @property
    def kernel_func_from_peak(self):
        if self.kernel_shape == 'bigaussian_from_peak':
            return self.bigaussian
        elif self.kernel_shape == 'gaussian_from_peak':
            return self.gaussian
    
    def set_kernels(self, use_bigaussian=False):
        """
        Based on the peak properties, find bigaussian parameters to 
        approximate spectrum. Creates kernel_params where cols correspond to 
        peaks and rows different properties. Properties are: m/z, vertical 
        shift, intensity at max, sigma left, sigma right
        """
        assert hasattr(self, 'peaks'), 'call set peaks first'
        if use_bigaussian: 
            self.kernel_shape = 'bigaussian'
            self.kernel_params = np.zeros((len(self.peaks), 5))
        else:
            self.kernel_shape = 'gaussian'
            self.kernel_params = np.zeros((len(self.peaks), 4))
        kernel_func_from_peak = self.kernel_func_from_peak
        
        for idx in range(len(self.peaks)):
            self.kernel_params[idx, :] = kernel_func_from_peak(idx)
        # vertical shifts get taken care of by taking sum
        self.kernel_params[:, 1] = 0
        
    def plt_summed(self, plt_kernels=False):
        assert hasattr(self, 'kernel_params'), 'call set_kernels first'
        # calculate approximated signal by summing up kernels
        intensities_approx = np.zeros_like(self.intensities)
        plt.figure()
        
        for i in range(len(self.peaks)):
            y = self.bigaussian(self.mzs, *self.kernel_params[i, :])
            intensities_approx += y
            if plt_kernels:
                plt.plot(self.mzs, y)
        plt.plot(self.mzs, self.intensities, label='summed intensity')
        plt.plot(self.mzs, intensities_approx, label='estimated')
        plt.legend()
        plt.show()
        
    def bin_spectra(self, reader: ReadBrukerMCF):
        """For each spectrum find overlap between kernels and signal."""
        def _bin_spectrum(spectrum, idx):
            """Find intensities of compound based on kernels."""
            if (len(spectrum.mzs) != len(self.mzs)) \
                    or (not np.allclose(spectrum.mzs, self.mzs)):
                spectrum.resample(self.mzs)
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
            line_spectrum = (spectrum.intensities @ kernels) * dmz
            self.line_spectra[idx, :] = line_spectrum
        
        assert hasattr(self, 'kernel_params'), 'calculate kernels with set_kernels'
        if not hasattr(reader, 'indices'):
            reader.create_indices()
        
        indices_spectra = reader.indices
        N_spectra = len(indices_spectra)  # number of spectra in mcf file
        N_peaks = len(self.peaks)  # number of identified peaks
        self.line_spectra = np.zeros((N_spectra, N_peaks))  # result array
        
        # precompute bigaussians
        dmz = self.mzs[1] - self.mzs[0]
        kernels = np.zeros((N_peaks, len(self.mzs)))
        if self.kernel_shape == 'bigaussian':
            for idx_peak in range(N_peaks):
                # x_c, y0, H, sigma_l, sigma_r
                sigma_l = self.kernel_params[idx_peak, 3]
                sigma_r = self.kernel_params[idx_peak, 4]
                H = np.sqrt(2 / np.pi) / (sigma_l + sigma_r)  # normalization constant
                kernels[idx_peak] = self.bigaussian(
                    self.mzs, 
                    x_c=self.kernel_params[idx_peak, 0], 
                    y0=0,  # assert spectra have baseline subtracted
                    H=H,  # normalized kernels
                    sigma_l=sigma_l, 
                    sigma_r=sigma_r
                )
            kernels = kernels.T
        elif self.kernel_shape == 'gaussian':
            for idx_peak in range(N_peaks):
                # x_c, y0, H, sigma_l, sigma_r
                sigma = self.kernel_params[idx_peak, 3]
                H = 1 / (np.sqrt(2 * np.pi) * sigma)  # normalization constant
                kernels[idx_peak] = self.gaussian(
                    self.mzs, 
                    x_c=self.kernel_params[idx_peak, 0], 
                    y0=0,  # assert spectra have baseline subtracted
                    H=H,  # normalized kernels
                    sigma=sigma
                )
            kernels = kernels.T
        
        # iterate over spectra and bin according to kernels
        print(f'binning {N_spectra} spectra into {N_peaks} bins ...')
        time0 = time.time()
        for it, idx_spectrum in enumerate(indices_spectra):
            spectrum = reader.get_spectrum(int(idx_spectrum)) 
            _bin_spectrum(spectrum, it)
            if it % 10 ** (np.around(np.log10(N_spectra), 0) - 2) == 0:
                time_now = time.time()
                time_elapsed = time_now - time0
                predict = time_elapsed * N_spectra / (it + 1)
                print(f'estimated time left: {(predict - time_elapsed):.1f} s')
        print('done binning spectra')
        
    def binned_spectra_to_df(self, reader: ReadBrukerMCF):
        """Turn the line_spectra into the familiar df with R, x, y columns."""
        if hasattr(self, 'feature_table'):
            return self.feature_table
        assert hasattr(self, 'line_spectra'), 'create line spectra with bin_spectra'
        if not hasattr(reader, 'spots'):
            reader.create_spots()
        
        df = pd.DataFrame(
            data=self.line_spectra.copy(), 
            columns=np.around(self.kernel_params[:, 0], 4).astype(str)
        )
        # add R, x, y columns
        names = reader.spots.names
        str_prefix = r'R(\d+)X'
        str_x = r'R\d+X(.*?)Y'
        str_y = r'Y(.*?)$'
        
        def rxy(name):
            r = int(re.findall(str_prefix, name)[0])
            x = int(re.findall(str_x, name)[0])
            y = int(re.findall(str_y, name)[0])
            return [r, x, y]
        RXYs = np.array([rxy(name) for name in names])
        df['R'] = RXYs[:, 0]
        df['x'] = RXYs[:, 1]
        df['y'] = RXYs[:, 2]
        self.feature_table = df
        return self.feature_table
    
    def get_kernel_params_df(self):
        assert hasattr(self, 'kernel_params'), 'call set_kernels'
        if self.kernel_shape == 'bigaussian':
            columns = ['mz', 'I0', 'H', 'sigma_l', 'sigma_r']
        elif self.kernel_shape == 'gaussian':
            columns = ['mz', 'I0', 'H', 'sigma']
        df = pd.DataFrame(data=self.kernel_params, columns=columns)
        return df
        
    def save(self):
        """Save object to d-folder."""
        dict_backup = self.__dict__.copy()
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
        

if __name__ == '__main__':
    pass
    
    
            
        

