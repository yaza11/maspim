"""This module allows the usage of functions from the R rtms package in python."""
import os
# specify the R installation folder here (required by rpy2 package)
R_HOME = r"C:\Program Files\R\R-4.3.2"  # your installation path here
os.environ["R_HOME"] = R_HOME  # adding R_HOME folder to environment parameters
os.environ["PATH"]   = R_HOME + ";" + os.environ["PATH"]  # and to system path 

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
        if limits is not None:
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
        """Resample mzs and intensities to regular intervals."""
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
        ints_ip = np.interp(mzs_ip, self.mzs, self.intensities)
        # overwrite
        self.mzs = mzs_ip
        self.intensities = ints_ip
        
    def to_pandas(self):
        df = pd.DataFrame({'mz': self.mzs, 'intensity': self.intensities})
        return df
    
    def copy(self):
        rspectrum = [self.mzs.copy(), self.intensities.copy()]
        new_spectrum = Spectrum(rspectrum)
        return new_spectrum
    
            
class ReadBrukerMCF:
    def __init__(self, path_d_folder: str):
        self.path_d_folder = path_d_folder
        
    def create_reader(self):
        print('creating BrukerMCF reader ...')
        self.reader = rtms.newBrukerMCFReader(self.path_d_folder)
        print('done creating reader')
        
    def create_indices(self):
        assert hasattr(self, 'reader'), 'create a reader with create_reader first'
        self.indices = np.array(rtms.getBrukerMCFIndices(self.reader))
        
    def get_spectrum(self, index : int, **kwargs) -> Spectrum:
        rspectrum = rtms.getSpectrum(self.reader, index)
        # convert to python
        spectrum = Spectrum(rspectrum, **kwargs)
        return spectrum
    
    def set_meta_data(self):
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
        assert hasattr(self, 'metaData'), 'call set_meta_data first'
        # find entries
        idx_center = self.metaData.PermanentName == 'Q1Mass'
        idx_size = self.metaData.PermanentName == 'Q1Res'
        # only keep value (not 1320.0 m/z)
        mass_center = float(self.metaData.loc[idx_center, 'Value'].iat[0].split()[0])
        mass_window_size = float(self.metaData.Value[idx_size].iat[0].split()[0])
        self.limits = (mass_center - mass_window_size / 2, mass_center + mass_window_size / 2)
        
        
    
    def create_spots(self):
        assert hasattr(self, 'reader'), 'create a reader with create_reader first'
        print('creating spots table ...')
        rspots = rtms.getBrukerMCFSpots(self.reader)
        self.spots = Spots(rspots)
        print('done creating spots table')
    
    def get_spectrum_by_spot(self, spot: str):
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
    def __init__(
            self, 
            reader: ReadBrukerMCF,
            limits: tuple[float] = None, 
            delta_mz: float = 1e-4,
            indices: Iterable = None,
            load = False
    ):
        self.path_d_folder = reader.path_d_folder
        if load:
            self.load()
        else:
            self.limits = limits
            self.delta_mz = delta_mz
            self.initiate(reader, indices, limits)
        
    def set_masses(self):
        smallest_mz = int(self.limits[0] / self.delta_mz) * self.delta_mz
        # round to next biggest multiple of delta_mz
        biggest_mz = (int(self.limits[1] / self.delta_mz) + 1) * self.delta_mz
        # equally spaced
        self.mzs =  np.arange(smallest_mz, biggest_mz + self.delta_mz, self.delta_mz)
        self.intensities = np.zeros_like(self.mzs)
    
    def initiate(self, reader, indices, limits):
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
        # spectrum = spectrum.copy()
        # check if resampling is necessary
        dmzs = np.diff(spectrum.mzs)
        if not np.allclose(dmzs[1:], dmzs[0]) or dmzs[0] != self.delta_mz:
            spectrum.resample(self.mzs)

        self.intensities += spectrum.intensities
    
    def add_all_spectra(self, reader):
        if not hasattr(reader, 'indices'):
            reader.create_indices()
        indices = reader.indices
        N = len(indices)
        print(f'adding up {N} spectra ...')
            
        time0 = time.time()
        for it, index in enumerate(indices):
            spectrum = reader.get_spectrum(int(index)) 
            self.add_spectrum(spectrum)
            time_now = time.time()
            if it % 10 ** (np.around(np.log10(98), 0) - 2) == 0:
                time_elapsed = time_now - time0
                predict = time_elapsed * N / (it + 1)
                print(f'estimated time left: {(predict - time_elapsed):.1f} s')
        # subtract baseline
        self.intensities -= self.intensities.min()
        print('done adding up spectra')
        
    def set_peaks(self, prominence=None, width=3, **kwargs):
        if prominence is None:
            median = np.median(self.intensities)
            prominence = .1 * median
        
        self.peaks, self.properties = find_peaks(
            self.intensities, prominence=prominence, width=width, **kwargs
        )
    
    def bigaussian_from_peak(self, peak_idx):
        """Find kernel parameters for a peak with the shape of a bigaussian."""
        assert hasattr(self, 'peaks'), 'call set_peaks first' 
        mz_idx = self.peaks[peak_idx]  # mz index of of center 
        mz_c = self.mzs[mz_idx] # center of gaussian
        # height at center of peak - prominence
        I0 = self.intensities[mz_idx] - self.properties['prominences'][peak_idx]
        H = self.intensities[mz_idx] - I0 # corresponding height
        # width of peak at half maximum
        FWHM_l = self.mzs[
            (self.properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r = self.mzs[
            (self.properties["right_ips"][peak_idx] + .5).astype(int)
        ]
        # convert FWHM to standard deviation
        sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
        sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))        
        return mz_c, I0, H, sigma_l, sigma_r
    
    @staticmethod
    def bigaussian(x: np.ndarray, x_c, y0, H, sigma_l, sigma_r):
        """Evaluate bigaussian for mass vector based on parameters."""
        x_l = x[x <= x_c]
        x_r = x[x > x_c]
        y_l =  y0 + H * np.exp(-1/2 * ((x_l - x_c) / sigma_l) ** 2)
        y_r =  y0 + H * np.exp(-1/2 * ((x_r - x_c) / sigma_r) ** 2)
        return np.hstack([y_l, y_r])
    
    def set_kernels(self):
        """Based on the peak properties, find bigaussian parameters to approximate spectrum."""
        assert hasattr(self, 'peaks'), 'call set peaks first'
        self.kernel_params = np.zeros((len(self.peaks), 5))
        for idx in range(len(self.peaks)):
            self.kernel_params[idx, :] = self.bigaussian_from_peak(idx)
        # vertical shifts get taken care of by taking sum
        self.kernel_params[:, 1] = 0
        
    def plt_kernels(self):
        assert hasattr(self, 'kernel_params'), 'call set_kernels first'
        # calculate approximated signal by summing up kernels
        intensities_approx = np.zeros_like(self.intensities)
        for i in range(len(self.peaks)):
            intensities_approx += self.bigaussian(
                self.mzs, *self.kernel_params[i, :]
            )
            
        plt.figure()
        plt.plot(self.mzs, self.intensities, label='summed intensity')
        plt.plot(self.mzs, intensities_approx, label='estimated')
        plt.legend()
        plt.show()
        
    def bin_spectra(self, reader):
        def _bin_spectrum(spectrum, idx):
            if (len(spectrum.mzs) != len(self.mzs)) \
                    or (not np.allclose(spectrum.mzs, self.mzs)):
                spectrum.resample(self.mzs)
            # line_spectrum = np.zeros(N_peaks)
            # for idx_peak in range(N_peaks):
            #     weighted_signal = spectrum.intensities * bigaussians[idx_peak, :]
            #     # weight is the integrated weighted signal
            #     # line_spectrum[idx_peak] = np.trapz(weighted_signal, x=self.mzs)
            #     line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
            line_spectrum = (spectrum.intensities @ bigaussians) * dmz
            self.line_spectra[idx, :] = line_spectrum
        
        assert hasattr(self, 'kernel_params'), 'calculate kernels with set_kernels'
        if not hasattr(reader, 'indices'):
            reader.create_indices()
        
        indices_spectra = reader.indices
        N_spectra = len(indices_spectra)
        N_peaks = len(self.peaks)
        self.line_spectra = np.zeros((N_spectra, N_peaks))
        
        # precompute bigaussians
        dmz = self.mzs[1] - self.mzs[0]
        bigaussians = np.zeros((N_peaks, len(self.mzs)))
        for idx_peak in range(N_peaks):
            # x_c, y0, H, sigma_l, sigma_r
            sigma_l = self.kernel_params[idx_peak, 3]
            sigma_r = self.kernel_params[idx_peak, 4]
            H = np.sqrt(2 / np.pi) / (sigma_l + sigma_r)  # normalization constant
            bigaussians[idx_peak] = self.bigaussian(
                self.mzs, 
                x_c=self.kernel_params[idx_peak, 0], 
                y0=0,  # assert spectra have baseline subtracted
                H=H,  # normalized kernels
                sigma_l=sigma_l, 
                sigma_r=sigma_r
            )
        bigaussians = bigaussians.T
        
        # iterate over spectra and bin according to kernels
        print(f'binning {N_spectra} spectra into {N_peaks} bins ...')
        time0 = time.time()
        for it, idx_spectrum in enumerate(indices_spectra):
            spectrum = reader.get_spectrum(int(idx_spectrum)) 
            _bin_spectrum(spectrum, it)
            time_now = time.time()
            if it % 10 ** (np.around(np.log10(N_spectra), 0) - 2) == 0:
                time_elapsed = time_now - time0
                predict = time_elapsed * N_spectra / (it + 1)
                print(f'estimated time left: {(predict - time_elapsed):.1f} s')
        print('done binning spectra')
        
    def binned_spectra_to_df(self, reader):
        assert hasattr(self, 'line_spectra'), 'create line spectra with bin_spectra'
        if not hasattr(self.reader, 'spots'):
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
        
        
        
    def save(self):
        file = self.path_d_folder + '/' + 'spectra_object.pickle'
        with open(file, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        
    def load(self):
        file = self.path_d_folder + '/' + 'spectra_object.pickle'
        with open(file, 'rb') as inp:
            self.__dict__ = pickle.load(inp).__dict__
        
            
    
if __name__ == '__main__':
    pass
    
    
            
        

