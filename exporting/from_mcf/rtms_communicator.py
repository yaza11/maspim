"""This module allows the usage of functions from the R rtms package in python."""
from exporting.from_mcf.helper import get_r_home

from functools import lru_cache
from typing import Iterable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


# specify the R installation folder here (required by rpy2 package)
try:
    R_HOME = get_r_home()
except EnvironmentError:
    R_HOME = r"C:\Program Files\R\R-4.3.2"  # your installation path here
os.environ["R_HOME"] = R_HOME  # adding R_HOME folder to environment parameters
os.environ["PATH"] = R_HOME + ";" + os.environ["PATH"]  # and to system path

from rpy2.robjects.packages import importr, isinstalled

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
            assert limits[0] < limits[
                1], 'left bound has to be smaller, e.g. limits=(600, 2000)'
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

    def resample(
            self,
            delta_mz: float | Iterable[float] = 1e-4,
            check_intervals=False
    ):
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
        assert hasattr(
            self, 'reader'), 'create a reader with create_reader first'
        self.indices = np.array(rtms.getBrukerMCFIndices(self.reader))

    def create_spots(self):
        """Create spots object with indices and names."""
        assert hasattr(
            self, 'reader'), 'create a reader with create_reader first'
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
        mass_center = float(
            self.metaData.loc[idx_center, 'Value'].iat[0].split()[0])
        mass_window_size = float(
            self.metaData.Value[idx_size].iat[0].split()[0])
        self.limits = (mass_center - mass_window_size / 2,
                       mass_center + mass_window_size / 2)

    def get_spectrum(self, index: int, **kwargs) -> Spectrum:
        """Get spectrum in mcf file by index (R index, so 1-based)."""
        rspectrum = rtms.getSpectrum(self.reader, int(index))
        # convert to python
        spectrum = Spectrum(rspectrum, **kwargs)
        return spectrum

    def set_mzs(self, mzs: Iterable[float]):
        self.mzs = mzs

    @lru_cache
    def get_spectrum_resampled_intensities(self, index: int) -> np.ndarray[float]:
        spectrum: Spectrum = self.get_spectrum(index)
        spectrum.resample(self.mzs)
        return spectrum.intensities

    def get_spectrum_by_spot(self, spot: str):
        """Get spectrum by spot-name (e.g. R00X102Y80)."""
        assert hasattr(self, 'spots'), 'create spots with create_spots first'
        # find corresponding index
        # index in spots may be shifted or have missing values
        # index corresponding to name in the spots table
        matches = np.argwhere(self.spots.names == spot)[0]
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


if __name__ == '__main__':
    pass
