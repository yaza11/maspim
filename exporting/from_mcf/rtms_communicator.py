"""This module allows the usage of functions from the R rtms package in python."""
from exporting.from_mcf.helper import get_r_home, ReaderBaseClass, Spots, Spectrum

from typing import Iterable
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

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


class ReadBrukerMCF(ReaderBaseClass):
    """
    Python intersection for rtms ReadBrukerMCF.

    This class allows to read MALDI data from disk files directly (mcf files with the corresponding
    index files.)

    Example usage
    -------------
    Import and initialize a reader
    >>> from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
    >>> reader = ReadBrukerMCF(path="/path/to/d_folder.d")
    >>> reader.create_reader()  # this can take a while
    Get information about the indices and meta data
    >>> reader.create_indices()
    >>> reader.set_meta_data()
    In case a QTOF was used, the masss window limits can be set from the meta data:
    >>> reader.set_meta_data()
    Now we can get spectra, either by providing an index
    >>> spec = reader.get_spectrum(1000)
    or a spot name
    >>> spec = reader.get_spectrum_by_spot('R00X100Y085')
    If you define an m/z vector, it is also possible to resample the spectra right away
    (this is not the recommended way, the Project class takes care of the details)
    >>> reader.set_mzs(np.arange(spec.limits[0], spec.limits[1], 1e-4))
    >>> intensities = reader.get_spectrum_resampled_intensities(1000)

    Notes
    -----
    The creation and reading of information from this class is fairly slow. Usually downstream
    analysis requires reading in information multiple times. It is therefore recommended to
    create an hdf5 file, if you can affort the disk space (See the documentation of hdf5Reader).
    """

    def __init__(self, path_d_folder: str):
        """
        Initializer

        Parameters
        ----------
        path_d_folder : str
            Path to the d-folder containing the mcf-files on disk.
        """
        self.path_d_folder: str = path_d_folder

    def create_reader(self):
        """Create a new BrukerMCFReader object."""
        logger.info('creating BrukerMCF reader, this may take a while ...')
        self.reader = rtms.newBrukerMCFReader(self.path_d_folder)
        logger.info('done creating reader')

    def create_indices(self):
        """Create indices of spectra in mcf file."""
        assert hasattr(
            self, 'reader'), 'create a reader with create_reader first'
        # get indices from reader
        self.indices: np.ndarray[int] = np.array(rtms.getBrukerMCFIndices(self.reader))

    def create_spots(self):
        """Create spots object with indices and names."""
        assert hasattr(
            self, 'reader'), 'create a reader with create_reader first'
        logger.info('creating spots table ...')
        # get spots from reader
        rspots = rtms.getBrukerMCFSpots(self.reader)
        self.spots: Spots = Spots(rspots)
        logger.info('done creating spots table')

    def set_meta_data(self):
        """Fetch metadata for measurement from mcf file and turn into df."""
        # arbitrary index, metaData should be the same for all spectra
        metaData = rtms.getBrukerMCFAllMetadata(self.reader, index=1)
        self.metaData: pd.DataFrame = pd.DataFrame({
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
        idx_center: pd.Series = self.metaData.PermanentName == 'Q1Mass'
        idx_size: pd.Series = self.metaData.PermanentName == 'Q1Res'
        assert ((n_entries_center := idx_center.sum()) == 1) and ((n_entries_size := idx_size.sum()) == 1), \
            (f'found {n_entries_center} entries for Q1Mass and {n_entries_size} for Q1Res in metadata, ' +
            'are you sure your data was obtained using a QTOF?')
        # only keep value (not 1320.0 m/z)
        mass_center: float = float(
            self.metaData.loc[idx_center, 'Value'].iat[0].split()[0])
        mass_window_size: float = float(
            self.metaData.Value[idx_size].iat[0].split()[0])
        self.limits: tuple[float, float] = (
            mass_center - mass_window_size / 2,
            mass_center + mass_window_size / 2
        )

    def get_spectrum(
            self,
            index: int,
            poly_coeffs: np.ndarray[float] | None = None,
            **kwargs
    ) -> Spectrum:
        """
        Get spectrum in mcf file by index (R index, so 1-based).

        Parameters
        ----------
        index: int
            The 1-based index of the spectrum.
        poly_coeffs: np.ndarray[float], optional
            Coefficients of the calibration polynomial. The default is None
            and will not apply a calibration.
        kwargs:
            Additional keywords used to initialize the spectrum instance (e.g. limits).

        Returns
        -------
        spectrum: Spectrum
            The resampled (and possibly calibrated) spectrum.
        """
        rspectrum = rtms.getSpectrum(self.reader, int(index))
        # convert to python
        spectrum: Spectrum = Spectrum(rspectrum, **kwargs)

        if poly_coeffs is not None:
            spectrum: Spectrum = self.calibrate_spectrum(spectrum, poly_coeffs)

        return spectrum

    def set_mzs(self, mzs: Iterable[float]):
        """Set the mz values for other classes to use to resample spectra"""
        self.mzs: np.ndarray[float] = np.array(mzs).astype(float)

    def get_spectrum_by_spot(self, spot: str) -> Spectrum:
        """
        Get spectrum by spot-name (e.g. R00X102Y80).

        Parameters
        ----------
        spot: str
            The name of the spot.

        Returns
        -------
        spectrum: Spectrum
            The spectrum corresponding to the spot.
        """
        assert hasattr(self, 'spots'), 'create spots with create_spots first'
        # find corresponding index
        # index in spots may be shifted or have missing values
        # index corresponding to name in the spots table
        matches: np.ndarray[int] = np.argwhere(self.spots.names == spot)[0]
        assert len(matches) == 1, f'{spot} not a valid spot, check spots.names'

        idx_spot: int = matches[0]
        # the corresponding value
        # int is necessary because rpy2 is picky about types and argwhere
        # returns np.intc, not int
        idx_spectrum: int = int(self.spots.idxs[idx_spot])
        spectrum: Spectrum = self.get_spectrum(idx_spectrum)
        return spectrum


if __name__ == '__main__':
    pass
