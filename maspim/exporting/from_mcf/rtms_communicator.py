"""This module allows the usage of functions from the R rtms package in python."""
import warnings

import pandas as pd
import numpy as np
import os
import logging
from functools import cached_property
from pyrtms.rtmsBrukerMCFReader import newBrukerMCFReader, getBrukerMCFIndices, getBrukerMCFAllMetadata, \
    getBrukerMCFSpots, getBrukerMCFSpectrum, RtmsBrukerMCFReader, BatchProcessor

from typing import Iterable

from maspim.exporting.from_mcf.helper import ReaderBaseClass, Spots, Spectrum, \
    apply_calibration, split_spot
from maspim.util.convenience import check_attr

logger = logging.getLogger(__name__)


class ReadBrukerMCF(ReaderBaseClass):
    """
    Python intersection for rtms ReadBrukerMCF.

    This class allows to read MALDI data from disk files directly (mcf files with the corresponding
    index files.)

    Example usage
    -------------
    Import and initialize a reader
    >>> from maspim import ReadBrukerMCF
    >>> reader = ReadBrukerMCF(path_d_folder="/path/to/d_folder.d")
    Now we can get spectra, either by providing an index
    >>> spec = reader.get_spectrum(1000)

    Notes
    -----
    The creation and reading of information from this class is fairly slow. Usually downstream
    analysis requires reading in information multiple times. It is therefore recommended to
    create a hdf5 file, if you can affort the disk space (See the documentation of hdf5Reader).
    """
    limits: tuple[float, float] | None = None
    path_d_folder: str | None = None
    reader: RtmsBrukerMCFReader | None = None
    mzs: np.ndarray[float] | None = None

    def __init__(
            self,
            path_d_folder: str,
            limits: tuple[float, float] | None = None,
            check_mzs_are_equal: bool = True
    ) -> None:
        """
        Initializer

        Parameters
        ----------
        path_d_folder : str
            Path to the d-folder containing the mcf-files on disk.
        """
        self.path_d_folder: str = path_d_folder
        self.limits: tuple[float, float] | None = limits

        self._create_reader()
        self.set_mzs(check_mzs_are_equal)

    def _create_reader(self):
        """Create a new BrukerMCFReader object."""
        logger.warning('creating BrukerMCF reader, this may take a while ...')
        self.reader: RtmsBrukerMCFReader = newBrukerMCFReader(self.path_d_folder)
        logger.info('done creating reader')

    @property
    def indices(self):
        """Create indices of spectra in mcf file."""
        assert check_attr(self, 'reader'), \
            'create a reader with _create_reader first'
        # get indices from reader
        return self.reader.spotTable.index

    @cached_property
    def spots(self):
        return self.reader.get_spots()

    @property
    def rs(self):
        return self.spots.SpotNumber.apply(lambda spotname: split_spot(spotname)[0])

    @property
    def xs(self):
        return self.spots.SpotNumber.apply(lambda spotname: split_spot(spotname)[1])

    @property
    def ys(self):
        return self.spots.SpotNumber.apply(lambda spotname: split_spot(spotname)[2])

    @property
    def metaData(self):
        """Fetch metadata for measurement from mcf file and turn into df."""
        # metaData should be the same for all spectra, cached by pyrtms
        return self.reader.metadataDF

    def set_casi_window(self):
        """Set mass window limits from CASI values in metadata.

        Continuous Accumulation of Selected Ions (CASI)
        """
        assert check_attr(self, 'metaData'), 'call set_meta_data first'

        if check_attr(self, 'limits'):
            logger.warning(f'overwriting previous mass window: {self.limits}')

        # find entries
        # not sure what DC and CID do, but for measurements without valid
        # CASI window Q1DC and Q1CID are both 'off', whereas Q1DC is 'on' for
        # measurements where
        idx_dc: pd.Series = self.metaData.PermanentName == 'Q1DC'
        idx_cid: pd.Series = self.metaData.PermanentName == 'Q1CID'
        idx_center: pd.Series = self.metaData.PermanentName == 'Q1Mass'
        idx_size: pd.Series = self.metaData.PermanentName == 'Q1Res'

        # check if the number of entries is 1
        n_entries_size = idx_size.sum()
        assert (
                ((n_entries_center := idx_center.sum()) == 1)
                and (n_entries_size == 1)
        ), (
                f'found {n_entries_center} entries for Q1Mass and {n_entries_size} '
                f'for Q1Res in metadata, ' +
                'are you sure your data was obtained using a QTOF?'
        )
        # check if either dc or cid value is 'on'
        if (
                (self.metaData.loc[idx_dc, 'Value'].iat[0] == 'off')
                and (self.metaData.loc[idx_cid, 'Value'].iat[0] == 'off')
        ):
            logger.error(
                'CID and DC value are set to "off" which probably means that '
                'no QTOF was used. Consider setting the limits manually.'
            )

        # only keep value (not 1320.0 m/z)
        mass_center: float = float(
            self.metaData.loc[idx_center, 'Value'].iat[0].split()[0])
        mass_window_size: float = float(
            self.metaData.Value[idx_size].iat[0].split()[0])
        if mass_window_size < 1e-9:
            raise ValueError(
                f'found unrealistic mass window parameters, consider specifying'
                f' the mass window manually'
            )
        self.limits: tuple[float, float] = (
            mass_center - mass_window_size / 2,
            mass_center + mass_window_size / 2
        )

    def get_spectrum(
            self,
            index: int = None,
            poly_coeffs: np.ndarray[float] | None = None,
            limits: tuple[float, float] | None = None
    ) -> Spectrum:
        """
        Get spectrum in mcf file by index (R index, so 1-based).

        Parameters
        ----------
        index: int
            The index of the spectrum.
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
        rspectrum = self.reader.get_spectrum(index)
        # convert to python
        if limits is None:
            limits = self.limits
        spectrum: Spectrum = Spectrum(rspectrum, limits=limits)

        if poly_coeffs is not None:
            spectrum: Spectrum = apply_calibration(spectrum, poly_coeffs)

        return spectrum

    def set_mzs(self, check_mzs=True):
        """Set the mz values for other classes to use to resample spectra.

        The mz array is shared with the pyrtms reader and read-only, copy it
        before modifying it.
        """
        self.mzs: np.ndarray[float] = self.reader.get_spectrum(0)[0]
        if not check_mzs:
            return
        # all spectra share the same m/z axis if their calibration parameters are identical
        cols = ["frequencyHigh", "frequencyWidth", "size", "alpha", "beta"]
        assert len(self.reader.spotTable[cols].drop_duplicates()) == 1, \
            ("Encountered spectra with non-equally sampled mz values. "
             "This is unsupported behavior.")

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
        raise NotImplementedError("this feature is deprecated")

        assert check_attr(self, 'spots'), 'create spots with create_spots first'
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

    def get_intensities_for_array_indices(self, expr) -> np.ndarray[np.float64]:
        """
        Obtain intensities for the specified indices as a 2D array
        (n_spectra, n_mzs).

        Supports every indexing method from numpy arrays.
        """
        indices = self.indices[expr]
        res = BatchProcessor(reader=self.reader, show_progress=False).get_mul_spectra(
            indices=indices, intensities_only=True)
        return np.stack([intensities for _, intensities in res]).astype(np.float64)

    def close(self) -> None:
        """Close the reader and release the file handle on the mcf file."""
        self.reader.close()


if __name__ == '__main__':
    pass
