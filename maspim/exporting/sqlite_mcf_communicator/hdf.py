"""This module implements an HDF5 handler."""
import h5py
import numpy as np
import os
import logging

from typing import Iterable

import pandas as pd
from pyrtms.rtmsBrukerMCFReader import RtmsBrukerMCFReader
from tqdm import tqdm

from maspim.exporting.from_mcf.helper import get_mzs_for_limits, Spectrum
from maspim.util.convenience import get_disk_file

logger = logging.getLogger(__name__)


class Hdf5Handler:
    """
    This class allows the interaction and creation of hdf5 files.

    Notice that the saved hdf5 file will reach a considerable size (assuming a sampling
    rate of 0.1 mDa a measurement with 20_000 spectra and a mass window of 100 Da would
    occupy 150 GB, so use the write function with caution).

    Example Usage
    -------------
    >>> from maspim.exporting.sqlite_mcf_communicator.hdf import Hdf5Handler
    >>> brukerReader = RtmsBrukerMCFReader.from_dir('some/d/folder.d')
    >>> brukerReader._create_reader()
    >>> brukerReader.create_indices()
    >>> brukerReader.set_meta_data()
    >>> brukerReader.set_casi_window()  # very important, otherwise the saved file will be huge
    >>> hdf = Hdf5Handler('some/d/folder.d')
    This creates the disk file
    >>> hdf.write(brukerReader)
    After that, for the initicailzation it suffices to call
    >>> hdf = Hdf5Handler('some/d/folder.d')
    and you have the same functionality as for the ReadBrukerMCF reader, e.g.
    >>> hdf.get_spectrum(1000)
    """
    _save_in_d_folder: bool = True

    indices: np.ndarray[int] = None
    mzs: np.ndarray[float] = None
    all_mz_values_same: bool = True  # not necessarily True for RtmsBrukerMCFReader
    limits: tuple[float | int, float | int] = None
    casi_window: list[float, float] = None
    # get_intensities_for_array_indices only reads the requested mz window
    reads_windows: bool = True

    path_folder: str = None
    path_file: str = None

    def __init__(self, *, path_file: str = None, path_folder: str = None, tag: str = None) -> None:
        """
        Initializer.

        Parameters
        ----------
        path_file : str
            Path and file name of the hdf5 file (e.g. 'path/to/file.hdf5') or d-folder
            (e.g. 'path/to/d_folder.d')
        """
        self._set_files(path_file, path_folder, tag)
        self._check_modify_date()
        self._post_init()

    def _set_files(self, path_file: str | None, path_folder: str | None, tag: str | None) -> None:
        """Infer the file name, d-folder and folder from input."""
        if (path_folder is None) and (path_file is None):
            path_folder = ''
        if (path_file is not None) and (path_folder is not None):
            logger.warning('If folder and file are provided, only the file will be used, ignoring the folder')
        if (path_file is not None) and (tag is not None):
            logger.warning('If file is provided, tag will be ignored.')

        if path_file is not None:  # hdf 5 file provided
            assert path_file.split('.')[-1] == 'hdf5'
            path_folder: str = os.path.dirname(path_file)
        elif path_folder is not None:
            file: str = get_disk_file(self, path_folder, tag)
            path_file: str = os.path.join(path_folder, file)

        self.path_folder: str = path_folder
        self.path_file: str = path_file

    def _check_modify_date(self) -> None:
        """Throw warning if mcf file was modified after hdf file."""
        if not os.path.exists(self.path_file):
            return
        time_hdf: float = os.path.getmtime(self.path_file)
        modify_times = [
            os.path.getmtime(os.path.join(self.path_folder, file))
            for file in os.listdir(self.path_folder)
            if ('mcf' in file.split('.')[-1]) and (file != 'Storage.mcf_idx')
        ]
        if len(modify_times) == 0:
            logger.info(
                'No mcf files found, so not checking modify times'
            )
            return

        time_mcf: float = max(modify_times)
        if time_mcf > time_hdf:
            logger.warning(
                'mcf files were modified after the creation of the hdf5 file, you may want to '
                'create a new hdf5 file.'
            )

    def _post_init(self) -> None:
        """Set metadata and limit from hdf5 file."""
        if not os.path.exists(self.path_file):
            return

        # keys: list[str] = ['indices', 'mzs']
        with h5py.File(self.path_file, 'r') as f:
            # for key in keys:
            #     if key in f:
            #         self.__dict__[key] = f[key][:]
            self.mzs: np.ndarray = np.asarray(f['mzs'])
            self.indices = self._read_indices(f)
            if (key := 'limits') in f.attrs:
                self.limits = f.attrs[key]
            if (key := 'casi_window') in f.attrs:
                self.casi_window = f.attrs[key]

    def _read_indices(self, f: h5py.File) -> np.ndarray[int]:
        """
        Read the spectrum indices from the file.

        Files written with the R rtms reader store 1-based indices (1, ..., N),
        while spectra are 0-based for the pyrtms reader and the spot table.
        Those indices are converted to 0-based.
        """
        indices: np.ndarray[int] = np.asarray(f['indices'])
        n_rows: int = f['intensities'].shape[0]
        if (len(indices) == n_rows) and np.array_equal(indices, np.arange(1, n_rows + 1)):
            logger.warning(
                f'{self.path_file} stores 1-based indices (written with the R rtms '
                f'reader), converting them to 0-based. Consider rewriting the hdf5 file.'
            )
            indices = np.arange(n_rows)
        return indices

    def _index_to_row(self, index: int) -> int:
        """Return the row in the intensities dataset of the spectrum with the given index."""
        if (getattr(self, '_row_lookup', None) is None) or (self._row_lookup[0] is not self.indices):
            self._row_lookup = (self.indices, {int(idx): row for row, idx in enumerate(self.indices)})
        return self._row_lookup[1][int(index)]

    def create_indices(self):
        """
        Create indices.

        Not necessary to call, but for compatability with ReadBrukerMCF class.
        """
        with h5py.File(self.path_file, 'r') as f:
            self.indices = self._read_indices(f)

    def add_metadata(self, reader: RtmsBrukerMCFReader) -> None:
        """Add index, measurement parameters and coordinates to hdf5 metadata"""

        spots = reader.spots
        met = reader.metadataDF
        indices = spots.index.to_numpy()

        with h5py.File(self.path_file, 'a') as f:
            if 'indices' not in f.keys():
                f.create_dataset('indices', data=indices)
        if 'instrument_settings' not in f.keys():
            met.to_hdf(self.path_file, key='instrument_settings', mode='a')

    @property
    def metadataDF(self) -> pd.DataFrame:
        """Return the metadata as a pandas DataFrame"""
        return pd.read_hdf(self.path_file, key='instrument_settings')

    def set_casi_window(self, casi_window: list[float, float] = None) -> None:
        self.casi_window = casi_window

    def write(
            self,
            reader: RtmsBrukerMCFReader,
            mzs: np.ndarray[float] = None,
            limits: tuple[float, float] = None,
            delta_mz=1e-4,
            show_progress: bool = False,
            **kwargs,
    ) -> None:
        """
        Using the ReadBrukerMCF reader, write an hdf5 file.

        This is recommended, if you can afford the extra disk space (e.g. ~150 GB for
        a measurement with 20_000 spectra and a mass range of 100 Da at a coverage of
        0.1 mDa)

        Parameters
        ----------
        reader: ReadBrukerMCF
            The Bruker reader used to open and read data from the mcf files.
        mzs: np.ndarray[float], optional
            The mz values used to resample the spectra. Default is a value every delta_mz.
        delta_mz: float, optional
            The sampling rate for intensities in the spectra. The default is 1e-4 Da (0.1 mDa).
        """
        assert hasattr(reader, 'indices'), 'call create_indices first'
        assert hasattr(reader, 'limits') or (mzs is not None), \
            'if mzs is not provided, reader must have limits'

        indices: np.ndarray[int] = reader.indices
        N: int = len(indices)

        if limits is None:
            limits = reader.casi_window

        if mzs is None:
            mzs = get_mzs_for_limits(limits, delta_mz)

        # stored as float64 (64 bit, so 8 byte per element)
        size_GB: float = N * len(mzs) * 8 / 1024 ** 3
        # warn above 100 GB
        if size_GB > 1024:
            logger.warning(f'Creating hdf5 file on disk with {size_GB / 1024:.1f} TB')
        elif size_GB > 100:
            logger.warning(f'Creating hdf5 file on disk with {size_GB:.1f} GB')
        elif size_GB > 1:
            logger.info(f'Creating hdf5 file on disk with {size_GB:.1f} GB')
        else:
            logger.info(f'Creating hdf5 file on disk with {size_GB * 1024:.1f} MB')

        with h5py.File(self.path_file, 'w') as f:
            # use file name as group name
            # group_name: str = os.path.basename(reader.path_d_folder)
            # f.create_group(group_name)
            data_shape: tuple[int, int] = (len(indices), len(mzs))
            dset = f.create_dataset(
                'intensities',
                shape=data_shape,
                dtype='float64',
                chunks=(1, len(mzs)),
                # according to test, lzf is faster to read, about 10 % smaller on disk and has similar write times to no compression
                compression=kwargs.pop('compression', 'lzf'),
                **kwargs,
            )
            # read, resample and write all spectra
            idx: int
            for it, idx in tqdm(
                    enumerate(indices),
                    total=N,
                    desc='Writing to hdf5',
                    smoothing=50/N,
                    disable=not show_progress
            ):
                _mzs, _ints = reader.get_spectrum(idx, return_mzs=True, limits=limits)
                dset[it, :] = np.interp(mzs, _mzs, _ints)

            f.create_dataset('indices', data=indices, dtype='int')
            f.create_dataset('mzs', data=mzs, dtype='float')

            f.attrs['limits'] = limits
            f.attrs['casi_window'] = reader.casi_window

        self.indices: np.ndarray[int] = indices
        self.limits: tuple[float, float] = limits
        self.mzs: np.ndarray[float] = mzs

        self.add_metadata(reader)

    def read(self, indices: Iterable[int] | None = None) -> dict[str, np.ndarray]:
        """
        Read spectra with given indices from hdf5 file.

        Parameters
        ----------
        indices: Iterable[int], optional
            The indices of the spectra to read. If not provided, all spectra will be read.

        Returns
        -------
        out: dict[str, np.ndarray]
            A dictionary containing
            - mzs: the mz values at which spectra were sampled
            - intensities: a 2D array with shape (N, M) where N is the number of spectra
              and M is the number of mz values.
        """
        with (h5py.File(self.path_file, 'r') as f):
            # (converted) indices, rows of the dataset follow their order
            indices_hpf5 = self.indices

            if indices is None:
                mask = np.ones_like(self.indices, dtype=bool)
            elif len(indices) == 1:
                assert indices[0] in indices_hpf5, \
                    f'{indices[0]} is not a valid index in {indices_hpf5}'
                mask = self._index_to_row(indices[0])
            else:
                mask = [idx in indices for idx in indices_hpf5]
                missing = [idx for idx in indices if idx not in indices_hpf5]
                assert len(missing) == 0, \
                    f'some indices were not found: {missing}'

            dset = f['intensities']

            intensities = dset[mask, :]
            masses = self.mzs

            return {
                'mzs': masses,
                'intensities': intensities
            }

    @property
    def rs(self):
        with (h5py.File(self.path_file, 'r') as f):
            return np.asarray(f['R'])

    @property
    def xs(self):
        with (h5py.File(self.path_file, 'r') as f):
            return np.asarray(f['X'])
    @property
    def ys(self):
        with (h5py.File(self.path_file, 'r') as f):
            return np.asarray(f['Y'])

    @property
    def instrument_settings(self):
        return pd.read_hdf(self.path_file, key='instrument_settings')

    # def get_intensities_for_array_indices(
    #         self,
    #         expr,
    #         mz_slice: slice = slice(None),
    #         **_
    # ) -> np.ndarray[np.float64]:
    #     """
    #     Obtain intensities for the specified indices as a 2D array
    #     (n_spectra, n_mzs).
    #
    #     Supports every indexing method from numpy arrays.
    #
    #     Parameters
    #     ----------
    #     expr:
    #         Positions in self.indices (rows of the dataset) of the spectra to read.
    #     mz_slice: slice, optional
    #         Only read these intensities (refers to self.mzs). The default reads all.
    #     """
    #     rows: np.ndarray[int] = np.atleast_1d(np.arange(len(self.indices))[expr])
    #     with (h5py.File(self.path_file, 'r') as f):
    #         dset = f['intensities']
    #         if (len(rows) > 0) and np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
    #             # consecutive rows can be read as one block
    #             return dset[rows[0]:rows[0] + len(rows), mz_slice]
    #         # h5py requires increasing indices
    #         order: np.ndarray[int] = np.argsort(rows)
    #         out: np.ndarray[float] = np.empty((len(rows), len(self.mzs[mz_slice])), dtype=dset.dtype)
    #         out[order] = dset[rows[order], mz_slice]
    #         return out

    def get_spectrum(
            self,
            index: int,
            return_mzs: bool,
            intensities_only: bool = None,
            limits: tuple[float, float] = None
    ) -> tuple[np.ndarray[float] | None, np.ndarray[float]]:
        """
        Return the spectrum to the Spots index (starting at 1)

        Parameters
        ----------
        index: int
            The index of the spectrum to retrieve.
        poly_coeffs: np.ndarray[float], optional
            The calibration function to apply before returning the spectrum.
        kwargs: dict, optional
            Additional keywords provided to the initialization of the Spectrum (e.g. limits).

        Returns
        -------
        spectrum: Spectrum
            The (calibrated) spectrum.
        """
        if return_mzs is not None and intensities_only is not None:
            raise ValueError('Provide only one of return_mzs and intensity_only, not both.')
        if intensities_only is not None:
            return_mzs = not intensities_only

        if limits is not None:
            mz_idx_lower: int = np.searchsorted(self.mzs, limits[0])
            mz_idx_upper: int = np.searchsorted(self.mzs, limits[1]) + 1
            mz_idcs = slice(mz_idx_lower, mz_idx_upper)
        else:
            mz_idcs = slice(None)

        with h5py.File(self.path_file, 'r') as f:
            intensities = f['intensities'][self._index_to_row(index), mz_idcs]

        mzs = self.mzs[mz_idcs] if return_mzs else None
        return mzs, intensities


if __name__ == '__main__':
    hdf5_reader = Hdf5Handler(
        r"C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i\2018_08_27 Cariaco 490-495 alkenones.d\Spectra.hdf5"
    )

    reader = RtmsBrukerMCFReader.from_dir(
        r'C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i\2018_08_27 Cariaco 490-495 alkenones.d'
    )

    hdf5_reader.write(reader=reader)