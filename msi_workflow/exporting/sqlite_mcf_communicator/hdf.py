"""This module implements an HDF5 handler."""
import h5py
import numpy as np
import os
import logging

from typing import Iterable
from tqdm import tqdm

from msi_workflow.exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from msi_workflow.exporting.from_mcf.helper import get_mzs_for_limits, ReaderBaseClass, Spectrum

logger = logging.getLogger(__name__)


class hdf5Handler(ReaderBaseClass):
    """
    This class allows the interaction and creation of hdf5 files.

    Notice that the saved hdf5 file will reach a considerable size (assuming a sampling
    rate of 0.1 mDa a measurement with 20_000 spectra and a mass window of 100 Da would
    occupy 150 GB, so use the write function with caution).

    Example Usage
    -------------
    >>> from msi_workflow.exporting.sqlite_mcf_communicator.hdf import hdf5Handler
    >>> from msi_workflow.exporting.from_mcf.rtms_communicator import ReadBrukerMCF
    >>> brukerReader = ReadBrukerMCF('some/d/folder.d')
    >>> brukerReader.create_reader()
    >>> brukerReader.create_indices()
    >>> brukerReader.set_meta_data()
    >>> brukerReader.set_casi_window()  # very important, otherwise the saved file will be huge
    >>> hdf = hdf5Handler('some/d/folder.d')
    This creates the disk file
    >>> hdf.write(brukerReader)
    After that, for the initicailzation it suffices to call
    >>> hdf = hdf5Handler('some/d/folder.d')
    and you have the same functionality as for the ReadBrukerMCF reader, e.g.
    >>> hdf.get_spectrum(1000)
    """
    def __init__(self, path_file: str) -> None:
        """
        Initializer.

        Parameters
        ----------
        path_file : str
            Path and file name of the hdf5 file (e.g. 'path/to/file.hdf5') or d-folder
            (e.g. 'path/to/d_folder.d')
        """
        self._set_files(path_file)
        self._check_modify_date()
        self._post_init()

    def _set_files(self, path_file: str) -> None:
        """Infere the file name, d-folder and folder from input."""
        if path_file.split('.')[-1] == 'hdf5':  # hdf 5 file provided
            path_d_folder: str = os.path.dirname(path_file)
        elif os.path.isdir(path_file) and (path_file.split('.')[-1] == 'd'):  # d folder provided
            path_d_folder: str = path_file
            path_file: str = os.path.join(path_d_folder, 'Spectra.hdf5')
        else:
            raise FileNotFoundError(
                'provided path must either be the hdf file ending in .hdf5 or the d-folder containing the hdf file'
            )
        self.file: str = os.path.basename(path_file)
        self.d_folder: str = os.path.basename(path_d_folder)
        self.path_folder: str = os.path.dirname(path_d_folder)

    def _check_modify_date(self) -> None:
        """Throw warning if mcf file was modified after hdf file."""
        if not os.path.exists(self.path_file):
            return
        time_hdf: float = os.path.getmtime(self.path_file)
        time_mcf: float = max([
            os.path.getmtime(os.path.join(self.path_d_folder, file))
            for file in os.listdir(self.path_d_folder) 
            if ('mcf' in file.split('.')[-1]) and (file != 'Storage.mcf_idx')
        ])
        if time_mcf > time_hdf:
            logger.warning(
                'mcf files were modified after the creation of the hdf5 file, you may want to '
                'create a new hdf5 file.'
            )

    def _post_init(self) -> None:
        """Set metadata and limit from hdf5 file."""
        if not os.path.exists(self.path_file):
            return

        keys: list[str] = ['indices', 'mzs']
        with h5py.File(self.path_file, 'r') as f:
            for key in keys:
                if key in f:
                    self.__dict__[key] = f[key][:]
            if (key := 'limits') in f.attrs:
                self.limits = f.attrs[key]

    @property
    def path_d_folder(self) -> str:
        """Path to d-folder."""
        return os.path.join(self.path_folder, self.d_folder)

    @property
    def path_file(self) -> str:
        """Path to disk-file."""
        return os.path.join(self.path_d_folder, self.file)

    def create_indices(self):
        """
        Create indices.

        Not necessary to call, but for compatability with ReadBrukerMCF class.
        """
        with h5py.File(self.path_file, 'r') as f:
            self.indices = f['indices'][:]

    def create_reader(self):
        """For compatibility with ReadBrukerMCF class."""
        pass

    def write(
            self,
            reader: ReadBrukerMCF,
            mzs: np.ndarray[float] | None = None,
            delta_mz=1e-4
    ) -> None:
        """
        Using the ReadBrukerMCF reader, write an hdf5 file.

        This is recommended, if you can affort the extra disk space (e.g. ~150 GB for
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
        limits: tuple[float, float] = reader.limits
        N: int = len(indices)

        if mzs is None:
            mzs = get_mzs_for_limits(limits, delta_mz)

        # stored as float8 apparently
        size_GB: float = N * len(mzs) * 8 / 1024 ** 3
        logger.info(f'Creating hdf5 file on disk with {size_GB:.1f} GB')

        with h5py.File(self.path_file, 'w') as f:
            # use file name as group name
            group_name: str = os.path.basename(reader.path_d_folder)
            f.create_group(group_name)
            data_shape: tuple[int, int] = (len(indices), len(mzs))
            dset = f.create_dataset(
                'intensities',
                shape=data_shape,
                dtype='float',
                chunks=(1, len(mzs))
            )
            # read, resample and write all spectra
            for it, idx in tqdm(
                    enumerate(indices),
                    total=N,
                    desc='Writing to hdf5',
                    smoothing=50/N
            ):
                spec: Spectrum = reader.get_spectrum(idx)
                spec.resample(mzs)
                dset[it, :] = spec.intensities

            f.create_dataset('indices', data=indices, dtype='int')
            f.create_dataset('mzs', data=mzs, dtype='float')

            f.attrs['limits'] = limits

        self.indices: np.ndarray[int] = indices
        self.limits: tuple[float, float] = limits
        self.mzs: np.ndarray[float] = mzs

    def read(
            self,
            indices: Iterable[int] | None = None,
    ) -> dict[str, np.ndarray]:
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
        with h5py.File(self.path_file, 'r') as f:
            indices_hpf5 = f['indices'][:]

            if indices is None:
                mask = np.ones_like(self.indices, dtype=bool)
            elif len(indices) == 1:
                mask = indices[0]
            else:
                mask = [idx in indices for idx in indices_hpf5]

            dset = f['intensities']

            intensities = dset[mask, :]
            masses = self.mzs

            return {
                'mzs': masses,
                'intensities': intensities
            }

    def get_spectrum(
            self,
            index: int,
            poly_coeffs: np.ndarray[float] | None = None,
            **kwargs
    ) -> Spectrum:
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
        index -= 1  # convert 1 based to 0 based index

        with h5py.File(self.path_file, 'r') as f:
            intensities = f['intensities'][index, :]

        spectrum = Spectrum((self.mzs, intensities), **kwargs)

        if poly_coeffs is not None:
            spectrum = self.calibrate_spectrum(spectrum, poly_coeffs)

        return spectrum
