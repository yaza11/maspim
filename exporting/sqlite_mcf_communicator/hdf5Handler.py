from exporting.from_mcf.rtms_communicator import ReadBrukerMCF, Spectrum
from exporting.from_mcf.helper import get_mzs_for_limits

import h5py
import numpy as np
import os
import time

from typing import Iterable


class hdf5Handler:
    def __init__(self, path_file: str):
        if path_file.split('.')[-1] == 'hdf5':
            self.path_file: str = path_file
        elif os.path.isdir(path_file) and (path_file.split('.')[-1] == 'd'):
            self.path_d_folder = path_file
            path_file = os.path.join(path_file, 'Spectra.hdf5')
            self.path_file = path_file
        else:
            raise FileNotFoundError(
                'provided path must either be the hdf file ending in .hdf5 or the d-folder containing the hdf file'
            )

        self._post_init()

    def _post_init(self):
        if not os.path.exists(self.path_file):
            return

        keys = ['indices', 'mzs']
        with h5py.File(self.path_file, 'r') as f:
            for key in keys:
                if key in f:
                    self.__dict__[key] = f[key][:]
            if (key := 'limits') in f.attrs:
                self.limits = f.attrs[key]

    def create_indices(self):
        with h5py.File(self.path_file, 'r') as f:
            self.indices = f['indices'][:]

    def write(
            self,
            reader: ReadBrukerMCF,
            mzs: np.ndarray[float] | None = None,
            delta_mz=1e-4
    ):

        assert hasattr(reader, 'indices'), 'call create_indices first'
        assert hasattr(reader, 'limits') or (mzs is not None), \
            'if mzs is not provided, reader must have limits'

        indices = reader.indices
        N: int = len(indices)

        if mzs is None:
            mzs = get_mzs_for_limits(reader.limits, delta_mz)

        with h5py.File(self.path_file, 'w') as f:
            # use file name as group name
            group_name = os.path.basename(reader.path_d_folder)
            f.create_group(group_name)
            data = np.zeros((len(indices), len(mzs)))
            dset = f.create_dataset(
                'intensities',
                data=data,
                dtype='float',
                chunks=(1, len(mzs))
            )

            time0 = time.time()
            print_interval = 10 ** (np.around(np.log10(N), 0) - 2)
            print('writing to hdf5')
            for it, idx in enumerate(indices):
                spec: Spectrum = reader.get_spectrum(idx)
                spec.resample(mzs)
                dset[it, :] = spec.intensities

                if it % print_interval == 0:
                    time_now = time.time()
                    time_elapsed = time_now - time0
                    predict = time_elapsed * N / (it + 1)  # s
                    left = predict - time_elapsed
                    left_min, left_sec = divmod(left, 60)
                    print(end='\x1b[2K')
                    print(
                        f'estimated time left: {str(int(left_min)) + " min" if left_min != 0 else ""} {left_sec:.1f} sec',
                        end='\r'
                    )
            print()
            print('\n')

            f.create_dataset('indices', data=indices, dtype='int')
            f.create_dataset('mzs', data=mzs, dtype='float')

            f.attrs['limits'] = reader.limits

        self.indices = reader.indices
        self.limits = reader.limits
        self.mzs = mzs

    def read(
            self,
            indices: Iterable[int] | None = None,
    ) -> dict[str, np.ndarray]:

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

    def get_spectrum(self, index: int, **kwargs) -> Spectrum:
        """Return the spectrum to the Spots index (starting at 1)"""
        index -= 1  # convert 1 based to 0 based index

        res = self.read([index])

        spectrum = Spectrum((res['mzs'], res['intensities']), **kwargs)
        return spectrum

    def get_spectrum_resampled_intensities(self, index: int) -> np.ndarray[float]:
        spectrum = self.get_spectrum(index)
        spectrum.resample(self.mzs)
        return spectrum.intensities

    def get_mass(self, mz: float, sigma: float | None, FWHM: float | None = None):
        pass
