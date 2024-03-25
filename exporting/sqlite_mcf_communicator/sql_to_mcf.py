from exporting.from_sqlite.parser import parse_sqlite

import os
import numpy as np

names = ['xy', 'snr', 'mzs', 'intensities', 'fwhm']


def get_path_sql(path_d_folder: str) -> str:
    """For given d folder, return the path to the sql file."""
    return os.path.join(path_d_folder, 'peaks.sqlite')


def find_files(path_d_folder: str) -> dict[str, str]:
    """Return dict of files relevant for sqlite."""
    files = os.listdir(path_d_folder)

    targets = [t + '.npy' for t in names]
    out = {}
    for n, t in zip(names, targets):
        if t in files:
            out[n] = t

    return out


def load_files(path_d_folder, targets=None) -> dict[str, np.ndarray]:
    """Load targeted files."""
    if targets is None:
        targets = names.copy()

    files = find_files(path_d_folder)
    out = {}
    for t in targets:
        out[t] = np.load(os.path.join(path_d_folder, files[t]), allow_pickle=True)

    return out


def read_sql(path_d_folder: str) -> dict[str, np.ndarray]:
    file = get_path_sql(path_d_folder)
    xy, mzs, intensities, snrs, fwhms = parse_sqlite(file)
    # turn into dict
    out = {
        'xy': xy,
        'mzs': mzs,
        'intensities': intensities,
        'snrs': snrs,
        'fwhms': fwhms
    }
    return out


def get_sql_files(path_d_folder: str) -> dict[str, np.ndarray]:
    files = find_files(path_d_folder)
    if any([target not in files.keys() for target in names]):
        return read_sql(path_d_folder)
    return load_files(path_d_folder)


def reconstruct_from_sql(path_d_folder: str):
    pass
