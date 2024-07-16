"""Obtain snr, intensities and fwhm from the SQL file. Complementary to the from_sqlite module."""
import os
import numpy as np

from src.exporting.from_sqlite.parser import parse_sqlite

names: list[str] = ['xy', 'snr', 'mzs', 'intensities', 'fwhm']


def get_path_sql(path_d_folder: str) -> str:
    """For given d folder, return the path to the sql file."""
    return os.path.join(path_d_folder, 'peaks.sqlite')


def find_files(path_d_folder: str) -> dict[str, str]:
    """Return dict of files relevant for sqlite."""
    files: str = os.listdir(path_d_folder)

    targets: list[str] = [t + '.npy' for t in names]
    out: dict[str, str] = {}
    for n, t in zip(names, targets):
        if t in files:
            out[n] = t

    return out


def load_files(path_d_folder, targets: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load targeted files."""
    if targets is None:
        targets = names.copy()

    files: dict[str, str] = find_files(path_d_folder)
    out: dict[str, np.ndarray] = {}
    for t in targets:
        out[t] = np.load(os.path.join(path_d_folder, files[t]), allow_pickle=True)

    return out


def read_sql(path_d_folder: str) -> dict[str, np.ndarray]:
    """Read data from an sql file."""
    file: str = get_path_sql(path_d_folder)
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
    """Create new or load existing sql files."""
    files: dict[str, str] = find_files(path_d_folder)
    if any([target not in files.keys() for target in names]):
        return read_sql(path_d_folder)
    return load_files(path_d_folder)
