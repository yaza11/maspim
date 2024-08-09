import os

import scipy
import numpy as np
import pandas as pd

from tqdm import tqdm

from mfe.from_txt import (
    get_ref_peaks, parse_da_export,
    create_bin, combine_spectrum,
    Spectrum as mfe_spectrum
)

from msi_workflow.util import Convinience
from msi_workflow.util.convinience import check_attr


def msi_from_txt(raw_txt_path: str, **kwargs) -> dict:
    with open(raw_txt_path, encoding="utf8") as raw_txt:
        lines = raw_txt.readlines()

    # only keep the lines that contain the spectrum, i.e., starts with 'R'
    lines = [line for line in lines if line.startswith('R')]

    out = {}
    for line in tqdm(lines, total=len(lines)):
        coord, spec = parse_da_export(line, **kwargs)
        out[coord] = spec
    return out


def create_feature_table(
        spectrum_dict: dict[tuple, mfe_spectrum],
        ref_peaks: np.ndarray,
        tol: int | float = 10,
        normalization: str = 'None'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create binned feature table with designated bin size.

    Parameters:
    --------
        ref_peaks: a list of reference peaks to which the samples are aligned.
            spectrum_dict: a dictionary object with key as spot coordinates and
            spectrum as value.
        tol: tolerance for peak alignment.
        normalization: method for normalization ('None', 'MinMax', etc.).

    Returns:
    --------
        feature_table: a DataFrame object containing the feature table.
        err_table: a DataFrame object containing the mz error of each aligned
            peak in each spot for accuracy evaluation.
    """
    binned_spectrum_dict: dict[tuple[int, ...], mfe_spectrum] = {}
    for spot, spectrum in tqdm(spectrum_dict.items(), desc="Binning the spectra"):
        _, spec = create_bin(spot, spectrum, ref_peaks, tol)
        binned_spectrum_dict[spot] = spec

    primer_df = pd.DataFrame(np.full(ref_peaks.shape, np.nan), index=list(ref_peaks))

    combined_spectrum_dict: dict = {}
    err_dict = {}
    for spot, binned_spectrum in tqdm(
            binned_spectrum_dict.items(),
            desc="Combining the binned spectra"
    ):
        _, ft, err = combine_spectrum(
            [],  # isn't actually used
            binned_spectrum,
            primer_df,
            normalization=normalization
        )
        combined_spectrum_dict[spot] = ft
        err_dict[spot] = err

    spots = np.array(list(combined_spectrum_dict.keys()))

    result_arr = scipy.sparse.vstack(list(combined_spectrum_dict.values())).toarray()
    err_arr = scipy.sparse.vstack(list(err_dict.values())).toarray()

    feature_table = pd.DataFrame(result_arr, columns=list(ref_peaks))
    err_table = pd.DataFrame(err_arr, columns=list(ref_peaks))

    feature_table['R'], feature_table['x'], feature_table['y'] = (
        spots[:, 0], spots[:, 1], spots[:, 2])
    err_table['R'], err_table['x'], err_table['y'] = (
        spots[:, 0], spots[:, 1], spots[:, 2])

    return feature_table, err_table


class DataAnalysisExport(Convinience):
    path_file: str | None = None
    _peak_thr: float | None = None
    _normalization: str | None = None

    _spectrum_dict: dict[tuple, mfe_spectrum] | None = None
    _ref_peaks: dict[float, np.ndarray] | None = None
    _feature_table: pd.DataFrame | None = None
    _error_table: pd.DataFrame | None = None

    def __init__(self, path_file: str, peak_th=.1, normalization='None'):
        assert normalization in ['None', 'median']
        assert (peak_th >= 0) and (peak_th <= 1)

        self.path_file = path_file
        self.path_folder = os.path.dirname(path_file)
        self._peak_th = float(peak_th)
        self._normalization = normalization

    def set_feature_table(self):
        params = dict(peak_th=self._peak_th, normalization=self._normalization)
        self._spectrum_dict = msi_from_txt(self.path_file)
        self._ref_peaks = get_ref_peaks(self._spectrum_dict, peak_th=params['peak_th'])

        # find the peaks in the reference sample
        feature_table, error_table = create_feature_table(
            self._spectrum_dict,
            self._ref_peaks[self._peak_th],
            normalization=self._normalization
        )
        self._feature_table = feature_table
        self._error_table = error_table

    def require_feature_table(self) -> pd.DataFrame:
        if not check_attr(self, '_feature_table'):
            self.set_feature_table()
        return self._feature_table

    @property
    def feature_table(self) -> pd.DataFrame:
        return self.require_feature_table()

    @property
    def error_table(self) -> pd.DataFrame:
        self.require_feature_table()
        return self._error_table

    @property
    def peak_th(self) -> float:
        return self._peak_th

    @property
    def normalization(self) -> str:
        return self._normalization
