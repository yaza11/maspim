import os
import re
import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd

from maspim.util.convenience import DFolderManager


def get_rxy(spot_names: Iterable[str]) -> np.ndarray[int]:
    # add R, x, y columns
    str_prefix: str = r'R(\d+)X'
    str_x: str = r'R\d+X(.*?)Y'
    str_y: str = r'Y(.*?)$'

    def rxy(name: str) -> list[int]:
        """Obtain x, y, and r value from name."""
        r: int = int(re.findall(str_prefix, name)[0])
        x: int = int(re.findall(str_x, name)[0])
        y: int = int(re.findall(str_y, name)[0])
        return [r, x, y]

    rxys: np.ndarray[int] = np.array([rxy(name) for name in spot_names])

    return rxys


def get_spots(path_d_folder: str, from_peaks: bool = None) -> pd.DataFrame:
    """Fetch spot info either from ImagingInfo.xml or peaks.sqlite"""
    if os.path.exists(os.path.join(path_d_folder, 'ImagingInfo.xml')) and (from_peaks is not True):
        ii = ImagingInfoXML(path_d_folder=path_d_folder)
        df = ii.feature_table
        return df.loc[:, ['spotName', 'R', 'x', 'y']]
    elif os.path.exists(file := os.path.join(path_d_folder, 'peaks.sqlite')):
        conn = sqlite3.connect(file)
        df = pd.read_sql_query(
            "SELECT SpotName,RegionNumber,XIndexPos,YIndexPos from Spectra",
            conn
        )
        df.columns = ['spotName', 'R', 'x', 'y']
        return df
    raise FileNotFoundError(
        f'Could not find peaks.sqlite or ImagingInfo.xml in {path_d_folder}'
    )


class ImagingInfoXML(DFolderManager):
    _feature_table = None

    def __init__(
            self,
            path_folder: str = None,
            path_d_folder: str = None,
            path_file: str = None
    ):
        assert (
                (path_folder is not None)
                or (path_d_folder is not None)
                or (path_file is not None)
        ), \
            'specify one of the parameters'

        if path_file is not None:
            self.path_file = path_file
            return
        if (path_folder is not None) and (path_d_folder is None):
            self.path_folder = path_folder
            self.set_d_folder(path_d_folder)
            path_d_folder = self.path_d_folder
        if path_file is None:
            self.path_file = os.path.join(path_d_folder, 'ImagingInfo.xml')

        assert os.path.exists(self.path_file), \
            f'make sure the file is named correctly, could not find {self.path_file}'

    def _re_all(self, key: str) -> np.ndarray[str]:
        with open(self.path_file, 'r') as f:
            xml: str = f.read()
            matches: list[str] = re.findall(rf'<{key}>(.*?)</{key}>', xml)
            return np.array(matches)

    @property
    def count(self) -> np.ndarray[int]:
        return self._re_all('count').astype(int)

    @property
    def indices(self) -> np.ndarray[int]:
        return self.count

    @property
    def spotName(self) -> np.ndarray[str]:
        return self._re_all('spotName')

    @property
    def minutes(self) -> np.ndarray[float]:
        return self._re_all('minutes').astype(float)

    @property
    def tic(self) -> np.ndarray[float]:
        return self._re_all('tic').astype(float)

    @property
    def maxpeak(self) -> np.ndarray[float]:
        return self._re_all('maxpeak').astype(float)

    def set_feature_table(self) -> None:
        RXYs: np.ndarray = get_rxy(self.spotName)
        self._feature_table = pd.DataFrame({
            'count': self.count,
            'spotName': self.spotName,
            'R': RXYs[:, 0],
            'x': RXYs[:, 1],
            'y': RXYs[:, 2],
            'minutes': self.minutes,
            'tic': self.tic,
            'maxpeak': self.maxpeak
        })

    @property
    def feature_table(self):
        if self._feature_table is None:
            self.set_feature_table()
        return self._feature_table
