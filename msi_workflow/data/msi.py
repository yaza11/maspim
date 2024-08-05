"""This module implements the MSI class."""
from __future__ import annotations

import os
import pandas as pd
import logging

from msi_workflow.exporting.from_mcf.spectrum import Spectra
from msi_workflow.data.main import Data
from msi_workflow.project.file_helpers import search_keys_in_xml
from msi_workflow.util.convinience import check_attr

logger = logging.getLogger(__name__)


class MSI(Data):
    """
    Class to wrap and process mass spectrometry imaging data.

    This object needs a table with compounds as columns
    (plus the x and y coordiantes of the data pixels). Each row corresponds to a data pixel. The
    recommended way is to use the set_feature_table_from_spectra method, but it is always possible
    to inject the data by
    >>> msi._feature_table = ft
    where msi is the MSI instance and ft is a pandas dataframe with data, x and y columns.

    Example Usage
    -------------
    Import
    >>> from msi_workflow import MSI
    Initialize
    >>> msi = MSI(path_d_folder='path/to/your/d_folder.d')
    If there are multiple mis files in the parent folder, it is recommended to provide
    the path_mis_file parameter as well
    >>> msi = MSI(path_d_folder='path/to/your/d_folder.d', path_mis_file='path/to/your/mis/file.mis')
    In that case the object will infere the data pixel resolution from the mis file.

    Set the feature table (here assuming that a spectra object has been saved to disk before):
    >>> from msi_workflow import Spectra
    >>> spec = Spectra(path_d_folder='path/to/your/d_folder.d', load=True)
    >>> msi.set_feature_table_from_spectra(spec)

    Now we are ready to do some analysis, e.g. nonnegative matrix factorization
    >>> msi.plot_nmf(k=5)
    """

    _save_in_d_folder: bool = True
    _save_attrs: set[str] = {
        'd_folder',
        'mis_file',
        'distance_pixels',
        '_feature_table',  # it could be processed, so not necessarily redundant information
        'depth_section',
        'age_span'
    }

    def __init__(
            self,
            path_d_folder: str,
            *,
            path_mis_file: str | None = None,
            distance_pixels: int | float | None = None,
    ) -> None:
        """
        Initializer of MSI class.

        Parameters
        ----------
        path_d_folder : str
            path to the d-folder plus the name of the d-folder. Example: 'path/to/your/d_folder.d'.
        path_mis_file : str, optional
            path to the mis file as well as its name. Recommended to provide if there are multiple mis files.
        distance_pixels : int, optional
            The distance (in micrometer) between data points. Will be looked up in mis file if not provided.
        """
        self._set_files(path_d_folder, path_mis_file)
        
        if distance_pixels is not None:
            self._distance_pixels = distance_pixels

    def _set_files(self, path_d_folder: str, path_mis_file: str) -> None:
        path_folder1, d_folder = os.path.split(path_d_folder)
        if path_mis_file is not None:
            path_folder2, mis_file = os.path.split(path_mis_file)
            assert os.path.samefile(path_folder1, path_folder2), \
                "Mis and d folder should be in the same directory"
            self.mis_file: str = mis_file
        self.path_folder: str = path_folder1
        self.d_folder: str = d_folder

    @property    
    def path_d_folder(self) -> str:
        return os.path.join(self.path_folder, self.d_folder)

    @property
    def path_mis_file(self):
        return os.path.join(self.path_folder, self.mis_file)

    def set_distance_pixels(
            self, distance_pixels: float | int | None = None
    ):
        """Set distance of pixels in micrometers either directly or from mis."""
        if distance_pixels is not None:
            self._distance_pixels = distance_pixels
            return

        logger.info(f"reading pixel distance from {self.path_mis_file}")

        # should be x,x (distance in x, y in um)
        distances: str | list[str] = search_keys_in_xml(
            self.path_mis_file, ['Raster']
        )['Raster']
        if type(distances) is list:
            distance: str = distances[0]
            assert all([d == distance for d in distances]), \
                "found different raster sizes in mis file, cannot handle this"
        else:
            distance: str = distances
        distance_t: list[str] = distance.split(',')
        assert (d := distance_t[0]) == distance_t[1], \
            'cant handle grid with different distances in x and y'
        self._distance_pixels = float(d)
    
    def set_feature_table_from_spectra(self, spectra: Spectra):
        """
        Set the feature table from a spectra object.
        """
        assert check_attr(spectra, 'feature_table')

        self._feature_table: pd.DataFrame = spectra.feature_table.copy()
    

if __name__ == '__main__':
    pass
