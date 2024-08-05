"""This module implements the XRF class."""
from __future__ import annotations

import re
import os
import numpy as np
import pandas as pd

from msi_workflow.data.main import Data
from msi_workflow.project.file_helpers import find_matches
from msi_workflow.res.constants import elements


def handle_video_file(
        folder: str,
        file_name: str
) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a csv file with image data and returns values as vectors.

    This function reads a csv file with grayscale values of the photo,
    constructs the x and y pixel coordinates and returns the flattened photo and coordinates.

    Parameters
    ----------
    folder : str
        Absolute path to the folder with the csv file
    file_name: str
        Name of the file.

    Returns
    -------
    out: list[np.ndarray, np.ndarray, np.ndarray]
        List containing the grayscale vector, the x coordinate vector and the y coordinate vector.

    """
    file_path: str = os.path.join(folder, file_name)
    gray: pd.DataFrame = pd.read_csv(file_path, sep=';', header=None)
    N_y, N_x = gray.shape
    X, Y = np.meshgrid(range(N_x), range(N_y))

    v_x: np.ndarray = X.ravel()
    v_y: np.ndarray = Y.ravel()
    v_gray: np.ndarray = gray.to_numpy().ravel()
    return [v_gray, v_x, v_y]


def txt_to_vec(folder: str, file_name: str) -> np.ndarray[float]:
    """Read a txt file separated by ; and return a numpy vector."""
    file_path = os.path.join(folder, file_name)
    df = pd.read_csv(file_path, sep=';', header=None)
    return df.to_numpy().ravel()


class XRF(Data):
    """
    Class to wrap and process mass spectrometry imaging data.

    This object compiles a feature table from txt files in a folder (txt files were exported,
    example: folder name 'S0343a_480-485cm', file name 'S0343a_Al.txt'). It is recommended to leave
    the folder name consistent with the file names. Further, the original image should be exported and
    contian the keyword 'Mosaic'. This will be necessary for the image classes.

    The feature table also contains information about the x and y coordiantes of the data pixels.
    Each row corresponds to a data pixel.

    Example Usage
    -------------
    Import
    >>> from msi_workflow import XRF
    Initialize
    >>> xrf = XRF(path_folder='path/to/your/folder')
    By default the measurement name will be infered from the folder name and the distance_pixels
    read from the bcf file. If multiple exports are located in the folder, the measurement name
    should be changed to the desired export:
    >>> xrf = XRF(path_folder='path/to/your/folder', measurement_name='D0343a')
    Set the feature table.
    >>> xrf.set_feature_table_from_txts()

    Now we are ready to do some analysis, e.g. nonnegative matrix factorization
    >>> xrf.plot_nmf(k=5)
    """
    _save_attrs: set[str] = {
        'default_file_type',
        'measurement_name',
        'prefix_files',
        'distance_pixels',
        '_feature_table',  # it could be processed, so not necessarily redundant information
        'depth_section',
        'age_span'
    }

    def __init__(
            self, 
            path_folder: str,
            distance_pixels: int | None = None,
            measurement_name: str = None
    ) -> None:
        """
        Initialize with a folder.

        """
        self.path_folder = path_folder
        if distance_pixels is not None:
            self.distance_pixels = distance_pixels
        if measurement_name is not None:
            self.measurement_name = measurement_name
        else:
            self._set_measurement_name()
            
    def _set_measurement_name(self):
        """
        Infer the measurement name from the folder name.

        Folder should have measurement name in it --> a capital letter, 4 digits and
        a lower letter. Example: S0343a
        """
        folder = os.path.split(self.path_folder)[1]
        pattern = r'^[A-Z]\d{3,4}[a-z]'
        
        match = re.match(pattern, folder)
        result = match.group() if match else None
        if result is None:
            raise OSError(
                f'Folder {folder} does not contain measurement name at ' +
                f'beginning, please rename folder ' +
                'or provide the measurement name upon initialization.',
            )
        else:
            self.measurement_name = result

    def _get_element_txts(self, tag: str | None = None) -> tuple[str, dict[str, str]]:
        """
        Data is stored in txt files where each file has a name of the format
        [tag]_[el].txt, so group txt files based on tag and pick the group 
        closest to the tag.

        Parameters
        ----------
        tag: str, optional
            A substring that must be present in the file name. This defaults to the measurement name.

        Returns
        -------
        closest_match: str
            The file with the closest match to the tag.
        res_dict: dict[str, str]
            A dictionary mapping elements to file names.
        """
        if tag is None:
            tag = self.measurement_name
        
        files: list[str] = find_matches(
            folder=self.path_folder,
            file_types='txt',
            return_mode='all'
        )
        
        els = set(elements.Abbreviation) | {'Video 1'}
        pres = []
        posts = []
        els_found = []
        for file in files:
            # split files at last occurring _
            *pre, post = file.split('_')
            pre = '_'.join(pre)  # in case there are multiple _ in the name
            el = post.split('.')[0]  # split of the .txt
            if el not in els:
                continue
            pres.append(pre)
            posts.append(post)
            els_found.append(el)
            
        closest_match: str = find_matches(
            files=list(set(pres)),
            substrings=tag
        )
        
        file_group = [
            '_'.join([pre, post])
            for pre, post in zip(pres, posts) if pre == closest_match
        ]
        
        return closest_match, dict(zip(els_found, file_group))

    def set_feature_table_from_txts(self, **kwargs) -> None:
        """
        Set the feature table with element images from txt files present in the folder.

        Parameters
        ----------
        kwargs: dict,
            Optional keyword arguments to be passed to _get_elements_txts.
        """
        # find all relevant files
        # tuple[str, dict[str, str]]
        self.prefix_files, files = self._get_element_txts(**kwargs)

        vecs: list[np.ndarray[float]] = []
        keys: list[str] = []

        # file names of data are in format
        # [S | D][measurement_number][measurement_key]_[element | Video 1].[txt]
        # example: D0343c_Mg.txt
        for element, file in files.items():
            if 'Video' in file:
                vecs.extend(
                    handle_video_file(self.path_folder, file))
                keys.extend(['L', 'x', 'y'])
            else:
                vecs.append(txt_to_vec(self.path_folder, file))
                keys.append(element)
        # combine to feature_table
        self._feature_table = pd.DataFrame(data=np.vstack(vecs).T, columns=keys)


if __name__ == '__main__':
    pass
