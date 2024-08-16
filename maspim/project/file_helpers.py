import os
import re
import sqlite3

import numpy as np
import pandas as pd

from typing import Iterable
from textdistance import damerau_levenshtein as textdistance


def find_matches(
        substrings: str | list[str] | None = None, 
        files: None | list[str] = None, 
        folder: str | None = None, 
        file_types: str | list[str] | None = None,
        must_include_substrings: bool = False,
        return_mode: str = 'best'
) -> str | list[str]:
    assert (files is not None) or (folder is not None), \
        'Provide either the folder or a list of files.'
    assert return_mode in ('best', 'valid', 'all'), \
        f'return mode must be one of "best" or "valid", not {return_mode}'
    
    if substrings is None:
        substrings = ['']
    elif type(substrings) is str:
        substrings = [substrings]
    if type(file_types) is str:
        file_types = [file_types]
    
    if files is None:
        files = os.listdir(folder)
    
    if must_include_substrings:
        # exclude files that do not contain substring
        files = [
            file for file in files if 
            all(substring in file for substring in substrings)
        ]    
    if file_types is not None:
        # exclude files whose suffix does not match
        files = [file for file in files if file.split('.')[-1] in file_types]
    # return all files matching criteria
    if return_mode in ('valid', 'all'):
        return files
    elif return_mode == 'best':
        distances = [textdistance(''.join(substrings), file) for file in files]
        idx_min = np.argmin(distances)
        return files[idx_min]

def get_folder_structure(path):
    # Initialize the result dictionary with folder 
    # name, type, and an empty list for children 
    result = {
        'name': os.path.basename(path), 
        'type': 'folder', 
        'children': []
    } 
  
    # Check if the path is a directory 
    if not os.path.isdir(path): 
        return result 
  
    # Iterate over the entries in the directory 
    for entry in os.listdir(path): 
       # Create the full path for the current entry 
        entry_path = os.path.join(path, entry) 
  
        # If the entry is a directory, recursively call the function 
        if os.path.isdir(entry_path): 
            result['children'].append(get_folder_structure(entry_path)) 
        # If the entry is a file, create a dictionary with name and type 
        else: 
            result['children'].append({'name': entry, 'type': 'file'}) 
  
    return result 


def find_files(folder_structure: dict[str, dict | str], *names, by_suffix=False):
    # first level entries
    children = folder_structure['children']
    # initiate dict with matches
    matches = {}
    # iterate over entries
    for child in children:
        # get name of child
        name = child['name']
        if by_suffix:
            name = name.split('.')[-1] if '.' in name else ''
        if name in names:
            matches[name] = child['name']
    return matches


def get_mis_file(path_folder, name_file: str | None = None) -> str:
    """Find the name of the mis file inside the .i folder"""
    # folder_structure = get_folder_structure(path_folder)
    # return find_files(folder_structure, 'mis', by_suffix=True)['mis']
    if name_file is None:
        name_file = os.path.basename(path_folder).split('.')[0] + '.mis'
    return find_matches(name_file, folder=path_folder, file_types='mis')


def get_d_folder(path_folder, return_mode='best') -> str:
    """Get the name of the .d folder inside the .i folder"""
    return find_matches(folder=path_folder, file_types='d', return_mode=return_mode)

def search_keys_in_xml(path_mis_file, keys):
    # iniate list of lists for values
    out_dict = {key: [] for key in keys}
    # open xml
    with open(path_mis_file) as xml:
        # parse through lines
        for line in xml:
            line = line.replace('/', '')
            # search for keys
            for key in keys:
                key_xml = f'<{key}>'
                if key_xml in line:
                    value = line.split(key_xml)[1]
                    out_dict[key].append(value)
    for key, value in out_dict.items():
        if len(value) == 1:
            out_dict[key] = value[0]
    return out_dict


def get_image_file(path_folder):
    path_mis_file = os.path.join(path_folder, get_mis_file(path_folder))
    return search_keys_in_xml(path_mis_file, ['ImageFile'])['ImageFile']


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


class ImagingInfoXML:

    _feature_table = None

    def __init__(
            self,
            path_folder: str | None = None,
            path_d_folder: str | None = None,
            path_file: str | None = None
    ):
        assert (
                (path_folder is not None)
                or (path_d_folder is not None)
                or (path_file is not None)
        ), \
            'specify one of the parameters'

        if path_file is not None:
            self.path_file = path_file
        elif (path_folder is not None) and (path_d_folder is None):
            path_d_folder = os.path.join(path_folder, get_d_folder(path_folder))
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


def get_spots(path_d_folder: str) -> pd.DataFrame:
    if os.path.exists(os.path.join(path_d_folder, 'ImagingInfo.xml')):
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

if __name__ == '__main__':
    substring = 'S0343c'
    folder = r'D:\Cariaco line scan Xray\uXRF slices\S0343c_490-495cm'

    print(find_matches(
        [substring, 'Fe'],
        folder=folder,
        file_type='txt',
        return_mode='valid',
        must_include_substrings=True
    ))
