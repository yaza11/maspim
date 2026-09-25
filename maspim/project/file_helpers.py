import os
import logging
import xml.etree.ElementTree as ET

import numpy as np

from typing import Iterable, Literal
from textdistance import damerau_levenshtein as textdistance

logger = logging.getLogger(__name__)


def find_matches(
        substrings: str | list[str] | None = None,
        files: None | list[str] = None,
        folder: str | None = None,
        file_types: str | list[str] | None = None,
        must_include_substrings: bool = False,
        return_mode: Literal['best', 'valid', 'all'] = 'all'
) -> str | list[str] | None:
    """
    In a folder (or in the specified files), find files mathcing the file type and/or substring(s)

    Either returns all matches or the closest match to the substring
    """
    assert (files is not None) or (folder is not None), \
        'Provide either the folder or a list of files.'
    assert return_mode in (return_modes := ('best', 'valid', 'all')), \
        f'return mode must be one of {return_modes}, not {return_mode}'

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
    if len(files) == 0:
        msg = f'did not find any file in {files} with {file_types=}'
        if must_include_substrings:
            msg += f'and {substrings=}'
        # premature exit for best
        # (this actually an error as best suggests there is at least one match)
        if return_mode == 'best':
            logger.error(msg)
            return
        logger.info(msg)
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


def find_files(
        folder_structure: dict[str, dict | str],
        *target_names: str,
        match_mode: Literal['exact', 'file_type', 'ends_with_name', 'keyword'] = 'exact',
        keyword: str = None,
        require_unique_matches: bool = True
) -> dict[str, list[str]] | dict[str, str]:
    # first level entries
    children: dict[str, dict | str] = folder_structure['children']
    # initiate dict with matches
    #  keys are the target names, values a  list of potential matches
    matches: dict[str, list[str]] = {k: [] for k in target_names}
    # iterate over entries
    for child in children:
        # get name of child
        name: str = child['name']
        if match_mode == 'file_type':
            suffix = name.split('.')[-1]
            for target in target_names:
                if target.split('.')[-1] == suffix:
                    matches[target].append(name)
        elif match_mode == 'exact':
            for target in target_names:
                if target == name:
                    matches[target].append(name)
        elif match_mode == 'ends_wth_name':
            for target in target_names:
                if name.endswith(target):
                    matches[target].append(name)
        elif match_mode == 'keyword':
            for target in target_names:
                if (target in name) and (keyword in name):
                    matches[target].append(name)
    if require_unique_matches:
        out = {}
        for k, v in matches.items():
            assert (n := len(v)) <= 1, \
                f'found target {n} matches for {k} with {match_mode=} and {keyword=} but was expecting zero or one.'
            if n == 0:
                continue
            out[k] = v[0]
        matches = out
    return matches


def get_mis_file(path_folder, name_file: str | None = None, return_mode='all') -> str | None:
    """Find the name of the mis file inside the .i folder"""
    # folder_structure = get_folder_structure(path_folder)
    # return find_files(folder_structure, 'mis', by_suffix=True)['mis']
    if name_file is None:
        name_file = os.path.basename(path_folder).split('.')[0] + '.mis'
    matches = find_matches(name_file, folder=path_folder, file_types='mis', return_mode=return_mode)
    if matches is None:
        raise FileNotFoundError(f'Could not find mis file inside {path_folder}')
    return matches


def get_mis_info(path_mis_file: str) -> dict[str, list | str | None]:
    xml_tree = ET.parse(path_mis_file)
    root = xml_tree.getroot()

    # get area
    for child in root:
        if child.tag == 'Area':
            break
    else:
        raise ValueError('Could not find area')
    resolution = None
    points = []
    for el in child:
        if el.tag == 'Raster':
            resolution = el.text
        elif el.tag == 'Point':
            points.append(el.text)
    return dict(Raster=resolution, Point=points)


def get_d_folder(path_folder, return_mode: str = 'all') -> str | list[str] | None:
    """Get the name of the .d folder inside the .i folder"""
    matches = find_matches(folder=path_folder, file_types='d', return_mode=return_mode)
    if matches is None:
        raise FileNotFoundError(f'No d folder found inside {path_folder}')
    return matches


def search_keys_in_xml(path_mis_file: str, keys: Iterable[str]) -> dict[str, list[str] | str]:
    # initiate list of lists for values
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


def get_resolution_msi(path_mis_file: str) -> float:
    """
    Read the spot resolution from the mis file and return the resolution in micrometer. Assumes that the resolution
    in the x and y direction is the same."""
    distances: str | list[str] = search_keys_in_xml(
        path_mis_file, ['Raster']
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
    return float(d)


def get_mis_image_file(path_mis_file: str) -> str:
    return search_keys_in_xml(path_mis_file, ['ImageFile'])['ImageFile']


if __name__ == '__main__':
    # substring = 'S0343c'
    # folder = r'D:\Cariaco line scan Xray\uXRF slices\S0343c_490-495cm'
    #
    # print(find_matches(
    #     [substring, 'Fe'],
    #     folder=folder,
    #     file_type='txt',
    #     return_mode='valid',
    #     must_include_substrings=True
    # ))
    pass
