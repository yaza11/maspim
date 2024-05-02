from __future__ import annotations

from data.cDataClass import Data
from Project.file_helpers import find_matches
from res.constants import elements
from util.manage_obj_saves import class_to_attributes, Data_nondata_columns

import re
import os
import pickle
import numpy as np
import pandas as pd


def handle_video_file(folder, file_name):
    file_path = os.path.join(folder, file_name)
    gray = pd.read_csv(file_path, sep=';', header=None)
    N_y, N_x = gray.shape
    X, Y = np.meshgrid(range(N_x), range(N_y))

    v_x = X.ravel()
    v_y = Y.ravel()
    v_gray = gray.to_numpy().ravel()
    return [v_gray, v_x, v_y]


def txt_to_vec(folder: str, file_name: str) -> np.ndarray[float]:
    """Read a txt file separated by ; and return a numpy vector."""
    file_path = os.path.join(folder, file_name)
    df = pd.read_csv(file_path, sep=';', header=None)
    return df.to_numpy().ravel()


class XRF(Data):
    def __init__(
            self, 
            path_folder: str,
            distance_pixels = None, 
            measurement_name: str = None
    ):
        self.verbose = False
        self.plts = False
        
        self.path_folder = path_folder
        if distance_pixels is not None:
            self.distance_pixles = distance_pixels
        if measurement_name is not None:
            self.measurement_name = measurement_name
        else:
            self._set_measurement_name()
            
    def _set_measurement_name(self):
        # folder should have measurement name in it --> a captial letter, 4 digits and 
        # a lower letter
        folder = os.path.split(self.path_folder)[1]
        pattern = r'^[A-Z]\d{3,4}[a-z]'
        
        match = re.match(pattern, folder)
        result = match.group() if match else None
        if result is None:
            raise OSError(
                f'Folder {folder} does not contain measurement name at beginning, please rename folder',
            )
        else:
            self.measurement_name = result

    def load(self) -> None:
        """Actions to performe when object was loaded from disc."""
        if __name__ == '__main__':
            raise RuntimeError('Cannot load obj from file where it is defined.')
        
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_folder, name), 'rb') as f:
            obj = pickle.load(f)
        
        self.__dict__ = obj.__dict__
        
        self.plts = False
        self.verbose = False

    def save(self, save_only_relevant_cols: bool = False) -> None:
        """Actions to performe before saving to disc."""
        if __name__ == '__main__':
            raise RuntimeError('Cannot save object from the file in which it is defined.')
        dict_backup = self.__dict__.copy()
        # delete all attributes that are not flagged as relevant
        existent_attributes = list(self.__dict__.keys())
        keep_attributes = set(existent_attributes) & class_to_attributes(self)
        nondata_columns_to_keep = list(Data_nondata_columns & set(self.feature_table.columns))
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)
        # drop data columns from current feature table
        if hasattr(self, 'feature_table') and save_only_relevant_cols:
            # make sure feature table is sorted
            self.feature_table = self.feature_table\
                .sort_values(by=['y', 'x']).reset_index(drop=True)
            self.feature_table = self.feature_table.loc[
                :, self.get_data_columns + nondata_columns_to_keep
            ].copy()
        
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup

    def get_element_txts(self, tag: str = None) -> tuple[str, dict[str, str]]:
        """
        Data is stored in txt files where each file has a name of the format
        [tag]_[el].txt, so group txt files based on tag and pick the group 
        closest to the tag.
        
        """
        if tag is None:
            tag = self.measurement_name
        
        files: list[str] = find_matches(
            folder=self.path_folder,
            file_types='txt',
            return_mode='all'
        )
        
        els = set(elements.Abbreviation) | set(['Video 1'])
        pres = []
        posts = []
        els_found = []
        for file in files:
            # split files at last occuring _
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
        
        file_group = ['_'.join([pre, post]) for pre, post in zip(pres, posts) if pre == closest_match]
        
        return closest_match, dict(zip(els_found, file_group))

    def set_feature_table_from_txts(self, **kwargs) -> None:
        # find all relevant files
        # tuple[str, dict[str, str]]
        self.prefix_files, files = self.get_element_txts(**kwargs)

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
        self.feature_table = pd.DataFrame(data=np.vstack(vecs).T, columns=keys)
    
    def get_data_columns(self):
        if ('feature_table' not in self.__dict__) or (self.feature_table is None):
            return None
        columns = self.feature_table.columns
        
        
        # data columns are elements
        columns_valid = [col for col in columns if
                         col in list(elements.Abbreviation)]
    
        data_columns = np.array(columns_valid)
        return data_columns
    
    def combine_with(self, other: XRF) -> XRF:
        """Combine two objects"""
        def both_have(attr: str, obj1 = self, obj2 = other) -> bool:
            return hasattr(obj1, attr) & hasattr(obj2, attr)
        
        assert type(self) == type(other), 'objects must be of the same type'
        assert 'depth' in self.feature_table.columns, \
            'found no depth column in first object'
        assert 'depth' in other.feature_table.columns, \
            'found no depth column in second object'
        assert self.feature_table.depth.max() <= other.feature_table.depth.min(), \
            'first object must have smaller depths and depth intervals cannot overlap'

        # determine the new folder
        paths: list[str] = [self.path_folder, other.path_folder]
        new_path: str = os.path.commonpath(paths)
        print(f'found common path: {new_path}')

        Data_new: XRF = XRF(
            path_folder=new_path, 
            distance_pixels=self.distance_pixels
        )
        
        self_df: pd.DataFrame = self.feature_table.copy()
        other_df: pd.DataFrame = other.feature_table.copy()

        # set min to 0
        other_df.loc[:, 'x'] -= other_df.x.min()
        # shift by last value of this df
        other_df.loc[:, 'x'] += self_df.x.max()
        if both_have('x_ROI', self_df, other_df):
            # set min to 0
            other_df.loc[:, 'x_ROI'] -= other_df.x_ROI.min()
            # shift by last value of this df
            other_df.loc[:, 'x_ROI'] += self_df.x_ROI.max()
        
        new_df: pd.DataFrame = pd.concat(
            [self_df, other_df], 
            axis=0
        ).reset_index(drop=True)
        Data_new.feature_table = new_df

        return Data_new


if __name__ == '__main__':
    pass