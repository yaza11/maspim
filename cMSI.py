from __future__ import annotations
from util.manage_obj_saves import class_to_attributes, Data_nondata_columns

from exporting.from_mcf.cSpectrum import Spectra

from data.cDataClass import Data
from Project.file_helpers import get_mis_file, search_keys_in_xml

import os
import pickle
import pandas as pd
import numpy as np

class MSI(Data):
    def __init__(
            self,
            path_d_folder: str,
            distance_pixels: int | float | None = None
    ):
        self.path_d_folder = path_d_folder
        
        if distance_pixels is not None:
            self.distance_pixels = distance_pixels

        self.plts = False
        self.verbose = False
        
    def load(self, **kwargs) -> None:
        """Actions to performe when object was loaded from disc."""
        if __name__ == '__main__':
            raise RuntimeError('Cannot load obj from file where it is defined.')
        
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_d_folder, name), 'rb') as f:
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
        with open(os.path.join(self.path_d_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup

    def set_distance_pixels(
            self, distance_pixels: float | int | None = None
    ):
        """set distance of pixels in micrometers either directly or from mis."""
        if distance_pixels is not None:
            self.distance_pixels = distance_pixels
            return 
        
        # move up one folder
        path_folder = os.path.dirname(self.path_d_folder) 
        path_mis_file = os.path.join(path_folder, get_mis_file(path_folder))
        # should be x,x (distance in x, y in um)
        distances: str | list[str] = search_keys_in_xml(
            path_mis_file, ['Raster']
        )['Raster']
        if type(distances) is list:
            distance: str = distances[0]
            assert all([d == distance for d in distances]), \
            "found different raster sizes in mis file, cannot handle this"
        else:
            distance: str = distances
        distance_t: tuple[str, str] = distance.split(',')
        assert (d:= distance_t[0]) == distance_t[1], \
            'cant handle grid with different distances in x and y'
        self.distance_pixels = float(d)
    
    def set_feature_table_from_spectra(self, spectra: Spectra):
        has_df = hasattr(spectra, 'feature_table')
        assert has_df or hasattr(spectra, 'line_spectra'), \
            'spectra object must have a feature table'
        if has_df:
            self.feature_table = spectra.feature_table.copy()
        else:
            self.feature_table: pd.DataFrame = spectra.binned_spectra_to_df().copy()
    
    def get_data_columns(self):
        if ('feature_table' not in self.__dict__) or (self.feature_table is None):
            return None
        columns = self.feature_table.columns
        # only return cols with masses
        columns_valid = [col for col in columns if str(
            col).replace('.', '', 1).isdigit()]
        
        data_columns = np.array(columns_valid)
        return data_columns
    
    def combine_with(self, other: MSI) -> MSI:
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
        paths: list[str] = [self.path_d_folder, other.path_d_folder]
        new_path: str = os.path.commonpath(paths)
        print(f'found common path: {new_path}')

        Data_new = MSI(
            path_d_folder=new_path, 
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
    

def test_features():
    o = MSI('490-495', 'Alkenones')
    o.plt_photo()
    o.load_feature_table()
    o.plt_NMF(k=3)
    o.plt_PCA()
    o.plt_kmeans(n_clusters=3)
    o.plt_img_from_feature_table()


if __name__ == '__main__':
    DC = MSI((505, 510), 'FA')
    DC.load()
    DC.plts = True
    DC.sget_photo_ROI()
