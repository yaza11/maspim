import os
import numpy as np
import pandas as pd
import pickle

from collections.abc import Iterable

def check_file_integrity(
        file: str, is_file: bool = True, suffixes: list[str] = None
) -> bool:
    """Check if a given file exists and optionally is of right type."""
    if os.path.exists(file):
        if is_file != os.path.isfile(file):
            return False
        elif is_file and (suffixes is not None):
            if (suffix := os.path.splitext(file)[1][1:]) not in suffixes:
                return False
            else:
                return True
        else:
            return True
    return False


class AgeModel:
    def __init__(
        self, 
        path_file: str | None = None,
        depth: Iterable | None = None,
        age: Iterable | None = None,
        column_depth: str = 'depth',
        column_age: str = 'age',
        **kwargs_read_file
   ):
        assert (path_file is not None) or ((depth is not None) and (age is not None)),\
            'provide either a file or a depth and age vector'

        # assign depth and age directly
        if (depth is not None) and (age is not None):
            assert len(depth) == len(age), \
                'depth and age must have same number of entries'
            self.df: pd.DataFrame = pd.DataFrame({column_depth: depth, column_age: age})
        # read from file
        else:
            self.path_file: str = path_file
            self.column_depth: str = column_depth
            self.column_age: str = column_age
            self.read_file(**kwargs_read_file)
            
    def read_file(self, **kwargs) -> None:
        file_types: list[str] = ['txt', 'csv', 'xlsx', 'pickle']
        if os.path.isdir(self.path_file) \
                and ('AgeModel.pickle' in os.listdir(self.path_file)):
            self.load()
            return
        assert check_file_integrity(self.path_file, is_file=True, suffixes=file_types), \
            f'check file name and type (must be in {file_types})'
            
        if (suffix := os.path.splitext(self.path_file)[1]) in ('.csv', '.txt'):
            self.df: pd.DataFrame = pd.read_csv(self.path_file, **kwargs)
        elif suffix == '.xlsx':
            self.df: pd.DataFrame = pd.read_excel(self.path_file, **kwargs)
        # strip whitespaces
        try:
            self.df: pd.DataFrame = self.df.map(lambda x: x.strip() if isinstance(x, str) else x)
        except:
            self.df: pd.DataFrame = self.df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        self.df.columns = self.df.columns.str.strip()

        # try to infere missing columns
        if self.column_depth not in self.df.columns:
            try_cols = {'depth', 'd', 'mbsf', 'depths'}
            for col in self.df.columns:
                # account for possible units e.g. "depth [m]" or "depth in cm"
                if col.lower().split()[0] in try_cols:
                    self.column_depth = col
                    break
            # in case no match was found
            if self.column_depth not in self.df.columns:
                raise KeyError('could not find a column for depth')

        if self.column_age not in self.df.columns:
            try_cols = {'age', 'ages', 'yrs'}
            for col in self.df.columns:
                if col.lower().split()[0] in try_cols:
                    self.column_age = col
                    break
            # in case no match was found
            if self.column_age not in self.df.columns:
                raise KeyError(f'could not find a column for age {self.column_age} in {list(self.df.columns)}')
                
    def add_depth_offset(self, depth_offset: float | int):
        self.df['depth'] += depth_offset

    def convert_depth_scale(self, factor: float):
        self.df['depth'] *= factor
            
    @property
    def depth(self):
        return self.df.depth.to_numpy()
    
    @property
    def age(self):
        return self.df.age.to_numpy()
        
    def depth_to_age(self, depth: float) -> float:
        """
        Return the corresponding core age (a b2k) for a given depth (m).

        Parameters
        ----------
        depth : float
            depth in core.

        Raises
        ------
        ValueError
            If depths in age model are not monotonically increasing.

        Returns
        -------
        float
            interpolated age in a b2k.

        """
        if not np.all(np.diff(self.depth) > 0):
            print('Depths not always strictly increasing!')
        if not np.all(np.diff(self.depth) >= 0):
            raise ValueError('Depths not always increasing!')
        # lineraly interpolate between values
        return np.interp(depth, self.depth, self.age)
    
    
    def __add__(self, other):
        assert isinstance(other, AgeModel), 'Can only combine age models'
        assert self.depth[-1] <= other.depth[0], \
            'last depth of first model has to be smaller than second'
        assert self.age[-1] <= other.age[0], \
            'last age of first model has to be smaller than second'
        new_depth = np.concatenate([self.depth, other.depth])
        new_age = np.concatenate([self.age, other.age])
        new = AgeModel(depth=new_depth, age=new_age)
        return new
    
    def __repr__(self):
        return str(self.df)
    
    def load(self):
        # load from path_file
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_file, name), 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__
        
    
    def save(self, target_folder = None):
        if target_folder is None:
            target_folder = self.path_file
        assert os.path.isdir(target_folder), \
            'specify a directory, not a file'
        
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(target_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)