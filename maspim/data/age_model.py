"""This module implements the AgeModel class."""
import os
import numpy as np
import pandas as pd
import logging

from typing import Self
from collections.abc import Iterable

from matplotlib import pyplot as plt

from maspim.util import Convenience

logger = logging.getLogger(__name__)


def check_file_integrity(
        file: str, is_file: bool = True, suffixes: list[str] = None
) -> tuple[bool, str]:
    """Check if a given file exists and optionally is of right type."""
    if os.path.exists(file):
        if is_file != os.path.isfile(file):
            return False, f'{file} is a directory, not a file'
        elif is_file and (suffixes is not None):
            if (suffix := os.path.splitext(file)[1][1:]) not in suffixes:
                return False, f'{suffix=} is not an option (must be in {suffixes})'
            else:
                return True, ''
        else:
            return True, ''
    return False, f'{file=} does not exist'


class AgeModel(Convenience):
    """
    Offer depth-age conversions for tabled data.

    This object allows to read in tabulated age data, interpolate between
    missing depth and combine multiple objects.

    Example Usage
    -------------
    Usually the depth-age information is stored in a separate file. The file
    reader is a wrapper around the pandas read functions, hence, to read data
    in correctly, the user is referenced to the documentation of read_csv,
    read_excel, depending on the file type. For a tab separated file with
    column names 'age' and 'depth' the initalization can look like this:
    >>> from maspim import AgeModel
    >>> age_model = AgeModel(path_file='path/to/file.txt', sep='\t', index_col=False)
    The class expects the depth data to be in cm below seaflow and the age in
    years. Oftentimes this is not the case. Here, the add_depth_offset and
    convert_depth_scale methods are handy (call order does not matter, but a
    different depth offset is required after converting the depth-scale)
    >>> age_model.convert_depth_scale(1 / 10)  # converts mm to cm
    >>> age_model.add_depth_offset(500)  # age model starts add 500 cmbsf
    Those parameters can also be provided upon initialization
    >>> age_model = AgeModel(
    >>>     path_file='path/to/file.txt',
    >>>     depth_offset=5000,
    >>>     conversion_to_cm=1 / 10,
    >>>     sep='\t',
    >>>     index_col=False
    >>> )
    In this case the depth offset is applied first. By default, the age model
    will be saved in the same folder from which the data has been loaded
    >>> age_model.save()  # saves the object in 'path/to'
    This can be changed by providing a path
    >>> age_model.save('path/to/desired/folder')
    Age models can be combined, if the depths do not overlap
    >>> age_model1 = AgeModel(path1, ...)
    >>> age_model2 = AgeModel(path2, ...)
    >>> age_model_combined = age_model1 + age_model2
    """
    path_folder: str | None = None
    _in_file: str | None = None
    _save_file: str | None = None

    _save_in_d_folder: bool = True

    _save_attrs: set[str] = {
        'df',
        '_in_file',
        '_save_file',
        'column_age',
        'column_depth'
    }

    def __init__(
            self,
            *,
            path_file: str | None = None,
            depth: Iterable | None = None,
            age: Iterable | None = None,
            column_depth: str = 'depth',
            column_age: str = 'age',
            depth_offset: float | int = 0,
            conversion_to_cm: int | float = 1,
            **kwargs_read_file
    ):
        """
        Constructor for the AgeModel.

        Always defines a df attribute (dataframe containing the age model)
        Depending on the provided variables, sets a path_file, depth and age column.

        It is assumed that the age depth column is in cm and is absolute or that the depth_offset and
        the conversion_to_cm parameters are specified.

        :param path_file: file from which to load the age model, see _read_file for more information
        :param depth: depth vector to be used and
        :param age: age vector to be used for the age model, age and depth must
            have the same length
        :param column_depth: name of the depth column in the data frame
        :param column_age: name of the age column in the data frame
        :param depth_offset: depth offset to be added to the depth column
        :param conversion_to_cm: conversion factor to be applied to the depth column
        :param kwargs_read_file: additional keyword arguments passed on to the pandas reader
        """
        assert (path_file is not None) or ((depth is not None) and (age is not None)), \
            'provide either a file or a depth and age vector'

        self._set_files(path_file)

        self.column_depth: str = column_depth
        self.column_age: str = column_age
        if (depth is not None) and (age is not None):  # assign depth and age directly
            assert len(depth) == len(age), \
                'depth and age must have same number of entries'
            self.df: pd.DataFrame = pd.DataFrame({column_depth: depth, column_age: age})
        else:  # read from file
            self._read_file(depth_offset, conversion_to_cm, **kwargs_read_file)

    def _set_files(self, path_file: str | None) -> None:
        self._save_file: None | str = None if path_file is None else os.path.basename(path_file)
        if path_file is None:
            return

        # check if file is directory, in that case require an AgeModel.pickle file
        if os.path.isdir(path_file):
            assert 'AgeModel.pickle' in os.listdir(path_file), \
                (f'A folder ({path_file} was provided as input, but no saved '
                 f'"AgeModel.pickle" file was found')
            self.path_folder: str = path_file
            self._in_file: str = 'AgeModel.pickle'

        # possible file types from which to read age model
        file_types: list[str] = ['txt', 'csv', 'xlsx', 'pickle']
        valid, msg = check_file_integrity(path_file, is_file=True, suffixes=file_types)
        assert valid, msg

        self.path_folder: str = os.path.dirname(path_file)
        self._in_file: str = os.path.basename(path_file)

    @property
    def path_file(self) -> str:
        assert (self.path_folder is not None) and (self._save_file is not None)
        return os.path.join(self.path_folder, self._save_file)

    @path_file.setter
    def path_file(self, path_file: str) -> None:
        self.path_folder: str = os.path.dirname(path_file)
        self._save_file: str = os.path.basename(path_file)
        self._in_file: str = self._save_file

    def _read_file(self, depth_offset: float | int, conversion_to_cm: float | int, **kwargs) -> None:
        """
        Read the age model from the specified file.

        The provided file can be any of txt, csv, xlsx or a saved Age Model file (pickle file).
        It is assumed that columns are named 'age' and 'depth' (or similar) or specified.
        This method also handles the depth offset and conversion of the depth scale to cm. The age values are
        assumed to be in yrs b2k.
        """
        file: str = os.path.join(self.path_folder, self._in_file)

        if (suffix := os.path.splitext(file)[1]) in ('.csv', '.txt'):
            self.df: pd.DataFrame = pd.read_csv(file, **kwargs)
        elif suffix == '.xlsx':
            self.df: pd.DataFrame = pd.read_excel(file, **kwargs)
        else:  # suffix pickle
            self.load()

        # strip whitespaces
        self.df: pd.DataFrame = self.df.map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

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

        self.add_depth_offset(depth_offset=depth_offset)
        self.convert_depth_scale(factor=conversion_to_cm)

    def add_depth_offset(self, depth_offset: float | int):
        """Apply a depth offset to the depth column"""
        self.df.loc[:, self.column_depth] += depth_offset

    def add_age_offset(self, age_offset: float | int):
        """Apply a depth offset to the depth column"""
        self.df.loc[:, self.column_age] += age_offset

    def convert_depth_scale(self, factor: float | int):
        """Apply a conversion factor"""
        self.df.loc[:, self.column_depth] *= factor

    def convert_age_scale(self, factor: float | int):
        """Apply a conversion factor"""
        self.df.loc[:, self.column_age] *= factor

    @property
    def depth(self):
        return self.df.loc[:, self.column_depth].to_numpy()

    @property
    def age(self):
        return self.df.loc[:, self.column_age].to_numpy()

    def depth_to_age(
            self,
            depth: float | Iterable[float]
    ) -> float | np.ndarray[float]:
        """
        Return the corresponding core age (a b2k) for a given depth (cm).

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
            logger.warning('Depths not always strictly increasing!')
        if not np.all(np.diff(self.depth) >= 0):
            raise ValueError('Depths not always increasing!')
        # lineraly interpolate between values
        return np.interp(depth, self.depth, self.age)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.depth, self.age)
        ax.set_xlabel('Depth in cmbsf')
        ax.set_ylabel('Age in yrs b2k')
        ax.grid(True)

        plt.show()

    def __add__(self, other: Self) -> Self:
        """
        Allow the combination of age models.

        This action is only defined for age models with non-overlapping age models

        Parameters
        ----------
        other: another AgeModel instance

        Returns
        -------

        """
        # type(other) is type(self) more elegant?
        assert isinstance(other, AgeModel), 'Can only combine age models'
        assert self.depth[-1] <= other.depth[0], \
            f'last depth of first model has to be smaller than second but is {self.depth[-1]} and {other.depth[0]}'
        assert self.age[-1] <= other.age[0], \
            f'last age of first model has to be smaller than second but is {self.age[-1]} and {other.age[0]}'
        new_depth = np.concatenate([self.depth, other.depth])
        new_age = np.concatenate([self.age, other.age])
        new = AgeModel(depth=new_depth, age=new_age)
        return new
