from copy import deepcopy
from typing import Iterable, Callable
import numpy as np
import os
import pickle
import logging

from util.manage_obj_saves import class_to_attributes

logger = logging.getLogger("msi_workflow." + __name__)


def verbose_function(func=None):
    def verbose_wrapper(self, *args, **kwargs):
        if self.verbose:
            print(f"Calling function: {func.__name__}")
        return func(self, *args, **kwargs)

    if func is None:
        return lambda f: verbose_function(f)

    return verbose_wrapper


def return_existing(attr_name: str) -> Callable:
    """Return attribute if it exists, otherwise fall back to function."""

    def return_existing_decorator(fallback_function):
        def return_existing_wrapper(self, *args, **kwargs):
            if hasattr(self, attr_name):
                return getattr(self, attr_name)
            else:
                return fallback_function(self, *args, **kwargs)

        return return_existing_wrapper

    return return_existing_decorator


class Convinience:
    def load(self):
        assert hasattr(self, 'path_folder'), \
            'object does not have a path_folder attribute'

        name: str = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        file: str = os.path.join(self.path_folder, name)

        assert os.path.exists(
            file), f'found no saved object in folder {self.path_folder if self.path_folder != "" else "."}'

        with open(file, 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__
        logger.info(f'loading object with keys {obj.__dict__.keys()}')

    def save(self):
        assert hasattr(self, 'path_folder'), \
            'object does not have a path_folder attribute'
        # delete all attributes that are not flagged as relevant
        dict_backup: dict = self.__dict__.copy()
        keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
        existent_attributes = list(self.__dict__.keys())
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)

        logger.info(f'saving image object with {self.__dict__.keys()} to {self.path_folder}')
        name: str = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup

    def copy(self):
        return deepcopy(self)

    def get_closest_mz(
            self,
            mz: float | str,
            cols: Iterable | None = None,
            max_deviation: float | None = None,
            return_deviation: bool = False
    ) -> str | tuple[str, float] | None:
        """
        Return the closest mz value in the msi data.

        Parameters
        ----------
        mz : mz value as float or string of one compound for which the mz value
            within the data is not known
        cols: Iterable | None, optional
            The columns in which to search the closest value. If cols is None,
            tries to get cols of feature_table or feature_table_zone_averages.
            If both fail, raises an attribute error.
        max_deviation : float, optional
            the maximum of allowed deviation of the mz value from that given
        return_deviation: bool, optional
            If this is True, return a tuple with closest mz found, deviation.
        Returns
        -------
        None
            Returns None if the closest mz is above deviation.
        tuple
            Returns a tuple of mz and deviation if return_deviation=True
        float
            returns the closest mz value within the provided list of mz values if
            deviation is small enough, otherwise returns None.
        """
        if cols is None:
            if hasattr(self, 'feature_table'):
                cols = np.array(self.feature_table.columns).astype(str)
            elif hasattr(self, 'feature_table_zone_averages'):
                cols = np.array(self.feature_table_zone_averages.columns).astype(str)
            else:
                raise AttributeError('Could not find feature table. Pass cols')
        else:
            cols = np.array(cols).astype(str)

        # check if mz already in cols
        if str(mz) in cols:
            if return_deviation:
                return str(mz), 0
            return str(mz)

        # find closest mz in numeric columns
        mz_f = float(mz)
        cols_f = np.array([float(col) for col in cols if str(col).replace('.', '').isnumeric()])

        # get idx of closest mz
        idx = (np.abs(mz_f - cols_f)).argmin()
        # get deviation
        deviation = np.abs(mz_f - cols_f[idx])
        if return_deviation:
            out = str(cols_f[idx]), deviation
        else:
            out = str(cols_f[idx])
        if max_deviation is None:
            return out
        # check if deviation is within tolerance
        elif deviation <= max_deviation:
            return out
        return None
