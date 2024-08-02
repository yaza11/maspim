from copy import deepcopy
from typing import Iterable, Callable, Any, Self
import numpy as np
import os
import pickle
import logging

from msi_workflow.util.manage_obj_saves import class_to_attributes

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


class_in_d_folder = {  # True if saved in d-folder
    'ImageClassified': False,
    'ImageROI': False,
    'ImageSample': False,
    'SampleImageHandlerMSI': False,
    'SampleImageHandlerXRF': False,
    'TimeSeries': True,
    'Spectra': True,
    'XRF': False,
    'MSI': True,
    'AgeModel': True,
    'XRay': False,
    'Mapper': False
}


def check_attr(obj, attr_name: str, check_nonempty: bool = False) -> bool:
    """
    Check whether an attribute exists and is valid.

    :param obj: Object to check
    :param attr_name: Name of the attribute
    :param check_nonempty: If True then also check if any of the values truthy
    """
    valid: bool = hasattr(obj, attr_name) and (getattr(obj, attr_name) is not None)
    if not check_nonempty:
        return valid
    return valid and np.any(getattr(obj, attr_name))


class Convinience:
    def _get_disc_folder_and_file(self) -> tuple[str, str]:
        assert (check_attr(self, 'path_folder')
                or check_attr(self, 'path_file')), \
            'object does not have a path_folder attribute'

        class_name: str = str(self.__class__).split('.')[-1][:-2]
        file_name: str = class_name + '.pickle'

        if class_in_d_folder[class_name]:
            folder: str = self.path_d_folder
        else:
            folder: str = self.path_folder

        file: str = os.path.join(folder, file_name)

        return folder, file

    def __repr__(self) -> str:
        out: list[str] = []
        for k, v in self.__dict__.items():
            out.append(f'{k}: {str(v)}')
        return '\n'.join(out)

    def _post_load(self):
        pass

    def _pre_load(self):
        pass

    def load(self):
        self._pre_load()

        folder, file = self._get_disc_folder_and_file()

        if not os.path.exists(file):
            raise FileNotFoundError(
                f'found no saved object in folder '
                f'{folder if folder != "" else "."} with name '
                f'{os.path.basename(file)}'
            )

        # for backwards compatibility, filter out attributes that are no longer
        # desired to load
        filter_attr = class_to_attributes(self)

        with open(file, 'rb') as f:
            obj: object | dict = pickle.load(f)
            if type(obj) is not dict:  # legacy
                obj: dict[str, Any] = obj.__dict__
            # filter out attributes that are not supposed to be saved
            load_attr: set[str] = filter_attr & set(obj.keys())
            # generate new dict, that only has the desired attributes
            obj_new: dict[str, Any] = {key: obj[key] for key in load_attr}
            # merge the objects dict with the disk dict, overwriting
            # instance attributes with saved once, if they both exist
            self.__dict__ |= obj_new
            
        logger.info(f'loaded {self.__class__.__name__} with keys {load_attr}')

        self._post_load()

    def _pre_save(self):
        pass

    def _post_save(self):
        pass

    def save(self):
        """Save class __dict__ instance to file."""
        self._pre_save()

        folder, file = self._get_disc_folder_and_file()

        # discard all attributes that are not flagged as relevant
        keep_attributes: set[str] = set(self.__dict__.keys()) & class_to_attributes(self)

        # new dict with only the desired attributes
        save_dict: dict[str, Any] = {key: self.__dict__[key] for key in keep_attributes}

        logger.info(f'saving image object with {self.__dict__.keys()} to {folder}')
        with open(file, 'wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

        self._post_save()

    def copy(self):
        return deepcopy(self)

    def get_closest_mz(
            self,
            mz: float | str,
            cols: Iterable | None = None,
            max_deviation: float | None = None,
            return_deviation: bool = False
    ) -> str | None | tuple[str | None, float]:
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
            if check_attr(self, 'feature_table'):
                cols = np.array(self.feature_table.columns).astype(str)
            else:
                raise AttributeError('Could not find feature table. Pass cols')
        else:
            cols = np.array(cols).astype(str)

        # check if mz already in cols
        if str(mz) in cols:
            return (str(mz), 0) if return_deviation else str(mz)

        # find closest mz in numeric columns
        try:
            mz_f = float(mz)
        except ValueError:
            raise AttributeError(
                f'Tried to find {mz}, which is not in the feature table and also not a number.' +
                ' Make sure the feature you want to get is actually in the feature table!'
            )
        cols_f = np.array([
            float(col)
            for col in cols
            if str(col).replace('.', '').isnumeric()
        ])

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
        return (None, deviation) if return_deviation else None
