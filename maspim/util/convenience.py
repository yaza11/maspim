from copy import deepcopy
from typing import Iterable, Callable, Any
import numpy as np
import os
import pickle
import logging

import pandas as pd

logger = logging.getLogger(__name__)


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
            if check_attr(self, attr_name):
                return getattr(self, attr_name)
            else:
                return fallback_function(self, *args, **kwargs)

        return return_existing_wrapper

    return return_existing_decorator


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


def _format_float(v: float | Any, precision=4) -> str:
    v_str: str = str(v)
    return (str(round(float(v), precision))
            if ('.' in v_str) and (v_str.replace('.', '').isdigit())
            else v_str)


def _format_iterable(v: Iterable, n=10) -> str:
    v_str = '[' + ', '.join([_format_float(e) for i, e in enumerate(v) if i < n])
    v_str += ', ...]' if len(v) > n else ']'
    return v_str


def object_to_string(obj: object | dict, pad=0) -> str:
    out: list[str] = []
    if isinstance(obj, dict):
        dict_items = obj.items()
    else:
        dict_items = obj.__dict__.items()
    for i, (k, v) in enumerate(dict_items):
        if isinstance(v, np.ndarray):
            v_str = f'Numpy array of type {v.dtype}, with shape {v.shape}'
        elif isinstance(v, pd.DataFrame):
            v_str = (f'Pandas DataFrame with columns {_format_iterable(v.columns)}, '
                     f'indices {_format_iterable(v.index)} '
                     f'and shape {v.shape}')
        elif isinstance(v, dict):
            v_str = ('\n' + ' ' * (len(k) + 2))\
                .join([f'{k}: {v}' for k, v in v.items()])
        elif isinstance(v, list | tuple | set):
            v_str = _format_iterable(v)
        else:
            v_str = str(v)
        if i == 0:
            out.append(f'{k}: {v_str}')
        else:
            out.append(' ' * pad + f'{k}: {v_str}')
    return '\n'.join(out)


class Convenience:
    path_d_folder: str | None = None
    path_folder: str | None = None

    _save_attrs: set[str] | None = None
    _save_in_d_folder: bool = False

    @property
    def feature_table(self) -> pd.DataFrame | None:
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement a feature_table property'
        )

    def _get_disc_folder_and_file(
            self, tag: str | None = None
    ) -> tuple[str, str]:
        assert (check_attr(self, 'path_folder')
                or check_attr(self, 'path_file')), \
            'object does not have a path_folder attribute'

        class_name: str = str(self.__class__).split('.')[-1][:-2]

        if tag is None and check_attr(self, '_tag'):
            tag = self.__getattribute__('_tag')

        if tag is not None:
            file_name: str = f'{class_name}_{tag}.pickle'
        else:
            file_name: str = f'{class_name}.pickle'

        if self._save_in_d_folder:
            if not check_attr(self, 'path_d_folder'):
                logger.warning(
                    'object does not have a path_d_folder attribute, saving in '
                    f'{self.path_folder} instead'
                )
                folder: str = self.path_folder
            else:
                folder: str = self.path_d_folder
        else:
            assert check_attr(self, 'path_folder'), \
                'object does not have a path_folder attribute'
            folder: str = self.path_folder

        file: str = os.path.join(folder, file_name)

        return folder, file

    @property
    def save_file(self):
        return self._get_disc_folder_and_file()[1]

    def get_save_file(self, tag: str | None = None):
        return self._get_disc_folder_and_file(tag=tag)[1]

    def __repr__(self) -> str:
        return object_to_string(self)

    def _pre_load(self):
        pass

    def _post_load(self):
        pass

    def load(self, tag: str | None = None):
        self._pre_load()

        folder, file = self._get_disc_folder_and_file(tag=tag)

        if not os.path.exists(file):
            raise FileNotFoundError(
                f'found no saved object in folder '
                f'{folder if folder != "" else "."} with name '
                f'{os.path.basename(file)}'
            )

        with open(file, 'rb') as f:
            obj: object | dict = pickle.load(f)
            if type(obj) is not dict:  # legacy
                obj: dict[str, Any] = obj.__dict__
            # filter out attributes that are not supposed to be saved
            if has_save_attr := check_attr(self, '_save_attrs'):
                load_attr: set[str] = self._save_attrs & set(obj.keys())
            else:  # load everything
                load_attr: set[str] = set(obj.keys())
            if (
                    has_save_attr and
                    (len(discarded := obj.keys() - self._save_attrs) > 0)
            ):
                logger.warning(f'discarded attributes {discarded} when loading {file}')
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

    def save(self, tag: str | None = None):
        """Save class __dict__ instance to file."""
        self._pre_save()

        folder, file = self._get_disc_folder_and_file(tag)

        # discard all attributes that are not flagged as relevant
        keep_attributes: set[str] = set(self.__dict__.keys())
        if self._save_attrs is not None:
            keep_attributes &= self._save_attrs

        # new dict with only the desired attributes
        save_dict: dict[str, Any] = {key: self.__dict__[key] for key in keep_attributes}

        logger.info(f'saving image object with {keep_attributes} to {folder}')
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
