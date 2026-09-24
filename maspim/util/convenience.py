from copy import deepcopy
from typing import Iterable, Callable, Any, Self
import numpy as np
import os
import pickle
import logging

import pandas as pd

from maspim.project.file_helpers import get_d_folder, get_mis_file

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
    """Convert data of an object to a string for better printing"""
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


def get_disk_file(obj: object, path_folder: str, tag: str = None) -> str:
    """
    Get the disk file name for an object.

    The file is prefixed by maspim to mark maspim objects followed by the folder name from which it originates to avoid
    potential clashes if multiple objects are stored in the same folder, followed by the object name and finally a tag
    (tag is optional).
    """
    if hasattr(obj, "__name__"):  # for uninitialized class objects
        class_name: str = obj.__name__
    else:  # for initialized class objects
        class_name: str = obj.__class__.__name__

    file_name_prefix: str = 'maspim'
    # strip .i or .d from path_folder
    folder: str = os.path.basename(path_folder)
    if folder.endswith('.i') or folder.endswith('.d'):
        folder = folder[:-2]

    if tag is not None:
        file_name: str = f'{file_name_prefix}_{folder}_{class_name}_{tag}.pickle'
    else:
        file_name: str = f'{file_name_prefix}_{folder}_{class_name}.pickle'
    return file_name


class MzUtility:
    _feature_table: pd.DataFrame = None

    @property
    def feature_table(self) -> None | pd.DataFrame:
        return self._feature_table

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


class Convenience:
    """Object for managing file storage."""
    d_folder: str = None
    path_folder: str = None
    path_file: str = None

    _save_attrs: set[str] = None  # list of properties to be saved on file
    _save_in_d_folder: bool = False

    def __init__(self, path_folder: str = None):
        self.path_folder = path_folder

    @classmethod
    def from_file(cls, path_folder: str = None, path_file: str = None, tag: str = None) -> Self:
        assert (path_folder is not None) ^ (path_file is not None), 'provide either path_folder or path_file (but not both)'
        if (path_file is not None) and (tag is not None):
            logger.warning('tag is ignored when loading from file')

        new = cls(path_folder)
        if path_folder is not None:
            path_file = new.get_save_file(path_folder=path_folder, tag=tag)
        new.load(path_file=path_file, tag=tag)
        return new

    def __repr__(self) -> str:
        return object_to_string(self)

    @property
    def save_in_d_folder(self) -> bool:
        return self._save_in_d_folder

    def get_save_file(self, path_folder: str = None, tag: str = None) -> str:
        """Return the folder and the file name (including the path)"""
        assert (check_attr(self, 'path_folder')
                or check_attr(self, 'path_file')), \
            'object does not have a path_folder attribute'
        if path_folder is not None:
            path_folder: str = path_folder
            if self._save_in_d_folder:
                assert path_folder.endswith('.d'), \
                    f'{self.__class__.__name__} is supposed to be stored in d_folder, but specified path_folder {path_folder} does not end with .d'
        elif self._save_in_d_folder:
            assert (check_attr(self, 'path_d_folder')), \
                f'{self.__class__.__name__} is supposed to be stored in d_folder, but attribute not set'
            path_folder: str = self.path_d_folder
        else:
            path_folder: str = self.path_folder

        if (tag is None) and check_attr(self, '_tag'):
            tag: str = self.__getattribute__('_tag')

        # changed in version 1.5.2: include name of the d folder in file name
        #  this allows having multiple d folders in the same i folder
        file_name: str = get_disk_file(
            self,
            path_folder=path_folder,
            tag=tag
        )

        return os.path.join(path_folder, file_name)

    def _pre_load(self):
        pass

    def _post_load(self):
        pass

    def load(self, path_folder: str = None, path_file: str = None, tag: str = None) -> Self:
        """Load maspim pickle object from file."""
        self._pre_load()

        # check file exists
        if path_file is None:
            path_file: str = self.get_save_file(path_folder=path_folder, tag=tag)
        if path_folder is None:
            path_folder = os.path.dirname(path_file)
        if not os.path.exists(path_file):
            file = os.path.basename(path_file)
            raise FileNotFoundError(
                f'found no saved object in folder '
                f'{path_folder} with name '
                f'{file}'
            )

        with open(path_file, 'rb') as f:
            obj: object | dict = pickle.load(f)
            if type(obj) is not dict:  # legacy support
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
                logger.warning(f'discarded attributes {discarded} when loading object from {path_file}')
            # generate new dict, that only has the desired attributes
            obj_new: dict[str, Any] = {key: obj[key] for key in load_attr}
            # merge the objects dict with the disk dict, overwriting
            # instance attributes with saved once, if they both exist
            self.__dict__ |= obj_new
            if self.save_in_d_folder:
                self.path_folder = os.path.dirname(path_folder)
                self.d_folder = os.path.basename(path_folder)
            else:
                self.path_folder = path_folder
            
        logger.info(f'loaded {self.__class__.__name__} with keys {load_attr}')

        self._post_load()
        return self

    def _pre_save(self):
        pass

    def _post_save(self):
        pass

    def save(self, tag: str | None = None):
        """Save class __dict__ instance to file."""
        self._pre_save()

        path_file: str = self.get_save_file(tag=tag)

        # discard all attributes that are not flagged as relevant
        keep_attributes: set[str] = set(self.__dict__.keys())
        if self._save_attrs is not None:
            keep_attributes &= self._save_attrs

        # new dict with only the desired attributes
        save_dict: dict[str, Any] = {key: self.__dict__[key] for key in keep_attributes}

        logger.info(f'saving object with {keep_attributes} to {path_file}')
        with open(path_file, 'wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

        self._post_save()

    def copy(self):
        return deepcopy(self)


class DFolderManager(Convenience):
    d_folder: str = None
    path_folder: str = None

    @classmethod
    def from_path_d_folder(cls, path_d_folder: str):
        assert os.path.isdir(path_d_folder) & os.path.exists(path_d_folder) & path_d_folder.endswith('.d'), \
            f'path_d_folder {path_d_folder} is not a directory'
        path_folder = os.path.dirname(path_d_folder)
        new = cls(path_folder)
        new.set_d_folder(path_d_folder)
        return new

    def set_d_folder(self, path_d_folder: str = None):
        if path_d_folder is not None:
            self.d_folder = os.path.basename(path_d_folder)
            return
        # attempt to find d folder
        d_folders: list[str] = get_d_folder(self.path_folder)
        if len(d_folders) > 1:
            raise ValueError(
                'Multiple d folders found in folder. Please specify the d folder.'
            )
        self.d_folder = d_folders[0]

    @property
    def path_d_folder(self):
        if self.d_folder is None:
            raise AttributeError(f'{self.__class__.__name__} does not have a d_folder attribute, set it with set_d_folder')
        return os.path.join(self.path_folder, self.d_folder)


class MisFileManager(Convenience):
    path_folder = None
    mis_file = None

    @classmethod
    def from_mis_file(cls, path_mis_file: str):
        assert os.path.isfile(path_mis_file) & os.path.exists(path_mis_file) & path_mis_file.endswith('.mis'), \
            f'path_mis_file {path_mis_file} is not a valid file'
        path_folder = os.path.dirname(path_mis_file)
        new = cls(path_folder)
        new.set_mis_file(path_mis_file)
        return new

    def set_mis_file(self, path_mis_file: str = None):
        if path_mis_file is not None:
            self.mis_file: str = os.path.basename(path_mis_file)
            return
        # attempt to set mis file from folder
        mis_files = get_mis_file(self.path_folder)
        if len(mis_files) > 1:
            raise ValueError(
                'Multiple mis files found in folder. Please specify the mis file.'
            )
        self.mis_file: str = mis_files[0]

    @property
    def path_mis_file(self):
        if self.mis_file is None:
            raise AttributeError(f'{self.__class__.__name__} does not have a d_folder attribute, set it with set_mis_file')
        return os.path.join(self.path_folder, self.mis_file)
