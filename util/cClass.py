from typing import Iterable, Callable
import numpy as np


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
    @verbose_function
    def check_attribute_exists(self, attribute: str) -> bool:
        """Returns true if attr exists (even if it is None)."""
        return hasattr(self, attribute)

    @verbose_function
    def check_attributes_exist(self, attributes: list[str]) -> bool:
        if isinstance(attributes, str):
            raise AttributeError('attributes are of type str')
        for attribute in attributes:
            if not self.check_attribute_exists(attribute):
                return False
        return True

    @verbose_function
    def get_attribute(self, attribute: str) -> object | None:
        """Return a class attribute if it exists."""
        if attribute in self.__dict__:
            return self.__dict__[attribute]
        return None

    @verbose_function
    def get_attributes(self, attributes: Iterable[str]) -> list[object | None]:
        """Return a dict of class attributes."""
        values = []
        for attribute in attributes:
            values.append(self.get_attribute(attribute))
        return values

    @verbose_function
    def sget(
            self,
            attributes: Iterable[str] | str,
            set_function: Callable,
            *args,
            overwrite: bool = False,
            **kwargs
    ) -> object | list[object]:
        """Call a set function and return it's output or return attributes."""
        # if attribute is str, wrap it in list
        if flag_notlist := isinstance(attributes, str):
            attributes = [attributes]

        if not overwrite:
            # out may contain Nones
            values = self.get_attributes(attributes)
            # break if any of the values is None and call the set_function
            for val in values:
                if val is None:
                    set_function(*args, overwrite=overwrite, **kwargs)
                    break
        if flag_notlist:
            return self.get_attributes(attributes)[0]
        return self.get_attributes(attributes)

    @verbose_function
    def manage_sget(
            self,
            attributes: Iterable[str] | str,
            function: Callable,
            *args,
            is_get_function=True,
            **kwargs
    ) -> list[object] | object:
        """Check for each attribute, if it exists. If at least one does not
        exist, call function. Return vals of function are expected to be in the
        same order as the attributes."""
        if (flag_notlist := isinstance(attributes, str)):
            attributes = [attributes]
        # not all attributes exist
        if not self.check_attributes_exist(attributes):
            if is_get_function:
                rets = function(*args, **kwargs)
                if flag_notlist:
                    rets = [rets]
            # set function
            else:
                function(*args, **kwargs)
                rets = self.get_attributes(attributes)
        # attributes exist, so get them
        else:
            rets = self.get_attributes(attributes)

        if is_get_function:
            for attribute, ret in zip(attributes, rets):
                self.__setattr__(attribute, ret)
        assert self.check_attributes_exist(attributes), f'functions output \
did not match input attributes. Got {len(attributes)} attribute but \
{len(rets)} values.'
        if flag_notlist:
            return self.get_attribute(attributes[0])
        return self.get_attributes(attributes)

    def get_section_formatted(
            self, section: tuple[int] | None = None
    ) -> tuple[tuple[int], str]:
        """Convert to tuple with (top, bottom) in cm as ints."""
        if section is None:
            section = self._section

        if isinstance(section, str):
            d1, d2 = section.split('-')
            section = (d1, d2)
        if isinstance(section, tuple):
            d1 = int(section[0])
            d2 = int(section[1])
        # make sure d2 > d1
        if d1 > d2:
            d = d2
            d2 = d1
            d1 = d
        return (d1, d2), f'{d1}-{d2}'

    def get_closest_mz(
            self,
            mz: float | str,
            cols: Iterable | None = None,
            max_deviation: float | None = None,
            return_deviation: bool = False
    ) -> str | tuple[str, float]:
        """
        Return the closest mz value in the msi data.

        Parameters
        ----------
        mz : mz value as float or string of one compound for which the mz value
            within the data is not known
        cols: Iterable | None, optional
            The columns in which to search the closest value. If cols is None,
            tries to get cols of current_feature_table or feature_table_zone_averages.
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
            if self.check_attribute_exists('current_feature_table'):
                cols = np.array(self.current_feature_table.columns).astype(str)
            elif self.check_attribute_exists('feature_table_zone_averages'):
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
