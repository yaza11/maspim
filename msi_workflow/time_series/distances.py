"""Different distance measures"""
from typing import Iterable

import numpy as np
import pandas as pd


def sign_corr(a: Iterable, b: Iterable) -> float:
    """
    Return the fraction of like signs over overall signs.

    The value is normed to be between -1 and 1 where 1 is returned, if all
    signs match and -1 if all signs oppose. Nans will be ignored.
    a and b must have the same length.

    Parameters
    ----------
    a : Iterable
        DESCRIPTION.
    b : Iterable
        DESCRIPTION.

    Returns
    -------
    float
        number of same signs over number of entries.

    """
    assert len(a) == len(b), 'a and b must have same length'
    r = np.nanmean(np.sign(a) == np.sign(b)) * 2 - 1
    return r


def sign_weighted_corr(
        a: np.ndarray | pd.Series,
        b: np.ndarray | pd.Series,
        w: np.ndarray | pd.Series
) -> float:
    """
    Return the fraction of like signs over overall signs.

    The value is normed to be between -1 and 1 where 1 is returned, if all
    signs match and -1 if all signs oppose. Nans will be ignored.
    a and b must have the same length.

    Parameters
    ----------
    a : First vector
    b : Second vector
    w : Weights

    Returns
    -------
    float
        number of same signs over number of entries weighted.

    """
    assert len(a) == len(b), 'a and b must have same length'
    assert len(a) == len(w), 'weights must have same length as a and b'
    assert np.min(w) >= 0, 'weights should be bigger than 0'

    r = np.sum((np.sign(a) == np.sign(b)) * w) / len(w) * 2 - 1
    return r
