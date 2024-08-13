"""Functions for converting coordinates."""
import warnings
import numpy as np
import pandas as pd


def kartesian_to_polar(v_x, v_y=None) -> tuple:
    """
    Convert vectors or point of x, y to r, phi.

    Parameters
    ----------
    v_x : array of floats 1D or tuple of (float: x, float: y)
        Either x-vector or (x, y) tuple.
    v_y : array of floats 1D or None, optional
        If v_x is not a tuple, v_y will be expected to hold an y-vector with 
        same length as the x-vector. The default is None.

    Returns
    -------
    v_r : 1D array of floats or float
        The radii.
    v_phi : 1D array of floats or float
        The angles.

    """
    if (v_y is None) and (len(v_x) == 2):
        v_x, v_y = v_x
    # polar
    v_r = np.sqrt(v_x ** 2 + v_y ** 2)
    v_phi = np.arctan2(v_y, v_x)
    return v_r, v_phi


def polar_to_kartesian(
        v_r: np.ndarray[float], v_phi: np.ndarray[float]
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Convert r and phi values back to kartesian coordinates.

    v_r and v_phi must have the same shape.

    v_r: array-like
        The radii.
    v_phi: array-like
        The angles.

    Returns
    -------
    v_x : array-like
        The x-coordinates
    v_y : array-like
        The y-coordinates
    """
    v_x: np.ndarray[float] = v_r * np.cos(v_phi)
    v_y: np.ndarray[float] = v_r * np.sin(v_phi)
    return v_x, v_y


def rescale_values(
        a: np.ndarray | pd.DataFrame,
        new_min: float,
        new_max: float,
        old_min: float | None = None,
        old_max: float | None = None,
        axis: int | None = None
) -> np.ndarray | pd.DataFrame:
    """
    Rescale values to specified range. Ignores nans.

    Parameters
    ----------
    a : np.ndarray | pd.DataFrame
        Array to be rescaled.
    new_min: float
        New lowest min value
    new_max: float
        New lowest max value
    old_min: float, optional
        The old lowest min. Useful if not the entire possible value range is
        populated. If not provided, will be inferred from a
    old_max: float, optional
        The old lowest max. Useful if not the entire possible value range is
        populated. If not provided, will be inferred from a
    axis: int, optional
        Axis along which to rescale. Defaults to all.
    """
    assert np.all(new_max >= new_min), 'max has to be bigger than min'
    if np.any(new_min == new_max, axis=axis):
        warnings.warn('found same min and max in rescaled values')

    if old_max is None:
        old_max = np.nanmax(a, axis=axis)
    if old_min is None:
        old_min = np.nanmin(a, axis=axis)
    if np.any(old_min == old_max, axis=axis):
        warnings.warn('found same min and max in  input values')
    return (a - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = np.random.random((100, 100)) * 100
    #  a[a < 10] = np.nan
    b = rescale_values(a, -.5, 2, axis=0)

    print(np.nanmin(a), np.nanmax(a))
    print(np.nanmin(b), np.nanmax(b))

    plt.imshow(a)
    plt.show()
    plt.imshow(b)
    plt.show()
