import numpy as np

def kartesian_to_polar(v_x, v_y=None):
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
    v_x: np.ndarray[float] = v_r * np.cos(v_phi)
    v_y: np.ndarray[float] = v_r * np.sin(v_phi)
    return v_x, v_y


def rescale_values(
        a: np.ndarray,
        new_min: float, new_max: float,
        old_min: float | None = None, old_max: float | None = None,
        axis: int | None = None
) -> np.ndarray:
    """Rescale values to specified range. Ignores nans."""
    assert new_max > new_min, 'max has to be bigger than min'
    if old_max is None:
        old_max = np.nanmax(a, axis=axis)
    if old_min is None:
        old_min = np.nanmin(a, axis=axis)
    if np.any(old_min == old_max, axis=axis):
        print('[W] found same min and max in rescale_values')
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
