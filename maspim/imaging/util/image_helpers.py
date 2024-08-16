"""Helper functions."""
import numpy as np
import pandas as pd
import cv2
import logging

from typing import Iterable
from scipy.spatial import KDTree
from tqdm import tqdm

from maspim.imaging.util.image_convert_types import ensure_image_is_gray
from maspim.imaging.util.image_plotting import plt_cv2_image
from maspim.imaging.util.image_processing import threshold_background_as_min
from maspim.imaging.util.coordinate_transformations import rescale_values

logger = logging.getLogger(__name__)


def ensure_odd(x: int) -> int:
    """
    Add 1 if x is even.

    Parameters
    ----------
    x : int
        possibly even int.

    Returns
    -------
    x : int
        odd int.

    """
    if not x % 2:
        x += 1
    return x


def get_half_width_padded(image: np.ndarray) -> np.ndarray:
    """ Zeropad the image by half the height to left and right."""
    width: int = image.shape[0]
    half_width: int = (width + 1) // 2
    image_padded: np.ndarray = np.pad(
        image,
        ((0, 0), (half_width, half_width + 1))
    )
    return image_padded


def first_nonzero(
        arr: np.ndarray, axis: int, invalid_val: float | int = -1
) -> np.ndarray[int]:
    """
    Return indices of first nonzero entries in an array along an axis

    Parameters
    ----------
    arr : np.ndarray
        Input array
    axis : int
        Axis along which to look
    invalid_val : int, flaot, optional
        Fill value for invalid rows/ columns. The default is -1.

    Returns
    -------
    An array of indices
    """
    # https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
    mask: np.ndarray[bool] = arr != 0
    return np.where(
        mask.any(axis=axis),
        mask.argmax(axis=axis),
        invalid_val
    )


def last_nonzero(arr, axis, invalid_val=-1):
    """
    Return indices of last nonzero entries in an array along an axis

    Parameters
    ----------
    arr : np.ndarray
        Input array
    axis : int
        Axis along which to look
    invalid_val : int, flaot, optional
        Fill value for invalid rows/ columns. The default is -1.

    Returns
    -------
    An array of indices
    """
    mask: np.ndarray[bool] = arr != 0
    val: int = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(
        mask.any(axis=axis),
        val,
        invalid_val
    )


def min_max_extent_layer(mask_layer: np.ndarray) -> np.ndarray[int]:
    """Return mask that restricts mask to extent of sample."""
    hor_min: np.ndarray[int] = first_nonzero(mask_layer, 0, mask_layer.shape[0])
    hor_max: np.ndarray[int] = last_nonzero(mask_layer, 0, -1)
    # which columns in mask contain not only zeros
    col_valid: np.ndarray[bool] = (hor_min < mask_layer.shape[0]) & (hor_max > -1)
    mask_extent: np.ndarray = np.zeros_like(mask_layer)
    # iterate over cols
    for j in range(mask_layer.shape[1]):
        if col_valid[j]:
            slice_valid = np.index_exp[hor_min[j]:hor_max[j] + 1, j]
            mask_extent[slice_valid] = 1
    return mask_extent


def filter_contours_by_size(
        contours: Iterable[np.ndarray[int]],
        image_shape: tuple[int, ...],
        threshold_size: float=.1) -> list[np.ndarray[int]]:
    """
    Filter out contours that are too small.

    To be significant contours have to be larger than 10 % of width or
    height of image.

    Parameters
    ----------
    contours : Iterable[np.ndarray[int]]
        The contours to be filtered.
    image_shape: tuple[int, ...]
        Shape of the image.
    threshold_size: float
        Minimum extent requireed in either x or y direction.

    Returns
    -------
    list[np.ndarray[int]]
        List of filtered contours.
    """
    height, width = image_shape[:2]
    contours_filtered_size: list[np.ndarray[int]] = []
    for contour in contours:
        # get size of area encircled by contour
        xy: np.ndarray[int] = contour[:, 0, :]
        x: np.ndarray[int] = xy[:, 0]
        y: np.ndarray[int] = xy[:, 0]
        dx: int = np.max(x) - np.min(x)
        dy: int = np.max(y) - np.min(y)

        if (dy > threshold_size * height) or (dx > threshold_size * width):
            contours_filtered_size.append(contour)
    return contours_filtered_size


def find_points_in_grid(
        grid_points: np.ndarray,
        points: np.ndarray,
        **kwargs
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    """
    Find indices of points inside a grid using a kdtree.

    Make sure no nan-values are present.

    Parameters
    ----------
    grid_points: np.ndarray
        Points on a grid
    points: np.ndarray
        Points to query. Do not have to be part of the grid.
    kwargs: Additional keyword arguments for the KDTree query method.

    Returns
    -------
    distances: np.ndarray[float]
        Distances of points to the closest points in the grid.
    idxs: np.ndarray[int]
        Indices of grid-points.
    """
    assert ~np.any(np.isnan(grid_points)), 'grid points contain nan'
    assert ~np.any(np.isnan(points)), 'points contain nan'
    # for each pixel in the target image, find the corresponding pixels
    # in the from_image

    # initiate the kdtree
    kdtree: KDTree = KDTree(grid_points)

    # search the nearest neighbours
    distances, idxs = kdtree.query(points, **kwargs)
    # idxs --> index of closest neighbour in grid_points
    return distances, idxs


def exclude_missing_pixels_in_feature_table(ft: pd.DataFrame) -> np.ndarray[bool]:
    """
    Find 2D mask for pixels in feature table that are missing.

    Find the area spanned by values in the x- and y-columns and return a 2D mask
    where missing pixels are set to False.

    Parameters
    ----------
    ft: pd.DataFrame
        Input table

    Returns
    -------
    mask: np.ndarray[bool]
        Array where missing pixels are set to False.
    """
    assert 'x' in ft.columns, 'ft must have x column'
    assert 'y' in ft.columns, 'ft must have y column'
    # get _extent of feature table
    x_min_FT: int = ft.x.min()
    x_max_FT: int = ft.x.max()
    y_min_FT: int = ft.y.min()
    y_max_FT: int = ft.y.max()

    X_ft, Y_ft = np.meshgrid(
        np.arange(x_min_FT, x_max_FT + 1, 1, dtype=int),
        np.arange(y_min_FT, y_max_FT + 1, 1, dtype=int)
    )
    # possible points
    p_grid: np.ndarray[int] = np.c_[X_ft.ravel(), Y_ft.ravel()]
    # actual points
    p_ft: np.ndarray[int] = np.c_[ft.x, ft.y]
    # check for each grid point if it is in the feature table
    distances, idxs = find_points_in_grid(p_ft, p_grid)
    # all idxs with distances > 0 are not in the feature table
    mask_in_ft: np.ndarray[bool] = (distances == 0).reshape(X_ft.shape)
    return mask_in_ft

def get_foreground_pixels_and_threshold(
        image: np.ndarray,
        obj_color: str,
        method: str = 'otsu'
) -> tuple[np.ndarray, float | int]:
    """
    Binarize an image into fore- and background pixels.

    Parameters
    ----------
    image:  np.ndarray
        image to be binarized, will be converted to grayscale automatically
    obj_color: str
        'light' if region of interest is lighter than rest, 'dark' otherwise
    method: str, optional.
        method to be used. Options are 'local-min' and 'otsu'. Default is otsu.

    Returns
    -------
    mask, threshold
    """
    methods: tuple[str, ...] = ('otsu', 'local-min')
    obj_colors: tuple[str, str] = ('light', 'dark')
    assert method in methods, f'Method {method} not {methods}'
    assert obj_color in obj_colors, f'Color {obj_color} not {obj_colors}'

    image_grayscale: np.ndarray = ensure_image_is_gray(image)
    image_grayscale: np.ndarray[np.uint8] = rescale_values(
        image_grayscale, 0, 255
    ).astype(np.uint8)

    # define the threshold type depending on the object color
    if obj_color == 'dark':
        thr = cv2.THRESH_BINARY_INV
    else:
        thr = cv2.THRESH_BINARY

    # create binary image with the specified method
    if method == 'otsu':
        logger.info('Determining threshold for background intensity with OTSU.')
        thr_background, mask_foreground = cv2.threshold(
            image_grayscale, 0, 1, thr + cv2.THRESH_OTSU)
    elif method == 'local_min':
        logger.info('Determining threshold for background intensity with local-min.')
        thr_background = threshold_background_as_min(image_grayscale)
        _, mask_foreground = cv2.threshold(
            image_grayscale, thr_background, 1, thr)
    else:
        raise KeyError(f'{method=} is not a valid option. Choose one of\
    "otsu", "local_min".')
    return mask_foreground, thr_background

def get_simplified_image(
        image: np.ndarray,
        factors: Iterable[int] | None = None,
        plts: bool = False
) -> np.ndarray[np.uint8]:
    """
    Return a simplified version of the input binary image.

    Simplification is achieved by stepping up the size of a median blur
    filter. Filter size is limited to 255

    Parameters
    ----------
    image: np.ndarray
        image to be simplified. If image is not binary, obj_color is expected to be passed.
    factors : Iterable[int], optional
        factors to be used for the median filter. Values are clipped to 255.
        If not provided, factors are powers of 2 from 10 to 5.
    plts: bool, optional

    Returns
    -------
    image_binary : np.ndarray
        Simplified image.

    """
    assert (len(np.unique(image)) <= 2), \
        'provide a binary image'
    image_binary: np.ndarray = ensure_image_is_gray(image)

    if factors is None:
        factors: list[int] = [1024, 512, 256, 128, 64, 32]
    for factor in factors:
        # smooth edge
        # limit kernel size to 255 as larger kernel sizes are not supported
        kernel_size: int = np.min([int(np.max(image_binary.shape) // factor), 255])
        if not kernel_size % 2:
            kernel_size += 1
        image_binary: np.ndarray[np.uint8] = cv2.medianBlur(
            image_binary, kernel_size).astype(np.uint8)
        if plts:
            plt_cv2_image(image_binary, f'{factor=}, {kernel_size=}')

    return image_binary


def restore_unique_values(
        image: np.ndarray[int], unique_values: Iterable[int]
) -> np.ndarray[int]:
    """
    E.g. warping interpolates intensities. For classification images this is
    not desired. This function restores the original unique values by rounding
    to the nearest unique value.
    """
    image_new = np.zeros_like(image)
    unique_values: np.ndarray[int] = np.sort(unique_values)
    n_unique: int = len(unique_values)

    for i, uval in tqdm(enumerate(unique_values), total=n_unique, desc='setting unique values'):
        left_uval: int = unique_values[i - 1] if i > 0 else -np.infty
        right_uval: int = unique_values[i + 1] if i < n_unique - 1 else np.infty
        left_bound: float = (left_uval + uval) / 2
        right_bound: float = (uval + right_uval) / 2
        mask = (image >= left_bound) & (image < right_bound)
        image_new[mask] = uval
    return image_new


if __name__ == '__main__':
    pass
