import numpy as np
import pandas as pd
import cv2
import skimage
import matplotlib.pyplot as plt
import skimage
import scipy
from scipy.spatial import KDTree


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


def get_half_width_padded(image):
    """ Zeropad the image by half the height to left and right."""
    width = image.shape[0]
    half_width = (width + 1) // 2
    image_padded = np.pad(image, ((0, 0), (half_width, half_width)))
    return image_padded


# https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def min_max_extent_layer(mask_layer):
    """Return mask that restricts mask to extent of probe."""
    hor_min = first_nonzero(mask_layer, 0, mask_layer.shape[0])
    hor_max = last_nonzero(mask_layer, 0, -1)
    # which columns in mask contain not only zeros
    col_valid = (hor_min < mask_layer.shape[0]) & (hor_max > -1)
    mask_extent = np.zeros_like(mask_layer)
    # iterate over cols
    for j in range(mask_layer.shape[1]):
        if col_valid[j]:
            slice_valid = np.index_exp[hor_min[j]:hor_max[j] + 1, j]
            mask_extent[slice_valid] = 1
    return mask_extent


def filter_contours_by_size(contours, image_shape, threshold_size=.1):
    # filter out contours that are too small to be any significant area
    # have to be larger than 10 % of width or height of image
    height, width = image_shape[:2]
    contours_filtered_size = []
    for contour in contours:
        # get size of area encircled by contour
        xy = contour[:, 0, :]
        x = xy[:, 0]
        y = xy[:, 0]
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)

        if (dy > threshold_size * height) or (dx > threshold_size * width):
            contours_filtered_size.append(contour)
    return contours_filtered_size


def find_points_in_grid(grid_points: np.ndarray, points: np.ndarray, **kwargs):
    assert ~np.any(np.isnan(grid_points)), 'grid points contain nan'
    assert ~np.any(np.isnan(points)), 'points contain nan'
    # for each pixel in the target image, find the corresponding pixels
    # in the from_image

    # initiate the kdtree
    kdtree = KDTree(grid_points)

    # search the nearest neighbours
    distances, idxs = kdtree.query(points, **kwargs)
    # idxs --> index of closest neighbour in grid_points
    return distances, idxs


def exclude_missing_pixels_in_feature_table(ft: pd.DataFrame):
    """Find 2D mask for pixels in feature table."""
    assert 'x' in ft.columns, 'ft must have x column'
    assert 'y' in ft.columns, 'ft must have y column'
    # get extent of feature table
    x_min_FT = ft.x.min()
    x_max_FT = ft.x.max()
    y_min_FT = ft.y.min()
    y_max_FT = ft.y.max()

    X_ft, Y_ft = np.meshgrid(
        np.arange(x_min_FT, x_max_FT + 1, 1, dtype=int),
        np.arange(y_min_FT, y_max_FT + 1, 1, dtype=int)
    )
    # possible points
    p_grid = np.c_[X_ft.ravel(), Y_ft.ravel()]
    # actual points
    p_ft = np.c_[ft.x, ft.y]
    # check for each grid point if it is in the feature table
    distances, idxs = find_points_in_grid(p_ft, p_grid)
    # all idxs with distances > 0 are not in the feature table
    mask_in_ft = (distances == 0).reshape(X_ft.shape)
    return mask_in_ft


if __name__ == '__main__':
    pass
