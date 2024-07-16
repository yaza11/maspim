"""Pure python implementation of the hog algorithm"""
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable
from skimage.feature._hoghistogram import hog_histograms


def hog_gradient(image):
    # gradient in row-direction (y)
    g_row = np.empty(image.shape, dtype=image.dtype)
    # 0 at boundaries
    g_row[0, :] = 0
    g_row[-1, :] = 0
    # f(x + 2) - f(x)
    g_row[1:-1, :] = image[2:, :] - image[:-2, :]
    
    g_col = np.empty(image.shape, dtype=image.dtype)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]

    return g_row, g_col


def hog_normalize_block(block, method, eps=1e-5):
    # normalize values in block with specified method
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
        # limit values to 0.2
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out**2) + eps**2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out


def slow_cell_hog(
    magnitude,
    orientation,
    orientation_start,
    orientation_end,
    cell_columns,
    cell_rows,
    column_index, 
    row_index,
    size_columns,
    size_rows,
    range_rows_start,
    range_rows_stop,
    range_columns_start, 
    range_columns_stop
):
    """Calculation of the cell's HOG value

    Parameters
    ----------
    magnitude : ndarray
        The gradient magnitudes of the pixels.
    orientation : ndarray
        Lookup table for orientations.
    orientation_start : float
        Orientation range start.
    orientation_end : float
        Orientation range end.
    cell_columns : int
        Pixels per cell (rows).
    cell_rows : int
        Pixels per cell (columns).
    column_index : int
        Block column index.
    row_index : int
        Block row index.
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    range_rows_start : int
        Start row of cell.
    range_rows_stop : int
        Stop row of cell.
    range_columns_start : int
        Start column of cell.
    range_columns_stop : int
        Stop column of cell

    Returns
    -------
    total : float
        The total HOG value.
    """
    total = 0.

    # add up magnitudes of cell
    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]

    # average
    return total / (cell_rows * cell_columns)


def slow_hog_histograms(
        gradient_columns,
        gradient_rows,
        cell_columns, 
        cell_rows,
        size_columns,
        size_rows,
        number_of_cells_columns,
        number_of_cells_rows,
        number_of_orientations,
        orientation_histogram
):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Parameters
    ----------
    gradient_columns : ndarray
        First order image gradients (rows).
    gradient_rows : ndarray
        First order image gradients (columns).
    cell_columns : int
        Pixels per cell (rows).
    cell_rows : int
        Pixels per cell (columns).
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    number_of_cells_columns : int
        Number of cells (rows).
    number_of_cells_rows : int
        Number of cells (columns).
    number_of_orientations : int
        Number of orientation bins.
    orientation_histogram : ndarray
        The histogram array which is modified in place.
    """
    magnitude = np.hypot(gradient_columns, gradient_rows)
    orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180

    r_0 = cell_rows / 2  # center pixel in cell
    c_0 = cell_columns / 2  # center pixel in cell
    cc = cell_rows * number_of_cells_rows  # total number of pixels in row-direction
    cr = cell_columns * number_of_cells_columns  # total number of pixels in column-direction
    # sum from center - 1 / 2 to center + 1 / 2 
    range_rows_stop = (cell_rows + 1) / 2
    range_rows_start = -(cell_rows / 2)
    range_columns_stop = (cell_columns + 1) / 2
    range_columns_start = -(cell_columns / 2)
    number_of_orientations_per_180 = 180. / number_of_orientations

    # compute orientations integral images
    # fill orientation histogram angle by angle
    for i in range(number_of_orientations):
        # isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        r = r_0
        r_i = 0
        # orientation histogram for every cell
        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                orientation_histogram[r_i, c_i, i] = \
                    slow_cell_hog(
                        magnitude, orientation,
                        orientation_start, orientation_end,
                        cell_columns, cell_rows, c, r,
                        size_columns, size_rows,
                        range_rows_start, range_rows_stop,
                        range_columns_start, range_columns_stop
                    )
                c_i += 1
                c += cell_columns

            r_i += 1
            r += cell_rows



def hog(
        image, 
        n_cells: int | tuple[int, int], 
        cells_per_block: int | tuple[int, int], 
        n_orientations: int, 
        block_norm='L2-Hys'
):
    if not hasattr(n_cells, '__iter__'):
        n_cells = (n_cells) * 2
    if not hasattr(cells_per_block, '__iter__'):
        cells_per_block = (cells_per_block) * 2
    n_cells_row, n_cells_col = n_cells

    s_row, s_col = image.shape[:2]  # pixels in row- and column-direction
    c_row, c_col = s_row // n_cells_row, s_col // n_cells_col  # pixels per cell
    b_row, b_col = cells_per_block

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    if n_blocks_col <= 0 or n_blocks_row <= 0:
        min_row = b_row * c_row
        min_col = b_col * c_col
        raise ValueError(
            'The input image is too small given the values of '
            'pixels_per_cell and cells_per_block. '
            'It should have at least: '
            f'{min_row} rows and {min_col} cols.'
        )

    g_row, g_col = hog_gradient(image)
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)

    # compute orientations integral images
    orientation_histogram = np.zeros(
        (n_cells_row, n_cells_col, n_orientations), dtype=float
    )

    # 3D array with shape n_cells_col, n_cells_row, n_orientations 
    # functionally the same as slow_hog_histograms, but faster implementation
    hog_histograms(
        g_col,
        g_row,
        c_col,
        c_row,
        s_col,
        s_row,
        n_cells_col,
        n_cells_row,
        n_orientations,
        orientation_histogram
    )

    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, n_orientations)
    )

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r : r + b_row, c : c + b_col, :]
            normalized_blocks[r, c, :] = hog_normalize_block(block, method=block_norm)

    angles = np.pi * (np.arange(n_orientations) + 0.5) / n_orientations

    return orientation_histogram, normalized_blocks, angles


if __name__ == '__main__':
    from PIL import Image
    img = np.array(Image.open('C:/Users/Yannick Zander/Downloads/2020_03_23_Cariaco_535-540cm_ROI.bmp').convert('L'))

    n_cells_col = 10
    n_cells_row = int(img.shape[1] / img.shape[0] * n_cells_col)
    n_cells = (n_cells_col, n_cells_row)
    n_pixels_per_cell_col = img.shape[0] / n_cells_col
    n_pixels_per_cell_row = img.shape[1] / n_cells_row

    g_row, g_col = hog_gradient(img)

    plt.imshow(g_row)
    plt.show()
    plt.imshow(g_col)
    plt.show()
    hist, blocks, angles = hog(
        img, n_cells=n_cells, cells_per_block=(1, 1), n_orientations=18
    )
    # hist_ = blocks[:, :, 0, 0, :]

    angles_array = angles[hist.argmax(axis=-1)]
    plt.imshow(angles_array)

    u = np.cos(angles_array)
    v = np.sin(angles_array)

    y, x = np.meshgrid(
        np.linspace(
            n_pixels_per_cell_col / 2, 
            img.shape[0] - n_pixels_per_cell_col / 2,
            n_cells_col
        ), 
        np.linspace(
            n_pixels_per_cell_row / 2, 
            img.shape[1] - n_pixels_per_cell_row / 2,
            n_cells_row
        )
    )

    plt.imshow(img)
    plt.quiver(x, y, u, v, angles='xy')
