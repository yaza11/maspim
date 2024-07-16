import re
from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache
def get_da_export_data(
        file_path: str
) -> tuple[np.ndarray[str], list[np.ndarray[float]], list[np.ndarray[float]], list[np.ndarray[float]]]:
    """Read data from txt file and return pixels and spectra."""
    spectra_mzs: list[np.ndarray[float]] = []
    spectra_intensities: list[np.ndarray[float]] = []
    spectra_SNR: list[np.ndarray[float]] = []
    pixel_names: list[str] = []
    with open(re.findall(r'(?:file:///)?(.+)', file_path)[0]) as f:
        # first line is number of pixels
        for i, line in enumerate(f):
            # skip first line (only contains information about the number of pixels)
            if i == 0:
                continue
            # values in line are separated by semicolons
            pixel_entries: list[str] = line.replace(',', '.').split(';')
            pixel_name, n_mzs = pixel_entries[:2]
            # making use of start:stop:step notation
            mzs: list[str] = pixel_entries[2::3]
            intensities: list[str] = pixel_entries[3::3]
            signal_to_noise_ratios: list[str] = pixel_entries[4::3]

            pixel_names.append(pixel_name)
            spectra_mzs.append(np.array(mzs, dtype=float))
            spectra_intensities.append(np.array(intensities, dtype=float))
            spectra_SNR.append(np.array(signal_to_noise_ratios, dtype=float))
    return np.array(pixel_names), spectra_mzs, spectra_intensities, spectra_SNR


def get_da_export_ion_image(
        mz: str | float,
        pixel_names: np.ndarray[str],
        spectra_mzs: list[np.ndarray[float]],
        data: list[np.ndarray[float]],
        norm_spectra: bool = False,
        tolerance: float = 3e-3,
        **_
) -> pd.DataFrame:
    """
    Given mz and data, create dataframe object.

    Parameters
    ----------
    mz : str | float
        The mass to plot.
    pixel_names : list[str]
        list of pixel names.
    spectra_mzs : list[float]
        list of masses in each spectum.
    data : list[float]
        list of intensities or SNRs for each pixel.
    norm_spectra : bool, optional
        If true, each spectrum will be scaled to its median. The default is False.
    tolerance : float, optional
        Width of the filter function. For The default is 0.003.

    Returns
    -------
    FT : pd.DataFrame
        The ion image inside a feature table.

    """
    mz: float = float(mz)
    # if kernel_mode is gauss, width defines 3 * sigma
    # otherwise width is the width of the window
    N_pixels: int = len(pixel_names)

    img_x: np.ndarray[int] = np.empty(N_pixels, dtype=int)
    img_y: np.ndarray[int] = np.empty(N_pixels, dtype=int)

    ion_image: np.ndarray[float] = np.zeros(N_pixels, dtype=float)
    # iterate over pixels (=lines in txt file)
    for idx_pixel, (pixel, mzs, intensities) in enumerate(zip(
            pixel_names, spectra_mzs, data
    )):
        # set x and y values
        img_x[idx_pixel] = int(re.findall('X(.*)Y', pixel)[0])  # x coordinate
        img_y[idx_pixel] = int(re.findall('Y(.*)', pixel)[0])  # y coordinate

        if len(intensities) == 0:  # empty, skip to next spectrum
            continue
        if norm_spectra:  # set median = 1
            m: float = np.median(intensities[intensities > 0])
            intensities /= m

        mask: np.ndarray[bool] = (mzs > mz - tolerance) & (mzs < mz + tolerance)
        if not np.any(mask):  # no peak within tolerance
            continue
        ion_image[idx_pixel] = np.max(intensities[mask])

    FT: pd.DataFrame = pd.DataFrame(data=ion_image, columns=[mz])
    FT['x'] = img_x
    FT['y'] = img_y

    return FT
