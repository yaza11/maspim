from imaging.util.Image_convert_types import convert, ensure_image_is_gray, infere_mode

from typing import Callable

import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import skimage
import scipy
from scipy import interpolate


def adaptive_mean_with_mask(
        src, maxValue, adaptiveMethod, thresholdType, blockSize, C,
        mask=None):
    """
    Apply adaptive mean on image with mask.

    This function is designed to resemble the cv2 adpativeMeanThreshold
    function and has the same call signature but additionally accepts a mask.

    Parameters
    ----------
    src : array[uint8]
        The image to apply the mask to.
    maxValue : uint8
        The value asigned to pixels above the threshold. Usually 255.
    adaptiveMethod : int
        The method to use. Currently the only option is
        cv2.ADAPTIVE_THRESH_MEAN_C.
    thresholdType : int
        Type of threshold for the binarisation. Options are cv2.THRESH_BINARY
        and cv2.THRESH_BINARY_INV.
    blockSize : odd int
        size of the kernel.
    C : float
        Value to be added before determening the threshold. I cannot think of
        a case where you would not want to use 0.
    mask : array[bool], optional
        The mask to use with the same width and height as src.
        The default is None.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    dst : array[bool]
        Same shape as src. The classified values for each pixel. If pixel is
        lighter than surroundings, the asigned value will be maxValue.

    """
    if adaptiveMethod != cv2.ADAPTIVE_THRESH_MEAN_C:
        raise NotImplementedError()

    # initiate arrays
    dst = np.zeros_like(src)

    if mask is None:
        mask = np.ones_like(src)
    # create extended array with adequate boundary condition
    # by how many pixels the src will be extended
    extend_by = (blockSize - 1) // 2
    src_extended = cv2.copyMakeBorder(
        src, top=extend_by, bottom=extend_by, left=extend_by,
        right=extend_by,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED)
    mask_extended = cv2.copyMakeBorder(
        mask, top=extend_by, bottom=extend_by, left=extend_by,
        right=extend_by,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED)

    footprint = np.ones((blockSize, blockSize))
    mean_pixel_values = skimage.filters.rank.mean(
        image=src_extended, footprint=footprint, mask=mask_extended)
    mean_pixel_values = mean_pixel_values[
        extend_by:-extend_by, extend_by:-extend_by]

    if thresholdType == cv2.THRESH_BINARY:
        dst[src - C > mean_pixel_values] = maxValue
    elif thresholdType == cv2.THRESH_BINARY_INV:
        dst[src - C < mean_pixel_values] = maxValue

    return dst


# https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image
def filter_with_mask_by_rescaling_weights(
        image: np.ndarray[float],
        mask_nonholes: np.ndarray[float],
        filter_function: Callable,
        set_mask_pixels_zero: bool = False,
        **kwargs
) -> np.ndarray:
    image_filtered = filter_function(image * mask_nonholes, **kwargs)

    weights = filter_function(mask_nonholes, **kwargs)

    # normalize smoothed by weights
    mask_valid_weights = weights > 0

    image_filtered[mask_valid_weights] /= weights[mask_valid_weights]
    # set data in invalid pixels to 0
    if set_mask_pixels_zero:
        image_filtered *= mask_nonholes
    return image_filtered


def adaptive_mean_with_mask_by_rescaling(
        image: np.ndarray,
        mask_nonholes: np.ndarray,
        ksize: int,
        thresholdType: int,
        maxValue: np.uint8,
        C: np.uint8 = 0,
        **kwargs
) -> np.ndarray[np.uint8]:
    """Apply adaptive mean to image with mask by rescaling mask weights."""
    assert len(image.shape) == 2, 'image has to be grayscale'
    assert (mask_nonholes.max() <= 1) and (mask_nonholes.min() >= 0), \
        'mask should hold values between 0 and 1'

    extend_by = (ksize[0] - 1) // 2
    image_extended = cv2.copyMakeBorder(
        image, top=extend_by, bottom=extend_by, left=extend_by, right=extend_by,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED)
    mask_extended = cv2.copyMakeBorder(
        mask_nonholes, top=extend_by, bottom=extend_by, left=extend_by, right=extend_by,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED)

    slice_nonborder = np.index_exp[extend_by:-extend_by, extend_by:-extend_by]

    image_mean = filter_with_mask_by_rescaling_weights(
        image=image_extended.astype(np.float32),
        mask_nonholes=mask_extended.astype(np.float32),
        filter_function=cv2.blur,
        ksize=ksize,
        **kwargs
    )[slice_nonborder]
    # set pixels above mean to light
    image_light = np.zeros((image.shape), dtype=bool)
    if thresholdType == cv2.THRESH_BINARY:
        image_light[image - C > image_mean] = maxValue
    elif thresholdType == cv2.THRESH_BINARY_INV:
        image_light[image - C < image_mean] = maxValue
    return image_light


def remove_outliers_by_median(image, kernel_size_median=7,
                              threshold_replace_median=10):
    """
    Remove outliers with median filter inplace.

    Replace pixels within given kernel region that deviate more than
    specified threshold from median by median.

    Parameters
    ----------
    image : array, optional
        The image on which to perform the denoising. The default is None.
    kernel_size_median : odd int, optional
        quadratic kernel. The default is 7.
    threshold_replace_median : int, optional
        How much the pixel is allowed to deviate from the median for 8bit
        image. The default is 10.

    Returns
    -------
    image_denoised : array
        denoised version of image. Function acts inplace

    """
    # create median
    image_median = cv2.medianBlur(image, kernel_size_median)
    # replace noisy pixels by mean
    image_denoised = image

    if infere_mode(image) == 'L':
        mask = np.abs(image - image_median
                      ) > threshold_replace_median
        image_denoised[mask] = image_median[mask]
    else:
        for d in range(3):
            mask = np.abs(image[:, :, d] - image_median[:, :, d]
                          ) > threshold_replace_median
            image_denoised[:, :, d][mask] = image_median[:, :, d][mask]
    return image_denoised


def threshold_background_as_min(image_gray, plts=False):
    """
    Find threshold for binarisation as local minimum.

    For a given grayscale image, find the local minimum in the histogram
    for a bimodal distribution.
    The local minimum between the two peaks will be the threshold.

    Parameters
    ----------
    image_gray : TYPE
        DESCRIPTION.
    plts : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    thr : int
        DESCRIPTION.

    """
    image_gray = ensure_image_is_gray(image_gray)

    # transform to vector
    v = image_gray.ravel()
    values, bins, _ = plt.hist(v, bins=256, range=(0, 256))
    # smooth the values
    values_smooth = scipy.ndimage.gaussian_filter1d(
        values, sigma=10, mode='constant', cval=0)
    # find local minima
    idx_mins = scipy.signal.argrelmin(values_smooth)[0]
    if len(idx_mins) == 0:
        return None
    # take the lowest between the two maxs
    thr = idx_mins[np.argmin(values_smooth[idx_mins])]
    if plts:
        plt.figure()
        plt.plot(values, label='distribution of intensities')
        plt.plot(values_smooth, label='distribution of smoothed intensities')
        for idx_min in idx_mins:
            plt.axvline(idx_min, linestyle='--')
        plt.axvline(thr, color='k', label='determined threshold')
        plt.legend()
        plt.xlabel('pixel intensity')
        plt.ylabel('counts')
        plt.title('Determination of background threshold')
    return thr


def func_on_image_with_mask(
        image, mask, func, *args, return_argument_idx=None,
        image_type='cv', mask_type='cv', **kwargs):
    """
    Apply cv2 function with mask on copy.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    func : TYPE
        function to apply.
    return_argument_idx : None or int, optional
        If the function returns more than one value, specifying this value
        tells the function which return value to use. The default is None.
    image_type : TYPE, optional
        DESCRIPTION. The default is 'cv'.
    mask_type : TYPE, optional
        DESCRIPTION. The default is 'cv'.

    Returns
    -------
    image_out : TYPE
        DESCRIPTION.

    """
    # convert to np
    image_np = convert(image_type, 'np', image)
    mask = convert(mask_type, 'np', mask).astype(bool)
    # get relevant pixles (e.g. where mask is True)
    loc = np.where(mask)
    # get the corresponding values
    values = image_np[mask]
    # apply function
    if return_argument_idx is None:
        new_values = func(values, *args, **kwargs)
    else:
        new_values = func(values, *args, **kwargs)[return_argument_idx]
    # initiate copy
    image_enhanced = image_np.copy()
    # set values in copy accordingly
    for i, coord in enumerate(zip(loc[0], loc[1])):
        image_enhanced[coord[0], coord[1]] = new_values[i][0]
    # convert back
    image_out = convert('np', image_type, image_enhanced)
    return image_out


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0):
    """
    Interpolate missing values in an array.

    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    # https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def downscale_image(image: np.ndarray, scale_factor: float = 1) -> np.ndarray:
    """Downscale image by factor smaller than 1."""
    if scale_factor >= 1:
        raise ValueError('A scaling factor of 1 has no effect and \
a value of more than 1 will result in a larger image, choose a \
value in the interval (0, 1).')
    downscaled_image = cv2.resize(
        image, fx=scale_factor, fy=scale_factor, dsize=None,
        interpolation=cv2.INTER_NEAREST)
    return downscaled_image


def upscale_image(image=None, scale_factor=1):
    """Upscale image by factor smaller than 1."""
    if scale_factor <= 1:
        raise ValueError('A scaling factor of 1 has no effect and \
a value of less than 1 will result in a smaller image, choose a \
value in the interval (1, infty).')
    upscaled_image = cv2.resize(
        image, fx=scale_factor, fy=scale_factor, dsize=None,
        interpolation=cv2.INTER_NEAREST)
    return upscaled_image


def auto_downscaled_image(
        image: np.ndarray, minpixels: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Downsample image such that the shorter side has the desired amount of pixels.
    """
    n_pixels: int = np.min(image.shape[:2])
    # if image is smaller than specified minpixles, return image with
    # scalefactor 1
    if n_pixels <= minpixels:
        return image, 1
    # scale down to minpixels
    scale_factor: float = minpixels / n_pixels
    image: np.ndarray = downscale_image(image, scale_factor)
    return image, scale_factor


def main():
    """Test functionality of all functions."""
    try:
        # setup
        from Image_plotting import plt_cv2_image
        test_img = convert('np', 'cv', skimage.data.brick())
        mask = test_img < 120
        plt_cv2_image(test_img, 'original')
        plt_cv2_image(mask, 'mask')
        # adaptive mean on mask
        plt_cv2_image(adaptive_mean_with_mask(
            test_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0),
            'adaptive mean on mask'
        )
        # outlier removal
        plt_cv2_image(remove_outliers_by_median(test_img, 11, 20), 'outliers removed')
        # threshold as min of bimodal distribution
        threshold_background_as_min(test_img, plts=True)

        # cv function on mask
        plt_cv2_image(func_on_image_with_mask(
            test_img,
            mask,
            cv2.threshold,
            127,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            return_argument_idx=1), 'otsu binarisation of foreground'
        )
        # missing values interpolation
        img_missing = test_img.astype(float)
        img_missing[mask] = np.nan
        plt_cv2_image(interpolate_missing_pixels(img_missing, mask),
                      'interpolated missing pixels')
        plt_cv2_image(upscale_image(downscale_image(test_img, 1 / 5), 5),
                      'down and upscaled image')
        return True
    except Exception as e:
        print(e)
        return False


def compare_adaptive_means():
    from imaging.util.Image_plotting import plt_cv2_image
    test_img = convert('np', 'cv', skimage.data.brick())
    mask = (test_img > 100).astype(np.uint8)
    plt_cv2_image(test_img, 'original')
    plt_cv2_image(mask, 'mask')

    a1 = adaptive_mean_with_mask(
        src=test_img,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=0,
        mask=mask)
    plt_cv2_image(a1, 'adaptive mean on mask (cv style)')

    a2 = adaptive_mean_with_mask_by_rescaling(
        image=test_img,
        mask_nonholes=mask,
        thresholdType=cv2.THRESH_BINARY,
        maxValue=1,
        C=0,
        ksize=(11, 11))
    plt_cv2_image(a2, 'adaptive mean on mask (rescaled weights)')
    plt_cv2_image(a1 - a2, 'difference')


if __name__ == '__main__':
    # setup
    # from Image_plotting import plt_cv2_image
    # test_img = convert('np', 'cv', skimage.data.brick())
    # mask = (test_img > 100).astype(np.uint8)
    compare_adaptive_means()
