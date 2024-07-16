"""Helper module for finding a sub-region with different scale."""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from skimage.feature import SIFT, match_descriptors, plot_matches, daisy
from skimage.transform import AffineTransform, warp
from skimage.color import rgb2gray

from scipy.optimize import minimize
from PIL import Image as PIL_Image
from typing import Sequence

logger = logging.getLogger(__name__)

def match_template_scale(
        image: np.ndarray[float], 
        template: np.ndarray[float], 
        N_guesses: int = 20,
        fine_tune: bool = True, 
        plts: bool = False, 
        method: str = 'Powell'
) -> tuple[tuple[int, int], float]:
    """
    Find scale and location of a template within an image.
    
    This function is build around openCVs matchTemplate function which expects
    the image and template to be at the same scale. This function takes the 
    maximum of the matches. 

    Parameters
    ----------
    image : np.ndarray[float]
        The image in which to search the template.
    template : np.ndarray[float]
        The template to be searched.
    N_guesses : int, optional
        The number of guesses before the fine-tuning. Used for a coarse 
        estimation of the scale. The default is 10.
    fine_tune : bool, optional
        Option to use a fine-tuning step to nail down the scale in the vicinity
        of the coarse guess. The default is True.
    plts : bool, optional
        Option to plot the final result. The default is False.
    method : str, optional
        method keyword passed on to scipy.optimize.minimize. 
        The default is 'Powell'. Must be an optimizer that accepts bounds.

    Returns
    -------
    (tuple[tuple[int], int])
        The location (top left corner) and scale to convert and insert the 
        template into the original image.

    """
    def eval_fit_for_scale(scale: float | int, return_loc: bool = False) -> float | Sequence[float, tuple[int, int]]:
        """
        Rescale the template to a certain scale and return the highest value of the match template cv2 function.

        Option to return the location as well (disabled by default).

        For invalid scales, this function returns the value 100.
        """
        dim_new: np.ndarray[int] = (np.asarray(template.shape[::-1]) * scale).astype(int)
        template_rescaled: np.ndarray = cv2.resize(template, dim_new)
        # catch invalid dimensions (e.g. template bigger than image) prematurely
        if (image.shape[0] <= template_rescaled.shape[0]) or (image.shape[1] <= template_rescaled.shape[1]):
            r: float = 1.
            if return_loc:
                return r, (-1, -1)
            return r
        res: np.ndarray = cv2.matchTemplate(image, template_rescaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if return_loc:
            return -max_val, max_loc
        return -max_val
    
    # guess the scale
    # (conversion factor to bring the template to the same resolution as the image)
    # image ROI should cover 5 cm section 
    # image should cover 5 cm plus stuff at the side
    scale_guess: float = image.shape[1] / template.shape[1]
    
    # scales to try for coarse guess
    scales: np.ndarray[float] = np.linspace(scale_guess / 10, scale_guess, N_guesses)

    # initialize holders for guesses
    min_val_best: float = np.infty
    loc_best: tuple[int, int] = (0, 0)
    scale_best: float = 0
    for scale in scales:
        min_val, loc = eval_fit_for_scale(scale, True)
        # update best guess
        if min_val < min_val_best:
            min_val_best: float = min_val
            loc_best: tuple[int, int] = loc
            scale_best: float = scale

    # fine tune step with optimizer
    if fine_tune:
        idx_best: int = np.argwhere(scales == scale_best)[0][0]
        bound_lower: float = scales[idx_best - 1] if idx_best > 0 else scales[idx_best]
        bound_upper: float = scales[idx_best + 1] if idx_best < len(scales) - 1 else scales[idx_best]
        # use optimizer to search 
        res = minimize(
            eval_fit_for_scale,
            x0=scale_best,
            args=(False),
            method=method,
            bounds=[(bound_lower, bound_upper)]
        )
        if res.success:
            scale_best: float = res.x[0]
            loc_best: tuple[int, int] = eval_fit_for_scale(scale_best, True)[1]
        else:
            logger.warning('optimizer did not succeed')
            logger.warning(res.message)
            logger.warning('using coarse scale match')

    if plts:
        plt_match_template_scale(image, template, scale_best, loc_best)

    return loc_best, scale_best

def plt_match_template_scale(
        image: np.ndarray,
        template: np.ndarray,
        scale: float,
        loc: tuple[int, int]
) -> None:
    """
    Plot the result of a match_template_scale call.

    Parameters
    ----------
    image: np.ndarray
        original image
    template: np.ndarray
        region to search
    scale: float
        scale with which to convert the template
    loc: tuple[int, int]
        location (upper left corner) of the template within image

    Returns
    -------
    None
    """
    dim: np.ndarray[int] = (np.asarray(template.shape[::-1]) * scale + .5).astype(int)
    template_best: np.ndarray[int] = cv2.resize(template, dim)

    xs_corner: tuple[int, int] = [loc[0], loc[0] + dim[0]]
    ys_corner: tuple[int, int] = [loc[1], loc[1] + dim[1]]

    canvas: np.ndarray = image.copy()
    canvas[
        loc[1]:loc[1] + dim[1],
        loc[0]:loc[0] + dim[0]
    ] = template_best

    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].imshow(image)
    axs[0].set_title('original image')

    axs[1].imshow(canvas)
    axs[1].scatter(xs_corner, ys_corner, c='r', marker='+', label='corners')
    axs[1].set_title('image with inserted template')
    plt.legend()
    plt.show()



def find_ROI_in_image(
        file_image: str = None, 
        file_image_roi: str = None,
        image: np.ndarray = None,
        image_roi: np.ndarray = None,
        **kwargs: dict
) -> tuple[tuple[int, int], int]:
    """
    This function is a wrapper around the match_template_scale to locate the
     position and scale of a ROI in an image.

    Parameters
    ----------
    :param file_image: image file to be used as the source image in
        case image is not provided.
    :param file_image_roi: image file to be used as the template in
        case image_roi is not provided.
    :param image: the image to be used as source image.
    :param image_roi: the template to be searched in the source image.
    :param kwargs: optional keyword arguments to be passed to match_template_scale

    Returns
    -------
    loc_best, scale_best
        The location and scale of the template inside the source image.
    """
    assert (file_image is not None) or (image is not None), \
        'either specify the file path or pass the image directly'
    assert (file_image_roi is not None) or (image_roi is not None), \
        'either specify the file path or pass the image directly'
    if file_image is not None:
        image = np.asarray(PIL_Image.open(file_image))
    if len(image.shape) == 3:
        image = rgb2gray(image)
        
    image = image.astype(np.float32)
    image /= image.max()
    
    if file_image_roi is not None:
        image_roi = np.asarray(
            PIL_Image.open(file_image_roi)
        ).sum(axis=-1)
    if len(image_roi.shape) == 3:
        image_roi = rgb2gray(image_roi)
    image_roi = image_roi.astype(np.float32)
    image_roi /= image_roi.max()
    
    return match_template_scale(image, image_roi, **kwargs)

def feature_matching():
    raise NotImplementedError('doesnt work really well')
    img = np.asarray(PIL_Image.open(
        r'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_480-510cm_100µm '
        r'slices/S0343c_490-495cm/Caricao_490-495cm_100µm_Mosaic.tif'
    ).convert('L'))

    ROI = (pd.read_csv(
        r'D:/Cariaco line scan Xray/uXRF slices/S0343 Cariaco_'
        r'480-510cm_100µm slices/S0343c_490-495cm/S0343c_Video 1.txt',
        sep=';'
    ).to_numpy()  / 2 ** 8).astype(np.uint8)
    
    sift = SIFT()
    # sift = daisy
    
    sift.detect_and_extract(img)
    keypoints_img = sift.keypoints
    descriptors_img = sift.descriptors
    
    sift.detect_and_extract(ROI)
    keypoints_ROI = sift.keypoints
    descriptors_ROI = sift.descriptors
    
    plt.imshow(img)
    plt.scatter(keypoints_img[:, 1], keypoints_img[:, 0], s=.1, c='r')
    plt.show()
    
    plt.imshow(ROI)
    plt.scatter(keypoints_ROI[:, 1], keypoints_ROI[:, 0], s=.1, c='r')
    plt.show()
    
    matches_img_ROI = match_descriptors(descriptors_img, descriptors_ROI)
    
    plt.imshow(img)
    plt.scatter(keypoints_img[matches_img_ROI[:, 0], 1], keypoints_img[matches_img_ROI[:, 0], 0], s=.1, c='r')
    plt.show()
    
    plt.imshow(ROI)
    plt.scatter(keypoints_ROI[matches_img_ROI[:, 1], 1], keypoints_ROI[matches_img_ROI[:, 1], 0], s=.1, c='r')
    plt.show()
    
    # fit
    
    tf = AffineTransform()
    tf.estimate(
        keypoints_img[matches_img_ROI[:, 0], :],
        keypoints_ROI[matches_img_ROI[:, 1], :]
    )
    warped = warp(ROI, tf, output_shape=img.shape)
    
    plt.imshow(warped)
