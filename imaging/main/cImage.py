from imaging.interactive import InteractiveImage
from res.constants import key_light_pixels, key_dark_pixels, key_hole_pixels
from util.manage_obj_saves import class_to_attributes
from util.cClass import Convinience, return_existing
from imaging.misc.fit_distorted_rectangle import find_layers, distorted_rect
from imaging.misc.find_punch_holes import find_holes

import imaging.util.Image_convert_types as Image_convert_types
from imaging.util.coordinate_transformations import rescale_values
from imaging.util.Image_convert_types import ensure_image_is_gray
from imaging.util.Image_plotting import plt_cv2_image, plt_contours, plt_rect_on_image
from imaging.util.Image_processing import (
    adaptive_mean_with_mask_by_rescaling,
    remove_outliers_by_median,
    threshold_background_as_min,
    func_on_image_with_mask,
    auto_downscaled_image,
    downscale_image, adaptive_mean_with_mask
)

from imaging.util.Image_geometry import (
    calculate_directionality_PCA,
    star_domain_contour
)

from imaging.util.Image_helpers import (
    ensure_odd,
    get_half_width_padded,
    min_max_extent_layer,
    filter_contours_by_size,
    get_foreground_pixels_and_threshold, get_simplified_image
)

from imaging.util.Image_boxes import get_mean_intensity_box, region_in_box, get_ROI_in_image

import pickle
import pandas as pd
import os
import cv2
import scipy
import functools
import numpy as np
import matplotlib.pyplot as plt
import logging

from skimage.segmentation import expand_labels
from scipy.optimize import minimize
from typing import Iterable, Self
from PIL import Image as PIL_Image

logger = logging.getLogger("msi_workflow." + __name__)


class Image(Convinience):
    """Base function to get sample images and analyze them."""

    def __init__(
            self,
            obj_color: str,
            path_image_file: str | None = None,
            image: np.ndarray[float | int] | None = None,
            image_type: str = 'cv',
            path_folder: str | None = None
    ) -> None:
        """Initiator.

        Parameters
        ----------
        obj_color : str
            The foreground color of the object in the image. Either 'light' or 'dark'.
            This is required for working with thresholded images is desired.
        path_image_file : str
            The file path to an image file to be read.
        image : np.ndarray[float | int]
            Alternatively, an image can be provided directly.
        image_type: str
            If the input image is not a cv image, provide this keyword argument. Options are 'cv', 'np', 'pil'
            for images read or processed with OpenCV, numpy or PILLOW respectively.

        """
        assert (path_image_file is not None) or (image is not None), \
            "Must provide either path or image"
        image_types: tuple[str, ...] = ('cv', 'np', 'pil')
        assert image_type in image_types, f'valid image types are {image_types}, depending on the source of the image'
        obj_colors: tuple[str, str] = ('light', 'dark')
        assert obj_color in obj_colors, f'valid object colors are {obj_colors}'

        self.obj_color: str = obj_color

        if path_image_file is not None:
            self.path_image_file: str = path_image_file
            image = cv2.imread(path_image_file)
            if path_folder is None:
                path_folder: str = os.path.dirname(path_image_file)

        # set _image_original
        self._from_image(image, image_type)

        self.path_folder = path_folder if path_folder is not None else ''

    def _from_image(self, image: np.ndarray, image_type: str) -> None:
        """
        Set attributes from the image and type.

        This function ensures that the image is oriented horizontally, and sets the original image as a cv image.
        """
        image = Image_convert_types.convert(
            image_type, 'cv', image.copy()
        )
        # make sure image is oriented horizontally
        h, w, *_ = image.shape
        if h > w:
            logger.info('swapped axes of input image to ensure horizontal orientation')
            # swapaxes returns a view by default
            image = image.copy().swapaxes(0, 1)
        self._hw = h, w
        self._image_original: np.ndarray[int | float] = image
        self._image: np.ndarray[int | float] = self._image_original.copy()

    @classmethod
    def from_disk(cls, path_folder: str) -> Self:
        # initiate dummy object that provides all, albeit nonsensical, parameters
        dummy: Self = cls(path_folder=path_folder, image=np.ones((3, 3)), obj_color='light')
        dummy.load()
        # load messes with _image, _image_original, the constructor can take care of that
        new: Self = cls(
            obj_color=dummy.obj_color,
            path_image_file=dummy.__dict__.get('path_image_file'),
            image=dummy.__dict__.get('_image_original'),
            image_type='cv',
            path_folder=path_folder,
        )

        dummy.__dict__.update(new.__dict__)

        return dummy

    @property
    def image(self) -> np.ndarray[int | float]:
        """Return a copy of the original image"""
        return self._image.copy()

    @functools.cached_property
    def image_grayscale(self) -> np.ndarray:
        """Return a grayscale version of the original image."""
        return ensure_image_is_gray(self.image).copy()

    def set_foreground_thr_and_pixels(
            self, thr_method: str = 'otsu', plts: bool = False, **_
    ) -> None:
        """
        Set threshold for foreground pixels and thresholded binary image.

        Parameters
        ----------
        thr_method : str, optional
            The method to use for thresholding. The default is 'otsu'.
            The other option is 'local-min'.
        plts : bool, optional
            Whether to plot the foreground pixels.

        Returns
        -------
        None

        """
        mask, thr = get_foreground_pixels_and_threshold(
            image=self.image,
            obj_color=self.obj_color,
            method=thr_method
        )
        self._thr_background: int | float = thr
        self._mask_foreground: np.ndarray[int] = mask
        if plts:
            plt_cv2_image(mask, 'Identified foreground pixels')

    def require_foreground_thr_and_pixels(
            self, **kwargs
    ) -> tuple[float | int, np.ndarray[int]]:
        """Make sure the foreground mask and threshold exists before returning it"""
        if not hasattr(self, '_mask_foreground'):
            self.set_foreground_thr_and_pixels(**kwargs)
        return self._thr_background, self._mask_foreground

    @property
    def image_binary(self):
        return self.require_foreground_thr_and_pixels()[1]

    @property
    def mask_foreground(self):
        """A mask where foreground pixels are True and background pixels are False"""
        return self.require_foreground_thr_and_pixels()[1]

    @property
    def thr_foreground(self):
        """The gloabla threshold where foreground are separated from background pixels."""
        return self.require_foreground_thr_and_pixels()[0]

    def get_binarisation_of_foreground(
            self,
            image: np.ndarray | None = None,
            mask: np.ndarray[int | bool] | None = None,
            plts: bool = False
    ) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        """
        OTSU-binarisation of foreground pixels.

        Parameters
        ----------
        image : cv2 image | None
            imagefor which to create binarisation.
        mask : cv2 image of uint8
            Pixels with foreground are 255.
        plts : bool, optional
            Plot resulting light and dark masks.

        Returns
        -------
        light_pixels : cv2 image
            The light pixels.
        dark_pixels : cv2 image
            The dark pixels.
        """
        if image is None:
            image = self.image
        if mask is None:
            mask = self.mask_foreground

        # create instances of functions
        func_light = functools.partial(
            cv2.threshold,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        func_dark = functools.partial(
            cv2.threshold,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        light_pixels = func_on_image_with_mask(
            image, mask, func_light, return_argument_idx=1) == 255

        dark_pixels = func_on_image_with_mask(
            image, mask, func_dark, return_argument_idx=1) == 255

        light_pixels = light_pixels.astype(np.uint8)
        dark_pixels = dark_pixels.astype(np.uint8)

        if plts:
            plt_cv2_image(light_pixels, 'identified light pixels')
            plt_cv2_image(dark_pixels, 'identified dark pixels')

        return light_pixels, dark_pixels

    def set_simplified_image(self, **kwargs) -> None:
        self._image_simplified: np.ndarray[int] = get_simplified_image(self.image_binary, **kwargs)

    def require_simplified_image(self, **kwargs) -> np.ndarray:
        if not hasattr(self, '_image_simplified'):
            self.set_simplified_image(**kwargs)
        return self._image_simplified

    @property
    def image_simplified(self) -> np.ndarray:
        return self.require_simplified_image()

    def set_main_contour(
            self,
            method: str = 'take_largest',
            filter_by_size: float = .3,
            plts: bool = False
    ):
        """
        Find and return the contour surrounding the sample. Uses the binary image.

        Several methods exist, so far take_largest seems to work best.
            'take_largest' picks the longest identified contour of the image
            'filter_by_size' concatenates all contours above the provided threshold.
            'star_domain' assumes that every point on the boundary can be reached in a straight line from the center.
            'convex_hull' returns the convex hull of all concatenated contours.

        Parameters
        ----------
        method
        filter_by_size
        plts

        Defines
        -------
        _main_contour
            The contour surrounding the sample as an array where each row describes a point.
        """
        methods = ('take_largest', 'star_domain', 'filter_by_size', 'convex_hull')
        assert method in methods, f"{method=} is not an option. Valid options are {methods}"

        image_binary = self.image_binary

        contours, _ = cv2.findContours(
            image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if plts:
            plt_contours(contours, image_binary, 'all detected contours')

        # take the largest one
        if method == 'take_largest':
            contour = contours[
                np.argmax([contour.shape[0] for contour in contours])]

        # filter out smaller contours only if method is not taking the largest
        # contour and filter_by_size has not been set to false
        elif method == 'filter_by_size':
            contours = filter_contours_by_size(
                contours, image_binary.shape, threshold_size=filter_by_size)
            contour = np.concatenate(contours)

        # this method will convert the domain to proxystardomain
        elif method == 'star_domain':
            # combine all contours into one array
            contour = np.concatenate(contours)
            center_points_x = [round(image_binary.shape[1] / 4 * (i + 1))
                               for i in range(3)]
            center_points_y = [round(image_binary.shape[0] / 2)] * 3
            center_points = list(zip(center_points_x, center_points_y))
            for center_point in center_points:
                contour = star_domain_contour(
                    contour=contour,
                    center_point=center_point,
                    smoothing_size=1,
                    plts=self.plts)

        elif method == 'convex_hull':
            contour = np.concatenate(contours)
            contour = cv2.convexHull(contour)
        else:
            raise KeyError(
                f"{method=} is not an option. Valid options are {methods}")

        if plts:
            plt_contours(
                contours=[contour],
                image=image_binary,
                title='main contour'
            )

        self._main_contour = contour

    def require_main_contour(self, **kwargs):
        if not hasattr(self, '_main_contour'):
            self.set_main_contour(**kwargs)
        return self._main_contour

    @property
    def main_contour(self):
        return self.require_main_contour()

    def set_age_span(self, age_span: tuple[float | int, float | int]):
        """
        Set the age span of the sample as a tuple (in yrs).

        This allows a more precise definition of the kernel size used in the classification.
        """
        self.age_span: tuple[float | int, float | int] = age_span

    def set_average_width_yearly_cycle(self, pixels: int | None = None) -> float:
        """Calculate how many cycles are in the interval and their av width."""
        if pixels is not None:
            self._average_width_yearly_cycle = pixels
            return
        assert hasattr(self, 'age_span'), 'call set_age_span'
        pixels_x = self.image.shape[1]
        # calculate the number of expected cycles from the age difference for
        # the depth interval of the slice
        self._average_width_yearly_cycle = pixels_x / abs(self.age_span[1] - self.age_span[0])

    @property
    def average_width_yearly_cycle(self):
        if not hasattr(self, '_average_width_yearly_cycle'):
            if not hasattr(self, 'age_span'):
                raise KeyError('Define an age span before calculating the average width of annual layers.')
            else:
                self.set_average_width_yearly_cycle()
        return self._average_width_yearly_cycle

    def plot(self, **kwargs):
        plt_cv2_image(image=self.image, **kwargs)


class ImageSample(Image):
    """
    Define sample area.

    This function uses a multistep approach to find the sample area. Oftentimes it is necessary to define the
    object color, which tells the algorithms if the pixels of the samples are lighter than the background (in which
    case the 'obj_color' keyword should be set to 'light') or darker (obj_color='dark'). It is recommended to save

    Example Usage
    -------------
    >>> from msi_workflow.imaging.cImage import ImageSample
    Create an ImageSample object from an image on disk
    >>> i = ImageSample(path_image_file="/path/to/your/file")
    in this case the object color will be infered, it is adviced to set it manually
    >>> i = ImageSample(path_image_file="/path/to/your/file", obj_color='light')
    or
    >>> i = ImageSample(path_image_file="/path/to/your/file", obj_color='dark')
    depending on your sample. The resolution of the image matters for downstream applications (e.g. combination of
    image with MSI measurement. Therefore, for MSI applications it is adviced to use the project class which takes
    care of finding the right image file (specified in the mis-file).
    Alternatively, one can load in a previously saved ImageSample instance by providing the folder path
    >>> i = ImageSample.from_path("path/to/your/folder")
    or
    >>> i = ImageSample(path_image_file='path/to/your/file')
    >>> i.load()

    It is recommended to stick to the properties, e.g. image, image_grayscale, image_binary, image_simplified, main_contour.
    The most important property is the image_sample_area which is the final result of performing all the steps of finding
    the sample area, so initiating and checking a new instance could look like this:
    >>> from msi_workflow.imaging.cImage import ImageSample
    >>> i = ImageSample(path_image_file="/path/to/your/file")
    >>> i.set_sample_area()
    >>> i.plot_overview()
    >>> i.save()
    """

    def __init__(
            self,
            path_folder: str | None = None,
            image: np.ndarray[float | int] | None = None,
            image_type: str = 'cv',
            path_image_file: str | None = None,
            obj_color: str | None = None
    ):
        """Initiator."""
        super().__init__(
            path_folder=path_folder,
            path_image_file=path_image_file,
            image=image,
            image_type=image_type,
            obj_color='light'  # give a dummy, will be overwritten
        )

        # overwrite the obj color attribute of the super init method
        self.obj_color = self._get_obj_color() if obj_color is None else obj_color

    def _get_obj_color(self, region_middleground=.8, **_):
        """
        Determine if middleground is light or dark by comparing averages.

        Parameters
        ----------
        image : uint8 array.
            The image to check.
        region_middleground: float between 0 and 1.
            The region defined as the middle ground in each direction from the
            center expressed as a fraction. The default is .8.

        Returns
        -------
        str
            'light' if middle-ground is lighter than whole image
            and 'dark' otherwise.

        """
        image_gray = self.image_grayscale

        height, width = image_gray.shape[:2]
        # determine indizes of box
        idx_height_min = round((1 - region_middleground) * height)
        idx_height_max = round(region_middleground * height)
        idx_width_min = round((1 - region_middleground) * width)
        idx_width_max = round(region_middleground * width)
        image_region = image_gray[idx_height_min:idx_height_max,
                       idx_width_min:idx_width_max]

        # of values for 4 channels take first one
        # (only nonzero for grayscale img)
        image_region_mean = cv2.mean(image_region)[0]
        image_mean = cv2.mean(image_gray)[0]
        if image_region_mean > image_mean:
            obj_color = 'light'
        else:
            obj_color = 'dark'

        logger.info(f'obj appears to be {obj_color}')

        return obj_color

    def get_sample_area_box(
            self,
            dilate_factor: float = 1,
            plts: bool = False,
            **kwargs
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Use optimizer to find sample area in image as a horizontal box..

        Parameters
        ----------
        dilate_factor : float, optional
            value used to dilate box. The default is 1. Value smaller than 1:
                will be added to the determined ratio
            Value bigger than 1:
                will be multiplied with the determined box ratio
        plts: bool, optional
            if True, will plot the box

        Returns
        -------
        image_roi, xywh
            The image inside the box and corner as well as width and height of box.

        """

        def metric(x0):
            """
            Calculate the difference in pixel intensity for a specified box.

            Parameters
            ----------
            x0 : tuple
                The center and ratios of the box.

            Returns
            -------
            float
                negative absolute difference of inside and outside pixel
                intensities normed by the area of the box.

            """
            box_ratio_x, box_ratio_y, center_box_x, center_box_y = x0
            center_box = (
                    np.array([center_box_x, center_box_y]) + .5).astype(int)
            mean_box, mean_rest = get_mean_intensity_box(
                image_downscaled,
                box_ratio_x=box_ratio_x,
                box_ratio_y=box_ratio_y,
                center_box=center_box)

            fraction_area = box_ratio_y * box_ratio_x
            return -np.abs(mean_box - mean_rest) * fraction_area

        image_binary = self.image_binary
        image_downscaled, scale_factor = auto_downscaled_image(image_binary)
        # initiate center_box
        middle_y: int = round(image_downscaled.shape[0] / 2)
        middle_x: int = round(image_downscaled.shape[1] / 2)

        logger.info('searching optimal parameters for box')

        x0 = np.array([.5, .5, middle_x, middle_y])
        params = minimize(
            metric,  # function to minimize
            x0=x0,  # start values
            method='Nelder-Mead',  # method
            bounds=[  # bounds of parameters
                (0, 1),
                (0, 1),
                (0, image_downscaled.shape[1]),
                (0, image_downscaled.shape[0])
            ]
        )
        # determined values
        box_ratio_x, box_ratio_y, center_box_x, center_box_y = params.x
        center_box: tuple[int, int] = (center_box_x, center_box_y)
        logger.info(f'found box with {params.x}')
        logger.info(f'solver converged: {params.success}')

        # get params of box from those determined by the optimizer
        box_params: dict[str, float] = region_in_box(
            image=image_downscaled, box_ratio_x=box_ratio_x,
            box_ratio_y=box_ratio_y, center_box=center_box)
        if plts:
            plt_rect_on_image(image_downscaled, box_params, title='Detected ROI of sample', **kwargs)

        # dilate the box slightly for finer sample definition
        if dilate_factor > 1:
            box_ratio_x *= dilate_factor
            box_ratio_y *= dilate_factor
        elif dilate_factor < 1:
            box_ratio_x += dilate_factor
            box_ratio_y += dilate_factor
        # calculate new box
        if dilate_factor != 1:
            box_params = region_in_box(
                image=image_downscaled, box_ratio_x=box_ratio_x,
                box_ratio_y=box_ratio_y, center_box=center_box)

        x = box_params['x']
        y = box_params['y']
        w = box_params['w']
        h = box_params['h']

        # scale image back to original resolution
        if (scale_factor != 1) and (scale_factor is not None):
            x = round(x / scale_factor)
            y = round(y / scale_factor)

            w = round(w / scale_factor)
            h = round(h / scale_factor)
        # select the ROI in the image with original scale
        image_ROI = self.image[y:y + h, x:x + w].copy()

        if plts:
            plt_cv2_image(image_ROI, 'detected ROI')

        return image_ROI, (x, y, w, h)

    def get_sample_area_from_contour(self, plts: bool = False, **kwargs):
        """
        Get the roi spanned by the main contour.

        It is recommended to use this function in conjunction with get_sample_area_box since the algorithm to find
        the main contour can fail if the sample material is split into multiple parts (see get_sample_area).
        """
        contour = self.require_main_contour(**kwargs)

        x, y, w, h = cv2.boundingRect(contour)

        image_roi = self.image[y:y + h, x:x + w]

        if plts:
            plt_cv2_image(
                image_roi, 'detected ROI as bounding box of main contour'
            )

        return image_roi, (x, y, w, h)

    def set_sample_area(self, plts: bool = False, interactive: bool = False, **_) -> None:
        """
        Find the sample area of a sample in multiple steps.

        Firstly, use the get_sample_area_box function  and dilate the result to get a rough estimate that definitely
        includes the entire sample. Then, find the contour of the simplified area and its bounding box.

        Parameters
        ----------
        plts
        interactive: bool,
            If True, opens window for user to define the sample area

        Returns
        -------
        None
        """
        if interactive:
            self._user_sample_area()
            # redefine foreground
            self.set_image
            self.set_foreground_thr_and_pixels()
            self.set_simplified_image()
            return

        # find the rough region of interest with box
        image_box, (xb, yb, wb, hb) = self.get_sample_area_box(dilate_factor=0.1, plts=plts)
        # set as new image
        image_sub: ImageSample = ImageSample(image=image_box, obj_color=self.obj_color)
        # set image simplified for contour to use
        image_sub._mask_foreground = image_sub.image_simplified
        # find the refined area as the extent of the simplified binary image
        _, (xc, yc, wc, hc) = image_sub.get_sample_area_from_contour(method='filter_by_size', plts=plts)

        # stack the offsets of the two defined ROI's since the second ROI is
        # placed in the first one
        x = xc + xb
        y = yc + yb
        w = wc
        h = hc

        image_roi = self.image[y: y + h, x: x + w].copy()

        if plts:
            plt_cv2_image(
                image_roi,
                'final ROI as defined by get_sample_area'
            )

        self._image_roi = image_roi
        self._xywh_ROI = (x, y, w, h)

    def _user_sample_area(self):
        interactive_image: InteractiveImage = InteractiveImage(self.image, mode='rect')
        interactive_image.show()
        if len(interactive_image.x_data) != 2:
            logger.error('Please provide exactly two points')
            return
        xs = (np.array(interactive_image.x_data) + .5).astype(int)
        ys = (np.array(interactive_image.y_data) + .5).astype(int)

        x = min(xs)
        y = min(ys)
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)

        self._xywh_ROI = (x, y, w, h)
        self._image_roi = self.image[y: y + h, x: x + w].copy()

    def require_image_sample_area(
            self, **kwargs
    ) -> tuple[np.ndarray[np.uint8], tuple[int, ...]]:
        """Set and return area of the sample in the image."""
        if (not hasattr(self, '_xywh_ROI')) and (not hasattr(self, '_image_roi')):
            self.set_sample_area(**kwargs)
        elif not hasattr(self, '_image_roi'):
            self._image_roi = self.get_sample_area_from_xywh()
        return self._image_roi, self._xywh_ROI

    @property
    def image_sample_area(self):
        x, y, w, h = self.require_image_sample_area()[1]
        return self.image[y:y + h, x:x + w].copy()

    def get_sample_area_from_xywh(self):
        assert hasattr(self, '_xywh_ROI'), 'no roi found, call require_image_sample_area'
        image = self.image
        x, y, w, h = self._xywh_ROI
        return image[y:y + h, x:x + w].copy()

    def plot_overview(self, **kwargs):
        # original image
        image_box, (xb, yb, wb, hb) = self.get_sample_area_box(dilate_factor=0.1)
        # set as new image
        image_sub: ImageSample = ImageSample(image=image_box, obj_color=self.obj_color)
        # set image simplified for contour to use
        image_sub._mask_foreground = image_sub.image_simplified
        # find the refined area as the extent of the simplified binary image
        _, (xc, yc, wc, hc) = image_sub.get_sample_area_from_contour(method='filter_by_size')

        # stack the offsets of the two defined ROI's since the second ROI is
        # placed in the first one
        x = xc + xb
        y = yc + yb
        w = wc
        h = hc

        fig, axs = plt.subplots(nrows=2, ncols=2, **kwargs)

        plt_cv2_image(
            fig=fig,
            ax=axs[0, 0],
            image=self.image,
            title='input image',
            no_ticks=True
        )
        plt_rect_on_image(
            fig=fig,
            ax=axs[0, 1],
            image=self.image_binary,
            title='box dilated by 10 %',
            no_ticks=True,
            hold=True,
            box_params=region_in_box(image=self.image_binary, x=xb, y=yb, w=wb, h=hb)
        )
        plt_contours(
            fig=fig,
            ax=axs[1, 0],
            image=np.stack([image_sub.image_simplified] * 3, axis=-1) * image_sub.image,
            contours=image_sub.main_contour,
            title='contour in simplified sub-region',
            hold=True,
            no_ticks=True
        )
        plt_cv2_image(
            fig=fig,
            ax=axs[1, 1],
            image=self.require_image_sample_area()[0],
            title='final sample region', no_ticks=True
        )
        fig.tight_layout()
        plt.show()


class ImageROI(Image):
    """
    Classify foreground pixels into light and dark.

    This class distinguishes light and dark pixels in an image by using the adaptive_mean_filter together with a
    mask specifying fore- and background pixels. It is assumed that the image does not contain too much empty space at
    the boundaries, since this may cause issues downstream.

    If the image is derived from an ImageSample object, it is advised to iniate the ImageROI instance from
    the 'from_parent' constructor:
    >>> from msi_workflow.imaging.cImage import ImageROI
    >>> i = ImageSample(...)
    >>> i.set_from_parent(...)
    >>> ir = ImageROI.from_parent(i)

    Otherwise the instance can be initiated using the default constructor by either providing an image or a file to an image.
    For file management it is required to specify the path_folder, if path_image_file is not provided (otherwise the folder
    will be infered from the image file path). So initiating could look like this (obj_color is always required):
    >>> from msi_workflow.imaging.cImage import ImageROI
    >>> ir = ImageROI(obj_color='light', path_image_file='path/to/your/image/file.png')
    or
    >>> your_image = np.random.random((100, 200))
    >>> ir = ImageROI(obj_color='light', image=your_image)

    Classify the foreground pixels works best if the age span is known, in which case the kernel size can be estimated
    to cover about 2 years:
    >>> ir.set_age_span((0, 100))  # sample covers ages from 0 to 100 yrs
    >>> ir.set_classification_adaptive_mean_filter()
    otherwise
    >>> ir.set_classification_varying_kernel_size()
    can be used which takes the median classification of a bunch of classifications with filters at different sizes.
    This method is more prone to noise.
    >>> ir.require_classification()
    is a convinience function which chooses the adaptive mean method if an age span has been set and the varying kernel
    size approach otherwise.

    Under the hood, the classified images are pre- and post-processed, which can be controlled by setting keyword arguments.
    This includes smoothing/ denoising the input image and excluding small blobs from the final classification. The choice
    of parameters can be viewed in
    >>> ir._params

    It is also possible to find punchholes (the positions where square-shaped holes have been punched into the sediment
    for later registration):
    >>> ir.set_punchholes(remove_gelatine=True, plts=True)

    if path_folder or path_image_file has been specified, you can save the instance to disk:
    >>> ir.save()
    """

    def __init__(
            self,
            obj_color: str,
            path_folder: str | None = None,
            image: np.ndarray[float | int] | None = None,
            image_type: str = 'cv',
            path_image_file: str | None = None,
    ):
        """Initiator."""
        # options mutable by user
        super().__init__(
            image=image,
            image_type=image_type,
            path_image_file=path_image_file,
            path_folder=path_folder,
            obj_color=obj_color
        )

    @classmethod
    def from_parent(cls, parent: ImageSample) -> Self:
        """Alternative constructor for instantiating an object from a parent ImageSample instance."""
        new: Self = cls(
            path_folder=parent.path_folder,
            image=parent.image_sample_area,
            path_image_file=None,
            obj_color=parent.obj_color
        )

        return new

    def get_params_laminae_classification(
            self, image_gray_shape: tuple[int, ...], **kwargs
    ) -> dict:
        """Set default params for classification and overwrite by kwargs."""
        # set default values
        params = {
            'remove_outliers': False,
            'use_bilateral_filter': False,
            'remove_small_areas': False,
            'kernel_size_median': 7,
            'threshold_replace_median': 10,
            'sigmaColor': 100,
            'sigmaSpace': 3,
            'kernel_size_adaptive': None,
            'estimate_kernel_size_from_age_model': True,
            'threshold_size_contours': .01,
            'area_percentile': 2}
        # replace passed values
        for key in kwargs:
            if key in params:
                params[key] = kwargs[key]

        if not hasattr(self, 'age_span'):
            params['estimate_kernel_size_from_age_model'] = False

        # update kernel_size to potentially match ROI
        if params['estimate_kernel_size_from_age_model']:
            logger.info('Estimating kernel size from age model (square with \
2x expected thickness of one year).')
            kernel_size_adaptive = ensure_odd(
                int(self.average_width_yearly_cycle * 2))
            params['kernel_size_adaptive'] = kernel_size_adaptive
        elif params['kernel_size_adaptive'] is None:
            logger.info('estimating adaptive kernel size from image dimensions')
            kernel_size_adaptive = ensure_odd(np.min(image_gray_shape) // 10)
            params['kernel_size_adaptive'] = kernel_size_adaptive

        logger.debug('Using the following parameters:')
        logger.debug(params)

        return params

    def get_preprocessed_for_classification(
            self,
            plts: bool = False,
            **kwargs
    ) -> tuple[np.ndarray, np.ndarray, dict[str, bool | int | float]]:
        """
        Preprocess an image for classification (remove noise).

        this method performs the following steps
        1. remove outliers by detecting large differences between original
            image and its median filtered version

        2. identify back- and foreground pixels with otsu-filter
        3. apply bilateral filter
        """
        image_gray = self.image_grayscale

        # update params with kwargs
        params = self.get_params_laminae_classification(
            image_gray.shape, **kwargs
        )
        if plts:
            plt_cv2_image(image_gray, 'input in grayscale')

        # get mask_foreground matching image_gray
        mask_foreground = self.mask_foreground

        if plts:
            plt_cv2_image(
                mask_foreground,
                title=f'foreground pixels (thr={self.thr_foreground})'
            )

        # remove outliers
        if params['remove_outliers']:
            logger.info('Removing outliers with median filter.')
            image_gray = remove_outliers_by_median(
                image_gray, kernel_size_median=params['kernel_size_median'],
                threshold_replace_median=params['threshold_replace_median'])
            if plts:
                plt_cv2_image(image_gray, 'Outliers removed')

        if params['use_bilateral_filter']:
            logger.info('Applying bilateral filter.')
            image_gray = cv2.bilateralFilter(image_gray, d=-1,
                                             sigmaColor=params['sigmaColor'],
                                             sigmaSpace=params['sigmaSpace'])
            if plts:
                plt_cv2_image(image_gray, 'Bilateral filter')

        return image_gray, mask_foreground, params

    def get_postprocessed_image_from_classification(
            self,
            image_light: np.ndarray,
            mask_foreground: np.ndarray,
            params: dict,
            plts: bool = False
    ) -> np.ndarray:
        """Postprocess a classified image (remove small features)."""
        if params['remove_small_areas']:
            logger.info('Removing small blobs.')
            # find contours
            contours_light, _ = cv2.findContours(
                image_light.astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            contours_light = filter_contours_by_size(
                contours_light, image_light.shape,
                threshold_size=params['threshold_size_contours'])
            # get their areas
            conts_areas = []
            for contour in contours_light:
                conts_areas.append(cv2.contourArea(contour))
            conts_areas = np.array(conts_areas)
            # get pth percentile
            area_pth_percentile = np.percentile(
                conts_areas, params['area_percentile'])
            # calculate equivalent diameter
            diam_pth_percentile = np.sqrt(4 * area_pth_percentile / np.pi)
            # set the kernel to be 3 times the size of the diameter
            kernel_median = int(3 * diam_pth_percentile + .5)
            # ensure oddity
            if not kernel_median % 2:
                kernel_median += 1
            kernel_median = np.min([kernel_median, 255])
            image_light = cv2.medianBlur(image_light, kernel_median)

        image_light_laminae = (
                image_light & mask_foreground).astype(np.uint8)
        image_dark_laminae = (
                mask_foreground & (~image_light)).astype(np.uint8)
        image_classification = image_light_laminae * key_light_pixels + \
                               image_dark_laminae * key_dark_pixels

        if plts:
            plt_cv2_image(
                image_classification, 'final classification')

        return image_classification

    def set_classification_adaptive_mean(self, plts: bool = False, **kwargs) -> tuple[np.ndarray, dict]:
        """Classify image with adaptive mean threshold filter."""

        image_gray, mask_foreground, params = \
            self.get_preprocessed_for_classification(**kwargs)

        # adaptive thresholding
        logger.info('adaptive thresholding with mask')

        image_light = adaptive_mean_with_mask_by_rescaling(
            image=image_gray,
            maxValue=1,
            thresholdType=cv2.THRESH_BINARY,
            ksize=(params['kernel_size_adaptive'], params['kernel_size_adaptive']),
            C=0,
            mask_nonholes=mask_foreground
        )
        image_light *= mask_foreground.astype(bool)

        image_classification = \
            self.get_postprocessed_image_from_classification(
                image_light,
                mask_foreground,
                params
            )

        if plts:
            plt_cv2_image(image_classification, title='classified image')

        self._image_classification = image_classification
        self._params = params

    def set_classification_varying_kernel_size(self, scaling=2, plts: bool = False, **kwargs) -> np.ndarray[int]:
        """
        Binarize foreground pixels by taking the median of adaptive mean classifications across different scales.

        Parameters
        ----------
        scaling: float
            controls the downscaling factor

        Returns
        -------

        """
        image, mask, params = \
            self.get_preprocessed_for_classification(**kwargs)
        height, width = image.shape[:2]

        if self.obj_color == 'dark':
            threshold_type = cv2.THRESH_BINARY_INV
        else:
            threshold_type = cv2.THRESH_BINARY

        i_max = int(np.emath.logn(scaling, height))
        heights = height / scaling ** np.arange(i_max)
        # get last index where there are more than sixteen vertical pixels
        i_max = np.arange(i_max)[heights > 16][-1]

        res: np.ndarray[bool] = np.zeros((height, width, i_max), dtype=bool)

        for it in range(i_max):
            # downscale
            new_width = round(width / scaling ** it)
            new_height = round(height / scaling ** it)
            dim = (new_width, new_height)
            image_downscaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            mask_downscaled = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)
            # filter
            image_filtered = adaptive_mean_with_mask(
                src=image_downscaled,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=threshold_type,
                blockSize=3,
                C=0,
                mask=mask_downscaled
            ) * mask_downscaled
            # upscale
            image_rescaled = cv2.resize(image_filtered, (width, height), interpolation=cv2.INTER_NEAREST)
            res[:, :, it] = image_rescaled.astype(bool)

        res = np.median(res, axis=-1).astype(bool)
        mask = mask.astype(bool)

        image_light = (
                res & mask).astype(np.uint8)
        # image_dark = (
        #         mask & (~res)).astype(np.uint8)
        # image_classification = image_light * key_light_pixels + \
        #                        image_dark * key_dark_pixels

        image_classification = \
            self.get_postprocessed_image_from_classification(
                image_light,
                mask,
                params
            )

        self._image_classification = image_classification
        self._params = params | {'scaling': scaling}

        if plts:
            plt_cv2_image(image_classification, title='classified image')

    def require_classification(self):
        """Create and return the image classification with parameters."""
        if not hasattr(self, '_image_classification'):
            if not hasattr(self, 'age_span'):
                logger.warning('No age span specified, falling back to more general method')
                self.set_classification_varying_kernel_size()
            else:
                self.set_classification_adaptive_mean()

        return self._image_classification

    @property
    def image_classification(self):
        return self.require_classification()

    def set_punchholes(
            self,
            remove_gelatine: bool = True,
            interactive: bool = False,
            **kwargs: dict
    ) -> None:
        """
        Add punchholes to the current instance.

        Parameters
        ----------
        remove_gelatine: bool
            Try to remove gelatine residuals by masking foreground values with the simplified image since the simplified
            image usually does not contain the gelatine.
            If false, uses the binary image
        kwargs: dict, optional
            Extra keywords to be passed on to find_holes.
        """
        if interactive:
            self._user_punchholes()
            return

        img: np.ndarray[np.uint8] = self.image_binary
        if remove_gelatine:
            img *= self.image_simplified

        self._punchholes, self._punchhole_size = find_holes(
            img,
            obj_color=self.obj_color,
            **kwargs
        )

    def _user_punchholes(self) -> None:
        interactive_image = InteractiveImage(self.image, mode='punchholes')
        interactive_image.show()

        self._punchholes = tuple(zip(interactive_image.y_data, interactive_image.x_data))
        self._punchhole_size: int = interactive_image._punchhole_size

    def require_punchholes(self, *args, **kwargs) -> tuple[list[np.ndarray[int], np.ndarray[int]], float]:
        if not hasattr(self, "_punchholes"):
            self.set_punchholes(*args, **kwargs)
        return self._punchholes, self._punchhole_size

    def plot_overview(self):
        if not hasattr(self, "_punchholes"):
            logger.warning("Punchholes not set with required parameter 'remove_gelatine', this may affect performance.")
        self.require_punchholes(remove_gelatine=True)
        hole_size = round(self._punchhole_size)

        fig, axs = plt.subplots(nrows=2, ncols=2, layout="constrained")
        plt_cv2_image(fig=fig, ax=axs[0, 0], image=self.image, title="Input image", no_ticks=True)
        plt_cv2_image(fig=fig, ax=axs[0, 1], image=self.get_preprocessed_for_classification()[0],
                      title="Preprocessed image",
                      no_ticks=True)
        plt_cv2_image(fig=fig, ax=axs[1, 0], image=self.image_classification, title="Classified image",
                      no_ticks=True)

        fig, ax11 = plt_rect_on_image(
            fig=fig,
            ax=axs[1, 1],
            image=self.image_binary,
            box_params=region_in_box(
                image=self.image_binary,
                point_topleft=np.array(self._punchholes[0])[::-1] - hole_size / 2,
                point_bottomright=np.array(self._punchholes[0])[::-1] + hole_size / 2
            ),
            no_ticks=True,
            hold=True
        )
        plt_rect_on_image(
            fig=fig,
            ax=ax11,
            image=self.image_binary,
            box_params=region_in_box(
                image=self.image_binary,
                point_topleft=np.array(self._punchholes[1])[::-1] - hole_size / 2,
                point_bottomright=np.array(self._punchholes[1])[::-1] + hole_size / 2
            ),
            no_ticks=True,
            title='Detected punch-holes'
        )

        # fig.constrained_layout()
        plt.show()


class ImageClassified(Image):
    """
    Characterise and modify the classified layers.

    Example Usage
    -------------
    >>> from msi_workflow.imaging.cImage import ImageClassified, ImageROI
    initiate from parent object, most common use case.
    >>> ir = ImageROI.from_disk('path/to/your/folder')
    >>> ic = ImageClassified.from_parent(ir)
    >>> ic.set_seeds(plts=True, peak_prominence=.1)
    >>> ic.set_params_laminae_simplified()
    >>> ic.set_quality_score()
    or, doing it all in one step
    >>> ic.set_laminae_params_table()
    and save to disk
    >>> ic.save()
    which can then be loaded
    >>> ic = ImageClassified.from_disk('path/to/your/folder')
    View the results
    >>> ic.plot_overview()
    """

    def __init__(
            self,
            obj_color: str,
            path_folder: str | None = None,
            image: np.ndarray[float | int] | None = None,
            image_classification: np.ndarray[int] | None = None,
            image_type: str = 'cv',
            path_image_file: str | None = None
    ):
        """Initiator."""
        super().__init__(
            path_folder=path_folder,
            path_image_file=path_image_file,
            image=image,
            image_type=image_type,
            obj_color=obj_color
        )

        if image_classification is not None:
            self._image_classification = image_classification

    @classmethod
    def from_parent(cls, parent: ImageROI) -> Self:
        """Alternative constructor for instantiating an object from a parent ImageSample instance."""
        new: Self = cls(
            path_folder=parent.path_folder,
            image=parent.image,
            image_classification=parent.image_classification,
            path_image_file=None,
            obj_color=parent.obj_color
        )

        if hasattr(parent, 'age_span'):
            new.age_span = parent.age_span

        return new

    @property
    def image_classification(self) -> np.ndarray[int]:
        assert hasattr(self, '_image_classification'), 'Image classification has not been provided!'
        return self._image_classification.copy()

    @staticmethod
    def column_wise_average(image: np.ndarray, mask: np.ndarray[int | bool]):
        mask = mask.astype(bool)
        image = image.copy().astype(float)
        image *= mask.astype(int)  # mask hole values
        image[mask] = rescale_values(image[mask], 0, 1)  # normalize
        col_sum: np.ndarray[float] = image.sum(axis=0)
        valid_sum: np.ndarray[int] = mask.sum(axis=0)
        # exclude columns with no sample material
        mask_nonempty_col: np.ndarray[bool] = valid_sum > 0
        # divide colwise sum of c by number of foreground pixels
        # empty cols will have a value of 0
        av: np.ndarray[float] = np.zeros(image.shape[1])
        av[mask_nonempty_col] = col_sum[mask_nonempty_col] / valid_sum[mask_nonempty_col]
        # center
        av -= .5
        av[~mask_nonempty_col] = 0
        return av

    def set_seeds(
            self,
            in_classification: bool = True,
            peak_prominence: float = 0,
            hold=False,
            min_distance=None,
            plts: bool = False,
            **kwargs

    ) -> None:
        """Find peaks in col-wise averaged classification."""
        mask_foreground: np.ndarray[int] = self.mask_foreground
        horizontal_extent = mask_foreground.shape[0]

        if in_classification:
            image_light = self.image_classification.copy()
        else:
            image_light = self.image_grayscale.copy()

        brightness = self.column_wise_average(image_light, mask=self.mask_foreground)

        yearly_thickness = self.average_width_yearly_cycle
        if min_distance is None:
            min_distance = yearly_thickness / 4

        assert ~np.isnan(brightness).any(), 'brightness array contains nan'

        # find light and dark seeds spaced at least half a year apart
        seeds_light, props_light = scipy.signal.find_peaks(
            brightness,
            distance=min_distance,
            prominence=peak_prominence
        )
        width_light = scipy.signal.peak_widths(
            brightness, seeds_light)

        seeds_dark, props_dark = scipy.signal.find_peaks(
            -brightness,
            distance=min_distance,
            prominence=peak_prominence
        )
        width_dark = scipy.signal.peak_widths(
            -brightness, seeds_dark)

        if plts:
            # brightness and peaks
            fig, axs = plt.subplots(nrows=2, sharex=True)
            axs[0].plot(brightness, label='brightness', linewidth=.9)

            # dots on brightness plot
            axs[0].plot(seeds_light, brightness[seeds_light], 'ro', alpha=.5)
            axs[0].plot(seeds_dark, brightness[seeds_dark], 'mo', alpha=.5)

            # lines for widths
            axs[0].hlines(*width_light[1:], colors='r', alpha=.5)
            axs[0].hlines(*width_dark[1:], colors='m', alpha=.5)

            axs[0].set_xlim((0, len(brightness) - 1))

            axs[0].grid('on')
            axs[0].set_ylabel('light/(light + dark)')

            # axs[0].legend(bbox_to_anchor=(0, -0.2, 1, 0.2), loc="lower left",
            #               mode="expand", borderaxespad=0, ncol=3)

            axs[1].imshow(image_light, aspect='auto', interpolation='none', cmap='gray')
            # dots on image of classification
            axs[1].plot(seeds_light, horizontal_extent / 2 * (np.ones_like(seeds_light) + .05),
                        'ro', alpha=.5, label='seeds light', markersize=4)
            axs[1].plot(seeds_dark, horizontal_extent / 2 * (np.ones_like(seeds_dark) - .05),
                        'mo', alpha=.5, label='seeds dark', markersize=4)

            fig.suptitle(f'detected peaks (light: {len(seeds_light)}, \
dark: {len(seeds_dark)}) \n with prominence greater than {peak_prominence}.')

            fig.tight_layout()
            if not hold:
                plt.show()
            else:
                return fig, axs

        self.seeds_light = seeds_light
        self.seeds_dark = seeds_dark
        self.width_light = width_light[0]
        self.width_dark = width_dark[0]
        self.prominences_light = props_light['prominences']
        self.prominences_dark = props_dark['prominences']

    def get_seeds_above_prominence(self, peak_prominence):
        """Return seeds above given prominence for light and dark as tuple."""
        seeds_light = self.seeds_light[self.prominences_light > peak_prominence]
        seeds_dark = self.seeds_dark[self.prominences_dark > peak_prominence]
        return seeds_light, seeds_dark

    def set_params_laminae_simplified(
            self,
            peak_prominence: float = 0,
            height0_mode: str = 'use_peak_widths',
            downscale_factor: float = 1,
            **kwargs
    ) -> None:
        """Run optimizer to find layer for each peak above prominence."""
        image_classification = self.image_classification.copy()
        # get seeds above prominence
        seeds_light, seeds_dark = self.get_seeds_above_prominence(peak_prominence)

        if height0_mode == 'use_age_model':
            height0s_light = height0s_dark = self.average_width_yearly_cycle
        elif height0_mode == 'use_peak_widths':
            height0s_light = self.width_light.copy()
            height0s_dark = self.width_dark.copy()
        else:
            raise KeyError(f'height0_mode has to be one of use_peak_widths, \
use_age_model, not {height0_mode}')

        # downscale
        if downscale_factor != 1:
            image_classification = downscale_image(image_classification, downscale_factor)
            seeds_light = np.round(seeds_light * downscale_factor).astype(int)
            seeds_dark = np.round(seeds_dark * downscale_factor).astype(int)
            height0s_light = height0s_light * downscale_factor
            height0s_dark = height0s_dark * downscale_factor

        # find layers
        dataframe_params_light = find_layers(
            image_classification, seeds_light, height0s_light, color='light',
            **kwargs
        )
        dataframe_params_light['width'] = self.width_light
        dataframe_params_light['prominence'] = self.prominences_light

        dataframe_params_dark = find_layers(
            image_classification, seeds_dark, height0s_dark, color='dark',
            **kwargs
        )
        dataframe_params_dark['width'] = self.width_dark
        dataframe_params_dark['prominence'] = self.prominences_dark

        for attr in ['width_light', 'width_dark', 'prominences_light', 'prominences_dark']:
            self.__delattr__(attr)

        dataframe_params = pd.concat(
            [dataframe_params_light, dataframe_params_dark]
        ).sort_values(by='seed', ignore_index=True)
        if downscale_factor != 1:
            dataframe_params['seed'] = (dataframe_params['seed'] / downscale_factor).round()
            dataframe_params['height'] = dataframe_params['height'] / downscale_factor

        # make sure data types are right
        dataframe_params = dataframe_params.astype(
            {'seed': int, 'a': float, 'b': float, 'c': float, 'd': float,
             'height': float, 'success': bool, 'color': str}
        )
        self.params_laminae_simplified = dataframe_params

    def get_region_from_params(self, idx):
        """Get the region from an index in params table."""
        width = self.image_classification.shape[0]
        row = self.params_laminae_simplified.iloc[idx, :]
        coeffs = row.iloc[1:4 + 1].to_numpy()
        height = row['height']
        layer_region = distorted_rect(width, height, coeffs).astype(np.int32)
        factor_c = 255  # factor of color
        factor_s = 1  # factor of signum
        if row.color == 'dark':
            factor_c = 127
            factor_s = -1
        return layer_region, factor_c, factor_s

    def get_region_in_image_from_params(self, image, idx):
        """Get the region in an image from params."""
        region_layer, _, _ = self.get_region_from_params(idx)
        width = image.shape[0]
        row = self.params_laminae_simplified.iloc[idx, :]
        slice_region = np.index_exp[:, row.seed: row.seed + width]
        return get_half_width_padded(image)[slice_region]

    def rate_quality_layer(
            self,
            idx: int,
            keys_classification: Iterable[int] | None = None,
            labels_classification: Iterable[str] | None = None,
            **kwargs
    ):
        """Calculate hom, cont, brightness for layer in self.params for idx."""
        if keys_classification is None:
            keys_classification = [key_light_pixels, key_dark_pixels, key_hole_pixels]
        if labels_classification is None:
            labels_classification = ['light', 'dark', 'hole']
        # get region of the layer
        region_layer, _, _ = self.get_region_from_params(idx)
        region_layer = region_layer.astype(bool)
        region_classification = self.get_region_in_image_from_params(
            self.image_classification, idx
        )
        region_grayscale = self.get_region_in_image_from_params(
            self.image_grayscale, idx
        )
        if any([val not in keys_classification for val in np.unique(region_classification)]):
            raise KeyError(f'values in classification array \
({np.unique(region_classification)}) do not \
match the passed classification keys ({keys_classification}).')
        labels_to_keys = dict(zip(labels_classification, keys_classification))

        # number of pixels classified as light in region
        mask_light = (region_classification == labels_to_keys['light'])
        mask_dark = (region_classification == labels_to_keys['dark'])
        sum_lights = np.sum(mask_light * region_layer)
        sum_darks = np.sum(mask_dark * region_layer)
        # only lights in layer --> hom = 1
        # only darks in layer --> hom = -1
        homogeneity = (sum_lights - sum_darks) / (sum_lights + sum_darks)
        # no holes --> 1, only holes --> 0
        mask_valid = region_layer & (mask_light | mask_dark)
        mask_extent = min_max_extent_layer(mask_valid)
        continuity = np.sum(mask_valid) / np.sum(mask_extent)
        # brightness: average grayscale intensity in region_layer
        #   --> values between 0 and 255
        # excluding holes
        brightness = np.sum(region_grayscale * mask_valid) / np.sum(mask_valid)

        return np.array([homogeneity, continuity, brightness])

    def set_quality_score(self, plts: bool = False):
        """Add quality and other props to params table."""

        def quality(homogeneity, continuity, contrast):
            return homogeneity * continuity * contrast

        def calc_contrasts(brightnesses):
            # contrast: luminance difference / average luminance
            # average luminance: brightness (neighbours + center) / 2
            # luminance difference: brightness middle - neighbours
            # for top / bottom only consider layer below / above as neighbour
            # value range contrast:
            #   max luminence diff: +/- 255
            #   --> contrast in [-2, 2]
            # so let's not take the average of center and neighbours but their sum
            # this changes contrast by factor of 2, therefore contrast in [-1, 1]
            # (Michelson contrast)
            contrast = np.zeros(params.shape[0])

            # apply reflecting boundary condition
            brightnesses = np.append(
                np.insert(brightnesses, 0, brightnesses[1]), brightnesses[-2]
            )
            for idx in range(1, N_layers + 1):
                neighbours = (brightnesses[idx - 1] + brightnesses[idx + 1]) / 2
                center = brightnesses[idx]
                sum_cn = center + neighbours
                contrast[idx - 1] = (center - neighbours) / sum_cn
            return contrast

        params = self.params_laminae_simplified
        N_layers = params.shape[0]

        # initiate temporary arrays for hom, cont, bright
        criteria_quality_columns = ['homogeneity', 'continuity', 'brightness']
        criteria_array = np.zeros((N_layers, len(criteria_quality_columns)))

        # iterate over laminae
        for idx in range(N_layers):
            # calculate homogeneity, continuity, brightness (in L)
            # later add contrast and quality
            criteria_quality_layer = self.rate_quality_layer(idx)
            criteria_array[idx, :] = criteria_quality_layer

        # add homogeneity, continuity, brightness, contrast, quality columns
        for idx, criterium in enumerate(criteria_quality_columns):
            params[criterium] = criteria_array[:, idx]

        params['contrast'] = calc_contrasts(params.brightness.to_numpy())
        params['quality'] = params.apply(
            lambda row:
            quality(row.homogeneity, row.continuity, row.contrast),
            axis=1
        )
        if plts:
            self.plt_quality()

    def set_laminae_images_from_params(
            self, ignore_conflicts=True, plts: bool = False
    ):
        """
        Create images with simplified laminae.

        (seed idx as value and light/dark as value), conflicts.
        """
        assert self.params_laminae_simplified is not None, \
            'create simplified laminae with \
simplify_laminae before calling create_simplified_laminae_classification.'

        image_classification = self.image_classification

        width = image_classification.shape[0]
        half_width = (width + 1) // 2

        # get padded versions of classification and grayscale
        image_classification_pad = get_half_width_padded(image_classification)

        # create image with idxs
        #   idx = seed * color
        image_seeds_pad = np.zeros(image_classification_pad.shape,
                                   dtype=np.int32)
        conflicts_pad = np.zeros(image_classification_pad.shape,
                                 dtype=bool)

        assert image_classification_pad.shape == image_seeds_pad.shape

        # iterate over laminae (start with lowest quality)
        for idx in self.params_laminae_simplified.quality.sort_values().index:
            seed = self.params_laminae_simplified.loc[idx, 'seed']
            layer_region, factor_c, factor_s = self.get_region_from_params(idx)
            mask_region = layer_region.astype(bool)

            # indizes in the padded images corresponding to current region
            slice_region = np.index_exp[:, seed: seed + width]

            # when the area we want to asign is already occupied
            conflict = (image_seeds_pad[slice_region] != 0) & mask_region
            image_seeds_slice = image_seeds_pad[slice_region]
            image_seeds_slice[mask_region] = factor_s * seed * layer_region[mask_region]
            # add conflicting pixels
            conflicts_pad[slice_region] |= conflict

        if not ignore_conflicts:
            # set conflicting pixels to 0
            image_seeds_pad[conflicts_pad] = 0

        # crop off the zero padding
        slice_original = np.index_exp[:, half_width:-half_width]

        self.image_seeds = image_seeds_pad[slice_original]
        conflicts = conflicts_pad[slice_original]
        # set background pixels to 0
        self.image_seeds *= self.mask_foreground
        conflicts &= self.mask_foreground.astype(bool)

        if plts:
            self.plt_image_seeds_and_classification()
            # pixels with conflict (conflict --> True)

            plt.imshow(conflicts, interpolation='none')
            plt.title('conflicts')
            plt.show()

    def get_image_expanded_laminae(self):
        assert hasattr(self, 'image_seeds'), 'call set_laminae_images_from_params'
        img = self.image_seeds
        img_e = expand_labels(img, distance=np.min(img.shape))
        img_e *= self.mask_foreground
        return img_e

    def get_image_simplified_classification(self):
        assert hasattr(self, 'image_seeds'), \
            'call set_laminae_images_from_params'

        isc = np.sign(self.image_seeds)
        isc[isc == 1] = key_light_pixels
        isc[isc == -1] = key_dark_pixels

        return isc

    def set_laminae_params_table(self, **kwargs):
        # set seeds with their prominences
        logger.info("setting seeds")
        self.set_seeds(**kwargs)
        # initiate params dataframe with seeds and params for distorted rects
        logger.info("finding distorted rects")
        self.set_params_laminae_simplified(**kwargs)
        # create output images for further analysis
        logger.info("creating image")
        # add quality criteria for each layer
        logger.info("calculating quality score")
        self.set_quality_score()
        # create classification image
        self.set_laminae_images_from_params()

    def plt_quality(self, take_abs=True, hold=False, fig=None, ax=None):
        params = self.params_laminae_simplified
        height_img, width_img = self.image_grayscale.shape
        # overview plot
        if fig is None:
            fig, axs = plt.subplots(nrows=2, sharex=True)
        else:
            axs = [ax]
            hold = True

        # if called from outside (e.g. ax provided), only quality criteria will be plotted
        is_outside_call = len(axs) != 2

        if take_abs:
            vecs = [params.homogeneity.abs(),
                    params.continuity,
                    params.contrast.abs(),
                    params.quality]
            labels = ['|homogeneity|', 'continuity', '|contrast|', 'quality']
        else:
            vecs = [params.homogeneity.abs(),
                    params.continuity,
                    params.contrast.abs(),
                    params.quality]
            labels = ['homogeneity', 'continuity', 'contrast', 'quality']
        xs = params.seed

        for vec, label in zip(vecs, labels):
            # scale
            s = np.max([1, np.floor(1 / vec.max())])
            axs[0].plot(xs,
                        vec * s,
                        label=label + r' $\cdot$ ' + str(int(s)),
                        linewidth=.9)
        axs[0].grid('on')
        axs[0].set_xlim((0, width_img))
        axs[0].legend(bbox_to_anchor=(0, 1, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=1 if is_outside_call else 3)

        # plt.imshow(self.image_grayscale)
        if not is_outside_call:
            axs[1].imshow(self.image_classification, interpolation='none', aspect='auto', cmap='gray')
            fig.tight_layout()

        if not hold:
            plt.show()
        else:
            return fig, axs

    def plt_image_seeds_and_classification(self):
        # image_simplified_classification
        isc = self.get_image_simplified_classification()

        # simplified light/dark/uncertain
        plt.subplot(211)
        plt.imshow(isc,
                   interpolation='none',
                   vmin=0, vmax=255
                   )
        plt.title('simplified classification')

        # different laminae (pos = light)
        plt.subplot(212)
        plt.imshow(self.image_seeds, interpolation='none', cmap='rainbow')
        plt.title('zones')
        plt.show()

    def plot_overview(self):
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, layout='constrained')

        plt_cv2_image(image=self.image, ax=axs[0, 0], fig=fig, title='Original image', no_ticks=True)
        plt_cv2_image(image=self.image_classification, ax=axs[0, 1], fig=fig, title='Classification input',
                      no_ticks=True)
        plt_cv2_image(image=self.get_image_simplified_classification(), ax=axs[1, 0], fig=fig,
                      title='Simplified classification', no_ticks=True)
        # returns quality where axs = [ax]
        fig, axs = self.plt_quality(fig=fig, ax=axs[1, 1])
        ax11 = axs[0]
        ax11.set_xlabel('depth (pixels)')
        ax11.set_ylabel('scaled criterion')
        plt.show()
