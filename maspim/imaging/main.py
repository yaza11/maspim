"""Main image module. Implements the ImageSample, ImageROI and ImageClassified classes."""
import skimage.transform
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
from typing import Iterable, Self, Any, Literal
from tqdm import tqdm

from maspim.imaging.interactive import InteractiveImage
from maspim.imaging.register.helpers import Mapper
from maspim.imaging.register.descriptor import Descriptor
from maspim.res.constants import key_light_pixels, key_dark_pixels, key_hole_pixels
from maspim.util.convenience import Convenience, check_attr
from maspim.imaging.misc.fit_distorted_rectangle import find_layers, distorted_rect
from maspim.imaging.misc.find_punch_holes import find_holes

import maspim.imaging.util.image_convert_types as image_convert_types
from maspim.imaging.util.coordinate_transformations import rescale_values
from maspim.imaging.util.image_convert_types import ensure_image_is_gray
from maspim.imaging.util.image_plotting import plt_cv2_image, plt_contours, plt_rect_on_image
from maspim.imaging.util.image_processing import (
    adaptive_mean_with_mask_by_rescaling,
    remove_outliers_by_median,
    func_on_image_with_mask,
    auto_downscaled_image,
    downscale_image, adaptive_mean_with_mask
)

from maspim.imaging.util.image_geometry import star_domain_contour

from maspim.imaging.util.image_helpers import (
    ensure_odd,
    get_half_width_padded,
    min_max_extent_layer,
    filter_contours_by_size,
    get_foreground_pixels_and_threshold, get_simplified_image, restore_unique_values
)

from maspim.imaging.util.image_boxes import get_mean_intensity_box, region_in_box

logger = logging.getLogger(__name__)


class Image(Convenience):
    """
    Base function to get sample images and analyze them.

    Can be used on its own for basic functionality, but generally not recommended.
    """
    _average_width_yearly_cycle: float | None = None
    _hw: tuple[int, int] | None = None
    _image: np.ndarray[int | float] | None = None
    _image_simplified: np.ndarray[int] | None = None
    _main_contour: np.ndarray[int] | None = None
    _mask_foreground: np.ndarray[bool] | None = None
    _thr_background: float | int = None

    age_span: tuple[float, float] | None = None
    image_file: str | None = None
    path_folder: str | None = None
    obj_color: str | None = None

    _save_attrs: set[str] = {
        'age_span',
        '_average_width_yearly_cycle',
        'image_file',
        '_image',
        'obj_color',
        '_mask_foreground'  # TODO: could alternatively save parameters with which to create mask
    }

    def __init__(
            self,
            obj_color: Literal['light', 'dark'],
            path_image_file: str | None = None,
            image: np.ndarray[float | int] | None = None,
            mask_foreground: np.ndarray | None = None,
            image_type: str = 'cv',
            path_folder: str | None = None
    ) -> None:
        """Initiator.

        Parameters
        ----------
        obj_color : str
            The foreground color of the object in the image. Either 'light' or
            'dark'. This is required for working with thresholded images is
            desired.
        path_image_file : str, optional
            The file path to an image file to be read.
        image : np.ndarray[float | int], optional
            Alternatively, an image can be provided directly.
        mask_foreground: np.ndarray, optional
            Mask specifying foreground pixels. Will be determined automatically
            using the obj_color if not provided.
        image_type: str, optional
            If the input image is not a cv image, provide this keyword argument.
            Options are 'cv', 'np', 'pil' for images read or processed with
            OpenCV, numpy or PILLOW respectively.
        path_folder : str, optional
            Folder in which the image or saved object is located. If not provided,
            will be inferred from path_image_file.
            If that is also not provided, will be an empty string.
        """
        assert (path_image_file is not None) or (image is not None), \
            "Must provide either path or image"
        image_types: tuple[str, ...] = ('cv', 'np', 'pil')
        assert image_type in image_types, (f'valid image types are {image_types},'
                                           f' depending on the source of the image')
        obj_colors: tuple[str, str] = ('light', 'dark')
        assert obj_color in obj_colors, f'valid object colors are {obj_colors}'

        if path_image_file is not None:
            image = cv2.imread(path_image_file)
            assert image is not None, f"Could not load image from {path_image_file}"
            self.image_file: str = os.path.basename(path_image_file)
            if path_folder is None:
                path_folder: str = os.path.dirname(path_image_file)

        self.obj_color: str = obj_color

        if mask_foreground is not None:
            assert mask_foreground.ndim == 2, f'mask must be 2D'
            assert len(np.unique(mask_foreground)) <= 2, \
                f'mask should contain at most 2 unique values'
            assert mask_foreground.shape == image.shape[:2], \
                (f'mask and image dimensions must match but found mask: '
                 f'{mask_foreground.shape} and image: {image.shape[:2]}')
            self._mask_foreground: np.ndarray[np.uint8] = (
                mask_foreground > 0
            ).astype(np.uint8) * 255
            self._thr_background: int = -1

        # set _image_original
        self._from_image(image, image_type)

        self.path_folder: str = path_folder if path_folder is not None else ''

    @property
    def path_image_file(self) -> str | None:
        """Compose the path of the image file from folder and image file."""
        # image file was not provided
        if not check_attr(self, 'image_file'):
            return
        # path_folder was not provided
        if not check_attr(self, 'path_folder'):
            return
        return os.path.join(self.path_folder, self.image_file)

    def _from_image(self, image: np.ndarray | None, image_type: str) -> None:
        """
        Set attributes from the image and type.

        This function ensures that the image is oriented horizontally,
        and sets the original image as a cv image.
        """
        image: np.ndarray[int | float] = image_convert_types.convert(
            image_type, 'cv', image.copy()
        )
        # make sure image is oriented horizontally
        h, w, *_ = image.shape
        if h > w:
            logger.info(
                'swapped axes of input image to ensure horizontal orientation'
            )
            # swapaxes returns a view by default
            image: np.ndarray[int | float] = image.copy().swapaxes(0, 1)
        self._hw = h, w
        self._image: np.ndarray[int | float] = image

    @classmethod
    def from_disk(cls, path_folder: str, tag: str | None = None) -> Self:
        """Load an image object from disk."""
        # initiate dummy object that provides all, albeit nonsensical, parameters
        dummy: Self = cls(path_folder=path_folder,
                          image=np.ones((3, 3)),
                          obj_color='light')
        dummy.load(tag)
        # load messes with _image, _image_original, the constructor can take care of that
        new: Self = cls(
            obj_color=dummy.obj_color,
            path_image_file=dummy.path_image_file,
            image=dummy.__dict__.get('_image'),
            image_type='cv',
            path_folder=path_folder,
        )

        # overwrite attributes
        dummy.__dict__.update(new.__dict__)

        return dummy

    @property
    def image(self) -> np.ndarray[int | float]:
        """Return a copy of the original image"""
        return self._image.copy()

    def _require_image_grayscale(self):
        if not check_attr(self, '_image_grayscale'):
            self._image_grayscale = ensure_image_is_gray(self.image)
        return self._image_grayscale

    @property
    def image_grayscale(self) -> np.ndarray:
        """Return a grayscale version of the original image."""
        # TODO: check if (and where) copy is necessary
        # everywhere else the instances themselves are returned
        return self._require_image_grayscale().copy()

    def set_foreground_thr_and_pixels(
            self, thr_method: str = 'otsu', plts: bool = False, **kwargs
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
        logger.debug(f'determining foreground pixels with {thr_method=}.')

        mask, thr = get_foreground_pixels_and_threshold(
            image=self._image,
            obj_color=self.obj_color,
            method=thr_method,
            plts=plts,
            **kwargs
        )
        self._thr_background: int | float = thr
        self._mask_foreground: np.ndarray[int] = mask
        if plts:
            plt_cv2_image(mask, 'Identified foreground pixels')

    def _require_foreground_thr_and_pixels(
            self, **kwargs
    ) -> tuple[float | int, np.ndarray[int]]:
        """Make sure the foreground mask and threshold exists before returning it"""
        if not check_attr(self, '_mask_foreground'):
            self.set_foreground_thr_and_pixels(**kwargs)
        return self._thr_background, self._mask_foreground

    @property
    def image_binary(self) -> np.ndarray[int]:
        return self._require_foreground_thr_and_pixels()[1]

    @property
    def mask_foreground(self) -> np.ndarray[int]:
        """A mask where foreground pixels are True and background pixels are False."""
        return self._require_foreground_thr_and_pixels()[1]

    @property
    def thr_foreground(self) -> int | float:
        """The global threshold where foreground are separated from background pixels."""
        return self._require_foreground_thr_and_pixels()[0]

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
            image for which to create binarisation.
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
            image: np.ndarray[int | float] = self.image
        if mask is None:
            mask: np.ndarray[bool] = self.mask_foreground

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
            image, mask, func_light, return_argument_idx=1
        ) == 255

        dark_pixels = func_on_image_with_mask(
            image, mask, func_dark, return_argument_idx=1
        ) == 255

        light_pixels = light_pixels.astype(np.uint8)
        dark_pixels = dark_pixels.astype(np.uint8)

        if plts:
            plt_cv2_image(light_pixels, 'identified light pixels')
            plt_cv2_image(dark_pixels, 'identified dark pixels')

        return light_pixels, dark_pixels

    def set_simplified_image(self, **kwargs) -> None:
        """
        Apply median filters and increasing scales to the binary image.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to pass to get_simplified image

        Notes
        -----
        Defines image_simplified.
        """
        self._image_simplified: np.ndarray[int] = get_simplified_image(
            self.image_binary, **kwargs
        )

    def _require_simplified_image(self, **kwargs) -> np.ndarray[int]:
        """Set the simplified image if it does not exist and then return it."""
        if not check_attr(self, '_image_simplified'):
            self.set_simplified_image(**kwargs)
        return self._image_simplified

    @property
    def image_simplified(self) -> np.ndarray[int]:
        """Fetch simplified image."""
        return self._require_simplified_image()

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

        Notes
        -----
        Defines _main_contour
            The contour surrounding the sample as an array where each row
            describes a point.
        """
        methods: tuple[str, ...] = (
            'take_largest', 'star_domain', 'filter_by_size', 'convex_hull'
        )
        assert method in methods, \
            f"{method=} is not an option. Valid options are {methods}"

        image_binary: np.ndarray[int] = self.image_simplified

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
            contour: np.ndarray[int] = np.concatenate(contours)

        # this method will convert the domain to proxystardomain
        elif method == 'star_domain':
            # combine all contours into one array
            contour: np.ndarray[int] = np.concatenate(contours)
            center_points_x: list[int] = [
                round(image_binary.shape[1] / 4 * (i + 1))
                for i in range(3)
            ]
            center_points_y: list[int] = [round(image_binary.shape[0] / 2)] * 3
            center_points: list[tuple[int, int]] = list(zip(
                center_points_x, center_points_y
            ))
            for center_point in center_points:
                contour: np.ndarray[int] = star_domain_contour(
                    contour=contour,
                    center_point=center_point,
                    smoothing_size=1,
                    plts=plts
                )

        elif method == 'convex_hull':
            contour: np.ndarray[int] = np.concatenate(contours)
            contour: np.ndarray[int] = cv2.convexHull(contour)
        else:
            raise KeyError(
                f"{method=} is not an option. Valid options are {methods}")

        if plts:
            plt_contours(
                contours=[contour],
                image=image_binary,
                title='main contour'
            )

        self._main_contour: np.ndarray[int] = contour

    def _require_main_contour(self, **kwargs) -> np.ndarray[int]:
        """Set contour, if necessary and return it."""
        if not check_attr(self, '_main_contour'):
            self.set_main_contour(**kwargs)
        return self._main_contour

    @property
    def main_contour(self) -> np.ndarray[int]:
        """Set contour, if necessary and return it."""
        return self._require_main_contour()

    def set_age_span(self, age_span: tuple[float | int, float | int]) -> None:
        """
        Set the age span of the sample as a tuple (in yrs).

        This allows a more precise definition of the kernel size used in the
        classification.

        Parameters
        ----------
        age_span : Iterable[float | int]
            2-tuple like. The age span of the sample covered in the image.
            Lower bound should come first.
        """
        assert age_span[0] < age_span[1], 'first value should be strictly lower'
        assert len(age_span) == 2, \
            'provide an upper and lower value (e.g. age_span=[0, 100])'
        self.age_span: tuple[float | int, float | int] = age_span

    def set_average_width_yearly_cycle(self, pixels: int | None = None) -> None:
        """
        Calculate how many cycles are in the interval and their av width.

        Parameters
        ----------
        pixels : int, optional
            The number of pixels covered by a cycle. If not provided, will be calculated
            from the age span.
        """
        if pixels is not None:
            self._average_width_yearly_cycle = pixels
            return
        assert check_attr(self, 'age_span'), \
            'call set_age_span'
        pixels_x: int = self.image.shape[1]
        # calculate the number of expected cycles from the age difference for
        # the depth interval of the slice
        self._average_width_yearly_cycle: float = pixels_x / abs(self.age_span[1] - self.age_span[0])

    @property
    def average_width_yearly_cycle(self) -> float:
        """Set and return the average width of a year in pixels."""
        if not check_attr(self, '_average_width_yearly_cycle'):
            assert check_attr(self, 'age_span'), \
                'Define an age span before calculating the average width of annual layers.'
            self.set_average_width_yearly_cycle()
        return self._average_width_yearly_cycle

    def plot(self, **kwargs) -> None | tuple[plt.Figure, plt.Axes]:
        """
        Plot the original image.

        Parameters
        ----------
        kwargs : dict
            Additional keywords passed on to plt_cv2_image.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None,
            Figure and Axes, if hold is set to True.
        """
        return plt_cv2_image(image=self.image, **kwargs)


class ImageSample(Image):
    """
    Define sample area.

    This function uses a multistep approach to find the sample area. Oftentimes
    it is necessary to define the object color, which tells the algorithms if
    the pixels of the samples are lighter than the background (in which case
    the 'obj_color' keyword should be set to 'light') or darker
    (obj_color='dark'). It is recommended to save

    Example Usage
    -------------
    >>> from maspim import ImageSample
    Create an ImageSample object from an image on disk
    >>> i = ImageSample(path_image_file="/path/to/your/file")
    in this case the object color will be infered, it is adviced to set it manually
    >>> i = ImageSample(path_image_file="/path/to/your/file",obj_color='light')
    or
    >>> i = ImageSample(path_image_file="/path/to/your/file",obj_color='dark')
    depending on your sample. The resolution of the image matters for downstream
    applications (e.g. combination of image with MSI measurement. Therefore,
    for MSI applications it is adviced to use the project class which takes
    care of finding the right image file (specified in the mis-file).
    Alternatively, one can load in a previously saved ImageSample instance by
    providing the folder path
    >>> i = ImageSample.from_disk("path/to/your/folder")
    or
    >>> i = ImageSample(path_image_file='path/to/your/file')
    >>> i.load()

    It is recommended to stick to the properties, e.g. image, image_grayscale,
    image_binary, image_simplified, main_contour. The most important property
    is the image_sample_area which is the final result of performing all the
    steps of finding the sample area, so initiating and checking a new instance
    could look like this:
    >>> from maspim import ImageSample
    >>> i = ImageSample(path_image_file="/path/to/your/file")
    >>> i.set_sample_area()
    >>> i.plot_overview()
    >>> i.save()
    """

    _save_attrs: set[str] = {
        'age_span',
        '_average_width_yearly_cycle',
        'image_file',
        '_image',
        'obj_color',
        '_xywh_ROI',
        '_hw',
        '_mask_foreground'
    }

    def __init__(
            self,
            *,
            path_folder: str | None = None,
            image: np.ndarray[float | int] | None = None,
            mask_foreground: np.ndarray | None = None,
            image_type: str = 'cv',
            path_image_file: str | None = None,
            obj_color: str | None = None
    ) -> None:
        """Initiator.

        Parameters
        ----------
        obj_color : str
           The foreground color of the object in the image. Either 'light' or 'dark'.
           This is required for working with thresholded images is desired.
        path_image_file : str, optional
           The file path to an image file to be read.
        image : np.ndarray[float | int], optional
           Alternatively, an image can be provided directly.
        mask_foreground: np.ndarray, optional
            Mask specifying foreground pixels. Will be determined automatically
            using the obj_color if not provided.
        image_type: str, optional
           If the input image is not a cv image, provide this keyword argument.
           Options are 'cv', 'np', 'pil' for images read or processed with
           OpenCV, numpy or PILLOW respectively.
        path_folder : str, optional
           Folder in which the image or saved object is located. If not provided,
           will be inferred from path_image_file.
           If that is also not provided, will be an empty string.

        """
        # super call
        super().__init__(
            path_folder=path_folder,
            path_image_file=path_image_file,
            image=image,
            mask_foreground=mask_foreground,
            image_type=image_type,
            obj_color='light'  # give a dummy, will be overwritten
        )

        # overwrite the obj color attribute of the super init method
        if obj_color is not None:
            assert obj_color in ['light', 'dark'], \
                'obj_color must be either "light" or "dark"!'
            self.obj_color: str = obj_color
        else:
            self.obj_color: str = self._get_obj_color()

    def _pre_save(self):
        # if image_file is defined, we don't need to store the original image
        if check_attr(self, 'path_image_file'):
            self._save_attrs.remove('_image')

    def _post_save(self):
        self._save_attrs.add('_image')

    def _post_load(self):
        if not check_attr(self, '_image'):
            assert check_attr(self, 'path_image_file'), \
                'loaded corrupted instance with neither image nor image_file'
            self._image = cv2.imread(self.path_image_file)

    def _get_obj_color(self, region_middleground: float = .8, **_) -> str:
        """
        Determine if middle-ground is light or dark by comparing averages.

        Parameters
        ----------
        image : uint8 array.
            The image to check.
        region_middleground: float between 0 and 1.
            The region defined as the middle ground in each direction from the
            center expressed as a fraction. The default is .8.

        Returns
        -------
        obj_color: str
            'light' if middle-ground is lighter than whole image
            and 'dark' otherwise.

        """
        image_gray: np.ndarray[int] = self.image_grayscale

        height, width = image_gray.shape[:2]
        # determine indizes of box
        idx_height_min: int = round((1 - region_middleground) * height)
        idx_height_max: int = round(region_middleground * height)
        idx_width_min: int = round((1 - region_middleground) * width)
        idx_width_max: int = round(region_middleground * width)
        image_region: int = image_gray[
            idx_height_min:idx_height_max,
            idx_width_min:idx_width_max
        ]

        # of values for 4 channels take first one
        # (only nonzero for grayscale img)
        image_region_mean: float = cv2.mean(image_region)[0]
        image_mean: float = cv2.mean(image_gray)[0]
        if image_region_mean > image_mean:
            obj_color: str = 'light'
        else:
            obj_color: str = 'dark'

        logger.info(f'obj appears to be {obj_color}')

        return obj_color

    def get_sample_area_box(
            self,
            dilate_factor: float = 1,
            plts: bool = False,
            extent_x: tuple[int, int] | None = None,
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
        extent_x : tuple[int, int], optional
            extent of the sample in the x direction (taken from mis file) as a tuple 
            where the first value is the left and the second the right bound.
        plts: bool, optional
            if True, will plot the box

        Returns
        -------
        _image_roi, xywh
            The image inside the box and corner as well as width and height of box.

        """

        def metric(x0: np.ndarray[float]) -> float:
            """
            Calculate the difference in pixel intensity for a specified box.

            Parameters
            ----------
            x0 : array-like 4-tuple
                The center and ratios of the box.

            Returns
            -------
            float
                negative absolute difference of inside and outside pixel
                intensities normed by the area of the box.

            """
            box_ratio_x, box_ratio_y, center_box_x, center_box_y = x0
            center_box: np.ndarray[int] = np.around(
                [center_box_x, center_box_y],
                0
            ).astype(int)
            mean_box, mean_rest = get_mean_intensity_box(
                image_downscaled,
                box_ratio_x=box_ratio_x,
                box_ratio_y=box_ratio_y,
                center_box=center_box
            )

            fraction_area: float = box_ratio_y * box_ratio_x
            return -np.abs(mean_box - mean_rest) * fraction_area

        def metric_width(x0: np.ndarray[float]) -> float:
            """
            Calculate the difference in pixel intensity for the height of a box.

            Parameters
            ----------
            x0 : array-like 2-tuple
                The center and ratios of the box in the y-direction.

            Returns
            -------
            float
                negative absolute difference of inside and outside pixel
                intensities normed by the area of the box.

            """
            box_ratio_y, center_box_y = x0
            center_box = (
                np.array([center_box_x, center_box_y]) + .5
            ).astype(int)
            mean_box, mean_rest = get_mean_intensity_box(
                image_downscaled,
                box_ratio_x=box_ratio_x,
                box_ratio_y=box_ratio_y,
                center_box=center_box
            )

            fraction_area: float = box_ratio_y * box_ratio_x
            return -np.abs(mean_box - mean_rest) * fraction_area

        image_binary: np.ndarray[int] = self.image_binary
        image_downscaled, scale_factor = auto_downscaled_image(image_binary)
        # initiate center_box
        middle_y: int = round(image_downscaled.shape[0] / 2)
        middle_x: int = round(image_downscaled.shape[1] / 2)

        logger.info('searching optimal parameters for box')

        if extent_x is None:
            x0: np.ndarray[float] = np.array([.5, .5, middle_x, middle_y])
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
        else:
            logger.info(f'got horizontal extent {extent_x}, only optimizing vertical extent')
            box_ratio_x: float = (extent_x[1] - extent_x[0]) / image_binary.shape[1]
            center_box_x: float = (extent_x[1] - extent_x[0]) / 2 + extent_x[0]
            center_box_x *= scale_factor
            x0: np.ndarray[float] = np.array([.5, middle_y])
            params = minimize(
                metric_width,  # function to minimize
                x0=x0,  # start values
                method='Nelder-Mead',  # method
                bounds=[  # bounds of parameters
                    (0, 1),
                    (0, image_downscaled.shape[1]),
                ]
            )
            # determined values
            box_ratio_y, center_box_y = params.x

        center_box: tuple[float, float] = (center_box_x, center_box_y)
        logger.info(f'found box with {params.x}')
        logger.info(f'solver converged: {params.success}')

        # get params of box from those determined by the optimizer
        box_params: dict[str, Any] = region_in_box(
            image=image_downscaled,
            center_box=center_box,
            box_ratio_x=box_ratio_x,
            box_ratio_y=box_ratio_y
        )
        if plts:
            plt_rect_on_image(image_downscaled,
                              box_params,
                              title='Detected ROI of sample',
                              **kwargs)

        # dilate the box slightly for finer sample definition
        if dilate_factor > 1:
            box_ratio_x *= dilate_factor
            box_ratio_y *= dilate_factor
        elif dilate_factor < 1:
            box_ratio_x += dilate_factor
            box_ratio_y += dilate_factor
        # calculate new box
        if dilate_factor != 1:
            box_params = region_in_box(image=image_downscaled,
                                       center_box=center_box,
                                       box_ratio_x=box_ratio_x,
                                       box_ratio_y=box_ratio_y)

        x: int = box_params['x']
        y: int = box_params['y']
        w: int = box_params['w']
        h: int = box_params['h']

        # scale image back to original resolution
        if (scale_factor != 1) and (scale_factor is not None):
            x: int = round(x / scale_factor)
            y: int = round(y / scale_factor)

            w: int = round(w / scale_factor)
            h: int = round(h / scale_factor)
        # select the ROI in the image with original scale
        image_ROI: np.ndarray = self.image[y:y + h, x:x + w].copy()

        if plts:
            plt_cv2_image(image_ROI, 'detected ROI')

        return image_ROI, (x, y, w, h)

    def get_sample_area_from_contour(
            self,
            plts: bool = False,
            **kwargs: Any
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Get the roi spanned by the main contour.

        It is recommended to use this function in conjunction with
        get_sample_area_box since the algorithm to find the main contour can
        fail if the sample material is split into multiple parts
        (see get_sample_area).

        Parameters
        ----------
        plts: bool, optional
            If True, will plot the identified bounding box on top of the image.
            The default is False.
        kwargs: Any, optional
            Keyword arguments for _require_main_contour.

        Returns
        -------
        _image_roi: np.ndarray
            The image section inside the ROI.
        x, y, w, h: tuple[int, int, int, int]
            The coordinates of the box.
        """
        contour = self._require_main_contour(**kwargs)

        x, y, w, h = cv2.boundingRect(contour)

        image_roi = self.image[y:y + h, x:x + w]

        if plts:
            plt_cv2_image(
                image_roi, 'detected ROI as bounding box of main contour'
            )

        return image_roi, (x, y, w, h)

    def set_sample_area(
            self,
            plts: bool = False,
            interactive: bool = False,
            extent_x: tuple[int, int] | None = None,
            **_
    ) -> None:
        """
        Find the sample area of a sample in multiple steps.

        Firstly, use the get_sample_area_box function  and dilate the result
        to get a rough estimate that definitely includes the entire sample.
        Then, find the contour of the simplified area and its bounding box.

        Parameters
        ----------
        plts: bool, optional
            Whether to plot the identified ROI. Default is False.
        interactive: bool, optional
            If True, opens window for user to define the sample area
        extent_x : tuple[int, int], optional
            extent of the sample in the x direction (taken from mis file) as a tuple
            where the first value is the left and the second the right bound.

        """
        if interactive:
            self._user_sample_area()
            # redefine foreground
            self.set_foreground_thr_and_pixels()
            self.set_simplified_image()
            return

        # find the rough region of interest with box
        # if extent_x is defined, skip dilation step
        df: float = .1 if extent_x is None else 1.
        image_box, (xb, yb, wb, hb) = self.get_sample_area_box(
            extent_x=extent_x,
            dilate_factor=df,
            plts=plts
        )
        # set as new image
        image_sub: ImageSample = ImageSample(
            image=image_box,
            obj_color=self.obj_color,
            mask_foreground=self.mask_foreground[yb:yb+hb, xb:xb+wb]
        )
        # set image simplified for contour to use
        image_sub._mask_foreground = image_sub.image_simplified
        # find the refined area as the _extent of the simplified binary image
        _, (xc, yc, wc, hc) = image_sub.get_sample_area_from_contour(
            method='filter_by_size', plts=plts
        )

        # stack the offsets of the two defined ROI's since the second ROI is
        # placed in the first one
        x = xc + xb
        y = yc + yb
        w = wc
        h = hc

        image_roi: np.ndarray = self.image[y: y + h, x: x + w].copy()

        if plts:
            plt_cv2_image(
                image_roi,
                'final ROI as defined by get_sample_area'
            )

        self._image_roi: np.ndarray = image_roi
        self._xywh_ROI: tuple[int, int, int, int] = (x, y, w, h)

    def _user_sample_area(self) -> None:
        """Set image ROI by points defined by user."""
        interactive_image: InteractiveImage = InteractiveImage(self.image, mode='rect')
        interactive_image.show()
        if len(interactive_image.x_data) != 2:
            logger.error('Please provide exactly two points')
            return
        xs: np.ndarray[int] = np.around(
            interactive_image.x_data, 0
        ).astype(int)
        ys: np.ndarray[int] = np.around(
            interactive_image.y_data, 0
        ).astype(int)

        x: int = min(xs)
        y: int = min(ys)
        w: int = max(xs) - min(xs)
        h: int = max(ys) - min(ys)

        self._xywh_ROI: tuple[int, int, int, int] = (x, y, w, h)
        self._image_roi: np.ndarray = self.image[y: y + h, x: x + w].copy()

    def get_sample_area_from_xywh(self):
        """Get the region of the image corresponding to sample area from xywh."""
        assert check_attr(self, '_xywh_ROI'), \
            'no roi found, call require_image_sample_area'
        image: np.ndarray = self.image
        x, y, w, h = self._xywh_ROI
        return image[y:y + h, x:x + w].copy()

    def require_image_sample_area(
            self, overwrite: bool = False, **kwargs
    ) -> tuple[np.ndarray[np.uint8], tuple[int, ...]]:
        """Set and return area of the sample in the image."""
        # does has _xywh_ROI and _image_ROI
        if overwrite:
            self.set_sample_area(**kwargs)
            return self._image_roi, self._xywh_ROI

        if (
                check_attr(self, '_xywh_ROI')
                and check_attr(self, '_image_roi')
        ):
            return self._image_roi, self._xywh_ROI

        if check_attr(self, '_xywh_ROI'):  # only has _xywh_ROI
            self._image_roi: np.ndarray = self.get_sample_area_from_xywh()
            return self._image_roi, self._xywh_ROI

        self.set_sample_area(**kwargs)
        return self._image_roi, self._xywh_ROI


    @property
    def xywh_ROI(self):
        return self.require_image_sample_area()[1]

    @property
    def image_sample_area(self) -> np.ndarray:
        """Return a copy of the sample area."""
        x, y, w, h = self.require_image_sample_area()[1]
        return self.image[y:y + h, x:x + w].copy()

    def plot_sample_area(self, **kwargs: Any) -> None | tuple[plt.Figure, plt.Axes]:
        """Plot the detected sample area as rectangle on original image."""
        assert check_attr(self, '_xywh_ROI'), \
            'call require_image_sample_area first'
        xb, yb, wb, hb = self._xywh_ROI

        return plt_rect_on_image(
            image=self.image,
            title=kwargs.pop('title', 'Detected sample area'),
            no_ticks=kwargs.pop('no_ticks', True),
            box_params=region_in_box(
                image=self.image_binary,
                x=xb,
                y=yb,
                w=wb,
                h=hb
            ),
            **kwargs
        )

    def plot_overview(self, **kwargs: Any) -> None:
        """Plot diagnostic images"""
        # original image
        image_box, (xb, yb, wb, hb) = self.get_sample_area_box(dilate_factor=0.1)
        # set as new image
        image_sub: ImageSample = ImageSample(image=image_box, obj_color=self.obj_color)

        fig, axs = plt.subplots(nrows=2, ncols=2, **kwargs)

        plt_cv2_image(
            fig=fig,
            ax=axs[0, 0],
            image=self.image,
            title='input image',
            swap_rb=True,
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
            swap_rb=True,
            hold=True,
            no_ticks=True
        )
        plt_cv2_image(
            fig=fig,
            ax=axs[1, 1],
            image=self.image_sample_area,
            title='final sample region', no_ticks=True,
            swap_rb=True
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
    >>> from maspim import ImageROI
    >>> i = ImageSample()
    >>> i.from_disk(...)
    >>> ir = ImageROI.from_parent(i)

    Otherwise, the instance can be initiated using the default constructor by either providing an image or a file to an image.
    For file management it is required to specify the path_folder, if path_image_file is not provided (otherwise the folder
    will be infered from the image file path). So initiating could look like this (obj_color is always required):
    >>> from maspim import ImageROI
    >>> ir = ImageROI(obj_color='light', path_image_file='path/to/your/image/file.png')
    or
    >>> your_image = np.random.random((100, 200))
    >>> ir = ImageROI(obj_color='light', image=your_image)

    Classify the foreground pixels works best if the age span is known, in which case the kernel size can be estimated
    to cover about 2 years:
    >>> ir.set_age_span((0, 100))  # sample covers ages from 0 to 100 yrs
    >>> ir.set_classification_adaptive_mean()
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

    _save_attrs: set[str] = {
        'age_span',
        '_average_width_yearly_cycle',
        'image_file',
        '_image',
        'obj_color',
        '_xywh_ROI',
        '_hw',
        '_image_classification',
        '_params',
        '_punchholes',
        '_punchhole_size',
        '_mask_foreground'
    }

    def __init__(
            self,
            obj_color: str | None = None,
            path_folder: str | None = None,
            image: np.ndarray[float | int] | None = None,
            mask_foreground: np.ndarray | None = None,
            has_no_holes: bool = False,
            image_type: str = 'cv',
            path_image_file: str | None = None,
            age_span: tuple[float | int, float | int] | None = None,
            **_
    ) -> None:
        """Initiator.

        Parameters
        ----------
        obj_color : str
            The foreground color of the object in the image. Either 'light' or 'dark'.
            This is required for working with thresholded images is desired.
        path_image_file : str, optional
            The file path to an image file to be read.
        image : np.ndarray[float | int], optional
            Alternatively, an image can be provided directly.
            mask_foreground: np.ndarray, optional
            Mask specifying foreground pixels. Will be determined automatically
            using the obj_color if not provided.
        image_type: str, optional
            If the input image is not a cv image, provide this keyword argument. Options are 'cv', 'np', 'pil'
            for images read or processed with OpenCV, numpy or PILLOW respectively.
        path_folder : str, optional
            Folder in which the image or saved object is located. If not provided,
            will be inferred from path_image_file.
            If that is also not provided, will be an empty string.
        age_span: tuple[float | int, float | int], optional
            The age span covered by the sample.
        has_no_holes: bool, optional
            If the sample does not have any holes, specifying the obj_color and
            mask_foreground is not necessary. In this case set this parameter
            to True.

        """
        if has_no_holes:
            # define dummies
            if obj_color is None:
                obj_color = 'light'
        else:
            assert obj_color is not None, \
                ('if the sample has holes, specifying the object color is '
                 'necessary to determine which areas are the background and '
                 'which is the sample. If this does not apply to your image, '
                 'set "has_no_holes=True"')

        super().__init__(
            image=image,
            image_type=image_type,
            mask_foreground=mask_foreground,
            path_image_file=path_image_file,
            path_folder=path_folder,
            obj_color=obj_color
        )

        if has_no_holes and (mask_foreground is None):
            self._mask_foreground = np.full_like(self.image_grayscale, 1, dtype=np.uint8)
            self._thr_background = 0

        self.age_span: tuple[float | int, float | int] | None = age_span

    @classmethod
    def from_parent(cls, parent: ImageSample, **kwargs) -> Self:
        """
        Alternative constructor for instantiating an object from a parent
        ImageSample instance.
        """
        image = parent.image_sample_area.copy()
        x, y, w, h = parent.xywh_ROI
        mask_foreground = parent.mask_foreground[y: y + h, x: x + w]

        new: Self = cls(
            path_folder=parent.path_folder,
            image=image,
            mask_foreground=mask_foreground,
            path_image_file=None,
            obj_color=parent.obj_color,
            **kwargs
        )

        if check_attr(parent, 'age_span') and ('age_span' not in kwargs):
            new.age_span = parent.age_span

        return new

    def _get_params_laminae_classification(
            self, image_gray_shape: tuple[int, ...], **kwargs
    ) -> dict[str, bool | int | float]:
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

        if not check_attr(self, 'age_span'):
            params['estimate_kernel_size_from_age_model'] = False

        # update kernel_size to potentially match ROI
        if params['estimate_kernel_size_from_age_model']:
            logger.info(
                'Estimating kernel size from age model (square with ' +
                '2x expected thickness of one year).'
            )
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

    def _get_preprocessed_for_classification(
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

        Parameters
        ----------
        plts : bool, optional
            If True, will plot the preprocessed image.
        kwargs: dict
            Optional keywords for _get_params_laminae_classification.

        Returns
        -------
        image_gray: np.ndarray
            The preprocessed single-channel image.
        mask_foreground: np.ndarray[int]
            A mask of foreground pixels.
        params : dict[str, bool | int | float]
            A dict specifying the parameters used.
        """
        image_gray: np.ndarray = self.image_grayscale

        # update params with kwargs
        params: dict[str, bool | int | float] = self._get_params_laminae_classification(
            image_gray.shape, **kwargs
        )
        if plts:
            plt_cv2_image(image_gray, 'input in grayscale')

        # get mask_foreground matching image_gray
        # adaptive mean filter requires values between 0 and 1, so divide by max
        mask_foreground: np.ndarray[int] = (self.mask_foreground.copy() / self.mask_foreground.max()).astype(int)

        if plts:
            plt_cv2_image(
                mask_foreground,
                title=f'foreground pixels (thr={self.thr_foreground})'
            )

        # remove outliers
        if params['remove_outliers']:
            logger.info('Removing outliers with median filter.')
            image_gray = remove_outliers_by_median(
                image_gray,
                kernel_size_median=params['kernel_size_median'],
                threshold_replace_median=params['threshold_replace_median']
            )
            if plts:
                plt_cv2_image(image_gray, 'Outliers removed')

        if params['use_bilateral_filter']:
            logger.info('Applying bilateral filter.')
            image_gray: np.ndarray = cv2.bilateralFilter(
                image_gray,
                d=-1,
                sigmaColor=params['sigmaColor'],
                sigmaSpace=params['sigmaSpace']
            )
            if plts:
                plt_cv2_image(image_gray, 'Bilateral filter')

        return image_gray, mask_foreground, params

    @staticmethod
    def _get_postprocessed_image_from_classification(
            image_light: np.ndarray[int],
            mask_foreground: np.ndarray[int],
            params: dict[str, bool | int | float],
            plts: bool = False
    ) -> np.ndarray[int]:
        """
        Postprocess a classified image (remove small features).

        Parameters
        ----------
        image_light: np.ndarray[int]
            The input image to be cleaned.
        mask_foreground: np.ndarray[int]
            A mask specifying the foreground pixels.
        params: dict
            A dictionary defining parameters for the cleaning routines.
        plts: bool, optional
            If True, will plot the cleaned classification. The default is False.

        Returns
        -------
        image_classification: np.ndarray[int]
            The cleaned image classification where each pixel has either
            a value corresponding to background, foreground light or
            foreground dark.

        Notes
        -----
        image_dark will be infered from image_light and mask_foreground.
        """
        if params['remove_small_areas']:
            logger.info('Removing small blobs.')
            # find contours
            contours_light, _ = cv2.findContours(
                image_light.astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            contours_light: np.ndarray[int] = filter_contours_by_size(
                contours_light, image_light.shape,
                threshold_size=params['threshold_size_contours'])
            # get their areas
            conts_areas: list[float] = []
            for contour in contours_light:
                conts_areas.append(cv2.contourArea(contour))
            conts_areas: np.ndarray[float] = np.array(conts_areas)
            # get pth percentile
            area_pth_percentile: float = np.percentile(
                conts_areas, params['area_percentile'])
            # calculate equivalent diameter
            diam_pth_percentile: float = np.sqrt(4 * area_pth_percentile / np.pi)
            # set the kernel to be 3 times the size of the diameter
            kernel_median: int = round(3 * diam_pth_percentile)
            # ensure oddity
            if not kernel_median % 2:
                kernel_median += 1
            kernel_median: int = np.min([kernel_median, 255])
            image_light: np.ndarray[int] = cv2.medianBlur(
                image_light.astype(np.uint8), kernel_median
            )

        image_light_laminae: np.ndarray[np.uint8] = (
                image_light & mask_foreground).astype(np.uint8)
        image_dark_laminae: np.ndarray[np.uint8] = (
                mask_foreground & (~image_light)).astype(np.uint8)
        image_classification: np.ndarray[np.uint8] = (
            image_light_laminae * key_light_pixels +
            image_dark_laminae * key_dark_pixels
        )

        if plts:
            plt_cv2_image(
                image_classification, 'final classification')

        return image_classification

    def set_classification_adaptive_mean(
            self,
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Set the classified image (light, dark, background).

        Parameters
        ----------
        plts: bool, optional
            Whether to plot the final image.
        kwargs: dict, optional
            Additional keywords for pre- and postprocessing.
        """
        image_gray, mask_foreground, params = \
            self._get_preprocessed_for_classification(**kwargs)

        # adaptive thresholding
        logger.info('adaptive thresholding with mask')

        image_light: np.ndarray = adaptive_mean_with_mask_by_rescaling(
            image=image_gray,
            maxValue=1,
            thresholdType=cv2.THRESH_BINARY,
            ksize=(params['kernel_size_adaptive'], params['kernel_size_adaptive']),
            C=0,
            mask_nonholes=mask_foreground
        )
        image_light *= mask_foreground.astype(bool)

        image_classification = \
            self._get_postprocessed_image_from_classification(
                image_light,
                mask_foreground,
                params
            )

        if plts:
            plt_cv2_image(image_classification, title='classified image')

        self._image_classification: np.ndarray[np.uint8] = image_classification
        self._params: dict[str, bool | int | float] = params

    def set_classification_varying_kernel_size(
            self,
            scaling=2,
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Binarize foreground pixels by taking the median of adaptive mean
        classifications across different scales.

        Parameters
        ----------
        scaling: float
            controls the downscaling factor from step to step. Defaults to 2.
        plts: bool, optional
            If True, will plot the resulting classification.
        kwargs: dict, optional
            Optional keyword arguments to pass to the pre- and postprocessing
            functions.
        """
        image, mask, params = \
            self._get_preprocessed_for_classification(**kwargs)
        height, width = image.shape[:2]

        if self.obj_color == 'dark':
            threshold_type = cv2.THRESH_BINARY_INV
        else:
            threshold_type = cv2.THRESH_BINARY

        i_max: int = int(np.emath.logn(scaling, height))
        heights: np.ndarray[float] = height / scaling ** np.arange(i_max)
        # get last index where there are more than sixteen vertical pixels
        i_max: int = np.arange(i_max)[heights > 16][-1]

        res: np.ndarray[bool] = np.zeros((height, width, i_max), dtype=bool)

        for it in tqdm(range(i_max)):
            # downscale
            new_width: int = round(width / scaling ** it)
            new_height: int = round(height / scaling ** it)
            dim: tuple[int, int] = (new_width, new_height)
            image_downscaled: np.ndarray = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            mask_downscaled: np.ndarray = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)
            image_downscaled: np.ndarray[np.uint8] = rescale_values(image_downscaled, 0, 255).astype(np.uint8)
            # filter
            image_filtered: np.ndarray[np.uint8] = adaptive_mean_with_mask(
                src=image_downscaled,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=threshold_type,
                blockSize=3,
                C=0,
                mask=mask_downscaled
            ) * mask_downscaled
            # upscale
            image_rescaled: np.ndarray = cv2.resize(image_filtered, (width, height), interpolation=cv2.INTER_NEAREST)
            res[:, :, it] = image_rescaled.astype(bool)

        res: np.ndarray[bool] = np.median(res, axis=-1).astype(bool)
        mask: np.ndarray[bool] = mask.astype(bool)

        image_light: np.ndarray[np.uint8] = (
                res & mask).astype(np.uint8)
        # image_dark = (
        #         mask & (~res)).astype(np.uint8)
        # image_classification = image_light * key_light_pixels + \
        #                        image_dark * key_dark_pixels

        image_classification: np.ndarray[np.uint8] = \
            self._get_postprocessed_image_from_classification(
                image_light,
                mask,
                params
            )

        self._image_classification: np.ndarray[np.uint8] = image_classification
        self._params: dict[str, bool | int | float] = params | {'scaling': scaling}

        if plts:
            plt_cv2_image(image_classification, title='classified image')

    def require_classification(
            self, overwrite: bool = False, **kwargs
    ) -> np.ndarray[np.uint8]:
        """Create and return the image classification with parameters."""
        if not check_attr(self, '_image_classification') or overwrite:
            if not check_attr(self, 'age_span'):
                logger.warning(
                    'No age span specified, falling back to more general method'
                )
                self.set_classification_varying_kernel_size(**kwargs)
            else:
                self.set_classification_adaptive_mean(**kwargs)

        return self._image_classification

    @property
    def image_classification(self) -> np.ndarray[np.uint8]:
        """Create and return the image classification with parameters."""
        return self.require_classification()

    def set_punchholes(
            self,
            remove_gelatine: bool = True,
            interactive: bool = False,
            **kwargs
    ) -> None:
        """
        Add punch-holes to the current instance.

        Parameters
        ----------
        remove_gelatine: bool
            Try to remove gelatin residuals by masking foreground values
            with the simplified image since the simplified
            image usually does not contain the gelatin.
            If false, uses the binary image.
        interactive: bool, optional
            If this is set to True, will open a window for the user to click
            on the punch-holes.
        kwargs: Any, optional
            Extra keywords to be passed on to find_holes.
        """
        if interactive:
            self._user_punchholes()
            return

        # need copy, otherwise mask in ImageROI will be modified
        img: np.ndarray[np.uint8] = self.image_binary.copy()
        if remove_gelatine:
            img *= self.image_simplified

        kwargs_ = kwargs.copy()
        if 'obj_color' in kwargs_:
            kwargs_.pop('obj_color')

        self._punchholes, self._punchhole_size = find_holes(
            img,
            obj_color='light',  # image binary takes object color into account
            **kwargs_
        )

    def _user_punchholes(self) -> None:
        interactive_image: InteractiveImage = InteractiveImage(
            image_convert_types.swap_RB(self.image), mode='punchholes'
        )
        interactive_image.show()

        self._punchholes: tuple[tuple[int, int]] = tuple(
            zip(interactive_image.y_data, interactive_image.x_data)
        )
        self._punchhole_size: int = interactive_image._punchhole_size

    def require_punchholes(
            self, *args, overwrite: bool = False, **kwargs
    ) -> tuple[list[np.ndarray[int]] | tuple[tuple[int, int]], float]:
        if (not check_attr(self, "_punchholes")) or overwrite:
            self.set_punchholes(*args, **kwargs)
        return self._punchholes, self._punchhole_size

    @property
    def punchholes(
            self
    ) -> list[np.ndarray[int]] | tuple[tuple[int, int]]:
        if not check_attr(self, '_punchholes'):
            self.require_punchholes()
        return self._punchholes

    def plot_punchholes(
            self,
            image_name: str = 'image_grayscale',
            fig: plt.Figure | None = None,
            axs: Iterable[plt.Axes] | None = None,
            hold=False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        image = self.__getattribute__(image_name)

        if fig is None:
            assert axs is None, "If ax is provided, must also provide fig"
            fig, axs = plt.subplots()
        else:
            assert axs is not None, "If fig is provided, must also provide ax"

        hole_size: int = round(self._punchhole_size)

        _, ax11 = plt_rect_on_image(
            fig=fig,
            ax=axs,
            image=image,
            box_params=region_in_box(
                image=self.image_binary,
                point_topleft=np.array(self._punchholes[0])[::-1] - hole_size / 2,
                point_bottomright=np.array(self._punchholes[0])[::-1] + hole_size / 2
            ),
            no_ticks=True,
            hold=True
        )
        fig, axs = plt_rect_on_image(
            fig=fig,
            ax=ax11,
            image=image,
            box_params=region_in_box(
                image=self.image_binary,
                point_topleft=np.array(self._punchholes[1])[::-1] - hole_size / 2,
                point_bottomright=np.array(self._punchholes[1])[::-1] + hole_size / 2
            ), no_ticks=True,
            title='Detected punch-holes',
            hold=True
        )

        if hold:
            return fig, axs

        plt.show()

    def plot_overview(
            self,
            fig: plt.Figure | None = None,
            axs: Iterable[plt.Axes] | None = None,
            hold=False
    ) -> None | tuple[plt.Figure, Iterable[plt.Axes]]:
        """Plot the original, preprocessed, classified image and the punchholes."""
        if not check_attr(self, "_punchholes"):
            logger.warning(
                "Punchholes not set with required parameter 'remove_gelatine', "
                "this may affect performance."
            )
        self.require_punchholes(remove_gelatine=True)
        hole_size: int = round(self._punchhole_size)

        if fig is None:
            assert axs is None, "If ax is provided, must also provide fig"
            fig, axs = plt.subplots(nrows=2, ncols=2, layout="constrained")
        else:
            assert axs is not None, "If fig is provided, must also provide ax"

        # will plot only the binary image with punch-holes
        axs = np.array(axs)
        only_final = axs.shape == (1,)

        if not only_final:
            plt_cv2_image(fig=fig,
                          ax=axs[0, 0],
                          image=self.image,
                          title="Input image",
                          no_ticks=True)
            plt_cv2_image(fig=fig,
                          ax=axs[0, 1],
                          image=self._get_preprocessed_for_classification()[0],
                          title="Preprocessed image",
                          no_ticks=True)
            plt_cv2_image(fig=fig,
                          ax=axs[1, 0],
                          image=self.image_classification,
                          title="Classified image",
                          no_ticks=True)

        _, ax11 = plt_rect_on_image(
            fig=fig,
            ax=axs[0] if only_final else axs[1, 1],
            image=self.image_classification if only_final else self.image_binary,
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
            image=self.image_classification if only_final else self.image_binary,
            box_params=region_in_box(
                image=self.image_binary,
                point_topleft=np.array(self._punchholes[1])[::-1] - hole_size / 2,
                point_bottomright=np.array(self._punchholes[1])[::-1] + hole_size / 2
            ), no_ticks=True,
            title='Detected punch-holes'
        )

        if hold:
            return fig, axs

        plt.show()


class ImageClassified(Image):
    """
    Characterise and modify the classified layers.

    Example Usage
    -------------
    >>> from maspim import ImageClassified, ImageROI
    initiate from parent object, most common use case.
    >>> ir = ImageROI.from_disk('path/to/your/folder')
    >>> ic = ImageClassified.from_parent(ir)
    >>> ic.set_seeds(peak_prominence=.1,plts=True)
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

    _image_classification: np.ndarray = None
    _image_classification_corrected: np.ndarray = None
    _image_corrected: np.ndarray = None

    _seeds_light: np.ndarray[int] = None
    _seeds_dark: np.ndarray[int] = None
    _width_light: np.ndarray[float] = None
    _width_dark: np.ndarray[float] = None
    _prominences_light: np.ndarray[float] = None
    _prominences_dark: np.ndarray[float] = None

    _qualities: np.ndarray[float] = None

    image_seeds: np.ndarray[int] = None
    params_laminae_simplified: pd.DataFrame = None

    _save_attrs: set[str] = {
        'age_span',
        '_average_width_yearly_cycle',
        'image_file',
        '_image',
        'obj_color',
        '_xywh_ROI',
        '_hw',
        'params_laminae_simplified',
        'image_seeds',
        '_image_classification',
        '_mask_foreground'
    }

    def __init__(
            self,
            obj_color: Literal['light', 'dark'] = None,
            path_folder: str = None,
            image: np.ndarray[float | int] = None,
            mask_foreground: np.ndarray = None,
            has_no_holes: bool = False,
            image_classification: np.ndarray[int] = None,
            image_type: str = 'cv',
            path_image_file: str = None,
            age_span: tuple[float | int, float | int] = None,
            use_tilt_correction: bool = True,
            **_
    ):
        """Initiator.

        Parameters
        ----------
        obj_color : str
           The foreground color of the object in the image. Either 'light' or 'dark'.
           This is required for working with thresholded images is desired.
        path_image_file : str, optional
           The file path to an image file to be read.
        image : np.ndarray[float | int], optional
           Alternatively, an image can be provided directly.
        mask_foreground: np.ndarray, optional
            Mask specifying foreground pixels. Will be determined automatically
            using the obj_color if not provided.
        image_type: str, optional
           If the input image is not a cv image, provide this keyword argument.
           Options are 'cv', 'np', 'pil' for images read or processed with
           OpenCV, numpy or PILLOW respectively.
        path_folder : str, optional
           Folder in which the image or saved object is located. If not provided,
           will be inferred from path_image_file.
           If that is also not provided, will be an empty string.
        age_span: tuple[float | int, float | int], optional
            The age span covered by the sample.
        use_tilt_correction: bool, optional
            If this is set to True, the input images will be transformed such
            that laminae are roughly distortion free. This allows to define
            laminae solemnly by their width. The image_seeds is transformed
            back such that downstream applications remain unaffected by this
            parameter.
        has_no_holes: bool, optional
            If the sample does not have any holes, specifying the obj_color and
            mask_foreground is not necessary. In this case set this parameter
            to True.
        """
        if has_no_holes:
            # define dummies
            if obj_color is None:
                obj_color = 'light'
        else:
            assert obj_color is not None, \
                ('if the sample has holes, specifying the object color is '
                 'necessary to determine which areas are the background and '
                 'which is the sample. If this does not apply to your image, '
                 'set "has_no_holes=True"')

        super().__init__(
            path_folder=path_folder,
            path_image_file=path_image_file,
            image=image,
            mask_foreground=mask_foreground,
            image_type=image_type,
            obj_color=obj_color
        )

        if has_no_holes and (mask_foreground is None):
            self._mask_foreground = np.full_like(self.image_grayscale, 1, dtype=np.uint8)
            self._thr_background = 0

        if image_classification is not None:
            assert image_classification.shape[:2] == self._image.shape[:2], (
                'image_classification and image should have the same shape ' +
                'along the first two axes' +
                f'but have shapes {image_classification.shape[:2]}' +
                f' and {self._image.shape[:2]}'
            )
            self._image_classification: np.ndarray[int] = image_classification

        self.age_span: tuple[float | int, float | int] | None = age_span
        self.use_tilt_correction: bool = use_tilt_correction

    @classmethod
    def from_parent(cls, parent: ImageROI, **kwargs) -> Self:
        """
        Alternative constructor for instantiating an object from a parent
        ImageSample instance.
        """
        new: Self = cls(
            path_folder=parent.path_folder,
            image=parent.image,
            mask_foreground=parent.mask_foreground,
            image_classification=parent.image_classification,
            path_image_file=None,
            obj_color=parent.obj_color,
            **kwargs
        )

        if check_attr(parent, 'age_span'):
            new.age_span = parent.age_span

        return new

    @staticmethod
    def column_wise_average(
            image: np.ndarray, mask: np.ndarray[int | bool]
    ) -> np.ndarray[float]:
        """
        Calculate the column-wise (y-direction) average intensity.

        Parameters
        ----------
        image : np.ndarray
            The image for which to calculate the column-wise averages.
        mask : np.ndarray[int | bool]
            Pixels to exclude from the mean values.

        Returns
        -------
        av: np.ndarray
            Vector of averages with same length as shape of the input image
            in the x-direction.
        """
        mask: np.ndarray[bool] = mask.astype(bool)
        image: np.ndarray[float] = image.copy().astype(float)
        image *= mask.astype(int)  # mask hole values
        image[mask] = rescale_values(image[mask], 0, 1)  # normalize
        col_sum: np.ndarray[float] = image.sum(axis=0)
        valid_sum: np.ndarray[int] = mask.sum(axis=0)
        # exclude columns with no sample material
        mask_nonempty_col: np.ndarray[bool] = valid_sum > 0
        # divide colwise sum of c by number of foreground pixels
        # empty cols will have a value of 0
        av: np.ndarray[float] = np.zeros(image.shape[1])
        av[mask_nonempty_col] = (col_sum[mask_nonempty_col]
                                 / valid_sum[mask_nonempty_col])
        # center
        av -= .5
        av[~mask_nonempty_col] = 0
        return av

    def _set_corrected_using_mapper(self, mapper: Mapper) -> None:
        """Set the tilt corrected images using a transformation object."""
        self._image_corrected: np.ndarray = mapper.fit(
            self._image, preserve_range=True)
        ic_corrected: np.ndarray[int] = mapper.fit(
            self._image_classification, preserve_range=True
        )
        # restore original values
        self._image_classification_corrected = restore_unique_values(
            ic_corrected, (0, 127, 255)
        )
        self._tilt_descriptor: Mapper = mapper

    def get_descriptor(
            self,
            nx_pixels_downscaled: int = 500,
            **kwargs
    ) -> Descriptor:
        """
        Get a descriptor object that is initialized with a downscaled image. Descriptor objects are used to find
        the tilt correction.

        Parameters
        ----------
        nx_pixels_downscaled: int, optional
            The number of pixels in the x-direction provided to the descriptor.
            For memory efficiency it is recommended to downscale the image. The
            default is 500 (so image provided to Descriptor has 500 pixels in
            the x-direction).
        kwargs: Any
            Additional keywords for descriptor initialization and fit.

        Returns
        -------

        """
        # downscaled image has at most nx_pixels_downscaled pixels in x-direction
        downscale_factor: float = min((
            nx_pixels_downscaled / self._image.shape[1],
            1
        ))
        downscaled_shape = (
            round(self._image.shape[0] * downscale_factor),
            round(self._image.shape[1] * downscale_factor)
        )
        image_downscaled: np.ndarray = skimage.transform.resize(
            self._image, downscaled_shape
        )

        logger.info(
            f'initializing descriptor with image of shape '
            f'{image_downscaled.shape} (instead of {self._image.shape})'
        )
        # TODO: set max_size and min_size from age model, if available
        descriptor = Descriptor(image=image_downscaled, **kwargs)

        descriptor.set_conv()
        descriptor.fit(**kwargs)

        return descriptor

    def set_corrected_images(
            self,
            descriptor: Descriptor = None,
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Use Descriptor class to find tilts of layers in the image and set
        corrected images.

        Parameters
        ----------
        descriptor: Descriptor, optional
            Descriptor used for the tilt correction
        plts: bool, optional
            If this option is set to True, plot the kerrnels, kernels on image,
            parameter images, angles and corrected image of the Descriptor.
        kwargs: dict, optional
            Keyword arguments for initialization and setting the fit of
            desciptor.
        """
        assert check_attr(self, '_image_classification'), \
            'image_classification has not been provided'

        mapper = Mapper(self._image.shape, self.path_folder, 'tilt_correction')

        logger.info('getting new tilt correction transformation')

        if descriptor is None:
            descriptor = self.get_descriptor(**kwargs)

        U = descriptor.get_shift_matrix(self._image.shape[:2])
        mapper.add_UV(U=U)
        mapper.save()

        self._set_corrected_using_mapper(mapper)

        # upscale and undistort values of descriptor as qualities
        self._qualities: np.ndarray[float] = mapper.fit(
            skimage.transform.resize(
                descriptor.vals,
                (self._image.shape[0], self._image.shape[1])
            ),
            preserve_range=True
        ).mean(axis=0)

        if plts:
            descriptor.plot_kernels()
            descriptor.plot_kernel_on_img()
            descriptor.plot_parameter_images()
            descriptor.plot_quiver()
            descriptor.plot_corrected()

    def require_corrected_images(
            self, overwrite: bool = False, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return tilt corrected RGB and classified image. This method will return existing tilt corrected images if
        attempt to load a mapping, if it exists already or create a new mapping if necessary."""

        # return an existing corrected image
        if check_attr(self, '_image_corrected') and (not overwrite):
            return (
                self._image_corrected.copy(),
                self._image_classification_corrected.copy()
            )

        # load a mapping, apply it and return the result
        mapper = Mapper(self._image.shape, self.path_folder, 'tilt_correction')
        if os.path.exists(mapper.save_file) and (not overwrite):
            logger.info('loading tilt correction transformation')
            mapper.load()
            self._set_corrected_using_mapper(mapper)
            return (
                self._image_corrected.copy(),
                self._image_classification_corrected.copy()
            )

        # nothing found, create a new mapping
        self.set_corrected_images(**kwargs)
        return (
            self._image_corrected.copy(),
            self._image_classification_corrected.copy()
        )

    @property
    def image_uncorrected(self) -> np.ndarray:
        return self._image

    @property
    def image_corrected(self) -> np.ndarray:
        return self.require_corrected_images()[0]

    @property
    def image(self) -> np.ndarray:
        return self.image_corrected if self.use_tilt_correction else self.image_uncorrected

    @property
    def image_classification_uncorrected(self) -> np.ndarray:
        return self._image_classification

    @property
    def image_classification_corrected(self) -> np.ndarray:
        return self.require_corrected_images()[1]

    @property
    def image_classification(self) -> np.ndarray:
        return (
            self.image_classification_corrected
            if self.use_tilt_correction
            else self.image_classification_uncorrected
        )

    def set_seeds(
            self,
            image: np.ndarray | None = None,
            in_classification: bool = True,
            peak_prominence: float = 0,
            min_distance: int | None = None,
            plts: bool = False,
            hold=False,
            **_
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """
        Find peaks in col-wise averaged classification.

        Parameters
        ----------
        image: np.ndarray, optional
            An image that matches the shape of the input image along the depth
            direction. If not provided, in_classification will be used to
            determine the image used for getting the brightness function.
        in_classification : bool, optional
            If True, will use the classified image to identify peaks. The
            default is True.
        peak_prominence : float, optional
            Minimum required prominence for peaks. The default is 0 (all peaks).
            Values should be between 0 and 1.
        min_distance : int, optional
            Minimum distance between peaks. None defaults to 1 / 4 the distance
            of a yearly cycle.
        plts: bool, optional
            If True, will plot the identified peaks. The default is False.
        hold: bool, optional
            If True, will return the created figure and axes.
        """
        mask_foreground: np.ndarray[int] = self.mask_foreground

        if image is not None:
            assert image.shape[1] == self.image.shape[1], \
                (f'input image must match the image provided on initialization,'
                 f'so expecting {self.image.shape[1]} but got {image.shape[1]}')
            image_light = image
            mask_foreground = np.ones_like(image_light, dtype=bool)
        elif in_classification:
            image_light: np.ndarray = self.image_classification.copy()
        else:
            image_light: np.ndarray = self.image_grayscale.copy()

        brightness: np.ndarray[float] = self.column_wise_average(
            image_light, mask=mask_foreground
        )

        horizontal_extent: int = image_light.shape[0]

        if check_attr(self, 'age_span') and (self.age_span is not None):
            yearly_thickness: float = self.average_width_yearly_cycle
        else:
            yearly_thickness: float = 12.  # results in min distance of 3 pixels
        if min_distance is None:
            min_distance: float = yearly_thickness / 4
        # min_distance must be at least 1
        min_distance = max(1., min_distance)

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
        width_dark: tuple[np.ndarray[float], ...] = scipy.signal.peak_widths(
            -brightness,
            seeds_dark
        )

        self._seeds_light: np.ndarray[int] = seeds_light
        self._seeds_dark: np.ndarray[int] = seeds_dark
        self._width_light: np.ndarray[float] = width_light[0]
        self._width_dark: np.ndarray[float] = width_dark[0]
        self._prominences_light: np.ndarray[float] = props_light['prominences']
        self._prominences_dark: np.ndarray[float] = props_dark['prominences']

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

    def reduce_to_n_factor_seeds(
            self, factor_above_age_span: float | None = None
    ) -> None:
        """
        After setting peaks we want to keep more than predicted but not all
        peaks
        """
        assert check_attr(self, 'age_span'), 'need an age span for this'
        if factor_above_age_span is None:
            factor_above_age_span = 1.5
        if factor_above_age_span < 1:
            logger.warning('a factor of less than 1 will keep less peaks than '
                           'predicted by age model')

        idx: int = int(
            (self.age_span[1] - self.age_span[0]) * factor_above_age_span
        )

        proms_light = self._prominences_light.copy()
        proms_light.sort()  # low to high
        if idx + 1 >= len(proms_light):
            thr_light = 0.
        else:
            thr_light = proms_light[idx]

        proms_dark = self._prominences_dark.copy()
        proms_dark.sort()
        if idx + 1 >= len(proms_dark):
            thr_dark = 0.
        else:
            thr_dark = proms_dark[idx]

        thr: float = max((thr_light, thr_dark))

        mask_light = self._prominences_light < thr
        mask_dark = self._prominences_dark < thr

        logger.info(f'Reducing number of layers to a factor of '
                    f'{factor_above_age_span}, so light: '
                    f'{len(self._prominences_light)} -> {mask_light.sum()}, '
                    f'dark: '
                    f'{len(self._prominences_dark)} -> {mask_dark.sum()}, ')

        self._seeds_light: np.ndarray[int] = self._seeds_light[mask_light]
        self._width_light: np.ndarray[float] = self._width_light[mask_light]
        self._prominences_light: np.ndarray[float] = self._prominences_light[mask_light]

        self._seeds_dark: np.ndarray[int] = self._seeds_dark[mask_dark]
        self._width_dark: np.ndarray[float] = self._width_dark[mask_dark]
        self._prominences_dark: np.ndarray[float] = self._prominences_dark[mask_dark]

    # TODO: option to fix layer widths
    def set_params_laminae_simplified(
            self,
            height0_mode: Literal['use_peak_widths', 'use_age_model'] = 'use_peak_widths',
            downscale_factor: float = 1,
            **kwargs
    ) -> None:
        """
        Run optimizer to find layer for each peak above prominence.

        Parameters
        ----------
        height0_mode: str, optional
            How to determine the initial guess for the thickness of layers.
            The default is 'use_peak_widths'. Another option is to use the
            thickness predicted by the age model (same for each layer) by
            specifying 'use_age_model'.
        downscale_factor: int, optional
            The downsampling applied before fitting the rectangle.
            The default is 1 (no downscaling). Acceptable values have to be
            bigger than or equal to 1.
        """
        def set_params_laminae(color: str) -> pd.DataFrame:
            is_light: bool = color == 'light'

            dataframe_params: pd.DataFrame = find_layers(
                image_classification,
                seeds_light if is_light else seeds_dark,
                height0s_light if is_light else height0s_dark,
                color=color,
                degree=pol_degree,
                **kwargs
            )
            dataframe_params['width'] = (
                self._width_light if is_light else self._width_dark
            )
            dataframe_params['prominence'] = (
                self._prominences_light if is_light else self._prominences_dark
            )
            return dataframe_params

        pol_degree = kwargs.pop('degree', 0 if self.use_tilt_correction else 3)

        image_classification: np.ndarray[int] = self.image_classification.copy()

        if height0_mode == 'use_age_model':
            height0s_light: float = self.average_width_yearly_cycle
            height0s_dark: float = self.average_width_yearly_cycle
        elif height0_mode == 'use_peak_widths':
            height0s_light: np.ndarray[float] = self._width_light.copy()
            height0s_dark: np.ndarray[float] = self._width_dark.copy()
        else:
            raise KeyError(f'height0_mode has to be one of use_peak_widths, \
use_age_model, not {height0_mode}')

        # downscale
        if downscale_factor != 1:
            image_classification: np.ndarray = downscale_image(
                image_classification, downscale_factor
            )
            seeds_light: np.ndarray[int] = np.round(
                self._seeds_light * downscale_factor
            ).astype(int)
            seeds_dark: np.ndarray[int] = np.round(
                self._seeds_dark * downscale_factor
            ).astype(int)
            height0s_light: np.ndarray[float] = height0s_light * downscale_factor
            height0s_dark: np.ndarray[float] = height0s_dark * downscale_factor
        else:
            seeds_light = self._seeds_light
            seeds_dark = self._seeds_dark

        # find layers
        dataframe_params_light: pd.DataFrame = set_params_laminae('light')

        dataframe_params_dark: pd.DataFrame = set_params_laminae('dark')

        # we don't need them anymore
        self._width_light = None
        self._width_dark = None
        self._prominences_light = None
        self._prominences_dark = None

        dataframe_params: pd.DataFrame = pd.concat(
            [dataframe_params_light, dataframe_params_dark]
        ).sort_values(by='seed', ignore_index=True)
        if downscale_factor != 1:
            dataframe_params['seed'] = (
                    dataframe_params['seed'] / downscale_factor
            ).round()
            dataframe_params['height'] = dataframe_params['height'] / downscale_factor

        # make sure data types are right
        dataframe_params: pd.DataFrame = dataframe_params.astype(
            {'seed': int, 'a': float, 'b': float, 'c': float, 'd': float,
             'height': float, 'success': bool, 'color': str}
        )

        self.params_laminae_simplified: pd.DataFrame = dataframe_params

    def _get_region_from_params(self, idx: int) -> tuple[np.ndarray[int], int, int]:
        """
        Get the region from an index in params table.

        Parameters
        ----------
        idx: int
            The index of the layer.

        Returns
        -------
        layer_region: np.ndarray[int]
            The distorted layer as a squre-shaped array.
        factor_c: int
            Classification value (either 255 or 127).
        factor_s: int
            Sign value (either 1 or -1).
        """
        width: int = self.image_classification.shape[0]
        row: pd.Series = self.params_laminae_simplified.iloc[idx, :]
        coeffs: np.ndarray[float] = row.iloc[1:4 + 1].to_numpy()
        height: float = row['height']
        layer_region: np.ndarray[int] = distorted_rect(width, height, coeffs).astype(np.int32)
        factor_c: int = 255  # factor of color
        factor_s: int = 1  # factor of signum
        if row.color == 'dark':
            factor_c: int = 127
            factor_s: int = -1
        return layer_region, factor_c, factor_s

    def _get_region_in_image_from_params(self, image: np.ndarray, idx: int) -> np.ndarray:
        """
        Get the region in an image from params.

        Parameters
        ----------
        image: np.ndarray
            Image in which to get the region viewed by the layer.
        idx: int
            Index of the layer.

        Returns
        -------
        region: np.ndarray
            Square shaped region of image around center of layer.
        """
        region_layer, _, _ = self._get_region_from_params(idx)
        width: int = image.shape[0]
        row: pd.Series = self.params_laminae_simplified.iloc[idx, :]
        slice_region = np.index_exp[:, row.seed: row.seed + width]

        region: np.ndarray = get_half_width_padded(image)[slice_region]
        # TODO: fix this properly, region_classification sometimes not sqaure shaped
        if region.shape[0] != region.shape[1]:
            logger.warning('zeropadded region to be square shape')
            logger.warning(f'shape before: {region.shape}')
            pad_length: int = region.shape[0] - region.shape[1]
            region: np.ndarray = np.pad(
                region,
                ((0, 0), (0, pad_length))
            )
            logger.info(f'shape after: {region.shape}')

        return region

    def _rate_quality_layer(
            self,
            idx: int,
            keys_classification: tuple[int, int, int] | None = None,
            **_
    ) -> np.ndarray[float]:
        """
        Calculate hom, cont, brightness for layer in self.params for idx.

        Parameters
        ----------
        idx: int
            Index of the layer.
        keys_classification: tuple[int], optional.
            Values classifying light, dark and hole pixels. The default is
            255, 127, 0

        Returns
        -------
        out: np.ndarray[float]
            Vector containing the homogeneity, continuity and brightness.

        """
        if keys_classification is None:
            keys_classification: tuple[int, int, int] = (
                key_light_pixels, key_dark_pixels, key_hole_pixels
            )

        labels_classification: tuple[str, str, str] = ('light', 'dark', 'hole')
        # get region of the layer
        region_layer, _, _ = self._get_region_from_params(idx)
        region_layer: np.ndarray[bool] = region_layer.astype(bool)
        region_classification: np.ndarray = self._get_region_in_image_from_params(
            self.image_classification, idx
        )
        region_grayscale: np.ndarray = self._get_region_in_image_from_params(
            self.image_grayscale, idx
        )
        if any([
                val not in keys_classification
                for val in np.unique(region_classification)
        ]):
            raise KeyError(
                f'values in classification array ({np.unique(region_classification)}) '
                f'do not match the passed classification keys ({keys_classification}).'
            )
        labels_to_keys: dict[str, int] = dict(
            zip(labels_classification, keys_classification)
        )

        # number of pixels classified as light in region
        mask_light: np.ndarray[bool] = (region_classification == labels_to_keys['light'])
        mask_dark: np.ndarray[bool] = (region_classification == labels_to_keys['dark'])
        sum_lights: float = np.sum(mask_light * region_layer)
        sum_darks: float = np.sum(mask_dark * region_layer)

        # catch edge case "empty layers"
        # this should mean that there are no nans, since sum_darks + sum_lights
        # should always be less than mask_extent (difference is the number of
        # hole pixels) and hence continuity and brightness are well-defined
        if sum_darks + sum_lights == 0:
            return np.array([0, 0, 0])

        # only lights in layer --> hom = 1
        # only darks in layer --> hom = -1
        homogeneity: float = (sum_lights - sum_darks) / (sum_lights + sum_darks)
        # no holes --> 1, only holes --> 0
        mask_valid: np.ndarray[bool] = region_layer & (mask_light | mask_dark)
        mask_extent: np.ndarray[bool] = min_max_extent_layer(mask_valid)

        continuity: float = np.sum(mask_valid) / np.sum(mask_extent)
        # brightness: average grayscale intensity in region_layer
        #   --> values between 0 and 255
        # excluding holes
        brightness: float = np.sum(region_grayscale * mask_valid) / np.sum(mask_valid)

        return np.array([homogeneity, continuity, brightness])

    def set_quality_score(self, plts: bool = False, **_) -> None:
        """
        Add quality and other props to params table.

        Parameters
        ----------
        plts: bool, optional
            If True, will plot the quality parameters.
        """

        def quality(homogeneity: float, continuity: float, contrast: float) -> float:
            return homogeneity * continuity * contrast

        def calc_contrasts(brightnesses: np.ndarray[float]) -> np.ndarray[float]:
            """
            For a brightness vector, calculate the contrasts of consecutive layers.

            Parameters
            ----------
            brightnesses: np.ndarray[float]
                Vector with intensities

            Returns
            -------
            contrasts: np.ndarray[float]
                Corresponding contrasts. The first value is 0 to make the
                input and output match in their lengths.

            Notes
            -----
            contrast: luminance difference / average luminance
            average luminance: brightness (neighbours + center) / 2
            luminance difference: brightness middle - neighbours
            for top / bottom only consider layer below / above as neighbour
            value range contrast:
              max luminance diff: +/- 255
              --> contrast in [-2, 2]
            so let's not take the average of center and neighbours but their sum
            this changes contrast by factor of 2, therefore contrast in [-1, 1]
            (Michelson contrast)
            """
            contrast: np.ndarray[float] = np.zeros(params.shape[0])

            # apply reflecting boundary condition
            brightnesses: np.ndarray[float] = np.append(
                np.insert(brightnesses, 0, brightnesses[1]),  # b|abc ...
                brightnesses[-2]  # ... def|e
            )
            for idx in range(1, N_layers + 1):
                neighbours: float = (brightnesses[idx - 1] + brightnesses[idx + 1]) / 2
                center: float = brightnesses[idx]
                sum_cn: float = center + neighbours
                contrast[idx - 1] = (center - neighbours) / sum_cn
            return contrast

        params: pd.DataFrame = self.params_laminae_simplified
        N_layers: int = params.shape[0]

        # initiate temporary arrays for hom, cont, bright
        criteria_quality_columns: list[str] = [
            'homogeneity', 'continuity', 'brightness'
        ]
        criteria_array: np.ndarray[float] = np.zeros(
            (N_layers, len(criteria_quality_columns))
        )

        # iterate over laminae
        for idx in tqdm(range(N_layers), desc='Calculating layer qualities'):
            # calculate homogeneity, continuity, brightness (in L)
            # later add contrast and quality
            criteria_quality_layer: np.ndarray[float] = self._rate_quality_layer(idx)
            criteria_array[idx, :] = criteria_quality_layer

        # add homogeneity, continuity, brightness, contrast, quality columns
        for idx, criterium in enumerate(criteria_quality_columns):
            params.loc[:, criterium] = criteria_array[:, idx]

        params.loc[:, 'contrast'] = calc_contrasts(params.brightness.to_numpy())
        params.loc[:, 'quality'] = params.apply(
            lambda row:
            quality(row.homogeneity, row.continuity, row.contrast),
            axis=1
        )
        if plts:
            self.plot_quality()

    def set_laminae_images_from_params(
            self, ignore_conflicts=True, plts: bool = False, **_
    ) -> None:
        """
        Create images with simplified laminae.

        This function iterates through the laminae, starting with the lowest
        quality.

        Parameters
        ----------
        ignore_conflicts: bool, optional
            Whether to ignore conflicting pixels (those that are assigned to
            more than one laminae in regions of intersect). Defaults to True.
            If False, conflicting pixels will be excluded from layers.
        plts: bool, optional
            Whether to plot the classified and conflict image.

        """
        assert self.params_laminae_simplified is not None, \
            ('create simplified laminae with simplify_laminae before calling ' +
             'create_simplified_laminae_classification.')

        image_classification: np.ndarray[int] = self.image_classification

        width: int = image_classification.shape[0]
        half_width: int = (width + 1) // 2

        # get padded versions of classification and grayscale
        image_classification_pad: np.ndarray[int] = get_half_width_padded(
            image_classification
        )

        # create image with idxs
        #   idx = seed * color
        image_seeds_pad: np.ndarray[int] = np.zeros(
            image_classification_pad.shape,
            dtype=np.int32
        )
        conflicts_pad: np.ndarray[bool] = np.zeros(
            image_classification_pad.shape,
            dtype=bool
        )

        assert image_classification_pad.shape == image_seeds_pad.shape, \
            ('image_classification shape does not match. This can happen if ' +
             'you use a newer ImageROI instance.')

        N: int = self.params_laminae_simplified.shape[0]
        # iterate over laminae (start with the lowest quality)
        for idx in tqdm(
                self.params_laminae_simplified.quality.sort_values().index,
                total=N,
                desc='Setting simplified laminae',
                smoothing=50/N
        ):
            seed: pd.Series = self.params_laminae_simplified.loc[idx, 'seed']
            layer_region, factor_c, factor_s = self._get_region_from_params(idx)
            mask_region: np.ndarray[bool] = layer_region.astype(bool)

            # indices in the padded images corresponding to current region
            slice_region = np.index_exp[:, seed: seed + width]

            # sometimes there are out of bounds issues
            width_slice: int = image_seeds_pad[slice_region].shape[1]
            mask_region: np.ndarray[int] = mask_region[:, :width_slice]
            layer_region: np.ndarray = layer_region[:, :width_slice]

            # when the area we want to assign is already occupied
            conflict: np.ndarray[bool] = (
                    (image_seeds_pad[slice_region] != 0) & mask_region
            )

            image_seeds_slice: np.ndarray[int] = image_seeds_pad[slice_region]
            image_seeds_slice[mask_region] = factor_s * seed * layer_region[mask_region]
            # add conflicting pixels
            conflicts_pad[slice_region] |= conflict

        if not ignore_conflicts:
            # set conflicting pixels to 0
            image_seeds_pad[conflicts_pad] = 0

        # crop off the zero padding
        slice_original = np.index_exp[
            :, half_width:half_width + image_classification.shape[1]
        ]

        self.image_seeds: np.ndarray[int] = image_seeds_pad[slice_original]
        conflicts: np.ndarray[bool] = conflicts_pad[slice_original]
        # set background pixels to 0
        self.image_seeds *= self.mask_foreground
        conflicts &= self.mask_foreground.astype(bool)

        if plts:
            self.plot_image_seeds_and_classification()
            # pixels with conflict (conflict --> True)

            plt.imshow(conflicts, interpolation='none')
            plt.title('conflicts')
            plt.show()

    def get_image_expanded_laminae(self) -> np.ndarray[int]:
        """
        Get an image where labels are expanded to fill all pixels that are not holes.

        Returns
        -------
        image_expanded : np.ndarray[int]
            Image with expanded labels.
        """
        assert check_attr(self, 'image_seeds'), 'call set_laminae_images_from_params'
        img = self.image_seeds
        img_e = expand_labels(img, distance=np.min(img.shape))
        img_e *= self.mask_foreground
        return img_e

    def get_image_simplified_classification(
            self, expanded: bool = False
    ) -> np.ndarray[int]:
        """
        Get an image with light and dark pixels.

        Returns
        -------
        image_simplified : np.ndarray[int]
            The image with light, dark and hole pixels.
        """
        assert check_attr(self, 'image_seeds'), \
            'call set_laminae_images_from_params'

        if expanded:
            isc: np.ndarray[int] = np.sign(self.get_image_expanded_laminae())
        else:
            isc: np.ndarray[int] = np.sign(self.image_seeds)

        isc[isc == 1] = key_light_pixels
        isc[isc == -1] = key_dark_pixels

        return isc

    def filter_bad_laminae(self, quality_threshold: float = 0., **_):
        """
        Filters out laminae below the specified threshold. 0 is a save value
        since values below 0 indicate that the layer is of the opposite class.

        This function is intended to be called after set_quality_score and before
        reduce_laminae.
        """
        assert check_attr(self, 'params_laminae_simplified'), 'call set_params_laminae_simplified first'
        assert 'quality' in self.params_laminae_simplified.columns, 'call set_quality_score first'

        qualities: pd.Series = self.params_laminae_simplified.quality

        mask_valid = qualities > quality_threshold
        logger.info(f'filtering out {(~mask_valid).sum()} laminae ({(~mask_valid).mean():.0%}) '
                    f'that fall below the quality threshold {quality_threshold}')
        self.params_laminae_simplified: pd.DataFrame = self.params_laminae_simplified.loc[mask_valid, :].reset_index(drop=True)

    def reduce_laminae(
            self, plts: bool = False, n_expected: int | None = None, **kwargs
    ) -> None:
        """
        Reduce number of layers to that predicted by age model.

        This function combines layers starting from the lowest quality one until the number of
        layers is exactly the same as that of the age model

        Parameters
        ----------
        plts : bool, optional
            Plot the new classified image and quality scores. Defaults to False
        n_expected : int, optional
            Number of expected layers. If not provided, attempts to use the age_span
        kwargs : dict, optional
            Additional keyword arguments passed on to set_laminae_images
        """
        def reduce(df_: pd.DataFrame) -> pd.DataFrame:
            """
            Combine duplicate seed entries.

            Heights are summed, poly-coeffs are averaged, quality is maximized.
            """
            # TODO: take weighted averages (weights=areas)
            df_: pd.DataFrame = df_.sort_values(by=['sseed', 'quality'])
            grouper = df_.groupby(by='sseed')
            summed: pd.DataFrame = grouper.sum()
            # lowest quality is first so take last
            highest: pd.DataFrame = grouper.tail(1)

            meaned: pd.Series = df_\
                .drop(columns='color')\
                .groupby(by='sseed')\
                .mean()

            summed.reset_index(inplace=True, drop=False)
            highest.reset_index(inplace=True, drop=True)
            meaned.reset_index(inplace=True, drop=True)

            highest.loc[:, 'height'] = summed.loc[:, 'height']
            highest.loc[:, ['a', 'b', 'c', 'd']] = meaned.loc[:, ['a', 'b', 'c', 'd']]

            return highest

        assert check_attr(self, 'age_span') or (n_expected is not None), (
            'reduce_laminae requires either an age_span ' +
            'or the number of expected layers, exiting method'
        )
        assert check_attr(self, 'params_laminae_simplified'), \
            'call set_params_laminae_simplified'
        assert 'quality' in self.params_laminae_simplified.columns, \
            'internal error: quality criterion missing'
        assert check_attr(self, 'age_span'), \
            'no age span set, call set_age_span'

        df: pd.DataFrame = self.params_laminae_simplified.copy()

        # add number color column
        colors: np.ndarray[int] = np.array([
                1 if c == 'light' else -1
                for c in df.color
        ])
        seeds: np.ndarray[int] = df.seed.to_numpy() * colors

        df.loc[:, 'sseed'] = seeds

        # remove duplicate layers (same seed and color)
        df: pd.DataFrame = reduce(df)

        # determine how many layers have to be removed
        # actual - expected
        if n_expected is None:
            n_expected: int = round(self.age_span[1] - self.age_span[0])
        n_light: int = (np.unique(df.sseed) > 0).sum()
        n_dark: int = (np.unique(df.sseed) < 0).sum()
        n_light_excess: int = max((n_light - n_expected, 0))  # at least 0
        n_dark_excess: int = max((n_dark - n_expected, 0))  # at least 0
        logger.info(
            f'expecting {n_expected} each, ' +
            f'found {n_light=} and {n_dark=} layers'
        )
        if (not n_light_excess > 0) and (not n_dark_excess > 0):
            logger.warning('Since number of layers is too low, exiting method call.')
            return

        logger.info(
                f'removing the layers with the {n_light_excess} lowest light '
                f'and the {n_dark_excess} lowest dark qualities'
        )

        mask_light = df.loc[:, 'sseed'] > 0
        mask_dark = df.loc[:, 'sseed'] < 0
        indices_light_too_low = df\
            .loc[mask_light, 'quality']\
            .nsmallest(n=n_light_excess)\
            .index
        indices_dark_too_low = df\
            .loc[mask_dark, 'quality']\
            .nsmallest(n=n_dark_excess)\
            .index
        drop_rows = np.concatenate((indices_dark_too_low, indices_light_too_low))
        df.drop(index=drop_rows, inplace=True)
        df.sort_values(by='seed', inplace=True)
        df.reset_index(inplace=True, drop=True)

        self.params_laminae_simplified: pd.DataFrame = df

        n_light: int = (np.unique(df.sseed) > 0).sum()
        n_dark: int = (np.unique(df.sseed) < 0).sum()
        logger.info(
            f'expecting {n_expected}, ' +
            f'now found {n_light=} and {n_dark=} layers'
        )

        logger.info('updating qualities')
        self.set_quality_score(plts=plts)
        logger.info('updating classification images')
        self.set_laminae_images_from_params(plts=plts, **kwargs)

    def plot_quality(self, take_abs=True, hold=False, fig=None, ax=None):
        """Plot quality criteria."""
        assert self.params_laminae_simplified is not None, 'call set_params_laminae_simplified first'

        params = self.params_laminae_simplified
        height_img, width_img = self.image_grayscale.shape
        # overview plot
        if fig is None:
            fig, axs = plt.subplots(nrows=2, sharex=True)
        else:
            axs = [ax]
            hold = True

        # if called from outside (e.g. ax provided), only quality criteria
        # will be plotted
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
        axs[0].legend(bbox_to_anchor=(0, 1, 1, 0.2),
                      loc="lower left",
                      mode="expand",
                      borderaxespad=0,
                      ncol=1 if is_outside_call else 3)

        if not is_outside_call:
            axs[1].imshow(self.image_classification,
                          interpolation='none',
                          aspect='auto',
                          cmap='gray')
            fig.tight_layout()

        if not hold:
            plt.show()
        else:
            return fig, axs

    def plot_image_seeds_and_classification(self):
        """Plot the simplified laminae"""
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

    def plot_overview(
            self,
            fig: plt.Figure | None = None,
            axs: Iterable[plt.Axes] | None = None,
            hold=False
    ) -> None | tuple[plt.Figure, Iterable[plt.Axes]]:
        """Plot an overview graph."""

        if fig is None:
            assert axs is None, "If ax is provided, must also provide fig"
            fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True,
                                    layout='constrained')
        else:
            assert axs is not None, "If fig is provided, must also provide ax"

        # will plot only the simplified classification
        axs = np.array(axs)
        only_final = axs.shape == (1,)

        if not only_final:
            plt_cv2_image(image=self.image,
                          ax=axs[0, 0],
                          fig=fig,
                          title='Input image (tilt corrected: '
                                f'{self.use_tilt_correction})',
                          no_ticks=True,
                          swap_rb=True)
            plt_cv2_image(image=self.image_classification,
                          ax=axs[0, 1],
                          fig=fig,
                          title='Classification input',
                          no_ticks=True)
            self.plot_quality(fig=fig, ax=axs[1, 1])

        plt_cv2_image(image=self.get_image_simplified_classification(),
                      ax=axs[0] if only_final else axs[1, 0],
                      fig=fig,
                      title='Simplified classification',
                      no_ticks=True)

        if not only_final:
            ax = axs[0, 0]
            ax.set_xlabel('depth (pixels)')
            ax.set_ylabel('scaled criterion')

        if hold:
            return fig, axs

        plt.show()

    def set_laminae_params_table(self, **kwargs):
        """Do all steps at once."""
        # set seeds with their prominences
        logger.info("setting seeds")
        self.set_seeds(**kwargs)
        if self.age_span is not None:
            self.reduce_to_n_factor_seeds(kwargs.pop('factor_above_age_span', None))
        else:
            logger.warning('Age span not set, cannot rude seeds')
        # initiate params dataframe with seeds and params for distorted rects
        logger.info("finding distorted rects")
        self.set_params_laminae_simplified(**kwargs)
        # create output images for further analysis
        logger.info("creating image")
        # add quality criteria for each layer
        logger.info("calculating quality score")
        self.set_quality_score(**kwargs)
        self.filter_bad_laminae(**kwargs)
        self.reduce_laminae(**kwargs)

    def require_laminae_params_table(self, **kwargs):
        if (self.params_laminae_simplified is not None) and ('homogeneity' not in self.params_laminae_simplified.columns):
            self.set_quality_score(plts=kwargs.get('plts', False))
        if self.params_laminae_simplified is None:
            self.set_laminae_params_table(**kwargs)
        return self.params_laminae_simplified
