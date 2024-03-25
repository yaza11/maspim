from res.constants import key_light_pixels, key_dark_pixels, key_hole_pixels
from util.manage_obj_saves import class_to_attributes
from util.cClass import Convinience, verbose_function, return_existing
from imaging.misc.fit_distorted_rectangle import find_layers, distorted_rect
from imaging.misc.find_punch_holes import find_holes

import imaging.util.Image_convert_types as Image_convert_types
from imaging.util.coordinate_transformations import rescale_values
from imaging.util.Image_convert_types import ensure_image_is_gray
from imaging.util.Image_plotting import plt_cv2_image, plt_contours, plt_rect_on_image
from imaging.util.Image_processing import (adaptive_mean_with_mask,
                              adaptive_mean_with_mask_by_rescaling,
                              remove_outliers_by_median,
                              threshold_background_as_min,
                              func_on_image_with_mask,
                              auto_downscaled_image,
                              downscale_image)

from imaging.util.Image_geometry import (calculate_directionality_PCA,
                            calculate_directionality_moments,
                            kartesian_to_polar,
                            polar_to_kartesian,
                            contour_to_xy,
                            star_domain_contour)

from imaging.util.Image_helpers import (ensure_odd,
                           get_half_width_padded,
                           first_nonzero,
                           last_nonzero,
                           min_max_extent_layer,
                           filter_contours_by_size)

from imaging.util.Image_boxes import get_mean_intensity_box, region_in_box, get_ROI_in_image

from data.file_helpers import get_d_folder, get_image_file

import pickle
import pandas as pd
import PIL
import os
import cv2
import scipy
import skimage
import functools
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import expand_labels

from scipy.optimize import minimize
from typing import Iterable


class Image(Convinience):
    """Base function to get sample images and analyze them."""
    image_type_default = 'cv'
    def __init__(
        self, path_folder, image, image_type='cv', obj_color=None,
    ):
        """Initiator."""
        # options mutable by user
        self.plts = False
        self.verbose = False
        
        self.path_folder = path_folder
        
        # get the image from the inputs
        self.image_type = image_type
        self._image_original = self.ensure_image_is_cv(image)
        # make sure image is oriented horizontally
        h, w, *_ = self._image_original.shape 
        if h > w:
            print('swapped axes of input image to ensure horizontal orientation')
            self._image_original = self._image_original.swapaxes(0, 1)
        self._hw = h, w
        
        if obj_color is None:
            self.sget_obj_color()
        else:
            self.obj_color=obj_color
            
        self.set_current_image()

    @verbose_function
    def load(self):
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_folder, name), 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__

    def sget_image_original(self):
        return self._image_original

    @verbose_function
    def sget_image_grayscale(self) -> np.ndarray:
        """Set _image_original_grayscale if it does not exist and return it."""
        return self.manage_sget(
            '_image_original_grayscale',
            ensure_image_is_gray,
            image=self.sget_image_original().copy()
        )

    @verbose_function
    def set_current_image(self, **kwargs) -> None:
        self.current_image = self.sget_image_original().copy()

    @verbose_function
    def sget_current_image(self, **kwargs) -> np.ndarray:
        return self.sget(
            'current_image', self.set_current_image, **kwargs
        )

    @verbose_function
    def ensure_image_is_cv(self, image):
        """
        Convert input image to cv inplace.

        Parameters
        ----------
        image : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        image in cv.

        """
        image = Image_convert_types.convert(
            self.image_type, 'cv', image)
        self.image_type = 'cv'
        return image

    @verbose_function
    def ensure_image_is_binary(self, image):
        image = ensure_image_is_gray(image)
        if set(np.unique(image)) != set(np.array([0, 255])):
            _, image = self.get_foreground_thr_and_pixels(image)
        return image

    @verbose_function
    def use_channel(self, key: str | int, image=None):
        cv_channel_to_idx = {'B': 0, 'G': 1, 'R': 2}
        PIL_channel_to_idx = {'R': 0, 'G': 1, 'B': 2}
        image = self.get_image(image, copy=True)
        if len(image.shape) != 3:
            raise ValueError('Cannot apply use_channel on singlechannel image')

        if isinstance(key, int):
            idx = key
        elif self.image_type == 'cv':
            cv_channel_to_idx[key]
        elif self.image_type == 'PIL':
            PIL_channel_to_idx[key]
        image = image[:, :, idx]
        return image

    @verbose_function
    def get_foreground_thr_and_pixels(
            self, image_gray: np.ndarray, thr_method: str = 'otsu', **kwargs
    ) -> tuple[float | int, np.ndarray]:
        """
        Return threshold for foreground pixels and binary image.

        Parameters
        ----------
        image : cv2 image | None, optional
            image for which to find foreground pixels. The default is None.
        thr_method : str, optional
            The method to use for thresholding. The default is 'otsu'.
            The other option is 'local-min'.

        Returns
        -------
        thr background, binary image (foreground pixels = True).

        """
        image_gray = ensure_image_is_gray(image_gray)

        # define the threshold type depending on the object color
        if self.sget_obj_color() == 'dark':
            thr = cv2.THRESH_BINARY_INV
        else:
            thr = cv2.THRESH_BINARY

        # create binary image with the specified method
        if thr_method.lower() == 'otsu':
            if self.verbose:
                print('Determining threshold for background intensity with OTSU.')
            thr_background, mask_foreground = cv2.threshold(
                image_gray, 0, 1, thr + cv2.THRESH_OTSU)
        elif thr_method.lower() == 'local_min':
            if self.verbose:
                print('Determining threshold for background intensity with local-min.')
            thr_background = threshold_background_as_min(image_gray)
            _, mask_foreground = cv2.threshold(
                image_gray, thr_background, 1, thr)
        else:
            raise KeyError(f'{thr_method=} is not a valid option. Choose one of\
"otsu", "local_min".')

        if self.plts:
            plt_cv2_image(mask_foreground, 'Identified foreground pixels')
        return thr_background, mask_foreground

    def sget_foreground_thr_and_pixels(
            self, thr_method='otsu', **kwargs
    ) -> tuple[float | int, np.ndarray[int]]:
        if not self.check_attributes_exist(['mask_foreground', 'thr_background']):
            self.thr_background, self.mask_foreground = self.get_foreground_thr_and_pixels(
                image_gray=self.sget_image_grayscale(),
                thr_method=thr_method,
                **kwargs
            )
        return self.thr_background, self.mask_foreground

    @verbose_function
    def get_binarisation_of_foreground(
            self,
            image: np.ndarray,
            mask: np.ndarray[int | bool] | None = None,
            overwrite: bool = False) -> tuple[np.ndarray[np.uint8]]:
        """
        OTSU-binarisation of foreground pixels.

        Parameters
        ----------
        image : cv2 image | None
            imagefor which to create binarisation.
        mask : cv2 image of uint8
            Pixels with foreground are 255.

        Returns
        -------
        light_pixels : cv2 image
            The light pixels.
        dark_pixels : cv2 image
            The dark pixels.

        """
        if mask is None:
            assert self.check_attribute_exists('mask_foreground')
            mask = self.mask_foreground.copy()

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
        print(light_pixels.shape)
        dark_pixels = func_on_image_with_mask(
            image, mask, func_dark, return_argument_idx=1) == 255

        if self.plts:
            plt_cv2_image(light_pixels, 'identified light pixels')
            plt_cv2_image(dark_pixels, 'identified light pixels')

        return light_pixels, dark_pixels

    @verbose_function
    def get_simplified_image(
            self, image_binary: np.ndarray, factors: Iterable[int] | None = None
    ) -> np.ndarray:
        """
        Return a simplified version of the input binary image.

        If image is not binary, it will be converted to binary image.
        Simplification is achieved by stepping up the size of a median blur
        filter. Filter size is limited to 255

        Parameters
        ----------
        image_binary : np.ndarray
            Input.

        Returns
        -------
        image_binary : np.ndarray
            Output image.

        """
        image_binary = self.ensure_image_is_binary(image_binary)

        if factors is None:
            factors = [1024, 512, 256, 128, 64, 32]
        for factor in factors:
            # smooth edge
            # limit kernel size to 255 as larger kernel sizes are not supported
            kernel_size = np.min([np.max(image_binary.shape) // factor, 255])
            if not kernel_size % 2:
                kernel_size += 1
            image_binary = cv2.medianBlur(
                image_binary, kernel_size).astype(np.uint8)
            if self.plts:
                plt_cv2_image(image_binary, f'{factor=}, {kernel_size=}')
        return image_binary

    @verbose_function
    def sget_simplified_image(self) -> np.ndarray:
        if not self.check_attribute_exists('image_binary'):
            self.image_binary = self.get_simplified_image(image_binary=self.sget_foreground_thr_and_pixels()[1])
        return self.image_binary

    @verbose_function
    def get_main_contour(
            self, image_binary, method='take_largest', filter_by_size=.3):

        image_binary = self.ensure_image_is_binary(image_binary)

        contours, _ = cv2.findContours(
            image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if self.plts:
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
                f"{method=} is not an option. Valid options are \
('take_largest', 'star_domain', 'filter_by_size', 'convex_hull')")

        if self.plts:
            plt_contours(
                contours=[contour],
                image=image_binary,
                title='main contour'
            )

        return contour
    
    def sget_main_contour(self):
        if (main_contour := self.__dict__.get('main_contour')) is not None:
            return main_contour
        self.main_contour = self.get_main_contour(self.sget_simplified_image())
        return self.main_contour

    @verbose_function
    def save(self):
        # delete all attributes that are not flagged as relevant
        dict_backup = self.__dict__.copy()
        keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
        existent_attributes = list(self.__dict__.keys())
        verbose = self.verbose
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)
        if verbose:
            print(f'saving image object with {self.__dict__.keys()}')
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup


class ImageSample(Image):
    """Find image on disc, find ROI."""
    def __init__(
        self, 
        path_folder: str | None = None, 
        image: np.ndarray[float | int] | None = None, 
        image_type: str = 'cv', 
        path_image_file: str | None = None, 
        obj_color: str | None = None
    ):
        """Initiator."""
        assert (path_folder is not None) or (image is not None) or (path_image_file is not None), \
            'provide either a path (to the folder) or an image'
        # options mutable by user
        self.plts: bool = False
        self.verbose: bool = False
        
        if path_folder is not None:
            self.path_folder: str = path_folder
        else: 
            self.path_folder: str = ''
        
        # get the image from the inputs
        self.image_type: str = image_type
        if image is None:
            if path_image_file is None:
                path_image_file = os.path.join(
                    self.path_folder, get_image_file(self.path_folder)
                )
            image: np.ndarray[np.uint8] = cv2.imread(path_image_file)
        
        self._image_original: np.ndarray[np.uint8] = self.ensure_image_is_cv(image)
        # make sure image is oriented horizontally
        h, w, *_ = self._image_original.shape 
        if h > w:
            print('swapped axes of input image to ensure horizontal orientation')
            self._image_original = self._image_original.swapaxes(0, 1)
        self._hw = h, w
        
        if obj_color is None:
            self.sget_obj_color()
        else:
            self.obj_color=obj_color
            
        self.set_current_image()

    @verbose_function
    def set_obj_color(self, region_middleground=.8, **kwargs):
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
        image_gray = self.sget_image_grayscale()

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
        if self.verbose:
            print(f'obj appears to be {obj_color}')

        self.obj_color = obj_color

    def sget_obj_color(self):
        return self.sget('obj_color', self.set_obj_color, is_get_function=False)

    @verbose_function
    def get_sample_area_box(
        self, image: np.ndarray, dilate_factor: float = 1, **kwargs
    ) -> tuple[np.ndarray, tuple[int]]:
        """
        Use optimizer to find samplearea of box.

        Parameters
        ----------
        image : array, optional
            DESCRIPTION. The default is None.
        image_thresh : array, optional
            image used to find the box. The default is None.
        dilate_factor : float, optional
            value used to dilate box. The default is 1. Value smaller than 1:
                will be added to the determined ratio
            Value bigger than 1:
                will be multiplied with the determined box ratio

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        image_binary = self.ensure_image_is_binary(image)
        image_downscaled, scale_factor = auto_downscaled_image(
            image_binary)

        def metric(x0):
            """
            Calcualte the difference in pixel intensity for a specified box.

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
            mean_box = mean_box[0]
            mean_rest = mean_rest[0]
            fraction_area = box_ratio_y * box_ratio_x
            return -np.abs(mean_box - mean_rest) * fraction_area

        # initiate center_box
        middle_y = round(image_downscaled.shape[0] / 2)
        middle_x = round(image_downscaled.shape[1] / 2)

        if self.verbose:
            print('searching optimal parameters for box')
        params = minimize(
            metric,  # function to minimize
            x0=[.5, .5, middle_x, middle_y],  # start values
            method='Nelder-Mead',  # method
            bounds=((0, 1), (0, 1), (0, image_downscaled.shape[1]),
                    (0, image_downscaled.shape[0]))  # bounds of parameters
        )
        # determined values
        box_ratio_x, box_ratio_y, center_box_x, center_box_y = params.x
        center_box = (center_box_x, center_box_y)
        if self.verbose:
            print(f'found box with {params.x}')
            print(f'solver converged: {params.success}')

        # get params of box from those determined by the optimizer
        box_params = region_in_box(
            image=image_downscaled, box_ratio_x=box_ratio_x,
            box_ratio_y=box_ratio_y, center_box=center_box)
        if self.plts:
            plt_rect_on_image(image_downscaled, box_params, title='Detected ROI of sample', **kwargs)

        # dilate the box slightly for finer sampledefinition
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
        image_ROI = image[y:y + h, x:x + w].copy()

        if self.plts:
            plt_cv2_image(image_ROI, 'detected ROI')

        return image_ROI, (x, y, w, h)

    @verbose_function
    def get_sample_area_contours(
            self, image, image_thresh=None, filter_by_size=.1,
            filter_by_ratio=.5, method='hierarchy-tree',
            automatic_downscale=1000, scale_factor=None, **kwargs):
        """
        Find contours in image using cv2's findContours.

        Parameters
        ----------
        image : uint8 array
            image in which to find contours.
        method : str
            One of ('hierarhy-tree', 'double-otsu'). hierarchy-tree is suitable
            for samples that show clear lamination and should be faster whereas
            the double-otsu method applies otsu to the region spanned by each
            contour to the grayscale image.

        Returns
        -------
        cnts : list[np.array]
            list of contours in which the list entries are numpy arrays with the
            pixels corresponding to the contour of each shape.

        """
        raise NotImplementedError('Outdated')
        image_downscaled, scale_factor = auto_downscaled_image(image)
        image_gray = ensure_image_is_gray(image=image_downscaled)

        # get thresholded image
        if image_thresh is None:
            if self.verbose:
                print('getting contours from classifying laminae')
            # get the light pixels
            image_thresh, _ = self.classify_laminae_by_threshold(
                image=image_gray,
                use_mask_for_adaptive=True,
                add_thr_background=True,
                threshold_contour_size=filter_by_size)

        contours, hierarchy_tree = cv2.findContours(image_thresh,
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)

        if self.plts:
            plt_contours(
                image_thresh, 'image and contours used to find contours')

        # reformat from (1, n_contours, 4) to (n_contours, 4)
        # rows in hierarchy_tree correspond to indices of
        # [Next, Previous, First_Child, Parent]
        hierarchy_tree = hierarchy_tree[0, :, :]
        # filter out compounds that do not show directionality
        if filter_by_ratio > 0:
            if self.verbose:
                print('filtering contours by directionality')
            contours_filtered_ratio = []
            for contour_idx, contour in enumerate(contours):
                contours_in_contour = []
                child_idx = hierarchy_tree[contour_idx, 2]
                if method == 'hierarchy-tree':
                    # only for contours that have children
                    while child_idx != -1:
                        print(child_idx)
                        # get the child contour
                        contours_in_contour.append(contours[child_idx])
                        # get index of next contour on same level
                        child_idx = hierarchy_tree[child_idx, 0]
                elif method == 'double-otsu':
                    # skip contours with no children
                    if child_idx == -1:
                        continue
                    # find section corresponding to contour
                    xc, yc, wc, hc = cv2.boundingRect(contour)
                    image_section = image_gray[
                        yc: yc + hc, xc: xc + wc
                    ].copy()
                    # apply large median filter
                    ksize = np.min([wc, hc]) // 10
                    if not ksize % 2:
                        ksize += 1
                    ksize = np.min([ksize, 255])
                    image_section = cv2.medianBlur(
                        image_section, ksize=ksize)
                    # build in extra binary-step to bring out laminae
                    _, image_section_binary = cv2.threshold(
                        src=image_section, thresh=0, maxval=255,
                        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours_in_contour, _ = cv2.findContours(
                        image_section_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                else:
                    raise NotImplementedError(f"{method=} is not implemented. \
Choose one of ('hierarhy-tree', 'double-otsu').")

                n_contours = len(contours_in_contour)
                if n_contours >= 2:
                    _, difference, _ = calculate_directionality_PCA(
                        contours_in_contour)

                    if self.plts:
                        # get the region of current contour
                        xc, yc, wc, hc = cv2.boundingRect(contour)
                        if method == 'hierarchy-tree':
                            canvas = image
                        elif method == 'double-otsu':
                            canvas = image_section
                        self.plt_contours(
                            contours_in_contour, canvas,
                            f'1 - cosine similarity: {difference} for\
{n_contours} contours')

                        if difference < filter_by_ratio:
                            contours_filtered_ratio.append(contour)
            contours = contours_filtered_ratio

        # find the bounding box
        all_contours_combined = np.concatenate(contours, axis=0)
        x, y, w, h = cv2.boundingRect(all_contours_combined)

        if self.plts:
            plt_contours(contours, image, 'all detected contours')

        if scale_factor != 1:
            x = int(x / scale_factor)
            y = int(y / scale_factor)

            w = int(w / scale_factor)
            h = int(h / scale_factor)
        image_ROI = image[y:y + h, x:x + w].copy()

        if self.plts:
            plt_cv2_image(image_ROI, 'detected ROI')

        return image_ROI, (x, y, w, h)

    @verbose_function
    def get_sample_area_main_contour(
            self, image, method='take_largest'):
        contour = self.get_main_contour(image, method=method)

        x, y, w, h = cv2.boundingRect(contour)

        image_ROI = image[y:y + h, x:x + w].copy()

        if self.plts:
            plt_cv2_image(
                image_ROI, 'detected ROI as bounding box of main contour'
            )

        return image_ROI, (x, y, w, h)

    @verbose_function
    def get_sample_area(self, image, **kwargs):
        # find the rough region of interest with box
        image_ROI_box, (xb, yb, wb, hb) = self.get_sample_area_box(
            image, dilate_factor=0.1)
        # get the foreground pixels in the ROI
        _, ROI_binary = self.get_foreground_thr_and_pixels(image_ROI_box, **kwargs)
        # simplify the binary image
        ROI_simplified = self.get_simplified_image(ROI_binary)
        # find the refined area as the extent of the simplified binary image
        _, (xc, yc, wc, hc) = self.get_sample_area_main_contour(
            ROI_simplified, method='filter_by_size')

        # stack the offsets of the two defined ROI's since the second ROI is
        # placed in the first one
        x = xc + xb
        y = yc + yb
        w = wc
        h = hc

        image_ROI = image[y: y + h, x: x + w].copy()

        if self.plts:
            plt_cv2_image(image_ROI, 'final ROI as defined by \
get_sample_area')

        return image_ROI, (x, y, w, h)

    def sget_sample_area(
            self, **kwargs
    ) -> tuple[np.ndarray[np.uint8], tuple[int]]:
        """Set and return area of the sample in the image."""
        return self.manage_sget(
            ['image_ROI', 'xywh_ROI'],
            self.get_sample_area,
            image=self.sget_current_image(),
            **kwargs
        )
    
    def get_sample_area_from_xywh(self):
        assert hasattr(self, 'xywh_ROI'), 'call sget_sample_area first'
        image = self._image_original
        x, y, w, h = self.xywh_ROI
        return image[y:y + h, x:x + w].copy()


class ImageROI(Image):
    """Create obj from xywh, classify laminae."""
    def __init__(
        self, path_folder = None, image = None, image_type='cv', obj_color=None
    ):
        """Initiator."""
        # options mutable by user
        self.plts: bool = False
        self.verbose: bool = False
        
        if path_folder is not None:
            self.path_folder: str = path_folder
        else:
            self.path_folder: str = ''
        
        # get the image from the inputs
        self.image_type: str = image_type
        if image is None:
            IS: ImageSample = self.get_image_sample()
            assert hasattr(IS, 'xywh_ROI'), \
                'save an ImageSample object with detected sample area first'
            image: np.ndarray = IS.get_sample_area_from_xywh()
        if obj_color is not None:
            self.obj_color: str = obj_color
        
        self._image_original: np.ndarray = self.ensure_image_is_cv(image)
            
        self.set_current_image()
    
    
    def get_image_sample(self):
        IS = ImageSample(self.path_folder, image=None)
        IS.load()
        return IS
    
    def get_parent_color_and_extent(
            self, overwrite: bool = False
    ) -> tuple[str, tuple[int]]:
        """Return obj color and ROI extent in original image."""
        IS = self.get_image_sample()
        # check if the saved obj has neccessary attributes (if it exists)
        if overwrite:
            compute_area = True
        elif os.path.exists(os.path.join(self.path_folder, 'ImageSample.pickle')):
            IS.load()
            if hasattr(IS, 'xywh_ROI'):
                compute_area = False
            else:
                compute_area = True
        else:
            compute_area = True
        
        if compute_area:
            IS.sget_sample_area()
        return IS.obj_color, IS.xywh_ROI

    def sget_parent_color_and_extent(self) -> tuple[str, tuple[int]]:
        """Set and return obj color and ROI extent."""
        return self.manage_sget(
            ['obj_color', 'xywh_ROI'],
            self.get_parent_color_and_extent
        )

    def sget_obj_color(self):
        """Set and return object color."""
        if hasattr(self, 'obj_color'):
            return self.obj_color
        try:
            return self.sget_parent_color_and_extent()[0]
        except KeyError as e:
            print(f'{e}, assuming object is dark')
            self.obj_color = 'dark'
            return self.obj_color
            
            

    @return_existing('_image_original')
    def sget_image_original(self) -> np.ndarray:
        """Set and return original image."""
        parent = ImageSample(self._section, self._window)
        image = parent.sget_image_original()
        
        return get_ROI_in_image(
            image=image, xywh_ROI=self.sget_parent_color_and_extent()[1]
        )

    def set_age_span(self, age_span: tuple):
        self.age_span = age_span

    def get_average_width_yearly_cycle(self) -> float:
        """Calculate how many cycles are in the interval and their av width."""
        assert hasattr(self, 'age_span'), 'call set_age_span'
        pixels_x = self.sget_image_original().shape[1]
        # calculate the number of expected cycles from the age difference for
        # the depth interval of the slice
        average_thickness_pixels = pixels_x / (self.age_span[1] - self.age_span[0])
        return average_thickness_pixels

    @verbose_function
    def find_best_kernel_size_adaptive(self, image_gray):
        """Estimate kernel size from image properties."""
        raise NotImplementedError('this function isnt fully functional yet')
        image_gray = self.ensure_image_is_gray(image_gray)

        max_kernel_size = np.min(image_gray.shape)
        kernel_size_ratio0 = 0.1

        def evaluate_directionality(x0):
            kernel_size_ratio = x0[0]
            kernel_size_adaptive = round(kernel_size_ratio * max_kernel_size)
            if not kernel_size_adaptive % 2:
                kernel_size_adaptive += 1
            image_light = cv2.adaptiveThreshold(
                src=image_gray,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=kernel_size_adaptive,
                C=0)
            contours, _ = cv2.findContours(image_light, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
            contours_filtered = []
            for contour in contours:
                if contour.shape[-1] > 10:
                    contours_filtered.append(contour)

            if len(contours_filtered) > 2:
                directionality = 1 - \
                    calculate_directionality_PCA(contours_filtered)
            else:
                directionality = 1
            canvas = image_light.copy()
            if self.plts:
                for idx, contour in contours_filtered:
                    cv2.drawContours(canvas, contours, idx,
                                     127, max_kernel_size // 20)
                    self.plt_cv2_image(canvas, f'{kernel_size_adaptive=}')
            return directionality

        params = minimize(evaluate_directionality, x0=(
            kernel_size_ratio0), method='BFGS', bounds=((0, 1)))
        print(params)
        print(params.x[0] * max_kernel_size)

    def get_params_laminae_classification(
            self, image_gray_shape: tuple[int], **kwargs
    ) -> dict:
        """Set default params for classification and overwrite by kwargs."""
        # set default values
        params = {
            'remove_outliers': False,
            'use_bilateral_filter': True,
            'use_adaptive_threshold': True,
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

        # update kernel_size to potentially match ROI
        if params['estimate_kernel_size_from_age_model']:
            if self.verbose:
                print('Estimating kernel size from age model (square with \
2x expected thickness of one year).')
            kernel_size_adaptive = ensure_odd(
                int(self.get_average_width_yearly_cycle() * 2))
            params['kernel_size_adaptive'] = kernel_size_adaptive
        elif params['kernel_size_adaptive'] is None:
            if self.verbose:
                print('estimating adaptive kernel size from image dimensions')
            kernel_size_adaptive = ensure_odd(np.min(image_gray_shape) // 10)
            params['kernel_size_adaptive'] = kernel_size_adaptive

        if self.verbose:
            print('Using the following parameters:')
            print(params)

        return params

    @verbose_function
    def get_preprocessed_for_classification(
            self,
            image_gray: np.ndarray,
            **kwargs
    ) -> tuple[np.ndarray, dict[str, bool | int | float]]:
        """Preprocess an image for classification (remove noise)."""
        image_gray = ensure_image_is_gray(image_gray)
        # this method performs the following steps
        # 1. remove outliers by detecting large differences between original
        #   image and its median filtered version
        # 2. identify back- and foreground pixels with otsu-filter
        # 3a. apply bilateral filter
        # (3b. apply Gauss filter) -> default: don't apply

        # update params with kwargs
        params = self.get_params_laminae_classification(
            image_gray.shape, **kwargs
        )
        if self.plts:
            plt_cv2_image(image_gray, 'input in grayscale')

        # get mask_foreground matching image_gray
        thr_background, mask_foreground = self.get_foreground_thr_and_pixels(
            image_gray)

        if self.plts:
            plt_cv2_image(
                mask_foreground,
                title=f'foreground pixels (thr={thr_background})'
            )

        # remove outliers
        if params['remove_outliers']:
            if self.verbose:
                print('Removing outliers with median filter.')
            image_gray = remove_outliers_by_median(
                image_gray, kernel_size_median=params['kernel_size_median'],
                threshold_replace_median=params['threshold_replace_median'])
            if self.plts:
                plt_cv2_image(image_gray, 'Outliers removed')

        if params['use_bilateral_filter']:
            if self.verbose:
                print('Applying bilateral filter.')
            image_gray = cv2.bilateralFilter(image_gray, d=-1,
                                             sigmaColor=params['sigmaColor'],
                                             sigmaSpace=params['sigmaSpace'])
            if self.plts:
                plt_cv2_image(image_gray, 'Bilateral filter')

        return image_gray, mask_foreground, params

    @verbose_function
    def get_postprocessed_image_from_classification(
            self,
            image_light: np.ndarray,
            mask_foreground: np.ndarray,
            params: dict
    ) -> np.ndarray:
        """Postprocess a classified image (remove small features)."""
        if params['remove_small_areas']:
            if self.verbose:
                print('Removing small blobs.')
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

        if self.plts:
            plt_cv2_image(
                image_classification, 'final classification')

        return image_classification

    @verbose_function
    def get_classification_adaptive_mean(
            self, image_gray: np.ndarray, overwrite: bool = False, **kwargs
    ) -> tuple[np.ndarray, dict]:
        """Classify image with adaptive mean threshold filter."""
        # this method works in the following steps
        # 1. remove outliers by detecting large differences between original
        #   image and its median filtered version
        # 2. identify back- and foreground pixels with otsu-filter
        # 3a. apply bilateral filter
        # (3b. apply Gauss filter) -> default: don't apply
        # 4. classify pixels by adaptive mean threshold filter or
        #   otsu binarisation for foreground pixels
        # 5. remove small blobs by calculating their diameter
        #   (default: second percentile will be removed)

        image_gray, mask_foreground, params = \
            self.get_preprocessed_for_classification(image_gray, **kwargs)

        # adaptive thresholding
        if self.verbose:
            print('adaptive thresholding with mask')

        image_light = adaptive_mean_with_mask_by_rescaling(
            image=image_gray,
            maxValue=1,
            thresholdType=cv2.THRESH_BINARY,
            ksize=(params['kernel_size_adaptive'], params['kernel_size_adaptive']),
            C=0,
            mask_nonholes=mask_foreground
        )
        image_light *= mask_foreground.astype(bool)

        if self.plts:
            plt_cv2_image(image_light, title='light pixels')

        image_classification = \
            self.get_postprocessed_image_from_classification(
                image_light,
                mask_foreground,
                params
            )
        return image_classification, params

    # TODO: option for ignoring age model 
    def sget_classification_adaptive_mean(self):
        """Create and return the image classification with parameters."""
        return self.manage_sget(
            ['image_classification', 'params_classification'],
            self.get_classification_adaptive_mean,
            image_gray=self.sget_image_grayscale()
        )

    def set_punchholes(self, remove_gelatine: bool, side: str, **kwargs):
        if not remove_gelatine:
            img: np.ndarray[np.uint8] = self.sget_image_grayscale()
        else:
            img: np.ndarray[bool] = self.sget_simplified_image() \
                * self.sget_foreground_thr_and_pixels()[1]
            
        self.punchholes: list[np.ndarray[int]] = find_holes(
            img,
            obj_color=self.obj_color,
            side=side,
            **kwargs
        )
        

class ImageClassified(Image):
    """Characterise and modify the classified layers."""
    def __init__(
        self, path_folder, image = None, image_type='cv', obj_color=None
    ):
        """Initiator."""
        # options mutable by user
        self.plts = False
        self.verbose = False
        if obj_color:
            self.obj_color = obj_color
        
        self.path_folder = path_folder
        
        # get the image from the inputs
        self.image_type = image_type
        if image is None:
            IR = self.get_image_roi()
            assert hasattr(IR, 'xywh_ROI'), \
                'save an ImageROI object with detected sample area first'
            image = IR.sget_image_original()
        
        self._image_original = self.ensure_image_is_cv(image)
            
        self.set_current_image()
    
    def get_image_roi(self):
        IR = ImageROI(self.path_folder, image=None)
        IR.load()
        return IR
    
    @verbose_function
    def get_parent_classification_and_color(self, overwrite: bool = False) -> np.ndarray:
        """Get classified image."""
        IR = self.get_image_roi()
        # check if the saved obj has neccessary attributes (if it exists)
        if overwrite:
            compute_classification = True
        elif os.path.exists(os.path.join(self.path_folder, 'ImageROI.pickle')):
            IR.load()
            if hasattr(IR, 'image_classification'):
                compute_classification = False
            else:
                compute_classification = True
        else:
            compute_classification = True
        
        if compute_classification:
            IR.sget_classification_adaptive_mean()
        return IR.image_classification, IR.sget_obj_color()

    @verbose_function
    def sget_image_classification_and_color(self) -> np.ndarray:
        """Set and get classification as original image."""
        if hasattr(self, 'obj_color'):
            self._image_classification = self.get_parent_classification_and_color()[0]
            return self._image_classification, self.obj_color
        return self.manage_sget(
            ['_image_classification', 'obj_color'], self.get_parent_classification_and_color
        )

    @verbose_function
    def sget_image_classification(self, **kwargs):
        """Return image classification from parent."""
        return self.sget_image_classification_and_color()[0]

    @verbose_function
    def sget_mask_foreground(self):
        """Return mask of pixels that are not holes."""
        return self.sget_image_classification() != key_hole_pixels

    @verbose_function
    def sget_obj_color(self):
        """Return obj color from parent."""
        return self.sget_image_classification_and_color()[1]

    def set_age_span(self, age_span: tuple):
        self.age_span = age_span

    def get_average_width_yearly_cycle(self) -> float:
        """Calculate how many cycles are in the interval and their av width."""
        assert hasattr(self, 'age_span'), 'call set_age_span'
        pixels_x = self.sget_image_original().shape[1]
        # calculate the number of expected cycles from the age difference for
        # the depth interval of the slice
        average_thickness_pixels = pixels_x / (self.age_span[1] - self.age_span[0])
        return average_thickness_pixels

    @verbose_function
    def sget_average_width_yearly_cycle(self):
        return self.manage_sget(
            'average_width_yearly_cycle',
            self.get_average_width_yearly_cycle
        )

    @verbose_function
    def get_image_original(self, overwrite=False):
        """Get ROI image."""
        # check if the saved obj has neccessary attributes (if it exists)
        o = ImageROI(self._section, self._window)
        return o.sget_image_original()

    @verbose_function
    def sget_image_original(self) -> np.ndarray:
        """Set and return original ROI image."""
        return self.manage_sget('_image_original', self.get_image_original)

    @verbose_function
    def set_seeds(
            self,
            in_classification: bool = True,
            peak_prominence: float = 0,
            hold=False,
            min_distance=None,
            **kwargs

    ) -> None:
        """Find peaks in col-wise averaged classification."""
        mask_foreground = self.sget_mask_foreground()
        horizontal_extent = mask_foreground.shape[0]

        if in_classification:
            image_light = self.sget_image_classification().copy()
            # set everything that is not light to 0
            image_light[image_light != key_light_pixels] = 0

            sum_lights = (image_light / image_light.max()).sum(axis=0)
        # in gray
        else:
            image_light = self.sget_image_grayscale().copy()
            # scaled from -0.5 to .5 and set holes to 0
            image_light = rescale_values(
                image_light * mask_foreground, 0, 1, 0, 255
            ) * mask_foreground

            sum_lights = image_light.sum(axis=0)
        sum_foreground = mask_foreground.sum(axis=0)
        # exclude columns with no sample material
        mask_nonempty_col = sum_foreground > 0
        # divide colwise sum of c by number of foreground pixels
        # empty cols will have a value of 0
        brightness = np.zeros(image_light.shape[1])
        brightness[mask_nonempty_col] = sum_lights[mask_nonempty_col] / sum_foreground[mask_nonempty_col]
        brightness -= .5
        brightness[~mask_nonempty_col] = 0

        yearly_thickness = self.sget_average_width_yearly_cycle()
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

        if self.plts:
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

    @verbose_function
    def get_seeds_above_prominence(self, peak_prominence):
        """Return seeds above given prominence for light and dark as tuple."""
        seeds_light = self.seeds_light[self.prominences_light > peak_prominence]
        seeds_dark = self.seeds_dark[self.prominences_dark > peak_prominence]
        return seeds_light, seeds_dark

    @verbose_function
    def set_params_laminae_simplified(
            self,
            peak_prominence: float,
            height0_mode: str = 'use_peak_widths',
            downscale_factor: float = 1,
            **kwargs
    ) -> None:
        """Run optimizer to find layer for each peak above prominence."""
        image_classification = self.sget_image_classification().copy()
        # get seeds above prominence
        seeds_light, seeds_dark = self.get_seeds_above_prominence(peak_prominence)

        if height0_mode == 'use_age_model':
            height0s_light = height0s_dark = self.sget_average_width_yearly_cycle() / 2
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
            plts=self.plts
        )
        dataframe_params_light['width'] = self.width_light
        dataframe_params_light['prominence'] = self.prominences_light

        dataframe_params_dark = find_layers(
            image_classification, seeds_dark, height0s_dark, color='dark',
            plts=self.plts
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

    @verbose_function
    def get_region_from_params(self, idx):
        """Get the region from an index in params table."""
        width = self.sget_image_classification().shape[0]
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

    @verbose_function
    def get_region_in_image_from_params(self, image, idx):
        """Get the region in an image from params."""
        region_layer, _, _ = self.get_region_from_params(idx)
        width = image.shape[0]
        row = self.params_laminae_simplified.iloc[idx, :]
        slice_region = np.index_exp[:, row.seed: row.seed + width]
        return get_half_width_padded(image)[slice_region]

    @verbose_function
    def rate_quality_layer(
            self,
            idx,
            keys_classification=[255, 127, 0],
            labels_classification=['light', 'dark', 'hole'],
            **kwargs
    ):
        """Calculate hom, cont, brightness for layer in self.params for idx."""
        # get region of the layer
        region_layer, _, _ = self.get_region_from_params(idx)
        region_layer = region_layer.astype(bool)
        region_classification = self.get_region_in_image_from_params(
            self.sget_image_classification(), idx)
        region_grayscale = self.get_region_in_image_from_params(
            self.sget_image_grayscale(), idx
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

    @verbose_function
    def set_quality_score(self):
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
        if self.plts:
            self.plt_quality()

    @verbose_function
    def set_laminae_images_from_params(
            self, ignore_conflicts=True, **kwargs
    ):
        """
        Create images with simplified laminae.

        (seed idx as value and light/dark as value), conflicts.
        """
        assert self.params_laminae_simplified is not None, \
            'create simplified laminae with \
simplify_laminae before calling create_simplified_laminae_classification.'

        image_classification = self.sget_image_classification(**kwargs)

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
        self.image_seeds *= self.sget_mask_foreground()
        conflicts *= self.sget_mask_foreground()

        if self.plts:
            self.plt_image_seeds_and_classification()
            # pixels with conflict (conflict --> True)

            plt.imshow(conflicts, interpolation='none')
            plt.title('conflicts')
            plt.show()
            
    def get_image_expanded_laminae(self):
        assert hasattr(self ,'image_seeds'), 'call set_laminae_images_from_params'
        img = self.image_seeds
        img_e = expand_labels(img, distance=np.min(img.shape))
        img_e *= self.sget_mask_foreground()
        return img_e

    def get_image_simplified_classification(self):
        assert self.check_attribute_exists('image_seeds')

        isc = np.sign(self.image_seeds)
        isc[isc == 1] = key_light_pixels
        isc[isc == -1] = key_dark_pixels

        return isc

    @verbose_function
    def set_laminae_params_table(self, **kwargs):
        # set seeds with their prominences
        if self.verbose:
            print("setting seeds")
        self.set_seeds(**kwargs)
        # initiate params dataframe with seeds and params for distorted rects
        if self.verbose:
            print("finding distorted rects")
        self.set_params_laminae_simplified(**kwargs)
        # create output images for further analysis
        if self.verbose:
            print("creating image")
        # add quality critera for each layer
        if self.verbose:
            print("calculating quality score")
        self.set_quality_score()
        # create classifiaction image
        self.set_laminae_images_from_params(**kwargs)

    def plt_quality(self, take_abs=True, hold=False):
        params = self.params_laminae_simplified
        height_img, width_img = self.sget_image_grayscale().shape
        # overview plot
        fig, axs = plt.subplots(nrows=2, sharex=True)
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
                      mode="expand", borderaxespad=0, ncol=3)

        # plt.imshow(self.sget_image_grayscale())
        axs[1].imshow(self.sget_image_classification(), interpolation='none', aspect='auto', cmap='gray')

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


# TODO: overwrite or delete
def ImageSample_from_Image(Img: Image) -> ImageSample:
    # initialize
    ImgP = ImageSample(section=(0, 1), window='none', data_type='none')
    # copy data over
    ImgP._image_original = Img._image_original
    ImgP.image_type = Img.image_type
    return ImgP


def ImageROI_from_ImageSample(ImgP: ImageSample) -> ImageROI:
    ImgROI = ImageROI(
        ImgP._section, ImgP._window, ImgP._data_type, 
        image_type_default=ImgP.current_image
    )
    ImgROI.obj_color = ImgP.sget_obj_color()
    ImgROI._image_original = ImgP.sget_sample_area()[0]
    ImgROI.xywh_ROI = ImgP.sget_sample_area()[1]
    return ImgROI

def ImageClassified_from_ImageROI(ImgROI: ImageROI) -> ImageClassified:
    ImgC = ImageClassified(
        section=ImgROI._section, 
        window=ImgROI._window, 
        image=ImgROI.sget_image_original()
    )
    ImgC._image_classification = ImgROI.sget_classification_adaptive_mean()[0]
    ImgC.obj_color = ImgROI.sget_obj_color()
    return ImgC
    


def full_initialization_standard_params(
        section, window, plts=False, verbose=False, only_steps={1, 2, 3}
):
    """Initialize a section and window with standard parameters."""
    if 1 not in only_steps:
        pass
    else:
        ISample = ImageSample(section, window)
        ISample.obj_color = 'light'
        ISample.plts = plts
        ISample.verbose = verbose
        ISample.sget_sample_area()
        ISample.save()
        del ISample

    if 2 not in only_steps:
        pass
    else:
        IROI = ImageROI(section, window)
        IROI.plts = plts
        IROI.verbose = verbose
        IROI.sget_classification_adaptive_mean()
        IROI.save()
        del IROI

    if 3 not in only_steps:
        pass
    else:
        if window.lower() != 'xrf':
            downscale_factor = 1 / 16
        elif window.lower() == 'xrf':
            downscale_factor = 1

        peak_prominence = .1
        max_slope = .1

        IClassified = ImageClassified(section, window)
        IClassified.plts = plts
        IClassified.verbose = verbose
        IClassified.set_laminae_params_table(
            peak_prominence=peak_prominence,
            max_slope=max_slope,
            downscale_factor=downscale_factor)
        IClassified.save()
        del IClassified


def full_initalization_section(section, **kwargs):
    """Initialize all windows in a given section."""
    for window in windows_all:
        full_initialization_standard_params(section, window, **kwargs)


def test_img_from_MSI(section=(490, 495), window='Alkenones'):
    I_obj = ImageSample(image_type_input='msi', section=section, window=window)
    # I_obj.classify_laminae_in_ROI()
    return I_obj


def test_img_from_XRF(section=(490, 495)):
    I_obj = Image(image_type_input='xrf', section=section, window='xrf')
    I_obj.classify_laminae_in_ROI()
    return I_obj
