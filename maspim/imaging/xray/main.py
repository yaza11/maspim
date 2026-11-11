"""
This module implements the XRay class, which offers to obtain xray regions from specific depths.
"""
from typing import Self

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.stats import linregress
from skimage.transform import rotate

from maspim.imaging.main import ImageSample, ImageROI
from maspim.imaging.util.image_convert_types import ensure_image_is_gray
from maspim.imaging.util.image_plotting import plt_cv2_image

logger = logging.getLogger(__name__)


class XRayROI(ImageROI):
    """ImageROI of a depth section of an XRay scan (see XRay.get_roi_from_section)."""


class XRay(ImageSample):
    """
    XRay class. Build around the image sample class.

    This class allows to obtain ImageROI objects for specific depth sections
    within the X-Ray scan of a core. It follows the same logic as the other
    image classes: every step is set explicitly and if a prerequisite is
    missing, the error message names the set method to call first.

    Example usage
    -------------
    create a new object
    >>> from maspim import XRay
    >>> xray = XRay(path_folder='your/path/to/the/folder')
    >>> xray.set_image(path_image_file='your/path/to/the/image/file', obj_color='dark')
    >>> xray.set_depth_section((100, 200))  # depth covered by the scan in cm
    the scan may be slightly rotated. The rotation can be found from the main
    contour, a rotated version of the scan is a new instance
    >>> xray.set_foreground_mask()
    >>> xray.set_simplified_image()
    >>> xray.set_main_contour()
    >>> xray.set_rotation()
    >>> xray = xray.get_rotated()
    set the sample area
    >>> xray.set_foreground_mask()
    >>> xray.set_sample_area()
    oftentimes the plastic liner is included in the determined sample area
    because it has the same color as the sediment. In that case the following
    method can help
    >>> xray.remove_bars()
    save result into the folder
    >>> xray.save()

    load a saved object
    >>> xray = XRay.load_from_file(path_folder='your/path/to/the/folder')

    obtain an XRayROI object for a specific depth section
    the depth section can be specified in multiple ways but always expects
    values in same unit as the depth_section
    >>> roi: XRayROI = xray.get_roi_from_section((100, 105))
    or
    >>> roi: XRayROI = xray.get_roi_from_section(section_start=100, section_end=105)
    or
    >>> roi: XRayROI = xray.get_roi_from_section(section_start=100, section_length=5)

    using the same syntax, it is also possible to only return the array
    >>> img: np.ndarray = xray.get_section((100, 105))
    """
    depth_section: tuple[float, float] = None
    angle: float = None
    center_xy: tuple[float, float] = None
    is_rotated: bool = False
    bars_removed: bool = False

    _save_attrs: set[str] = {
        'age_span',
        'average_width_yearly_cycle',
        'image_file',
        'image',
        'obj_color',
        'xywh_ROI',
        'main_contour',
        'height_width',
        'mask_foreground',
        'min_intensity_foreground',
        'depth_section',
        'angle',
        'center_xy',
        'is_rotated',
        'bars_removed'
    }

    def set_depth_section(self, depth_section: tuple[float, float]) -> None:
        """Set the depth section covered by the scan as (top, bottom) in cm."""
        assert depth_section[0] < depth_section[1], \
            ('first value should be the depth closer to the surface, not ' +
             f'{depth_section}')
        self.depth_section: tuple[float, float] = depth_section

    @staticmethod
    def _section_args_to_tuple(
            section_start: int | float | tuple[int | float, int | float],
            section_end: int | float | None = None,
            section_length: int | float = 5,
    ) -> tuple[float | int, float | int]:
        """
        Interpret arguments for specifying a depth section. By default, it is assumed that a section is 5 cm long.

        Parameters
        ----------
        section_start: int | float | tuple[int | float, int | float]
            The start of the section or end and start as tuple
        section_end: int | float | None (default = None)
            The end of the section. If not specified, will be calculated from section_length
        section_length: int | float (default = 5)
            The length of the section. Will not be used if section_end is specified.

        Returns
        -------
        tuple[float | int, float | int]
            The start and end of the section.
        """
        if isinstance(section_start, tuple | list):
            # provided section_start actually was a tuple
            section_start, section_end = section_start
        elif section_end is None:
            # section end not specified
            section_end = section_start + section_length

        assert section_start < section_end, \
            ('first value should be the depth closer to the surface, not ' +
             f'{(section_start, section_end)}')

        return section_start, section_end

    def set_rotation(
            self,
            angle: float | None = None,
            center_xy: tuple[int | float, int | float] | None = None
    ) -> None:
        """
        Set the rotation of the scan.

        If no angle is provided, use cv's minAreaRect function on the main
        contour to find the angle and center. Use get_rotated to obtain the
        rotated instance.

        Parameters
        ----------
        angle: float, optional
            Rotation angle in degrees.
        center_xy: tuple[float, float], optional
            Center of the rotation. Defaults to the center of the image if an
            angle is provided.
        """
        self._check_is_set('image', 'set_image')
        if angle is None:
            self._check_is_set('main_contour', 'set_main_contour')
            # angle in clockwise direction
            center_xy, _, angle = cv2.minAreaRect(self.main_contour)
            angle = angle % 90
            if angle > 45:
                angle -= 90
        elif center_xy is None:
            h, w = self.image.shape[:2]
            center_xy = (w / 2 - .5, h / 2 - .5)

        logger.info(
            f'found an angle of {angle:.1f} degrees ' +
            f'centered at {center_xy}'
        )

        self.center_xy: tuple[int | float, int | float] = center_xy
        self.angle: float = angle

    def get_rotated(self) -> Self:
        """
        Return a new instance in which image and foreground mask (if set) are
        rotated by the angle set with set_rotation. Depth section and object
        color are copied, the sample area has to be set on the new instance.
        """
        if self.is_rotated:
            raise ValueError('this instance is already rotated')
        self._check_is_set('image', 'set_image')
        self._check_is_set('angle', 'set_rotation')

        center_x, center_y = self.center_xy

        def rotate_image(image: np.ndarray) -> np.ndarray:
            # skimage rotates counter-clockwise, minAreaRect defines clockwise
            # center is expected as (col, row)
            return rotate(
                image,
                angle=self.angle,
                center=(center_x, center_y),
                preserve_range=True
            )

        new: Self = self.__class__(path_folder=self.path_folder)
        new.set_image(
            image=rotate_image(self.image).astype(self.image.dtype),
            image_type='cv',
            obj_color=self.obj_color
        )
        if self.mask_foreground is not None:
            new.set_foreground_mask_from_other(
                rotate_image(self.mask_foreground.astype(float)) > .5,
                thr_background=self.min_intensity_foreground
            )
        new.depth_section = self.depth_section
        new.age_span = self.age_span
        new.angle = self.angle
        new.center_xy = self.center_xy
        new.is_rotated = True
        return new

    def _get_section_slice(
            self,
            section_start: int | float | tuple[int | float, int | float],
            section_end: int | float | None = None,
            section_length: int | float = 5
    ) -> slice:
        """Columns of the sample area corresponding to a depth section."""
        self._check_is_set('depth_section', 'set_depth_section')
        self._check_is_set('xywh_ROI', 'set_sample_area')
        section_start, section_end = self._section_args_to_tuple(
            section_start, section_end, section_length
        )
        # length of section in cm
        section_core: float | int = self.depth_section[1] - self.depth_section[0]

        # find indices corresponding to depth
        w = self.xywh_ROI[2]
        return slice(
            round(w * (section_start - self.depth_section[0]) / section_core),
            round(w * (section_end - self.depth_section[0]) / section_core)
        )

    def get_section(
            self,
            section_start: int | float | tuple[int | float, int | float],
            section_end: int | float | None = None,
            section_length: int | float = 5,
            plts: bool = False
    ) -> np.ndarray[int]:
        """
        Crop a depth section from the sample area of the core image and return it.

        section_start can be a float or a tuple. If it is a tuple,
        the first value will be used as start and the second as end depth.
        Otherwise, if section_end is not specified, it will be inferred from
        the section_length.

        Parameters
        ----------
        section_start: int | float | tuple[int | float, int | float]
            The start of the section or end and start as tuple
        section_end: int | float | None (default = None)
            The end of the section. If not specified, will be calculated from
            section_length
        section_length: int | float (default = 5)
            The length of the section. Will not be used if section_end is specified.
        plts: bool (default = False)
            Whether to create a plot where the desired section is highlighted.

        Returns
        -------
        np.ndarray[int]
            The desired depth section of the xray image.

        """
        mask: slice = self._get_section_slice(section_start, section_end, section_length)
        roi: np.ndarray = self.get_sample_area_from_xywh()

        if plts:
            section_start, section_end = self._section_args_to_tuple(
                section_start, section_end, section_length
            )
            section_core: float | int = self.depth_section[1] - self.depth_section[0]
            # plot the desired section as a red rectangle on the XRay ROI
            plt_cv2_image(roi, hold=True)
            w, h = self.xywh_ROI[2:]
            plt.vlines(
                np.linspace(
                    start=0,
                    stop=w,
                    num=int(section_core // (section_end - section_start)) + 1,
                    endpoint=True
                )[1:-1],
                ymin=0, ymax=h, colors='r'
            )
            plt.fill_between(x=(mask.start, mask.stop), y1=h, color='r', alpha=.2)
            plt.gca().axis('off')
            plt.title(f'Section between {section_start} and {section_end} cm')
            plt.show()

        return roi[:, mask]

    def get_roi_from_section(
            self,
            section_start: int | float | tuple[int | float, int | float],
            section_end: int | float | None = None,
            section_length: int | float = 5,
            path_folder: str | None = None,
            plts: bool = False,
            **_
    ) -> XRayROI:
        """
        Get an XRayROI object from a specific section.

        Image and foreground mask (if set) of the section are passed on.

        Parameters
        ----------
        section_start: int | float | tuple[int | float, int | float]
            The start of the section or end and start as tuple
        section_end: int | float | None (default = None)
            The end of the section. If not specified, will be calculated from section_length
        section_length: int | float (default = 5)
            The length of the section. Will not be used if section_end is specified.
        path_folder: str, optional
            Folder of the new object. Defaults to the folder of this object.
        plts: bool (default = False)
            Whether to create a plot where the desired section is highlighted.

        Returns
        -------
        roi: XRayROI
            The desired depth section of the xray image as an XRayROI object.

        """
        image: np.ndarray = self.get_section(
            section_start, section_end, section_length, plts=plts
        )
        roi: XRayROI = XRayROI(
            path_folder=path_folder if path_folder is not None else self.path_folder
        )
        roi.set_image(image=image, image_type='cv', obj_color=self.obj_color)
        if self.mask_foreground is not None:
            x, y, w, h = self.xywh_ROI
            mask_section: np.ndarray = self.mask_foreground[y:y + h, x:x + w][
                :, self._get_section_slice(section_start, section_end, section_length)
            ]
            roi.set_foreground_mask_from_other(
                mask_section, thr_background=self.min_intensity_foreground
            )
        return roi

    def remove_bars(self, n_sections: int = 10, plts: bool = False, **_) -> None:
        """
        Remove liner bars at top and bottom of the image.

        This function determines inflection points in n_sections depth-wise
        average brightnesses to determine the positioning of the liner along
        those transects. A line is then fitted through the top and bottom
        boundary respectively and values outside the area between those lines
        nullified. The sample area is then updated.

        This function assumes that the background is white.

        The transects ideally look something like this (inflection points
        marked with x):
              ___                ___
            |    |             |    |
           |      x            x     |
           |      |___________|      |
           |                         |
        __|                           |______

        Parameters
        ----------
        n_sections : int (default = 10)
            The number of transects to be used for the line fits.
        plts : bool (default = False)
            Plot the estimated position of liners and cleaned image

        Returns
        -------
        None
        """
        def find_bounds(image_section_: np.ndarray) -> tuple[int, int]:
            """Determine the inflection points in a given section."""
            # average out in the depth-wise direction
            brightness_1d: np.ndarray[float] = np.mean(image_section_, axis=1)
            n: int = len(brightness_1d)  # length of transect
            # right and left side of the core
            upper: np.ndarray[float] = brightness_1d[:n // 2]
            lower = brightness_1d[n // 2:]
            # discard everything outside center of dark bar
            upper_crop_idx: int = np.argmin(upper)
            lower_crop_idx: int = np.argmin(lower)
            upper_c: np.ndarray[float] = upper[upper_crop_idx:]
            lower_c: np.ndarray[float] = lower[:lower_crop_idx]
            # set boundary where signal starts to drop (~ point of maximum change)
            # can't rely on bright gap between casing and sediment
            upper_infl: int = np.argwhere(np.diff(upper_c) > 1)[-1][0]  # rising flank
            lower_infl: int = np.argwhere(np.diff(lower_c) < -1)[0][0]  # falling flank
            # shift back
            upper_infl += upper_crop_idx
            lower_infl += n // 2
            return upper_infl, lower_infl

        if self.bars_removed:
            logger.warning('bars already removed, exiting method')
            return
        self._check_is_set('obj_color', 'set_image (with obj_color)', 'infer_obj_color')
        self._check_is_set('depth_section', 'set_depth_section')
        self._check_is_set('xywh_ROI', 'set_sample_area')

        image_sample_area: np.ndarray = self.get_sample_area_from_xywh()

        # determine upper and lower bounds for all sections
        section_length: float = (self.depth_section[1] - self.depth_section[0]) / n_sections
        upper_bounds: np.ndarray[int] = np.zeros(n_sections, dtype=int)
        lower_bounds: np.ndarray[int] = np.zeros(n_sections, dtype=int)
        for i_section in range(n_sections):
            section_start: float = self.depth_section[0] + i_section * section_length
            image_section: np.ndarray = ensure_image_is_gray(self.get_section(
                section_start, section_length=section_length
            ))
            if self.obj_color == 'light':
                # invert image section because find bounds assumes white background
                image_section: np.ndarray = image_section.max() - image_section

            u, l = find_bounds(image_section)
            upper_bounds[i_section] = u
            lower_bounds[i_section] = l

        # lin fit
        # center points of sections
        x_c: np.ndarray = np.linspace(0, self.xywh_ROI[2], n_sections + 2, endpoint=True)[1:-1]
        # x pixel values in image
        xs: np.ndarray = np.arange(0, self.xywh_ROI[2])
        upper_m, upper_b, *_ = linregress(x_c, upper_bounds)
        # values of line
        upper_bound = upper_b + upper_m * xs

        lower_m, lower_b, *_ = linregress(x_c, lower_bounds)
        # values of line
        lower_bound = lower_b + lower_m * xs

        if plts:
            # plot determined inflection points and boundaries on image
            plt.figure()
            plt.imshow(image_sample_area)
            plt.scatter(x_c, upper_bounds)
            plt.plot(xs, upper_bound)
            plt.scatter(x_c, lower_bounds)
            plt.plot(xs, lower_bound)
            plt.show()

        # set everything outside bounds to 255 (or 0)
        if self.obj_color == 'dark':
            fill_val: int = image_sample_area.max()
        else:
            fill_val: int = image_sample_area.min()
        _, Y = np.meshgrid(np.arange(0, self.xywh_ROI[2]), np.arange(0, self.xywh_ROI[3]))
        mask: np.ndarray[bool] = (Y < upper_bound) | (Y > lower_bound)
        temp_roi = image_sample_area.copy()
        temp_roi[mask] = fill_val
        # recast to fit ROI to new sample
        x, y, w, h = self.xywh_ROI
        upper_new = int(np.min(upper_bound))  # floor
        lower_new = int(np.max(lower_bound) + 1)  # ceil
        y_new = y + upper_new
        h_new = lower_new - upper_new
        # update image and sample area
        image: np.ndarray = self.image.copy()
        image[y:y + h, x:x + w] = temp_roi
        self.image = image
        self.__dict__.pop('image_grayscale', None)
        # the image no longer matches the file, so it has to be saved
        self.image_file = None
        self.xywh_ROI = (x, y_new, w, h_new)
        self.bars_removed: bool = True

        if plts:
            # plot resulting new image ROI
            plt_cv2_image(self.get_sample_area_from_xywh())
