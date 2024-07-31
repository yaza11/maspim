"""
This module implements the XRay class, which offers to obtain xray regions from specific depths.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.stats import linregress

from msi_workflow.imaging.main import ImageSample, ImageROI
from msi_workflow.imaging.util.image_convert_types import ensure_image_is_gray
from msi_workflow.imaging.util.image_plotting import plt_cv2_image

logger = logging.getLogger(__name__)


class XRay(ImageSample):
    """
    XRay class. Build around the image sample class.

    This class allows to obtain ImageSample objects for specific sections within the X-Ray scan.

    Example usage
    -------------
    create a new object
    >>> xray = XRay(
    >>>     path='your/path/to/the/image/file',
    >>>     depth_section=(100, 200),  # span of the entire image in cm
    >>>     obj_color = 'dark'  # sediment color compared to background
    >>> )
    set and get the sample area
    >>> xray.sget_sample_area()
    oftentimes the plastic liner is included in the determined sample area
    because it has the same color as the sediment. In that case the following method can help
    >>> xray.remove_bars()
    save result into folder where the image file is from
    >>> xray.save()

    if loaded from file, it is not necessary to specify the object color and depth section arguments again
    >>> xray = XRay(path='your/path/to/the/image/file')
    >>> xray.load()

    obtain an ImageROI object for a specific depth section
    the depth section can be specified in multiple ways but always expects values in same unit as the depth_section
    specified upon initialization
    >>> img: ImageROI = xray.get_ImageROI_from_section((100, 105))
    or
    >>> img: ImageROI = xray.get_ImageROI_from_section(section_start=100, section_end=105)
    or
    >>> img: ImageROI = xray.get_ImageROI_from_section(section_start=100, section_length=5)

    using the same syntax, it is also possible to only return the array
    >>> img: np.ndarray = xray.get_section((100, 105))

    """

    def __init__(
            self,
            depth_section: tuple[float, float],
            path_folder: str | None = None,
            image: np.ndarray[float | int] | None = None,
            image_type: str = 'cv',
            path_image_file: str | None = None,
            obj_color: str | None = None
    ) -> None:
        """
        Initialize the object.

        If it is intended to load the object from disc after initialization, it suffices to provide the path_image_file.

        Parameters
        ----------
        path_image_file: str
            The image file with the XRay measurement.
        depth_section: tuple[float, float] | None
            The depth section covered by the photo as a tuple in cm.
            This is required if the object is not loaded from disc
        obj_color: str (default= 'dark')
            The relative brightness of the object compared to background. Options are 'dark', 'light'.
        """

        super().__init__(
            path_folder=path_folder,
            image=image,
            image_type=image_type,
            path_image_file=path_image_file,
            obj_color=obj_color
        )

        self.depth_section = depth_section
        self._bars_removed: bool = False

    def _section_args_to_tuple(
            self,
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
        if isinstance(section_start, tuple):
            # provided section_start actually was a tuple
            section_start, section_end = section_start
        elif section_end is None:
            # section end not specified
            section_end = section_start + section_length

        assert section_start < section_end, \
            f'first value should be the depth closer to the surface, not {(section_start, section_end)}'

        return section_start, section_end

    def get_section(
            self,
            section_start: int | float | tuple[int | float, int | float],
            section_end: int | float | None = None,
            section_length: int | float = 5,
            plts: bool = False
    ) -> np.ndarray[int]:
        """
        Crop a depth section from the core image and return it.
        
        section_start can be a float or a tuple. If it is a tuple, 
        the first value will be used as start and the second as end depth. 
        Otherwise, if section_end is not specified, it will be inferred from the section_length.

        Parameters
        ----------
        section_start: int | float | tuple[int | float, int | float]
            The start of the section or end and start as tuple
        section_end: int | float | None (default = None)
            The end of the section. If not specified, will be calculated from section_length
        section_length: int | float (default = 5)
            The length of the section. Will not be used if section_end is specified.
        plts: bool (default = False)
            Whether to create a plot where the desired section is highlighted.

        Returns
        -------
        np.ndarray[int]
            The desired depth section of the xray image.

        """

        assert self.depth_section is not None, 'specify the depth section if not loaded'
        section_start, section_end = self._section_args_to_tuple(
            section_start, section_end, section_length
        )
        roi: np.ndarray = self.image_sample_area
        # length of section in cm
        section_core: float | int = self.depth_section[1] - self.depth_section[0]

        # find indices corresponding to depth
        w = self._xywh_ROI[2]
        mask: slice = slice(
            round(w * (section_start - self.depth_section[0]) / section_core),
            round(w * (section_end - self.depth_section[0]) / section_core)
        )

        if plts:
            # plot the desired section as a red rectangle on the XRay ROI
            plt_cv2_image(roi, hold=True)
            w, h = self._xywh_ROI[2:]
            plt.vlines(
                np.linspace(
                    start=0,
                    stop=w,
                    num=int(section_core // section_length) + 1,
                    endpoint=True
                )[1:-1],
                ymin=0, ymax=h, colors='r'
            )
            plt.fill_between(x=(mask.start, mask.stop), y1=h, color='r', alpha=.2)
            plt.gca().axis('off')
            plt.title(f'Section between {section_start} and {section_end} cm')
            plt.show()

        return roi[:, mask]

    def get_ImageROI_from_section(
            self,
            section_start: int | float | tuple[int | float, int | float],
            section_end: int | float | None = None,
            section_length: int | float = 5,
            **kwargs: dict
    ) -> ImageROI:
        """
        Get an ImageROI object from a specific section.

        Parameters
        ----------
        section_start: int | float | tuple[int | float, int | float]
            The start of the section or end and start as tuple
        section_end: int | float | None (default = None)
            The end of the section. If not specified, will be calculated from section_length
        section_length: int | float (default = 5)
            The length of the section. Will not be used if section_end is specified.
        plts: bool (default = False)
            Whether to create a plot where the desired section is highlighted.

        Returns
        -------
        roi: ImageROI
            The desired depth section of the xray image as an ImageROI object.

        """
        depth_section: tuple[float | int, float | int] = self._section_args_to_tuple(
            section_start, section_end, section_length
        )
        img: np.ndarray[int] = self.get_section(depth_section, **kwargs)
        roi: ImageROI = ImageROI(
            path_folder=os.path.dirname(self.path_image_file),
            image=img,
            obj_color=self.obj_color
        )
        return roi

    def remove_bars(self, n_sections: int = 10, plts: bool = False) -> None:
        """
        Remove liner bars at top and bottom of the image.

        This function determines inflection points in n_sections depth-wise
        average brightnesses to determine the positioning of the liner along
        those transects. A line is then fitted through the top and bottom
        boundary respectively and values outside the area between those lines
        nullified. The roi is then updated.

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

        if self._bars_removed:
            logger.warning('bars already removed, exiting method')
            return

        # determine upper and lower bounds for all sections
        section_length: float = (self.depth_section[1] - self.depth_section[0]) / n_sections
        upper_bounds: np.ndarray[int] = np.zeros(n_sections, dtype=int)
        lower_bounds: np.ndarray[int] = np.zeros(n_sections, dtype=int)
        for i_section in range(n_sections):
            section_end: float = self.depth_section[0] + i_section * section_length
            image_section: np.ndarray = ensure_image_is_gray(self.get_section(
                section_end, section_length=section_length
            ))
            if self.obj_color == 'light':
                # invert image section because find bounds assumes white background
                image_section: np.ndarray = image_section.max() - image_section

            u, l = find_bounds(image_section)
            upper_bounds[i_section] = u
            lower_bounds[i_section] = l

        # lin fit
        # center points of sections
        x_c: np.ndarray = np.linspace(0, self._xywh_ROI[2], n_sections + 2, endpoint=True)[1:-1]
        # x pixel values in image
        xs: np.ndarray = np.arange(0, self._xywh_ROI[2])
        upper_m, upper_b, *_ = linregress(x_c, upper_bounds)
        # values of line
        upper_bound = upper_b + upper_m * xs

        lower_m, lower_b, *_ = linregress(x_c, lower_bounds)
        # values of line
        lower_bound = lower_b + lower_m * xs

        if plts:
            # plot determined inflection points and boundaries on image
            plt.figure()
            plt.imshow(self.image_sample_area)
            plt.scatter(x_c, upper_bounds)
            plt.plot(xs, upper_bound)
            plt.scatter(x_c, lower_bounds)
            plt.plot(xs, lower_bound)
            plt.show()

        # set everything outside bounds to 255 (or 0)
        if self.obj_color == 'dark':
            fill_val: int = self.image_sample_area.max()
        else:
            fill_val: int = self.image_sample_area.min()
        _, Y = np.meshgrid(np.arange(0, self._xywh_ROI[2]), np.arange(0, self._xywh_ROI[3]))
        mask: np.ndarray[bool] = (Y < upper_bound) | (Y > lower_bound)
        temp_roi = self.image_sample_area.copy()
        temp_roi[mask] = fill_val
        # recast to fit ROI to new sample
        x, y, w, h = self._xywh_ROI
        upper_new = int(np.min(upper_bound))  # floor
        lower_new = int(np.max(lower_bound) + 1)  # ceil
        y_new = y + upper_new
        h_new = lower_new - upper_new
        # update extent of ROI
        self._xywh_ROI = (x, y_new, w, h_new)
        self._image[y:y + h, x:x + w] = temp_roi
        self._bars_removed: bool = True

        if plts:
            # plot resulting new image ROI
            plt_cv2_image(self.image_sample_area)
