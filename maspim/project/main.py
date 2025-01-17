"""
This module contains the project class which is used to manage various objects for XRF and MSI measurements.
"""
import functools
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from matplotlib import patches
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Iterable, Self, Any
from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw

from maspim.data.helpers import plot_comp, transform_feature_table, plot_comp_on_image, get_comp_as_img
from maspim.exporting.legacy.data_analysis_export import DataAnalysisExport
from maspim.exporting.legacy.ion_image import (get_da_export_ion_image,
                                               get_da_export_data)
from maspim.imaging.util.image_boxes import region_in_box
from maspim.imaging.util.image_geometry import ROI
from maspim.imaging.util.image_plotting import plt_rect_on_image, plt_cv2_image
from maspim.time_series.helpers import get_averaged_tables

from maspim.util import Convenience

from maspim.exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from maspim.exporting.from_mcf.helper import Spectrum, get_mzs_for_limits
from maspim.exporting.from_mcf.spectrum import Spectra, MultiSectionSpectra
from maspim.exporting.sqlite_mcf_communicator.hdf import hdf5Handler

from maspim.data.msi import MSI
from maspim.data.xrf import XRF
from maspim.data.age_model import AgeModel

from maspim.project.file_helpers import (
    get_folder_structure, find_files, get_mis_file, get_d_folder,
    search_keys_in_xml, get_image_file, find_matches, ImagingInfoXML, get_rxy,
    get_spots
)

from maspim.imaging.main import ImageSample, ImageROI, ImageClassified
from maspim.imaging.util.image_convert_types import (
    ensure_image_is_gray, PIL_to_np, convert
)
from maspim.imaging.util.coordinate_transformations import rescale_values
from maspim.imaging.util.find_xrf_roi import find_ROI_in_image, plt_match_template_scale
from maspim.imaging.xray.main import XRay, XRayROI
from maspim.imaging.register.transformation import Transformation
from maspim.imaging.register.helpers import Mapper

from maspim.time_series.main import TimeSeries
from maspim.time_series.proxy import UK37
from maspim.util.convenience import check_attr, object_to_string
from maspim.util.read_msi_align import get_teaching_points, get_teaching_point_pairings_dict

PIL_Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class SampleImageHandlerMSI(Convenience):
    """
    Given the mis file and folder, find image and area (of MSI) of sample.

    Example usage:
    # initialize
    i_handler = SampleImageHandlerMSI(path_folder='path/to/your/msi/folder')
    # read the extent of the data from the reader (if provided) or the ImagingInfoXML file
    i_handler.set_extent_data()
    # set the photo specified in the mis file (PIL Image)
    i_handler.set_photo()
    # only after photo and data extent have been set, the set_photo_roi function becomes available
    # with match_pixels set to True, this will return the photo in the data ROI where each pixel
    # in the image corresponds to a data point
    roi = i_handler.set_photo_roi()
    # save the handler inside the d folder for faster future usage
    i_handler.save()


    # if an instance has been saved before, the handler can be loaded
    # initialize
    i_handler = SampleImageHandlerMSI(path_folder='path/to/your/msi/folder')
    i_handler.load()

    """
    path_folder: str | None = None
    d_folder: str | None = None
    image_file: str | None = None
    mis_file: str | None = None

    image: PIL_Image.Image | None = None
    _extent_spots: tuple[int, int, int, int] | None = None
    _data_roi_xywh: tuple[int, int, int, int] | None = None
    _photo_roi_xywh: tuple[int, int, int, int] | None = None

    _save_attrs = {
        '_extent_spots',
        'd_folder',
        'image_file',
        'mis_file',
        '_extent_spots'
        '_data_roi_xywh'
        '_photo_roi_xywh',
    }

    def __init__(
            self,
            path_folder: str,
            path_d_folder: str | None = None,
            path_mis_file: str | None = None
    ) -> None:
        """
        Initialize paths for folder, mis file, d folder and ImageSample object.

        Parameters
        ----------
        path_folder : str
            The folder containing the d folder, mis file and sample photos.
        path_d_folder : str
            The d folder inside the folder. If not provided, the folder name
            is searched inside the path_folder
            Specifying this is only necessary when multiple d folders are
            inside the folder.
        path_mis_file: str, optional
            Path and name of the mis file to use.

        Returns
        -------
        None.

        """
        self.path_folder: str = path_folder
        if path_mis_file is not None:
            self.mis_file: str = os.path.basename(path_mis_file)
        else:
            self.mis_file: str = get_mis_file(self.path_folder)
        if path_d_folder is not None:
            self.d_folder = os.path.basename(path_d_folder)
        else:
            self.d_folder = get_d_folder(self.path_folder)

        self.image_file = get_image_file(self.path_folder)

    @property
    def path_d_folder(self):
        return os.path.join(self.path_folder, self.d_folder)

    @property
    def path_mis_file(self):
        return os.path.join(self.path_folder, self.mis_file)

    @property
    def path_image_file(self):
        return os.path.join(self.path_folder, self.image_file)

    @functools.cached_property
    def image(self) -> PIL_Image.Image:
        """
        Set the photo from the determined file as PIL image.
        """
        image: PIL_Image.Image = PIL_Image.open(self.path_image_file)
        return image

    def set_extent_data(
            self,
            reader: ReadBrukerMCF | None = None,
            spot_info: ImagingInfoXML | pd.DataFrame | None = None
    ) -> None:
        """
        Get spot names from MCFREader and set extent of pixels based on that.

        Parameters
        ----------
        reader : ReadBrukerMCF, optional
            Reader to get the pixel indices. The default is None. If not speci-
            fied, this method will create a new ImagingInfoXML instance in this scope.
        spot_info : ImagingInfoXML, optional
            An ImagingInfoXML instance. If not specified, this method will
            call get_spots to try fetching spots from the xml or sqlite file.

        Returns
        -------
        None.

        """
        # no reader specified or reader does not have spots
        if (reader is None) or (not check_attr(reader, 'spots')):
            if spot_info is None:  # create new instance
                spot_info: pd.DataFrame = get_spots(
                    path_d_folder=self.path_d_folder
                )
            pixel_names = spot_info.spotName  # get pixel names from reader
        else:
            pixel_names = reader.spots.names  # get pixel names from reader
        # initialize the _extent
        xmin: int = 65536
        xmax: int = 0
        ymin: int = 65536
        ymax: int = 0
        for pixel_name in pixel_names:
            # values in line are separated by semicolons
            img_x = int(re.findall('X(.*)Y', pixel_name)[0])  # x coordinate
            img_y = int(re.findall('Y(.*)', pixel_name)[0])  # y coordinate
            # update boundaries if more extreme value is found
            if img_x > xmax:
                xmax: int = img_x
            if img_x < xmin:
                xmin: int = img_x
            if img_y > ymax:
                ymax: int = img_y
            if img_y < ymin:
                ymin: int = img_y
        self._extent_spots: tuple[int, int, int, int] = (xmin, xmax, ymin, ymax)

    @property
    def extent_spots(self) -> tuple[int, int, int, int]:
        if not check_attr(self, '_extent_spots'):
            self.set_extent_data()
        return self._extent_spots

    def _draw_measurement_area(self, canvas: PIL_Image.Image) -> PIL_Image.Image:
        assert check_attr(self, 'points'), 'call set_photo_roi'
        canvas = canvas.copy()
        draw = PIL_ImageDraw.Draw(canvas)

        # define linewidth in terms of image size
        linewidth = round(min(canvas._size[:2]) / 100)

        if len(self.points) < 3:  # for rectangle
            # p1 --> smaller x value
            points_: list[tuple[int, int]] = self.points.copy()
            # the PIL rectangle function is very specific about the order of points
            points_.sort()
            p1, p2 = points_
            # swap ys of p1 and p2
            if p1[1] > p2[1]:
                p1_ = (p1[0], p2[1])
                p2_ = (p2[0], p1[1])
                p1 = p1_
                p2 = p2_
            points_: list[tuple[int, int]] = [p1, p2]
            draw.rectangle(points_, outline=(255, 0, 0), width=linewidth)
        else:
            draw.polygon(self.points, outline=(255, 0, 0), width=linewidth)
        return canvas

    def set_photo_roi(
            self,
            match_roi_data: bool = True,
            plts: bool = False,
            **_
    ) -> None:
        """
        Match image and data pixels and set extent of the measurement area in
        data and photo pixels.

        Parameters
        ----------
        match_roi_data : bool, optional
            Whether to resize the image to the datapoints. The default is True.
            False will return the original image cropped to the measurement 
            region.
        plts : bool, optional
            Whether to plot inbetween results. The default is False.

        Returns
        -------
        None

        """
        assert check_attr(self, '_extent_spots'), 'call set_extent_data'

        # search the mis file for the point data and image file
        mis_dict: dict = search_keys_in_xml(self.path_mis_file, ['Point'])
        assert len(mis_dict) > 0, 'found no region in mis file'

        # get points specifying the measurement area
        points_mis: list[str] = mis_dict['Point']
        # format self.points
        self.points: list[tuple[int, int]] = []
        # get the points of the defined area
        for point in points_mis:
            p: tuple[int, int] = (int(point.split(',')[0]), int(point.split(',')[1]))
            self.points.append(p)

        if plts:
            canvas = self._draw_measurement_area(self.image)
            img = np.array(canvas)
            plt.figure()
            plt.imshow(img, interpolation='None')
            plt.show()

        # get the _extent of the image
        points_x: list[int] = [p[0] for p in self.points]
        points_y: list[int] = [p[1] for p in self.points]

        # the _extent of measurement area in pixel coordinates
        x_min_area: int = np.min(points_x)
        x_max_area: int = np.max(points_x)
        y_min_area: int = np.min(points_y)
        y_max_area: int = np.max(points_y)

        # get _extent of data points in txt-file
        x_min_FT, x_max_FT, y_min_FT, y_max_FT = self.extent_spots

        # resize region in photo to match data points
        if match_roi_data:
            img_resized = self.image.resize(
                (x_max_FT - x_min_FT + 1, y_max_FT - y_min_FT + 1),  # new number of pixels
                box=(x_min_area, y_min_area, x_max_area, y_max_area),  # area of photo
                resample=PIL_Image.Resampling.LANCZOS  # supposed to be best
            )
        else:  # crop original image to data region
            img_resized = self.image.crop(
                (x_min_area, y_min_area, x_max_area, y_max_area))
        # xywh of data ROI in original image, photo units
        xp: int = x_min_area  # lower left corner
        yp: int = y_min_area  # lower left corner
        wp: int = x_max_area - x_min_area  # width
        hp: int = y_max_area - y_min_area  # height
        # xywh of data, data units
        xd: int = x_min_FT
        yd: int = y_min_FT
        wd: int = x_max_FT - x_min_FT
        hd: int = y_max_FT - y_min_FT

        self._photo_roi_xywh: tuple[int, ...] = (xp, yp, wp, hp)  # photo units
        self._data_roi_xywh: tuple[int, ...] = (xd, yd, wd, hd)  # data units
        self._image_roi: np.ndarray[int] = img_resized

    @property
    def photo_roi_xywh(self) -> tuple[int, ...]:
        if not check_attr(self, '_photo_roi_xywh'):
            self.set_photo_roi()
        return self._photo_roi_xywh

    @property
    def data_roi_xywh(self) -> tuple[int, ...]:
        if not check_attr(self, '_data_roi_xywh'):
            self.set_photo_roi()
        return self._data_roi_xywh

    @property
    def image_roi(self):
        """Uses grayscale values resampled at data points."""
        if check_attr(self, '_image_roi'):
            return self._image_roi
        x_min_FT, x_max_FT, y_min_FT, y_max_FT = self.extent_spots
        x_min_area, y_min_area, wp, hp = self.photo_roi_xywh
        x_max_area, y_max_area = x_min_area + wp, y_min_area + hp

        self._image_roi = self.image.resize(
            (x_max_FT - x_min_FT + 1, y_max_FT - y_min_FT + 1),  # new number of pixels
            box=(x_min_area, y_min_area, x_max_area, y_max_area),  # area of photo
            resample=PIL_Image.Resampling.LANCZOS  # supposed to be best
        )

    def plot_shots(self, s=.1):
        """Plot positions of measurement points on the sample area."""
        # first, draw bounds of measurement area
        if not check_attr(self, 'points'):
            self.set_photo_roi()
        draw = self._draw_measurement_area(self.image)
        img = PIL_to_np(draw)

        # get data coordinates
        spots_df: pd.DataFrame = get_spots(self.path_d_folder)
        # convert to pixel coordinates using Data
        data: MSI = MSI(path_d_folder=self.path_d_folder)
        data.inject_feature_table_from(spots_df, supress_warnings=True)
        data.pixels_get_photo_ROI_to_ROI(data_ROI_xywh=self._data_roi_xywh,
                                         photo_ROI_xywh=self._photo_roi_xywh,
                                         image_ROI_xywh=(0, 0, img.shape[1], img.shape[0]))

        plt.imshow(img)

        plt.scatter(data.feature_table.x_ROI, data.feature_table.y_ROI, s=s)
        plt.show()

    def plot_overview(
            self, fig: plt.Figure | None = None, ax: plt.Axes | None = None, hold=False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plot the image with identified region of measurement."""
        if not check_attr(self, 'points'):
            self.set_photo_roi()
        draw = self._draw_measurement_area(self.image)
        img = PIL_to_np(draw)

        x, y, w, h = self.photo_roi_xywh
        rect_photo = patches.Rectangle((x, y), w, h, fill=False, edgecolor='g')

        if fig is None:
            assert ax is None, "If ax is provided, must also provide fig"
            fig, ax = plt.subplots()
        else:
            assert ax is not None, "If fig is provided, must also provide ax"
        plt.imshow(img)
        ax.add_patch(rect_photo)
        if hold:
            return fig, ax
        plt.show()


class SampleImageHandlerXRF(Convenience):
    """
    Image handler for XRF measurements. Finds the image region corresponding
    to the measurement area.

    If path_image_roi_file is not specified, it is assumed that the image file
    ROI is the same as the measurement area.
    path_folder is only used for loading and saving the object.
    Since the object is saved with a non-unique name, multiple measurements
    should be located in separate folders.

    The path_image_roi_file should be a txt file ending in _Video.txt

    Example usage:
    # initialize
    # make sure to pass both the image_file and image_roi_file if the
    # measurement does not cover the entire sediment sample!
    i_handler = SampleImageHandlerXRF(
        path_folder='path/to/your/xrf/measurement',
        path_image_file='path/to/your/photo',
        path_image_roi_file='path/to/your/video/file'
    )
    # make sure to call set_photo before set_extent_data()
    # set the photo of the sample and the ROI
    # if path_image_roi_file is not specified, image will be the same as _image_roi
    i_handler.set_photo()

    # read the extent of the data
    i_handler.set_extent_data()

    # only after photo and data extent have been set, the set_photo_roi function
    # becomes available with match_pixels set to True, this will return the
    # photo in the data ROI where each pixel in the image corresponds to a data point
    roi = i_handler.set_photo_roi()
    # save the handler inside the d folder for faster future usage
    i_handler.save()

    # if an instance has been saved before, the handler can be loaded
    # initialize
    i_handler = SampleImageHandlerMSI(path_folder='path/to/your/msi/folder')
    i_handler.load()
    """
    path_folder: str | None = None
    image_file: str | None = None
    image_roi_file: str | None = None
    roi_is_image: bool | None = None

    image: PIL_Image.Image | None = None
    _extent: tuple[int, int, int, int] | None = None
    _extent_spots: tuple[int, int, int, int] | None = None
    _data_roi_xywh: tuple[int, int, int, int] | None = None
    _photo_roi_xywh: tuple[int, int, int, int] | None = None
    _scale_conversion: float | None = None

    _save_attrs: set[str] = {
        'image_file',
        'image_roi_file',
        'roi_is_image',
        '_extent',
        '_extent_spots',
        '_data_roi_xywh',
        '_photo_roi_xywh',
        '_scale_conversion'
    }

    def __init__(
            self,
            path_folder: str | None = None,
            path_image_file: str | None = None,
            path_image_roi_file: str | None = None
    ) -> None:
        """
        Initialize. Set paths.

        Parameters
        ----------
        path_folder : str, optional
            Folder to load and save the object
        path_image_file: str
            The file with the photo of the sample
        path_image_roi_file: str, optional
            If not provided, it is assumed that no photo of the sample exists.

        Returns
        -------
        None.

        """
        assert (path_folder is not None) or (path_image_file is not None), \
            "provide either a folder or the image file"

        if path_folder is not None:
            self.path_folder: str = path_folder
        elif path_image_file is not None:
            self.path_folder: str = os.path.dirname(path_image_file)

        if path_image_roi_file is None:
            path_image_roi_file: str = path_image_file

        self.roi_is_image: bool = path_image_file == path_image_roi_file
        if path_image_file is not None:
            self.image_file: str = os.path.basename(path_image_file)
            self.image_roi_file: str = os.path.basename(path_image_roi_file)

    @property
    def path_image_file(self):
        assert self.image_file is not None
        assert self.path_folder is not None
        return os.path.join(self.path_folder, self.image_file)

    @property
    def path_image_roi_file(self):
        assert self.image_roi_file is not None
        assert self.path_folder is not None
        return os.path.join(self.path_folder, self.image_roi_file)

    def set_photo(self) -> None:
        """Set the image and ROI from the files."""

        def txt2uint8(path_file: str) -> np.ndarray[np.uint8]:
            """Convert a txt file containing values seperated by ; into an array."""
            arr: np.ndarray = pd.read_csv(path_file, sep=';').to_numpy()
            # convert int64 to uint8
            arr: np.ndarray[np.uint8] = (arr / 2 ** 8).astype(np.uint8)
            return arr

        if os.path.splitext(self.path_image_file)[1] == '.txt':
            arr: np.ndarray[np.uint8] = txt2uint8(self.path_image_file)
            self.image: PIL_Image.Image = PIL_Image.fromarray(arr, 'L')
        else:
            self.image: PIL_Image.Image = PIL_Image.open(self.path_image_file)

        if self.roi_is_image:
            self._image_roi: PIL_Image.Image = self.image.copy()
        else:
            arr: np.ndarray[np.uint8] = txt2uint8(self.path_image_roi_file)
            self._image_roi: PIL_Image.Image = PIL_Image.fromarray(arr, 'L')

    def set_extent_data(self, **kwargs: dict) -> None:
        """
        Set the extent of the spots. This function has to be called after set_photo

        If the image is the same as the ROI, this will simply be the full extent.
        Otherwise, cv2's template matching function will be evoked with varying
        scaling factors to obtain the precise position and scale of the ROI
        inside the of the image.

        Parameters
        ----------
        kwargs: dict
            Optional parameters for find_ROI_in_image.

        Returns
        -------
        None
        """
        assert check_attr(self, 'image'), 'call set_photo first'

        if self.roi_is_image:
            # xmin, xmax, ymin, ymax
            self._extent_spots: tuple[int, ...] = (
                0, self.image._size[1], 0, self.image._size[0]
            )
        else:
            # function expects arrays, not PIL_Images
            loc, scale = find_ROI_in_image(
                image=PIL_to_np(self.image),
                image_roi=PIL_to_np(self._image_roi),
                **kwargs
            )

            # convert image_roi resolution to image resolution
            self._scale_conversion: float = scale
            # xmin, xmax, ymin, ymax
            self._extent = (
                loc[0],
                loc[0] + round(scale * self._image_roi.size[0]),  # size is (width, height)
                loc[1],
                loc[1] + round(scale * self._image_roi.size[1])
            )
            self._extent_spots = tuple(round(p / scale) for p in self._extent)
        logger.info(
            f'found the extent of the data to be {self._extent} (pixel coordinates)'
        )

    def plot_extent_data(self):
        """
        Using the scale conversion factor and top left corner of the ROI,
        plot an image where the section corresponding to the ROI is replaced.
        This function can only be used after set_extent_data

        """
        assert check_attr(self, '_scale_conversion'), 'call set_extent_data'

        scale: float = self._scale_conversion
        loc: tuple[int, int] = (self._extent[0], self._extent[2])

        plt_match_template_scale(
            image=ensure_image_is_gray(PIL_to_np(self.image)),
            template=PIL_to_np(self._image_roi),
            loc=loc,
            scale=scale
        )

    def set_photo_roi(self, match_pxls: bool = True) -> None:
        """
        Get the image ROI corresponding to the measurement area.

        Parameters
        ----------
        match_pxls: bool, the default is True.
            If this is set to True, the returned image ROI will have the same
            resolution as the data pixels.

        Returns
        -------
        image_roi : PIL_Image
            The image ROI corresponding to the measurement area.
        """
        assert check_attr(self, '_extent_spots'), 'call set_extent_data first'
        # get _extent of data points in txt-file
        x_min_area, x_max_area, y_min_area, y_max_area = self._extent
        x_min_meas, x_max_meas, y_min_meas, y_max_meas = self._extent_spots

        # resize region in photo to match data points
        if match_pxls:
            img_resized = self._image_roi
        else:
            img_resized = self.image.crop(
                (x_min_area, y_min_area, x_max_area, y_max_area)
            )

        self._photo_roi_xywh = (  # photo units
            x_min_area,
            y_min_area,
            abs(x_max_area - x_min_area),
            abs(y_max_area - y_min_area)
        )
        self._data_roi_xywh = (  # data units
            0,  # XRF coordinates always start at 0
            0,  # XRF coordinates always start at 0
            abs(x_max_meas - x_min_meas),
            abs(y_max_meas - y_min_meas)
        )
        self._image_roi = img_resized

    @property
    def photo_roi_xywh(self) -> tuple[int, ...]:
        if not check_attr(self, '_photo_roi_xywh'):
            self.set_photo_roi()
        return self._photo_roi_xywh

    @property
    def data_roi_xywh(self) -> tuple[int, ...]:
        if not check_attr(self, '_data_roi_xywh'):
            self.set_photo_roi()
        return self._data_roi_xywh

    def plot_overview(
            self, fig: plt.Figure | None = None, ax: plt.Axes | None = None, hold=False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        img = PIL_to_np(self.image)

        x, y, w, h = self._photo_roi_xywh
        rect_photo = patches.Rectangle((x, y), w, h, fill=False, edgecolor='g')

        if fig is None:
            assert ax is None, "If ax is provided, must also provide fig"
            fig, ax = plt.subplots()
        else:
            assert ax is not None, "If fig is provided, must also provide ax"
        plt.imshow(img)
        ax.add_patch(rect_photo)
        if hold:
            return fig, ax
        plt.show()


class ProjectBaseClass:
    """
    Abstract base class for ProjectMSI and ProjectXRF.
    """
    # placeholders for objects
    _age_model: AgeModel | None = None
    depth_span: tuple[float, float] | None = None
    age_span: tuple[float, float] | None = None

    holes_data = None
    holes_xray = None

    _image_handler: SampleImageHandlerMSI | SampleImageHandlerXRF | None = None
    _image_sample: ImageSample = None
    _image_roi: ImageROI = None
    _image_classified: ImageClassified = None

    path_folder: str | None = None
    path_d_folder: str | None = None

    _da_export: DataAnalysisExport | None = None
    _spectra: Spectra = None
    _data_object: MSI | XRF = None
    _xray_long: XRay | None = None
    _xray: XRayROI | None = None
    _time_series: TimeSeries = None
    # flags
    _is_laminated = None
    _is_MSI = None

    def __repr__(self) -> str:
        return object_to_string(self)

    def _update_files(self):
        """Placeholder for children"""
        raise NotImplementedError()

    def forget(self, attribute: str):
        """
        Forget a certain attribute (set it to None).

        This also works for attributes by first mapping the public property
        to the private attribute.
        """


        private_attributes: list[str] = [
            k[1:] for k in self.__dir__()
            if k.startswith('_') and not k.startswith('__')
        ]

        forget_attr: str = (attribute
                            if attribute not in private_attributes
                            else f'_{attribute}')
        logger.info(f'Forgetting attribute: {forget_attr}')

        assert forget_attr in self.__dir__(), \
            f'Instance does not have attribute "{attribute}"'

        setattr(self, forget_attr, None)

    def set_age_model(self, path_file: str | None = None, **kwargs_read) -> None:
        self._age_model: AgeModel = AgeModel(
            path_file=path_file, **kwargs_read
        )
        self._age_model.path_file = self.path_d_folder
        self._age_model.save()

        self._update_files()

    def require_age_model(
            self,
            path_file: str | None = None,
            overwrite: bool = False,
            **kwargs_read: dict
    ) -> AgeModel:
        """
        Load an age model from the d folder if it exists, from another file if
        specified in path_file or create a new object by reading data from 
        path_file with the passed keyword arguments.

        Parameters
        ----------
        path_file : str, optional
            File from where to read the AgeModel object (if load is True) or
            the file from which to create an AgeModel object. 
            The default is None.
        load : bool, optional
            Set to true if you want to load an AgeModel object. 
            The default is True.
        **kwargs_read : dict
            Keyword arguments to be passed to AgeModel if the object is not 
            loaded.

        Returns
        -------
        None.

        """
        # return existing instance
        if (self._age_model is not None) and (not overwrite):
            return self._age_model
        # if an age model save is located in the folder, load it
        elif check_attr(self, 'AgeModel_file') and (not overwrite):
            self._age_model: AgeModel = AgeModel(
                path_file=os.path.join(
                    self.path_d_folder,
                    self.__getattribute__('AgeModel_file')
                )
            )
        # if a file is provided, load it from the file provided
        elif (not overwrite) and (path_file is not None):
            self._age_model: AgeModel = AgeModel(path_file=path_file, **kwargs_read)
        else:  # otherwise create new age model
            self.set_age_model(path_file, **kwargs_read)
        return self._age_model

    @property
    def age_model(self) -> AgeModel:
        return self.require_age_model()

    @age_model.setter
    def age_model(self, age_model: AgeModel):
        assert isinstance(age_model, AgeModel), 'Provided value should be an instance of AgeModel.'
        self._age_model = age_model

    def set_depth_span(self, depth_span: tuple[float, float]) -> None:
        """
        Set the depth span as a tuple where the first element is the upper
        depth and the second element the lower depth in cm.

        Example usage
        -------------
        >>> p.set_depth_span((490, 495))
        sets the depth span to 490-495 cm for a project


        Parameters
        ----------
        depth_span: tuple[float]


        Returns
        -------
        None
        """
        self.depth_span: tuple[float, float] = depth_span

    def set_age_span(
            self,
            depth_span: tuple[float, float] | None = None,
            age_span: tuple[float, float] | None = None
    ) -> None:
        """
        Set the age span of the measurement either from the age model or the
        provided tuple.

        Age is expected to be in years before the year 2000 (b2k).

        Example usage
        -------------
        >>>p.set_age_span(age_span=(11400, 11420))
        sets the age span from 11400 to 11420 yrs b2k.
        >>>p.set_age_span(depth_span=(490, 495))
        sets the age span using the age model.
        >>>p.set_age_span()
        sets the age span using the age model and the previously specified
        depth span.

        Returns
        -------
        None
        """
        assert (
            (depth_span is not None) or
            (self.depth_span is not None) or
            (self.age_span is not None)
        ), 'specify the depth in cm or the age span in yrs b2k'
        assert (age_span is not None) or (self.age_model is not None)
        if depth_span is None:
            depth_span: tuple[float, float] = self.depth_span

        if age_span is not None:
            self.age_span: tuple[float, float] = age_span
        else:
            self.age_span: tuple[float, float] = tuple(
                self.age_model.depth_to_age(depth_span)
            )

    def set_image_handler(self, *args, **kwargs):
        """Placeholder for children"""
        raise NotImplementedError()

    def require_image_handler(self, *args, **kwargs):
        """Placeholder for children"""
        raise NotImplementedError()

    @property
    def image_handler(self):
        return self.require_image_handler()

    def set_image_sample(
            self,
            obj_color: str | None = None,
            use_extent_from_handler: bool = True,
            use_extent_from_mis: bool = None,
            **kwargs: Any
    ) -> None:
        """
        Set image_sample from image_handler.

        This method is available after the image_handler has been set.

        Parameters
        ----------
        obj_color: str, optional
            The color of the sample relative to the background (either 'dark'
            or 'light'). If not specified, this parameter will be determined
            automatically or loaded from the saved object, if available.
        use_extent_from_mis: bool, optional
            If True, will determine the image extent from the measurement area
            defined in the mis file and the optimizer otherwise.
        kwargs: Any, optional
            keyword arguments passed on to ImageSample.sget_sample_area and save

        Returns
        -------
        None

        """
        # pass image file from handler if it has it else the image
        # that way we can save disk space as ImageSample only saves the Image
        # if it does not know the image file
        if use_extent_from_mis is True:
            logger.warning('"use_extent_from_mis" option has been renamed to '
                           '"use_extent_from_handler" and will be removed in '
                           'the future')
            use_extent_from_handler = use_extent_from_mis

        # fetch image from path in handler
        if check_attr(self.image_handler, 'image_file'):
            image_file: str = self.image_handler.image_file
            path_image_file: str = os.path.join(self.path_folder, image_file)
            image_kwargs: dict = dict(path_image_file=path_image_file)
        else:
            image_kwargs: dict = dict(image=self.image_handler.image,
                                      image_type='pil')
            logger.warning('Handler does not have an image file, this should '
                           'not happen and will increase the disk space needed')

        # if obj color is not provided, attempt to estimate it from region in
        # image handler
        if check_attr(self, '_image_handler'):
            logger.info('attempting to estimate obj_color using measurement area in ')

        self._image_sample: ImageSample = ImageSample(path_folder=self.path_folder,
                                                      obj_color=obj_color,
                                                      **image_kwargs)
        # attempt to set photo ROI on image handler
        if (
                use_extent_from_handler and
                (not check_attr(self._image_sample, '_xywh_ROI'))
        ):
            try:
                self.image_handler.set_photo_roi()
                assert check_attr(self.image_handler, '_photo_roi_xywh'), \
                    'Need an image handler with photo ROI'
            except Exception as e:
                logger.error(e)
                logger.error(
                    'Could not set photo ROI, continuing with fitting box'
                )
                use_extent_from_handler = False

        if use_extent_from_handler:
            x_start: int = self.image_handler.photo_roi_xywh[0]
            x_end: int = x_start + self.image_handler.photo_roi_xywh[2]
            extent_x: tuple[int, int] = (x_start, x_end)
        else:
            extent_x: None = None

        thr_method = kwargs.pop('thr_method',
                                'otsu' if self._is_laminated else 'slic')
        logging.info(f'estimating foreground pixels with method {thr_method}')
        self._image_sample.set_foreground_thr_and_pixels(
            measurement_area_xywh=self.image_handler.photo_roi_xywh,
            thr_method=thr_method,
            **kwargs
        )
        self._image_sample.set_sample_area(extent_x=extent_x, **kwargs)
        self._image_sample.save(kwargs.get('tag'))

        self._update_files()

    def require_image_sample(
            self,
            obj_color: str | None = None,
            overwrite: bool = False,
            tag: str | None = None,
            **kwargs: Any
    ) -> ImageSample:
        # return existing
        if check_attr(self, '_image_sample') and (not overwrite):
            return self._image_sample
        # load and return
        if (
                check_attr(self, 'ImageSample_file')
                or (tag is not None)
        ) and (not overwrite):
            logger.info('loading ImageSample')
            self._image_sample: ImageSample = ImageSample(
                path_folder=self.path_folder,
                image=self.image_handler.image,
                image_type='pil',
                obj_color=obj_color
            )

            if not os.path.exists(self._image_sample.save_file):
                logger.warning(f'Could not find ImageSample with {tag=}')

            self._image_sample.load(tag)
            # overwrite obj_color
            if obj_color is not None:
                self._image_sample.obj_color = obj_color
            if check_attr(self._image_sample, '_xywh_ROI'):
                return self._image_sample

            logger.warning(
                'loaded partially initialized ImageSample, overwriting '
                'loaded ImageSample with fully initialized object'
            )

        logger.info("Initializing new ImageSample instance")
        # either overwrite or loaded partially processed obj
        self.set_image_sample(
            obj_color=(
                self._image_sample.obj_color
                if check_attr(self, '_image_sample') and not overwrite
                else obj_color),  # try to use stored obj_color
            tag=tag,
            **kwargs
        )

        return self._image_sample

    @property
    def image_sample(self):
        return self.require_image_sample()

    def set_image_roi_from_ion_image(self, comp: str | int | float, **kwargs) -> None:
        """
        Initialize an ImageROI instance from an ion image in the feature table.

        This may be especially useful for XRF data. The ion image will be taken
        from the data_object and values scaled to be compatible with uint8.
        This is achieved by rescaling values between 0 and the 95th percentile
        to 0 and 255.

        Parameters
        ----------
        comp: str | int | float
            Compound to be used for the input image
        kwargs: Any
            Additional parameters.

        Returns
        -------
        None
        """
        assert check_attr(self, '_data_object'), \
            'initialize data object first'
        assert comp in self._data_object.feature_table.columns, \
            f'{comp=} not found in data_object'

        image, *_ = get_comp_as_img(data_frame=self.data_object.feature_table,
                                    comp=comp,
                                    exclude_holes=False)
        # some functions require the image to be of type uint8
        image: np.ndarray[np.uint8] = np.around(rescale_values(
            image,
            0,
            255,
            0,
            np.nanquantile(image, .95)
        )).astype(np.uint8)

        image_roi: ImageROI = ImageROI(
            image=image,
            obj_color='light',
            age_span=self.age_span
        )

        # if image roi is set from an ion image, image_sample and image_handler
        # have to be modified by setting the sample area to the measurement area
        if (
                check_attr(self, '_image_sample')
                and check_attr(self, '_image_sample')
        ):
            self.image_sample._xywh_ROI = self.image_handler.photo_roi_xywh
            x, y, w, h = self.image_sample.xywh_ROI
            # now we can be sure data roi is same as sample roi
            self.image_handler._data_roi_xywh = (0, 0, w, h)
            self.image_sample._image_roi = self.image_sample.image[
                                           y: y + h, x: x + w
                                           ].copy()

        else:
            logger.warning(
                'Was not able to set sample area to measurement area because '
                'either sample or handler is not set'
            )

        self._image_roi: ImageROI = image_roi
        if self.age_span is not None:
            self._image_roi.age_span = self.age_span

        self._image_roi.require_classification(**kwargs)
        self._image_roi.set_punchholes(**kwargs)

        self._image_roi.save(kwargs.get('tag'))

        self._update_files()

    def set_image_roi_from_parent(self, **kwargs) -> None:
        """
        Set an ImageROI instance.

        This function can only be called after set_image_sample has been called.
        """
        # create _image_roi using image from image_sample
        self._image_roi: ImageROI = ImageROI.from_parent(self.image_sample, **kwargs)
        if self.age_span is not None:
            self._image_roi.age_span = self.age_span

        self._image_roi.require_classification(**kwargs)
        self._image_roi.set_punchholes(**kwargs)

        self._image_roi.save(kwargs.get('tag'))

        self._update_files()

    def set_image_roi(self, **kwargs):
        warnings.warn('the set_image_roi function will be replaced by '
                      'set_image_roi_from_parent in the future.')
        self.set_image_roi_from_parent(**kwargs)

    def require_image_roi(
            self,
            overwrite: bool = False,
            tag: str | None = None,
            source: str = 'parent',
            **kwargs
    ) -> ImageROI:
        def add_age_span() -> ImageROI:
            if check_attr(self, 'age_span'):
                self._image_roi.age_span = self.age_span
            return self._image_roi

        # return existing
        if (self._image_roi is not None) and (not overwrite):
            return add_age_span()
        # load from disk
        if check_attr(self, 'ImageROI_file') and (not overwrite):
            logger.info('loading ImageROI')
            self._image_roi: ImageROI = ImageROI.from_disk(
                path_folder=os.path.join(self.path_folder),
                tag=tag
            )
            return add_age_span()

        assert source in (source_options := ['parent', 'comp']), \
            f'source must be one of {source_options}, not {source}'
        # from parent
        if source == 'parent':
            logger.info('Creating new ImageROI instance from parent')
            assert check_attr(self, '_image_sample'), \
                'set image_sample first or specify a different source'
            self.set_image_roi_from_parent(**kwargs)
            return self.image_roi
        # from ion image
        if source == 'comp':
            logger.info('Creating new ImageROI instance from ion image')
            assert check_attr(self, '_data_object'), \
                'set data_object first or specify a different source'
            self.set_image_roi_from_ion_image(**kwargs)
            return self.image_roi

    @property
    def image_roi(self) -> ImageROI:
        return self.require_image_roi()

    def set_tilt_corrector(self, **kwargs) -> None:
        assert check_attr(self, '_image_roi'), \
            'need image_roi to initialize tilt corrector'
        if not check_attr(self, '_image_classified'):
            self.require_image_classified(full=False)

        self.image_classified.set_corrected_image(**kwargs)
        self._update_files()

    def require_tilt_corrector(self, overwrite=False, **kwargs) -> Mapper:
        assert check_attr(self, '_image_roi'), \
            'need image_roi to initialize tilt corrector'
        mapper = Mapper(self.image_roi.image.shape,
                        self.path_folder,
                        'tilt_correction')
        if overwrite or (not os.path.exists(mapper.save_file)):
            logger.info(f'Could not find mapper with file {mapper.save_file}, '
                        f'creating new instance')
            self.set_tilt_corrector(**kwargs)

        # mapper should exist now in either case
        mapper.load()
        return mapper

    def set_image_classified(
            self,
            peak_prominence: float = .01,
            downscale_factor: float = 1 / 16,
            reduce_laminae: bool = True,
            use_tilt_correction: bool | None = None,
            full: bool = False,
            **kwargs
    ) -> None:
        """
        Set an ImageClassified instance, assuming an ImageROI instance has been
        saved before.

        Parameters
        ----------
        peak_prominence: float (default =0.1)
            The relative peak prominence above which peaks in the brightness
            function are considered as relevant.
        downscale_factor: float (default = 1 / 16)
            Downscaling factor to be applied to the image before tuning the
            parameters. The closer this value is to 0, the worse the
            resolution, but the faster the convergence.
        reduce_laminae: bool, optional
            Reduce the number of layers to that predicted by the age model.
            The default is True.
        full: bool, optional
            Whether to perform all processing steps, including determining
            laminae parameters. The default is False

        Returns
        -------
        None
        """
        if use_tilt_correction is None:
            use_tilt_correction = self._is_laminated

        # initialize ImageClassified
        self._image_classified: ImageClassified = ImageClassified.from_parent(
            self.image_roi, use_tilt_correction=use_tilt_correction, **kwargs
        )
        if (
                not check_attr(self._image_classified, 'age_span')
                and check_attr(self, 'age_span')
        ):
            self._image_classified.age_span = self.age_span

        if full:
            logger.info('setting laminae in ImageClassified ...')
            self._image_classified.set_laminae_params_table(
                peak_prominence=peak_prominence,
                downscale_factor=downscale_factor,
                **kwargs
            )
            if reduce_laminae:
                self._image_classified.reduce_laminae(**kwargs)
        self._image_classified.save(kwargs.get('tag'))
        self._update_files()

    def require_image_classified(
            self,
            overwrite: bool = False,
            tag: str = None,
            **kwargs
    ) -> ImageClassified:
        if (self._image_classified is not None) and (not overwrite):
            return self._image_classified

        if (
                check_attr(self, 'ImageClassified_file')
                or (tag is not None)
        ) and (not overwrite):
            logger.info('loading ImageClassified ...')
            self._image_classified: ImageClassified = ImageClassified.from_parent(
                self.image_roi, **kwargs
            )
            try:
                self._image_classified.load(tag)

                if check_attr(self, 'age_span'):
                    self._image_classified.age_span = self.age_span
                if (not check_attr(self._image_classified,
                                   'params_laminae_simplified')):
                    logger.warning(
                        'Loaded ImageClassified is expected to have a '
                        'params table but found none'
                    )
                else:
                    return self._image_classified
            except FileNotFoundError:
                logger.warning(f'Could not find ImageClassified with {tag=}')

        self.set_image_classified(**kwargs)

        return self._image_classified

    @property
    def image_classified(self) -> ImageClassified:
        return self.require_image_classified()

    def require_images(self, overwrite: bool = False, **kwargs) -> None:
        """Set the image handler, sample and roi, and if the sediment is
        laminated classified.

        Parameters
        ----------
        overwrite: bool
            If False, will attempt to load objects from disk. If True, will
            overwrite saved states.
        kwargs: Any
            keywords for the setters.
        """
        self.require_image_handler(overwrite=overwrite, **kwargs)
        self.require_image_sample(overwrite=overwrite, **kwargs)
        self.require_image_roi(overwrite=overwrite, **kwargs)
        if self._is_laminated:
            self.require_image_classified(overwrite=overwrite, **kwargs)

    def add_image_attributes(self, **kwargs):
        def add_or_warn(obj_name: str, func, **kwargs_):
            if check_attr(self, obj_name):
                func(**kwargs_)
            else:
                logger.warning(
                    'Unable to call {func.__name__} because {obj_name} is not '
                    'initialized. Call require_{obj_name} first.'
                )

        self.add_tic()

        add_or_warn('_image_handler', self.add_pixels_ROI)
        add_or_warn('_image_sample', self.add_photo, **kwargs)
        add_or_warn('_image_roi', self.add_holes, **kwargs)
        add_or_warn(
            '_image_roi', self.add_light_dark_classification, **kwargs
        )
        if self._is_laminated:
            add_or_warn(
                '_image_classified',
                self.add_laminae_classification,
                **kwargs
            )
        add_or_warn('_xray', self.add_xray)

    def require_data_object(self, *args, **kwargs):
        """Overwritten by children"""
        raise NotImplementedError()

    @property
    def data_object(self):
        return self.require_data_object()

    def add_tic(self, imaging_info_xml: ImagingInfoXML | None = None):
        """Add the total ion count (TIC) for each pixel to the feature table
        of the data obj."""
        assert self._data_object is not None, "Set data object first with require_data_object"

        if imaging_info_xml is None:
            imaging_info_xml: ImagingInfoXML = ImagingInfoXML(
                path_d_folder=self.path_d_folder
            )
        ft_imaging: pd.DataFrame = imaging_info_xml.feature_table.loc[
            :, ['R', 'x', 'y', 'tic', 'minutes']
        ]
        self.data_object.inject_feature_table_from(
            pd.merge(self.data_object.feature_table,
                     ft_imaging,
                     how="left"),
            supress_warnings=True
        )

    def add_pixels_ROI(self) -> None:
        """
        Add image pixels to data points in the feature table of the data_object.

        This requires that the image handler and sample have been set.
        Creates new columns x_ROI and y_ROI for the pixel coordinates in the
        feature table.
        """
        assert self._image_sample is not None, 'call set_image_sample first'
        assert self._data_object is not None, 'call set_data_object'

        attrs: tuple[str, ...] = ('_image_roi', '_photo_roi_xywh', '_data_roi_xywh')
        if not all([check_attr(self.image_handler, attr) for attr in attrs]):
            self.image_handler.set_photo_roi()
        image_ROI_xywh: tuple[int, ...] = self.image_sample.xywh_ROI
        data_ROI_xywh: tuple[int, ...] = self.image_handler._data_roi_xywh
        photo_ROI_xywh: tuple[int, ...] = self.image_handler._photo_roi_xywh

        self.data_object.pixels_get_photo_ROI_to_ROI(
            data_ROI_xywh, photo_ROI_xywh, image_ROI_xywh
        )

    def add_photo(self, median: bool = False, **_) -> None:
        """
        Add the gray-level values of the photo to the feature table of the data_object.

        This function has to be called after add_pixels_ROI.
        In general the resolution of the data points is smaller than that of the photo.
        By default, the closest value is used.
        The column in the feature table is 'L'

        median: bool (default False)
            If median is True, the median intensities inside the photo will be
            ascribed to each data point.
            If median is False, the closest values will be used.

        Returns
        -------
        None
        """
        assert self._data_object is not None, 'set data object first'
        assert 'x_ROI' in self.data_object.feature_table.columns, \
            'add x_ROI, y_ROI coords with add_pixels_ROI'
        image = ensure_image_is_gray(
            self.image_sample.image_sample_area
        )
        self.data_object.add_attribute_from_image(image, 'L', median=median)

    def add_holes(self, **kwargs) -> None:
        """
        Add classification for holes and sample to the feature table of the
        data_object.

        This function has to be called after the ImageROI and data object have
        been set. Uses the closest pixel value.
        The new column is called 'valid' where holes are associated with a
        value of 0.

        Returns
        -------
        None
        """
        kwargs_ = kwargs.copy()
        kwargs_['median'] = False
        assert self._image_roi is not None, 'set image_roi first'
        assert self._data_object is not None, 'set data_object first'
        assert 'x_ROI' in self._data_object.columns, 'call add_pixels_ROI first'

        image = self.image_roi.image_binary
        self.data_object.add_attribute_from_image(image, 'valid', **kwargs_)
        # round and int
        self.data_object.feature_table.valid = np.round(
            self.data_object.feature_table.valid
        ).astype(int)

    def add_light_dark_classification(self, **kwargs) -> None:
        """
        Add light and dark classification to the feature table of the data_object.
        This only considers the foreground (valid) pixels.

        The new column inside the feature table of the data_object is called 'classification'.

        Returns
        -------
        None
        """
        assert self._image_roi is not None, 'call set_image_roi_from_parent'
        assert self._data_object is not None, 'set data_object first'

        image: np.ndarray[int] = self.image_roi.image_classification
        self.data_object.add_attribute_from_image(image, 'classification', **kwargs)
        # round and int
        self.data_object.feature_table.valid = np.round(
            self.data_object.feature_table.valid
        ).astype(int)

    def data_object_apply_tilt_correction(self) -> None:
        assert not self.corrected_tilt, 'tilt has already been corrected'
        assert self._data_object is not None, 'set data_object.'
        assert 'x_ROI' in self._data_object.columns, 'call add_pixels_ROI first'

        mapper = Mapper(
            image_shape=(-1, -1),
            path_folder=self.path_folder,
            tag='tilt_correction'
        )

        if self.image_classified is None:
            logger.warning('no image_classified set, not correcting tilts')
            return
        elif not self.image_classified.use_tilt_correction:
            logger.warning(
                'image classified set, but tilt correction set to False, '
                'not correcting tilts'
            )
            return

        if not os.path.exists(mapper.save_file):
            raise FileNotFoundError(
                f'expected to find a Mapping at '
                f'{mapper.save_file}, call require_tilt_corrector'
            )

        mapper.load()
        # get transformed coordinates
        XT, YT = mapper.get_transformed_coords()

        # insert into feature table
        self.data_object.add_attribute_from_image(
            XT, 'x_ROI_T', fill_value=np.nan
        )
        self.data_object.add_attribute_from_image(
            YT, 'y_ROI_T', fill_value=np.nan
        )

        # fit feature table
        self.data_object.inject_feature_table_from(
            transform_feature_table(self.data_object.feature_table),
            supress_warnings=True
        )
        logger.info('successfully loaded mapper and applied tilt correction')

        self._data_object.tilt_correction_applied = True

    def data_object_apply_transformation_old(self, mapper: Mapper) -> None:
        """Apply a mapping from a mapper object to the data."""
        assert not self.corrected_tilt, 'tilt has already been corrected'
        assert self._data_object is not None, 'set data_object.'
        assert 'x_ROI' in self._data_object.columns, 'call add_pixels_ROI first'

        # get transformed coordinates
        XT, YT = mapper.get_transformed_coords()

        # insert into feature table
        self.data_object.add_attribute_from_image(
            XT, 'x_ROI_T', fill_value=np.nan
        )
        self.data_object.add_attribute_from_image(
            YT, 'y_ROI_T', fill_value=np.nan
        )

        # fit feature table
        self.data_object.inject_feature_table_from(
            transform_feature_table(self.data_object.feature_table),
            supress_warnings=True
        )
        logger.info('successfully applied mapping')

    def data_object_apply_transformation(
            self,
            mapper: Mapper,
            keep_sparse: bool | None = None,
            sparsity_threshold: float = .5,
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Apply a mapping from a mapper object to the data.

        Parameters
        ----------
        mapper: Mapper
            Mapper object that has information of the x and y shifts. Must have
            the image from the sample area as input (or one with the same shape).
            The Transformation class rescales and stretches the source image to
            match it to the target's image shape by default.
        keep_sparse: bool, optional
            Whether to keep the intensity distribution sparse. This should be
            preferred for noisy data. None defaults to estimating this for each
            compound separately based on the sparsity threshold.
        sparsity_threshold: float, optional
            Ratio of non-zero values required for a compound to be not sparse.
        """
        def reformat_ion_image_to_mapper_input(values_) -> np.ndarray[float]:
            """
            Take the raveled intensities of an ion image and format them to
            match the shape of the target image.

            """
            # ion image has to match extent from the image inputted into the
            # mapper
            # first step: scale to photo resolution
            ion_image: np.ndarray = griddata(
                points,
                values_,
                (grid_x, grid_y),
                method='nearest',
                fill_value=0
            )

            # second step: zeropad/crop to match
            target_image[mask_ion_image] = ion_image

            return target_image

        assert self._data_object is not None, 'set data_object.'
        assert 'x_ROI' in self._data_object.columns, 'call add_pixels_ROI first'
        assert check_attr(self, '_image_handler'), \
            'need image handler to determine image region'

        keep_sparse_auto: bool = keep_sparse is None
        assert (sparsity_threshold >= 0) and (sparsity_threshold <= 1), \
            'sparsity threshold must be between 0 and 1'

        # container in which to place ion images for fitting
        # can use the same container for each ion image since the extent
        # of ion images remains the same
        source_sample_roi: ROI = ROI(self.image_sample.xywh_ROI)
        source_shape = (source_sample_roi.h, source_sample_roi.w)
        target_shape = mapper.image_shape

        logger.debug('source_shape', source_shape)
        logger.debug('target_shape', target_shape)

        # must be the same as with which the mapper was created
        # in Transformer, source image is rescaled to match it to target image
        # determine rescaling factors to match shape of target and source
        # source * factor = target <==> factor = target / source
        rescale_x: float = target_shape[1] / source_shape[1]
        rescale_y: float = target_shape[0] / source_shape[0]

        # scale x_ROI, y_ROI in feature table in order for
        # add_attribute_from_image to work correctly. Columns will be scaled
        # back in the end
        # avoid pandas complaining about dtypes
        ft = self.data_object.feature_table
        ft.loc[:, 'x_ROI'] = ft.loc[:, 'x_ROI'].astype(float) * rescale_x
        ft.loc[:, 'y_ROI'] = ft.loc[:, 'y_ROI'].astype(float) * rescale_y

        # rescale points
        x_roi_ft: pd.Series = self.data_object.feature_table.x_ROI
        y_roi_ft: pd.Series = self.data_object.feature_table.y_ROI
        points: np.ndarray[float] = np.c_[x_roi_ft, y_roi_ft]

        # roi area in terms of target image coordinates
        # (this is not the same as the measurement roi of the target because
        # measurement areas may cover different areas in source and target)
        source_meas_roi: ROI = ROI(self.image_handler.photo_roi_xywh)
        source_meas_roi_rescaled: ROI = source_meas_roi.resize(
            source_shape, target_shape
        )

        # sample roi of source in target pixel coordinates
        source_sample_roi_rescaled: ROI = source_sample_roi.resize(
            source_shape, target_shape
        )

        logger.debug(f'{source_sample_roi=}')
        logger.debug(f'{source_meas_roi=}')

        logger.debug('after rescaling to match target')
        logger.debug(f'{source_sample_roi_rescaled=}')
        logger.debug(f'{source_meas_roi_rescaled=}')

        # container for ion image intensities
        target_image: np.ndarray[float] = np.zeros(target_shape, dtype=float)

        # measurement relative to sample
        # find indices of overlap between data and sample roi's in target coordinates
        y_ion_roi: int = source_meas_roi_rescaled.y - source_sample_roi_rescaled.y
        if y_ion_roi < 0:  # measurement area starts above of sample area
            y_ion_roi = 0

        x_ion_roi: int = source_meas_roi_rescaled.x - source_sample_roi_rescaled.x
        if x_ion_roi < 0:  # measurement area starts above of sample area
            x_ion_roi = 0

        h_ion_roi: int = source_meas_roi_rescaled.h
        if (y_ion_roi + h_ion_roi) > source_sample_roi_rescaled.h:
            h_ion_roi: int = source_sample_roi_rescaled.h - y_ion_roi

        w_ion_roi: int = source_meas_roi_rescaled.w
        if (x_ion_roi + w_ion_roi) > source_sample_roi_rescaled.w:
            w_ion_roi: int = source_sample_roi_rescaled.w - x_ion_roi

        roi_ion: ROI = ROI(x_ion_roi, y_ion_roi, w_ion_roi, h_ion_roi)
        mask_ion_image = roi_ion.index_exp

        # regular grid spanning the overlap area of sample and measurement region
        grid_x, grid_y = np.meshgrid(
            np.arange(roi_ion.x, roi_ion.x + roi_ion.w),
            np.arange(roi_ion.y, roi_ion.y + roi_ion.h)
        )

        if plts:
            plt.figure()
            source_photo_shape = self.image_sample.image.shape[:2]
            area = np.zeros(source_photo_shape)
            area[source_sample_roi.get_mask_for_image(source_photo_shape)] = 1
            area[source_meas_roi.get_mask_for_image(source_photo_shape)] = 2
            plt.imshow(area)
            plt.title('Area of ion images overlapping sample area in photo')
            plt.show()

            plt.figure()
            area = np.zeros(target_shape)
            area[source_sample_roi_rescaled.reset_offset().index_exp] = 1
            area[source_meas_roi_rescaled.reset_offset().index_exp] = 2
            area[mask_ion_image] = 3
            plt.imshow(area)
            plt.title('Area of ion images overlapping sample area in roi after rescaling')
            plt.show()

        is_sparse: bool = keep_sparse  # will only be updated if keep_sparse_auto
        for comp in tqdm(
                self.data_object.data_columns,
                desc='transforming ion images'
        ):
            values: np.ndarray[float] = self.data_object.feature_table.loc[
                :, comp
            ].fillna(0).to_numpy()

            ion_image = reformat_ion_image_to_mapper_input(values)

            if keep_sparse_auto:
                is_sparse: bool = (values > 0).mean() < sparsity_threshold
                if is_sparse:
                    logger.info(f'found sparse compound: {comp}')

            warped_image: np.ndarray[float] = mapper.fit(
                ion_image,
                preserve_range=True,
                keep_sparse=is_sparse,
                **kwargs
            )

            # add to feature table
            self.data_object.add_attribute_from_image(
                image=warped_image, column_name=comp
            )
        # scale ROI coordinates back
        self.data_object.feature_table.loc[:, 'x_ROI'] = np.around(
            self.data_object.feature_table.loc[:, 'x_ROI'] / rescale_x
        ).astype(int)
        self.data_object.feature_table.loc[:, 'y_ROI'] = np.around(
            self.data_object.feature_table.loc[:, 'y_ROI'] / rescale_y
        ).astype(int)

    @property
    def corrected_tilt(self) -> bool:
        return self._data_object.tilt_correction_applied

    def add_laminae_classification(self, **kwargs) -> None:
        """
        Add light and dark laminae classification to the feature table of the
        data_object. This only considers the foreground (valid) pixels.

        This requires that the ImageClassified instance has already been set.

        The new column inside the feature table of the data_object is called
        'classification_s'. A column called 'classification_se' is also added,
        which is the classification image but laminae have been expanded to
        fill the entire image, except holes.

        Returns
        -------
        None
        """
        assert self._image_classified is not None, 'call set_image_classified'
        assert self._data_object is not None, 'set data_object first'

        assert (  # image classified either not tilt corrected or data object transformed
            (not self._image_classified.use_tilt_correction)
            or self._data_object.tilt_correction_applied
        ), (
            'found image_classified but use_tilt_correction is set to True and '
            'data object was not tilt-corrected. Please first '
            "transform the data_object's feature table"
        )

        image: np.ndarray[int] = self.image_classified.image_seeds
        image_e: np.ndarray[int] = self.image_classified.get_image_expanded_laminae()
        self.data_object.add_attribute_from_image(image, 'classification_s', **kwargs)
        self.data_object.add_attribute_from_image(image_e, 'classification_se', **kwargs)

    def add_depth_column(self, exclude_gaps: bool = True) -> None:
        """
        Add the depth column to the data_object using the depth span.

        Linearly map xs (from the data_object.feature_table) to depths
        (based on depth_span).

        If exclude_gaps is True, depth intervals where no sediment is present,
        will be excluded from the depth calculation (they will be assigned the
        same depth as the last valid depth).
        """
        assert self._data_object is not None, 'set the data_object first'
        assert self.depth_span is not None, 'set the depth_span first'
        if exclude_gaps:
            assert 'valid' in self.data_object.feature_table.columns,\
                'set holes in data_object first'

        min_depth, max_depth = self.depth_span
        # convert seed pixel coordinate to depth and depth to age
        x: pd.Series = self.data_object.feature_table.x_ROI

        self.data_object.feature_table['depth'] = np.nan

        if not exclude_gaps:
            depths = rescale_values(
                x,
                new_min=min_depth,
                new_max=max_depth,
                old_min=x.min(),
                old_max=x.max()
            )
            self.data_object.feature_table['depth'] = depths

        else:
            # row-wise check if any of the pixels are part of the sample
            valid_mask: np.ndarray[bool] = self.data_object.feature_table.pivot(
                index='x_ROI', columns='y_ROI', values='valid'
            ).any(axis=1).to_numpy()
            # set first valid entry to 0 as well because cumsum is used
            valid_mask[np.argwhere(valid_mask == True)[0]] = False
            ddepth: float = (max_depth - min_depth) / (valid_mask.sum())

            # invalid rows do not increase depth
            depths = np.cumsum(valid_mask.copy() * ddepth)
            depths += min_depth
            # assign a depth to each x value
            xs = np.unique(x)
            assert len(xs) == len(depths)
            mapper = dict(zip(xs, depths))

            def map_depth(row):
                return mapper[row.x_ROI]

            self.data_object.feature_table['depth'] = \
                self.data_object.feature_table.apply(map_depth, axis=1)
        # add new column to feature table

    def add_age_column(self, use_corrected: bool = False) -> None:
        """
        Add an age column to the data_object using the depth column and age model.

        Map the depths in the data_object.feature_table with the age model.

        Parameters
        ----------
        use_corrected : bool
            Whether to use the corrected depth column ('depth_corrected').
            data_object must have the corrected depth column, otherwise an error is raised.

        Returns
        -------
        None.
        """
        assert self._age_model is not None, 'set age model first'
        assert self._data_object is not None, f'did not set data_object yet'
        assert check_attr(self.data_object, 'feature_table'), \
            'must have data_object'
        if use_corrected:
            depth_col = 'depth_corrected'
        else:
            depth_col = 'depth'
        assert depth_col in self.data_object.feature_table.columns, \
            ('data object must have {depth_col} column, call add_depth_column '
             'and set the exclude_gaps accordingly')

        self.data_object.feature_table['age'] = self.age_model.depth_to_age(
            self.data_object.feature_table[depth_col]
        )

    def set_xray(
            self,
            path_image_file: str,
            overwrite_long: bool = False,
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Set the X-ray object from the specified image file and depth section.

        The object is accessible with p.xray with p being the project.
        This method performs all processing steps.


        """
        assert path_image_file is not None, \
            'Providing an image file is required for setting the xray instance'
        assert os.path.exists(path_image_file), f'could not find {path_image_file=}'
        assert (self.depth_span is not None) or ('depth_section' in kwargs), \
            'set depth span or pass the depth_section argument'

        self._xray_long = XRay(path_image_file=path_image_file, **kwargs)
        if (
                (not overwrite_long) and
                os.path.exists(self._xray_long.save_file)
        ):
            self._xray_long.load(kwargs.get('tag'))
        else:
            self._xray_long.require_image_rotated()
            self._xray_long.require_image_sample_area(plts=plts, **kwargs)
            self._xray_long.remove_bars(plts=plts, **kwargs)
            self._xray_long.save(kwargs.get('tag'))

        self._xray: XRayROI = self._xray_long.get_roi_from_section(
            self.depth_span, plts=plts
        )
        self._xray.save(kwargs.get('tag'))

        self._update_files()

    def require_xray(
            self,
            overwrite: bool = False,
            tag: str | None = None,
            **kwargs
    ) -> ImageROI:
        if check_attr(self, '_xray') and (not overwrite):
            return self._xray

        try_load_long = True

        # try to load from disk
        if kwargs.get('path_folder') is not None:
            path_folder: str = kwargs['path_folder']
        elif kwargs.get('path_image_file') is not None:
            path_folder: str = os.path.dirname(kwargs['path_image_file'])
        else:  # neither folder nor image file provided, cannot determine disk location
            try_load_long: bool = False
        # first, try loading an XRayROI instance
        if (
                check_attr(self, 'XRayROI_file')
                or (tag is not None)
        ) and (not overwrite):
            self._xray = XRayROI.from_disk(path_folder=self.path_folder)
            return self._xray

        # try loading an XRay long instead
        if tag is not None:
            name: str = f'XRay_{tag}.pickle'
        else:
            name: str = 'XRay.pickle'

        if (
                try_load_long and
                os.path.exists(os.path.join(path_folder, name))
        ) and (not overwrite):
            self._xray_long = XRay(**kwargs)
            self._xray_long.load(kwargs.get('tag'))
            self._xray: XRayROI = self._xray_long.get_roi_from_section(
                self.depth_span
            )
            self._xray.save(kwargs.get('tag'))
            self._update_files()
            return self._xray

        self.set_xray(**kwargs)
        return self._xray

    @property
    def xray(self):
        return self.require_xray()

    def set_punchholes_from_msi_align(
            self,
            path_file_msi_align: str | None = None
    ):
        assert os.path.exists(path_file_msi_align), \
            f'Provided file {path_file_msi_align} does not exist'
        assert path_file_msi_align.split(".")[1] == 'json', \
            'Provided file must be a json'

        # teaching points of the photo
        x_data, y_data, labels_data = get_teaching_points(
            self.image_sample.path_image_file,
            path_file_msi_align
        )
        # subtract the x and y ROI coordinates from the returned teaching points
        # since teaching points are set on the entire images
        x_ROI, y_ROI, *_ = self.image_sample.xywh_ROI

        x_data: np.ndarray[float] = np.array(x_data) - x_ROI
        y_data: np.ndarray[float] = np.array(y_data) - y_ROI

        # TODO: check if this is (x, y) or (y, x)
        self.holes_data: list[np.ndarray[int], np.ndarray[int]] = [
            np.around(x_data).astype(int), np.around(y_data).astype(int)
        ]

        # teaching points of the photo
        x_xray, y_xray, labels_xray = get_teaching_points(
            self.image_sample.path_image_file,
            path_file_msi_align
        )
        # subtract the x and y ROI coordinates from the returned teaching points
        # since teaching points are set on the entire images
        x_ROI, y_ROI, *_ = self._xray_long.xywh_ROI

        x_xray: np.ndarray[float] = np.array(x_xray) - x_ROI
        y_xray: np.ndarray[float] = np.array(y_xray) - y_ROI

        # find the three points corresponding to the image by labels
        x_xray_new: list[int] = []
        y_xray_new: list[int] = []

        with open(path_file_msi_align, 'r') as f:
            d: dict[str, Any] = json.load(f)

        pairings = d['pair_tp_str']

        mapping: dict[int, int] = get_teaching_point_pairings_dict(pairings)

        inverse_mapping: dict[int, int] = {v: k for k, v in mapping.items()}

        for label in labels_data:
            m: dict[int, int] = mapping if label in mapping else inverse_mapping
            # should have exactly one match
            idx: np.ndarray[int] = np.argwhere(labels_xray == m[label]).squeeze()
            assert idx.size == 1
            x_xray_new.append(x_xray[idx])
            y_xray_new.append(y_xray[idx])
        # closest in terms of relative image coordinates
        # for xd, yd in zip(x_data, y_data):
        #     # TODO: scale coordinates
        #     # TODO: center X-ray around section corresponding to image
        #     idx_min = np.argmin((x_xray - xd) ** 2 + (y_xray - yd) ** 2)
        #     x_xray_new.append(x_xray[idx_min])
        #     y_xray_new.append(y_xray[idx_min])

        # TODO: check if this is x, y or y, x
        self.holes_xray: list[np.ndarray[int], np.ndarray[int]] = [
            np.around(x_xray).astype(int), np.around(y_xray).astype(int)
        ]

    def set_punchholes(
            self,
            side_xray: str | None = None,
            side_data: str | None = None,
            overwrite_data: bool = False,
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Identify square-shaped holes at top or bottom of sample in xray section
        and MSI sample.

        Parameters
        ----------
        side_xray : str, optional
            The side on which the holes are expected to be found in the xray
            sample, either 'top' or 'bottom'.
        side_data : str, optional
            The side on which the holes are expected to be found in the data
            sample, either 'top' or 'bottom'.
        overwrite_data: bool, optional
            The default is False. If this is set to False, will fetch punch-holes
            from the ImageROI object that may have been defined manually earlier.
        plts : bool, optional
            If True, will plot inbetween and final results in the hole 
            identification.
        **kwargs : dict
            Additional kwargs passed on to the find_holes function.
        """
        assert self._image_roi is not None, 'call set_image_roi_from_parent first'

        if not (set_xray := check_attr(self, '_xray')):
            logger.warning(
                'No xray object set, can only set punch holes for MSI/XRF.'
            )

        if 'side' in kwargs:
            raise ValueError(
                'please provide "side_xray" and "side_data" seperately'
            )

        if set_xray:
            self.xray.set_punchholes(
                remove_gelatine=False, side=side_xray, plts=plts, **kwargs
            )
            self.xray.save(kwargs.get('tag'))

        if (not check_attr(self.image_roi, 'punchholes')) or overwrite_data:
            self.image_roi.set_punchholes(
                remove_gelatine=True, side=side_data, plts=plts, **kwargs
            )
            self.image_roi.save(kwargs.get('tag'))

        # copy over to object attributes
        if set_xray:
            self.holes_xray: list[np.ndarray[int], np.ndarray[int]] = self.xray.punchholes
        self.holes_data: list[np.ndarray[int], np.ndarray[int]] = self.image_roi.punchholes

    def add_depth_correction_with_xray(self, method: str = 'linear') -> None:
        """
        Add a column with corrected depth based on the punch hole correlation
        with the xray image.

        Depending on the specified method, 2 or 4 points are used for the
        depth correction:
            for the method linear, only the two punchholes are used
            for all other methods the top and bottom of the slice are also used

        Parameters
        ----------
        method: str (default 'linear')
            Options are 'linear' / 'l', 'cubic' / 'c'
            and 'piece-wise linear' / 'pwl'

        Returns
        -------
        None
        """
        def idx_to_depth(
                idx: np.ndarray[int], img_shape: tuple[int, ...]
        ) -> np.ndarray[float]:
            """Convert the index into the relative depth."""
            return idx / img_shape[1] * depth_section + self.depth_span[0]

        def append_top_bottom(
                arr: np.ndarray, img_shape: tuple[int, ...]
        ) -> np.ndarray:
            """Insert the indices for the top and bottom of the slice."""
            # insert top
            arr = np.insert(arr, 0, 0)
            # append bottom
            arr = np.append(arr, img_shape[1])
            return arr

        # methods to choose from
        methods: tuple[str, ...] = ('linear', 'cubic', 'piece-wise linear', 'l', 'c', 'pwl')
        assert method in methods, \
            f'method {method} is not valid, valid options are {methods}'
        assert self.holes_data is not None, 'call set_punchholes'
        assert self._data_object is not None, 'set data_object object first'
        assert 'depth' in self.data_object.feature_table.columns, 'set depth column'
        assert check_attr(self, '_image_roi'), 'call require_image_roi'
        assert check_attr(self, '_xray'), 'call require_xray'

        depth_section: float | int = self.depth_span[1] - self.depth_span[0]

        img_xray_shape: tuple[int, int] = self._xray.image.shape[:2]
        img_data_shape: tuple[int, int] = self._image_roi.image.shape[:2]

        # use holes as tie-points
        idxs_xray: np.ndarray[int] = np.array(
            [point[1] for point in self.holes_xray]
        )
        idxs_data: np.ndarray[int] = np.array(
            [point[1] for point in self.holes_data]
        )

        depths: pd.Series = self.data_object.feature_table['depth']

        if method not in ('linear', 'l'):
            idxs_xray: np.ndarray[int] = append_top_bottom(
                idxs_xray, img_xray_shape
            )
            idxs_data: np.ndarray[int] = append_top_bottom(
                idxs_data, img_data_shape
            )

        # depths xray --> assumed to be not deformed, therefore linear depth
        # increase
        if method in ('linear', 'l'):
            # linear function to fit xray depth to msi depth
            coeffs = np.polyfit(
                idx_to_depth(idxs_data, img_data_shape),
                idx_to_depth(idxs_xray, img_xray_shape),
                1
            )
            pol = np.poly1d(coeffs)
            depths_new = pol(depths)
        elif method in ('cubic', 'c'):
            # third degree polynomial fit with 4 points
            # top and bottom are part of tie-points
            coeffs = np.polyfit(
                idx_to_depth(idxs_data, img_data_shape),
                idx_to_depth(idxs_xray, img_xray_shape),
                3
            )
            pol = np.poly1d(coeffs)
            depths_new = pol(depths)
        elif method in ('piece-wise linear', 'pwl'):
            depths_new = np.zeros_like(depths)
            for i in range(len(idxs_xray) - 1):
                # get index pairs
                idxs_xray_pair: np.ndarray[int] = idxs_xray[i:i + 2]
                idxs_data_pair: np.ndarray[int] = idxs_data[i:i + 2]
                coeffs = np.polyfit(
                    idx_to_depth(idxs_data_pair, img_data_shape),
                    idx_to_depth(idxs_xray_pair, img_xray_shape),
                    1
                )
                pol = np.poly1d(coeffs)
                # indices corresponding to current idx pair
                depth_interval = idx_to_depth(idxs_data_pair, img_data_shape)
                mask = (depths >= depth_interval[0]) & \
                       (depths <= depth_interval[1])
                depths_new[mask] = pol(depths[mask])
        else:
            raise NotImplementedError('internal error')

        self.data_object.feature_table['depth_corrected'] = depths_new

    def _get_xray_transform(self) -> tuple[Mapper, bool]:
        # try to load mapper
        mapper = Mapper(path_folder=self.path_folder, tag='xray')
        if loaded := os.path.exists(mapper.save_file):
            mapper.load()
        return mapper, loaded

    def set_xray_transform(
            self,
            plts=False,
            is_piecewise=True,
            method='punchholes',
            flip_xray: bool | None = None,
            **_
    ) -> None:
        """
        This function may only be called if a data, _image_roi and xray object
        have been set, the depth span specified and the punch holes been added
        to the feature table.

        This method will use the punch hole positions and, depending on the
        method, the top and bottom of the sample to first fit the X-ray
        measurement to match the MSI one and then add the transformed X-ray
        image to the feature table. The piece-wise transformation requires that
        the teaching points span the entire ROI. Hence, the corners of the ROI
        are used as additional teaching points. The trafo object currently does
        not support other transformations than piece-wise linear, affine and
        holomorphic.

        Parameters
        ----------
        plts
        is_piecewise
        kwargs

        Returns
        -------

        """
        assert self._data_object is not None, 'set data_object object first'
        assert 'x_ROI' in self.data_object.feature_table.columns, 'call add_pixels_ROI'
        assert check_attr(self, '_image_roi'), 'call require_image_roi'
        assert check_attr(self, '_xray'), 'call require_xray'
        if method == 'punchholes':
            assert self.holes_data is not None, 'call set_punchholes first'
        if (method == 'punchholes') and (flip_xray is not None):
            logger.warning(
                'flipping will be ignored for punch hole method and instead '
                'inferred from the mappings'
            )

        # use transformer to handle rescaling
        t: Transformation = Transformation(
            source=self.xray.image,
            target=self.image_roi.image,
            source_obj_color=self.xray.obj_color,
            target_obj_color=self.image_roi.obj_color
        )
        if method == 'punchholes':
            points_xray: list[np.ndarray[int], np.ndarray[int]] = self.holes_xray.copy()
            points_data: list[np.ndarray[int], np.ndarray[int]] = self.holes_data.copy()

            # provide contours explicitly in case they were defined with
            # non-default parameters
            cont_data: np.ndarray[int] = self.image_roi.main_contour
            cont_xray: np.ndarray[int] = self.xray.main_contour

            t.estimate(
                method=method,
                is_piecewise=is_piecewise,
                points_source=points_xray,
                points_target=points_data,
                contour_source=cont_xray,
                contour_target=cont_data,
                is_rescaled=False
            )
        elif method == 'bounding_box':
            if flip_xray:
                t.estimate(method='flip_ud')
            t.estimate(
                method=method,
                plts=plts
            )
        else:
            raise KeyError(f'method {method} not valid for xray transformation')

        mapper = t.to_mapper(
            path_folder=self.path_folder,
            tag='xray'
        )
        mapper.save()

        if not plts:
            return

        warped_xray = t.fit()

        if method == 'punchholes':
            plt.figure()
            plt.imshow(warped_xray)
            plt.title('warped in transformer')
            plt.show()

            # tie points on msi image
            fig, ax = plt.subplots()
            ax.plot(cont_data[:, 0, 0], cont_data[:, 0, 1])
            plt_cv2_image(self.image_roi.image,
                          hold=True,
                          fig=fig,
                          ax=ax,
                          swap_rb=False)
            ax.scatter(
                [point[1] for point in points_data],
                [point[0] for point in points_data],
                color='r'
            )
            plt.show()

            # tie points on xray section
            fig, ax = plt.subplots()
            plt_cv2_image(t.source.image,
                          hold=True,
                          fig=fig,
                          ax=ax,
                          swap_rb=False)
            ax.plot(cont_xray[:, 0, 0] * t.source_rescale_width,
                    cont_xray[:, 0, 1] * t.source_rescale_height)
            ax.scatter(
                [point[1] * t.source_rescale_width for point in points_xray],
                [point[0] * t.source_rescale_height for point in points_xray],
                c='red'
            )
            plt.show()

        # warped xray image on top of msi
        plt.figure()
        img_xray = warped_xray.copy()
        if self.image_roi.obj_color != self.xray.obj_color:
            img_xray = img_xray.max() - img_xray
        img_xray = rescale_values(img_xray, 0, 255).astype(np.uint8)  # invert
        # img_xray = cv2.equalizeHist(img_xray)

        img_roi = self.image_roi.image_grayscale.astype(float)
        img_roi *= self.image_roi.mask_foreground
        img_roi = rescale_values(img_roi, 0, 255).astype(int)
        # img_roi = cv2.equalizeHist(img_roi.astype(np.uint8))

        img_cpr = np.stack([
            img_xray,
            img_roi // 2 + img_xray // 2,
            img_roi
        ], axis=-1).astype(int)
        plt_cv2_image(img_cpr, swap_rb=False, title='warped image')

    def require_xray_transform(self, **kwargs) -> Mapper:
        mapper, loaded = self._get_xray_transform()
        if not loaded:
            self.set_xray_transform(**kwargs)
            mapper, loaded = self._get_xray_transform()
            assert loaded, 'encountered internal error'
        return mapper

    def add_xray(self, plts: bool = False):
        """
        Add the X-ray measurement as a new column to the feature table of the
        data object.
        """
        assert check_attr(self, '_image_roi'), 'call require_image_roi'
        assert check_attr(self, '_xray'), 'call require_xray'
        mapper, loaded = self._get_xray_transform()
        assert loaded, 'Unable to load mapper, call require_xray_transform first'

        # use transformer to handle rescaling
        t: Transformation = Transformation(
            source=self.xray.image,
            target=self.image_roi.image,
            source_obj_color=self.xray.obj_color,
            target_obj_color=self.image_roi.obj_color
        )

        warped_xray: np.ndarray[float] = mapper.fit(
            t.source.image_grayscale, preserve_range=True
        )

        if plts:
            plt_cv2_image(self.xray.image, title='input image', swap_rb=False)
            plt_cv2_image(t.source.image, title='xray before warp', swap_rb=False)
            plt_cv2_image(warped_xray, title='xray after warp', swap_rb=False)

        # add to feature table
        self.data_object.add_attribute_from_image(warped_xray, 'xray')

    def set_combine_mapper(
            self,
            other: Self,
            self_tilt_correction: bool,
            other_tilt_correction: bool,
            mapping_method: list[str] | None = None,
            mapping_method_kwargs: list[dict] | None = None,
            plts: bool = False,
            **kwargs
    ) -> None:
        """
        Create a mapper that maps the sample region of another project onto this'
        and saves it to disk. Leaves the resolution unchanged.

        Parameters
        ----------
        other: Self
            Other project that is supposed to be combined with this one.
        self_tilt_correction: bool
            Whether a tilt correction shall be used for this project.
        other_tilt_correction: bool
            Whether a tilt correction shall be used for the other project.
        mapping_method: list[str], optional
            The steps to be performed to match the samples (see Transformation
            class for options). If not provided, will use the bounding box for
            a coarse match and if both samples are laminated the tilt correction
            as specified by self/other_tilt_correction and finally a laminae shift.
        mapping_method_kwargs: list[str], optional
            Specific kwargs for the mapping methods. Will combine with but
            overwrite kwargs. If mapping_method is not specified, this is ignored.
        plts: bool, optional
            Whether to plot inbetween results. The default is False.
        """
        if self_tilt_correction:
            assert check_attr(self, '_image_classified'), \
                'need image_classified object when tilt correction is desired.'
            if not os.path.exists(os.path.join(self.path_folder, 'tilt_correction')):
                logger.warning(
                    'did not find tilt correction for self in '
                    'set_combine_mapper, using default parameters to set Mapper.'
                )
            target = self.image_classified.image_corrected
        else:
            assert check_attr(self, '_image_roi'), \
                ('need image_roi object when no tilt correction is used in '
                 'set_combine_mapper.')
            target = self.image_roi.image

        # append name of other project as tag to mapper name
        identifier: str = os.path.basename(other.path_folder).split('.')[0]

        # for source, we must not use tilt corrected, otherwise there is nothing
        # left to correct in transformer
        # (image roi image is always uncorrected)
        source: ImageROI = other.image_roi.image

        t: Transformation = Transformation(
            source=source,
            target=target,
            source_obj_color=other.image_roi.obj_color,
            target_obj_color=self.image_roi.obj_color
        )

        if mapping_method is None:
            logger.info(
                'using default strategy of matching bounding boxes and, '
                'if the sample is laminated, the tilts and laminae'
            )
            logger.info('estimating match of bounding boxes')
            t.estimate('bounding_box', plts=plts, **kwargs)
            if other_tilt_correction:
                logger.info('estimating tilt correction')
                t.estimate('tilt', plts=plts, **kwargs)
            if self._is_laminated:
                if not other._is_laminated:
                    logger.warning('Not using lamination although other is laminated')
                logger.info('estimating laminae correction')
                t.estimate('laminae', plts=plts, **kwargs)
        else:
            if mapping_method_kwargs is None:
                mapping_method_kwargs = [{}] * len(mapping_method)

            logger.info(f'using custom mapping strategy: {mapping_method}')
            for meth, meth_kwargs in zip(mapping_method, mapping_method_kwargs):
                this_kwargs = kwargs | meth_kwargs
                logger.info(f'estimating {meth} with parameters {this_kwargs}')
                t.estimate(meth, plts=plts, **this_kwargs)

        if plts:
            t.plot_fit(use_classified=False)

        mapper = t.to_mapper(path_folder=self.path_folder,
                             tag=f'combine_with_{identifier}')
        mapper.save()

    def add_other_by_shift_and_rescale(self, other: Self):
        """
        Transform feature table of other such x_ROI, y_ROI match by only
        shifting and rescaling. It is assumed that rotation, tilting has been
        performed (see 'data_object_apply_transformation') or is not necessary.
        """
        # determine shift and scaling factor
        # respective corner points form lines that intersect in a point p
        # TODO: find transformation
        ...
        raise NotImplementedError

    def require_combine_mapper(
            self,
            other: Self,
            *args,
            overwrite: bool = False,
            **kwargs
    ) -> Mapper:
        """
        Get a combine mapper.

        If none exists on disk yet, it will be created using set_combine_mapper
        and the provided args and kwargs.
        """
        identifier: str = os.path.basename(other.path_folder).split('.')[0]

        # attempt to load
        mapper = Mapper(path_folder=self.path_folder,
                        tag=f'combine_with_{identifier}')
        if (not os.path.exists(mapper.save_file)) or overwrite:
            self.set_combine_mapper(other, *args, **kwargs)
        mapper.load()
        return mapper

    def _require_combine_mapper(self, *args, **kwargs):
        warnings.warn('[DEPRECATION] _require_combine_mapper will be removed '
                      'in favor of the '
                      'require_combine_mapper in a future version')
        return self.require_combine_mapper(*args, **kwargs)

    def transform_other_data(
            self,
            other: Self,
            use_tilt_correction: bool | Iterable = None,
            **kwargs
    ) -> None:
        """
        Match the coordinate system of another project with this one. Does not
        add the data of the other object to this one.

        This function makes use of the combine_mapper
        (see require_combine_mapper).

        Parameters
        ----------
        other: Self
            Other project.
        use_tilt_correction: bool | Iterable, optional
            Whether to apply a tilt correction. Can be a bool or 2-tuple-like
            of bools. A tuple will be interpreted as self_tilt_correction,
            other_tilt_correction.
        kwargs: Any
            Additional keywords for require_combine_mapper.
        """
        warnings.warn('Behavior of this function changed in version 1.2.2, '
                      'using the old function is discouraged as it may lead to '
                      'unexpected results. Instead use "require_combine_mapper" '
                      'and "data_object_apply_transformation"')

        assert check_attr(other, '_image_roi')
        assert check_attr(other, '_data_object')
        assert 'x_ROI' in other.data_object.columns

        assert check_attr(self, '_image_roi')
        assert check_attr(self, '_data_object')
        assert 'x_ROI' in self.data_object.columns

        # by default, apply tilt correction only to laminated sediments
        if use_tilt_correction is None:
            logger.info(
                'requiring tilt correction since this instance is laminated'
            )
            use_tilt_correction = self._is_laminated
        is_2tuple_bool = (hasattr(use_tilt_correction, '__iter__')
                          and (len(use_tilt_correction) == 2)
                          and all([isinstance(c, bool) for c in use_tilt_correction]))
        assert isinstance(use_tilt_correction, bool) or is_2tuple_bool, (
            f'correct_tilt must be either a bool or an 2-tuple'
            f' of bools, you provided {use_tilt_correction}'
        )

        if is_2tuple_bool:
            self_use_tilt_correction, other_use_tilt_correction = use_tilt_correction
        else:  # determine if tilts have to be corrected (still)
            self_use_tilt_correction = use_tilt_correction
            other_use_tilt_correction = use_tilt_correction

        self_correct_tilt = self_use_tilt_correction and (not self.corrected_tilt)
        other_correct_tilt = other_use_tilt_correction and (not other.corrected_tilt)
        logger.info(f'Determined {self_correct_tilt=} and {other_correct_tilt=}')

        # warn if corrected even though use_tilt_correction is set to False
        if self.corrected_tilt and (not self_use_tilt_correction):
            logger.warning('using tilt corrected feature table for this '
                           'project even though '
                           'use_tilt_correction is set to False')
        if other.corrected_tilt and (not other_use_tilt_correction):
            logger.warning('using tilt corrected feature table for other '
                           'project even though '
                           'use_tilt_correction is set to False')

        # apply tilt corrections
        if self_correct_tilt:
            assert check_attr(self, '_image_classified'), \
                'Need instance of ImageClassified for correcting tilt'

            self.require_tilt_corrector()
            self.data_object_apply_tilt_correction()

        # setting the warp mapper
        # perform tilt correction after matching bounding rectangles
        mapper_warp = self.require_combine_mapper(
            other,
            self_tilt_correction=self_correct_tilt,
            other_tilt_correction=other_correct_tilt,
            **kwargs
        )

        # apply mapping to other
        other.data_object_apply_transformation(mapper_warp, **kwargs)

    def set_time_series(
            self,
            is_continuous: bool = False,
            **kwargs
    ) -> None:
        assert self._data_object is not None, 'call require_data_object'

        if not is_continuous:
            assert self._image_classified is not None, \
                'call require_image_classified'
            assert check_attr(self.image_classified,
                              'params_laminae_simplified',
                              True), \
                'make sure image_classified has parameters for laminae'

        if 'L' not in self.data_object.columns:
            logger.warning(
                'No grayscale column set yet, proceeding anyway, but you might '
                'want to call add_photo'
            )
        if ignore_depth := ('depth' not in self.data_object.columns):
            logger.warning(
                'No depth column set yet, proceeding anyway, but you might want to '
                'add the depths to data_object with add_depth_column first'
            )
        if ignore_age := ('age' not in self.data_object.columns):
            logger.warning(
                'No age column set yet, proceeding anyway, but you might want to '
                'add the ages to data_object with add_age_column first'
            )

        ft_seeds_avg, ft_seeds_success, ft_seeds_std = get_averaged_tables(
            data_object=self.data_object,
            image_classified=self._image_classified,
            is_continuous=is_continuous,
            **kwargs
        )

        # add age column if it is not present yet
        if (not (ignore_age or ignore_depth)) and ('age' not in ft_seeds_avg.columns):
            logging.info(
                'Adding age column by converting depths to age using age model.'
            )
            ft_seeds_avg['age'] = self.age_model.depth_to_age(
                ft_seeds_avg.depth
            )

        self._time_series: TimeSeries = TimeSeries(self.path_d_folder)
        self._time_series.set_feature_tables(ft_seeds_avg,
                                             ft_seeds_success,
                                             ft_seeds_std)

        self._time_series.save(kwargs.get('tag'))

        self._update_files()

    def require_time_series(
            self,
            overwrite: bool = False,
            tag: str | None = None,
            **kwargs
    ) -> TimeSeries:
        if (self._time_series is not None) and (not overwrite):
            logger.info('returning cached time series')
            return self._time_series
        if (
                check_attr(self, 'TimeSeries_file') or
                (tag is not None)
        ) and (not overwrite):
            logger.info(f'fetching time series from {self.path_d_folder}')
            self._time_series = TimeSeries(self.path_d_folder)
            try:
                self._time_series.load(tag)
                if (
                        check_attr(self._time_series,
                                   '_feature_table',
                                   True)
                ):
                    return self._time_series
                logger.warning(
                    'Loaded time series instance is missing a feature table.'
                )
            except FileNotFoundError:
                logger.warning(f'did not find file with specified {tag=}.')

        logger.info('setting new time series instance')
        self.set_time_series(tag=tag, **kwargs)

        return self._time_series

    @property
    def time_series(self) -> TimeSeries:
        return self.require_time_series()

    def plot_comp(
            self,
            comp: str | float | int,
            source: str = 'data_object',
            tolerance: float = 3e-3,
            da_export_file: str | None = None,
            plot_on_background: bool = False,
            title: str | None = None,
            **kwargs
    ):
        """
        Plot a compound from different sources.

        Parameters
        ----------
        comp : str | float | int
            m/z or name of the compound or feature to plot.
        source: str
            Source attribute from which to fetch the values. Default is
            "data_object". Other possibilities are
            - reader for MSI
            - da_export for MSI
            - da_export_file for MSI
            - spectra for MSI
            - time_series
        tolerance : float
            Deviation for m/z's or compounds in reader or feature table.
        da_export_file: str
            File with DataAnalysis export to be used for plotting (only
            necessary if source is "da_export_file"
        plot_on_background: bool, optional
            If this is set to True, will use the photo as background for the
            ion images.
        title: str, optional
            Title of the figure
        kwargs: Any
            Additional keywords for IonImagePlotter.plot_comp


        """
        assert check_attr(self, source) if source != 'reader' else self._is_MSI, \
            f'{source} not set'

        plotter: IonImagePlotter = IonImagePlotter(
            project=self,
            source=source,
            tolerance=tolerance,
            da_export_file=da_export_file,
        )
        plotter.plot_comp(
            comp=comp,
            plot_on_background=plot_on_background,
            title=title,
            **kwargs
        )

    def plot_punchholes(self):
        assert (
                check_attr(self, 'holes_data')
                and check_attr(self, 'holes_xray')), \
            'Call set_punchholes before calling this method'
        fig, axs = plt.subplots(nrows=2)
        self.image_roi.plot_punchholes(fig=fig, axs=axs[0], hold=True)
        self.xray.plot_punchholes(fig=fig, axs=axs[1], hold=True)
        axs[0].set_title('Data')
        axs[1].set_title('X-ray')
        plt.show()

    def plot_overview(self):
        """Plot figures representing the current state of the project."""
        fig, axs = plt.subplots(
            nrows=5, ncols=2, figsize=(10, 25), frameon=False, layout='constrained'
        )
        if self._image_handler is not None:
            self.image_handler.plot_overview(fig=fig, ax=axs[0, 0], hold=True)
            axs[0, 0].set_title('Measurement region')
        if self._image_sample is not None:
            image = self.image_sample.image
            x, y, w, h = self.image_sample.xywh_ROI
            plt_rect_on_image(
                fig=fig,
                image=image,
                box_params=region_in_box(image=image, x=x, y=y, w=w, h=h),
                hold=True
            )
            axs[0, 1].set_title('Sample region')
        if self._image_roi is not None:
            self.image_roi.plot_overview(fig=fig, axs=[axs[1, 0]], hold=True)
            axs[1, 0].set_title('Classification and punch-holes')
        if self._image_classified is not None:
            self.image_classified.plot_overview(fig=fig, axs=[axs[1, 1]], hold=True)
            axs[1, 1].set_title('Laminae')


class ProjectXRF(ProjectBaseClass):
    _is_MSI: bool = False

    def __init__(
            self,
            path_folder: str,
            path_bcf_file: str | None = None,
            path_image_file: str | None = None,
            path_image_roi_file: str | None = None,
            measurement_name: str | None = None,
            is_laminated: bool = True
    ):
        self.path_folder: str = path_folder
        self.path_d_folder: str = self.path_folder

        if measurement_name is not None:
            self.measurement_name: str = measurement_name
        else:
            self._set_measurement_name()
        self._set_files(
            path_bcf_file, path_image_file, path_image_roi_file
        )
        self._is_laminated: bool = is_laminated

    def _set_measurement_name(self):
        # folder should have measurement name in it --> a capital letter, 4 digits and
        # a lower letter
        folder = os.path.split(self.path_folder)[1]
        pattern = r'^[A-Z]\d{3,4}[a-z]'

        match = re.match(pattern, folder)
        result = match.group() if match else None
        if result is None:
            logger.warning(
                f'Folder {folder} does not contain measurement name at beginning. '
                f'To mute this warning rename folder or provide the measurement '
                f'name upon initialization',
            )
            self.measurement_name = None
        else:
            self.measurement_name: str = result

    def _set_files(
            self,
            path_bcf_file: str | None,
            path_image_file: str | None,
            path_image_roi_file: str | None
    ) -> None:
        """Try to find files and infer measurement name."""
        files: list[str] = os.listdir(self.path_folder)

        if path_bcf_file is None:
            # it's fine to keep this on the error level without throwing an error
            bcf_file = find_matches(
                substrings=self.measurement_name,
                files=files,
                file_types='bcf'
            )
        else:
            bcf_file = os.path.basename(path_bcf_file)
        if path_image_file is None:
            image_file = find_matches(
                substrings='Mosaic',
                files=files,
                file_types=['tif', 'bmp', 'png', 'jpg'],
                must_include_substrings=True
            )
            if image_file is None:
                logger.info(
                    'found no image file containing "Mosaic", expanding search'
                    ' to anything containing "ROI"')

                image_file = find_matches(
                    substrings='ROI',
                    files=files,
                    file_types=['tif', 'bmp', 'png', 'jpg'],
                    must_include_substrings=True
                )
        else:
            image_file = os.path.basename(path_image_file)
        if path_image_roi_file is None:
            image_roi_file = find_matches(
                substrings='Video 1',
                files=files,
                file_types='txt',
                must_include_substrings=True,
            )
        else:
            image_roi_file = os.path.basename(path_image_roi_file)

        self.bcf_file: str = bcf_file
        self.image_file: str = image_file
        self.image_roi_file: str = image_roi_file

        targets_folder = {
            'XRF.pickle',
            'ImageSample.pickle',
            'ImageROI.pickle',
            'ImageClassified.pickle',
            'SampleImageHandlerXRF.pickle',
            'AgeModel.pickle',
            'XRayROI.pickle',
            'TimeSeries.pickle'
        }

        dict_files = {}
        for file in files:
            if file not in targets_folder:
                continue
            k_new = file.split('.')[0] + '_file'
            dict_files[k_new] = file

        self.__dict__ |= dict_files

    @property
    def path_bcf_file(self) -> str | None:
        if not check_attr(self, 'bcf_file'):
            return None
        return os.path.join(self.path_folder, self.bcf_file)

    @property
    def path_image_file(self):
        return os.path.join(self.path_folder, self.image_file)

    @property
    def path_image_roi_file(self):
        return os.path.join(self.path_folder, self.image_roi_file)

    def _update_files(self):
        self._set_files(
            path_bcf_file=self.path_bcf_file,
            path_image_file=self.path_image_file,
            path_image_roi_file=self.path_image_roi_file
        )

    def set_image_handler(self):
        """
        Initialize and SampleImageHandlerXRF object and set the photo.

        Returns
        -------
        None.

        """
        assert (self.image_file is not None) and check_attr(self, 'image_roi_file'), \
            'ensure the image files have good names (matching measurement name)'
        self._image_handler: SampleImageHandlerXRF = SampleImageHandlerXRF(
            path_folder=self.path_folder,
            path_image_file=self.path_image_file,
            path_image_roi_file=self.path_image_roi_file
        )
        if not check_attr(self.image_handler, '_extent_spots'):
            self._image_handler.set_photo()
            self._image_handler.set_extent_data()
            self._image_handler.save()
        if (
                (not check_attr(self._image_handler, '_image_roi'))
                or (not check_attr(self._image_handler, '_data_roi_xywh'))
        ):
            self._image_handler.set_photo_roi()
            self._image_handler.save()

        self._update_files()

    def require_image_handler(self, overwrite=False, **_) -> SampleImageHandlerXRF:
        if (self._image_handler is not None) and (not overwrite):
            return self._image_handler
        if check_attr(self, 'SampleImageHandlerXRF_file') and (not overwrite):
            self._image_handler: SampleImageHandlerXRF = SampleImageHandlerXRF(
                path_folder=self.path_folder,
                path_image_file=self.path_image_file,
                path_image_roi_file=self.path_image_roi_file
            )
            self._image_handler.load()
            self._image_handler.set_photo()

            if all([
                check_attr(self._image_handler, attr)
                for attr in ('_extent_spots', '_image_roi', '_data_roi_xywh')
            ]):
                return self._image_handler
            logger.warning(
                'found partially initialized SampleImageHandler, calling '
                'setter'
            )
        logger.info('Setting new image handler')
        self.set_image_handler()

        return self._image_handler

    def set_data_object(self, tag=None, **kwargs):
        self._data_object = XRF(
            path_folder=self.path_folder,
            measurement_name=self.measurement_name,
            **kwargs
        )
        self._data_object.set_feature_table_from_txts()
        self._data_object.feature_table['R'] = 0
        self._data_object.save(tag)

    def require_data_object(
            self,
            overwrite: bool = False,
            tag: str | None = None,
            **kwargs
    ) -> XRF:
        if (self._data_object is not None) and (not overwrite):
            return self._data_object
        if (
                check_attr(self, 'XRF_file')
                or (tag is not None)
        ) and (not overwrite):
            self._data_object = XRF(
                path_folder=self.path_folder,
                measurement_name=self.measurement_name,
                **kwargs
            )
            try:
                self._data_object.load(tag)
                if check_attr(self._data_object, 'feature_table'):
                    return self._data_object
                logger.warning(
                    'loaded object does not have feature table, setting new one'
                )
            except FileNotFoundError:
                logger.warning('Could not find data object with the {tag=}')

        logger.info('Setting new data object')
        self.set_data_object(tag=tag, **kwargs)
        return self._data_object


class ProjectMSI(ProjectBaseClass):
    _is_MSI: bool = True
    uk37_proxy: UK37 | None = None

    def __init__(
            self,
            path_folder,
            depth_span: tuple[int | float, int | float] = None,
            d_folder: str | None = None,
            mis_file: str | None = None,
            is_laminated: bool = True
    ) -> None:
        """
        Initialization with folder.

        Parameters
        ----------
        path_folder : str
            path to folder with d-folder, mis file etc.
        depth_span : tuple[int], optional
            For core data, the depth section of the slice in cm. 
            The default is None. Certain features will not be available without 
            the depth section.
        d_folder : str, optional
            Name of the d-folder
        mis_file : str, optional,
            Name of the mis file in which measurement parameters are specified


        Returns
        -------
        None.

        """
        self.path_folder = path_folder
        if depth_span is not None:
            self.depth_span = depth_span

        self._set_files(d_folder, mis_file)
        self._is_laminated: bool = is_laminated

    def _set_files(self, d_folder: str | None = None, mis_file: str | None = None):
        """
        Find d folder, mis file and saved objects inside the d folder. If
        multiple d folders are inside the .i folder, the d folder must be
        specified

        Returns
        -------
        None.

        """
        folder_structure = get_folder_structure(self.path_folder)
        if d_folder is None:
            d_folders: list[str] = get_d_folder(self.path_folder, return_mode='valid')
            assert len(d_folders) < 2, \
                (f'Found multiple d folders {d_folders},'
                 ' please specify the name of the file by providing the d_folder'
                 ' keyword upon initialization.')
            assert len(d_folders) > 0, \
                (f'Found no d folder in {self.path_folder}, maybe the d '
                 f'folder does not end in .d?')
            d_folder: str = d_folders[0]
            # best guess for mis file name is that it is the same as the folder
            # name
            name_mis_file: str = d_folder.split('.')[0] + '.mis'
        else:
            name_mis_file: None = None
        if mis_file is None:
            mis_file: str = get_mis_file(self.path_folder, name_file=name_mis_file)

        dict_files: dict[str, str] = {
            'd_folder': d_folder,
            'mis_file': mis_file
        }

        if dict_files.get('d_folder') is not None:
            self.d_folder: str = dict_files['d_folder']
        else:
            raise FileNotFoundError(f'Found no d folder in {self.path_folder}')

        if dict_files.get('mis_file') is not None:
            self.mis_file: str = dict_files['mis_file']
        else:
            raise FileNotFoundError(f'Found no mis file in {self.path_folder}')

        # try finding savefiles inside d-folder
        targets_d_folder: list[str] = [
            'peaks.sqlite',
            'Spectra.pickle',
            'MSI.pickle',
            'AgeModel.pickle',
            'TimeSeries.pickle'
        ]
        targets_folder: list[str] = [
            'ImageSample.pickle',
            'ImageROI.pickle',
            'ImageClassified.pickle',
            'SampleImageHandlerMSI.pickle',
            'DataAnalysisExport.pickle',
            'XRayROI.pickle'
        ]

        # get d_folder
        idxs = np.where([
            entry['name'] == dict_files['d_folder']
            for entry in folder_structure['children']
        ])
        assert len(idxs[0]) == 1, 'found no or conflicting files, check folder'
        idx = idxs[0][0]
        dict_files_dfolder = find_files(
            folder_structure['children'][idx],
            *targets_d_folder
        )

        dict_files_folder = find_files(
            folder_structure,
            *targets_folder
        )

        for k, v in dict_files_dfolder.items():
            k_new = k.split('.')[0] + '_file'
            dict_files[k_new] = v
        for k, v in dict_files_folder.items():
            k_new = k.split('.')[0] + '_file'
            dict_files[k_new] = v

        if os.path.exists(os.path.join(self.path_d_folder, 'Spectra.hdf5')):
            dict_files['hdf_file'] = 'Spectra.hdf5'

        self.__dict__ |= dict_files

    def _update_files(self):
        self._set_files(
            d_folder=os.path.basename(self.path_d_folder),
            mis_file=os.path.basename(self.path_mis_file)
        )

    @property
    def path_d_folder(self):
        return os.path.join(self.path_folder, self.d_folder)

    @property
    def path_mis_file(self):
        return os.path.join(self.path_folder, self.mis_file)

    def set_image_handler(self, **kwargs) -> None:
        """
        Initialize and SampleImageHandlerMSI object and set the photo.

        Returns
        -------
        None.

        """
        self._image_handler = SampleImageHandlerMSI(
            path_folder=self.path_folder,
            path_d_folder=self.path_d_folder,
            path_mis_file=self.path_mis_file
        )

        self._image_handler.set_extent_data(
            reader=kwargs.get('reader'),
            spot_info=kwargs.get('spot_info')
        )
        self._image_handler.set_photo_roi(**kwargs)
        self._image_handler.save()
        self._update_files()

    def require_image_handler(
            self,
            overwrite: bool = False,
            **kwargs
    ) -> SampleImageHandlerMSI:
        # return existing
        if (self._image_handler is not None) and (not overwrite):
            return self._image_handler
        # load and set image
        if check_attr(self, 'SampleImageHandlerMSI_file') and (not overwrite):
            logger.info(f'loading SampleHandler from {self.path_folder}')
            self._image_handler = SampleImageHandlerMSI(
                path_folder=self.path_folder,
                path_d_folder=self.path_d_folder,
                path_mis_file=self.path_mis_file
            )
            self._image_handler.load()
            # make sure it has _extent_spots
            if check_attr(self._image_handler, '_extent_spots'):
                return self._image_handler
            logger.warning('Loaded image handler misses extent_spots')

        logger.info('Initiating new ImageHandler instance')
        self.set_image_handler(**kwargs)
        return self._image_handler

    def get_mcf_reader(self, **kwargs) -> ReadBrukerMCF:
        reader = ReadBrukerMCF(self.path_d_folder, **kwargs)
        reader.create_reader()
        reader.create_indices()
        reader.create_spots()
        reader.set_meta_data()
        if 'limits' not in kwargs:
            try:
                reader.set_casi_window()
            except ValueError as e:
                logger.error(e)
                logger.error('setting window to broadband (400, 2000)')
                reader.limits = (400, 2000)
        return reader

    def set_hdf_file_targets(
            self,
            targets: Iterable[float],
            tolerances: float | Iterable[float] = 3e-3 * 3,
            **kwargs
    ) -> None:
        """
        Sets an hdf5 file that only contains intensity values around targets.

        Parameters
        ----------
        targets : Iterable[float]
            The target masses
        tolerances : float | Iterable[float], optional
            The tolerances

        """
        if not hasattr(tolerances, '__iter__'):
            tolerances: list[float] = [tolerances] * len(targets)
        assert len(tolerances) == len(targets), \
            'must provide float or iterable of same length as targets for tolerances'

        reader: ReadBrukerMCF = kwargs.pop('reader', self.get_mcf_reader())
        mzs = get_mzs_for_limits(reader.limits,
                                 delta_mz=kwargs.pop('delta_mz', 1e-4))
        mask = np.zeros_like(mzs, dtype=bool)
        for target, tolerance in zip(targets, tolerances):
            mask_target = (mzs > target - tolerance) & (mzs < target + tolerance)
            mask |= mask_target
        self.set_hdf_file(reader=reader, mzs=mzs[mask], **kwargs)

    def set_hdf_file(
            self, reader: ReadBrukerMCF | None = None, **kwargs
    ) -> None:
        handler = hdf5Handler(self.path_d_folder)
        logger.info(f'creating hdf5 file in {self.path_d_folder}')

        if reader is None:
            reader: ReadBrukerMCF = self.get_mcf_reader()

        handler.write(reader, **kwargs)

        # update files
        self._update_files()

    def get_hdf_reader(self) -> hdf5Handler:
        reader = hdf5Handler(self.path_d_folder)
        return reader

    def require_hdf_reader(self, overwrite: bool = False) -> hdf5Handler:
        if (not check_attr(self, 'hdf_file')) or overwrite:
            reader: ReadBrukerMCF = self.get_mcf_reader()
            self.set_hdf_file(reader)
        return self.get_hdf_reader()

    def get_reader(self, prefer_hdf: bool = True) -> ReadBrukerMCF | hdf5Handler:
        if check_attr(self, 'hdf_file') and prefer_hdf:
            reader = self.get_hdf_reader()
        else:
            reader = self.get_mcf_reader()
        return reader

    def set_spectra(
            self,
            reader: ReadBrukerMCF | hdf5Handler = None,
            full: bool = True,
            spectra: Spectra | None = None,
            SNR_threshold: float = 2,
            targets: list[str | float] | None = None,
            plts: bool = False,
            **kwargs
    ):
        if reader is None:
            reader = self.require_hdf_reader()
        # create spectra object
        if spectra is None:
            self._spectra: Spectra = Spectra(reader=reader)
        else:
            self._spectra = spectra

        if not full:
            logger.info('Setting partially initialized spectra object')
            return

        if not np.any(self._spectra.intensities.astype(bool)):
            logger.info('spectra object does not have a summed intensity')
            self._spectra.add_calibrated_spectra(reader=reader, **kwargs)
            if plts:
                self._spectra.plot_calibration_functions(reader, n_plot=3)
        if not check_attr(self.spectra, '_peaks'):
            logger.info('spectra object does not have peaks')
            self._spectra.set_peaks(**kwargs)
        if not check_attr(self.spectra, '_kernel_params'):
            logger.info('spectra object does not have kernels')
            self._spectra.set_kernels(**kwargs)
        if targets is not None:
            self._spectra.set_targets(targets, reader=reader, plts=plts, **kwargs)
            self._spectra.filter_line_spectra(binned_snr_threshold=SNR_threshold, **kwargs)
        elif not check_attr(self._spectra, '_line_spectra', True):
            logger.info('spectra object does not have binned spectra')
            self._spectra.bin_spectra(reader, **kwargs)
            self._spectra.filter_line_spectra(binned_snr_threshold=SNR_threshold, **kwargs)
        if not check_attr(self.spectra, '_feature_table'):
            self._spectra.set_feature_table(**kwargs)

        if plts:
            self._spectra.plot_summed()

        self._spectra.save(kwargs.get('tag'))
        self._update_files()

    def require_spectra(
            self,
            overwrite: bool = False,
            tag: str | None = None,
            **kwargs
    ) -> Spectra:
        if check_attr(self, '_spectra') and (not overwrite):
            return self._spectra
        # load from existing file
        if (
            (
                check_attr(self, 'Spectra_file')
                or (tag is not None)
            ) and (not overwrite)
        ):
            self._spectra: Spectra = Spectra(
                path_d_folder=self.path_d_folder, initiate=False
            )
            if os.path.exists(self._spectra.get_save_file(tag=tag)):
                logger.info(f'loaded spectra with {tag=}')
                self._spectra.load(tag)
            else:
                self._spectra: None = None
                logger.warning(f'Could not find spectra object with {tag=} '
                               f'in {self.path_d_folder}')
                # if feature_table or line spectra are set, we are good to return
            if check_attr(self._spectra, '_line_spectra'):
                logger.info('loaded fully initialized spectra object')
                self._spectra.set_feature_table()
                return self._spectra
            elif check_attr(self._spectra, '_feature_table'):
                logger.info('loaded fully initialized spectra object')
                return self._spectra
            # if full is set to False, we can also return
            logger.warning('loaded partially initialized spectra object')
            if ('full' in kwargs) and (kwargs.get('full') is False):
                return self._spectra

        if self._spectra is None:
            logger.info('Initializing new spectra object')
        else:
            logger.info('Continuing processing of spectra object')
        self.set_spectra(
            spectra=self._spectra,  # continue from loaded spectra or None
            tag=tag,
            **kwargs
        )
        return self._spectra

    @property
    def spectra(self) -> Spectra:
        return self.require_spectra()

    def set_da_export(self, path_file: str, **kwargs):
        tag = kwargs.pop('tag', None)
        self._da_export = DataAnalysisExport(path_file=path_file, **kwargs)
        self._da_export.set_feature_table()
        self._da_export.save(tag)

        self._update_files()

    def require_da_export(
            self,
            overwrite=False,
            tag: str | None = None,
            **kwargs
    ) -> DataAnalysisExport:
        if (self._da_export is not None) and (not overwrite):
            return self._da_export

        if (
            (
                check_attr(self, 'DataAnalysisExport_file')
            ) and (not overwrite)
        ):
            self._da_export: DataAnalysisExport = DataAnalysisExport(
                path_file=''
            )
            try:
                self._da_export.load(tag)
                if check_attr(self._da_export, 'feature_table'):
                    logger.info('loaded fully initialized spectra object')
                    return self._da_export
                logger.warning('Loaded corrupted DataAnalysisExport object')
            except FileNotFoundError:
                logger.warning(f'Could not find DataAnalysisExport with {tag=}')

        logger.info('creating new DataAnalysisExport object')
        self.set_da_export(tag=tag, **kwargs)
        return self._da_export

    @property
    def da_export(self):
        return self.require_da_export()

    def set_data_object(self, source='spectra', **kwargs):
        self._data_object: MSI = MSI(
            self.path_d_folder, path_mis_file=self.path_mis_file
        )
        self._data_object.set_distance_pixels(kwargs.get('distance_pixels'))
        if source == 'spectra':
            assert (
                    check_attr(self.spectra, 'feature_table')
                    or check_attr(self.spectra, 'line_spectra')
            ), 'set spectra object first'
            self._data_object.inject_feature_table_from(self.spectra)
        elif source == 'da_export':
            assert (
                    check_attr(self.da_export, 'feature_table')
            ), 'set spectra object first'
            # use msi_feature_extraction
            self._data_object.inject_feature_table_from(self._da_export)
        else:
            raise NotImplementedError()

    def require_data_object(
            self,
            overwrite=False,
            tag: str | None = None,
            **kwargs
    ):
        # return an existing instance
        if (self._data_object is not None) and (not overwrite):
            return self._data_object
        # try to load an instance from disk
        if (
                check_attr(self, f'MSI_file')
                or (tag is not None)
        ) and (not overwrite):
            self._data_object: MSI = MSI(
                self.path_d_folder, path_mis_file=self.path_mis_file
            )
            try:
                self._data_object.load(tag)
                if check_attr(self.data_object, '_feature_table'):
                    # loaded good instance, can return
                    return self._data_object
                else:
                    logger.warning(
                        f'loaded corrupted {self._data_object.__class__.__name__} '
                        'instance, setting new object')
            except FileNotFoundError:
                logger.warning(f'Could not find data object with {tag=}')

        logger.info('Initializing new data object')
        self.set_data_object(tag=tag, **kwargs)
        return self._data_object

    def add_xrf(self, project_xrf: ProjectXRF, **kwargs) -> None:
        self.combine_with_project(project_xrf, tag='xrf', **kwargs)

    def set_UK37(
            self,
            correction_factor: float = 1,
            method_SST: str = 'bayspline',
            prior_std_bayspline: int = 10,
            **kwargs
    ):
        assert self._time_series is not None

        self.uk37_proxy = UK37(time_series=self.time_series, **kwargs)
        self.uk37_proxy.correct(correction_factor=correction_factor)
        self.uk37_proxy.add_SST(method=method_SST, prior_std=prior_std_bayspline)


class IonImagePlotter:
    def __init__(
            self,
            project: ProjectMSI | ProjectXRF | ProjectBaseClass,
            source: str,
            tolerance: float,
            da_export_file: str | None = None,
    ) -> None:
        sources = (
            'reader',
            'spectra',
            'data_object',
            'time_series',
            'da_export',
            'da_export_file'
        )
        self._is_MSI = isinstance(project, ProjectMSI)

        assert source in sources, \
            f'No source named {source}, must be one of {sources}'
        if source not in ('data_object', 'time_series'):
            assert self._is_MSI, \
                f'{source} is not a valid source for XRF'
        if source == 'da_export_file':
            assert da_export_file is not None, \
                ('if method da_export_file is used, '
                 'the da_export_file must be specified')
        if source in ('data_object', 'time_series', 'da_export'):
            assert check_attr(project, source), \
                f'make sure to set the {source} before selecting it as source'
        self._source = source

        self._tolerance: float = float(tolerance)

        assert (da_export_file is None) or os.path.exists(da_export_file), \
            f'provided {da_export_file=} does not exist'
        self._da_export_file = da_export_file

        self._project = project
        self._set_object()

    def _set_object(self):
        if self._source == 'reader':
            assert check_attr(reader := self._project.get_reader(), 'limits'), \
                'Source "reader" needs attribute "limits"'
            self._object = reader
            return
        if self._source != 'da_export_file':
            self._object = self._project.__getattribute__(self._source)
        # don't need to set an object for da_export_file

    def _set_compound(
            self,
            comp,
    ) -> None:
        """
        Find the target compound in the feature table (closest).

        Raises a value error if the tolerance is exceeded
        """
        if self._source == 'reader':
            if (comp < self._object.limits[0]) or (comp > self._object.limits[1]):
                logger.warning(
                    f'Target compound with mass {comp} is outside the mass window'
                    f'of the reader ({self._object.limits}')
        if self._source in ('da_export_file', 'reader'):
            self._comp = str(round(comp, 4))
            return
        comp_, distance = self._object.get_closest_mz(
            comp,
            max_deviation=self._tolerance,
            return_deviation=True
        )
        if comp_ is None:
            raise ValueError(
                f'No compound found within the tolerance ({self._tolerance * 1e3:.0f} '
                f'mDa), next compound is {distance * 1e3:.0f} mDa away'
            )
        self._comp: str = comp_

    def _pixel_table_from_xml(self) -> pd.DataFrame:
        """Construct a feature table from the xml file."""
        imaging_info: pd.DataFrame = get_spots(
            path_d_folder=self._project.path_d_folder
        )
        names: np.ndarray[str] = imaging_info.spotName
        rxys: np.ndarray[int] = get_rxy(names)
        cols: list = [self._comp, 'R', 'x', 'y']
        df_: pd.DataFrame = pd.DataFrame(
            data=np.zeros((len(names), len(cols))),
            columns=cols
        )
        # put R, x and y in feature table
        df_.iloc[:, 1:] = rxys
        return df_

    def _reader_setup(self) -> tuple[ReadBrukerMCF | hdf5Handler, pd.DataFrame]:
        """Get a reader and the feature table."""
        reader_: ReadBrukerMCF | hdf5Handler = self._project.get_reader()
        reader_.create_reader()
        reader_.create_indices()
        df_: pd.DataFrame = self._pixel_table_from_xml()
        return reader_, df_

    def _max_window_spec(self, spec: Spectrum) -> float:
        """Get the maximum intensity of a spectrum within the specified tolerance."""
        mask: np.ndarray[bool] = (
                (spec.mzs > float(self._comp) - self._tolerance)
                & (spec.mzs < float(self._comp) + self._tolerance)
        )
        return spec.intensities[mask].max()

    def _spectra_iterator(
            self,
            obj: Spectra | ReadBrukerMCF | hdf5Handler,
            reader: ReadBrukerMCF | hdf5Handler,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """Iterate over spectra and extract intensity of target."""
        # iterate over spectra and extract intensity
        n: int = len(reader.indices)
        for it, idx in tqdm(
                enumerate(reader.indices),
                desc=f'Fetching intensities from {obj.__class__.__name__}',
                total=n
        ):
            if isinstance(obj, (ReadBrukerMCF, hdf5Handler)):
                spec: Spectrum = obj.get_spectrum(idx)
            elif isinstance(obj, Spectra):
                spec: Spectrum = obj.get_spectrum(
                    reader, idx, only_intensity=False
                )
            else:
                raise NotImplementedError(
                    f'internal error for object of type {type(obj)}, '
                    f'possible types are {type(Spectra)}, '
                    f'{type(ReadBrukerMCF)}, {type(hdf5Handler)}'
                )
            # window
            df.loc[it, self._comp] = self._max_window_spec(spec)
        return df

    def _get_df(self) -> pd.DataFrame:
        if self._source == 'reader':
            reader, df = self._reader_setup()
            df: pd.DataFrame = self._spectra_iterator(reader, reader, df)
        elif self._source == 'spectra':
            reader, df = self._reader_setup()
            df: pd.DataFrame = self._spectra_iterator(self._object, reader, df)
        elif self._source == 'da_export_file':
            pixel_names, spectra_mzs, spectra_intensities, _ = get_da_export_data(
                self._da_export_file
            )
            df: pd.DataFrame = get_da_export_ion_image(
                mz=self._comp,
                pixel_names=pixel_names,
                spectra_mzs=spectra_mzs,
                data=spectra_intensities,
                tolerance=self._tolerance
            )
        elif self._source == 'data_object':
            df = self._project.data_object.feature_table
        elif self._source == 'time_series':
            df = self._project.data_object.feature_table
        elif self._source == 'da_export':
            df = self._project.da_export.feature_table
        else:
            raise NotImplementedError(f'{self._source} is not implemented')

        return df

    def _get_distance_pixels(self, **kwargs):
        if 'distance_pixels' in kwargs:
            return kwargs
        if not check_attr(
                self._data_object, '_distance_pixels'
        ):
            return kwargs
        kwargs['distance_pixels'] = self._data_object.distance_pixels
        return kwargs

    def _set_data_object(self, df):
        if self._is_MSI:
            self._data_object = MSI(
                self._project.path_d_folder,
                path_mis_file=self._project.path_mis_file
            )
        else:
            self._data_object = XRF(
                self._project.path_folder
            )

        self._data_object.inject_feature_table_from(df, supress_warnings=True)
        self._data_object.feature_table.columns = self._data_object.feature_table.columns.astype(str)
        if not check_attr(self._project, '_data_object'):
            return

        if 'x_ROI' in self._project.data_object.columns:
            self._data_object.feature_table.loc[:, ['x_ROI', 'y_ROI']] = \
                self._project.data_object.feature_table.loc[:, ['x_ROI', 'y_ROI']]
        if 'valid' in self._project.data_object.columns:
            self._data_object.feature_table.loc[:, 'valid'] = \
                self._project.data_object.feature_table.loc[:, 'valid']

    def plot_comp(
            self,
            comp: str | float | int,
            plot_on_background: bool = False,
            title: str | None = None,
            **kwargs
    ):
        self._set_compound(comp)

        df: pd.DataFrame = self._get_df()

        if self._source == 'time_series':
            self._object.plot_comp(self._comp, title=title, **kwargs)
            return

        self._set_data_object(df)

        # try to fetch distance_pixels
        kwargs = self._get_distance_pixels(**kwargs)

        if plot_on_background:
            background_image = convert(
                'cv', 'np', self._project.image_roi.image
            )
            plot_comp_on_image(
                comp=self._comp,
                background_image=background_image,
                data_frame=self._data_object.feature_table,
                title=title,
                **kwargs
            )
        else:
            plot_comp(
                data_frame=self._data_object.feature_table,
                title=title,
                comp=self._comp,
                **kwargs
            )


def get_project(is_MSI: bool, *args, **kwargs) -> ProjectMSI | ProjectXRF:
    if is_MSI:
        return ProjectMSI(*args, **kwargs)
    return ProjectXRF(*args, **kwargs)


def get_image_handler(
        is_MSI: bool, *args, **kwargs
) -> SampleImageHandlerMSI | SampleImageHandlerXRF:
    if is_MSI:
        return SampleImageHandlerMSI(*args, **kwargs)
    return SampleImageHandlerXRF(*args, **kwargs)


class MultiMassWindowProject(ProjectBaseClass):
    """
    other with multiple measurement windows (e.g. XRF and MSI or MSI with
    multiple mass windows).
    """

    def __init__(self, *projects: Iterable[ProjectMSI | ProjectXRF]):
        # map all projects onto first one
        assert len(projects) > 1, 'this object expects more than one project object to combine'
        project_main: ProjectMSI | ProjectXRF = projects[0]
        projects = projects[1:]
        for project in projects:
            self.combine(project_main, project)

    def combine(
            self,
            p_main: ProjectMSI | ProjectXRF,
            p_other: ProjectMSI | ProjectXRF,
            plts: bool = False
    ) -> ProjectMSI | ProjectXRF:
        def expand_image(img):
            # TODO: expand "image" to match ROI
            ...
            raise NotImplementedError()

        assert check_attr(p_other, '_image_roi') and check_attr(p_main, '_image_roi'), \
            'projects must have _image_roi objects'
        t = Transformation(source=p_other.image_roi, target=p_main.image_roi)

        # use bounding box first, image flow second
        t.estimate('bounding_box')
        t.estimate('image_flow')

        # TODO: optimize
        # TODO: issue: image region is not the same as measurement region
        for col in p_other.data_object.feature_table.columns:
            # TODO: fix broken get_comp_as_img reference
            img: np.ndarray[float] = p_other.data_object.get_comp_as_img(col, exclude_holes=False)
            img = expand_image(img)
            t.fit(img)
            p_main.data_object.add_attribute_from_image(img, column_name=col)

        # combine feature tables
        return p_main


class MultiSectionProject:
    """
    project with multiple depth sections combined, loses functionality for
    image objects.
    """

    def __init__(self, *projects: Iterable[ProjectMSI | ProjectXRF]):
        self._combine_spectra(*projects)
        self._combine_feature_tables(*projects)
        self.age_model: AgeModel = self._combine_age_models(*projects)

    def _combine_spectra(self, *projects):
        """Combine spectra using the readers"""
        # TODO: assertions
        readers: list[ReadBrukerMCF | hdf5Handler] = [
            project.get_reader() for project in projects
        ]
        self.spectra: MultiSectionSpectra = MultiSectionSpectra(readers=readers)

    def _combine_age_models(self):
        # TODO: this
        ...

    def _combine_feature_tables(self, *projects):
        # TODO: assertions
        # self.data_object = MultiSectionData(*folders)
        if check_attr(self.spectra, 'line_spectra') and not check_attr(self.spectra, 'feature_table'):
            self.spectra.feature_table = self.spectra.set_feature_table()
        if check_attr(self.spectra, 'feature_table'):
            self.data_object.feature_table = self.spectra.feature_table
        else:
            # TODO: this
            ...
            raise NotImplementedError()
