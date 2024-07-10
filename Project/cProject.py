"""
This module contains the Project class which is used to manage various objects for XRF and MSI measurements.
"""
from matplotlib import patches
from scipy.interpolate import griddata
from tqdm import tqdm

from data.helpers import plot_comp, transform_feature_table
from exporting.from_mcf.helper import Spectrum
from imaging.register.helpers import Mapper
from imaging.util.Image_boxes import region_in_box
from imaging.util.Image_plotting import plt_rect_on_image
from util import Convinience
from util.manage_obj_saves import class_to_attributes

from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler
from exporting.from_mcf.cSpectrum import Spectra, MultiSectionSpectra
from data.cMSI import MSI
from data.cXRF import XRF
from Project.file_helpers import (
    get_folder_structure, find_files, get_mis_file,
    get_d_folder, search_keys_in_xml, get_image_file, find_matches, ImagingInfoXML, get_rxy
)
from data.cAgeModel import AgeModel
from imaging.main.cImage import ImageSample, ImageROI, ImageClassified
from imaging.util.Image_convert_types import (
    ensure_image_is_gray, PIL_to_np
)
from imaging.util.coordinate_transformations import rescale_values
from imaging.util.find_XRF_ROI import find_ROI_in_image, plt_match_template_scale
from imaging.XRay.cXRay import XRay
from imaging.register.transformation import Transformation

import os
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import logging

from timeSeries.cTimeSeries import TimeSeries
from timeSeries.cProxy import RatioProxy, UK37

from typing import Iterable, Self

from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw

PIL_Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class SampleImageHandlerMSI(Convinience):
    """
    Given the mis file and folder, find image and area (of MSI) of sample.

    Example usage:
    # initialize
    i_handler = SampleImageHandlerMSI(path_folder='path/to/your/msi/folder')
    # read the extent of the data from the reader (if provided) or the ImagingInfoXML file
    i_handler.set_extent_data()
    # set the photo specified in the mis file (PIL Image)
    i_handler.set_photo()
    # only after photo and data extent have been set, the set_photo_ROI function becomes available
    # with match_pixels set to True, this will return the photo in the data ROI where each pixel
    # in the image corresponds to a data point
    roi = i_handler.set_photo_ROI()
    # save the handler inside the d folder for faster future usage
    i_handler.save()


    # if an instance has been saved before, the handler can be loaded
    # initialize
    i_handler = SampleImageHandlerMSI(path_folder='path/to/your/msi/folder')
    i_handler.load()

    """

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
            The d folder inside the folder. If not provided, the folder name is searched inside the path_folder
            Specifying this is only necessary when multiple d folders are inside the folder.
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

    def set_photo(self):
        """
        Set the photo from the determined file as PIL image.

        Returns
        -------
        None.

        """
        self.image: PIL_Image = PIL_Image.open(self.path_image_file)

    def set_extent_data(
            self,
            reader: ReadBrukerMCF | None = None,
            imaging_xml: ImagingInfoXML | None = None
    ) -> None:
        """
        Get spot names from MCFREader and set extent of pixels based on that.

        Parameters
        ----------
        reader : ReadBrukerMCF, optional
            Reader to get the pixel indices. The default is None. If not speci-
            fied, this method will create a new ImagingInfoXML instance in this scope.
        imaging_xml : ImagingInfoXML, optional
            An ImagingInfoXML instance. If not specified, this method will create a new instance.

        Returns
        -------
        None.

        """
        if (reader is None) or (not hasattr(reader, 'spots')):  # no reader specified or reader does not have spots
            if imaging_xml is None:  # create new instance
                imaging_xml: ImagingInfoXML = ImagingInfoXML(
                    path_d_folder=self.path_d_folder
                )
            pixel_names = imaging_xml.spotName  # get pixel names from reader
        else:
            pixel_names = reader.spots.names  # get pixel names from reader
        # initialize the extent
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
        self.extent_spots = (xmin, xmax, ymin, ymax)

    def set_photo_ROI(
            self,
            match_pxls: bool = True,
            plts: bool = False
    ) -> None:
        """
        Match image and data pixels and set extent in data and photo pixels.

        Parameters
        ----------
        match_pxls : bool, optional
            Whether to resize the image to the datapoints. The default is True.
            False will return the original image cropped to the measurement 
            region.
        plts : bool, optional
            Whether to plot inbetween results. The default is False.

        Returns
        -------
        None

        """
        assert hasattr(self, 'extent_spots'), 'call set_extent_data'
        if not hasattr(self, 'image'):
            self.set_photo()
        # search the mis file for the point data and image file
        mis_dict: dict = search_keys_in_xml(self.path_mis_file, ['Point'])

        # get points specifying the measurement area
        points_mis: list[str] = mis_dict['Point']
        # format points
        points: list[tuple[int, int]] = []
        # get the points of the defined area
        for point in points_mis:
            p: tuple[int, int] = (int(point.split(',')[0]), int(point.split(',')[1]))
            points.append(p)

        if plts:
            # draw measurement area on top of original image
            img_rect = self.image.copy()
            draw = PIL_ImageDraw.Draw(img_rect)
            # define linewidth in terms of image size
            linewidth = round(min(self.image._size[:2]) / 100)
            # the PIL rectangle function is very specific about the order of points
            if len(points) < 3:  # for rectangle
                # p1 --> smaller x value
                points_: list[tuple[int, int]] = points.copy()
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
                draw.polygon(points, outline=(255, 0, 0), width=linewidth)
            plt.figure()
            plt.imshow(img_rect, interpolation='None')
            plt.show()

        # get the extent of the image
        points_x: list[int] = [p[0] for p in points]
        points_y: list[int] = [p[1] for p in points]

        # the extent of measurement area in pixel coordinates
        x_min_area: int = np.min(points_x)
        x_max_area: int = np.max(points_x)
        y_min_area: int = np.min(points_y)
        y_max_area: int = np.max(points_y)

        # get extent of data points in txt-file
        x_min_FT, x_max_FT, y_min_FT, y_max_FT = self.extent_spots

        # resize region in photo to match data points
        if match_pxls:
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

        self.photo_ROI_xywh: tuple[int, ...] = (xp, yp, wp, hp)  # photo units
        self.data_ROI_xywh: tuple[int, ...] = (xd, yd, wd, hd)  # data units
        self.image_roi: np.ndarray[int] = img_resized

    def plot_overview(self):
        img = PIL_to_np(self.image)

        # x, y, w, h = self.data_ROI_xywh
        # rect_data = patches.Rectangle((x, y), w, h, fill=False, edgecolor='r')

        x, y, w, h = self.photo_ROI_xywh
        rect_photo = patches.Rectangle((x, y), w, h, fill=False, edgecolor='g')

        fig, ax = plt.subplots()
        plt.imshow(img)
        # ax.add_patch(rect_data)
        ax.add_patch(rect_photo)
        plt.show()


class SampleImageHandlerXRF(Convinience):
    """
    Image handler for XRF measurements. Finds the image region corresponding to the measurement area.

    If path_image_roi_file is not specified, it is assumed that the image file ROI is the same as the measurement area.
    path_folder is only used for loading and saving the object.
    Since the object is saved with a non-unique name, multiple measurements should be located in separate folders.

    The path_image_roi_file should be a txt file ending in _Video.txt

    Example usage:
    # initialize
    # make sure to pass both the image_file and image_roi_file if the measurement does not cover the entire
    # sediment sample!
    i_handler = SampleImageHandlerXRF(
        path_folder='path/to/your/xrf/measurement',
        path_image_file='path/to/your/photo',
        path_image_roi_file='path/to/your/video/file'
    )
    # make sure to call set_photo before set_extent_data()
    # set the photo of the sample and the ROI
    # if path_image_roi_file is not specified, image will be the same as image_roi
    i_handler.set_photo()

    # read the extent of the data
    i_handler.set_extent_data()

    # only after photo and data extent have been set, the set_photo_ROI function becomes available
    # with match_pixels set to True, this will return the photo in the data ROI where each pixel
    # in the image corresponds to a data point
    roi = i_handler.set_photo_ROI()
    # save the handler inside the d folder for faster future usage
    i_handler.save()

    # if an instance has been saved before, the handler can be loaded
    # initialize
    i_handler = SampleImageHandlerMSI(path_folder='path/to/your/msi/folder')
    i_handler.load()
    """

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

        if path_image_roi_file is None:
            path_image_roi_file: str = path_image_file

        self.ROI_is_image: bool = path_image_file == path_image_roi_file
        if path_image_file is not None:
            self.image_file: str = os.path.basename(path_image_file)
            self.image_roi_file: str = os.path.basename(path_image_roi_file)

    @property
    def path_image_file(self):
        return os.path.join(self.path_folder, self.image_file)

    @property
    def path_image_roi_file(self):
        return os.path.join(self.path_folder, self.image_roi_file)

    def set_photo(self) -> None:
        """Set the image and ROI from the files."""

        def txt2uint8(path_file: str) -> np.ndarray[np.uint8]:
            arr: np.ndarray = pd.read_csv(path_file, sep=';').to_numpy()
            # convert int64 to uint8
            arr: np.ndarray[np.uint8] = (arr / 2 ** 8).astype(np.uint8)
            return arr

        if os.path.splitext(self.path_image_file)[1] == '.txt':
            arr: np.ndarray[np.uint8] = txt2uint8(self.path_image_file)
            self.image: PIL_Image = PIL_Image.fromarray(arr, 'L')
        else:
            self.image: PIL_Image = PIL_Image.open(self.path_image_file)

        if self.ROI_is_image:
            self.image_roi: PIL_Image = self.image.copy()
        else:
            arr: np.ndarray[np.uint8] = txt2uint8(self.path_image_roi_file)
            self.image_roi: PIL_Image = PIL_Image.fromarray(arr, 'L')

    def set_extent_data(self, **kwargs: dict) -> None:
        """
        Set the extent of the spots. This function has to be called after set_photo

        If the image is the same as the ROI, this will simply be the full extent.
        Otherwise, cv2's template matching function will be evoced with varying scaling factors to obtain the
        precise position and scale of the ROI inside the of the image.

        Parameters
        ----------
        kwargs: dict
            Optional parameters for find_ROI_in_image.

        Returns
        -------
        None
        """
        assert hasattr(self, 'image'), 'call set_photo first'

        if self.ROI_is_image:
            # xmin, xmax, ymin, ymax
            self.extent_spots = (0, self.image._size[1], 0, self.image._size[0])
        else:
            # function expects arrays, not PIL_Images
            loc, scale = find_ROI_in_image(
                image=PIL_to_np(self.image),
                image_roi=PIL_to_np(self.image_roi),
                **kwargs
            )

            # convert image_ROI resolution to image resolution
            self.scale_conversion: float = scale
            # xmin, xmax, ymin, ymax
            self.extent = (
                loc[0],
                loc[0] + round(scale * self.image_roi.size[0]),  # size is (width, height)
                loc[1],
                loc[1] + round(scale * self.image_roi.size[1])
            )
            self.extent_spots = tuple(round(p / scale) for p in self.extent)
        logger.info(f'found the extent of the data to be {self.extent} (pixel coordinates)')

    def plt_extent_data(self):
        """
        Using the scale conversion factor and top left corner of the ROI,
        plot an image where the section corresponding to the ROI is replaced.
        This function can only be used after set_extent_data

        """
        assert hasattr(self, 'scale_conversion'), 'call set_extent_data'

        scale: float = self.scale_conversion
        loc: tuple[int, int] = (self.extent[0], self.extent[2])

        plt_match_template_scale(
            image=ensure_image_is_gray(PIL_to_np(self.image)),
            template=PIL_to_np(self.image_roi),
            loc=loc,
            scale=scale
        )

    def set_photo_ROI(self, match_pxls: bool = True) -> None:
        """
        Get the image ROI corresponding to the measurement area.

        Parameters
        ----------
        match_pxls: bool, the default is True.
            If this is set to True, the returned image ROI will have the same resolution as the data pixels.

        Returns
        -------
        image_ROI : PIL_Image
            The image ROI corresponding to the measurement area.
        """
        assert hasattr(self, 'extent_spots'), 'call set_extent_data first'
        # get extent of data points in txt-file
        x_min_area, x_max_area, y_min_area, y_max_area = self.extent
        x_min_meas, x_max_meas, y_min_meas, y_max_meas = self.extent_spots

        # resize region in photo to match data points
        if match_pxls:
            img_resized = self.image_roi
        else:
            img_resized = self.image.crop(
                (x_min_area, y_min_area, x_max_area, y_max_area)
            )

        self.photo_ROI_xywh = [  # photo units
            x_min_area,
            y_min_area,
            abs(x_max_area - x_min_area),
            abs(y_max_area - y_min_area)
        ]
        self.data_ROI_xywh = [  # data units
            0,  # XRF coordinates always start at 0
            0,  # XRF coordinates always start at 0
            abs(x_max_meas - x_min_meas),
            abs(y_max_meas - y_min_meas)
        ]
        self.image_roi = img_resized

    def plot_overview(self):
        img = PIL_to_np(self.image)

        # x, y, w, h = self.data_ROI_xywh
        # rect_data = patches.Rectangle((x, y), w, h, fill=False, edgecolor='r')

        x, y, w, h = self.photo_ROI_xywh
        rect_photo = patches.Rectangle((x, y), w, h, fill=False, edgecolor='g')

        fig, ax = plt.subplots()
        plt.imshow(img)
        # ax.add_patch(rect_data)
        ax.add_patch(rect_photo)
        plt.show()


class ProjectBaseClass:
    """
    Abstract base class for ProjectMSI and ProjectXRF.
    """
    # placeholders for objects
    age_model: AgeModel | None = None
    depth_span: tuple[float, float] | None = None
    age_span: tuple[float, float] | None = None
    holes_data = None
    image_handler: SampleImageHandlerMSI | SampleImageHandlerXRF | None = None
    image_sample: ImageSample = None
    image_roi: ImageROI = None
    image_classified: ImageClassified = None
    path_folder: str | None = None
    path_d_folder: str | None = None
    spectra: Spectra = None
    data_obj: MSI | XRF = None
    xray: XRay = None
    time_series: TimeSeries = None
    # flags
    _is_laminated = None
    _is_MSI = None
    _corrected_tilt: bool = False

    def _update_files(self):
        pass

    def load_age_model(self, path_file: str) -> None:
        """
        Load an AgeModel object from the specified file path and save it to the
        d folder for easier access later on.

        Parameters
        ----------
        path_file : str
            File of the age model object.

        Returns
        -------
        None.
        """
        self.age_model: AgeModel = AgeModel(path_file=path_file)

    def set_age_model(
            self,
            path_file: str | None = None,
            load: bool = True,
            **kwargs_read: dict
    ) -> None:
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
        # if an age model save is located in the folder, load it
        if hasattr(self, 'AgeModel_file') and load:
            self.age_model: AgeModel = AgeModel(self.path_d_folder)
        # if a file is provided, load it from the file provided
        elif load and (path_file is not None):
            self.load_age_model(path_file)
        # otherwise create new age model
        else:
            self.age_model: AgeModel = AgeModel(path_file, **kwargs_read)
            self.age_model.path_file = self.path_d_folder
            self.age_model.save()

        self._update_files()

    def set_depth_span(self, depth_span: tuple[float, float]) -> None:
        """
        Set the depth span as a tuple where the first element is the upper depth and the second element the lower depth
        in cm.

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
            self, depth_span: tuple[float, float] | None = None, age_span: tuple[float, float] | None = None
    ) -> None:
        """
        Set the age span of the measurement either from the age model or the provided tuple.

        Age is expected to be in years before the year 2000 (b2k).

        Example usage
        -------------
        >>>p.set_age_span(age_span=(11400, 11420))
        sets the age span from 11400 to 11420 yrs b2k.
        >>>p.set_age_span(depth_span=(490, 495))
        sets the age span using the age model.
        >>>p.set_age_span()
        sets the age span using the age model and the previously specified depth span.

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
            self.age_span: tuple[float, float] = tuple(self.age_model.depth_to_age(depth_span))

    def set_image_handler(self, **_):
        pass

    def set_image_sample(
            self,
            obj_color: str | None = None,
            overwrite: bool = False,
            **kwargs_area: dict
    ) -> None:
        """
        Set image_sample from image_handler.

        This method is available after the image_handler has been set.

        Parameters
        ----------
        obj_color: str, optional
            The color of the sample relative to the background (either 'dark' or 'light'). If not specified,
            this parameter will be determined automatically or loaded from the saved object, if available.
        kwargs_area: dict, optional
            keyword arguments passed on to ImageSample.sget_sample_area

        Returns
        -------
        None

        """
        assert self.image_handler is not None, \
            'call set_image_handler first'

        self.image_sample: ImageSample = ImageSample(
            path_folder=self.path_folder,
            image=self.image_handler.image,
            image_type='pil',
            obj_color=obj_color
        )
        if hasattr(self, 'ImageSample_file') and not overwrite:
            logger.info('loading ImageSample')
            self.image_sample.load()
        if not hasattr(self.image_sample, '_xywh_ROI'):
            try:
                self.image_handler.set_photo_ROI()
            except Exception as e:
                logger.error(e)
                logger.info('Could not set photo ROI, continuing with fitting box')
            if hasattr(self.image_handler, 'photo_ROI_xywh'):
                extent_x: tuple[int, int] = (
                    self.image_handler.photo_ROI_xywh[0], self.image_handler.photo_ROI_xywh[2]
                )
            else:
                extent_x: None = None
            self.image_sample.set_sample_area(extent_x=extent_x, **kwargs_area)
            self.image_sample.save()

        self._update_files()

    def set_image_roi(self, overwrite: bool = False) -> None:
        """
        Set an ImageROI instance.

        This function can only be called after set_image_sample has been called.

        Parameters
        ----------
        overwrite : bool, optional
            If True, initialize new ImageROI instance, even if there is one found on disk.

        Returns
        -------
        None

        """
        assert self.image_sample is not None, 'call set_image_sample first'

        # create image_roi using image from image_sample
        self.image_roi: ImageROI = ImageROI.from_parent(self.image_sample)
        if hasattr(self, 'ImageROI_file') and not overwrite:
            logger.info('loading ImageROI')
            self.image_roi.load()
            return

        # attempt to classify laminated sample
        if self.age_span is None:
            logger.info(
                'age_span has not been defined. If you want to analyze a laminated sediment, '
                'set the age_span and call this function again.'
            )
            return

        self.image_roi.age_span = self.age_span
        if not hasattr(self.image_roi, 'image_classification'):
            self.image_roi._require_classification()
        self.image_roi.save()

        self._update_files()

    def set_image_classified(
            self,
            peak_prominence: float = .1,
            max_slope: float = .1,
            downscale_factor: float = 1 / 16,
            overwrite: bool = False,
            **kwargs
    ) -> None:
        """
        Set an ImageClassified instance, assuming an ImageROI instance has been saved before.

        Parameters
        ----------
        peak_prominence: float (default =0.1)
            The relative peak prominence above which peaks in the brightness
            function are considered as relevant.
        max_slope: float (default =0.1)
            Maximum allowed slope of laminae. A slope of 0.1 corresponds to
            10 % or arctan(0.1) = 5.7 degrees
        downscale_factor: float (default = 1 / 16)
            Downscaling factor to be applied to the image before tuning the
            parameters. The closer this value is to 0, the worse the
            resolution, but the faster the convergence.

        Returns
        -------
        None
        """
        assert self.image_roi is not None, 'call set_image_roi before'

        # initialize ImageClassified
        self.image_classified: ImageClassified = ImageClassified.from_parent(
            self.image_roi, **kwargs
        )
        if hasattr(self, 'ImageClassified_file') and (not overwrite):
            logger.info('loading ImageClassified ...')
            self.image_classified.load()
        self.image_classified.age_span = self.age_span
        if not hasattr(self.image_classified, 'params_laminae_simplified'):
            logger.info('setting laminae in ImageClassified ...')
            self.image_classified.set_laminae_params_table(
                peak_prominence=peak_prominence,
                max_slope=max_slope,
                downscale_factor=downscale_factor,
                **kwargs
            )
            self.image_classified.save()

        self._update_files()

    def set_images(self, overwrite: bool = False, **kwargs):
        self.set_image_handler(overwrite=overwrite)
        self.set_image_sample(overwrite=overwrite, **kwargs)
        self.set_image_roi(overwrite=overwrite)
        if self._is_laminated:
            self.set_image_classified(overwrite=overwrite, **kwargs)

    def add_tic(self, imaging_info_xml: ImagingInfoXML | None = None):
        """Add the TIC for each pixel to the feature table of the data obj."""
        assert self.data_obj is not None, "No data object"

        if imaging_info_xml is None:
            imaging_info_xml: ImagingInfoXML = ImagingInfoXML(
                path_d_folder=self.path_d_folder
            )
        ft_imaging: pd.DataFrame = imaging_info_xml.feature_table.loc[
            :, ['R', 'x', 'y', 'tic', 'minutes']
        ]
        self.data_obj.feature_table = pd.merge(
            self.data_obj.feature_table, ft_imaging, how="left"
        )

    def add_pixels_ROI(self) -> None:
        """
        Add image pixels to data points in the feature table of the data_obj.

        This requires that the image handler and sample have been set.
        Creates new columns x_ROI and y_ROI for the pixel coordinates in the feature table.

        Returns
        -------

        """
        assert self.image_sample is not None, 'call set_image_sample first'
        assert self.image_handler is not None, 'call set_image_handler'

        attrs = ('image_roi', 'photo_ROI_xywh', 'data_ROI_xywh')
        if not all([hasattr(self.image_handler, attr) for attr in attrs]):
            self.image_handler.set_photo_ROI()

        image_ROI_xywh = self.image_sample._require_image_sample_area()[1]
        data_ROI_xywh = self.image_handler.data_ROI_xywh
        photo_ROI_xywh = self.image_handler.photo_ROI_xywh

        self.data_obj._pixels_get_photo_ROI_to_ROI(
            data_ROI_xywh, photo_ROI_xywh, image_ROI_xywh
        )

    def add_photo(self, median: bool = False) -> None:
        """
        Add the gray-level values of the photo to the feature table of the data_obj.

        This function has to be called after add_pixels_ROI.
        In general the resolution of the data points is smaller than that of the photo.
        By default, the closest value is used.
        The column in the feature table is 'L'

        median: bool (default False)
            If median is True, the median intensities inside the photo will be ascribed to each data point.
            If median is False, the closest values will be used.

        Returns
        -------
        None
        """
        assert self.data_obj is not None, 'set data object first'
        assert 'x_ROI' in self.data_obj.feature_table.columns, \
            'add x_ROI, y_ROI coords with add_pixels_ROI'
        image = ensure_image_is_gray(
            self.image_sample.image_sample_area
        )
        self.data_obj.add_attribute_from_image(image, 'L', median=median)

    def add_holes(self, **kwargs) -> None:
        """
        Add classification for holes and sample to the feature table of the data_obj.

        This function has to be called after the ImageROI and data object have been set.
        Uses the closest pixel value.
        The new column is called 'valid' where holes are associated with a value of 0.

        Returns
        -------
        None
        """
        assert self.image_roi is not None, 'set image_roi first'
        assert self.data_obj is not None, 'set data_object first'
        image = self.image_roi.image_binary
        self.data_obj.add_attribute_from_image(image, 'valid', median=False, **kwargs)

    def add_light_dark_classification(self, **kwargs) -> None:
        """
        Add light and dark classification to the feature table of the data_obj.
        This only considers the foreground (valid) pixels.

        The new column inside the feature table of the data_obj is called 'classification'.

        Returns
        -------
        None
        """
        assert self.image_roi is not None, 'call set_image_roi'
        assert self.data_obj is not None, 'set data_object first'

        image: np.ndarray[int] = self.image_roi.image_classification
        self.data_obj.add_attribute_from_image(image, 'classification', **kwargs)

    def add_laminae_classification(self, **kwargs) -> None:
        """
        Add light and dark laminae classification to the feature table of the data_obj.
        This only considers the foreground (valid) pixels.

        This requires that the ImageClassified instance has already been set.

        The new column inside the feature table of the data_obj is called 'classification_s'.
        A column called 'classification_se' is also added, which is the classification image but laminae have been
        expanded to fill the entire image, except holes.

        Returns
        -------
        None
        """
        assert self.image_classified is not None, 'call set_image_classified'
        assert self.data_obj is not None, 'set data_object first'

        image: np.ndarray[int] = self.image_classified.image_seeds
        image_e: np.ndarray[int] = self.image_classified.get_image_expanded_laminae()
        self.data_obj.add_attribute_from_image(image, 'classification_s', **kwargs)
        self.data_obj.add_attribute_from_image(image_e, 'classification_se', **kwargs)

    def add_depth_column(self, exclude_gaps: bool = True) -> None:
        """
        Add the depth column to the data_obj using the depth span.

        Linearly map xs (from the data_obj.feature_table) to depths
        (based on depth_span).

        If exclude_gaps is True, depth intervals where no sediment is present, will be excluded from the depth
        calculation (they will be assigned the same depth as the last valid depth).
        """
        assert self.data_obj is not None, 'set the data_obj first'
        assert self.depth_span is not None, 'set the depth_span first'
        if exclude_gaps:
            assert 'valid' in self.data_obj.feature_table.columns, 'set holes in data_obj first'

        min_depth, max_depth = self.depth_span
        # convert seed pixel coordinate to depth and depth to age
        x: pd.Series = self.data_obj.feature_table.x_ROI

        self.data_obj.feature_table['depth'] = np.nan

        if not exclude_gaps:
            depths = rescale_values(
                x,
                new_min=min_depth,
                new_max=max_depth,
                old_min=x.min(),
                old_max=x.max()
            )
            self.data_obj.feature_table['depth'] = depths

        else:
            # row-wise check if any of the pixels are part of the sample
            valid_mask: np.ndarray[bool] = self.data_obj.feature_table.pivot(
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

            self.data_obj.feature_table['depth'] = self.data_obj.feature_table.apply(map_depth, axis=1)
        # add new column to feature table

    def add_age_column(self, use_corrected: bool = False) -> None:
        """
        Add an age column to the data_obj using the depth column and age model.

        Map the depths in the data_obj.feature_table with the age model.

        Parameters
        ----------
        use_corrected : bool
            Whether to use the corrected depth column ('depth_corrected').
            data_obj must have the corrected depth column, otherwise an error is raised.

        Returns
        -------
        None.
        """
        assert self.age_model is not None, 'set age model first'
        assert self.data_obj is not None, f'did not set data_obj yet'
        assert hasattr(self.data_obj, 'feature_table'), 'must have data_obj'
        if use_corrected:
            depth_col = 'depth_corrected'
        else:
            depth_col = 'depth'
        assert depth_col in self.data_obj.feature_table.columns, \
            'data object must have {depth_col} column, call add_depth_column and set the exclude_gaps accordingly'

        self.data_obj.feature_table['age'] = self.age_model.depth_to_age(
            self.data_obj.feature_table[depth_col]
        )

    def set_xray(
            self, path_image_file,
            depth_section: tuple[float, float] | None = None,
            obj_color: str = 'dark',
            **_
    ):
        """
        Set the X-Ray object from the specified image file and depth section.

        The object is accessible with p.xray with p being the project.
        This method performs all processing steps.

        Parameters
        ----------
        path_image_file: str
            The image file with the XRay measurement.
        depth_section: tuple[float, float] | None
            The depth section covered by the photo as a tuple in cm.
            This is required if the object is not loaded from disk
        obj_color: str (default= 'dark')
            The relative brightness of the object compared to background.
            Options are 'dark', 'light'.
        """
        if depth_section is None:
            assert self.depth_span is not None, \
                'set depth span or pass the depth_section argument'

        self.xray: XRay = XRay(
            path_image_file=path_image_file,
            depth_section=depth_section,
            obj_color=obj_color
        )
        # try to load from disk
        folder: str = os.path.dirname(path_image_file)
        name: str = 'XRay.pickle'
        if os.path.exists(os.path.join(folder, name)):
            self.xray.load()
        else:
            self.xray._require_image_sample_area()
            self.xray.remove_bars()
            self.xray.save()

        self._update_files()

    def data_obj_apply_tilt_correction(self) -> None:
        assert not self._corrected_tilt, 'tilt has already been corrected'
        assert self.data_obj is not None, 'set data_obj.'
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
        if os.path.exists(mapper._get_disc_folder_and_file()[1]):
            mapper.load()
            # get transformed coordinates
            XT, YT = mapper.get_transformed_coords()  # as expected

            # insert into feature table
            self.data_obj.add_attribute_from_image(  # as expected
                XT, 'x_ROI_T', fill_value=np.nan
            )
            self.data_obj.add_attribute_from_image(  # as expected
                YT, 'y_ROI_T', fill_value=np.nan
            )

            # fit feature table
            self.data_obj.feature_table = transform_feature_table(
                self.data_obj.feature_table
            )
            logger.info('successfully loaded mapper and applied tilt correction')

            self._corrected_tilt: bool = True
        else:
            raise FileNotFoundError(
                f'expected to find a Mapping at '
                f'{mapper._get_disc_folder_and_file()[1]}, make sure to set '
                f'the image_classification with use_tilt_correction set to True'
            )
        self.image_classified.set_corrected_image()

    def set_punchholes(
            self,
            side_xray: str | None = None,
            side_data: str | None = None,
            plts: bool = False,
            **kwargs
    ):
        """
        Identify square-shaped holes at top or bottom of sample in xray section
        and MSI sample.

        Parameters
        ----------
        side_xray : str, optional
            The side on which the holes are expected to be found in the xray sample, either 'top'
            or 'bottom'.
        side_data : str, optional
            The side on which the holes are expected to be found in the data sample, either 'top'
            or 'bottom'.
        plts : bool, optional
            If True, will plot inbetween and final results in the hole 
            identification.
        **kwargs : dict
            Additional kwargs passed on to the find_holes function.

        Returns
        -------
        None.

        """
        assert self.xray is not None, 'call set_xray first'
        assert self.image_roi is not None, 'call set_image_roi first'

        if 'side' in kwargs:
            raise ValueError('please provide "side_xray" and "side_data" seperately')

        img_xray: ImageROI = self.xray.get_ImageROI_from_section(section_start=self.depth_span)
        img_xray.set_punchholes(remove_gelatine=False, side=side_xray, plts=plts, **kwargs)
        if not hasattr(self.image_roi, 'punchholes'):
            self.image_roi.set_punchholes(
                remove_gelatine=True, side=side_data, plts=plts, **kwargs
            )

        # copy over to object attributes
        self.holes_xray: list[np.ndarray[int], np.ndarray[int]] = img_xray._punchholes
        self.holes_data: list[np.ndarray[int], np.ndarray[int]] = self.image_roi._punchholes

    def add_depth_correction_with_xray(self, method: str = 'linear') -> None:
        """
        Add a column with corrected depth based on the punchhole correlation with the xray image.

        Depending on the specified method, 2 or 4 points are used for the depth correction:
            for the method linear, only the two punchholes are used
            for all other methods the top and bottom of the slice are also used

        Parameters
        ----------
        method: str (default 'linear')
            Options are 'linear' / 'l', 'cubic' / 'c' and 'piece-wise linear' / 'pwl'

        Returns
        -------
        None
        """
        def idx_to_depth(idx: int, img_shape: tuple[int, ...]) -> float:
            """Convert the index into the relative depth."""
            return idx / img_shape[1] * depth_section + self.depth_span[0]

        def append_top_bottom(arr: np.ndarray, img_shape: tuple[int, ...]) -> np.ndarray:
            """Insert the indices for the top and bottom of the slice."""
            # insert top
            arr = np.insert(arr, 0, 0)
            # append bottom
            arr = np.append(arr, img_shape[1])
            return arr

        # TODO: rewrite this to use mapper
        # methods to choose from
        methods: tuple[str, ...] = ('linear', 'cubic', 'piece-wise linear', 'l', 'c', 'pwl')
        assert method in methods, \
            f'method {method} is not valid, valid options are {methods}'
        assert self.holes_data is not None, 'call set_punchholes'
        assert self.data_obj is not None, 'set data_obj object first'
        assert 'depth' in self.data_obj.feature_table.columns, 'set depth column'
        assert self.depth_span is not None, 'set the depth section'

        depth_section: float | int = self.depth_span[1] - self.depth_span[0]

        img_xray_shape: tuple[int, ...] = self.xray.get_section(
            self.depth_span[0], self.depth_span[1]
        ).shape
        # eliminate gelatine pixels
        img_data_shape = self.image_roi.image_grayscale.shape

        # use holes as tie-points
        idxs_xray: np.ndarray[int] = np.array([point[1] for point in self.holes_xray])
        idxs_data: np.ndarray[int] = np.array([point[1] for point in self.holes_data])

        depths: pd.Series = self.data_obj.feature_table['depth']

        if method not in ('linear', 'l'):
            idxs_xray: np.ndarray[int] = append_top_bottom(idxs_xray, img_xray_shape)
            idxs_data: np.ndarray[int] = append_top_bottom(idxs_data, img_data_shape)

        # depths xray --> assumed to be not deformed, therefore linear depth increase
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
                idxs_xray_pair = idxs_xray[i:i + 2]
                idxs_data_pair = idxs_data[i:i + 2]
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
            raise NotImplementedError()

        self.data_obj.feature_table['depth_corrected'] = depths_new

    def add_xray(self, plts=False, is_piecewise=True, **_):
        """
        Add the X-Ray measurement as a new column to the feature table of the data object.

        This function may only be called if a data, image_roi and xray object have been set, the depth span specified
        and the punchholes been added to the feature table.

        This method will use the punchchole positions and, depending on the method, the top and bottom of the sample
        to first fit the X-Ray measurement to match the MSI one and then add the transformed X-Ray image to the
        feature table. The piece-wise transformation requires that the teaching points span the entire ROI. Hence,
        the corners of the ROI are used as additional teaching points. The trafo object currently does not support
        other transformations than piece-wise linear, affine and holomoprhic.

        Parameters
        ----------
        plts
        is_piecewise
        kwargs

        Returns
        -------

        """
        assert self.data_obj is not None, 'set data_obj object first'
        assert 'x_ROI' in self.data_obj.feature_table.columns, 'call add_pixels_ROI'
        assert self.xray is not None, 'call set_xray first'
        assert self.image_roi is not None, 'call set_image_roi first'
        assert self.depth_span is not None, 'set the depth section'
        assert self.holes_data is not None, 'call add_depth_correction_with_xray'

        img_xray: np.ndarray[int] = ensure_image_is_gray(
            self.xray.get_section(self.depth_span[0], self.depth_span[1])
        )
        roi_xray: ImageROI = ImageROI(
            image=img_xray, path_folder=None, obj_color=self.xray.obj_color
        )

        plt.figure()
        plt.imshow(img_xray)
        plt.title('input image')
        plt.show()

        # try to load mapper
        mapper = Mapper(path_folder=self.path_folder, tag='xray')
        if os.path.exists(mapper._get_disc_folder_and_file()[1]):
            is_loaded: bool = True
            mapper.load()
        else:
            is_loaded: bool = False
            # eliminate gelatine pixels
            img_data: np.ndarray[int] = (
                    self.image_roi._require_simplified_image()
                    * self.image_roi._require_foreground_thr_and_pixels()[1]
            )

            points_xray: list[np.ndarray[int], np.ndarray[int]] = self.holes_xray.copy()
            points_data: list[np.ndarray[int], np.ndarray[int]] = self.holes_data.copy()

            cont_data: np.ndarray[int] = self.image_roi._require_main_contour()
            cont_xray: np.ndarray[int] = roi_xray._require_main_contour()

            t: Transformation = Transformation(roi_xray, self.image_roi)
            t._transform_from_punchholes(
                is_piecewise=is_piecewise,
                points_source=points_xray,
                points_target=points_data,
                contour_source=cont_xray,
                contour_target=cont_data,
                is_rescaled=False
            )

            plt.figure()
            plt.imshow(t.fit())
            plt.title('warped in transformer')
            plt.show()

            mapper = t.to_mapper(
                path_folder=self.path_folder,
                tag='xray'
            )
            mapper.save()

        # use (new) transformer to handle rescaling
        t: Transformation = Transformation(source=roi_xray, target=self.image_roi)

        warped_xray: np.ndarray[float] = mapper.fit(
            t.source.image, preserve_range=True
        )

        plt.figure()
        plt.imshow(t.source.image)
        plt.title('xray before warp')

        plt.figure()
        plt.imshow(warped_xray)
        plt.title('xray after warp')

        # add to feature table
        self.data_obj.add_attribute_from_image(warped_xray, 'xray')

        if plts and not is_loaded:
            # tie points on msi image
            plt.figure()
            plt.plot(cont_data[:, 0, 0], cont_data[:, 0, 1])
            plt.imshow(img_data)
            plt.scatter(
                [point[1] for point in points_data],
                [point[0] for point in points_data],
                color='r'
            )
            plt.show()

            # tie points on xray section
            plt.figure()
            plt.imshow(img_xray)
            plt.plot(cont_xray[:, 0, 0], cont_xray[:, 0, 1])
            plt.scatter(
                [point[1] for point in points_xray],
                [point[0] for point in points_xray],
                c='red'
            )
            plt.show()

            # warped xray image on top of msi
            plt.figure()
            img_xray = warped_xray.copy()
            img_xray = (255 - img_xray).astype(np.uint8)  # invert
            mask = img_xray == 255
            img_xray[mask] = 0
            # img_xray = cv2.equalizeHist(img_xray)

            img_roi = self.image_roi.image_grayscale.astype(float)
            img_roi *= self.image_roi._require_foreground_thr_and_pixels()[1]
            img_roi = rescale_values(img_roi, 0, 255).astype(int)
            # img_roi = cv2.equalizeHist(img_roi.astype(np.uint8))

            img_cpr = np.stack([
                img_xray,
                img_roi // 2 + img_xray // 2,
                img_roi
            ], axis=-1)
            plt.imshow(img_cpr)
            plt.title('warped image')
            plt.show()
        elif plts:
            logger.warning('Cannot create plots if mapper is loaded.')

    def combine_with_project(
            self, project: Self, tag: str, plts: bool = False, **kwargs
    ) -> None:
        assert project.image_roi is not None
        assert project.data_obj is not None

        assert self.image_classified is not None
        assert self.data_obj is not None

        mapper = Mapper(path_folder=self.path_folder, tag=tag)

        if os.path.exists(mapper._get_disc_folder_and_file()[1]):
            mapper.load()
        else:
            target: np.ndarray = self.image_classified.image
            source: ImageROI = project.image_roi

            logger.info(
                f'loaded image from image_classified with tilt correction set '
                f'to {self.image_classified.use_tilt_correction}'
            )

            t = Transformation(
                source=source,
                target=target,
                target_obj_color=self.image_classified.obj_color
            )

            t.estimate('bounding_box', plts=plts, **kwargs)
            t.estimate('tilt', plts=plts, **kwargs)
            t.estimate('laminae', plts=plts, **kwargs)

            if plts:
                t.plot_fit(use_classified=False)

            mapper = t.to_mapper(path_folder=self.path_folder, tag='xrf')
            mapper.save()

        x_ROI = project.data_obj.feature_table.x_ROI
        y_ROI = project.data_obj.feature_table.y_ROI
        points = np.c_[x_ROI, y_ROI]
        _, _, w_ROI, h_ROI = project.image_handler.photo_ROI_xywh
        grid_x, grid_y = np.meshgrid(
            np.arange(w_ROI),
            np.arange(h_ROI)
        )
        for comp in tqdm(
                project.data_obj.get_data_columns(),
                desc='adding XRF ion images'
        ):
            values: np.ndarray[float] = project.data_obj.feature_table.loc[
                                        :, comp].to_numpy()

            # turn ion images into images that cover the image of the XRF photo
            ion_image: np.ndarray = griddata(
                points,
                values,
                (grid_x, grid_y),
                method='nearest',
                fill_value=0
            )

            # use (new) transformer to handle rescaling
            t: Transformation = Transformation(
                source=ion_image,
                target=self.image_classified.image,
                source_obj_color=project.image_classified.obj_color,
                target_obj_color=self.image_classified.obj_color
            )

            warped_xray: np.ndarray[float] = mapper.fit(
                t.source.image, preserve_range=True
            )
            # add to feature table
            self.data_obj.add_attribute_from_image(
                image=warped_xray, column_name=comp
            )

    def set_time_series(
            self,
            average_by_col: str = 'classification_se',
            overwrite: bool = False,
            plts: bool = False,
            **kwargs
    ) -> None:
        self.time_series = TimeSeries(self.path_d_folder)
        if hasattr(self, 'TimeSeries_file') and (not overwrite):
            self.time_series.load()
            if hasattr(self.time_series, 'feature_table'):
                return

        assert self.data_obj is not None, 'call set_object'
        assert 'L' in self.data_obj.feature_table.columns, \
            'call add_photo'
        assert 'x_ROI' in self.data_obj.feature_table.columns, \
            'call add_pixels_ROI'
        assert average_by_col in self.data_obj.feature_table.columns, \
            'call add_laminae_classification or use a different column name to average by'
        assert 'depth' in self.data_obj.feature_table.columns, \
            'add the depths to data_obj with add_depth_column first'
        assert 'age' in self.data_obj.feature_table.columns, \
            'add the ages to data_obj with add_age_column first'

        assert self.image_classified is not None, 'call set_image_classified'
        assert hasattr(self.image_classified, 'params_laminae_simplified'), \
            'could not find params table in classified image object, call set_image_classified'

        # x column always included in processing_zone_wise_average
        columns_feature_table = np.append(
            self.data_obj.get_data_columns(),
            ['x_ROI', 'R', 'L', 'depth']
        )
        ft_seeds_avg, ft_seeds_std, ft_seeds_success = self.data_obj.processing_zone_wise_average(
            zones_key=average_by_col,
            columns=columns_feature_table,
            calc_std=True,
            **kwargs
        )

        ft_seeds_avg = ft_seeds_avg.fillna(0)

        # add quality criteria
        cols_quals = ['homogeneity', 'continuity', 'contrast', 'quality']
        # only consider those seeds that are actually in the image
        seeds = self.image_classified.params_laminae_simplified.seed.copy()
        seeds *= np.array([
            1 if (c == 'light') else -1
            for c in self.image_classified.params_laminae_simplified.color]
        )

        row_mask = [seed in ft_seeds_avg.index for seed in seeds]
        # print(row_mask)
        # print('row mask (hits, length): ', sum(row_mask), len(row_mask))

        quals = self.image_classified.params_laminae_simplified.loc[
                row_mask, cols_quals + ['height']
        ].copy()
        # take weighted average for laminae with same seeds (weights are areas=heights)
        quals_weighted = quals.copy().mul(quals.height.copy(), axis=0)
        # reset height (otherwise height column would have values height ** 2
        quals_weighted['height'] = quals.height.copy()
        quals_weighted['seed'] = seeds
        quals_weighted = quals_weighted.groupby(by='seed').sum()
        quals_weighted = quals_weighted.div(quals_weighted.height, axis=0)
        quals_weighted.drop(columns=['height'], inplace=True)

        # join the qualities to the averages table
        ft_seeds_avg = ft_seeds_avg.join(quals_weighted, how='left')
        # insert infty for every column in success table that is not there yet
        missing_cols = set(ft_seeds_avg.columns).difference(
            set(ft_seeds_success.columns)
        )

        for col in missing_cols:
            ft_seeds_success.loc[:, col] = np.infty

        # plot the qualities
        if plts:
            plt.figure()
            plt.plot(quals_weighted.index, quals_weighted.quality, '+', label='qual')
            plt.plot(ft_seeds_avg.index, ft_seeds_avg.quality, 'x', label='ft')
            plt.legend()
            plt.xlabel('seed')
            plt.ylabel('quality')
            plt.title('every x should have a +')
            plt.show()

        # drop index (=seed) into dataframe
        ft_seeds_avg.index.names = ['seed']
        ft_seeds_std.index.names = ['seed']
        ft_seeds_success.index.names = ['seed']
        # reset index
        ft_seeds_avg.reset_index(inplace=True)
        ft_seeds_std.reset_index(inplace=True)
        ft_seeds_success.reset_index(inplace=True)

        ft_seeds_avg['seed'] = ft_seeds_std.seed.astype(int)
        ft_seeds_std['seed'] = ft_seeds_std.seed.astype(int)
        ft_seeds_success['seed'] = ft_seeds_std.seed.astype(int)

        # need to insert the x_ROI from avg
        ft_seeds_std['spread_x_ROI'] = ft_seeds_std.x_ROI.copy()
        ft_seeds_std['x_ROI'] = ft_seeds_avg.x_ROI.copy()

        ft_seeds_success['N_total'] = ft_seeds_success.x_ROI.copy()
        ft_seeds_success['x_ROI'] = ft_seeds_avg.x_ROI.copy()

        # sort by depth
        ft_seeds_avg = ft_seeds_avg.sort_values(by='x_ROI')
        ft_seeds_std = ft_seeds_std.sort_values(by='x_ROI')
        ft_seeds_success = ft_seeds_success.sort_values(by='x_ROI')

        # drop columns with seed == 0
        mask_drop = ft_seeds_avg.seed == 0
        ft_seeds_avg.drop(index=ft_seeds_avg.index[mask_drop], inplace=True)
        ft_seeds_std.drop(index=ft_seeds_std.index[mask_drop], inplace=True)
        ft_seeds_success.drop(index=ft_seeds_success.index[mask_drop], inplace=True)

        # add age column
        ft_seeds_avg['age'] = self.age_model.depth_to_age(ft_seeds_avg.depth)

        self.time_series.feature_table = ft_seeds_avg
        self.time_series.feature_table_standard_deviations = ft_seeds_std
        self.time_series.feature_table_successes = ft_seeds_success

        self.time_series.save()

        self._update_files()

    def plot_comp(
            self,
            comp: str | float | int,
            source: str,
            tolerance: float = 3e-3,
            title: str | None = None,
            **kwargs
    ):
        def pixel_table_from_xml() -> pd.DataFrame:
            """Construct a feature table from the xml file."""
            imaging_info: ImagingInfoXML = ImagingInfoXML(path_d_folder=self.path_d_folder)
            names: np.ndarray[str] = imaging_info.spotName
            rxys: np.ndarray[int] = get_rxy(names)
            cols: list = [comp, 'R', 'x', 'y']
            df_: pd.DataFrame = pd.DataFrame(
                data=np.zeros((len(names), len(cols))),
                columns=cols
            )
            # put R, x and y in feature table
            df_.iloc[:, 1:] = rxys
            return df_

        def reader_setup() -> tuple[ReadBrukerMCF | hdf5Handler, pd.DataFrame]:
            """Get a reader and the feature table."""
            reader_: ReadBrukerMCF | hdf5Handler = self.get_reader()
            reader_.create_reader()
            reader_.create_indices()
            df_: pd.DataFrame = pixel_table_from_xml()
            return reader_, df_

        def spectra_iterator(
                obj: Spectra | ReadBrukerMCF | hdf5Handler,
                reader_: ReadBrukerMCF | hdf5Handler, df_: pd.DataFrame
        ) -> pd.DataFrame:
            """Iterate over spectra and extract intensity of target."""
            comp_f: float = float(comp)
            # iterate over spectra and extract intensity
            n: int = len(reader_.indices)
            for it, idx in tqdm(
                    enumerate(reader_.indices),
                    desc=f'Fetching intensities from {obj.__class__.__name__}',
                    total=n
            ):
                if isinstance(obj, (ReadBrukerMCF, hdf5Handler)):
                    spec: Spectrum = obj.get_spectrum(idx)
                elif isinstance(obj, Spectra):
                    spec: Spectrum = obj._get_spectrum(
                        reader_, idx, only_intensity=False
                    )
                else:
                    raise NotImplementedError(
                        f'internal error for object of type {type(obj)}'
                    )
                # window
                df_.loc[it, comp] = max_window_spec(spec, comp_f)
            return df_

        def max_window_spec(spec: Spectrum, comp_f: float) -> float:
            """Get the maximum intensity of a spectrum within the specified tolerance."""
            mask: np.ndarray[bool] = (
                    (spec.mzs > comp_f - tolerance)
                    & (spec.mzs < comp_f + tolerance)
            )
            return spec.intensities[mask].max()

        def find_compound(
                obj: Spectra | TimeSeries,
                comp_: str | float
        ) -> str:
            """
            Find the target compound in the feature table (closest).

            Raises a value error if the tolerance is exceeded
            """
            comp_, distance = obj._get_closest_mz(
                comp_,
                max_deviation=tolerance,
                return_deviation=True
            )
            if comp_ is None:
                raise ValueError(
                    f'No compound found within the tolerance ({tolerance*1e3:.0f} '
                    f'mDa), next compound is {distance*1e3:.0f} mDa away'
                )
            return comp_

        assert source in ('reader', 'spectra', 'data_obj', 'time_series')

        if source in ('reader', 'spectra'):
            if not self._is_MSI:
                raise KeyError(f'{source} is not a valid source for XRF')
            reader, df = reader_setup()
            obj = self.spectra if source == 'spectra' else reader
            df: pd.DataFrame = spectra_iterator(obj, reader, df)
            plot_comp(data_frame=df, title=title, comp=comp, **kwargs)
        elif source in ('data_obj', 'time_series'):
            assert self.__getattribute__(source) is not None,\
                f'make sure to set the {source} before selecting it as source'
            obj = self.data_obj if source == 'data_obj' else self.time_series
            comp: str = find_compound(obj, comp)
            obj.plot_comp(title=title, comp=comp, **kwargs)


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
            self.measurement_name = measurement_name
        else:
            self._set_measurement_name()
        self._set_files(
            path_bcf_file, path_image_file, path_image_roi_file
        )
        self._is_laminated: bool = is_laminated

    def _set_measurement_name(self):
        # folder should have measurement name in it --> a captial letter, 4 digits and
        # a lower letter
        folder = os.path.split(self.path_folder)[1]
        pattern = r'^[A-Z]\d{3,4}[a-z]'

        match = re.match(pattern, folder)
        result = match.group() if match else None
        if result is None:
            raise OSError(
                f'Folder {folder} does not contain measurement name at beginning, please rename folder',
            )
        else:
            self.measurement_name = result

    def _set_files(
            self,
            path_bcf_file: str | None,
            path_image_file: str | None,
            path_image_roi_file: str | None
    ) -> None:
        """Try to find files and infer measurement name."""
        files: list[str] = os.listdir(self.path_folder)

        if path_bcf_file is None:
            bcf_file = find_matches(
                substrings=self.measurement_name,
                files=files,
                file_types='bcf'
            )
        else:
            bcf_file = os.path.basename(path_bcf_file)
        if path_image_file is None:
            try:
                image_file = find_matches(
                    substrings='Mosaic',
                    files=files,
                    file_types=['tif', 'bmp', 'png', 'jpg'],
                    must_include_substrings=True
                )
            except ValueError as e:
                # sometimes the image file does not contain 'Mosaic' in its name
                logger.info(e)
                logger.info('found no image file containing "Mosaic", expanding search')
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
            'AgeModel.pickle'
        }

        dict_files = {}
        for file in files:
            if file not in targets_folder:
                continue
            k_new = file.split('.')[0] + '_file'
            dict_files[k_new] = file

        self.__dict__ |= dict_files

    @property
    def path_bcf_file(self):
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

    def set_image_handler(self, overwrite=False):
        """
        Initialize and SampleImageHandlerXRF object and set the photo.

        Returns
        -------
        None.

        """
        assert (self.image_file is not None) and hasattr(self, 'image_roi_file'), \
            'ensure the image files have good names (matching measurement name)'
        self.image_handler: SampleImageHandlerXRF = SampleImageHandlerXRF(
            path_folder=self.path_folder,
            path_image_file=self.path_image_file,
            path_image_roi_file=self.path_image_roi_file
        )
        if hasattr(self, 'SampleImageHandlerXRF_file') and (not overwrite):
            self.image_handler.load()
            self.image_handler.set_photo()
        if not hasattr(self.image_handler, 'extent_spots'):
            self.image_handler.set_photo()
            self.image_handler.set_extent_data()
            self.image_handler.save()
        if (not hasattr(self.image_handler, 'image_roi')) or (not hasattr(self.image_handler, 'data_ROI_xywh')):
            self.image_handler.set_photo_ROI()
            self.image_handler.save()

        self._update_files()

    def set_object(self, **kwargs):
        self.data_obj = XRF(
            path_folder=self.path_folder,
            measurement_name=self.measurement_name,
            **kwargs
        )
        if hasattr(self, 'XRF_file'):
            self.data_obj.load()
            if hasattr(self.data_obj, 'feature_table'):
                logger.info('loaded data object')
                return
        self.data_obj.set_feature_table_from_txts()
        self.data_obj.feature_table['R'] = 0


class ProjectMSI(ProjectBaseClass):
    _is_MSI: bool = True

    def __init__(
            self,
            path_folder,
            depth_span: tuple[int | float] = None,
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
        Find d folder, mis file and saved objects inside the d folder. If multiple d folders are inside the .i folder,
        the d folder must be specified

        Returns
        -------
        None.

        """
        folder_structure = get_folder_structure(self.path_folder)
        if d_folder is None:
            d_folders = get_d_folder(self.path_folder, return_mode='valid')
            assert len(d_folders) == 1, \
                (f'Found multiple or no d folders {d_folders},'
                 ' please specify the name of the file by providing the d_folder'
                 ' keyword upon initialization.')
            d_folder = d_folders[0]
            name_mis_file = d_folder.split('.')[0] + '.mis'
        else:
            name_mis_file = None
        if mis_file is None:
            mis_file = get_mis_file(self.path_folder, name_file=name_mis_file)

        dict_files: dict[str, str] = {
            'd_folder': d_folder,
            'mis_file': mis_file
        }

        if dict_files.get('d_folder') is not None:
            self.d_folder = dict_files['d_folder']
        else:
            raise FileNotFoundError(f'Found no d folder in {self.path_folder}')

        if dict_files.get('mis_file') is not None:
            self.mis_file = dict_files['mis_file']
        else:
            raise FileNotFoundError(f'Found no mis file in {self.path_folder}')

        # try finding savefiles inside d-folder
        targets_d_folder = [
            'peaks.sqlite',
            'spectra_object.pickle',
            'MSI.pickle',
            'AgeModel.pickle',
            'Spectra.hdf5'
        ]
        targets_folder = [
            'ImageSample.pickle',
            'ImageROI.pickle',
            'ImageClassified.pickle',
            'SampleImageHandlerMSI.pickle',
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

    def set_image_handler(self, overwrite=False) -> None:
        """
        Initialize and SampleImageHandlerMSI object and set the photo.

        Returns
        -------
        None.

        """
        self.image_handler = SampleImageHandlerMSI(
            path_folder=self.path_folder,
            path_d_folder=self.path_d_folder,
            path_mis_file=self.path_mis_file
        )
        if hasattr(self, 'SampleImageHandlerMSI_file') and (not overwrite):
            logger.info(f'loading SampleHandler from {self.path_folder}')
            self.image_handler.load()
        if not hasattr(self.image_handler, 'extent_spots') or overwrite:
            self.image_handler.set_extent_data()
            self.image_handler.set_photo_ROI()
            self.image_handler.save()
        self.image_handler.set_photo()

    def get_mcf_reader(self) -> ReadBrukerMCF:
        reader = ReadBrukerMCF(self.path_d_folder)
        reader.create_reader()
        reader.create_indices()
        reader.set_meta_data()
        reader.set_QTOF_window()
        return reader

    def create_hdf_file(
            self, reader: ReadBrukerMCF | None = None, overwrite=False, **kwargs
    ) -> hdf5Handler:
        handler = hdf5Handler(self.path_d_folder)
        logger.info(f'creating hdf5 file in {self.path_d_folder}')

        if hasattr(self, 'Spectra_file') and not overwrite:
            logger.info('found hdf5 file, not creating new file because overwrite is set to false')
            return handler

        if reader is None:
            reader = self.get_mcf_reader()

        handler.write(reader, **kwargs)

        # update files
        self._update_files()

        return handler

    def get_hdf_reader(self) -> hdf5Handler:
        reader = hdf5Handler(self.path_d_folder)
        return reader

    def get_reader(self, prefer_hdf: bool = True) -> ReadBrukerMCF | hdf5Handler:
        if hasattr(self, 'Spectra_file') and prefer_hdf:
            reader = self.get_hdf_reader()
        else:
            reader = self.get_mcf_reader()
        return reader

    def require_hdf_reader(self) -> hdf5Handler:
        if not hasattr(self, 'Spectra_file'):
            reader: ReadBrukerMCF = self.get_mcf_reader()
            reader: hdf5Handler = self.create_hdf_file(reader)
        else:
            reader: hdf5Handler = self.get_hdf_reader()
        return reader

    def set_spectra(
            self,
            reader: ReadBrukerMCF | hdf5Handler = None,
            full: bool = True,
            overwrite: bool = False,
            SNR_threshold: float = 2,
            plts: bool = False

    ):
        # create spectra object
        if (
                (
                    hasattr(self, 'spectra_object_file')
                    or hasattr(self, 'Spectra_file')
                ) and not overwrite
        ):
            self.spectra = Spectra(
                path_d_folder=self.path_d_folder, initiate=False
            )
            self.spectra.load()
            if hasattr(self.spectra, 'feature_table'):
                logger.info('loaded fully initialized spectra object')
                return
            elif hasattr(self.spectra, 'line_spectra'):
                logger.info('loaded fully initialized spectra object')
                self.spectra.binned_spectra_to_df()
                self.spectra.save()
                return
            elif reader is None:
                reader = self.require_hdf_reader()
        else:
            if reader is None:
                reader = self.require_hdf_reader()
            self.spectra: Spectra = Spectra(reader=reader)
        if not full:
            logger.info('Setting partially initialized spectra object')
            return

        if not np.any(self.spectra.intensities.astype(bool)):
            logger.info('spectra object does not have a summed intensity')
            self.spectra.add_all_spectra(reader)
            self.spectra.subtract_baseline(plts=plts)
            self.spectra.set_calibrate_functions(reader=reader)
            self.spectra.plot_calibration_functions(reader, n=3)
            self.spectra.add_all_spectra(reader)
            self.spectra.subtract_baseline(overwrite=True)
        if not hasattr(self.spectra, 'peaks'):
            logger.info('spectra object does not have peaks')
            self.spectra.set_peaks()
        if not hasattr(self.spectra, 'kernel_params'):
            logger.info('spectra object does not have kernels')
            self.spectra.set_kernels()
        if not hasattr(self.spectra, 'line_spectra'):
            logger.info('spectra object does not have binned spectra')
            self.spectra.bin_spectra(reader)
            self.filter_line_spectra(SNR_threshold=SNR_threshold)
        if not hasattr(self.spectra, 'feature_table'):
            self.spectra.binned_spectra_to_df()

        if plts:
            self.spectra.plt_summed()

        self.spectra.save()
        self._update_files()

    def set_object(self, overwrite: bool = False):
        self.data_obj: MSI = MSI(self.path_d_folder, path_mis_file=self.path_mis_file)
        if hasattr(self, 'MSI_file') and not overwrite:
            self.data_obj.load()
            if hasattr(self.data_obj, 'feature_table'):
                return

        assert (
            (self.spectra is not None) and (
                hasattr(self.spectra, 'feature_table')
                or hasattr(self.spectra, 'line_spectra')
            )
        ), 'set spectra object first'

        self.data_obj.set_distance_pixels()
        self.data_obj.set_feature_table_from_spectra(self.spectra)

    def add_xrf(self, project_xrf: ProjectXRF, **kwargs) -> None:
        self.combine_with_project(project_xrf, tag='xrf', **kwargs)

    def set_UK37(
            self,
            correction_factor: float = 1,
            method_SST: str = 'BAYSPLINE',
            prior_std_bayspline: int = 10,
            **kwargs
    ):
        assert self.time_series is not None

        self.UK37_proxy = UK37(TS=self.time_series, **kwargs)
        self.UK37_proxy.correct(correction_factor=correction_factor)
        self.UK37_proxy.add_SST(method=method_SST, prior_std=prior_std_bayspline)


def get_project(is_MSI: bool, *args, **kwargs) -> ProjectMSI | ProjectXRF:
    if is_MSI:
        return ProjectMSI(*args, **kwargs)
    return ProjectXRF(*args, **kwargs)


class MultiMassWindowProject(ProjectBaseClass):
    """
    Project with multiple measurement windows (e.g. XRF and MSI or MSI with multiple mass windows).
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

        assert hasattr(p_other, 'image_roi') and hasattr(p_main, 'image_roi'), \
            'projects must have image_roi objects'
        t = Transformation(source=p_other.image_roi, target=p_main.image_roi)

        # use bounding box first, image flow second
        t.estimate('bounding_box')
        t.estimate('image_flow')

        # TODO: optimize
        # TODO: issue: image region is not the same as measurement region
        for col in p_other.data_obj.feature_table.columns:
            # TODO: fix broken get_comp_as_img reference
            img: np.ndarray[float] = p_other.data_obj.get_comp_as_img(col, exclude_holes=False)
            img = expand_image(img)
            t.fit(img)
            p_main.data_obj.add_attribute_from_image(img, column_name=col)

        # combine feature tables
        return p_main


class MultiSectionProject:
    """
    Project with multiple depth sections combined, loses functionality for
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
        # self.data_obj = MultiSectionData(*folders)
        if hasattr(self.spectra, 'line_spectra') and not hasattr(self.spectra, 'feature_table'):
            self.spectra.feature_table = self.spectra.binned_spectra_to_df()
        if hasattr(self.spectra, 'feature_table'):
            self.data_obj.feature_table = self.spectra.feature_table
        else:
            # TODO: this
            ...
            raise NotImplementedError()
