from util.manage_obj_saves import class_to_attributes

from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler
from exporting.from_mcf.cSpectrum import Spectra, MultiSectionSpectra
from data.cMSI import MSI
from data.cXRF import XRF
from data.file_helpers import (
    get_folder_structure, find_files, get_mis_file,
    get_d_folder, search_keys_in_xml, get_image_file, find_matches, ImagingInfoXML
)
from data.cAgeModel import AgeModel
from imaging.main.cImage import ImageSample, ImageROI, ImageClassified
from imaging.util.Image_convert_types import (
    ensure_image_is_gray, PIL_to_np, np_to_PIL
)
from imaging.util.coordinate_transformations import rescale_values
from imaging.util.find_XRF_ROI import find_ROI_in_image
from imaging.XRay.cXRay import XRay
from imaging.register.main import Transformation

import os
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from timeSeries.cTimeSeries import TimeSeries
from timeSeries.cProxy import RatioProxy, UK37

from typing import Iterable

from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw
PIL_Image.MAX_IMAGE_PIXELS = None


class SampleImageHandlerMSI:
    """Given the mis file and folder, find image and area (of MSI) of sample."""

    def __init__(self, path_folder: str, path_d_folder: str | None = None):
        """
        Initialize paths for folder, mis file, d folder and ImageSample object.

        Parameters
        ----------
        path_folder : str
            The folder containing the f folder, mis file and sample photos.

        Returns
        -------
        None.

        """
        self.path_folder = path_folder
        self.path_mis_file = os.path.join(
            self.path_folder, get_mis_file(self.path_folder)
        )
        if path_d_folder is not None:
            self.path_d_folder = path_d_folder
        else:
            self.path_d_folder = os.path.join(
                self.path_folder, get_d_folder(self.path_folder)
            )

        image_file = get_image_file(self.path_folder)
        self.path_image_file = os.path.join(
            self.path_folder, image_file
        )

    def load(self):
        """
        Load saved object attributes from disc (object is saved in the d folder).

        Returns
        -------
        None.

        """
        # name of file
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        # load object, overwrite attributes with save
        with open(os.path.join(self.path_d_folder, name), 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__

    def save(self):
        """
        Save object inside d folder.

        Returns
        -------
        None.

        """
        # keep copy to restore attributes
        dict_backup = self.__dict__.copy()
        keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
        existent_attributes = list(self.__dict__.keys())
        # delete all attributes that are not flagged as relevant
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        with open(os.path.join(self.path_d_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup

    def set_photo(self):
        """
        Set the photo from the determined file as PIL image.

        Returns
        -------
        None.

        """
        self.image = PIL_Image.open(self.path_image_file)

    def set_extent_data(
            self,
            reader: ReadBrukerMCF = None,
            imaging_xml: ImagingInfoXML = None
    ):
        """
        Get spot names from MCFREader and set extent of pixels based on that.

        Parameters
        ----------
        reader : ReadBrukerMCF, optional
            Reader to get the pixel indices. The default is None. If not speci-
            fied, this method will create a new reader in this scope.

        Returns
        -------
        None.

        """
        if (reader is None) or (not hasattr(reader, 'spots')):
            if imaging_xml is None:
                imaging_xml: ImagingInfoXML = ImagingInfoXML(
                    path_d_folder=self.path_d_folder
                )
            pixel_names = imaging_xml.spotName
        else:
            pixel_names = reader.spots.names
        xmin = np.infty
        xmax = -np.infty
        ymin = np.infty
        ymax = -np.infty
        for pixel_name in pixel_names:
            # values in line are separated by semicolons
            img_x = int(re.findall('X(.*)Y', pixel_name)[0])  # x coordinate
            img_y = int(re.findall('Y(.*)', pixel_name)[0])  # y coordinate
            if img_x > xmax:
                xmax = img_x
            if img_x < xmin:
                xmin = img_x
            if img_y > ymax:
                ymax = img_y
            if img_y < ymin:
                ymin = img_y
        self.extent_spots = (xmin, xmax, ymin, ymax)

    def get_photo_ROI(
            self,
            match_pxls: bool = True,
            plts: bool = False
    ):
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
        PIL.Image
            The photo in the (rectangular) measurement region.

        """
        assert hasattr(self, 'extent_spots'), 'call set_extent_data'
        if not hasattr(self, 'image'):
            self.set_photo()
        # search the mis file for the point data and image file
        mis_dict = search_keys_in_xml(self.path_mis_file, ['Point'])

        points_mis = mis_dict['Point']
        points = []
        # get the points of the defined area
        for point in points_mis:
            p = (int(point.split(',')[0]), int(point.split(',')[1]))
            points.append(p)

        if plts:
            img_rect = self.image.copy()
            draw = PIL_ImageDraw.Draw(img_rect)
            linewidth = round(min(self.image._size[:2]) / 100)
            # the PIL rectangle function is very specific about the order of points
            #   for rectangle
            if len(points) < 3:
                # p1 --> smaller x value
                points_ = points.copy()
                points_.sort()
                p1, p2 = points_
                # swap ys of p1 and p2
                if p1[1] > p2[1]:
                    p1_ = (p1[0], p2[1])
                    p2_ = (p2[0], p1[1])
                    p1 = p1_
                    p2 = p2_
                points_ = [p1, p2]
                draw.rectangle(points_, outline=(255, 0, 0), width=linewidth)
            else:
                draw.polygon(points, outline=(255, 0, 0), width=linewidth)
            plt.figure()
            plt.imshow(img_rect, interpolation='None')
            plt.show()

        # get the extent of the image
        points_x = [p[0] for p in points]
        points_y = [p[1] for p in points]

        # the extent of measurement area in pixel coordinates
        x_min_area = np.min(points_x)
        x_max_area = np.max(points_x)
        y_min_area = np.min(points_y)
        y_max_area = np.max(points_y)

        # get extent of data points in txt-file
        x_min_FT, x_max_FT, y_min_FT, y_max_FT = self.extent_spots

        # resize region in photo to match data points
        if match_pxls:
            img_resized = self.image.resize(
                (x_max_FT - x_min_FT + 1, y_max_FT - y_min_FT + 1),  # new number of pixels
                box=(x_min_area, y_min_area, x_max_area, y_max_area),  # area of photo
                resample=PIL_Image.Resampling.LANCZOS  # supposed to be best
            )
        else:
            img_resized = self.image.crop(
                (x_min_area, y_min_area, x_max_area, y_max_area))
        # xywh of data ROI in original image, photo units
        xp = x_min_area
        yp = y_min_area
        wp = x_max_area - x_min_area
        hp = y_max_area - y_min_area
        # xywh of data, data units
        xd = x_min_FT
        yd = y_min_FT
        wd = x_max_FT - x_min_FT
        hd = y_max_FT - y_min_FT

        self.photo_ROI_xywh = (xp, yp, wp, hp)  # photo units
        self.data_ROI_xywh = (xd, yd, wd, hd)  # data units
        self.image_roi = img_resized
        return self.image_roi


class SampleImageHandlerXRF:
    def __init__(
            self,
            path_folder: str = None,
            path_image_file: str = None,
            path_image_roi_file: str = None
    ):
        """
        Initialize.

        Parameters
        ----------
        path_image_file: str
            The file with the photo of the sample
        path_ROI_file: str, optional
            If not provided, it is assumed that no photo of the sample exists.

        Returns
        -------
        None.

        """
        assert (path_folder is not None) or (path_image_file is not None), \
            "provide either a folder or the image file"

        if path_folder is not None:
            self.path_folder = path_folder

        if path_image_roi_file is None:
            path_image_roi_file = path_image_file

        self.ROI_is_image: bool = path_image_file == path_image_roi_file
        if path_image_file is not None:
            self.path_image_file = path_image_file
            self.path_image_roi_file = path_image_roi_file

    def load(self):
        """
        Load saved object attributes from disc (object is saved in the d folder).

        Returns
        -------
        None.

        """
        # name of file
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        # path to save-fil
        # load object, overwrite attributes with save
        with open(os.path.join(self.path_folder, name), 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__

    def save(self):
        """
        Save object inside d folder.

        Returns
        -------
        None.

        """
        # keep copy to restore attributes
        dict_backup = self.__dict__.copy()
        keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
        existent_attributes = list(self.__dict__.keys())
        # delete all attributes that are not flagged as relevant
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'

        with open(os.path.join(self.path_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup

    def set_photo(self):
        """Set the image and ROI from the files."""
        def txt2uint8(path_file: str):
            arr = pd.read_csv(path_file, sep=';').to_numpy()
            # convert int64 to uint8
            arr = (arr / 2 ** 8).astype(np.uint8)
            return arr

        if os.path.splitext(self.path_image_file)[1] == '.txt':
            arr = txt2uint8(self.path_image_file)
            self.image = PIL_Image.fromarray(arr, 'L')
        else:
            self.image = PIL_Image.open(self.path_image_file)

        if self.ROI_is_image:
            self.image_roi = self.image.copy()
        else:
            arr = txt2uint8(self.path_image_roi_file)
            self.image_roi = PIL_Image.fromarray(arr, 'L')

    def set_extent_data(self, **kwargs):
        if self.ROI_is_image:
            self.extent_spots = (0, 0, self.image.shape[1], self.image.shape[0])
        else:
            loc, scale = find_ROI_in_image(
                image=PIL_to_np(self.image),
                image_roi=PIL_to_np(self.image_roi),
                **kwargs
            )

            # convert image_ROI resolution to image resolution
            self.scale_conversion = scale
            # xmin, xmax, ymin, ymax
            self.extent = (
                loc[0],
                loc[0] + int(scale * self.image_roi.size[0] + .5),  # size is (width, height)
                loc[1],
                loc[1] + int(scale * self.image_roi.size[1] + .5)
            )
            self.extent_spots = tuple(int(p / scale + .5) for p in self.extent)

    def get_photo_ROI(self, match_pxls=True, plts=False):
        assert hasattr(self, 'extent_spots'), 'call set_extent_data first'
        # get extent of data points in txt-file
        x_min_FT, x_max_FT, y_min_FT, y_max_FT = self.extent_spots
        x_min_area, x_max_area, y_min_area, y_max_area = self.extent

        # resize region in photo to match data points
        if match_pxls:
            img_resized = self.image_roi
        else:
            img_resized = self.image.crop(
                (x_min_area, y_min_area, x_max_area, y_max_area))

        self.photo_ROI_xywh = self.extent  # photo units
        self.data_ROI_xywh = self.extent_spots  # data units
        self.image_roi = img_resized
        return self.image_roi


class ProjectBaseClass:
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
        # save age model to folder where project data originates from
        if path_file != self.path_d_folder:
            self.age_model.save(self.path_d_folder)

    def set_age_model(
            self,
            path_file: str | None = None,
            load: bool = True,
            **kwargs_read
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
        if hasattr(self, 'AgeModel_file'):
            self.age_model: AgeModel = AgeModel(self.path_d_folder)
        # if a file is provided, load it from the file provided
        elif load and (path_file is not None):
            self.load_age_model(path_file)
        # otherwise create new age model
        else:
            self.age_model: AgeModel = AgeModel(path_file, **kwargs_read)
            self.age_model.path_file = self.path_d_folder
            self.age_model.save()

    def set_depth_span(self, depth_span: tuple[float]) -> None:
        self.depth_span: tuple[float] = depth_span

    def set_age_span(
            self, depth_span: tuple | None = None, age_span: tuple | None = None
    ) -> None:
        """
        Either by specifying a depth span and using the age model or by directly
        specifying the age_span, set an age span (in years before 2000)."""

        assert hasattr(self, 'depth_span') or (depth_span is not None) or (age_span is not None), \
            'specify the depth in cm'
        assert hasattr(self, 'age_model') or (age_span is not None)
        if depth_span is None:
            depth_span = self.depth_span

        if age_span is not None:
            self.age_span = age_span
        else:
            self.age_span = tuple(self.age_model.depth_to_age(depth_span))

    def add_depth_column(self) -> None:
        """
        Map xs (from the data_obj.feature_table) to depths 
        (based on depth_span).
        """
        assert hasattr(self, 'data_obj'), f'did not set data_obj yet'
        assert hasattr(self, 'depth_span'), 'set the depth_span first'

        min_depth, max_depth = self.depth_span
        # convert seed pixel coordinate to depth and depth to age
        x = self.data_obj.feature_table.x.abs()
        # seed = 0 corresponds to min_depth
        # seed.max corresponds to max_depth (roughly)
        depths = rescale_values(
            x,
            new_min=min_depth,
            new_max=max_depth,
            old_min=x.min(),
            old_max=x.max()
        )
        self.data_obj.feature_table['depth'] = depths

    def add_age_column(self):
        """
        Map the depths in the data_obj.feature_table with the age model.

        Returns
        -------
        None.

        """
        assert hasattr(self, 'age_model'), 'set age model first'
        assert hasattr(self, 'data_obj'), f'did not set data_obj yet'
        hasattr(self.data_obj, 'feature_table') and \
            'depth' in self.data_obj.feature_table.columns, \
            'data object must have depth column'

        self.data_obj.feature_table['age'] = self.age_model.depth_to_age(
            self.data_obj.feature_table.depth
        )

    def set_image_sample(self, obj_color=None, **kwargs_area):
        assert hasattr(self, 'image_handler'), \
            'call set_image_handler first'
        self.image_sample = ImageSample(
            self.path_folder, self.image_handler.image,
            image_type='pil', obj_color=obj_color
        )
        if hasattr(self, 'ImageSample_file'):
            print('loading ImageSample')
            self.image_sample.load()
        if not hasattr(self.image_sample, 'xywh_ROI'):
            self.image_sample.sget_sample_area(**kwargs_area)
            self.image_sample.save()

    def set_image_roi(self, obj_color=None):
        self.image_roi = ImageROI(
            self.path_folder, obj_color=obj_color
        )
        if hasattr(self, 'ImageROI_file'):
            print('loading ImageROI')
            self.image_roi.load()
            return
        assert hasattr(self, 'age_span'), 'specify the age span first'
        if (obj_color is None) or \
                (not hasattr(self, 'image_sample')) or \
                (not hasattr(self.image_sample, 'obj_color')):
            obj_color = None
        else:
            obj_color = self.image_sample.obj_color

        self.image_roi.age_span = self.age_span
        if not hasattr(self.image_roi, 'image_classification'):
            self.image_roi.sget_classification_adaptive_mean()
        self.image_roi.save()

    def set_image_classified(
            self, obj_color=None, peak_prominence=.1, max_slope=.1,
            downscale_factor=1 / 16
    ):
        if obj_color is not None:
            pass
        elif hasattr(self, 'image_sample') and hasattr(self.image_sample, 'obj_color'):
            obj_color = self.image_sample.obj_color
        elif hasattr(self, 'image_roi') and hasattr(self.image_roi, 'obj_color'):
            obj_color = self.image_roi.obj_color
        self.image_classified = ImageClassified(
            self.path_folder, obj_color=obj_color
        )
        if hasattr(self, 'ImageClassified_file'):
            print('loading ImageClassified ...')
            self.image_classified.load()
        self.image_classified.age_span = self.age_span
        if not hasattr(self.image_classified, 'params_laminae_simplified'):
            print('setting laminae in ImageClassified ...')
            self.image_classified.set_laminae_params_table(
                peak_prominence=peak_prominence,
                max_slope=max_slope,
                downscale_factor=downscale_factor
            )
            self.image_classified.save()

    def add_pixels_ROI(self):
        assert hasattr(self, 'image_sample'), 'call set_image_sample first'
        assert hasattr(self, 'image_handler'), 'call set_image_handler'

        if not hasattr(self.image_handler, 'image_roi'):
            self.image_handler.get_photo_ROI()

        image_ROI_xywh = self.image_sample.sget_sample_area()[1]
        data_ROI_xywh = self.image_handler.data_ROI_xywh
        photo_ROI_xywh = self.image_handler.photo_ROI_xywh
        self.data_obj.pixels_get_photo_ROI_to_ROI(
            data_ROI_xywh, photo_ROI_xywh, image_ROI_xywh
        )

    def add_photo(self):
        assert hasattr(self, 'data_obj'), 'set data object first'
        assert 'x_ROI' in self.data_obj.feature_table.columns, \
            'add x_ROI, y_ROI coords with add_pixels_ROI'
        image = ensure_image_is_gray(
            self.image_sample.sget_sample_area()[0]
        )
        self.data_obj.add_attribute_from_image(image, 'L', median=False)

    def add_holes(self):
        assert hasattr(self, f'image_roi')
        image = self.image_roi.get_foreground_thr_and_pixels(self.image_roi.sget_image_grayscale())[1]
        self.data_obj.add_attribute_from_image(image, 'valid', median=False)

    def add_light_dark_classification(self):
        assert hasattr(self, f'image_roi'), f'call set_image_roi'
        image = self.image_roi.image_classification
        self.data_obj.add_attribute_from_image(image, 'classification')

    def add_laminae_classification(self):
        assert hasattr(self, 'image_classified'), 'call set_image_classified'
        image = self.image_classified.image_seeds
        image_e = self.image_classified.get_image_expanded_laminae()
        self.data_obj.add_attribute_from_image(image, 'classification_s')
        self.data_obj.add_attribute_from_image(image_e, 'classification_se')

    def set_xray(
            self, path_image_file,
            depth_section=None, obj_color='dark', **kwargs
    ):
        self.xray = XRay(
            path_image_file, depth_section, obj_color=obj_color, **kwargs
        )
        # try to load from disc
        folder = os.path.dirname(path_image_file)
        name = 'XRay.pickle'
        if os.path.exists(os.path.join(folder, name)):
            self.xray.load()
            if hasattr(self.xray, 'image_ROI'):
                return
        self.xray.sget_sample_area()
        self.xray.remove_bars()
        self.xray.save()

    def set_punch_holes(self, side: str, plts: bool = False, **kwargs):
        """
        Identify squre-shaped holes at top or bottom of sample in xray section
        and MSI sample.

        Parameters
        ----------
        side : str, optional
            The side on which the holes are expected to be found, either 'top'
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
        assert hasattr(self, 'xray'), 'call set_xray first'
        assert hasattr(self, 'image_roi'), 'call set_image_roi first'

        img_xray: ImageROI = self.xray.get_ImageROI_from_section(self.depth_span)
        img_xray.set_punchholes(remove_gelatine=False, side=side, **kwargs)
        if not hasattr(self.image_roi, 'punchholes'):
            self.image_roi.set_punchholes(remove_gelatine=True, side=side, **kwargs)

        # copy over to object attributes
        self.holes_xray: list[np.ndarray[int]] = img_xray.punchholes
        self.holes_data: list[np.ndarray[int]] = self.image_roi.punchholes
        self.hole_side: str = side

    def set_depth_correction_with_xray(self, method='linear'):
        assert method in ('linear', 'cubic', 'piece-wise linear', 'l', 'c', 'pwl')
        assert hasattr(self, 'holes_data'), 'call set_punch_holes'
        assert hasattr(self, 'data_obj'), 'set data_obj object first'
        assert 'depth' in self.data_obj.feature_table.columns, 'set depth column'
        assert hasattr(self, 'depth_span'), 'set the depth section'

        def idx_to_depth(idx, img_shape):
            """Convert the index into the relative depth."""
            return idx / img_shape[1] * depth_section + self.depth_span[0]

        def append_top_bottom(arr: np.ndarray, img_shape):
            # insert top
            arr = np.insert(arr, 0, 0)
            # append bottom
            arr = np.append(arr, img_shape[1])
            return arr

        depth_section = self.depth_span[1] - self.depth_span[0]

        img_xray_shape = self.xray.get_section(
            self.depth_span[0], self.depth_span[1]
        ).shape
        # eliminate gelatine pixels
        img_data_shape = self.image_roi.sget_image_grayscale().shape

        # use holes as tie-points
        idxs_xray = np.array([point[1] for point in self.holes_xray])
        idxs_data = np.array([point[1] for point in self.holes_data])

        depths = self.data_obj.feature_table['depth']

        if method not in ('linear', 'l'):
            idxs_xray = append_top_bottom(idxs_xray, img_xray_shape)
            idxs_data = append_top_bottom(idxs_data, img_data_shape)

        # depths xray --> assumed to be not deformed, therefore linear depth increase
        if method in ('linear', 'l'):
            # linear function to transform xray depth to msi depth
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
                mask = (depths >= depth_interval[0]) &\
                    (depths <= depth_interval[1])
                depths_new[mask] = pol(depths[mask])
        else:
            raise NotImplementedError()

        self.data_obj.feature_table['depth_corrected'] = depths_new

    def add_xray(self, plts=False, is_piecewise=True, **kwargs):
        assert hasattr(self, 'data_obj'), 'set data_obj object first'
        assert 'x_ROI' in self.data_obj.feature_table.columns, 'call add_pixels_ROI'
        assert hasattr(self, 'xray'), 'call set_xray first'
        assert hasattr(self, 'image_roi'), 'call set_image_roi first'
        assert hasattr(self, 'depth_span'), 'set the depth section'
        assert hasattr(self, 'holes_data'), 'call set_depth_correction_with_xray'

        # no longer necessary since corners are used
        # def find_top_bottom_intersects(contour):
        #     # convert to polar
        #     contour_f = contour[:, 0, :].copy().astype(float)
        #     center = contour_f.mean(axis=0)  # x, y
        #     contour_f -= center  # shift to center
        #     contour_phi = np.arctan2(contour_f[:, 1], contour_f[:, 0])  # rad

        #     # find points closest to 0 and pi
        #     idx_lower = np.argmin(np.abs(contour_phi - np.pi))
        #     # express 3rd and 4th quadrant as negative angles
        #     contour_phi[contour_phi > np.pi] = \
        #         2 * np.pi - contour_phi[contour_phi > np.pi]
        #     idx_upper = np.argmin(np.abs(contour_phi - 0))
        #     # get corresponding points
        #     point_lower = contour[:, 0, :][idx_lower][::-1]
        #     point_upper = contour[:, 0, :][idx_upper][::-1]
        #     return [point_lower, point_upper]

        img_xray: np.ndarray[int] = ensure_image_is_gray(
            self.xray.get_section(self.depth_span[0], self.depth_span[1])
        )
        IR: ImageROI = ImageROI(
            image=img_xray, path_folder=None, obj_color=self.xray.obj_color
        )
        img_xray_simplified: np.ndarray[int] = IR.sget_simplified_image()
        # eliminate gelatine pixels
        img_data: np.ndarray[int] = self.image_roi.sget_simplified_image() \
            * self.image_roi.sget_foreground_thr_and_pixels()[1]

        points_xray: list[np.ndarray[int]] = self.holes_xray.copy()

        points_data: list[np.ndarray[int]] = self.holes_data.copy()

        cont_data: np.ndarray[int] = self.image_roi.get_main_contour(
            self.image_roi.sget_simplified_image()
        )
        cont_xray: np.ndarray[int] = self.xray.get_main_contour(
            img_xray_simplified
        )

        t: Transformation = Transformation(img_xray, self.image_roi)
        t._transform_from_punchholes(
            is_piecewise=is_piecewise,
            hole_side=self.hole_side,
            points_source=points_xray,
            points_target=points_data,
            contour_source=cont_xray,
            contour_target=cont_data
        )

        warped_xray: np.ndarray[float] = t.fit()

        warped_xray: np.ndarray[int] = rescale_values(
            warped_xray, new_min=0, new_max=255
        ).astype(int)
        # add to feature table
        self.data_obj.add_attribute_from_image(warped_xray, 'xray')

        if plts:
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

            img_roi = self.image_roi.sget_image_grayscale().astype(float)
            img_roi *= self.image_roi.sget_foreground_thr_and_pixels()[1]
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

    def set_time_series(
            self,
            average_by_col='classification_s',
            plts=False,
            **kwargs
    ) -> None:
        self.time_series = TimeSeries(self.path_folder)
        if hasattr(self, 'TimeSeries_file'):
            self.time_series.load()
            if hasattr(self.time_series, 'feature_table'):
                return self.time_series

        assert hasattr(self, 'data_obj'), 'call set_msi_object'
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

        assert hasattr(self, 'image_classified'), 'call set_image_classified'
        assert hasattr(self.image_classified, 'params_laminae_simplified'), \
            'could not find params table in classified image object, call set_image_classified'

        # x column always included in processing_zone_wise_average
        columns_feature_table = np.append(
            self.data_obj.get_data_columns(),
            ['x_ROI', 'R', 'L', 'depth', 'age']
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
        quals = self.image_classified.params_laminae_simplified.loc[
            :, ['seed', 'color'] + cols_quals
        ].copy()

        # sign the seeds in the quality dataframe
        mask_c = quals['color'] == 'light'
        quals.loc[mask_c, 'color'] = 1
        quals.loc[~mask_c, 'color'] = -1
        quals['seed'] *= quals['color']
        quals = quals.drop(columns='color')
        quals['seed'] = quals.seed.astype(int)
        quals = quals.set_index('seed')
        # join the qualities to the averages table
        ft_seeds_avg = ft_seeds_avg.join(quals, how='left')
        # insert infty for every column in success table that is not there yet
        missing_cols = set(ft_seeds_avg.columns).difference(
            set(ft_seeds_success.columns)
        )
        for col in missing_cols:
            ft_seeds_success[col] = np.infty

        # plot the qualities
        if plts:
            plt.figure()
            plt.plot(quals.index, quals.quality, '+', label='qual')
            plt.plot(ft_seeds_avg.index, ft_seeds_avg.quality, 'x', label='ft')
            plt.legend()
            plt.xlabel('seed')
            plt.ylabel('quality')
            plt.title('every x should have a +')
            plt.show()

        # need to insert the x_ROI from avg
        ft_seeds_std['spread_x_ROI'] = ft_seeds_std.x_ROI.copy()
        ft_seeds_success['N_total'] = ft_seeds_success.x_ROI.copy()
        ft_seeds_std['x_ROI'] = ft_seeds_avg.x_ROI.copy()
        ft_seeds_success['x_ROI'] = ft_seeds_avg.x_ROI.copy()
        # sort by depth
        ft_seeds_avg = ft_seeds_avg.sort_values(by='x_ROI')
        ft_seeds_std = ft_seeds_std.sort_values(by='x_ROI')
        ft_seeds_success = ft_seeds_success.sort_values(by='x_ROI')
        # drop index (=seed) into dataframe
        ft_seeds_avg.index.names = ['seed']
        ft_seeds_std.index.names = ['seed']
        ft_seeds_success.index.names = ['seed']
        # reset index
        ft_seeds_avg.reset_index(inplace=True)
        ft_seeds_avg['seed'] = ft_seeds_avg.seed.astype(int)
        ft_seeds_std.reset_index(inplace=True)
        ft_seeds_std['seed'] = ft_seeds_std.seed.astype(int)
        ft_seeds_success.reset_index(inplace=True)
        ft_seeds_success['seed'] = ft_seeds_success.seed.astype(int)

        self.time_series.feature_table = ft_seeds_avg
        self.time_series.feature_table_standard_deviations = ft_seeds_std
        self.time_series.feature_table_successes = ft_seeds_success

        if hasattr(self, 'age_span'):
            self.time_series.set_age_scale(self.time_series.feature_table.age.to_numpy())

        self.time_series.save()


class ProjectMSI(ProjectBaseClass):
    def __init__(self, path_folder, depth_span: tuple[int] = None, d_folder: str | None = None):
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
        file_d_folder : str, optional


        Returns
        -------
        None.

        """
        self.path_folder = path_folder
        if depth_span is not None:
            self.depth_span = depth_span

        self._set_files(d_folder)

    def _set_files(self, d_folder: str | None = None):
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
                f'Found multiple d folders {d_folders}, please specify the name of the file by providing the d_folder keyword upon initalization.'
            d_folder = d_folders[0]
            name_mis_file = d_folder.split('.')[0] + '.mis'
        else:
            name_mis_file = None
        dict_files: dict[str, str] = {
            'd_folder': d_folder,
            'mis_file': get_mis_file(self.path_folder, name_file=name_mis_file)
        }

        if dict_files.get('d_folder') is not None:
            self.path_d_folder = os.path.join(self.path_folder, dict_files['d_folder'])
        else:
            raise FileNotFoundError(f'Found no d folder in {self.path_folder}')

        if dict_files.get('mis_file') is not None:
            self.path_mis_file = os.path.join(self.path_folder, dict_files['mis_file'])
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
            entry['name'] == d_folder
            for entry in folder_structure['children']
        ])
        assert len(idxs) == 1, 'found no or conflicting files, check folder'
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

    def set_image_handler(self):
        """
        Initialize and SampleImageHandlerMSI object and set the photo.

        Returns
        -------
        None.

        """
        self.image_handler = SampleImageHandlerMSI(path_folder=self.path_folder, path_d_folder=self.path_d_folder)
        if hasattr(self, 'SampleImageHandlerMSI_file'):
            self.image_handler.load()
        if not hasattr(self.image_handler, 'extent_spots'):
            self.image_handler.set_extent_data()
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

        if hasattr(self, 'Spectra_file') and not overwrite:
            print('found hdf5 file, not creating new file because overwrite is set to false')
            return handler

        if reader is None:
            reader = self.get_mcf_reader()

        handler.write(reader, **kwargs)

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

    def set_spectra(self, reader: ReadBrukerMCF | hdf5Handler = None, full=True, plts=False):
        # create spectra object
        if hasattr(self, 'spectra_object_file'):
            self.spectra = Spectra(
                path_d_folder=self.path_d_folder, initiate=False
            )
            self.spectra.load()
            if hasattr(self.spectra, 'feature_table'):
                return
            elif hasattr(self.spectra, 'line_spectra'):
                self.spectra.binned_spectra_to_df(reader=self.get_reader())
                self.spectra.save()
                return
            elif reader is None:
                reader = self.get_reader()
        else:
            if reader is None:
                reader = self.get_reader()
            self.spectra = Spectra(reader=reader)
        if not full:
            return

        if not np.any(self.spectra.intensities.astype(bool)):
            self.spectra.add_all_spectra(reader)
            self.spectra.subtract_baseline(plts=plts)
        if not hasattr(self.spectra, 'peaks'):
            self.spectra.set_peaks()
        if not hasattr(self.spectra, 'kernel_params'):
            self.spectra.set_kernels()
        if not hasattr(self.spectra, 'line_spectra'):
            self.spectra.bin_spectra(reader)
        if not hasattr(self.spectra, 'feature_table'):
            self.spectra.binned_spectra_to_df(reader)

        if plts:
            self.spectra.plt_summed(plt_kernels=True)
            img = self.spectra.feature_table.pivot(
                index='x',
                columns='y',
                values=self.spectra.feature_table.columns[0]
            )
            plt.figure()
            plt.imshow(img)
            plt.show()

        self.spectra.save()

    def set_object(self, overwrite: bool = False):
        self.data_obj: MSI = MSI(self.path_d_folder)
        if hasattr(self, 'MSI_file') and not overwrite:
            self.data_obj.load()
            if hasattr(self.data_obj, 'feature_table'):
                return

        assert hasattr(self, 'spectra') and hasattr(self.spectra, 'feature_table'), \
            'set spectra object first'

        self.data_obj.set_distance_pixels()
        self.data_obj.set_feature_table_from_spectra(self.spectra)

    def get_proxy(self, mz_a: float | str, mz_b: float | str, **kwargs):
        assert hasattr(self, 'time_series'), 'set the time series object first'

        proxy: RatioProxy = RatioProxy(self.time_series, mz_a, mz_b, **kwargs)

        return proxy

    def set_UK37(
            self, correction_factor=1,
            method_SST='BAYSPLINE', prior_std_bayspline=10, **kwargs
    ):
        assert hasattr(self, 'time_series')

        self.UK37_proxy = UK37(self.time_series, **kwargs)
        self.UK37_proxy.add_UK_proxy(correction_factor=correction_factor)
        self.UK37_proxy.add_SST(method=method_SST, prior_std=prior_std_bayspline)


class ProjectXRF(ProjectBaseClass):
    def __init__(
            self,
            path_folder: str,
            path_bcf_file: str | None = None,
            path_image_file: str | None = None,
            path_image_roi_file: str | None = None,
            measurement_name: str | None = None
    ):
        self.path_folder: str = path_folder

        if measurement_name is not None:
            self.measurement_name = measurement_name
        else:
            self._set_measurement_name()
        self._set_files(
            path_bcf_file, path_image_file, path_image_roi_file
        )

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
        """Try to find files and infere measurement name."""
        files: list[str] = os.listdir(self.path_folder)

        if path_bcf_file is None:
            path_bcf_file = os.path.join(self.path_folder, find_matches(
                substrings=self.measurement_name,
                files=files,
                file_types='bcf'
            ))
        if path_image_file is None:
            path_image_file = os.path.join(self.path_folder, find_matches(
                substrings='Mosaic',
                files=files,
                file_types=['tif', 'bmp', 'png', 'jpg'],
                must_include_substrings=True
            ))
        if path_image_roi_file is None:
            path_image_roi_file = os.path.join(self.path_folder, find_matches(
                substrings='Video 1',
                files=files,
                file_types='txt',
                must_include_substrings=True,
            ))

        self.path_bcf_file: str = path_bcf_file
        self.path_image_file: str = path_image_file
        self.path_image_roi_file: str = path_image_roi_file

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

    def set_image_handler(self):
        """
        Initialize and SampleImageHandlerXRF object and set the photo.

        Returns
        -------
        None.

        """
        assert hasattr(self, 'path_image_file') and hasattr(self, 'path_image_roi_file'), \
            'ensure the image files have good names (matching measurement name)'
        self.image_handler: SampleImageHandlerXRF = SampleImageHandlerXRF(
            path_folder=self.path_folder,
            path_image_file=self.path_image_file,
            path_image_roi_file=self.path_image_roi_file
        )
        if hasattr(self, 'SampleImageHandlerXRF_file'):
            self.image_handler.load()
            self.image_handler.set_photo()
        if not hasattr(self.image_handler, 'extent_spots'):
            self.image_handler.set_photo()
            self.image_handler.set_extent_data()
            self.image_handler.save()

    def set_object(self, **kwargs):
        self.data_obj = XRF(
            path_folder=self.path_folder,
            measurement_name=self.measurement_name,
            **kwargs
        )
        if hasattr(self, 'XRF_file'):
            self.data_obj.load()
            if hasattr(self.data_obj, 'feature_table'):
                print('loaded data object')
                return
        self.data_obj.set_feature_table_from_txts()


def Project(is_MSI: bool, *args, **kwargs) -> ProjectMSI | ProjectXRF:
    if is_MSI:
        return ProjectMSI(*args, **kwargs)
    return ProjectXRF(*args, **kwargs)


class MultiMeasurementProject:
    def __init__(self, *projects: Iterable[ProjectMSI | ProjectXRF]):
        pass


    def combine_msi_xrf(self):
        assert hasattr(self, 'msi') and hasattr(self, 'xrf'), \
            'must have msi and xrf object to combine them'


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
        self.data_obj = MultiSectionData(*folders)
        if hasattr(self.spectra, 'line_spectra') and not hasattr(self.spectra, 'feature_table'):
            self.spectra.feature_table = self.spectra.binned_spectra_to_df()
        if hasattr(self.spectra, 'feature_table'):
            self.data_obj.feature_table = self.spectra.feature_table
        else:
            # TODO: this
            ...
            raise NotImplementedError()


