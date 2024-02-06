from util.manage_obj_saves import class_to_attributes
from imaging.util.coordinate_transformations import rescale_values

from exporting_mcf.rtms_communicator import ReadBrukerMCF, Spectra
from data.cMSI import MSI
from data.file_helpers import (get_folder_structure, find_files, get_mis_file, 
                               get_d_folder, search_keys_in_xml, get_image_file)
from data.cAgeModel import AgeModel
from imaging.main.cImage import ImageSample, ImageROI, ImageClassified
from imaging.util.Image_convert_types import convert, ensure_image_is_gray

import os
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt

from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw
PIL_Image.MAX_IMAGE_PIXELS = None

    
class SampleImageHandler:
    """Given the mis file and folder, find image and area (of MSI) of sample."""
    def __init__(self, path_folder):
        self.path_folder = path_folder
        self.path_mis_file = os.path.join(self.path_folder, get_mis_file(self.path_folder))
        self.path_d_folder = os.path.join(self.path_folder, get_d_folder(self.path_folder))
        image_file = get_image_file(self.path_folder)
        self.path_image_sample = os.path.join(
            self.path_folder, image_file
        )
        
    def load(self):
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        path_d_folder = os.path.join(
            self.path_folder, 
            get_d_folder(self.path_folder)
        )
        with open(os.path.join(path_d_folder, name), 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__
        
    def save(self):
        # delete all attributes that are not flagged as relevant
        dict_backup = self.__dict__.copy()
        keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
        existent_attributes = list(self.__dict__.keys())
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        path_d_folder = os.path.join(
            self.path_folder, get_d_folder(self.path_folder)
        )
        with open(os.path.join(path_d_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup
        
    def set_photo(self):
        if not hasattr(self, 'path_image_sample'):
            self.set_path_photo()
        self.image = PIL_Image.open(self.path_image_sample)
    
    def set_extent_data(self, reader: ReadBrukerMCF = None):
        """Get spot names from MCFREader and set extent of pixels"""
        if reader is None:
            reader = ReadBrukerMCF(self.path_d_folder)
            reader.create_reader()
        if not hasattr(reader, 'spots'):
            reader.create_spots()
            
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
            match_pxls=True,
            plts=False
    ):
        """Match image and data pixels. Set extent pixels"""
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
        self.photo_ROI_sample = img_resized
        return self.photo_ROI_sample


class Project:
    def __init__(self, path_folder, depth_section: tuple[int] = None):
        """
        Initialization with folder.

        Parameters
        ----------
        path_folder : str
            path to folder with d-folder, mis file etc.
        depth_section : tuple[int], optional
            For core data, the depth section of the slice in cm. 
            The default is None. Certain features will not be available without 
            the depth section.

        Returns
        -------
        None.

        """
        self.path_folder = path_folder
        if depth_section is not None:
            self.depth_section = depth_section
        
        self._set_files()

    def _set_files(self):
        folder_structure = get_folder_structure(self.path_folder)
        dict_files = {}
        dict_files['d_folder'] = get_d_folder(self.path_folder)
        dict_files['mis_file'] = get_mis_file(self.path_folder)
        
        # try finding savefiles inside d-folder
        targets_d_folder = [
            'peaks.sqlite', 
            'spectra_object.pickle', 
            'MSI.pickle',
            'ImageSample.pickle',
            'ImageROI.pickle',
            'ImageClassified.pickle',
            'SampleImageHandler.pickle',
            'AgeModel.pickle'
        ]
        idxs = np.where([
            entry['name'].split('.')[-1] == 'd' 
            for entry in folder_structure['children']
        ])
        assert len(idxs) == 1, 'found no or conflicting files, check folder'
        idx = idxs[0][0]
        dict_files_dfolder = find_files(
            folder_structure['children'][idx],
            *targets_d_folder
        )
        
        for k, v in dict_files_dfolder.items():
            k_new = k.split('.')[0] + '_file'
            dict_files[k_new] = v
        
        self.__dict__ |= dict_files
        for name in dict_files:
            self.__setattr__(
                'path_' + name, os.path.join(self.path_folder, dict_files[name])
            )
            
    def set_image_handler(self):
        self.image_handler = SampleImageHandler(self.path_folder)
        if hasattr(self, 'SampleImageHandler_file'):
            self.image_handler.load()
        if not hasattr(self.image_handler, 'extent_spots'):
            self.image_handler.set_extent_data()
            self.image_handler.save()
        self.image_handler.set_photo()
        
    def load_age_model(self, path_file: str):
        self.age_model = AgeModel(path_file=path_file)
        if path_file != self.path_d_folder:
            self.age_model.save(self.path_d_folder)
        
    def set_age_model(self, path_file: str = None, load=True, **kwargs_read):
        if hasattr(self, 'AgeModel_file'):
            self.age_model = AgeModel(self.path_d_folder)
        elif load and (path_file is not None):
            self.load_age_model(path_file)
        else:
            self.age_model = AgeModel(path_file, **kwargs_read)
            self.age_model.path_file = self.path_d_folder
            self.age_model.save()
                    
    def set_image_sample(self, obj_color=None, **kwargs_area):
        assert hasattr(self, 'image_handler'), \
            'call set_image_handler first'
        self.image_sample = ImageSample(
            self.path_folder, self.image_handler.image, 
            image_type='pil', obj_color=obj_color
        )
        if hasattr(self, 'ImageSample_file'):
            self.image_sample.load()
        if not hasattr(self.image_sample, 'xywh_ROI'):
            self.image_sample.sget_sample_area(**kwargs_area)
            self.image_sample.save()
            
    def set_age_span(self, depth_section: tuple =  None):
        assert hasattr(self, 'depth_section'), 'specify the depth in cm'
        assert hasattr(self, 'age_model')
        
        self.age_span = tuple(self.age_model.depth_to_age(self.depth_section))
        
        
    def set_image_roi(self, obj_color=None):
        assert hasattr(self, 'age_span'), 'specify the age spa first'
        if (obj_color is None) or \
                (not hasattr(self, 'image_sample')) or \
                (not hasattr(self.image_sample, 'obj_color')):
            obj_color = None
        else:
            obj_color = self.image_sample.obj_color
        self.image_roi = ImageROI(
            self.path_folder, obj_color=obj_color
        )
        if hasattr(self, 'ImageROI_file'):
            self.image_roi.load()
        self.image_roi.age_span = self.age_span
        if not hasattr(self.image_roi, 'image_classification'):
            self.image_roi.sget_classification_adaptive_mean()
        self.image_roi.save()
        
    def set_image_classified(
            self, obj_color=None, peak_prominence=.1, max_slope=.1, 
            downscale_factor=1 / 16
    ):
        if (obj_color is not None):
            pass
        elif hasattr(self, 'image_sample') and hasattr(self.image_sample, 'obj_color'):
            obj_color = self.image_sample.obj_color
        elif hasattr(self, 'image_roi') and hasattr(self.image_roi, 'obj_color'):
            obj_color = self.image_roi.obj_color
        self.image_classified = ImageClassified(
            self.path_folder, obj_color=obj_color
        )
        self.image_classified.verbose=True
        self.image_classified.plts=True
        self.image_classified.age_span = self.age_span
        self.image_classified.set_laminae_params_table(
            peak_prominence=peak_prominence,
            max_slope=max_slope,
            downscale_factor=downscale_factor
        )
        # TODO: check image_classified behaves corretly with age_sapn etc
        self.image_classified.save()
        
    
    def set_spectra(self, reader: ReadBrukerMCF = None, plts=False):
        if hasattr(self, 'spectra_object_file'):
            self.spectra = Spectra(load=True, path_d_folder=self.path_d_folder)
            if hasattr(self.spectra, 'feature_table'):
                return self.spectra
        
        # create reader object
        if reader is None:
            reader = ReadBrukerMCF(self.path_d_folder)
        if not hasattr(reader, 'reader'):
            reader.create_reader()
        if not hasattr(reader, 'indices'):
            reader.create_indices()
            
        # create spectra object
        self.spectra = Spectra(reader=reader)
        if not np.any(self.spectra.intensities.astype(bool)):
            self.spectra.add_all_spectra(reader)
        if not hasattr(self.spectra, 'peaks'):
            self.spectra.set_peaks()
        if not hasattr(self.spectra, 'kernel_params'):
            self.spectra.set_kernels()
        
        if plts:
            self.spectra.plt_kernels()
        if not hasattr(self.spectra, 'line_spectra'):
            self.spectra.bin_spectra(reader)
        if not hasattr(self.spectra, 'feature_table'):
            self.spectra.binned_spectra_to_df(reader)

        if plts:
            self.spectra.plt_summed(plt_kernels=True)
            img = self.spectra.feature_table.pivot(
                index='x', 
                columns='y', 
                values = self.spectra.feature_table.columns[0]
            )
            plt.imshow(img)
            plt.show()

        self.spectra.save()
    
    def set_msi_object(self):
        assert hasattr(self, 'image_sample'), 'call set_image_object first'
        assert hasattr(self, 'image_handler'), 'call set_image_handler'
        assert hasattr(self, 'spectra') and hasattr(self.spectra, 'feature_table'), \
            'set spectra object first'
        if not hasattr(self.image_handler, 'photo_ROI_sample'):
            self.image_handler.get_photo_ROI()
        # should be x,x (distance in x, y in um)
        distance_t: tuple[str] = search_keys_in_xml(
            self.path_mis_file, ['Raster']
        )['Raster'].split(',')
        assert (d:= distance_t[0]) == distance_t[1], \
            'cant handle grid with different distances in x and y'
        distance_pixels = float(d)
        self.msi = MSI(self.path_d_folder, distance_pixels=distance_pixels)
        self.msi.feature_table = self.spectra.feature_table
        image_ROI_xywh = self.image_sample.sget_sample_area()[1]
        data_ROI_xywh = self.image_handler.data_ROI_xywh
        photo_ROI_xywh = self.image_handler.photo_ROI_xywh
        self.msi.pixels_get_photo_ROI_to_ROI(
            data_ROI_xywh, photo_ROI_xywh, image_ROI_xywh
        )
        self.msi.save()
        
    def add_photo_to_msi(self):
        assert hasattr(self, 'msi'), 'set msi object first'
        
        image = ensure_image_is_gray(
            self.image_sample.sget_sample_area()[0]
        )
        
        self.msi.add_attribute_from_image(image, 'L', median=False)
        
    def add_depth_column(self):
        """Map xs to depths."""
        assert hasattr(self, 'depth_section'), 'set the depth_section first'
        min_depth, max_depth = self.depth_section
        # convert seed pixel coordinate to depth and depth to age
        x = self.msi.feature_table.x.abs()
        # seed = 0 corresponds to min_depth
        # seed.max corresponds to max_depth (roughly)
        depths = rescale_values(x, new_min=min_depth, new_max=max_depth, old_min=x.min(), old_max=x.max())
        self.msi.feature_table['depth'] = depths
        
    def add_age_column(self):
        assert hasattr(self, 'age_model'), 'set age model first'
        assert hasattr(self, 'msi') and hasattr(self.msi, 'feature_table') and \
            'depth' in self.msi.feature_table.columns, 'msi object must have depth column'
        
        self.msi.feature_table['age'] = self.age_model.depth_to_age(
            self.msi.feature_table.depth
        )
        
    def add_holes_to_msi(self):
        assert hasattr(self, 'image_roi')
        image = self.image_roi.get_foreground_thr_and_pixels(self.image_roi.sget_image_grayscale())[1]
        self.msi.add_attribute_from_image(image, 'valid', median=False)
        
    def add_classification_to_msi(self):
        pass
    

        
            
