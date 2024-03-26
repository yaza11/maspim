from imaging.main.cImage import ImageSample, ImageROI
from imaging.util.Image_convert_types import ensure_image_is_gray
from imaging.util.Image_plotting import plt_cv2_image

from util.manage_obj_saves import class_to_attributes

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

class XRay(ImageSample):
    def __init__(
            self, 
            path_image_file: str, 
            depth_section: tuple[float] = None,
            obj_color: str = 'dark'
    ):
        self.path_image_file  = path_image_file
        self.depth_section = depth_section
        
        self.verbose = False
        self.plts = False
        
        self._set_image()
        
        if obj_color is None:
            self.sget_obj_color()
        else:
            self.obj_color=obj_color
            
        self.set_current_image()
        
    def _set_image(self):
        self._image_original = cv2.imread(self.path_image_file)
        self.image_type = 'cv'
        h, w = self._image_original.shape [:2]
        if h > w:
            print('swapped axes of xray image to ensure horizontal orientation')
            self._image_original = self._image_original.swapaxes(0, 1)
        self._hw = self._image_original.shape[:2]
        
    def _section_args_to_tuple(
            self, 
            section_start: int | float | tuple[int | float], 
            section_end: int | float | None = None, 
            section_length: int | float = 5, 
    ) -> tuple[float | int]:
        if isinstance(section_start, tuple):
            section_start, section_end = section_start
        elif section_end is None:
            section_end = section_start + section_length
        assert section_start < section_end, 'first value should be the depth closer to the surface'

        return section_start, section_end

    def get_section(
            self, 
            section_start: int | float | tuple[int | float], 
            section_end: int | float | None = None, 
            section_length: int | float = 5, 
            plts: bool = False
    ) -> np.ndarray[int]:
        """
        Crop a depth section from the core image and return it.
        
        section_start can be a float or a tuple. If it is a tuple, 
        the first value will be used as start and the second as end depth. 
        Otherwise, if section_end is not specified, it will be infered from the section_length.
        """
        assert self.depth_section is not None, 'specify the depth section if not loaded'
        section_start, section_end = self._section_args_to_tuple(
            section_start, section_end, section_length
        )
        roi = self.sget_sample_area()[0]
        section_core = self.depth_section[1] - self.depth_section[0]
        # indices corresponding to depth
        w = self.xywh_ROI[2]
        mask = slice(
            int(w * (section_start - self.depth_section[0]) / section_core + .5),
            int(w * (section_end - self.depth_section[0]) / section_core + .5)
        )
        
        if plts:
            plt_cv2_image(roi, hold=True)
            w, h = self.xywh_ROI[2:]
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
            section_start: int | float | tuple[int | float], 
            section_end: int | float | None = None, 
            section_length: int | float = 5, 
            **kwargs
    ) -> ImageROI:
        depth_section: tuple[float | int] = self._section_args_to_tuple(
            section_start, section_end, section_length
        )
        img: np.ndarray[int] = self.get_section(depth_section, **kwargs)
        roi: ImageROI = ImageROI(
            path_folder=os.path.dirname(self.path_image_file), 
            image=img, 
            obj_color=self.obj_color
        )
        return roi
    
    def sget_sample_area(self, **kwargs):
        if hasattr(self, 'image_ROI'):
            return self.image_ROI, self.xywh_ROI
        
        self.image_ROI, self.xywh_ROI = self.get_sample_area(
            image=self.sget_image_original(), **kwargs
        )
        return self.image_ROI, self.xywh_ROI 
    
    def load(self):
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        path = os.path.dirname(self.path_image_file)
        with open(os.path.join(path, name), 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__
    
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
            print(f'saving xray object with {self.__dict__.keys()}')
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        path = os.path.dirname(self.path_image_file)
        with open(os.path.join(path, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup
    
    def remove_bars(self, n_sections = 10):
        def find_bounds(image_section):
            # average out in the depth-wise direction
            brightness_1d = np.mean(image_section, axis=1)
            n = len(brightness_1d)
            upper = brightness_1d[:n//2]
            lower = brightness_1d[n//2:]
            # discard everything outside center of dark bar
            upper_crop_idx = np.argmin(upper)
            lower_crop_idx = np.argmin(lower)
            upper_c = upper[upper_crop_idx:]
            lower_c = lower[:lower_crop_idx]
            # set boundary where signal starts to drop (~ point of maximum change)
            # can't rely on bright gap between casing and sediment
            upper_infl = np.argwhere(np.diff(upper_c) > 1)[-1][0]  # rising flank
            lower_infl = np.argwhere(np.diff(lower_c) < -1)[0][0]  # falling flank
            # shift back
            upper_infl += upper_crop_idx
            lower_infl += n // 2
            return upper_infl, lower_infl
            
        section_length = (self.depth_section[1] - self.depth_section[0]) / n_sections
        upper_bounds = np.zeros(n_sections, dtype=int)
        lower_bounds = np.zeros(n_sections, dtype=int)
        for i_section in range(n_sections):
            section_end = self.depth_section[0] + i_section * section_length
            image_section = ensure_image_is_gray(self.get_section(
                section_end, section_length=section_length
            ))
            u, l = find_bounds(image_section)
            upper_bounds[i_section] = u
            lower_bounds[i_section] = l
            
        # lin fit
        # center points of sections
        x_c = np.linspace(0, self.xywh_ROI[2], n_sections + 2, endpoint=True)[1:-1]
        xs = np.arange(0, self.xywh_ROI[2])
        upper_m, upper_b, *_ = linregress(x_c, upper_bounds)
        upper_bound = upper_b + upper_m * xs
        
        lower_m, lower_b, *_ = linregress(x_c, lower_bounds)
        lower_bound = lower_b + lower_m * xs
        
        if self.plts:
            plt.figure()
            plt.imshow(self.sget_sample_area()[0])
            plt.scatter(x_c, upper_bounds)
            plt.plot(xs, upper_bound)
            plt.scatter(x_c, lower_bounds)
            plt.plot(xs, lower_bound)
            plt.show()
            
        # set everything outside bounds to 255
        if self.obj_color == 'dark':
            fill_val = self.sget_sample_area()[0].max()
        else:
            fill_val = self.sget_sample_area()[0].min()
        _, Y = np.meshgrid(np.arange(0, self.xywh_ROI[2]), np.arange(0, self.xywh_ROI[3]))
        mask = (Y < upper_bound) | (Y > lower_bound) 
        self.image_ROI[mask] = fill_val
        # recast to fit ROI to new sample
        x, y, w, h = self.xywh_ROI
        upper_new = int(np.min(upper_bound))  # floor
        lower_new = int(np.max(lower_bound) + 1)  # ceil
        y_new = y + upper_new
        h_new = lower_new - upper_new
        self.xywh_ROI = (x, y_new, w, h_new)
        self.image_ROI = self.image_ROI[upper_new:lower_new, :]
        
        if self.plts:
            plt_cv2_image(self.image_ROI)
            
            
        
    
    
        
    
        