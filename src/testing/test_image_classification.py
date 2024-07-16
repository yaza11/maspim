import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import random_walker, slic
from skimage.transform import rescale
from skimage.color import rgb2gray

from imaging.main.cImage import ImageClassified, ImageROI
from imaging.util.Image_plotting import plt_cv2_image
from imaging.util.Image_processing import adaptive_mean_with_mask

from res.constants import key_light_pixels, key_dark_pixels

import cv2

def get_classification_varying_kernel_size(self, scaling=2) -> np.ndarray[int]:
    image: np.ndarray[int] = self.image_grayscale()
    height, width = image.shape[:2]
    
    if self.obj_color() == 'dark':
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY
    
    i_max = int(np.emath.logn(scaling, height))
    heights = height / scaling ** np.arange(i_max)
    # get last index where there are more than sixteen vertical pixels
    i_max = np.arange(i_max)[heights > 16][-1]
    
    res: np.ndarray[bool] = np.zeros((height, width, i_max), dtype=bool)
    
    mask: np.ndarray[np.uint8] = self._require_foreground_thr_and_pixels()[1]
    
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
    image_dark = (
        mask & (~res)).astype(np.uint8)
    image_classification = image_light * key_light_pixels + \
        image_dark * key_dark_pixels
        
    return image_classification


# path_folder = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i'
path_folder = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\505-510cm\2018_08_30 Cariaco 505-510 alkenones.i'

ir = ImageROI(path_folder)

# res = get_classification_varying_kernel_size(ir, scaling=2)
# ir.age_span = (0, 100)
# res2 = ir.get_classification_adaptive_mean(image_gray=ir.image_grayscale())[0]

# %%
# for i in range(res.shape[2]):
#     plt.figure()
#     plt.imshow(res[:, :, i], interpolation='none')
#     plt.show()
    
# %%
# plt.figure()
# plt.imshow(ir.image_grayscale())
# plt.show()
# plt.figure()
# plt.imshow(res, interpolation='none')
# plt.show()

# %%
mask_foreground = ir._require_foreground_thr_and_pixels()[1].astype(bool)
img = ir.image_grayscale()
imgc = img.copy().astype(float)
imgc[~mask_foreground] = np.nan
plt.imshow(imgc)
# res_c = res.copy().astype(float)
# res_c[~mask_foreground] = np.nan

# b = np.nanmean(res_c, axis=0)

# plt.imshow(res_c)
# plt.plot(b)
