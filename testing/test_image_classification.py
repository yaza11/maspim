import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import random_walker
from skimage.transform import rescale
from skimage.color import rgb2gray

from imaging.main.cImage import ImageClassified
from imaging.util.Image_plotting import plt_cv2_image

path_folder = 'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i'

IC = ImageClassified(path_folder)

# IC.set_seeds()

downscale = 1 / 32

img = IC.sget_image_original()
img = rescale(img, [downscale, downscale, 1])
img_foreground = rescale(IC.sget_mask_foreground(), downscale)
# %%
plt_cv2_image(img)

img_gray = rgb2gray(img) 

labels = np.zeros_like(img_gray, dtype=int)

labels[img_gray >= np.percentile(img_gray[img_foreground], 95)] = 255

labels[img_gray <= np.percentile(img_gray[img_foreground], 5)] = 127

labels[~img_foreground] = -1

plt_cv2_image(labels)

img_seg = random_walker(img_gray, labels)
plt_cv2_image(img_seg, title='segmentation')
# plt.imshow(IC.get_image_expanded_laminae())

