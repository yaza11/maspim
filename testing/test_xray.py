import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.XRay.cXRay import XRay
from imaging.util.Image_plotting import plt_cv2_image
from imaging.main.cImage import ImageSample
from imaging.misc.find_punch_holes import find_holes

import matplotlib.pyplot as plt
import cv2
import numpy as np

path_xray = r'D:/Cariaco line scan Xray/Cariaco Xray/sliced/MD_03_2621_480-510_sliced_1200dpi.tif'

xray = XRay(path_xray, depth_section=(480, 510))
xray.sget_sample_area()
xray.plts=True
xray.remove_bars()
xray.save()

xray = XRay(path_xray)
xray.load()
plt_cv2_image(xray.sget_sample_area()[0])

# %%

s_xray = xray.get_section(section_upper=490)
find_holes(s_xray, 'bottom', obj_color='dark', width_sector=.4, plts=True)    
    
# %%

path_image_folder = r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i'

IS = ImageSample(path_image_folder)
IS.load()

find_holes(IS.sget_sample_area()[0], plts=True, side='bottom', obj_color='light')

