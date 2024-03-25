import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.XRay.cXRay import XRay
from imaging.util.Image_plotting import plt_cv2_image
from imaging.main.cImage import ImageSample, ImageROI
from imaging.misc.find_punch_holes import find_holes
from imaging.register.main import Transformation
from data.cMSI import MSI


import matplotlib.pyplot as plt
import cv2
import numpy as np

path_xray = r'D:/Cariaco line scan Xray/Cariaco Xray/sliced/MD_03_2621_480-510_sliced_1200dpi.tif'
path_image_folder = r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i'
# path_image_folder = r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 FA.i'
# path_image_folder = r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 GDGT.i'
# path_image_folder = r'D:\Cariaco Data for Weimin\495-500cm\2018_08_28 Cariaco 495-500 alkenones.i' 
# path_image_folder = r'D:\Cariaco Data for Weimin\500-505cm\2018_08_29 Cariaco 500-505 GDGT.i'
# path_image_folder = r'D:\Cariaco Data for Weimin\505-510cm\2018_08_31 Cariaco 505-510 GDGT.i'  # <-- too much missing material
# path_image_file_SBB = r'C:/Users/yanni/Downloads/example_SBB/MV0811-14TC_0-5_A127_0001.tif'
# path_SBB = r'C:\Users\yanni\Downloads\example_SBB'

# xray = XRay(path_xray, depth_section=(480, 510))
# xray.sget_sample_area()
# xray.plts=True
# xray.remove_bars()
# xray.save()

xray = XRay(path_xray)
xray.load()
plt_cv2_image(xray.sget_sample_area()[0])
img_xray = xray.get_section(section_start=490, section_end=495)

IS = ImageSample(path_image_folder)
# IS.load()
img_msi = IS.sget_sample_area()[0]
# IS.save()
IR = ImageROI(image=img_msi, path_folder=path_image_folder, obj_color='light')

image = IR.sget_simplified_image() * IR.sget_foreground_thr_and_pixels()[1]

holes_xray = find_holes(img_xray, 'bottom', obj_color='dark',plts=True)    
holes_msi = find_holes(image, plts=True, side='bottom', obj_color='light')

