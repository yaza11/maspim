import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cProject import SampleImageHandlerXRF, Project
from data.cXRF import XRF

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

# handler = SampleImageHandlerXRF(
#     path_image_file='D:/Cariaco line scan Xray/uXRF slices/S0343c_490-495cm/PS343c 490-495cm Mosaic.bmp',
#     path_image_roi_file='D:/Cariaco line scan Xray/uXRF slices/S0343c_490-495cm/PS343 ctss _Video 1.txt'
# )

# handler.set_photo()
# handler.set_extent_data()

# img = handler.get_photo_ROI()
folder_xrf : str = r'D:\Cariaco line scan Xray\uXRF slices\S0343c_490-495cm'

pxrf = Project(is_MSI=False, path_folder=folder_xrf)

pxrf.set_image_handler()
pxrf.set_object()
pxrf.set_image_sample()
pxrf.set_image_roi()
# pxrf.data_obj.save()

# xrf: XRF = XRF(folder)
# print(xrf.get_element_txts()[1])
# xrf.set_feature_table_from_txts(tag='PS343 ctss ')

# xrf.analyzing_NMF(k=3)
# xrf.plt_NMF(3)



