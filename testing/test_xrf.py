import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

logging.basicConfig(level=logging.INFO)

from Project.cProject import get_project

logging.info("Starting script")

# handler = SampleImageHandlerXRF(
#     path_image_file='D:/Cariaco line scan Xray/uXRF slices/S0343c_490-495cm/PS343c 490-495cm Mosaic.bmp',
#     path_image_roi_file='D:/Cariaco line scan Xray/uXRF slices/S0343c_490-495cm/PS343 ctss _Video 1.txt'
# )

# handler.set_photo()
# handler.set_extent_data()

# img = handler.set_photo_ROI()
folder_xrf: str = r'D:\Cariaco line scan Xray\uXRF slices\S0343c_490-495cm'

pxrf = get_project(is_MSI=False, path_folder=folder_xrf)
pxrf.set_depth_span((490, 495))

pxrf.set_image_handler()

pxrf.set_age_model(path_file=r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\Age Model')
pxrf.set_age_span()

pxrf.set_image_sample()
pxrf.set_image_roi()
pxrf.set_image_classified()

pxrf.set_object()
pxrf.add_pixels_ROI()
pxrf.add_laminae_classification()
pxrf.add_depth_column()
pxrf.add_age_column()

pxrf.set_time_series()
# pxrf.data_obj.save()

# xrf: XRF = XRF(folder)
# print(xrf._get_element_txts()[1])
# xrf.set_feature_table_from_txts(tag='PS343 ctss ')

# xrf.analyzing_NMF(k=3)
# xrf.plt_NMF(3)
