import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project.cProject import get_project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)  # get some info during runtime

folder = (r'C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\490-495cm'
          r'\2018_08_27 Cariaco 490-495 alkenones.i')
# folder =r'F:\535-540cm\2020_03_23_Cariaco_535-540cm_Alkenones.i'
path_age_model = (r'C:/Users/Yannick Zander/Promotion/Cariaco MSI 2024/Age '
                  r'Model/480-510/480_510_MSI_age_model_mm_yr.txt')
path_xray = (r'C:/Users/Yannick Zander/Promotion/Cariaco line scan Xray/Cariaco '
             r'Xray/sliced/MD_03_2621_480-510_sliced_1200dpi.tif')

folder_xrf = (r'C:\Users\Yannick Zander\Promotion\Cariaco line scan Xray\uXRF '
              r'slices\S0343c_490-495cm')

depth_span = (490, 495)

params_age_model = dict(
    path_age_model=path_age_model,
    depth_offset_age_model=4800,
    conversion_to_cm_age_model=1 / 10
)

SNR_threshold = 2

p_xrf = get_project(is_MSI=False, path_folder=folder_xrf)
p_xrf.set_depth_span(depth_span)
p_xrf.require_age_model(params_age_model['path_age_model'], sep='\t', index_col=False, load=False)
p_xrf.age_model.add_depth_offset(params_age_model['depth_offset_age_model'])
p_xrf.age_model.convert_depth_scale(params_age_model['conversion_to_cm_age_model'])  # convert mm to cm
p_xrf.set_depth_span(depth_span)
p_xrf.set_age_span()

p_xrf.set_image_handler(overwrite=False)
p_xrf.set_image_sample(overwrite=False)
# p_xrf.image_sample.plot_overview()
p_xrf.set_image_roi()
p_xrf.set_image_classified(overwrite=False, plts=False)
p_xrf.set_data_object()
p_xrf.add_pixels_ROI()
p_xrf.add_photo()
p_xrf.add_holes()
p_xrf.data_obj_apply_tilt_correction()

p = get_project(is_MSI=True, path_folder=folder)
# reader = p.get_reader()
p.set_spectra(full=False)

p.set_depth_span(depth_span)
p.require_age_model(params_age_model['path_age_model'], sep='\t', index_col=False, load=False)
p.age_model.add_depth_offset(params_age_model['depth_offset_age_model'])
p.age_model.convert_depth_scale(params_age_model['conversion_to_cm_age_model'])  # convert mm to cm
p.set_depth_span(depth_span)
p.set_age_span()

p.set_image_handler(overwrite=False)
p.set_image_sample(overwrite=False)
# p.image_sample.plot_overview()
p.set_image_roi()
p.set_image_classified(overwrite=False, plts=False)

p.set_data_object()
p.add_pixels_ROI()
p.add_photo()
p.add_holes()

p.add_xrf(p_xrf, plts=True)

p.plot_comp('Fe', 'data_obj')
