import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from data.cProject import Project, SampleImageHandler, get_d_folder
from imaging.main.cImage import ImageSample, ImageROI, ImageClassified
from exporting.from_mcf.rtms_communicator import Spectra, ReadBrukerMCF
from imaging.util.Image_convert_types import ensure_image_is_gray

path_folder = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i'
path_xray = r'D:/Cariaco line scan Xray/Cariaco Xray/sliced/MD_03_2621_480-510_sliced_1200dpi.tif'
# path_folder = "D:/Promotion/Test data"

# con = ReadBrukerMCF(get_d_folder(path_folder))

# s = Spectra()

# P.set_age_model(
#     path_file=r'G:/Meine Ablage/Master Thesis/AgeModel/480_510_MSI_age_model_mm_yr.txt',
#     sep='\t',
#     index_col=False,
#     load=False
# )

def test_all(path_folder, depth_span=(490, 495), obj_color='light'):
    P = Project(path_folder)
    
    # age model (required for ImageROI (choice of filter size) and to add age to MSI)
    P.set_age_model()
    P.set_depth_span(depth_span=depth_span)  # required for age_span and add_depth_to_msi
    P.set_age_span()
    
    # spectra (required for set_msi)
    P.set_spectra()  
    
    # images
    P.set_image_handler()  # required for adding photos
    P.set_image_sample(obj_color=obj_color)  # required for adding photo to msi
    P.set_image_roi()  # required for adding hole, light, dark information to msi
    P.set_image_classified()  # for adding laminae information to msi
    
    # msi
    P.set_msi_object()
    P.add_msi_pixels_ROI()
    P.add_photo_to_msi()
    P.add_holes_to_msi()
    P.add_depth_column()
    P.add_age_column()
    P.add_light_dark_classification_to_msi()
    P.add_laminae_classification_to_msi()
    
    # time series
    P.set_msi_time_series()
    
    # msi object is not saved by default
    # P.msi.save()
    # P.set_msi_object will try to load a saved msi object
    return P

def test_msi_minimal(path_folder):
    P = Project(path_folder)
    
    P.set_spectra()
    P.set_msi_object()
    return P

def test_proxy(path_folder):
    P = test_all(path_folder)
    P.set_UK37()
    
    P.UK37_proxy.plot()
    
def test_punch_holes(path_folder, path_xray, depth_xray=None, side='bottom', plts=False):
    P = test_all(path_folder)
    P.set_xray(path_xray, depth_xray)
    P.set_holes(plts=plts, side=side)
    
    P.add_xray_to_msi(plts=True)

    # test all methods    
    depth = P.msi.feature_table.depth.copy()
    P.set_msi_depth_correction_with_xray(method='l')
    depth_linear = P.msi.feature_table.depth_corrected.copy()

    P.set_msi_depth_correction_with_xray(method='c')
    depth_cubic = P.msi.feature_table.depth_corrected.copy()

    P.set_msi_depth_correction_with_xray(method='pwl')
    depth_pw = P.msi.feature_table.depth_corrected.copy()

    depth_n = depth.to_numpy()
    o = np.argsort(depth_n)
    if plts:
        plt.figure()
        plt.plot(depth, depth, label='identity')
        plt.plot(depth, depth_linear, label='linear')
        plt.plot(depth_n, depth_cubic.to_numpy()[o], label='cubic')
        plt.plot(depth_n, depth_pw.to_numpy()[o], label='piece-wise linear')
        plt.legend()
        plt.show()
    

# %%

# P = test_msi_minimal(path_folder)

P = Project(path_folder, depth_span=(490, 495))
# P.set_spectra()
P.set_msi_object()
P.set_xray(path_xray)
P.set_age_model()
P.set_age_span()
P.set_image_roi()

# P = test_msi_minimal(path_folder)

# self = P

# reader = P.get_mcf_reader()
# idx = 10
# %%
# idx += 1
# reader.get_spectrum(idx, limits=(552.52, 552.66)).plot()
# P.msi.plt_comp('L')

# s = Spectra(path_d_folder=os.path.join(path_folder, get_d_folder(path_folder)), load=True)
