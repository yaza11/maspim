import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from data.cProject import Project, ImagingInfoXML
from imaging.main.cImage import ImageSample, ImageROI, ImageClassified
from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from exporting.from_mcf.cSpectrum import Spectra
from imaging.util.Image_convert_types import ensure_image_is_gray

path_folder = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i'
path_folder2 = r'D:/Cariaco Data for Weimin/495-500cm/2018_08_28 Cariaco 495-500 alkenones.i'
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
    P = Project(is_MSI=True, path_folder=path_folder)
    
    # age model (required for ImageROI (choice of filter size) and to add age to MSI)
    print('setting age model ...')
    P.set_age_model()
    print('setting depth span ...')
    P.set_depth_span(depth_span=depth_span)  # required for age_span and add_depth_to_msi
    print('setting age span ...')
    P.set_age_span()
    
    # spectra (required for set_msi)
    print('setting spectra ...')
    P.set_spectra()  
    
    # images
    print('setting image handler ...')
    P.set_image_handler()  # required for adding photos
    print('setting image_sample ...')
    P.set_image_sample(obj_color=obj_color)  # required for adding photo to msi
    print('setting image_roi ...')
    P.set_image_roi()  # required for adding hole, light, dark information to msi
    print('setting image_classified ...')
    P.set_image_classified()  # for adding laminae information to msi
    
    # msi
    print('setting data object ...')
    P.set_object()
    print('setting ROI')
    P.add_pixels_ROI()
    print('setting photo in ft ...')
    P.add_photo()
    print('adding hole classification to ft ...')
    P.add_holes()
    print('adding depth to ft ...')
    P.add_depth_column()
    print('adding age column to tf ...')
    P.add_age_column()
    print('adding light/dark classification to ft ...')
    P.add_light_dark_classification()
    print('adding laminae classifiaction to ft ...')
    P.add_laminae_classification()
    
    # time series
    print('setting time series ...')
    P.set_time_series()
    
    # msi object is not saved by default
    # P.msi.save()
    # P.set_msi_object will try to load a saved msi object
    return P

def test_msi_minimal(path_folder):
    P = Project(True, path_folder)
    
    P.set_spectra()
    P.set_object()
    return P

def test_proxy(path_folder):
    P = test_all(path_folder)
    P.set_UK37()
    
    P.UK37_proxy.plot()
    
def test_punch_holes(
        path_folder, path_xray, depth_xray=None, side='bottom', plts=False
):
    P = test_all(path_folder)
    print('setting xray object ...')
    P.set_xray(path_xray, depth_xray)
    print('setting punch_holes ...')
    P.set_punch_holes(plts=False, side=side)
    
    print('adding xray to ft ...')
    P.add_xray(plts=False)

    # test all methods    
    print('adding corrected depth linear ...')
    depth = P.data_obj.feature_table.depth.copy()
    P.set_depth_correction_with_xray(method='l')
    depth_linear = P.data_obj.feature_table.depth_corrected.copy()

    print('... cubic ...')
    P.set_depth_correction_with_xray(method='c')
    depth_cubic = P.data_obj.feature_table.depth_corrected.copy()

    print('... piecewise linear ...')
    P.set_msi_depth_correction_with_xray(method='pwl')
    depth_pw = P.data_obj.feature_table.depth_corrected.copy()

    if plts:
        msize=.5
        plt.figure()
        plt.plot(depth, depth - depth, 'o', markersize=msize, label='identity')
        plt.plot(depth, depth_linear - depth, 'o', markersize=msize, label='linear')
        plt.plot(depth, depth_cubic - depth, 'o', markersize=msize, label='cubic')
        plt.plot(depth, depth_pw - depth, 'o', markersize=msize, label='piece-wise linear')
        plt.xlabel('depth in cm')
        plt.ylabel('difference in cm')
        plt.legend()
        plt.show()
    return P


# self = Project(is_MSI = True, path_folder=path_folder)
# print(self.__dict__)
# self = test_all(path_folder)
# self = test_punch_holes(path_folder, path_xray, plts=True, depth_xray=(480, 510))
# self.set_spectra()
# self.set_object()
# self.set_depth_span((490, 495))
# self.set_xray(path_xray)
# self.set_image_roi()

# self.add_depth_column()

# self.set_punch_holes(side='bottom')

# self.set_image_handler()
# self.set_image_sample()
# self.add_pixels_ROI()

# self.add_xray(plts=True, is_piecewise=False)
# self.add_xray(plts=True, is_piecewise=True)


P = test_msi_minimal(path_folder)
P.set_depth_span((490, 495))
P.add_depth_column()
P2 = test_msi_minimal(path_folder2)
P2.set_depth_span((495, 500))
P2.add_depth_column()

msi = P.data_obj
msi2 = P2.data_obj

msi_new = msi.combine_with(msi2)
# P = Project(is_MSI=True, path_folder=path_folder, depth_span=(490, 495))
# P.set_spectra()
# P.set_object()
# P.set_spectra()
# P.set_msi_object()
# P.set_xray(path_xray)
# P.set_age_model()
# P.set_age_span()
# P.set_image_roi()

# P = test_msi_minimal(path_folder)

# self = P

# reader = P.get_mcf_reader()
# idx = 10

# idx += 1
# reader.get_spectrum(idx, limits=(552.52, 552.66)).plot()
# P.msi.plt_comp('L')

# s = Spectra(path_d_folder=os.path.join(path_folder, get_d_folder(path_folder)), load=True)
