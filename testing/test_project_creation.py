import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import logging

from Project.cProject import get_project

logger = logging.getLogger('msi_workflow.' + __name__)

path_folder = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i'
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
    P = get_project(is_MSI=True, path_folder=path_folder)

    # age model (required for ImageROI (choice of filter size) and to add age to MSI)
    logger.info('setting age model ...')
    P.set_age_model()
    logger.info('setting depth span ...')
    P.set_depth_span(depth_span=depth_span)  # required for age_span and add_depth_to_msi
    logger.info('setting age span ...')
    P.set_age_span()

    # spectra (required for set_msi)
    logger.info('setting spectra ...')
    P.set_spectra()

    # images
    logger.info('setting image handler ...')
    P.set_image_handler()  # required for adding photos
    logger.info('setting image_sample ...')
    P.set_image_sample(obj_color=obj_color)  # required for adding photo to msi
    logger.info('setting image_roi ...')
    P.set_image_roi()  # required for adding hole, light, dark information to msi
    logger.info('setting image_classified ...')
    P.set_image_classified()  # for adding laminae information to msi

    # msi
    logger.info('setting data object ...')
    P.set_object()
    logger.info('setting ROI')
    P.add_pixels_ROI()
    logger.info('setting photo in ft ...')
    P.add_photo()
    logger.info('adding hole classification to ft ...')
    P.add_holes()
    logger.info('adding depth to ft ...')
    P.add_depth_column()
    logger.info('adding age column to tf ...')
    P.add_age_column()
    logger.info('adding light/dark classification to ft ...')
    P.add_light_dark_classification()
    logger.info('adding laminae classifiaction to ft ...')
    P.add_laminae_classification()

    # time series
    logger.info('setting time series ...')
    P.set_time_series()

    # msi object is not saved by default
    # P.msi.save()
    # P.set_msi_object will try to load a saved msi object
    return P


def test_msi_minimal(path_folder):
    P = get_project(True, path_folder)

    P.set_spectra()
    P.set_object()
    return P


def test_proxy(path_folder):
    P = test_all(path_folder)
    P.set_UK37()

    P.UK37_proxy.plot()


def test_punch_holes(
        path_folder, path_xray, depth_xray=None, plts=False
):
    P = test_all(path_folder)
    logger.info('setting xray object ...')
    P.set_xray(path_xray, depth_xray)
    logger.info('setting punch_holes ...')
    P.set_punchholes(plts=False)

    logger.info('adding xray to ft ...')
    P.add_xray(plts=False)

    # test all methods    
    logger.info('adding corrected depth linear ...')
    depth = P.data_obj.feature_table.depth.copy()
    P.add_depth_correction_with_xray(method='l')
    depth_linear = P.data_obj.feature_table.depth_corrected.copy()

    logger.info('... cubic ...')
    P.add_depth_correction_with_xray(method='c')
    depth_cubic = P.data_obj.feature_table.depth_corrected.copy()

    logger.info('... piecewise linear ...')
    P.add_depth_correction_with_xray(method='pwl')
    depth_pw = P.data_obj.feature_table.depth_corrected.copy()

    if plts:
        msize = .5
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


# self = get_project(is_MSI = True, path_folder=path_folder)
# logger.info(self.__dict__)
self = test_all(path_folder)
self = test_punch_holes(path_folder, path_xray, plts=True, depth_xray=(480, 510))
# self.set_spectra()
# self.set_object()
# self.set_depth_span((490, 495))
# self.set_xray(path_xray)
# self.set_image_roi()

# self.add_depth_column()

# self.set_punchholes(side='bottom')

# self.set_image_handler()
# self.set_image_sample()
# self.add_pixels_ROI()

# self.add_xray(plts=True, is_piecewise=False)
# self.add_xray(plts=True, is_piecewise=True)


# P = test_msi_minimal(path_folder)
# P.set_depth_span((490, 495))
# P.add_depth_column()
# P2 = test_msi_minimal(path_folder2)
# P2.set_depth_span((495, 500))
# P2.add_depth_column()

# msi = P.data_obj
# msi2 = P2.data_obj

# msi_new = msi.combine_with(msi2)
# P = get_project(is_MSI=True, path_folder=path_folder, depth_span=(490, 495))
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
# P.msi.plot_comp('L')

# s = Spectra(path_d_folder=os.path.join(path_folder, get_d_folder(path_folder)), load=True)
