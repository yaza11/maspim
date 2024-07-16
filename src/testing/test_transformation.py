import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project.cProject import get_project
from data.cAgeModel import AgeModel

from imaging.main.cImage import ImageROI, ImageSample
from imaging.XRay.cXRay import XRay
from imaging.register.transformation import Transformation
from imaging.util.Image_plotting import plt_contours
from imaging.util.Image_convert_types import ensure_image_is_gray
from imaging.util.coordinate_transformations import rescale_values

import numpy as np
import matplotlib.pyplot as plt


def cpr_img(i1, i2):
    i1 = rescale_values(ensure_image_is_gray(i1), 0, 1)
    i2 = rescale_values(ensure_image_is_gray(i2), 0, 1)

    i_cpr = np.stack(
        [
            i1,
            i1 / 2 + i2 / 2,
            i2
        ], axis=-1
    )
    return i_cpr


def test_punch():
    path_folder = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i'
    path_xray = r'D:/Cariaco line scan Xray/Cariaco Xray/sliced/MD_03_2621_480-510_sliced_1200dpi.tif'

    pxray: XRay = XRay(path_xray)
    pmsi: ImageROI = ImageROI(path_folder)

    pxray.load()
    pmsi.load()

    px: ImageROI = pxray.get_ImageROI_from_section(490, 495)

    # add punchholes 
    px.set_punchholes(side='bottom', plts=False, remove_gelatine=False)
    pmsi.set_punchholes(side='bottom', plts=False, remove_gelatine=True)

    t: Transformation = Transformation(px, pmsi)

    # plt_contours(t.source._require_main_contour(), t.source.image())

    # t._transform_from_bounding_box(plts=True)
    # warped = t.fit()

    # plt.imshow(cpr_img(
    #     t.target.image_grayscale(), warped
    # ))
    # plt.title('cprs warped image and target based on sample area extent')

    t._transform_from_punchholes(
        points_source=t.source.punchholes,
        points_target=t.target.punchholes,
        is_piecewise=True,
        hole_side='bottom'
    )

    warped2 = t.fit()

    plt.imshow(cpr_img(
        t.target.image_grayscale(), ensure_image_is_gray(warped2)
    ))
    plt.title('cprs warped image and target based on punchholes')


def warping(p1, p2) -> tuple[Any, Any, Any]:
    p1.set_image_roi()

    p2.set_depth_span(depth_span)
    p2.age_model = age_model
    p2.set_age_span()

    p2.set_image_handler()

    p2.set_image_sample(obj_color='light')
    p2.set_image_roi()

    t = Transformation(source=p2.image_roi, target=p1.image_roi)

    t.estimate('bounding_box', plts=False)
    t.estimate('laminae', n_transects=3, plts=True, deg=5)
    # t.estimate('image_flow', plts=True, simplify=True, use_classified=True)

    img_transformed = t.fit(p2.image_roi.image_classification)

    fig, axs = t.plot_fit(use_classified=True, simplify=True)
    fig.savefig('register.png', dpi=600)

    return p1, p2, t


logging.basicConfig(level=logging.INFO)

depth_span = (490, 495)
target = ImageSample(
    path_image_file=r'C:/Users/Yannick Zander/Promotion/Cariaco MSI 2024/muXRF/S0343 Cariaco_480-510cm_100um slices/S0343c_490-495cm/Caricao_490-495cm_100um_Mosaic.tif'
)
source = ImageSample(
    path_image_file=r'C:/Users/Yannick Zander/Promotion/Cariaco MSI 2024/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenone_0000.tif'
)

path_age_model_480_510 = r'C:/Users/Yannick Zander/Promotion/Cariaco MSI 2024/Age Model/480-510/480_510_MSI_age_model_mm_yr.txt'
age_model = AgeModel(
    path_age_model_480_510,
    depth_offset=4800,
    conversion_to_cm=1 / 10,
    sep='\t',
    index_col=False
)

# apply tilt correct to target
t1 = Transformation(source=target, target=None)
t1.estimate('tilt')
target = t1.get_transformed_source()

target.age_span = age_model.depth_to_age(depth_span)
source.age_span = age_model.depth_to_age(depth_span)

t = Transformation(source=source, target=target)

t.estimate('bounding_box', plts=True)
t.estimate('tilt', plts=True)
t.estimate('laminae', plts=True, degree=5)
# t.estimate('image_flow', plts=True, simplify=True, use_classified=True)

t.plot_fit(use_classified=False)

mapper = t.to_mapper()
