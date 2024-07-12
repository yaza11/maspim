import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project.cProject import get_project
from imaging.main.cImage import ImageROI, ImageSample
from data.cAgeModel import AgeModel
from imaging.util.Image_boxes import region_in_box
from imaging.util.Image_plotting import plt_rect_on_image, plt_contours, plt_cv2_image
from imaging.register.transformation import Transformation

import matplotlib.pyplot as plt
import numpy as np
import logging


def detected_roi(p):
    i = p.image_sample
    _, xywh = i.get_sample_area_box()
    x, y, w, h = xywh
    box_params: dict[str, float] = region_in_box(image=i.image, x=x, y=y, w=w, h=h)

    fig, ax = plt.subplots()
    fig, ax = plt_rect_on_image(image=i.image_binary, box_params=box_params, no_ticks=True, fig=fig, ax=ax, hold=True)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig('sample_area.png', dpi=600, bbox_inches="tight", pad_inches=0)
    plt.show()


def detected_contour(p):
    i = p.image_sample
    img, xywh = i.get_sample_area_box(dilate_factor=.1)
    image_sub = ImageSample(image=img, obj_color=i.obj_color)
    # set image simplified for contour to use
    image_sub._mask_foreground = image_sub.image_simplified
    # find the refined area as the extent of the simplified binary image
    _, (xc, yc, wc, hc) = image_sub.get_sample_area_from_contour(method='filter_by_size')

    cont = image_sub.main_contour
    fig, ax = plt.subplots()
    fig, ax = plt_contours(image=image_sub.image, contours=cont, no_ticks=True, save_png='contour.png', fig=fig, ax=ax,
                           hold=True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # fig.tight_layout()
    fig.savefig('contour.png', dpi=600, bbox_inches="tight", pad_inches=0)
    plt.show()


def warping(p1, p2) -> tuple[Any, Any, Any]:
    p1.set_image_roi()

    p2.set_depth_span(depth_span)
    p2.age_model = age_model
    p2.set_age_span()

    p2.set_image_handler()

    p2.set_image_sample(obj_color='light')
    p2.set_image_roi()

    logging.basicConfig(level=logging.INFO)

    t = Transformation(source=p2.image_roi, target=p1.image_roi)

    t.estimate('bounding_box', plts=False)
    t.estimate('laminae', n_transects=3, plts=True, deg=5)
    # t.estimate('image_flow', plts=True, simplify=True, use_classified=True)

    img_transformed = t.fit(p2.image_roi.image_classification)
    # t.save('D:')

    fig, axs = t.plot_fit(use_classified=True, simplify=True)
    fig.savefig('register.png', dpi=600)

    return p1, p2, t


def detect_laminae(p):
    p.set_image_classified()
    i = p.image_classified

    img = i.get_image_expanded_laminae()
    img[img > 0] = 255
    img[img < 0] = 127

    fig, ax = plt.subplots()
    fig, ax = plt_cv2_image(image=img, no_ticks=True, fig=fig, ax=ax,
                            hold=True, cmap='plasma'
                            )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # fig.tight_layout()
    fig.savefig('laminae.png', dpi=600, bbox_inches="tight", pad_inches=0)
    plt.show()


depth_span = (490, 495)
p1_folder = r''
p2_folder = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\490-495cm\2018_08_27 Cariaco 490-495 FA.i'

p1 = get_project(is_MSI=p1_folder.startswith('C'), path_folder=p1_folder)
p2 = get_project(is_MSI=p2_folder.startswith('C'), path_folder=p2_folder)

path_age_model_480_510 = r'C:/Users/Yannick Zander/Promotion/Cariaco 2024/Age Model/480-510/480_510_MSI_age_model_mm_yr.txt'
age_model = AgeModel(path_age_model_480_510, depth_offset=4800, conversion_to_cm=1 / 10, sep='\t',
                     index_col=False)

p2.set_depth_span(depth_span)
p2.age_model = age_model
p2.set_age_span()

p2.set_image_handler()
p2.set_image_sample(obj_color='light')
p2.set_image_roi()


p1.set_depth_span(depth_span)
p1.age_model = age_model
p1.set_age_span()

p1.set_image_handler()
p1.set_image_sample(obj_color='light')
p1.set_image_roi()

# detected_roi(p2)
# detected_contour(p2)
# detect_laminae(p2)
p1, p2, t = warping(p1, p2)

# p1.image_roi.set_punchholes(side='bottom')
# p1.set_image_classified()

# p1.set_spectra()
# p1.set_data_object()

# p1.add_depth_column()
# p1.add_age_column()
# p1.add_pixels_ROI()
# p1.add_holes()
# p1.add_light_dark_classification(plts=True)
# p1.add_laminae_classification()
