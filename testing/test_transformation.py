import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.main.cImage import ImageROI
from imaging.XRay.cXRay import XRay
from imaging.register.main import Transformation
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
        ], axis = -1
    )
    return i_cpr

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

# plt_contours(t.source.sget_main_contour(), t.source.sget_image_original())

# t._transform_from_bounding_box(plts=True)
# warped = t.fit()



# plt.imshow(cpr_img(
#     t.target.sget_image_grayscale(), warped
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
    t.target.sget_image_grayscale(), ensure_image_is_gray(warped2)
))
plt.title('cprs warped image and target based on punchholes')    

