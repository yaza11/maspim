import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.main.cImage import (
    Image, ImageProbe_from_Image, ImageROI_from_ImageProbe, 
    ImageClassified_from_ImageROI
)
from imaging.util.Image_plotting import plt_cv2_image

I_HAVE_TIME = False

image_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'exampleData/example.tif'
)

# initialize from path
Img = Image(image_path=image_path)
# turn into probe object
ImgP = ImageProbe_from_Image(Img)
# find sample area
ImgP.sget_probe_area()
plt_cv2_image(ImgP.sget_probe_area()[0])
# initalize ROI
ImgROI = ImageROI_from_ImageProbe(ImgP)
# have to set manually for conversion from pixel to age
ImgROI._section = (490, 495)
ImgROI.sget_classification_adaptive_mean()
plt_cv2_image(ImgROI.sget_classification_adaptive_mean()[0])
# classification object
imgC = ImageClassified_from_ImageROI(ImgROI)
# this can take a while
if I_HAVE_TIME:
    imgC.set_laminae_params_table(
        peak_prominence=.1,
        max_slope=.1,
        downscale_factor=1/16
    )
