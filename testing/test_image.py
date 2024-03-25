import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.main.cImage import ImageSample
from imaging.register.main import Transformation

import matplotlib.pyplot as plt

file1 = r'D:/Cariaco line scan Xray/uXRF slices/S0343c_490-495cm/PS343c 490-495cm Mosaic.bmp'
file2 = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenone_0002.tif'

img1: ImageSample = ImageSample(path_image_file=file1)
img2: ImageSample = ImageSample(path_image_file=file2) 

t = Transformation(source=img1, target=img2)
t._transform_from_bounding_box(plts=False)
warped = t.fit()

fig, axs = plt.subplots(nrows=3)
axs[0].imshow(t.target.sget_image_original())
axs[0].set_title('target')

axs[1].imshow(t.source.sget_image_original())
axs[1].set_title('source')

axs[2].imshow(warped)
axs[2].set_title('warped')

plt.tight_layout()
plt.show()
