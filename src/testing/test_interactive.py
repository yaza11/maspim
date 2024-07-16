import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.main.cImage import ImageSample, ImageROI

import matplotlib.pyplot as plt

i = ImageSample(path_image_file=r'C:/Users/Yannick Zander/Promotion/Cariaco 2024/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenone_0002.tif')

i.set_sample_area(interactive=False)

# plt.figure()
# plt.imshow(i._image_roi)
# plt.show() 

ir = ImageROI.from_parent(i)
ir._user_punchholes()
ir.plot_overview()
