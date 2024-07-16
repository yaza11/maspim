import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.align_net.synthetic_images import training_pair
from imaging.main.cImage import ImageClassified, ImageROI, ImageSample
from imaging.register.helpers import get_transect

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# np.random.seed(0)
# expected, warped = training_pair((200, 800))

img = np.asarray(Image.open(r'C:/Users/Yannick Zander/Nextcloud2/Promotion/msi_workflow/imaging/register/2020_03_23_Cariaco_535-540cm_Fullerite_0001.bmp').convert('L'))

i = ImageSample(image=img, obj_color='light')
ir = ImageROI.from_parent(i)
ir.set_age_span((0, 100))
ic = ImageClassified.from_parent(ir)

plt.imshow(ic.image_classification)
plt.show()

n_transects = 10

seeds_light = []
seeds_dark = []
ys_light = []
ys_dark = []

width_transect = ic.image.shape[0] / n_transects
halfwidth_transect = width_transect / 2
ys = halfwidth_transect + np.array([width_transect * i for i in range(n_transects)])
for i, y in enumerate(ys):
    img = get_transect(ic.image, i, n_transects)
    img_c = get_transect(ic.image_classification, i, n_transects)
    ic_ = ImageClassified(obj_color='light', image=img, image_classification=img_c)    
    ic_.set_age_span((0, 100))
    # plt.imshow(ic_.image)
    # plt.show()
    # plt.imshow(ic_.image_classification)
    # plt.show()

    ic_.set_seeds(in_classification=False, peak_prominence=.1, plts=True)
    x_light = list(ic_._seeds_light)
    x_dark = list(ic_._seeds_dark)
    seeds_light.extend(x_light)
    seeds_dark.extend(x_dark)
    
    ys_light.extend([y] * len(x_light))
    ys_dark.extend([y] * len(x_dark))
    
plt.imshow(ic.image)
plt.plot(seeds_light, ys_light, 'ro', alpha=.5)
plt.plot(seeds_dark, ys_dark, 'mo', alpha=.5)

    

