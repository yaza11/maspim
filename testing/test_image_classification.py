import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import expand_labels

from imaging.main.cImage import ImageClassified

path_folder = 'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i'

IC = ImageClassified(path_folder)
IC.load()

plt.imshow(IC.get_image_expanded_laminae())

