import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.main.cImage import Image, ImageSample, ImageROI, ImageClassified
from imaging.register.transformation import Transformation

import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

def test_trafo():
    file1 = r'D:/Cariaco line scan Xray/uXRF slices/S0343c_490-495cm/PS343c 490-495cm Mosaic.bmp'
    file2 = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenone_0002.tif'
    
    img1: ImageSample = ImageSample(path_image_file=file1)
    img2: ImageSample = ImageSample(path_image_file=file2) 
    
    t = Transformation(source=img1, target=img2)
    t._transform_from_bounding_box(plts=False)
    warped = t.fit()
    
    fig, axs = plt.subplots(nrows=3)
    axs[0].imshow(t.target.image())
    axs[0].set_title('target')
    
    axs[1].imshow(t.source.image())
    axs[1].set_title('source')
    
    axs[2].imshow(warped)
    axs[2].set_title('warped')
    
    plt.tight_layout()
    plt.show()


def test_image_sample():
    path_test_file = r'C:/Users/Yannick Zander/Promotion/Cariaco 2024/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenone_0001.tif'
    i = ImageSample(path_image_file=path_test_file, obj_color='light')
    i.plot_overview()
    
path_test_file = r'F:\530-535cm\2019_07_08_Cariaco_530-535cm_Alkenones.i'
folder = os.path.dirname(path_test_file)

# i = ImageSample.from_disk(folder)

# i = ImageSample(path_image_file=path_test_file)
# i._require_image_sample_area(plts=True)

# i.require_sample_area
# i.plot_overview()
# i.save()

# ir = ImageROI.from_parent(i)
# ir.age_span = (11393.916733067728, 11495.626)
# ir._require_classification()
# ir.set_punchholes(True, plts=True)
# ir.plot_overview()

# ir = ImageROI.from_disk(folder)


ic = ImageClassified.from_disk(path_test_file)
ic.plot_image_seeds_and_classification()
ic.reduce_laminae()
ic.plot_image_seeds_and_classification()
# ic.age_span = (11393.916733067728, 11495.626)
# ic = ImageClassified.from_parent(ir)

# ic.set_seeds(plts=True, in_classification=True, peak_prominence=.1)
# ic._set_params_laminae_simplified()
# ic.set_quality_score(plts=True)
# ic.set_laminae_images_from_params(plts=True)
# ic.plot_overview()
# ic.save()

# ic.load()
# ic.plot_image_seeds_and_classification()
