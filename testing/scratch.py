import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imaging.main.cImage import ImageSample, ImageROI
from data.cAgeModel import AgeModel

import matplotlib.pyplot as plt
from PIL import Image

path_age_model = r'C:/Users/Yannick Zander/Promotion/Cariaco MSI 2024/Age Model/510-540/510_540_MSI_age_model_mm_yr.txt'
age_model = AgeModel(path_age_model, depth_offset=5100, conversion_to_cm=1 / 10, sep='\t',
                     index_col=False)


i = ImageSample(path_image_file=r'C:\Users\Yannick Zander\Nextcloud2\Promotion\msi_workflow\imaging\register\2020_03_23_Cariaco_535-540cm_Fullerite_0001.bmp')
ir = ImageROI.from_parent(i)
ir.age_span = age_model.depth_to_age((535, 540))

plt.imshow(ir.image)
plt.show()

plt.imshow(ir.image_classification)
plt.show()


im = Image.fromarray(ir.image_classification)
im.save(r"C:\Users\Yannick Zander\Nextcloud2\Promotion\msi_workflow\imaging\register\image_classified.png")
