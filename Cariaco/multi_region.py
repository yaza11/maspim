import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cProject import Project

folder = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\505-510cm\2018_08_30 Cariaco 505-510 FA.i'

depth_span = (505, 510)

path_age_model = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\Age Model\480_510_MSI_age_model_mm_yr.txt'

p = Project(is_MSI=True, path_folder=folder, d_folder='2018_08_30 Cariaco 505-510 FA A.d')
p.set_image_handler()
# p.image_handler.set_photo(plts=True)
p.image_handler.get_photo_ROI(plts=True)

p.set_spectra()
p.set_object()
p.set_depth_span((505, 507))
p.set_image_sample()
p.add_pixels_ROI()
p.add_photo()
p.data_obj.plt_comp('L', clip_at=1)
