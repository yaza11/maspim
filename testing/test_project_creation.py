import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from data.cProject import Project, SampleImageHandler
from imaging.main.cImage import ImageSample, ImageROI

path_folder = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i'
# path_folder = "D:/Promotion/Test data"

P = Project(path_folder, depth_section=(490, 495))

# P.set_age_model(
#     path_file=r'G:/Meine Ablage/Master Thesis/AgeModel/480_510_MSI_age_model_mm_yr.txt',
#     sep='\t',
#     index_col=False,
#     load=False
# )

P.set_age_model()
P.set_age_span()

P.set_spectra()  
P.set_image_handler()
P.set_image_sample(obj_color='light')
P.set_image_roi()
P.set_image_classified()
# P.set_msi_object()
# P.add_photo_to_msi()
# P.add_holes_to_msi()
# P.add_depth_column()
# P.add_age_column()

# P.msi.save()


# %%
from data.cMSI import MSI
path_d_folder = r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i\2018_08_27 Cariaco 490-495 alkenones.d'

msi = MSI(path_d_folder)
msi.load()


mz_C37_2 = 553.53188756536
mz_C37_3 = 551.51623750122

msi.plt_comp(mz_C37_2)
msi.plt_comp(mz_C37_3)

# ts = msi.processing_zone_wise_average(zones_key='depth', columns=['551.5174', '553.5323', 'age'])
# ts['Uk37'] = ts['553.5323'] / (ts['553.5323'] + ts['551.5174'])

# df = pd.read_csv(r'C:/Users/yanni/Downloads/MD03-2621_UK37_SST_BAYSPLINE.tab', sep='\t')
# df['Age'] *= 1000
# # %%
# mask = (df.Age >= ts.age.min()) & (df.Age <= ts.age.max())

# plt.plot(ts['age'], ts['Uk37'], label='depth-wise')
# plt.plot(df.loc[mask, 'Age'], df.loc[mask, "UK37"], label='reference')
# plt.xlabel('age in yr b2k')
# plt.ylabel("UK'37")
# plt.legend()
# plt.show()