import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cProject import Project
from timeSeries.cProxy import Sterane
from timeSeries.cTimeSeries import MultiSectionTimeSeries
from exporting.from_mcf.cSpectrum import MultiSectionSpectra
from res.constants import mC28, mC29

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folders = [
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\490-495cm\2018_08_27 Cariaco 490-495 FA.i',
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\495-500cm\2018_08_28 Cariaco 495-500 FA.i',
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\500-505cm\2018_08_29 Cariaco 500-505 FA.i',
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\505-510cm\2018_08_30 Cariaco 505-510 FA.i',
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\505-510cm\2018_08_30 Cariaco 505-510 FA.i'
]
depth_spans = [(490, 495), (495, 500), (500, 505), (505, 507), (507, 510)]

d_folders = {
    (505, 507): '2018_08_30 Cariaco 505-510 FA A.d',
    (507, 510): '2018_08_30 Cariaco 505-510 FA B.d'
}

path_age_model = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\Age Model\480_510_MSI_age_model_mm_yr.txt'

readers = []
ps = []

SNR_threshold = 0

# folders = ['C:/Users/Yannick Zander/Promotion/Test data'] * 2

for folder, depth_span in zip(folders, depth_spans):
    print(folder)
    # create hdf5
    if depth_span[0] >= 505:
        p = Project(is_MSI=True, path_folder=folder, d_folder=d_folders[depth_span])
    else:
        p = Project(is_MSI=True, path_folder=folder)
    reader = p.create_hdf_file()
    # set Spectra
    p.set_spectra(reader, full=False)
    readers.append(reader)
    p.set_age_model(path_age_model, sep='\t', index_col=False, load=False)
    p.age_model.add_depth_offset(4800)
    p.age_model.convert_depth_scale(1 / 10)  # convert mm to cm
    p.set_depth_span(depth_span)
    p.set_age_span()
    p.set_image_handler()
    p.set_image_sample()
    p.set_image_roi()
    p.set_image_classified()
    ps.append(p)

# initiate multi section spectra object
specs = MultiSectionSpectra(readers)
# perform the necessary processing steps
specs.full_targeted(readers=readers, targets=[mC28, mC29], integrate_peaks=False, SNR_threshold=4)

ts = []
for p, spec in zip(ps, specs.specs):
    p.spectra = spec
    p.set_object()
    p.add_pixels_ROI()
    p.add_photo()
    p.add_holes()
    p.add_depth_column()
    p.add_age_column()
    p.add_light_dark_classification()
    p.add_laminae_classification()
    if SNR_threshold > 0:
        # set intensities to zero if any of the compounds is zero
        cols = p.data_obj.get_data_columns()
        # test if all entries are nonzero
        mask_all_nonzero = p.data_obj.feature_table.loc[:, cols].all(axis='columns')
        p.data_obj.feature_table.loc[~mask_all_nonzero, cols] = 0
    # p.set_time_series(average_by_col='classification_s')
    
ts_combined = MultiSectionTimeSeries(ts)


rt = Sterane(ts_combined)

df = pd.read_csv(
    'C:/Users/Yannick Zander/Downloads/MD03-2621_UK37_SST_BAYSPLINE.tab', 
    sep='\t',
    skiprows=20
)

df.columns = ['age', 'UK37', 'UK37_corrected', 'SST']
df.age *= 1000

mask = (df.age >= rt.feature_table.age.min()) & (df.age <= rt.feature_table.age.max())

# %%
plt.figure()
plt.plot(
    rt.feature_table.age, 
    rt.feature_table.ratio, 
    label='Sterane ratio'
)
plt.plot(df.age[mask], df.UK37_corrected[mask], label='paper')
plt.xlabel('Age in yrs b2k')
plt.ylabel('proxy values')
plt.legend()
plt.show()
    
    