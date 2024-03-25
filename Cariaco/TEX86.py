import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cProject import Project
from timeSeries.cProxy import TEX86
from timeSeries.cTimeSeries import MultiSectionTimeSeries
from exporting.from_mcf.cSpectrum import MultiSectionSpectra
from res.constants import mGDGT1, mGDGT2, mGDGT3, mCren_p

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folders = [
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\490-495cm\2018_08_27 Cariaco 490-495 GDGT.i',
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\495-500cm\2018_08_28 Cariaco 495-500 GDGT.i',
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\500-505cm\2018_08_29 Cariaco 500-505 GDGT.i',
    r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\505-510cm\2018_08_31 Cariaco 505-510 GDGT.i'
]
depth_spans = [(490, 495), (495, 500), (500, 505), (505, 510)]

path_age_model = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\Age Model\480_510_MSI_age_model_mm_yr.txt'

readers = []
ps = []

SNR_threshold = 0

# folders = ['C:/Users/Yannick Zander/Promotion/Test data'] * 2

for folder, depth_span in zip(folders, depth_spans):
    # create hdf5
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
specs.full_targeted(readers=readers, targets=[mGDGT1, mGDGT2, mGDGT3, mCren_p], integrate_peaks=False, SNR_threshold=4)

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
    p.set_time_series(
        average_by_col='classification_s', 
        exclude_zeros=SNR_threshold>0  # only if filtering is active
    )
    ts.append(p.time_series)

# %%
ts_combined = MultiSectionTimeSeries(ts)

tx_combined = TEX86(ts_combined, valid_spectra_mode='any_above')
tx_combined.add_SST(method='conventional')
# tx_combined.add_SST(method='BAYSPAR', lat=10.5, lon=-65.16667)


# uk_combined.feature_table['UK37p_corrected'] = uk_combined.feature_table.UK37p * 1.194

df = pd.read_csv(
    'C:/Users/Yannick Zander/Downloads/MD03-2621_UK37_SST_BAYSPLINE.tab', 
    sep='\t',
    skiprows=20
)

df.columns = ['age', 'UK37', 'UK37_corrected', 'SST']
df.age *= 1000

tx_age = tx_combined.feature_table.age
tx_tx = tx_combined.feature_table.ratio
tx_SST = tx_combined.feature_table.SST


mask = (df.age >= tx_age.min()) & (df.age <= tx_age.max())

# %% UK val
# add standard errors
errs = tx_combined.get_feature_table_standard_errors()

# errs_tx = tx_combined.get_std_err_UK()
errs_tx = 0

plt.figure()
plt.fill_between(tx_age, tx_tx + errs_tx, tx_tx - errs_tx, alpha=.5, label='SE')
plt.plot(
    tx_age, 
    tx_tx, 
    label='TEX86'
)
plt.plot(df.age[mask], df.UK37_corrected[mask], label='paper Uk37')
plt.xlabel('Age in yrs b2k')
plt.ylabel('proxy values')
plt.legend()
plt.show()

# %%

# plt.figure()
# plt.plot(
#     uk_combined.feature_table.age, 
#     uk_combined.feature_table.UK37p_corrected, 
#     label='automatic'
# )
# plt.plot(df.age[mask], df.UK37_corrected[mask], label='paper')
# plt.xlabel('Age in yrs b2k')
# plt.ylabel('Uk37p values')
# plt.legend()
# plt.show()

# %% SST

# bin into same intervals
bin_width = 1
binned_SSTs = np.zeros_like(df.age[mask])
for idx, age_bin_center in enumerate(df.age[mask]):
    mask_bin = (tx_combined.feature_table.age >= age_bin_center - bin_width / 2) &\
        (tx_combined.feature_table.age < age_bin_center + bin_width / 2)
    binned_SSTs[idx] = tx_combined.feature_table.SST[mask_bin].mean()
    

plt.figure()
if "SST_lower" in tx_combined.feature_table:
    plt.fill_between(tx_age, tx_combined.feature_table.SST_lower, tx_combined.feature_table.SST_upper, alpha=.5, label='SE')
plt.plot(
    tx_age, 
    tx_SST, 
    label='TEX86'
)
plt.plot(
    df.age[mask], 
    binned_SSTs, 
    label='TEX86 binned'
)
plt.plot(df.age[mask], df.SST[mask], label='paper Uk37')
plt.xlabel('Age in yrs b2k')
plt.ylabel('SST in degrees C')
plt.legend()
plt.show()


