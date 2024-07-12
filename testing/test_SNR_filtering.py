import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project.cProject import get_project
from res.constants import mC37_2, mC37_3

import matplotlib.pyplot as plt
import numpy as np

folder = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i'

p = get_project(is_MSI=True, path_folder=folder)

reader = p.create_hdf_file()
# set Spectra
p.set_spectra(reader, full=True)
p.spectra.bin_spectra(reader, integrate_peaks=False)
p.set_image_handler()
p.set_image_sample()
p.set_image_roi()

snrs = p.spectra._get_SNR_table()
idxs_peaks = [np.argmin(np.abs(p.spectra.kernel_params[:, 0] - m)) for m in [mC37_2, mC37_3]]

# control
print(mC37_2, p.spectra.kernel_params[idxs_peaks[0], 0])
print(mC37_3, p.spectra.kernel_params[idxs_peaks[1], 0])


# %%
idx_spec = 3 + 1
print(snrs[idx_spec - 1, np.array(idxs_peaks)])

plt.plot(p.spectra.mzs, p.get_reader().get_spectrum_resampled_intensities(idx_spec))
plt.plot(p.spectra.mzs, p.spectra.noise_level, label='noise')
plt.vlines([mC37_2, mC37_3], 0, 1e6, color='k', linestyle='--')

# %%
thresholds = [0, 1, 2, 3, 4, 5]
for snr in thresholds:
    p.set_data_object(SNR_threshold=snr)
    
    p.add_pixels_ROI()
    p.add_photo()
    p.add_holes()
    
    p.data_obj.plot_comp(mC37_2)
