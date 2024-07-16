import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exporting.from_mcf.cSpectrum import Spectra
from exporting.from_mcf.rtms_communicator import ReadBrukerMCF, rtms
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler

import matplotlib.pyplot as plt
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)

# d_folder = r'C:\Users\Yannick Zander\Promotion\Test data\13012023_SBB_TG3A_05-10_Test2.d'
d_folder = r"C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\480-484cm\2018_08_24 Cariaco 480-484 100um alkenones.i\2018_08_24 Cariaco 480-484 alkenones.d"

# con = ReadBrukerMCF(d_folder)
# con.create_reader()
# con.create_indices()
# con.set_meta_data()
# con.set_QTOF_window()

reader = hdf5Handler(d_folder)
# # reader.write(con)
reader.create_indices()

# # df2 = reader.metaData
time0 = time.time()
spec = Spectra(path_d_folder=d_folder, initiate=True, reader=reader)
spec.add_all_spectra(reader=reader)
time1 = time.time()
print(f'adding up from hdf5 without calibration took {(time1-time0)//60:.0f} minutes and {(time1-time0) % 60:.0f} seconds')

# %%
time2 = time.time()
spec2 = Spectra(path_d_folder=d_folder, initiate=True, reader=reader)
spec2.add_all_spectra(reader=reader)
spec2.subtract_baseline()
spec2.set_calibrate_functions(calibrants_mz=[548.24410, 557.25231], reader=reader, SNR_threshold=2)
spec2.add_all_spectra(reader=reader, calibrate=True)
time3 = time.time()
print(f'adding up from hdf5 without calibration took {(time3-time2)//60:.0f} minutes and {(time1-time0) % 60:.0f} seconds')


plt.plot(spec.mzs, spec.intensities, label='not calibrated')
plt.plot(spec2.mzs, spec2.intensities, label='calibrated')
# plt.plot(spec2.mzs, spec2.noise_level, label='SNR')
plt.vlines([548.24410, 557.25231, 558.26544], 0, spec.intensities.max(), 'k')
plt.legend()
plt.show()

# spec.plt_summed()

# spec.subtract_baseline(plts=True)
# spec.set_calibrate_functions(use_sql=False, calibrants_mz=557.25231, reader=reader)

# %%
idx = 8699
spec_before = reader.get_spectrum(idx)
spec_after = reader.get_spectrum(idx, poly_coeffs=spec2.calibration_parameters[idx-1, :])
# spec_after = reader.get_spectrum(6655, poly_coeffs=[.0003, 0, -400])

print(np.poly1d(spec2.calibration_parameters[idx-1, :])(spec2.mzs))

plt.plot(spec_before.mzs, spec_before.intensities, label='not calibrated')
plt.plot(spec_after.mzs, spec_after.intensities, '--', label='calibrated')
plt.plot(spec2.mzs, spec2.noise_level, label='SNR')
plt.vlines([548.24410, 557.25231], 0, 40_000, 'k')
# plt.xlim(557.25231 - .01, 557.25231 + .01)
plt.legend()
plt.show()
