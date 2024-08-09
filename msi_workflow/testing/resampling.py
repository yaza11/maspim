import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from msi_workflow.exporting.from_mcf.rtms_communicator import ReadBrukerMCF

import numpy as np
import matplotlib.pyplot as plt


# d_folder = r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i\2018_08_27 Cariaco 490-495 alkenones.d'
d_folder = 'D:/Promotion/Test data/13012023_SBB_TG3A_05-10_Test2.d'

reader = ReadBrukerMCF(path_d_folder=d_folder)
reader.create_reader()
reader.create_indices()
reader.set_meta_data()
reader.set_casi_window()

spec_orig = reader.get_spectrum(1, limits=reader.limits)

spec_res = spec_orig.copy()
spec_res.resample(delta_mz=5e-4)

plt.plot(spec_orig.mzs, spec_orig.intensities, label='original')
plt.plot(spec_res.mzs, spec_res.intensities, label='resampled')

