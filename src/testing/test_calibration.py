import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from exporting.from_mcf.cSpectrum import Spectra
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler

import logging

logging.basicConfig(level=logging.INFO)

# path_d_folder = r'C:/Users/Yannick Zander/Promotion/Cariaco MSI 2024/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d'
path_d_folder = r'F:\520-525cm\2019_07_22_Cariaco_520-525cm_60um_FA.i\2019_07_22_Cariaco_520-525cm_60um_long FA.d'

reader = hdf5Handler(path_d_folder)

spec = Spectra(path_d_folder=path_d_folder, initiate=False)
spec.load()
# spec.path_d_folder = path_d_folder

spec.set_calibrate_functions(reader=reader, SNR_threshold=2, max_degree=1)
# spec.save()

spec.plot_calibration_functions(reader, n=5)
