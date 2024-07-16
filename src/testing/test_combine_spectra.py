import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exporting.from_mcf.cSpectrum import Spectra, MultiSectionSpectra
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler
from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
import matplotlib.pyplot as plt
import numpy as np

# path_d_folders = [
#     'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d',
#     'D:/Cariaco Data for Weimin/495-500cm/2018_08_28 Cariaco 495-500 alkenones.i/2018_08_28 Cariaco 495-500 alkenones.d',
#     'D:/Cariaco Data for Weimin/500-505cm/2018_08_29 Cariaco 500-505 alkenones.i/2018_08_29 Cariaco 500-505 alkenones.d',
#     'D:/Cariaco Data for Weimin/505-510cm/2018_08_30 Cariaco 505-510 alkenones.i/2018_08_30 Cariaco 505-510 alkenones.d'
# ]
# labels = ['490-495', '495-500', '500-505', '505-510']

path_d_folders = [
    '//hlabstorage/scratch/Yannick/13012023_SBB_TG3A_05-10_Test2/13012023_SBB_TG3A_05-10_Test2.d',
    '//hlabstorage/scratch/Yannick/13012023_SBB_TG3A_05-10_Test2/13012023_SBB_TG3A_05-10_Test2.d',
    '//hlabstorage/scratch/Yannick/13012023_SBB_TG3A_05-10_Test2/13012023_SBB_TG3A_05-10_Test2.d'
]

# readers = [hdf5Handler(path_d_folder) for path_d_folder in path_d_folders]
readers = [ReadBrukerMCF(path_d_folder) for path_d_folder in path_d_folders]
for reader in readers:
    reader.create_reader()

spectra = MultiSectionSpectra(readers)
spectra.full(readers)
