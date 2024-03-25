import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cProject import ProjectMSI
from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler
from data.file_helpers import get_d_folder, get_folder_structure

import time
import numpy as np

folders = [
    # r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i',
    r'D:/Cariaco Data for Weimin/495-500cm/2018_08_28 Cariaco 495-500 alkenones.i',
    r'D:/Cariaco Data for Weimin/500-505cm/2018_08_29 Cariaco 500-505 alkenones.i',
    r'D:/Cariaco Data for Weimin/505-510cm/2018_08_30 Cariaco 505-510 alkenones.i'
]

times = np.zeros(len(folders), dtype=float)

for idx, folder in enumerate(folders):
    tic = time.time()
    print(f'working on {folder} ...')
    
    path_d_folder = os.path.join(folder, get_d_folder(folder))
    
    reader = ReadBrukerMCF(path_d_folder=path_d_folder)
    reader.create_reader()
    reader.create_indices()
    reader.set_meta_data()
    reader.set_QTOF_window()
    
    handler = hdf5Handler(path_d_folder)
    handler.write(reader=reader)
    # pr = ProjectMSI(folder)
    # pr.set_spectra(plts=True)
    
    toc = time.time()
    dtime = (toc - tic) / 60  # min
    times[idx] = dtime
    print(f'This took {dtime:.1f} minutes.')
# pr = ProjectMSI(folders[0])
# pr.set_spectra()
