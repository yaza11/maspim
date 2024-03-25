import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler
from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from exporting.from_mcf.cSpectrum import Spectra

# d_folder = "D:/Promotion/Test data/13012023_SBB_TG3A_05-10_Test2.d"
d_folder = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d'

# reader = ReadBrukerMCF(d_folder)
# reader.create_reader()
# reader.create_indices()
# reader.set_meta_data()
# reader.set_QTOF_window()

# hdf_file = os.path.join(d_folder, "test.hdf5")

handler = hdf5Handler(d_folder)
# handler.write(reader)

spec = Spectra(reader=handler)
spec.full(reader=handler)
