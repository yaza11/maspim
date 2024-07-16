import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project.cProject import get_project
from res.constants import mC37_2, mC37_3

path_folder=r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i'

p = get_project(is_MSI=True, path_folder=path_folder)

reader = p.get_reader()

# reader = ReadBrukerMCF(path_d_folder)

p.set_spectra(reader=reader, full=True)

spec = p.spectra
# spec.full(reader)
# spec.save()

# spec.set_peaks()
# spec.set_kernels()
# spec.set_targets([mC37_2, mC37_3], method='nearest_peak', plts=True)

spec.set_peaks()
spec.set_kernels()
spec.set_targets([mC37_2, mC37_3], method='area_overlap', plts=True)

# spec.plt_summed(plt_kernels=True)
