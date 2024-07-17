from msi_workflow import get_project
from msi_workflow.res.compound_masses import mC37_2, mC37_3

plts: bool = True
path_i_folder: str = r'C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\484-485cm\2018_08_23 Cariaco 484-485 alkenones 100 um .i'

p = get_project(is_MSI=True, path_folder=path_i_folder)
p.require_images()  # sets or loads ImageHandler, ImageSample, ImageROI and ImageClassified
p.set_spectra(targets=[mC37_2, mC37_3])  # perform all steps to extract intensities from alkenones
p.set_data_object()
p.set_time_series(plts=plts)
p.set_UK37()
