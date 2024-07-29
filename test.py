from msi_workflow import get_project, AgeModel
from msi_workflow.res.compound_masses import mC37_2, mC37_3

age_model = AgeModel(depth=(480, 485), age=(0, 1))

plts: bool = True
path_i_folder: str = r'C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\484-485cm\2018_08_23 Cariaco 484-485 alkenones 100 um .i'

p = get_project(is_MSI=True, path_folder=path_i_folder)
p.depth_span = (482, 485)
p.age_model = age_model
p.require_images()  # sets or loads ImageHandler, ImageSample, ImageROI and ImageClassified
p.require_spectra(targets=[mC37_2, mC37_3])  # perform all steps to extract intensities from alkenones
p.require_data_object()
p.add_image_attributes()
# p.add_depth_column()
# p.add_age_column()
# p.require_time_series(plts=plts, overwrite=True, average_by_col='depth')
# p.set_UK37()


p.plot_comp(mC37_2, 'data_object', plot_on_background=True)