from msi_workflow import get_project, AgeModel
from msi_workflow.res.compound_masses import mC37_2, mC37_3

age_model = AgeModel(depth=(480, 485), age=(0, 1))

plts: bool = True
# path_i_folder: str = r'C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i'
path_i_folder = r'C:\Users\Yannick Zander\Promotion\Test data'

p = get_project(is_MSI=True, path_folder=path_i_folder)

hdf = p.get_hdf_reader()
mcf = p.get_mcf_reader()

hdf.get_spectrum(5).plot()
mcf.get_spectrum(5).plot()

# p.depth_span = (490, 495)
# p.age_model = age_model
# p.require_images()  # sets or loads ImageHandler, ImageSample, ImageROI and ImageClassified
# p.require_spectra(targets=[mC37_2, mC37_3])  # perform all steps to extract intensities from alkenones
# p.require_data_object()
# p.add_image_attributes()
# p.add_depth_column()
# p.add_age_column()
# p.require_time_series(plts=plts, overwrite=True, average_by_col='depth')
# p.set_UK37()

# p.image_classified.plot_overview()

# p.plot_comp('tic', 'data_object', plot_on_background=True, clip_above_percentile=.8, clip_below_percentile=.2)
# p.plot_comp('tic', 'data_object', plot_on_background=False, clip_above_percentile=.8, clip_below_percentile=0, SNR_scale=False)
