import os
import logging

from msi_workflow.project.main import get_project, ProjectMSI
from msi_workflow.data.age_model import AgeModel
from msi_workflow.exporting.from_mcf.spectrum import MultiSectionSpectra
from msi_workflow.timeSeries.time_series import MultiSectionTimeSeries

logger = logging.getLogger(__name__)


def get_long_time_series(
        folders: list[str],
        depth_spans: list[tuple[int, int]],
        age_model: AgeModel,
        targets: list[float],
        integrate_peaks: bool = False,
        SNR_threshold: float | int = 0,
        average_by_col: str = 'classification_s',
        d_folders: list[str] | None = None,
        tolerances: float | None = None,
        path_spectra_object: str | None = None
) -> tuple[MultiSectionTimeSeries, list[ProjectMSI], MultiSectionSpectra]:
    if d_folders is None:
        d_folders = [None] * len(folders)

    readers = []
    ps = []

    # setup projects
    # this includes all steps up to setting the data_object, since the bins are
    # estimated on the MultiSectionSpectra
    for folder, depth_span, d_folder in zip(folders, depth_spans, d_folders):
        # create hdf5
        p = get_project(is_MSI=True, path_folder=folder, d_folder=d_folder)
        reader = p.get_reader()
        # set Spectra
        p.set_spectra(full=False)
        readers.append(reader)
        # set age model
        p.age_model = age_model
        p.set_depth_span(depth_span)
        p.set_age_span()

        p.set_image_handler()
        p.set_image_sample()
        p.set_image_roi()
        p.set_image_classified()

        ps.append(p)

    # initiate multi section spectra object
    specs = MultiSectionSpectra(readers)
    # perform the necessary processing steps
    if (path_spectra_object is None) or (not os.path.exists(path_spectra_object)):
        specs.full_targeted(
            readers=readers,
            targets=targets,
            integrate_peaks=integrate_peaks,
            SNR_threshold=SNR_threshold,
            tolerances=tolerances
        )
        if path_spectra_object is not None:
            specs.save(path_spectra_object)
    else:
        logger.info('Loading long spectra object ...')
        specs.load(path_spectra_object)
        specs.set_peaks()
        specs.set_kernels()
        specs.set_targets(targets=targets, tolerances=tolerances)
        specs.distribute_peaks_and_kernels()
        specs.bin_spectra(readers=readers, integrate_peaks=integrate_peaks)
        specs.filter_line_spectra(SNR_threshold=SNR_threshold)
        specs.set_feature_table(readers=readers)

    # all steps after data_object is set
    ts = []
    for p, spec in zip(ps, specs.specs):
        p.spectra = spec
        p.set_data_object()

        p.add_pixels_ROI()
        p.add_photo()
        p.add_holes()
        p.add_depth_column()
        p.add_age_column()
        p.add_light_dark_classification()
        p.add_laminae_classification()

        if SNR_threshold > 0:
            # set intensities to zero if any of the compounds is zero
            cols = p.data_object.get_data_columns()
            # test if all entries are nonzero
            mask_all_nonzero = p.data_object.feature_table.loc[:, cols].all(axis='columns')
            p.data_object.feature_table.loc[~mask_all_nonzero, cols] = 0
        # p.set_time_series(average_by_col='classification_s')

        p.set_time_series(
            average_by_col=average_by_col,
            exclude_zeros=SNR_threshold > 0  # only if filtering is active
        )
        ts.append(p.time_series)

    path_folder_ts = None
    if path_spectra_object is not None:
        path_folder_ts = os.path.dirname(path_spectra_object)

    t: MultiSectionTimeSeries = MultiSectionTimeSeries(ts, path_folder=path_folder_ts)

    return t, ps, specs
