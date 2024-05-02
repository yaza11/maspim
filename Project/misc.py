import os

from Project.cProject import Project, ProjectMSI
from exporting.from_mcf.cSpectrum import MultiSectionSpectra
from timeSeries.cTimeSeries import MultiSectionTimeSeries


def get_long_time_series(
        folders: list[str],
        depth_spans: list[tuple[int, int]],
        params_age_models: list[dict[str, str | float | int]],
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
    # this includes all steps up to setting the data_obj, since the bins are
    # estimated on the MultiSectionSpectra
    for folder, depth_span, params_age_model, d_folder in zip(folders, depth_spans, params_age_models, d_folders):
        # create hdf5
        p = Project(is_MSI=True, path_folder=folder, d_folder=d_folder)
        reader = p.create_hdf_file()
        # set Spectra
        p.set_spectra(reader, full=False)
        readers.append(reader)
        # set age model
        p.set_age_model(params_age_model['path_age_model'], sep='\t', index_col=False, load=False)
        p.age_model.add_depth_offset(params_age_model['depth_offset_age_model'])
        p.age_model.convert_depth_scale(params_age_model['conversion_to_cm_age_model'])  # convert mm to cm
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
        print('Loading long spectra object ...')
        specs.load(path_spectra_object)
        specs.distribute_peaks_and_kernels()
        specs.bin_spectra(readers=readers, integrate_peaks=integrate_peaks)
        specs.filter_line_spectra(SNR_threshold=SNR_threshold)
        specs.binned_spectra_to_df(readers=readers)

    # all steps after data_obj is set
    ts = []
    for p, spec in zip(ps, specs.specs):
        p.spectra = spec
        p.set_object()

        p.add_pixels_ROI()
        p.add_photo()
        p.add_holes()
        p.add_depth_column()
        p.add_age_column()
        p.add_light_dark_classification()
        p.add_laminae_classification()

        if SNR_threshold > 0:
            # set intensities to zero if any of the compounds is zero
            cols = p.data_obj.get_data_columns()
            # test if all entries are nonzero
            mask_all_nonzero = p.data_obj.feature_table.loc[:, cols].all(axis='columns')
            p.data_obj.feature_table.loc[~mask_all_nonzero, cols] = 0
        # p.set_time_series(average_by_col='classification_s')

        p.set_time_series(
            average_by_col=average_by_col,
            exclude_zeros=SNR_threshold > 0  # only if filtering is active
        )
        ts.append(p.time_series)

    t: MultiSectionTimeSeries = MultiSectionTimeSeries(ts)

    return t, ps, specs