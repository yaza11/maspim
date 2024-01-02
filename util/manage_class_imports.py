from res import directory_paths

import os
import pickle
import numpy as np

folder_path = directory_paths.absolute_path_to_saves


def window_to_folder(window: str):
    d = {'alkenones': 'Alkenones',
         'fa': 'FA',
         'gdgt': 'GDGT',
         'xrf': 'XRF',
         'combined': 'combined',
         'alkenones_fa': 'Alkenones_FA'
         }
    window_l = window.lower()
    return d[window_l]


def obj_type_to_folder(obj_type: str) -> str:
    obj_type_l = obj_type.lower()
    if 'transformation' in obj_type_l:
        return 'Image_transformations'
    elif 'image' in obj_type_l:
        return 'Image_objects'
    elif 'timeseries' in obj_type_l:
        return 'TimeSeries_objects'
    elif ('data' in obj_type_l) or (obj_type_l in ('msi', 'xrf')):
        return 'DataClass_objects'
    elif 'metafeatures' in obj_type_l:
        return 'MetaFeatures_objects'
    else:
        raise KeyError


def get_file_path(
        obj: object | None = None,
        section: tuple[int, int] | None = None,
        window: str | None = None,
        obj_type: str | None = None
) -> str:
    assert (obj is not None) or (all([section, window, obj_type])), 'Define either obj \
or section, window and obj_type.'
    # get class name, section and window from obj
    if obj is not None:
        obj_type = obj.__class__.__name__
        if ('_section' in obj.__dict__) and ('_window' in obj.__dict__):
            section = obj._section
            window = obj._window
        elif ('section' in obj.__dict__) and ('window' in obj.__dict__):
            section = obj.section
            window = obj.window
        else:
            raise LookupError(f'')

    folder_obj_type = obj_type_to_folder(obj_type)
    section_str = f'{section[0]}-{section[1]}'
    folder_path_file = os.path.join(
        folder_path, folder_obj_type, section_str, window_to_folder(window))
    name = f'{obj_type}_obj_{section_str}cm_{window}.pkl'
    file = os.path.join(folder_path_file, name)
    return file


def check_file_exists(**kwargs) -> bool:
    file = get_file_path(**kwargs)
    return os.path.exists(file)


def save_obj(obj: object) -> None:
    if 'Image' in obj.__class__.__name__:
        file = get_file_path(obj=obj)
    else:
        file = get_file_path(obj=obj)

    with open(file, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_obj(section: tuple[int, int], window: str, obj_type: str):
    file = get_file_path(obj=None,
                         section=section,
                         window=window,
                         obj_type=obj_type
                         )

    with open(file, 'rb') as inp:
        obj = pickle.load(inp)
    return obj


def save_params(image_from_obj, image_to_obj, data, key, overwrite):
    section = image_to_obj._section
    file_name = f'{image_from_obj._window.lower()}_to_{image_to_obj._window.lower()}_{key}.npy'
    section_str = f'{section[0]}-{section[1]}'
    file = os.path.join(
        folder_path, 'Image_transformations', section_str, file_name)
    if (not os.path.exists(file)) or overwrite:
        if key != 'T':
            np.save(file, data)
        else:
            with open(file.replace('npy', 'pkl'), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_params(image_from_obj, image_to_obj, key):
    section = image_to_obj._section
    file_name = f'{image_from_obj._window.lower()}_to_{image_to_obj._window.lower()}_{key}.npy'
    section_str = f'{section[0]}-{section[1]}'
    file = os.path.join(
        folder_path, 'Image_transformations', section_str, file_name)
    if key == 'T':
        file = file.replace('npy', 'pkl')
    if os.path.exists(file):
        print(
            f'found transformation file {key} for {image_from_obj._window.lower()}_to_{image_to_obj._window.lower()}')
        if key != 'T':
            return np.load(file)
        with open(file, 'rb') as f:
            return pickle.load(f)
    print(f'could not find file {file}')
    return None


def func_transform(
        ROI_from, ROI_to, M, pixel_coords_X, pixel_coords_X_distorted, pixel_coords_Y):
    from distorted_images import rescale
    import cv2
    # rescale to match destionation
    ROI_rescaled = rescale(
        ROI_from, ROI_to)
    # apply the warp perspective
    ROI_transformed = cv2.warpPerspective(ROI_rescaled, M, dsize=(
        ROI_to.shape[1], ROI_to.shape[0]))
    # apply the stretching
    ROI_distorted = np.zeros_like(ROI_transformed)
    ROI_distorted[pixel_coords_Y, pixel_coords_X_distorted]\
        = ROI_transformed[pixel_coords_Y, pixel_coords_X]
    return ROI_distorted


def save_combined(section, dataframe):
    file = get_file_path(section, 'combined', 'DataClass')

    with open(file, 'wb') as outp:
        pickle.dump(dataframe, outp, pickle.HIGHEST_PROTOCOL)


def get_combined(section):
    file = get_file_path(section, 'combined', 'DataClass')
    if os.path.exists(file):
        print(
            f'found combined ft for {section}')
        with open(file, 'rb') as f:
            return pickle.load(f)
    print(f'did not find combined ft for {section} in {file}, returning None')
    return None


def test_smoothing():
    dtype = 'Alkenones'
    self = load_obj((490, 495), dtype, 'Image')

    # self.x = None
    # self.y = None

    # self.flow_feature_tables.append(self.current_feature_table)

    # self.plt_comp(str(553.5328))
    # self.plt_comp(str(558.3628))

    # self.plt_top_comps_laminated(
    #     use_intensities=False, use_KL_div=True, use_successes=False,
    #     scale=False,
    #     light_or_dark='dark', remove_holes=True, N_top=10)

    # self.plt_NMF(k=2, use_repeated_NMF=True, exclude_holes=True)
    # smoothed_feature_table = self.processing_perform_smoothing(
    #     kernel_size=5, sigma=1)
    # smoothed_feature_table['classification'] = self.current_feature_table['classification'].copy()
    # self.update_flow_feature_tables(smoothed_feature_table)

    # self.plt_comp(str(553.5328))
    # self.plt_comp(str(558.3628))

    # self.plt_NMF(k=2, use_repeated_NMF=True, exclude_holes=True)

    # self.plt_top_comps_laminated(
    #     use_intensities=True, use_KL_div=True, use_successes=True,
    #     scale=True,
    #     light_or_dark='dark', remove_holes=True, N_top=10)


if __name__ == '__main__':
    print(folder_path)
    # section = (490, 495)
    # window = 'Alkenones'

    # from cMSI import MSI
    # # from cDataClass import plt_comps

    # self = MSI(section, window)
    # self.verbose = True
    # # self.sget_feature_table(use_common_mzs=True)
    # # self.perform_all_initialization_steps()
    # self.load(use_common_mzs=True)
    # self.plt_comp(553.53188756536, exclude_holes=True)
    # self.plt_comp(551.51623750122, exclude_holes=True)

    # # zone wise averaged
    # ft_seeds = self.processing_zone_wise_average(zones_key='seed')\
    #     .sort_values(by='x').reset_index(drop=True)
    # ft_seeds['L'] = self.processing_zone_wise_average(zones_key='seed', columns=['L'])\
    #     .sort_values(by='x').reset_index(drop=True).L
    # # correlations with grayscale vals
    # cL = ft_seeds.corr().L

    # import matplotlib.pyplot as plt
    # comps = ['L', '53.5328']

    # def plt_comp_and_intensity(self, comp):
    #     self.distance_pixels = None
    #     self.plt_comp(comp)
    #     plt.plot(ft_seeds[comp], ft_seeds.x - self.get_x().min()[0])
    #     plt.ylim((0, self.get_x().max()[0] - self.get_x().min()[0]))
    #     plt.gca().invert_yaxis()
    #     plt.ylabel('x ordinate')
    #     plt.xlabel('av intensity in layer')

    # plt_comp_and_intensity(self, cL.sort_values().index[-5])

    # from cTimeSeries import TimeSeries
    # import matplotlib.pyplot as plt

    # TS = TimeSeries(section, window)
    # TS.plts = False
    # TS.verbose = True
    # # TS.sget_time_series_table()
    # TS.load()
    # # TS.sget_contrasts_table()

    # TS.plt_against_grayscale(['L', 'quality'], color_seasons=True, plt_weighted=False, norm_mode='upper_lower')
    # TS.save()
    # TS.verbose = False

    # from cMSI import MSI
    # alks = MSI(section, window)
    # alks.load()
    # alks.calculate_lightdark_rankings(
    #     use_intensities=True,
    #     use_KL_div=True,
    #     use_successes=True,
    #     calc_corrs=True
    # )
    # alks.plt_PCA_rankings(
    #     sign_criteria=True, add_laminae_averages=True,
    #     columns=['corr_L'])

    # # cancel signs of contrast with light/dark sign
    # ucontrasts = TS.feature_table_contrasts.loc[:, TS.sget_data_columns()].multiply(np.sign(TS.feature_table_contrasts.seed), axis=0)
    # av_seas = ucontrasts.mean(axis=0)
    # # weigh contrasts by quality
    # q = TS.feature_table_zone_averages.quality / TS.feature_table_zone_averages.quality.max()
    # c = TS.feature_table_zone_averages.contrast / TS.feature_table_zone_averages.contrast.max()
    # ucontrasts_q = ucontrasts.multiply(q, axis=0)
    # ucontrasts_q['x_ROI'] = TS.feature_table_zone_averages.x_ROI

    # ft_contrast = self.calculate_contrasts_simplified_laminae()

    # self.get_existing_feature_table_paths()
    # del self.feature_table_paths_dict
    # self.get_existing_feature_table_paths()

    # self.set_params_laminae_simplified(peak_prominence=.5, downscale_factor=1 / 4, max_slope=.5)

    # self.get_preprocessed_image_for_classification(image_gray=self._image_original.copy())
    # img_c = self.get_classification_adaptive_mean(self._image_original.copy())
    # plt_cv2_image(img_c)

    # self.resolve_intersects_by_quality(maxiter=3)

    # self.set_laminae_seeds(peak_prominence=.2, in_classification=True)
    # self.simplify_laminae(0.2, height0='use_peak_widths')
    # self.set_quality_score()

    # self.add_seed_classification()
    # self.add_simplified_laminae_classification()

    # from cDataClass import plt_comps
    # plt_comps(self.current_feature_table, ['553.5328', 'classification_s', 'classification'])

    # self.add_simplified_laminae_classification()
    # self.add_seed_classification()
    # self.plt_contrasts()
    # self.calculate_lightdark_rankings(use_intensities=True, use_KL_div=True, use_successes=True, calc_corrs=True)
    # self.plt_PCA_rankings(
    #     sign_criteria=False,
    #     add_seasonality=True,
    #     columns=['score', 'KL_div', 'corr_L', 'intensity_div'])

    # self.set_quality_score()

    # self.set_laminae_params_table(peak_prominence=0.2)

    # self.add_layer_classification()

    # self.laminae_seeds(peak_prominence=.2)
    # self.simplify_laminae(peak_prominence=.2)
    # save_obj(self)
    # self.create_simplified_laminae_classification()

    # self.flow_feature_tables.append(self.current_feature_table)
    # smoothed_feature_table = self.processing_perform_smoothing(
    #     kernel_size=5, sigma=1)
    # smoothed_feature_table['classification'] = self.current_feature_table['classification'].copy()
    # smoothed_feature_table['L'] = self.current_feature_table['L'].copy()
    # self.update_flow_feature_tables(smoothed_feature_table)

    # classification_column = 'classification'

    # sign_to_val = {-1: 127, 0: 0, 1: 255}
    # self.current_feature_table[classification_column] = \
    #     self.current_feature_table.apply(
    #         lambda row: sign_to_val[np.sign(row.seed)], axis=1)

    # self.calculate_lightdark_rankings(
    #     use_intensities=True, use_successes=True, use_KL_div=True,
    #     calc_corrs=True, scale=True, classification_column=classification_column
    # )

    # self.plt_PCA_rankings(
    #     sign_criteria=False,
    #     columns=['intensity_div', 'density_nonzero', 'KL_div', 'score',
    #              'corr_L', f'corr_{classification_column}']
    # )
