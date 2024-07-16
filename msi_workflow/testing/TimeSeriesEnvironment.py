from cTimeSeries import TimeSeries
from cTimeSeries import combine_sections as combine_sections_TS
from cProxy import RatioProxy
from constants import sections_all, mC37_2, mC37_3, transformation_target, windows_all, YD_transition, YD_transition_depth
from cMSI import MSI
from cDataClass import combine_sections as combine_sections_MSI
from cDataClass import Data, plt_comps
from cImage import ImageProbe
# from clean_FT_FA import clean_section

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt


def plt_PCA(TS, pca, pcs, N_top=3, title_appendix='', **kwargs):
    index_loadings = TS.get_data_columns()
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC' + str(i) for i in range(np.shape(pcs)[1])],
        index=index_loadings)

    top_compounds = pca.explained_variance_ratio_.cumsum()
    plt.figure()
    plt.plot(top_compounds)
    plt.grid('on')
    plt.xlabel('modes')
    plt.ylabel('cumulative explained variance')
    plt.show()

    fig, axs = plt.subplots(nrows=N_top,
                            ncols=2,
                            sharex='col',
                            figsize=(20, 20))
    # df with x, y like original FT
    x = TS.age_scale()
    for i in range(N_top):
        axs[i, 0].plot(x, pcs[:, i])
        if TS._data_type == 'xrf':
            labels = list(loadings.index)
        else:
            labels = [float(loading) for loading in loadings.index]
        axs[i, 1].stem(labels,
                       loadings['PC' + str(i)],
                       markerfmt=' ')

    title = f'PCA on {TS._section}cm, {TS._window} {title_appendix}'
    fig.suptitle(title)
    plt.tight_layout()


def pca_on_time_series(TS):
    # pca on table
    ft = TS.get_contrasts_table().loc[:, TS.get_data_columns()]
    assert not np.any(np.isnan(ft)), 'ft contains nans'

    # scale
    ft_s = StandardScaler().fit_transform(ft)

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(ft_s)

    plt_PCA(TS, pca, pcs)


def find_closests(values: np.ndarray, targets: np.ndarray):
    assert np.all(~np.isnan(values)), 'values contain nan'
    assert np.all(~np.isnan(targets)), 'targets contain nan'
    # Broadcasting to create a 2D array of absolute differences
    abs_diffs = np.abs(values[:, np.newaxis] - targets)

    # Finding the index of the closest value for each target
    closest_indices = np.argmin(abs_diffs, axis=0)

    # Retrieving the closest values for each target
    closest_values = values[closest_indices]
    return closest_values


def overwrite_seed(TS):
    """Overwrite seed with sign(seed) * x_ROI."""
    new = (np.sign(TS.feature_table_zone_averages.seed)
           * TS.feature_table_zone_averages.x_ROI
           ).copy()
    TS.feature_table_zone_averages['seed'] = new
    TS.feature_table_zone_successes['seed'] = new
    if 'feature_table_standard_deviations' in TS.__dict__:
        TS.feature_table_zone_successes['seed'] = new
    return TS


def combine_windows(section):
    window = transformation_target

    TS = TimeSeries(section, window)
    TS.set_time_series_tables(use_common_mzs=True)

    TS = overwrite_seed(TS)

    windows = windows_all.copy()
    if window in windows:
        windows.remove(window)

    for window in windows:
        print(section, window)
        ts = TimeSeries(section, window)
        ts.set_time_series_tables(use_common_mzs=True)

        cols = ts.get_data_columns() + ['seed']

        ts.correct_distortion()
        ts = overwrite_seed(ts)
        # find corresponding layer
        seed_new = find_closests(
            TS.feature_table_zone_averages.seed.to_numpy(),
            ts.feature_table_zone_averages.seed.to_numpy()
        )
        ts.feature_table_zone_averages['seed'] = seed_new
        ts.feature_table_zone_successes['seed'] = seed_new

        # merge with TS
        TS.feature_table_zone_averages = TS.feature_table_zone_averages.merge(
            ts.feature_table_zone_averages.loc[:, cols],
            on='seed',
            how='left'
        )
        TS.feature_table_zone_successes = TS.feature_table_zone_successes.merge(
            ts.feature_table_zone_successes.loc[:, cols],
            on='seed',
            how='left'
        )

    plt.imshow(TS.feature_table_zone_averages > 0)
    plt.show()

    TS._window = 'combined'
    return TS


def combine_sections_and_windows():
    from cImage import ImageProbe
    combined_section = (sections_all[0][0], sections_all[-1][-1])
    TS_combined = TimeSeries(combined_section, 'Alkenones')
    avs: list[pd.DataFrame] = []
    sucs: list[pd.DataFrame] = []
    x_ROI_offset: float = 0
    for section in sections_all:
        # add offset to x_ROI
        TS = combine_windows(section)
        # add offsets to x_ROI and seed
        av = TS.feature_table_zone_averages.copy()
        av['x_ROI'] += x_ROI_offset
        av['seed'] = (av.seed.abs() + x_ROI_offset) * np.sign(av.seed)
        avs.append(av)
        sucs.append(TS.get_feature_table_zone_successes().copy())
        I = ImageProbe(section, transformation_target)
        I.load()
        x_ROI_offset += I.xywh_ROI[2]
        del TS

    TS_combined.feature_table_zone_averages = pd.concat(avs, axis=0).reset_index(drop=True)
    TS_combined.feature_table_zone_successes = pd.concat(sucs, axis=0).reset_index(drop=True)

    TS_combined.feature_table_zone_successes['seed'] = TS_combined.feature_table_zone_averages.seed.copy()

    TS_combined.feature_table_zone_averages

    TS_combined._window = 'combined'

    plt.imshow(TS_combined.feature_table_zone_averages > 0)
    plt.show()

    return TS_combined


def combine_sections_FA():
    """Combine time series of multiple sections."""
    window = 'FA'
    from cImage import ImageProbe
    combined_section = (sections_all[0][0], sections_all[-1][-1])
    TS_combined = TimeSeries(combined_section, window)
    avs = []
    stds = []
    sucs = []
    x_ROI_offset = 0
    for section in sections_all:
        section_str = f'{section[0]}-{section[1]}'
        print(section_str)
        ft_path = rf'E:/Master_Thesis/raw_feature_tables/{section_str}/FA/feature_table_{section_str}_FA_mz388-481_0dot1.csv'
        # add offset to x_ROI
        TS = TimeSeries(section, window)
        TS.verbose = True
        TS.set_time_series_tables(ft_path=ft_path)
        av = TS.get_feature_table_zone_averages().copy()
        # add offsets to x_ROI and seed
        av['x_ROI'] += x_ROI_offset
        av['seed'] = (av.seed.abs() + x_ROI_offset) * np.sign(av.seed)
        avs.append(av)
        stds.append(TS.get_feature_table_zone_standard_deviations().copy())
        sucs.append(TS.get_feature_table_zone_successes().copy())
        I = ImageProbe(section, window)
        I.load()
        x_ROI_offset += I.xywh_ROI[2]

    TS_combined.feature_table_zone_averages = pd.concat(avs, axis=0).reset_index(drop=True)
    TS_combined.feature_table_zone_standard_deviations = pd.concat(stds, axis=0).reset_index(drop=True)
    TS_combined.feature_table_zone_successes = pd.concat(sucs, axis=0).reset_index(drop=True)

    TS_combined.feature_table_zone_standard_deviations['seed'] = TS_combined.feature_table_zone_averages.seed.copy()
    TS_combined.feature_table_zone_successes['seed'] = TS_combined.feature_table_zone_averages.seed.copy()

    TS_combined.feature_table_zone_averages

    return TS_combined


def get_FA_TS():
    window = 'FA'
    combined_section = (sections_all[0][0], sections_all[-1][-1])
    TS_combined = TimeSeries(combined_section, window)
    TS_combined.load()
    TS_combined.feature_table_zone_averages_clean, TS_combined.feature_table_zone_successes_clean = TS_combined.combine_duplicate_seed()

    return TS_combined


def combine_sections_FA_clean():
    """Combine time series of multiple sections."""
    window = 'FA'
    from cImage import ImageProbe
    combined_section = (sections_all[0][0], sections_all[-1][-1])
    TS_combined = TimeSeries(combined_section, window)
    avs = []
    stds = []
    sucs = []
    x_ROI_offset = 0
    for section in sections_all:
        section_str = f'{section[0]}-{section[1]}'
        print(section_str)
        # add offset to x_ROI
        TS = TimeSeries(section, window)
        TS.verbose = True
        fa_clean = clean_section(section)
        TS.set_time_series_tables(Data_obj=fa_clean)
        av = TS.get_feature_table_zone_averages().copy()
        # add offsets to x_ROI and seed
        av['x_ROI'] += x_ROI_offset
        av['seed'] = (av.seed.abs() + x_ROI_offset) * np.sign(av.seed)
        avs.append(av)
        stds.append(TS.get_feature_table_zone_standard_deviations().copy())
        sucs.append(TS.get_feature_table_zone_successes().copy())
        I = ImageProbe(section, window)
        I.load()
        x_ROI_offset += I.xywh_ROI[2]
        del fa_clean, I

    TS_combined.feature_table_zone_averages = pd.concat(avs, axis=0).reset_index(drop=True)
    TS_combined.feature_table_zone_standard_deviations = pd.concat(stds, axis=0).reset_index(drop=True)
    TS_combined.feature_table_zone_successes = pd.concat(sucs, axis=0).reset_index(drop=True)

    TS_combined.feature_table_zone_standard_deviations['seed'] = TS_combined.feature_table_zone_averages.seed.copy()
    TS_combined.feature_table_zone_successes['seed'] = TS_combined.feature_table_zone_averages.seed.copy()

    TS_combined.feature_table_zone_averages

    return TS_combined


def contrast(l, c, r):
    b = np.mean([l, r])
    return (c - b) / (c + b)


if __name__ == '__main__':
    window = 'FA'
    sections = [(490, 495), (495, 500), (500, 505), (505, 510)]
    # TS = combine_sections_TS(sections, window)
    # TS.save()
    TS = TimeSeries((490, 510), window)
    TS.load()

    # TS.feature_table_zone_successes['1'] = 100
    # TS.feature_table_zone_successes['2'] = 100
    # TS.feature_table_zone_successes['0'] = 100
    # TS.feature_table_zone_averages['1'] = 127.5 * (1 + np.sign(TS.feature_table_zone_averages.contrast))
    # TS.feature_table_zone_averages['2'] = 127.5 * (1 - np.sign(TS.feature_table_zone_averages.contrast))
    # TS.feature_table_zone_averages['0'] = 127.5
    # # TS.plt_against_grayscale(['1', '2', '0'], plt_contrasts=True)
    # TS.plt_against_grayscale(['1'], plt_contrasts=True)
    # s = TS.get_seasonalities(norm_weights=False, exclude_low_success=False)
    # print(s.loc[['1', '2', '0']])
