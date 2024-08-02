from cTimeSeries import TimeSeries
from cImage import ImageProbe
from cMSI import MSI
from cXRF import XRF
from cDataClass import Data, combine_sections
import cClass
from cMSI import MSI
import constants
from constants import mC37_2, sections_all, windows_all, YD_transition_depth

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats


def test_(section, window):
    if window.lower() != 'xrf':
        DC = MSI(section, window)
    else:
        DC = XRF(section, window)
    section_str = f'{section[0]}-{section[1]}'
    # ft_path = rf'E:/Master_Thesis/raw_feature_tables/{section_str}/FA/feature_table_{section_str}_FA_mz388-481_0dot1.csv'
    # DC.load(ft_path=ft_path)
    DC.load()
    # print(len(DC.current_feature_table.columns))
    DC.plot_comp('classification', exclude_holes=False)
    DC.plot_comp(443.136)
    # DC.plts = True
    # DC.sget_photo_ROI()
    # DC.plot_comp('classification')
    # DC.plot_comp('classification_s')
    # DC.plot_comp('seed')
    return DC


# windows = ['FA']
# sections = sections_all
# for section in sections:
#     for window in windows:
#         print(section, window)

#         section_str = f'{section[0]}-{section[1]}'
    # DC = MSI(section, window)
    # DC.verbose = True
    # DC.plts = True
    # # DC.load(use_common_mzs=True)
    # DC.current_feature_table = DC.load_feature_table(key=.1)
    # DC.sget_photo_ROI()
    # DC.combine_photo_feature_table()
    # DC._pixels_get_photo_ROI_to_ROI()
    # DC.add_graylevel_from_data_frame()
    # DC.add_laminae_classification()
    # DC.add_seed_classification()
    # DC.add_simplified_laminae_classification()
    # DC.plot_comp('classification')
    # DC.save()
    # test_(section, window)
#         # try:
#         #     if window.lower() != 'xrf':
#         #         DC = MSI(section, window)
#         #     else:
#         #         DC = XRF(section, window)
#         #     DC.verbose = True
#         #     DC.perform_all_initialization_steps()
#         #     del DC
#         # except Exception as e:
#         #     print(e)


def create_combined_FA():
    window = 'FA'
    combined_section = (sections_all[0][0], sections_all[-1][-1])
    MSI_combined = MSI(combined_section, window)
    fts = []
    x_ROI_offset = 0
    x_offset = 0
    for section in sections_all:
        print(section)
        section_str = f'{section[0]}-{section[1]}'
        ft_path = rf'E:/Master_Thesis/raw_feature_tables/{section_str}/FA/feature_table_{section_str}_FA_mz388-481_0dot1.csv'
        M = MSI(section, window)
        M.verbose = True
        M.load(ft_path=ft_path)
        M.current_feature_table['x_ROI'] += x_ROI_offset
        M.current_feature_table['x'] += x_offset
        M.current_feature_table['y'] -= M.current_feature_table.y.min()
        M.current_feature_table['seed'] = \
            (M.current_feature_table.seed.abs() + x_ROI_offset) * \
            np.sign(M.current_feature_table.seed)
        fts.append(M.current_feature_table)
        I = ImageProbe(section, window)
        I.load()
        x_ROI_offset += I.xywh_ROI[2]
        x_offset = M.current_feature_table.x.max()
        del I
        del M

    MSI_combined.current_feature_table = pd.concat(fts, axis=0).reset_index(drop=True)
    # save the data
    try:
        print('starting save')
        MSI_combined.current_feature_table.loc[:, list(MSI_combined.get_data_columns()) + ['x', 'y']].to_csv('E:/Master_Thesis/raw_feature_tables/combined/FA/feature_table_combined490-520_FA_mz388-481_0dot1.csv')
        MSI_combined.save()
        print('DONE!')
    except Exception as e:
        print(e)


def find_gray_mode(obj):
    L = obj.current_feature_table.L.loc[obj.nmf_xy.index]
    W = pd.DataFrame(obj.W, index=obj.nmf_xy.index)
    return W.corrwith(L)


def significance_seasonality(hold=False):
    # sections = sections_all
    # m = combine_sections(sections, 'Alkenones')
    # m.plot_comp('L')
    self = MSI((490, 510), 'FA')
    self.load()

    # create feature table with same intensity distribution but in ranom places
    data = self.data()

    def resample(df):
        # shuffle rows
        df = df.sample(frac=1, ignore_index=True)
        return df

    N_runs = 100

    seas = np.zeros((N_runs, len(self.get_data_columns())))

    TS = TimeSeries((490, 510), 'FA')
    TS.load()

    # for i in range(N_runs):
    #     print(round(i / N_runs * 100))
    #     self.current_feature_table.loc[:, self.get_data_columns()] = resample(data)
    #     ft_seeds_avg, ft_seeds_std, ft_seeds_success = self.processing_zone_wise_average(
    #         zones_key='seed',
    #         columns=self.get_data_columns(),
    #         correct_zeros=False,
    #         calc_std=True
    #     )
    #     ft_seeds_avg = ft_seeds_avg.fillna(0).reset_index(drop=True)
    #     ft_seeds_success = ft_seeds_success.reset_index(drop=True)

    #     TS.feature_table_zone_averages.loc[:, TS.get_data_columns()] = ft_seeds_avg
    #     TS.feature_table_zone_successes.loc[:, TS.get_data_columns()] = ft_seeds_success
    #     seas[i, :] = TS.get_seasonalities()
    # np.save('seas_FA_random100.npy', seas)
    seas = np.load('seas_FA_random100.npy')

    # calculate distance from mean
    means = np.nanmean(seas, axis=0)
    stds = np.nanstd(seas, axis=0)

    TS.load()
    sea = TS.get_seasonalities()
    n = len(sea)

    # factors = np.abs(means - sea) / stds
    SE = stds / np.sqrt(n)  # standard error
    factors = np.abs(means - sea) / SE  # standard score

    mask_both_zero = (stds == 0) & (sea == 0)
    print(f'excluding {sum(mask_both_zero)} compounds in SE plot')
    factors[mask_both_zero] = 0

    # t-test pval
    alpha = 5e-2 / n
    p = stats.ttest_1samp(seas, sea, axis=0).pvalue
    x = sea.abs()
    y = factors
    fig1 = plt.figure()
    plt.loglog(x, y, 'o', markersize=3)
    plt.xlabel('|seasonality|')
    plt.ylabel('standard score')
    for i, ls in zip((1, 2, 3), ('-', '--', ':')):
        plt.hlines(
            i,
            xmin=np.nanmin(x),
            xmax=np.nanmax(x),
            colors='k',
            linestyles=ls,
            label=fr'{i} SE'
        )
    plt.legend()
    plt.title('standard score vs seasonality strength')
    if not hold:
        plt.show()

    # p-val has to be lower than alpha to be considered significant (probability of occuring randomly below alpha)
    mask_insiginificant = p >= alpha
    fig2 = plt.figure()
    print(sum(x == 0), sum(mask_insiginificant))
    plt.loglog(x[mask_insiginificant], p[mask_insiginificant], 'ro', markersize=3, label=r'$H_0$ accepted (random)')
    plt.loglog(x[~mask_insiginificant], p[~mask_insiginificant], 'o', markersize=3, label=r'$H_0$ rejcted (not random)')
    plt.xlabel('|seasonality|')
    plt.ylabel('p-value')
    plt.hlines(
        alpha,
        xmin=np.nanmin(x),
        xmax=np.nanmax(x),
        colors='k',
        label=r' $n$ corrected $\alpha$'
    )
    plt.legend()
    plt.title('T-test against randomness')
    if not hold:
        plt.show()
    else:
        return fig1, fig2


if __name__ == '__main__':
    # significance_seasonality()

    a = MSI((490, 510), 'FA')
    a.load()
    a = a.split_at_depth(494.99)[0]
    a.plot_comp(mC37_2)

    a.estimate_nmf(k=5)

    # %%
    fig, axs = a.plot_nmf(k=5, hold=True)

    for ax in list(axs[:, 0]):
        ax.set_axis_off()

    for ax in list(axs[:, 1]):
        ax.axes.get_yaxis().set_visible(False)

    axs[-1, -1].set_xlabel('Da')

    plt.savefig(
        os.path.join(
            folder_thesis_images,
            '490-495_FA_NMF_k=5_repeated=False.png'
        ),
        dpi=low_res
    )
    plt.show()

    plt.show()

    # %%
    # figl, axsl = a.plt_top_comps_laminated(
    #     N_top=5, remove_holes=True, use_intensities=True, classification_column='classification_s',
    #     use_successes=True, use_KL_div=True, scale=True, hold=True, figsize=(6, 7))

    # axsl[0].set_ylabel(r'depth (cm)')
    # N_ticks = 11
    # axsl[0].set_yticks(np.linspace(0, 2000, N_ticks), np.linspace(0, 20, N_ticks).astype(int))
    # for ax in axsl:
    #     ax.set_xticks([])

    # figd, axsd = a.plt_top_comps_laminated(
    #     N_top=5, remove_holes=True, use_intensities=True, classification_column='classification_s',
    #     use_successes=True, use_KL_div=True, light_or_dark='dark', scale=True, hold=True, figsize=(6, 7))

    # axsd[0].set_ylabel(r'depth (cm)')
    # N_ticks = 11
    # axsd[0].set_yticks(np.linspace(0, 2000, N_ticks), np.linspace(0, 20, N_ticks).astype(int))
    # for ax in axsd:
    #     ax.set_xticks([])

    # a.plot_comp(mC37_2, exclude_holes=False)
    # print(
    #       'classified pixels in % (total)):', (
    #           1 - (
    #               a.current_feature_table.classification_s.isna()
    #           ).mean()
    #       ) * 100
    # )

    # print(
    #       'classified pixels in % (non-hole)):', (
    #           1 - (
    #               a.current_feature_table.classification_s.isna() & (a.current_feature_table.classification > 0)
    #           ).mean()
    #       ) * 100
    # )

    # a.calculate_lightdark_rankings(use_intensities=True, use_successes=True, use_KL_div=True)

    # a.plt_PCA_rankings(
    #     columns=['score', 'KL_div', 'density_nonzero', 'intensity_div'],
    #     add_annotations=False,
    #     add_laminae_averages=True,
    #     title='Biplot of metrices in alkenones window for seasonality score \n based on light/dark classifications',
    #     hold=True
    # )

    # b = MSI((490, 510), 'FA')
    # b.load()

    # print(
    #       'classified pixels in % (total)):', (
    #           1 - (
    #               b.current_feature_table.classification_s.isna()
    #           ).mean()
    #       ) * 100
    # )

    # print(
    #       'classified pixels in % (non-hole)):', (
    #           1 - (
    #               b.current_feature_table.classification_s.isna() & (b.current_feature_table.classification > 0)
    #           ).mean()
    #       ) * 100
    # )

    # a.load(use_common_mzs=True)

    # cac0 = a.get_data_columns()

    # a.plot_comp('L')

    # b = MSI((505, 510), 'Alkenones')

    # b.load(ft_path=r'E:/Master_Thesis/raw_feature_tables/extra tables/FA_dirty/feature_table_505-510_FA_mz388-481_0dot1.csv')

    # with open(r'E:\Master_Thesis\raw_feature_tables\490-520\FA\ref_peaks_after.pickle', 'rb') as f:
    #     ref_peaks_490_500 = pickle.load(f)

    # with open(r'E:\Master_Thesis\raw_feature_tables\490-520\FA\ref_peaks_before.pickle', 'rb') as f:
    #     ref_peaks_505_520 = pickle.load(f)

    # cols_drop_upper = list(set(a.get_data_columns()).difference(set(ref_peaks_490_500.astype(str))))
    # a.current_feature_table = a.current_feature_table.drop(columns=cols_drop_upper)

    # cols_drop_lower = list(set(b.get_data_columns()).difference(set(ref_peaks_505_520.astype(str))))
    # b.current_feature_table = b.current_feature_table.drop(columns=cols_drop_lower)

    # m = a.get_data_mean()
    # n = b.get_data_mean()

    # plt.stem(m.index.astype(float), m.values, label='490-495', markerfmt='', linefmt='C0')
    # plt.stem(n.index.astype(float), n.values, label='505-510', markerfmt='', linefmt='C1')
    # plt.legend()

    # # orange, blue
    # pairs = np.array([[390.37040, 390.37150],
    #                   [422.39620, 422.39831],
    #                   [441.3704, 441.372502],
    #                   [472.2898, 472.291899]])

    # self = m
    # self.processing_perform_smoothing(kernel_size=5)
    # self.current_feature_table = self.smoothed_feature_table
    # self.estimate_nmf(k=5, use_repeated_NMF=True, N_rep=10)
    # self.plot_nmf(k=5)
    # find_gray_mode(self)

    # m.plot_comp('classification', exclude_holes=False)

    # u, l = m.split_at_depth(YD_transition_depth)
    # u.plot_comp('classification', exclude_holes=False)
    # l.plot_comp('classification', exclude_holes=False)
    pass
