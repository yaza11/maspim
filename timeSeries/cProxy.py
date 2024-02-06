from res.constants import (
    mC37_2, mC37_3, sections_all, n_successes_required,
    window_to_type, distance_pixels, YD_transition, elements
)
from data.cMSI import MSI
from data.cDataClass import combine_sections
from imaging.main.cImage import ImageProbe
from timeSeries.cTimeSeries import TimeSeries

import re
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_feature_table_for_UKp():
    section = (490, 510)
    window = 'Alkenones'
    # get path
    m = MSI(section, window)
    m.load()
    # # inject FT in MSI obj
    ft = m.get_nondata().copy()
    m.current_feature_table = pd.read_csv(r'E:/Master_Thesis/raw_feature_tables/490-510/UK37/feature_table_490-510_Uk37.csv', index_col=0)

    # return m


def combine_sections(sections, **kwargs):
    """Combine time series of multiple sections."""
    window = 'Alkenones'
    from imaging.main.cImage import ImageProbe
    combined_section = (sections[0][0], sections[-1][-1])
    TS_combined = TimeSeries(combined_section, window)
    avs = []
    stds = []
    sucs = []
    x_ROI_offset = 0
    for section in sections:
        print(section)
        # add offset to x_ROI
        TS = TimeSeries(section, window)
        TS.set_time_series_tables(Data_obj=get_feature_table_for_UKp(section), **kwargs)
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


class UK37(TimeSeries):
    def __init__(self) -> None:
        """Initialize."""
        self.plts = False
        self.verbose = False

        self._data_type = 'msi'
        self._section = (sections_all[0][0], sections_all[-1][-1])
        self._window = 'Alkenones'
        self.distance_pixels = distance_pixels[self._data_type]

    def set_time_series_tables(self, correct_zeros=False, **kwargs):
        TS = combine_sections(sections_all, correct_zeros=correct_zeros, **kwargs)
        self.feature_table_zone_averages = TS.feature_table_zone_averages
        self.feature_table_zone_standard_deviations = TS.feature_table_zone_standard_deviations
        self.feature_table_zone_successes = TS.feature_table_zone_successes

    def get_table_w_N(self):
        columns = [
            'seed', str(mC37_2), str(mC37_3),
            'x_ROI', 'N_total', 'N_C37:2', 'N_C37:3'
        ]

        table = np.zeros(
            (self.get_feature_table_zone_averages().shape[0], len(columns))
        )
        table = pd.DataFrame(data=table, columns=columns)
        table['seed'] = self.get_feature_table_zone_averages().seed
        table[str(mC37_2)] = self.get_feature_table_zone_averages().loc[:, str(mC37_2)]
        table[str(mC37_3)] = self.get_feature_table_zone_averages().loc[:, str(mC37_3)]
        table['x_ROI'] = self.get_feature_table_zone_averages().x_ROI
        table['N_total'] = self.get_feature_table_zone_successes().N_total
        table['N_C37:2'] = self.get_feature_table_zone_successes()[str(mC37_2)]
        table['N_C37:3'] = self.get_feature_table_zone_successes()[str(mC37_3)]
        return table

    def combine_layers(self, table=None, diff_sign_condition=0):
        if table is None:
            table = self.get_table_w_N()

        s = np.sign(table.seed)
        mask_non_separated = np.diff(s) == diff_sign_condition

        # initiate new table
        combined_table = np.zeros(table.shape)
        combined_table = pd.DataFrame(data=combined_table, columns=table.columns)
        counter = 0
        idx_layer = 0
        while idx_layer < s.shape[0]:
            # skip check for the last entry
            if (idx_layer < s.shape[0] - 1) and mask_non_separated[idx_layer]:
                # weights for C
                n_upper_C = table.loc[idx_layer, str(mC37_2)]
                n_lower_C = table.loc[idx_layer + 1, str(mC37_2)]
                N_C = n_upper_C + n_lower_C
                if N_C == 0:
                    weight_upper_C = 0
                    weight_lower_C = 0
                else:
                    weight_upper_C = n_upper_C / N_C
                    weight_lower_C = n_lower_C / N_C

                # weights for seed and ROI
                n_upper_G = table.loc[idx_layer, 'N_total']
                n_lower_G = table.loc[idx_layer + 1, 'N_total']
                N_G = n_upper_G + n_lower_G
                if N_G == 0:
                    idx_layer += 1
                    continue
                else:
                    weight_upper_G = n_upper_G / N_G
                    weight_lower_G = n_lower_G / N_G

                # add up Ns
                combined_table.loc[counter, 'N_total'] = \
                    table.loc[idx_layer, 'N_total'] + \
                    table.loc[idx_layer + 1, 'N_total']
                combined_table.loc[counter, 'N_C37:2'] = \
                    table.loc[idx_layer, 'N_C37:2'] + \
                    table.loc[idx_layer + 1, 'N_C37:2']
                combined_table.loc[counter, 'N_C37:3'] = \
                    table.loc[idx_layer, 'N_C37:3'] + \
                    table.loc[idx_layer + 1, 'N_C37:3']
                # take weighted average
                combined_table.loc[counter, str(mC37_2)] = \
                    weight_upper_C * table.loc[idx_layer, str(mC37_2)] + \
                    weight_lower_C * table.loc[idx_layer + 1, str(mC37_2)]
                combined_table.loc[counter, str(mC37_3)] = \
                    weight_upper_C * table.loc[idx_layer, str(mC37_3)] + \
                    weight_lower_C * table.loc[idx_layer + 1, str(mC37_3)]
                combined_table.loc[counter, 'x_ROI'] = \
                    weight_upper_G * table.loc[idx_layer, 'x_ROI'] + \
                    weight_lower_G * table.loc[idx_layer + 1, 'x_ROI']
                if diff_sign_condition == 0:
                    combined_table.loc[counter, 'seed'] = round(
                        weight_upper_G * table.loc[idx_layer, 'seed'] +
                        weight_lower_G * table.loc[idx_layer + 1, 'seed']
                    )
                else:
                    combined_table.loc[counter, 'seed'] = round(
                        weight_upper_G * np.abs(table.loc[idx_layer, 'seed']) +
                        weight_lower_G * np.abs(table.loc[idx_layer + 1, 'seed'])
                    )
                idx_layer += 1
            else:
                combined_table.loc[counter, 'N_total'] = table.loc[idx_layer, 'N_total']
                combined_table.loc[counter, 'N_C37:2'] = table.loc[idx_layer, 'N_C37:2']
                combined_table.loc[counter, 'N_C37:3'] = table.loc[idx_layer, 'N_C37:3']

                combined_table.loc[counter, str(mC37_2)] = table.loc[idx_layer, str(mC37_2)]
                combined_table.loc[counter, str(mC37_3)] = table.loc[idx_layer, str(mC37_3)]
                combined_table.loc[counter, 'x_ROI'] = table.loc[idx_layer, 'x_ROI']
                combined_table.loc[counter, 'seed'] = table.loc[idx_layer, 'seed']
            counter += 1
            idx_layer += 1
        # drop empty rows
        combined_table = combined_table.loc[(combined_table != 0).any(axis=1)]
        if np.any(np.diff(np.sign(combined_table.seed)) == 0) and (diff_sign_condition == 0):
            print('reiterating ...')
            combined_table = self.combine_layers(combined_table)
        return combined_table

    def add_UK_proxy(self, table=None, corrected=True):
        if table is None:
            table = self.get_table_w_N()
        # calculate UK37 proxy
        C37_2 = table[str(mC37_2)]
        C37_3 = table[str(mC37_3)]

        UK37p = C37_2 / (C37_2 + C37_3)
        if corrected:
            UK37p *= 1.194

        table['UK37p'] = UK37p

        return table

    def add_SST(self, table=None, method='prahl', **kwargs):
        if table is None:
            table = self.get_table_w_N()
        # water temperature in degrees Celsius
        uks = table['UK37p'].fillna(0)
        if method == 'prahl':
            def SST_prahl(UK37p):
                return 29.41 * UK37p - 1.15
            table['SST'] = SST_prahl(uks)
        elif method == 'BAYSPLINE':
            import bayspline
            if 'prior_std' in kwargs:
                prior_std = kwargs['prior_std']
            else:
                prior_std = 10
            prediction = bayspline.predict_sst(uks, prior_std=prior_std)
            table['SST'] = prediction.percentile(q=50)
        else:
            raise NotImplementedError()

        return table


def plt_UK37():
    p = UK37()
    p.set_time_series_tables(correct_zeros=False, exclude_zeros=True)
    c = p.combine_layers()
    a = p.combine_layers(c, diff_sign_condition=np.diff(np.sign(c.seed))[0])
    a = p.add_UK_proxy(a, corrected=True)
    a = p.add_SST(a, method='BAYSPLINE')

    mask_success = a.SST > 0
    x = p.age_scale(x=a.x_ROI)[mask_success]
    y = a.SST[mask_success]

    plt.plot(x, y, color='red', alpha=.5)
    plt.plot(x, scipy.ndimage.gaussian_filter1d(y, sigma=1), color='black')
    plt.vlines(11_700, 0, 35, linestyles='solid', alpha=.75, label='Pl-H boundary', color='black', linewidth=2)
    plt.ylim((19, 31))
    plt.grid('on')


class RatioProxy(TimeSeries):
    """Construct proxy as ratio of compounds in existent feature table."""

    def __init__(
            self,
            TS: TimeSeries,
            mz_a: float | str,
            mz_b: float | str
    ) -> None:
        """Initialize."""
        l = [float(mz_a), float(mz_b)]
        self.mz_a = np.min(l)
        self.mz_b = np.max(l)

        self.copy_attributes(TS)
        self.add_proxy(valid_spectra_mode='both_above')

    def copy_attributes(self, TS):
        """Inherit attributes from TS, reduce ft to relevant columns."""
        for k, v in TS.__dict__.items():
            if 'feature_table' in k:  # is a feature table
                # drop all irrelevant data columns
                cols_to_drop = TS.get_data_columns().copy()
                cols_to_drop.remove(str(self.mz_a))
                cols_to_drop.remove(str(self.mz_b))
                v = v.drop(columns=cols_to_drop)
            self.__setattr__(k, v)

    def get_data_columns(self):
        """Modified data columns getter where ratio is data column."""
        columns = self.get_feature_table_zone_averages().columns
        columns_valid = []
        columns_xrf = [col for col in columns if
                       col in list(elements.Abbreviation)]
        columns_msi = [col for col in columns if str(
            col).replace('.', '', 1).isdigit()]
        columns_valid = columns_xrf + columns_msi
        if 'ratio' in self.get_feature_table_zone_averages().columns:
            columns_valid.append('ratio')
        return columns_valid

    def add_proxy(
            self, valid_spectra_mode: str, n_threshold: int | None = None
    ):
        """Add relative ratio proxy."""
        assert valid_spectra_mode in {'all_spectra', 'both_above', 'any_above', 'a_above', 'b_above'}, \
            f"succes_mode must be one of 'all_spectra', 'both_above', 'one_above', not {valid_spectra_mode}"

        if n_threshold is None:
            n_threshold = n_successes_required

        succ_a = self.feature_table_zone_successes[str(self.mz_a)]
        succ_b = self.feature_table_zone_successes[str(self.mz_b)]
        mask_a_valid = succ_a >= n_threshold
        mask_b_valid = succ_b >= n_threshold
        if valid_spectra_mode == 'all_spectra':
            mask_valid = np.ones(self.feature_table_zone_averages.shape[0], dtype=bool)
        elif valid_spectra_mode == 'any_above':
            mask_valid = mask_a_valid | mask_b_valid
        elif valid_spectra_mode == 'both_above':
            mask_valid = mask_a_valid & mask_b_valid
        elif valid_spectra_mode == 'a_above':
            mask_valid = mask_a_valid
        elif valid_spectra_mode == 'b_above':
            mask_valid = mask_b_valid
        else:
            raise NotImplementedError()

        I_a = self.feature_table_zone_averages[str(self.mz_a)]
        I_b = self.feature_table_zone_averages[str(self.mz_b)]
        ratio = (I_b / (I_a + I_b)).fillna(0)
        ratio[~mask_valid] = np.nan
        self.feature_table_zone_averages['ratio'] = ratio
        if valid_spectra_mode in ('all_spectra', 'any_above'):
            self.feature_table_zone_successes['ratio'] = \
                self.feature_table_zone_successes.loc[
                    :, [str(self.mz_a), str(self.mz_b)]
            ].max(axis=1)
        elif valid_spectra_mode == 'a_above':
            self.feature_table_zone_successes['ratio'] = \
                self.feature_table_zone_successes.loc[str(self.mz_a)]
        elif valid_spectra_mode == 'b_above':
            self.feature_table_zone_successes['ratio'] = \
                self.feature_table_zone_successes.loc[str(self.mz_b)]
        elif valid_spectra_mode == 'both_above':
            self.feature_table_zone_successes['ratio'] = \
                self.feature_table_zone_successes.loc[
                    :, [str(self.mz_a), str(self.mz_b)]
            ].min(axis=1)

    def get_seasonality_proxy(self):
        return self.get_seasonalities().ratio


# uk = UK37()
# uk.set_time_series_tables()

"""
# filter out unsuccessful layers
mask_enough_successes = Ns.drop(columns='x').fillna(0).min(axis=1) >= n_successes_required
x = np.abs(mask_enough_successes.index).to_numpy()

o = np.argsort(x)
mask = mask_enough_successes.to_numpy()

# order
mask = mask[o]
x = x[o]
SST = SST[o]

plt.plot(
    # np.abs(UK37p_GC_FID.index)[mask_enough_successes],
    # UK37p_GC_FID[mask_enough_successes],
    x[mask],
    SST[mask],
    # markerfmt=''
)
plt.xlabel('pixel x')
plt.ylabel(r'SST ($^\circ$C)')
plt.ylim((SST[mask].min(), SST[mask].max()))
# plt.ylabel('UK37')
# plt.ylim((UK37p_GC_FID.min(), UK37p_GC_FID.max()))
"""
