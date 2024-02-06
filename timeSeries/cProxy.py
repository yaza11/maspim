from res.constants import (
    mC37_2, mC37_3, sections_all, n_successes_required,
    window_to_type, distance_pixels, YD_transition, elements
)
from data.cMSI import MSI
from data.cDataClass import combine_sections
from timeSeries.cTimeSeries import TimeSeries

import re
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RatioProxy(TimeSeries):
    """Construct proxy as ratio of compounds in existent feature table."""

    def __init__(
            self,
            TS: TimeSeries,
            mz_a: float | str,
            mz_b: float | str,
            valid_spectra_mode='both_above'
    ) -> None:
        """Initialize."""
        l = [float(mz_a), float(mz_b)]
        self.mz_a = str(TS.get_closest_mz(np.min(l)))
        self.mz_b = str(TS.get_closest_mz(np.max(l)))

        self._copy_attributes(TS)
        self._add_proxy(valid_spectra_mode=valid_spectra_mode)

    def _copy_attributes(self, TS):
        """Inherit attributes from TS, reduce ft to relevant columns."""
        for k, v in TS.__dict__.items():
            if 'feature_table' in k:  # is a feature table
                # drop all irrelevant data columns
                cols_to_drop = TS.get_data_columns().copy()
                cols_to_drop.remove(self.mz_a)
                cols_to_drop.remove(self.mz_b)
                v = v.drop(columns=cols_to_drop)
            self.__setattr__(k, v)

    def get_data_columns(self):
        """Modified data columns getter where ratio is data column."""
        columns = self.get_feature_table().columns
        columns_valid = []
        columns_xrf = [col for col in columns if
                       col in list(elements.Abbreviation)]
        columns_msi = [col for col in columns if str(
            col).replace('.', '', 1).isdigit()]
        columns_valid = columns_xrf + columns_msi
        if 'ratio' in self.get_feature_table().columns:
            columns_valid.append('ratio')
        return columns_valid

    def _add_proxy(
            self, valid_spectra_mode: str, n_threshold: int | None = None
    ):
        """Add relative ratio proxy."""
        assert valid_spectra_mode in {'all_spectra', 'both_above', 'any_above', 'a_above', 'b_above'}, \
            f"succes_mode must be one of 'all_spectra', 'both_above', 'one_above', not {valid_spectra_mode}"

        if n_threshold is None:
            n_threshold = n_successes_required

        succ_a = self.feature_table_successes[self.mz_a]
        succ_b = self.feature_table_successes[self.mz_b]
        mask_a_valid = succ_a >= n_threshold
        mask_b_valid = succ_b >= n_threshold
        if valid_spectra_mode == 'all_spectra':
            mask_valid = np.ones(self.feature_table.shape[0], dtype=bool)
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

        I_a = self.feature_table[self.mz_a]
        I_b = self.feature_table[self.mz_b]
        ratio = (I_b / (I_a + I_b)).fillna(0)
        ratio[~mask_valid] = np.nan
        self.feature_table['ratio'] = ratio
        if valid_spectra_mode in ('all_spectra', 'any_above'):
            self.feature_table_successes['ratio'] = \
                self.feature_table_successes.loc[
                    :, [self.mz_a, self.mz_b]
            ].max(axis=1)
        elif valid_spectra_mode == 'a_above':
            self.feature_table_successes['ratio'] = \
                self.feature_table_successes.loc[self.mz_a]
        elif valid_spectra_mode == 'b_above':
            self.feature_table_successes['ratio'] = \
                self.feature_table_successes.loc[self.mz_b]
        elif valid_spectra_mode == 'both_above':
            self.feature_table_successes['ratio'] = \
                self.feature_table_successes.loc[
                    :, [self.mz_a, self.mz_b]
            ].min(axis=1)

    def get_seasonality_proxy(self):
        return self.get_seasonalities().ratio
    
class UK37(RatioProxy):
    def __init__(self, TS : TimeSeries, valid_spectra_mode='both_above') -> None:
        """Initialize."""
        self.plts = False
        self.verbose = False

        self.mz_a = str(TS.get_closest_mz(mC37_3))
        self.mz_b = str(TS.get_closest_mz(mC37_2))

        self._copy_attributes(TS)
        self._add_proxy(valid_spectra_mode=valid_spectra_mode)
        
    @property
    def C37_2(self):
        return self.feature_table.loc[:, self.mz_b]
    
    @property
    def C37_3(self):
        return self.feature_table.loc[:, self.mz_a]

    def combine_layers(self, table=None, diff_sign_condition=0):
        raise NotImplementedError('Depricated')
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

    def add_UK_proxy(self, corrected=False):
        # calculate UK37 proxy

        UK37p = self.C37_2 / (self.C37_2 + self.C37_3)
        if corrected:
            UK37p *= 1.194

        self.feature_table['UK37p'] = UK37p

    def add_SST(self, method='prahl', prior_std = 10, **kwargs):
        def SST_prahl(UK37p):
            return 29.41 * UK37p - 1.15
        assert 'UK37p' in self.feature_table.columns
        
        if method=='BAYSPLINE':
            try:
                import bayspline
            except ModuleNotFoundError as e:
                print(e)
                print('falling back to prahl method')
                method = 'prahl'
        
        UK37p = self.feature_table['UK37p'].copy()
        # water temperature in degrees Celsius
        UK37p = UK37p.fillna(0)
        if method == 'prahl':
            self.feature_table['SST'] = SST_prahl(UK37p)
        elif method == 'BAYSPLINE':
            prediction = bayspline.predict_sst(UK37p, prior_std=prior_std)
            self.feature_table['SST'] = prediction.percentile(q=50)
        else:
            raise NotImplementedError()
            
    def plot(self, sigma=1):
        mask_success = self.feature_table.SST > 0
        x = self.feature_table.age[mask_success]
        y = self.feature_table.SST[mask_success]
        
        plt.figure()
        plt.plot(x, scipy.ndimage.gaussian_filter1d(y, sigma=sigma), color='black')
        plt.vlines(YD_transition, 0, 35, 
                   linestyles='solid', alpha=.75, label='Pl-H boundary', 
                   color='black', linewidth=2
        )
        # plt.ylim((19, 31))
        plt.grid('on')
        plt.show()