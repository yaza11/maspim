from data.combine_feature_tables import combine_feature_tables
from res.compound_masses import (
    mC37_2, mC37_3,  # alkenones
    mGDGT0, mGDGT1, mGDGT2, mGDGT3, mCren_p,  # GDGTS
    mC24FA, mC26FA, mC28FA, mC30FA,  # FA's
    mC29stanol, mC29stenol, mC28, mC29  # steroids
)
from res.constants import YD_transition

from timeSeries.cTimeSeries import TimeSeries

import re
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable, Callable


class ProxyBaseClass(TimeSeries):
    def _copy_attributes(self, TS):
        """Inherit attributes from TS, reduce ft to relevant columns."""
        for k, v in TS.__dict__.items():
            if 'feature_table' in k:  # is a feature table
                # drop all irrelevant data columns
                cols_to_drop = TS.get_data_columns().copy()
                for mz in self.mzs:
                    cols_to_drop.remove(mz)
                v = v.drop(columns=cols_to_drop)
            self.__setattr__(k, v)

    def get_seasonality_proxy(self):
        return self.get_seasonalities().ratio

class Proxy(ProxyBaseClass):
    def __init__(
            self,
            TS: TimeSeries,
            mzs: Iterable[float],
            func: Callable,
            valid_spectra_mode: str = 'both_above',
            n_successes_required: int = 10
    ):
        self.mzs = []
        for mz in mzs:
            mz_c, diff = TS.get_closest_mz(mz, return_deviation=True)
            if diff > .1:
                print(
                    f'Found large deviation for {mz} ({mz_c}, distance: {diff}), make sure the mz is within the mass interval.'
                )
            self.mzs.append(mz_c)

        self._copy_attributes(TS)
        self._add_proxy(
            func, 
            valid_spectra_mode=valid_spectra_mode, 
            n_successes_required=n_successes_required
        )

    def _add_proxy(
            self, 
            func, 
            valid_spectra_mode, 
            n_successes_required, 
            column_name: str = 'ratio'
    ):
        assert valid_spectra_mode in (modes := {'all_spectra', 'any_above', 'all_above'}), \
            f"valid_spectra_mode must be one of {modes}, not {valid_spectra_mode}"

        succs = [self.feature_table_successes[mz] for mz in self.mzs]
        masks_valid = np.array([succ >= n_successes_required for succ in succs])

        if valid_spectra_mode == 'all_spectra':
            mask_valid = np.ones(self.feature_table.shape[0], dtype=bool)
        elif valid_spectra_mode == 'any_above':
            mask_valid = masks_valid.any(axis=0)
        elif valid_spectra_mode == 'all_above':
            mask_valid = masks_valid.all(axis=0)
        else:
            raise NotImplementedError()

        vecs = [self.feature_table[mz] for mz in self.mzs]
        ratio = func(*vecs)
        ratio[~mask_valid] = np.nan
        self.feature_table[column_name] = ratio
        if valid_spectra_mode in ('all_spectra', 'any_above'):
            self.feature_table_successes[column_name] = self.feature_table_successes.loc[
                :, self.mzs
            ].max(axis=1)
        elif valid_spectra_mode == 'all_above':
            self.feature_table_successes[column_name] = \
                self.feature_table_successes.loc[
                :, self.mzs
                ].min(axis=1)
        else:
            raise NotImplementedError()

class RatioProxy(ProxyBaseClass):
    """Construct proxy as ratio of compounds in existent feature table."""

    def __init__(
            self,
            TS: TimeSeries,
            mz_a: float | str,
            mz_b: float | str,
            valid_spectra_mode: str = 'all_above',
            n_successes_required: int = 10
    ) -> None:
        """Initialize."""
        l = [float(mz_a), float(mz_b)]
        self.mz_a = str(TS.get_closest_mz(np.min(l)))
        self.mz_b = str(TS.get_closest_mz(np.max(l)))
        self.mzs = [self.mz_a, self.mz_b]

        self._copy_attributes(TS)
        self._add_proxy(valid_spectra_mode=valid_spectra_mode, n_successes_required=n_successes_required)

    def _add_proxy(
            self, valid_spectra_mode: str, n_successes_required: int
    ):
        """Add relative ratio proxy."""
        assert valid_spectra_mode in (modes := {'all_spectra', 'all_above', 'both_above', 'any_above', 'a_above', 'b_above'}), \
            f"valid_spectra_mode must be one of {modes}, not {valid_spectra_mode}"

        succ_a = self.feature_table_successes[self.mz_a]
        succ_b = self.feature_table_successes[self.mz_b]
        mask_a_valid = succ_a >= n_successes_required
        mask_b_valid = succ_b >= n_successes_required
        if valid_spectra_mode == 'all_spectra':
            mask_valid = np.ones(self.feature_table.shape[0], dtype=bool)
        elif valid_spectra_mode == 'any_above':
            mask_valid = mask_a_valid | mask_b_valid
        elif valid_spectra_mode in ('both_above', 'all_above'):
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

    @property
    def ratio(self):
        return self.feature_table.ratio.copy()

    @property
    def SST(self):
        return self.feature_table.SST.copy()
    
class UK37(RatioProxy):
    def __init__(
            self,
            TS: TimeSeries | None = None,
            path_file: str | None = None,
            valid_spectra_mode: str = 'all_above',
            n_successes_required: int = 10
    ) -> None:
        """Initialize."""
        assert (TS is not None) or (path_file is not None), 'provide either TS or path_file'

        if TS is not None:
            super().__init__(
                TS,
                mz_a=mC37_2,
                mz_b=mC37_3,
                valid_spectra_mode=valid_spectra_mode,
                n_successes_required=n_successes_required
            )

            self.mC37_3 = self.mz_a
            self.mC37_2 = self.mz_b
        else:
            self.verbose = False
            self.plts = False
            self.load(path_file)


    @property
    def C37_2(self):
        return self.feature_table.loc[:, self.mC37_2].copy()
    
    @property
    def C37_3(self):
        return self.feature_table.loc[:, self.mC37_3].copy()

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

    def correct(self, correction_factor: float = 1):
        self.feature_table.ratio *= correction_factor

    def get_std_err(self):
        errs = self.get_feature_table_standard_errors()
        # chain rule
        uk_errs = errs.loc[:, self.mC37_2] * self.C37_3 / (self.C37_2 + self.C37_3) ** 2 + \
                  errs.loc[:, self.mC37_3] * self.C37_2 / (self.C37_2 + self.C37_3) ** 2
        return uk_errs

    def add_SST(
            self,
            method='prahl',
            prior_std=10,
            percentile_uncertainty: int = 5,
            **_: dict
    ):
        def SST_prahl(UK37p):
            return 29.41 * UK37p - 1.15
        
        if method == 'BAYSPLINE':
            try:
                import bayspline
            except ModuleNotFoundError as e:
                print(e)
                print('falling back to prahl method')
                method = 'prahl'
        

        UK37p = self.feature_table['ratio'].copy()
        mask_valid: pd.Series[bool] = ~UK37p.isna()

        # water temperature in degrees Celsius
        if method == 'prahl':
            SST = SST_prahl(UK37p[mask_valid])
        elif method == 'BAYSPLINE':
            prediction = bayspline.predict_sst(UK37p[mask_valid], prior_std=prior_std)
            SST = prediction.percentile(q=50)
            # add xth and 100-xth percentile to estiamte uncertainty
            self.feature_table['SST_lower'] = np.nan
            self.feature_table['SST_upper'] = np.nan
            self.feature_table.loc[mask_valid, 'SST_lower'] = prediction.percentile(q=percentile_uncertainty)
            self.feature_table.loc[mask_valid, 'SST_upper'] = prediction.percentile(q=100 - percentile_uncertainty)
        else:
            raise NotImplementedError()
        self.feature_table['SST'] = np.nan
        self.feature_table.loc[mask_valid, 'SST'] = SST

    def plot(self, sigma=1):
        mask_success = self.feature_table.SST > 0
        x = self.feature_table.age[mask_success]
        y = self.feature_table.SST[mask_success]
        
        plt.figure()
        plt.plot(x, scipy.ndimage.gaussian_filter1d(y, sigma=sigma), color='black')
        plt.vlines(YD_transition, y.min(), y.max(),
                   linestyles='solid', alpha=.75, label='Pl-H boundary', 
                   color='black', linewidth=2
        )
        # plt.ylim((19, 31))
        plt.xlabel('Age in yrs b2k')
        plt.ylabel('SST in degrees C')
        plt.grid('on')
        plt.show()

class TEX86(Proxy):
    def __init__(
            self,
            TS: TimeSeries | None = None,
            path_file: str | None = None,
            valid_spectra_mode: str = 'all_above',
            n_successes_required: int = 10,
            use_modified: bool = True
    ) -> None:
        """Initialize."""
        assert (TS is not None) or (path_file is not None), 'provide either TS or path_file'

        def TEX86H(*vecs):
            GDGT1, GDGT2, GDGT3, cren_p = vecs
            return np.log((GDGT2 + GDGT3 + cren_p) / (GDGT1 + GDGT2 + GDGT3 + cren_p))

        def TEX86L(*vecs):
            GDGT1, GDGT2, GDGT3, *_ = vecs
            return np.log(GDGT2 / (GDGT1 + GDGT2 + GDGT3))

        if TS is not None:
            super().__init__(
                TS,
                mzs = [mGDGT0, mGDGT1, mGDGT2, mGDGT3, mCren_p],
                func=TEX86L if use_modified else TEX86H,
                valid_spectra_mode=valid_spectra_mode,
                n_successes_required=n_successes_required
            )
        else:
            self.verbose = False
            self.plts = False
            self.load(path_file)

        if not use_modified:
            print("'Cannot differentiate isosteriomers Crenarchaeol and Cren', for MS data it is thus adviced to use the modified TEX86.")

        self.use_modified = use_modified

        self.mGDGT0, self.mGDGT1, self.mGDGT2, self.mGDGT3, self.mCren_p = self.mzs


    def add_SST(self, method='conventional', prior_std=10, percentile_uncertainty: int = 5, **kwargs: dict):
        def SST_original(ratio):
            return 68.4 * ratio + 38.6

        def SST_modified(ratio):
            # https://www.sciencedirect.com/science/article/pii/S0016703710003054
            return 67.5 * ratio + 46.9

        if method == 'BAYSPAR':
            try:
                import bayspar
            except ModuleNotFoundError as e:
                print(e)
                print('falling back to conventional method')
                method = 'conventional'

        ratio: pd.Series = self.feature_table['ratio'].copy()
        mask_valid: pd.Series[bool] = ~ratio.isna()

        # water temperature in degrees Celsius
        if method == 'conventional':
            SST_conv = SST_modified if self.use_modified else SST_original
            SST = SST_conv(ratio[mask_valid])
        elif method == 'BAYSPAR':
            prediction = bayspar.predict_seatemp(ratio[mask_valid], prior_std=prior_std, temptype='sst', **kwargs)
            SST = prediction.percentile(q=50)
            # add xth and 100-xth percentile to estiamte uncertainty
            self.feature_table['SST_lower'] = np.nan
            self.feature_table['SST_upper'] = np.nan
            self.feature_table.loc[mask_valid, 'SST_lower'] = prediction.percentile(q=percentile_uncertainty)
            self.feature_table.loc[mask_valid, 'SST_upper'] = prediction.percentile(q=100 - percentile_uncertainty)
        else:
            raise NotImplementedError()
        self.feature_table['SST'] = np.nan
        self.feature_table.loc[mask_valid, 'SST'] = SST

    def add_ring_index(self):
        # GDGT0 can be omitted since the multiplicity is 0
        self.feature_table['RI'] = (
            self.feature_table[self.mGDGT1] +
            2 * self.feature_table[self.mGDGT2] +
            3 * self.feature_table[self.mGDGT3] +
            4 * self.feature_table[self.mCren_p]
        ) / self.feature_table.loc[:, [self.mGDGT0, self.mGDGT1, self.mGDGT2, self.mGDGT3, self.mCren_p]].sum(axis='columns')

    def add_CCaT(self):
        # Crenarchaeol Caldarchaeol Tetraether index
        # CCaT = GDGT-5MS / (GDGT-0 + GDGT-5MS)
        # GDGT-5 is cren
        self.feature_table['CCaT'] = self.feature_table[self.mCren_p] / (
                self.feature_table[self.mGDGT0] + self.feature_table[self.mCren_p]
        )

    def add_methane_index(self):
        # https://www.sciencedirect.com/science/article/pii/S0012821X11003141?via%3Dihub
        u = self.feature_table[self.mGDGT1] + self.feature_table[self.mGDGT2] + self.feature_table[self.mGDGT3]
        self.feature_table['MI'] = (u) / (
                u + self.feature_table[self.mCren_p]
        )

    @property
    def GDGT0(self):
        return self.feature_table[self.mGDGT0].copy()

    @property
    def GDGT1(self):
        return self.feature_table[self.mGDGT1].copy()

    @property
    def GDGT2(self):
        return self.feature_table[self.mGDGT2].copy()

    @property
    def GDGT3(self):
        return self.feature_table[self.mGDGT3].copy()

    @property
    def Cren_p(self):
        return self.feature_table[self.mCren_p].copy()

    @property
    def RI(self):
        return self.feature_table.RI.copy()

    @property
    def MI(self):
        return self.feature_table.MI.copy()

    @property
    def CCaT(self):
        return self.feature_table.CCaT.copy()

    @property
    def SST(self):
        return self.feature_table.SST.copy()

class C29StanolStenol(Proxy):
    # ratio = stanol / stenol
    def __init__(
            self,
            TS: TimeSeries,
            valid_spectra_mode: str = 'all_above',
            n_successes_required: int = 10,
    ) -> None:
        """Initialize."""

        def ratio(*vecs):
            stanol, stenol = vecs
            return stanol / stenol

        super().__init__(
            TS,
            mzs=[mC29stanol, mC29stenol],
            func=ratio,
            valid_spectra_mode=valid_spectra_mode,
            n_successes_required=n_successes_required
        )

        self.mC29stanol, self.mC29stenol = self.mzs


class FA(Proxy):
    def __init__(
            self,
            TS: TimeSeries,
            valid_spectra_mode: str = 'all_above',
            n_successes_required: int = 10,
    ) -> None:
        """Initialize."""

        mzs = [mC24FA, mC26FA, mC28FA]

        self.mzs = []
        for mz in mzs:
            mz_c = str(TS.get_closest_mz(mz))
            self.mzs.append(mz_c)

        self._copy_attributes(TS)

        self.mC24FA, self.mC26FA, self.mC28FA = self.mzs

        funcs = [
            lambda *vecs: vecs[0] / (vecs[1] + vecs[2]),
            lambda *vecs: vecs[0] / (vecs[1]),
            lambda *vecs: vecs[0] / (vecs[2]),
            lambda *vecs: vecs[1] / (vecs[2])
        ]
        column_names = [
            'ratio_24_all',
            'ratio_24_26',
            'ratio_24_28',
            'ratio_26_28'
        ]

        for func, column_name in zip(funcs, column_names):
            self._add_proxy(
                func,
                valid_spectra_mode=valid_spectra_mode,
                n_successes_required=n_successes_required,
                column_name=column_name
            )


class Sterane(RatioProxy):
    def __init__(
            self,
            TS: TimeSeries,
            valid_spectra_mode: str = 'both_above',
            n_successes_required: int = 10
    ):
        super().__init__(TS, mz_a=mC28, mz_b=mC29, valid_spectra_mode=valid_spectra_mode, n_successes_required=n_successes_required)

        self.mC28 = self.mz_a
        self.mC29 = self.mz_b
    
class MultiSectionUK37(UK37):
    def __init__(self, uk37s: Iterable[UK37]):
        assert all([isinstance(uk37, UK37) for uk37 in uk37s])
        self.mz_a = uk37s[0].mz_a
        self.mz_b = uk37s[0].mz_b
        assert all([uk.mz_a == self.mz_a for uk in uk37s])
        assert all([uk.mz_b == self.mz_b for uk in uk37s])

        self.feature_table = combine_feature_tables([uk.feature_table for uk in uk37s])

        self.plts = False
        self.verbose = False

