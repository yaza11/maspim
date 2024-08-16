"""Calculate ratio proxies from time series data."""
from copy import deepcopy

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from maspim.util.convinience import check_attr

try:
    import bayspline
    UK37_SST_METHODS: list[str] = ['prahl', 'bayspline']
except ImportError:
    UK37_SST_METHODS: list[str] = ['prahl']
import logging

from typing import Iterable, Callable, Any

from maspim.data.combine_feature_tables import combine_feature_tables
from maspim.res.compound_masses import (
    mC37_2, mC37_3,  # alkenones
    mGDGT0, mGDGT1, mGDGT2, mGDGT3, mCren_p,  # GDGTS
    mC24FA, mC26FA, mC28FA, mC30FA,  # FA's
    mC29stanol, mC29stenol, mC28, mC29  # steroids
)
from maspim.res.constants import YD_transition
from maspim.time_series.main import TimeSeries


logger = logging.getLogger(__name__)

VALID_SPECTRA_MODES: tuple[str, ...] = ('all_spectra', 'any_above', 'all_above')


class Proxy(TimeSeries):
    """
    General proxy defined as a function of multiple compounds.

    This class inherits functionality from TimeSeries, but is quite different
    in how it is supposed to be used. Since this is a fairly high level object,
    it is expected to be used in a script-like fashion, rather than a pipeline.
    Therefore, the initialization is based on a time series object.
    """
    path_file: str | None = None

    _save_attrs = {
        'd_folder',
        'n_successes_required',
        '_feature_table',
        '_feature_table_standard_deviations',
        '_feature_table_successes',
        'mzs',
        'mzs_theo',
        'proxy_column_name'
    }

    def __init__(
            self,
            *,
            mzs: Iterable[float],
            func: Callable[[pd.Series, ...], pd.Series],
            time_series: TimeSeries | None = None,
            path_file: str | None = None,
            valid_spectra_mode: str = 'all_above',
            n_successes_required: int = 10,
            proxy_column_name: str = 'ratio',
            **_
    ) -> None:
        """
        Initialize the proxy.

        Parameters
        ----------
        mzs: Iterable[float]
            A vector-like object specifying the theoretical mz values needed to
            calculate the proxy.
        func: Callable[[pd.Series, ...], pd.Series]
            A function that takes compound intensities as vectors (in the same
            order as mzs) and returns corresponding proxy values.
        time_series: TimeSeries, optional
            The time series object holding the data of compounds
        path_file: str, optional
            Path to save and load the object from disk.
        valid_spectra_mode: str, optional
            Options are 'all_above', 'any_above' and 'all_spectra', which
            requires all, at least one or none of the compounds to be above
            the desired success number in each layer. The default is 'all_above'
        n_successes_required: int, optional
            The required number of successful pixels in each layer. The default
            is 10. Layers with fewer successes will be ignored.
        proxy_column_name: str, optional
            Name of the column with the proxy values. Defaults to 'ratio'.
        """
        assert (time_series is not None) or (path_file is not None), \
            'provide either time_series or path_file'

        if time_series is not None:
            self._set_mzs(mzs, time_series)
            self._copy_attributes(time_series)
            # has to happen after copy
            if path_file is not None:
                self.path_file: str = path_file
        else:
            if path_file is not None:
                self.path_file: str = path_file
            self.load()

        self._add_proxy(
            func,
            valid_spectra_mode=valid_spectra_mode,
            n_successes_required=n_successes_required,
            proxy_column_name=proxy_column_name
        )
        self.proxy_column_name: str = proxy_column_name

    def _set_mzs(self, mzs: Iterable[float], time_series: TimeSeries) -> None:
        self.mzs: list[str] = []
        self.mzs_theo: list[float] = []
        for mz in sorted(mzs):
            mz_c, diff = time_series.get_closest_mz(mz, return_deviation=True)
            if diff > .1:
                logger.warning(
                    f'Found large deviation for {mz} ({mz_c}, '
                    f'distance: {diff:.3f}), '
                    'make sure the mz is within the mass interval.'
                )
            self.mzs.append(mz_c)
            self.mzs_theo.append(mz)

    def _copy_attributes(self, time_series: TimeSeries) -> None:
        """Inherit attributes from time_series, reduce ft to relevant columns."""
        dict_new = {}
        iter_dict = deepcopy(time_series.__dict__)
        for k, v in iter_dict.items():
            if 'feature_table' in k:  # is a feature table
                # drop all irrelevant data columns
                cols_to_drop = time_series.data_columns.copy()
                for mz in self.mzs:  # keep mzs of interest
                    cols_to_drop.remove(mz)
                v_new = v.drop(columns=cols_to_drop)
            else:
                v_new = v
            # feature table only containing compounds of interest and
            # image features
            dict_new[k] = v_new
        self.__dict__ |= dict_new

    def _add_proxy(
            self, 
            func: Callable[[pd.Series, ...], pd.Series],
            valid_spectra_mode: str = 'all_above',
            n_successes_required: int = 10,
            proxy_column_name: str = 'ratio',
            **_
    ) -> None:
        """
        Add the proxy values to the feature table with name 'ratio'

        Parameters
        ----------
        func: Callable[[pd.Series, ...], pd.Series]
            A function that takes compound intensities as vectors (in the same
            order as mzs) and returns corresponding proxy values.
        valid_spectra_mode: str, optional
            Options are 'all_above', 'any_above' and 'all_spectra', which
            requires all, at least one or none of the compounds to be above
            the desired success number in each layer. The default is 'all_above'
        n_successes_required: int, optional
            The required number of successful pixels in each layer. The default
            is 10. Layers with fewer successes will be ignored.
        proxy_column_name: str, optional
            Name of the column with the proxy values. Defaults to 'ratio'.
        """
        assert valid_spectra_mode in VALID_SPECTRA_MODES, \
            (f"valid_spectra_mode must be one of {VALID_SPECTRA_MODES}, "
             f"not {valid_spectra_mode}")

        # the successes of every compound throughout the series
        succs: list[pd.Series] = [self.successes[mz] for mz in self.mzs]

        # which layers have enough successful spectra
        # build array by stacking series along axis 0
        masks_valid: np.ndarray[bool] = np.array([
            succ >= n_successes_required for succ in succs
        ])

        # construct vector for specifying which layers to keep
        if valid_spectra_mode == 'all_spectra':  # keep all
            mask_valid: np.ndarray[bool] = np.ones(
                self.feature_table.shape[0], dtype=bool
            )
        elif valid_spectra_mode == 'any_above':  # keep as long as one is above
            mask_valid: np.ndarray[bool] = masks_valid.any(axis=0)
        elif valid_spectra_mode == 'all_above':  # all have to be above thr
            mask_valid: np.ndarray[bool] = masks_valid.all(axis=0)
        else:  # there should be no other option
            raise NotImplementedError('internal error')

        # values of compounds
        vecs: list[pd.Series] = [self.feature_table[mz] for mz in self.mzs]
        # calculate proxy
        ratio: pd.Series = func(*vecs)
        # exclude layers not meeting success criterion
        ratio[~mask_valid] = np.nan
        # add to feature table
        self.feature_table[proxy_column_name] = ratio
        # set successes in success table
        if valid_spectra_mode in ('all_spectra', 'any_above'):
            self.successes[proxy_column_name] = self.successes.loc[
                :, self.mzs
            ].max(axis=1)
        elif valid_spectra_mode == 'all_above':
            self.successes[proxy_column_name] = self.successes.loc[
                :, self.mzs
            ].min(axis=1)
        else:
            raise NotImplementedError('internal error')

    @property
    def ratio(self) -> pd.Series:
        return self.feature_table.loc[:, self.proxy_column_name].copy()

    def get_seasonality_of_proxy(self) -> float:
        """Get the seosonality of the proxy"""
        return self.get_seasonalities()[self.proxy_column_name]

    def plot_proxy(self, sigma: int | float = 1) -> None:
        """
        Plot the proxy values.

        Parameters
        ----------
        sigma : int | float, optional
            The smoothing factor in the gaussian filter to apply. Defaults to 1.
        """
        x = self.feature_table.age
        y = self.feature_table.loc[:, self.proxy_column_name]

        plt.figure()
        plt.plot(x, scipy.ndimage.gaussian_filter1d(y, sigma=sigma), color='black')
        plt.vlines(
            YD_transition,
            y.min(),
            y.max(),
            linestyles='solid',
            alpha=.75,
            label='Pl-H boundary',
            color='black',
            linewidth=2
        )
        # plt.ylim((19, 31))
        plt.xlabel('Age in yrs b2k')
        plt.ylabel('Proxy')
        plt.grid(True)
        plt.show()

class RatioProxy(Proxy):
    """Construct proxy as ratio of compounds in existent feature table."""
    def __init__(self, mz_a: float | str, mz_b: float | str, **kwargs) -> None:
        """
        Initialize.

        Suitable for proxies of the form b / (a + b)

        Parameters
        ----------
        mz_a : float | str
            Compound a contributing to the ratio.
        mz_b : float | str
            Compound b contributing to the ratio. Compounds are sorted by mass
            automatically.
        kwargs: str | int | float
            Keyword arguments to pass to Proxy.__init__
        """
        def func(a: pd.Series, b: pd.Series) -> pd.Series:
            r = (b / (a + b)).fillna(0)
            return r

        super().__init__(
            mzs=(mz_a, mz_b),
            func=func,
            **kwargs
        )


class UK37(RatioProxy):
    """Class for calculating Uk37 and sst values from a time series object."""
    _method: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize.

        Na+ adducts are assumed to find the masses of compounds in the time
        series.

        Parameters
        ----------
        kwargs: Any
            Keyword arguments to pass to RatioProxy.__init__
        """
        super().__init__(
            mz_a=mC37_2,
            mz_b=mC37_3,
            **kwargs
        )
        self.mC37_3, self.mC37_2 = self.mzs

    @property
    def C37_2(self) -> pd.Series:
        """Values of the C37:2 alkenone"""
        return self.feature_table.loc[:, self.mC37_2].copy()
    
    @property
    def C37_3(self) -> pd.Series:
        """Values of the C37:3 alkenone"""
        return self.feature_table.loc[:, self.mC37_3].copy()

    def correct(self, correction_factor: float = 1):
        """Correction factor to be applied to proxy values."""
        self.feature_table.ratio *= correction_factor

    def get_std_err_ratio(self) -> pd.Series:
        """Calculate standard errors for the intensities of each proxy value."""
        errs = self.get_standard_errors()
        # chain rule
        ratio_errs: pd.Series = (
            np.abs(
                errs.loc[:, self.mC37_2]
                * self.C37_3 / (self.C37_2 + self.C37_3) ** 2
            ) + np.abs(
                errs.loc[:, self.mC37_3]
                * self.C37_2 / (self.C37_2 + self.C37_3) ** 2
            )
        )
        return ratio_errs

    def get_std_err_proxy(self) -> pd.Series:
        assert check_attr(self, '_method'), 'call add_SST first'

        if self._method == 'bayspline':
            upper = self.feature_table.loc[:, 'SST_lower']
            lower = self.feature_table.loc[:, 'SST_upper']
            return (upper - lower) / 2
        elif self._method == 'prahl':
            ratio_errs: pd.Series = self.get_std_err_ratio()
            # eq: SST = 29.41 * UK37p - 1.15
            return 29.41 * ratio_errs

    def add_SST(
            self,
            method: str = 'prahl',
            prior_std: int | float = 10,
            percentile_uncertainty: int = 5,
            **_: dict
    ):
        """
        Add SST values based on proxy values.

        Parameters
        ----------
        method : str, optional
            Method used to calculate SST values. Options are 'prahl' and
            'bayspline'. The later uses a Bayesian Spline model. Uses 'prahl'
            by default, which is the based on the fit commonly found in the
            literature of SST = (29.41 * Uk37p - 1.15) degC.
        prior_std : int, optional
            Prior standard for bayesian model. The default is 10.
        percentile_uncertainty: int, optional
            Desired level of uncertainty added as upper and lower bounds to
            plots. Interval is guaranteed to cover the uncertainty range
            [percentile_uncertainty, 100 - percentile_uncertainty].
            The default is 5. Only used for method
        """
        def SST_prahl(UK37p: pd.Series) -> pd.Series:
            return 29.41 * UK37p - 1.15

        assert method in UK37_SST_METHODS, \
            f'{method} is not available. Please choose one of {UK37_SST_METHODS}'

        UK37p: pd.Series = self.feature_table['ratio'].copy()
        mask_valid: pd.Series[bool] = ~UK37p.isna()

        # water temperature in degrees Celsius
        if method == 'prahl':
            SST: pd.Series = SST_prahl(UK37p[mask_valid])
        elif method == 'bayspline':
            prediction = bayspline.predict_sst(UK37p[mask_valid],
                                               prior_std=prior_std)
            SST = prediction.percentile(q=50)
            # add xth and 100-xth percentile to estiamte uncertainty
            self.feature_table['SST_lower'] = np.nan
            self.feature_table['SST_upper'] = np.nan
            self.feature_table.loc[mask_valid, 'SST_lower'] = \
                prediction.percentile(q=percentile_uncertainty)
            self.feature_table.loc[mask_valid, 'SST_upper'] = \
                prediction.percentile(q=100 - percentile_uncertainty)
        else:
            raise NotImplementedError('internal error')

        self._method: str = method
        self.feature_table['SST'] = np.nan
        self.feature_table.loc[mask_valid, 'SST'] = SST

    def plot_SST(self, sigma: int | float = 1) -> None:
        """
        Plot the reconstructed SST with uncertainties.

        Parameters
        ----------
        sigma : int | float, optional
            The smoothing factor in the gaussian filter to apply. Defaults to 1.
        """
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
        plt.grid(True)
        plt.show()

class ProxiesGDGT(Proxy):
    """
    ProxiesGDGT proxy class to reconstruct temperatures from GDGTs.

    Multiple methods are supported (known as TEX86H and TEX86L). Also , other
    proxies that can be calculated from GDGTs can be added:
    - The ring index (RI)
    - The Crenarchaeol Caldarchaeol Tetraether index (CCaT)
    - The methane index (MI)
    """
    def __init__(self, use_modified: bool = True, **kwargs) -> None:
        """Initialize.

        Na+ adducts are assumed to find the masses of compounds in the time
        series.

        Parameters
        ----------
        use_modified: bool, optional
            If True, will use the TEX86L formula, otherwise TEX86H.
            TEX86L = log(GDGT2 / (GDGT1 + GDGT2 + GDGT3))
            TEX86H = log((GDGT2 + GDGT3 + cren_p) / (GDGT1 + GDGT2 + GDGT3 + cren_p))
        kwargs: str | bool | int | float | None
            Keyword arguments for Proxy.__init__
        """
        def TEX86H(*vecs: pd.Series) -> pd.Series:
            cren_p, GDGT3, GDGT2, GDGT1, GDGT0 = vecs
            return np.log((GDGT2 + GDGT3 + cren_p) / (GDGT1 + GDGT2 + GDGT3 + cren_p))

        def TEX86L(*vecs: pd.Series) -> pd.Series:
            cren_p, GDGT3, GDGT2, GDGT1, GDGT0 = vecs
            return np.log(GDGT2 / (GDGT1 + GDGT2 + GDGT3))

        if not use_modified:
            logger.warning(
                "Cannot differentiate isosteriomers Crenarchaeol and Cren, "
                "for MS data it is thus advised to use the modified ProxiesGDGT."
            )

        super().__init__(
            mzs=(mCren_p, mGDGT3, mGDGT2, mGDGT1, mGDGT0),
            func=TEX86L if use_modified else TEX86H,
            **kwargs
        )

        self.use_modified = use_modified
        self.mCren_p, self.mGDGT3, self.mGDGT2, self.mGDGT1, self.mGDGT0,  = self.mzs


    def add_SST(
            self,
            method: str = 'conventional',
            prior_std: int | float = 10,
            percentile_uncertainty: int = 5,
            **kwargs
    ) -> None:
        """
        Calculate SST from GDGT abundances.

        Parameters
        ----------
        method : str, optional
            Method used to calculate the SST. 'conventional' just uses the
            equation (default behavior), 'BAYSPAR' uses the Bayesian model.
        prior_std: int | float, optional
            Prior std of the Bayesian model (default 10).
        percentile_uncertainty : int, optional
            Desired uncertainty levels in bayesian model (default 5).
        """
        def SST_original(ratio: pd.Series) -> pd.Series:
            """Formula for calculating SST from ration"""
            return 68.4 * ratio + 38.6

        def SST_modified(ratio: pd.Series) -> pd.Series:
            """New formula for calculating SST from ration"""
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
            SST_conv: Callable = SST_modified if self.use_modified else SST_original
            SST: pd.Series = SST_conv(ratio[mask_valid])
        elif method == 'BAYSPAR':
            prediction = bayspar.predict_seatemp(
                ratio[mask_valid],
                prior_std=prior_std,
                temptype='sst',
                **kwargs
            )
            SST = prediction.percentile(q=50)
            # add xth and 100-xth percentile to estiamte uncertainty
            self.feature_table['SST_lower'] = np.nan
            self.feature_table['SST_upper'] = np.nan
            self.feature_table.loc[mask_valid, 'SST_lower'] = prediction.percentile(q=percentile_uncertainty)
            self.feature_table.loc[mask_valid, 'SST_upper'] = prediction.percentile(q=100 - percentile_uncertainty)
        else:
            raise NotImplementedError('internal error')
        self.feature_table['SST'] = np.nan
        self.feature_table.loc[mask_valid, 'SST'] = SST

    def _add_ring_index(self):
        """Calculate ring index."""
        # GDGT0 can be omitted since the multiplicity is 0
        self.feature_table['RI'] = (
            self.feature_table[self.mGDGT1] +
            2 * self.feature_table[self.mGDGT2] +
            3 * self.feature_table[self.mGDGT3] +
            4 * self.feature_table[self.mCren_p]
        ) / self.feature_table.loc[:, [
            self.mGDGT0, self.mGDGT1, self.mGDGT2, self.mGDGT3, self.mCren_p
        ]].sum(axis='columns')

    def _add_CCaT(self) -> None:
        """
        Calculate the CCaT.

        Crenarchaeol Caldarchaeol Tetraether index
            CCaT = GDGT-5MS / (GDGT-0 + GDGT-5MS)

        GDGT-5 is cren
        """
        self.feature_table['CCaT'] = self.feature_table[self.mCren_p] / (
                self.feature_table[self.mGDGT0] + self.feature_table[self.mCren_p]
        )

    def _add_methane_index(self) -> None:
        """
        Calculate the methane index.

        MI = (GDGT1 + GDGT2 + GDGT3) / (GDGT1 + GDGT2 + GDGT3 + Cren_p)
        """
        # https://www.sciencedirect.com/science/article/pii/S0012821X11003141?via%3Dihub
        u: pd.Series = (
                self.feature_table[self.mGDGT1] +
                self.feature_table[self.mGDGT2] +
                self.feature_table[self.mGDGT3]
        )
        self.feature_table['MI'] = u / (
                u + self.feature_table[self.mCren_p]
        )

    @property
    def GDGT0(self) -> pd.Series:
        return self.feature_table[self.mGDGT0].copy()

    @property
    def GDGT1(self) -> pd.Series:
        return self.feature_table[self.mGDGT1].copy()

    @property
    def GDGT2(self) -> pd.Series:
        return self.feature_table[self.mGDGT2].copy()

    @property
    def GDGT3(self) -> pd.Series:
        return self.feature_table[self.mGDGT3].copy()

    @property
    def Cren_p(self) -> pd.Series:
        return self.feature_table[self.mCren_p].copy()

    @property
    def RI(self) -> pd.Series:
        if 'RI' not in self.feature_table.columns:
            self._add_ring_index()
        return self.feature_table.RI.copy()

    @property
    def MI(self) -> pd.Series:
        if 'MI' not in self.feature_table.columns:
            self._add_methane_index()
        return self.feature_table.MI.copy()

    @property
    def CCaT(self) -> pd.Series:
        if 'CCaT' not in self.feature_table.columns:
            self._add_CCaT()
        return self.feature_table.CCaT.copy()

    @property
    def SST(self):
        assert 'SST' in self.feature_table.columns, 'call add_SST first'
        return self.feature_table.SST.copy()

class C29StanolStenol(Proxy):
    """
    Proxy class for calculating the C29 stanol-stenol ratios.

    ratio = stanol / stenol
    """
    def __init__(self, **kwargs) -> None:
        """Initialize.

        Na+ adducts are assumed to find the masses of compounds in the time
        series.

        Parameters
        ----------
        kwargs: Keyword arguments forwarded to Proxy.__init__
        """
        def ratio(*vecs):
            stenol, stanol = vecs
            return stanol / stenol

        super().__init__(
            mzs=(mC29stenol, mC29stanol),
            func=ratio,
            **kwargs
        )

        self.mC29stenol, self.mC29stanol = self.mzs


class FA(Proxy):
    """
    Proxy class for calculating proxies from fatty acids.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize.

        Na+ adducts are assumed to find the masses of compounds in the time
        series.

        Parameters
        ----------
        kwargs: Keyword arguments forwarded to Proxy.__init__
        """

        mzs = (mC24FA, mC26FA, mC28FA)

        funcs = [
            lambda *vecs: vecs[0] / (vecs[1] + vecs[2]),
            lambda *vecs: vecs[0] / (vecs[1]),
            lambda *vecs: vecs[0] / (vecs[2]),
            lambda *vecs: vecs[1] / (vecs[2]),
            lambda *vecs: (24 * vecs[1] + 26 * vecs[2]) / (vecs[1] + vecs[2])
        ]
        column_names = [
            'ratio_24_all',
            'ratio_24_26',
            'ratio_24_28',
            'ratio_26_28',
            'av_chain_length'
        ]

        super().__init__(
            mzs=mzs,
            func=funcs[0],
            proxy_column_name=column_names[0],
            **kwargs
        )

        # add remaining proxies
        for func, col in zip(funcs[1:], column_names[1:]):
            self._add_proxy(
                func,
                proxy_column_name=col,
                **kwargs
            )


class Sterane(RatioProxy):
    """
    Proxy class for calculating ratio of C28 to (C28 + C29) steranes.
    """
    def __init__(self,**kwargs):
        """Initialize.

       Na+ adducts are assumed to find the masses of compounds in the time
       series.

       Parameters
       ----------
       kwargs: Keyword arguments forwarded to Proxy.__init__
       """
        super().__init__(
            mz_a=mC28,
            mz_b=mC29,
            **kwargs
        )

        self.mC28, self.mC29 = self.mzs


class Sterole(Proxy):
    """
    Proxy class for investigating sterols with 27 to 29 C atoms and 0 to 2
    double bounds
    """
    def __init__(self, **kwargs) -> None:
        # TODO
        ...
        raise NotImplementedError


class MultiSectionProxy(Proxy):
    def __init__(self, proxies: Iterable[Proxy]):
        assert all([isinstance(proxy, Proxy) for proxy in proxies])
        self.mzs = proxies[0].mzs
        assert all([proxy.mzs == self.mzs for proxy in proxies])

        self.feature_table = combine_feature_tables([proxy.feature_table for proxy in proxies])


