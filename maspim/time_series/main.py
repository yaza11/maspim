import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Iterable, Self, Callable
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
from scipy.signal.windows import blackman
from astropy.timeseries import LombScargle

from maspim.data.combine_feature_tables import combine_feature_tables
from maspim.data.main import DataBaseClass
from maspim.time_series.distances import sign_corr, sign_weighted_corr
from maspim.util.convinience import Convinience, check_attr
from maspim.res.constants import elements, YD_transition
from maspim.imaging.util.coordinate_transformations import rescale_values


logger = logging.getLogger(__name__)


class TimeSeries(DataBaseClass, Convinience):
    _save_in_d_folder: bool = True

    _feature_table: pd.DataFrame | None = None
    _feature_table_successes: pd.DataFrame | None = None
    _feature_table_standard_deviations: pd.DataFrame | None = None
    _feature_table_standard_errors: pd.DataFrame | None = None
    _contrasts: pd.DataFrame | None = None

    _data_columns: list[str] | None = None

    _save_attrs = {
        'd_folder',
        'n_successes_required',
        '_feature_table',
        '_feature_table_standard_deviations',
        '_feature_table_successes'
    }

    def __init__(
            self,
            path_folder: str | None = None,
            n_successes_required: int = 10
    ) -> None:
        """Initialize."""

        assert isinstance(n_successes_required, int), 'n_successes_required must be an integer'
        assert isinstance(path_folder, str | None), 'path_folder must be string or None'

        self.n_successes_required: int = n_successes_required

        self._set_files(path_folder)

        self._feature_table: pd.DataFrame = pd.DataFrame()
        self._feature_table_successes: pd.DataFrame = pd.DataFrame()
        self._feature_table_standard_deviations: pd.DataFrame = pd.DataFrame()


    def _sort_tables(self) -> None:
        if check_attr(self, '_feature_table'):
            self._feature_table = self._feature_table\
                .sort_values(by='x_ROI')\
                .reset_index(drop=True)
        if check_attr(self, '_feature_table_standard_deviations'):
            self._feature_table_standard_deviations = self._feature_table_standard_deviations \
                .sort_values(by='x_ROI') \
                .reset_index(drop=True)
        if check_attr(self, '_feature_table_successes'):
            self._feature_table_successes = self._feature_table_successes \
                .sort_values(by='x_ROI') \
                .reset_index(drop=True)

    def _post_load(self) -> None:
        self._sort_tables()

    def _set_files(self, path_folder: str) -> None:
        if path_folder is None:
            return

        # path_folder could be d-folder
        if path_folder[-2:] == '.d':
            logger.info('detected d-folder')
            path_folder, d_folder = os.path.split(path_folder)
            self.d_folder: str = d_folder
        else:
            logger.info('detected no d-folder')
            self.d_folder = ''
        self.path_folder = path_folder

    @property
    def path_d_folder(self) -> str:
        return os.path.join(self.path_folder, self.d_folder)

    @property
    def age(self) -> pd.Series:
        return self.feature_table.age

    @property
    def successes(self) -> pd.DataFrame:
        return self._feature_table_successes

    @property
    def deviations(self) -> pd.DataFrame:
        return self._feature_table_standard_deviations

    def set_feature_tables(
            self,
            intensities: pd.DataFrame,
            successes: pd.DataFrame | None,
            deviations: pd.DataFrame | None
    ) -> None:
        if successes is not None:
            assert intensities.shape[0] == successes.shape[0]
        if deviations is not None:
            assert intensities.shape[0] == deviations.shape[0]

        self._feature_table = intensities
        self._feature_table_successes = successes
        self._feature_table_standard_deviations = deviations

        self._sort_tables()

    def combine_duplicate_seed(self, weighted=False):
        """Combining layers with same seeds, information about quality is lost."""
        if weighted:
            raise NotImplementedError(
                'there is an issue with counts for quality criteria, check \
that before using this option'
            )
            seeds = self.feature_table.seed.copy()
            # take weighted average of rows that have the same xROI
            #   mult every comps in each layer by its n
            cols = self.data_columns + [
                'L', 'x_ROI', 'quality', 'homogeneity', 'continuity',
                'contrast', 'quality'
            ]
            ns = self.successes.loc[:, cols]
            sums = ns * self.feature_table.loc[:, cols]
            #   add x_ROI col
            sums['seed'] = seeds
            #   take weighted mean
            ns['seed'] = seeds
            wmean = sums.groupby('seed').sum() / ns.groupby('seed').sum()
            # set nans to 0 an drop xROI (index) into dataframe
            wmean = wmean.sort_values(by='x_ROI').fillna(0).reset_index()
            ns = ns.groupby('seed').sum().sort_values(by='x_ROI').reset_index()
        else:
            wmean = self.feature_table \
                .groupby('seed') \
                .mean() \
                .fillna(0) \
                .sort_values(by='x_ROI') \
                .reset_index()
            ns = self.successes \
                .groupby('seed') \
                .mean() \
                .sort_values(by='x_ROI') \
                .reset_index()

        return wmean, ns

    def get_standard_errors(self) -> pd.DataFrame:
        # standard error: sigma / sqrt(n)
        columns = list(set(self.deviations.columns)
                       & set(self.successes.columns)
                       - {'zones'})
        successes: np.ndarray = self.successes.loc[:, columns].to_numpy()
        # will encounter invalid values for 0 successes and any nans
        with np.errstate(invalid="ignore"):
            feature_table_standard_errors = np.divide(
                self.deviations.loc[:, columns],
                np.sqrt(successes),
                where=successes > 0,
                out=np.full_like(successes, np.nan)
            )
        # zones is more of an index
        feature_table_standard_errors.loc[:, 'zone'] = self.deviations.zone
        return feature_table_standard_errors

    @property
    def errors(self) -> pd.DataFrame:
        if not check_attr(self, '_feature_table_standard_errors'):
            self._feature_table_standard_errors: pd.DataFrame = self.get_standard_errors()
        return self._feature_table_standard_errors

    def _get_data_columns(self) -> list[str]:
        columns = self.columns
        columns_xrf = [col for col in columns if
                       col in list(elements.Abbreviation)]
        columns_msi = [col for col in columns if str(
            col).replace('.', '', 1).isdigit()]
        columns_valid = columns_xrf + columns_msi
        return columns_valid

    @property
    def data_columns(self):
        if not check_attr(self, '_data_columns'):
            self._data_columns = self._get_data_columns()
        return self._data_columns

    def get_weighted(
            self,
            feature_table: pd.DataFrame | None,
            use_L_contrasts: bool,
            **_
    ) -> pd.DataFrame:
        """
        Return average intensities weighted by quality of layers.

        Parameters
        ----------
        feature_table : pd.DataFrame
            The feature table in which to add the weights.
        use_L_contrasts : bool
            If True, will use contrasts signed by homogeneity, otherwise the
            geometric mean of the quality.

        Returns
        -------
        data_weighted : pd.DataFrame
            The feature table weighted by the quality.

        """
        if feature_table is None:
            feature_table = self.feature_table

        # get quality column
        if not use_L_contrasts:
            logger.info('using geometric mean of quality')
            q: pd.Series = (feature_table.quality
                            if 'quality' in feature_table
                            else self.feature_table.quality)
            # calc geometric mean
            q: pd.Series = q.abs().pow(1 / 3)
        # use contrasts of grayscale
        else:
            logger.info('using signed contrasts for quality')
            # try to take criteria from feature table
            c: pd.Series = (feature_table.contrast
                            if 'contrast' in feature_table.columns
                            else self.feature_table.contrast)
            h: pd.Series = (feature_table.homogeneity
                            if 'homogeneity' in feature_table.columns
                            else self.feature_table.homogeneity)
            q: pd.Series = c * np.sign(h)

        # set qs below 0 to 0
        mask_less_than_zero: pd.Series = q < 0
        q[mask_less_than_zero] = 0

        # weigh data by quality
        data_weighted: pd.DataFrame = feature_table.multiply(q, axis=0)
        # add weights columns
        data_weighted['weights'] = q
        return data_weighted

    def get_contrasts_table(
            self,
            feature_table: pd.DataFrame | None = None,
            subtract_mean: bool = False,
            columns: Iterable | None = None
    ) -> pd.DataFrame:
        """
        Return the contrasts for each layer in the averages table.

        Parameters
        ----------
        feature_table : pd.DataFrame | None, optional
            The feature table for which to calculate the conrasts.
            The default is None.
            This will default to get_feature_table_zone_averages
        subtract_mean: bool, optional
            Will subtract the mean of each time series before calculating the
            contrasts. This will default to False.
        columns: Iterable, optional
            The columns of which to take the contrasts. Defaults to all.

        Returns
        -------
        ft_contrast : pd.DataFrame
            The contrasts for each component and layer.
            Reflecting boundary conditions will be applied.
            x, x_ROI, seed will be copied over from
            get_feature_table_zone_averages, contrast will from
            get_feature_table_zone_averages will be renamed to L
        """
        if columns is not None:
            pass
        elif feature_table is None:
            columns: pd.Index = self.columns
        else:
            columns = feature_table.columns

        if feature_table is None:
            feature_table: pd.DataFrame = self.feature_table.loc[:, columns]

        # contrast calculation
        brightnesses: np.ndarray = feature_table.to_numpy()
        if subtract_mean:
            brightnesses = np.subtract(brightnesses, brightnesses.mean(axis=0))
        # add reflecting boundary condition
        brightnesses_bound = np.pad(brightnesses,
                                    ((1, 1), (0, 0)),
                                    mode='reflect')
        # define slices
        slice_center = np.index_exp[1:-1, :]
        slice_up = np.index_exp[:-2, :]
        slice_down = np.index_exp[2:, :]
        # get neighbour, center values
        neighbour_up: np.ndarray = brightnesses_bound[slice_up]
        neighbour_down: np.ndarray = brightnesses_bound[slice_down]
        neighbours: np.ndarray = (neighbour_up + neighbour_down) / 2
        center: np.ndarray = brightnesses_bound[slice_center]
        # calc contrast (prevent division by zero), invalid values will be 0
        contrast = np.divide(
            center - neighbours,
            center + neighbours,
            out=np.zeros_like(center),
            where=center + neighbours != 0
        )
        # put in feature table
        ft_contrast = pd.DataFrame(
            data=contrast,
            index=feature_table.index,
            columns=columns
        )
        # copy depth over from av intensity calculation
        if 'x' in columns:
            ft_contrast['x'] = self.feature_table.x.copy()
        if 'x_ROI' in columns:
            ft_contrast['x_ROI'] = self.feature_table.x_ROI.copy()
        if 'seed' in columns:
            ft_contrast['seed'] = self.feature_table.seed.copy()

        return ft_contrast

    def _get_contrast_errors(self) -> pd.DataFrame:
        """

        Notes
        -----
        Contrasts for i-th layer c[i] is calculated as follows:
                    l[i] - (l[i+1] + l[i-1]) / 2
            c[i] = -------------------------------
                    l[i] + (l[i+1] + l[i-1]) / 2

                    2 * l[i] - (l[i+1] + l[i-1])
            c[i] = -------------------------------
                    2 * l[i] + (l[i+1] + l[i-1])
        where l[i] is the intensity at the i-th layer. By the error propagation
        formula we get
                              4 * (l[i+1] + l[i-1])
            part l[i] =  ------------------------------
                          (2 * l[i] + l[i+1] + l[i-1]) ** 2

                                           4 * l[i]
            part l[i + 1] =  -----------------------------------
                              (2 * l[i] + l[i+1] + l[i-1]) ** 2

                                           4 * l[i]
            part l[i - 1] =  -----------------------------------
                              (2 * l[i] + l[i+1] + l[i-1]) ** 2

        https://www.wolframalpha.com/input?i=d%2Fdx+%28x+-+%28y+%2B+z%29+%2F+2%29+%2F+%28x+%2B+%28y+%2B+z%29+%2F+2%29
        https://www.wolframalpha.com/input?i=d%2Fdy+%28x+-+%28y+%2B+z%29+%2F+2%29+%2F+%28x+%2B+%28y+%2B+z%29+%2F+2%29
        https://www.wolframalpha.com/input?i=d%2Fdz+%28x+-+%28y+%2B+z%29+%2F+2%29+%2F+%28x+%2B+%28y+%2B+z%29+%2F+2%29

        So overall

            Delta c[i] =   part l[i] * Delta l[i]
                         + part l[i+1] * Delta l[i+1]
                         + part l[i-1] * Delta l[i-1]
        """
        columns = list(set(self.columns) & set(self.errors.columns))
        feature_table: pd.DataFrame = self.feature_table.loc[:, columns]

        # contrast calculation
        brightnesses: np.ndarray = feature_table.to_numpy()
        # add reflecting boundary condition
        brightnesses_bound: np.ndarray = np.pad(brightnesses,
                                                ((1, 1), (0, 0)),
                                                mode='reflect')
        # define slices
        slice_center = np.index_exp[1:-1, :]
        slice_up = np.index_exp[:-2, :]
        slice_down = np.index_exp[2:, :]
        # get neighbour, center values
        neighbour_up: np.ndarray = brightnesses_bound[slice_up]
        neighbour_down: np.ndarray = brightnesses_bound[slice_down]
        center: np.ndarray = brightnesses_bound[slice_center]

        v: np.ndarray = 2 * center + neighbour_down + neighbour_up
        part_center: np.ndarray = np.divide(
            4 * (neighbour_up + neighbour_down),
            v ** 2,
            out=np.zeros_like(center),
            where=v != 0
        )
        part_up: np.ndarray = np.divide(
            4 * center,
            v ** 2,
            out=np.zeros_like(center),
            where=v != 0
        )
        part_down: np.ndarray = part_up

        errors: np.ndarray = self.errors.loc[:, columns].to_numpy()
        errors: np.ndarray = np.pad(errors,
                                    ((1, 1), (0, 0)),
                                    mode='reflect')

        errors_center: np.ndarray = errors[slice_center]
        errors_up: np.ndarray = errors[slice_up]
        errors_down: np.ndarray = errors[slice_down]

        errors: np.ndarray = (np.abs(part_center * errors_center) +
                              np.abs(part_down * errors_down) +
                              np.abs(part_up * errors_up))

        errors_df = pd.DataFrame(data=errors, columns=columns)

        return errors_df

    @property
    def contrasts(self) -> pd.DataFrame:
        if not check_attr(self, '_contrasts'):
            self._contrasts = self.get_contrasts_table()
        return self._contrasts

    def get_envelope(self, comp: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return envelopes in the contrast signal for a given compound.

        This function finds local extrema and linearly interpolates values for
        every index.

        Parameters
        ----------
        comp : str
            The compound for which to calculate the envelope.

        Returns
        -------
        envelope_min : np.ndarray
            values of lower function for every index.
        envelope_max : TYPE
            values of upper function for every index.

        """
        # https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
        comp = self.get_closest_mz(comp)
        v = self.contrasts.loc[:, comp]
        idxs_local_mins = (np.diff(np.sign(np.diff(v))) > 0).nonzero()[0] + 1
        idxs_local_maxs = (np.diff(np.sign(np.diff(v))) < 0).nonzero()[0] + 1

        # interpolate to all indices
        idxs: np.ndarray[int] = np.arange(len(v), dtype=int)
        envelope_min: np.ndarray = np.interp(idxs,
                                             idxs_local_mins,
                                             v[idxs_local_mins])
        envelope_max: np.ndarray = np.interp(idxs,
                                             idxs_local_maxs,
                                             v[idxs_local_maxs])
        return envelope_min, envelope_max

    def get_seasonality(self, comp, window_size, dt=.5, plts=False):
        comp = self.get_closest_mz(comp)
        v = self.get_contrasts_table().loc[:, comp]
        t = self.age
        # get upper and lower envelope
        e_min, e_max = self.get_envelope(comp)

        s = (e_max - e_min) / 2

        s_median = np.median(s)

        t_i = np.arange(t.min(), t.max() + dt, dt)
        s_i = np.interp(t_i, t, s)

        # run av filter
        def moving_average(a):
            ret = np.cumsum(a, dtype=float)
            ret[window_size:] = ret[window_size:] - ret[:-window_size]
            return ret[window_size - 1:] / window_size

        s_ = moving_average(s_i)
        t_ = moving_average(t_i)

        if plts:
            plt.plot(t, v, label=str(comp))
            plt.plot(t, e_min, label='envelope', color='darkblue')
            plt.plot(t, e_max, color='darkblue')
            plt.plot(t_i, s_i, label='amp.')
            plt.plot(t_, s_, label=f'smoothed amp. ({window_size * dt:.1f} yrs)')

            if (t.min() < YD_transition) and (YD_transition < t.max()):
                logger.info('Pl-H transition in slice!')
                # add vertical line to mark the transition
                plt.vlines(YD_transition, -1, 1, linestyles='solid', alpha=.75, label='Pl-H boundary', color='black',
                           linewidth=2)

                # add horizontal lines for averages
                mask_pl = t > YD_transition
                mask_h = t < YD_transition
                pl_av = np.median(s[mask_pl])
                h_av = np.median(s[mask_h])
                plt.hlines(pl_av, t[mask_pl][0], t[mask_pl][-1], color='darkblue', alpha=.5, label='Pl med. seas.')
                plt.hlines(h_av, t[mask_h][0], t[mask_h][-1], color='darkred', alpha=.5, label='H med. seas.')
            plt.hlines(s_median, t[0], t[-1], color='black', alpha=.5, label='median seas.')

            plt.xlim((t.min(), t.max()))
            plt.ylim((v.min(), v.max()))
            plt.xlabel('age (yr B2K)')
            plt.ylabel('contrast')
            plt.legend()
            plt.show()

        return t_, s_, s_median

    def get_seasonalities(
            self,
            ft: pd.DataFrame | None = None,
            cols: Iterable | None = None,
            weighted: bool = True,
            exclude_low_success: bool = True,
            mult_n: bool = True,
            norm_weights: bool = False,
            **kwargs
    ) -> pd.Series:
        """
        Method for calculating seasonalities of each compound based on
        contrasts and layer qualities.

        Parameters
        ----------
        ft: pd.DataFrame, optional
            Feature table of contrasts. Will call get_contrast_table with the
            provided kwargs.
        cols: Iterable, optional
            Columns for which to calculate the seasonalities. If not provided,
            will use the data_colunms
        weighted: bool, optional
            If True, will weigh the time series by qualities. Will use kwargs,
            if provided.
        exclude_low_success: bool, optional
            Fill layers which are below the required success amount to nan.
            The default is True.
        mult_n: bool, optional
            If exclude_low_success is set to True, this option becomes available.
            If this is set to True (which is the default), the weighted table
            will be multiplied with the ratio of successful pixels within th
            layer.
        norm_weights: bool, optional
            If this is set to True, the sum values will be bound between -1 and
            1. If False (default), the median score for the seasonality will
            be returned.
        kwargs: Any
            Additional keyword arguments for methods mentioned above.

        Returns
        -------
        seasonalities: pd.Series
            The seasonalities for each compound specified by cols.
        """
        if cols is not None:
            pass
        elif ft is not None:
            cols = ft.columns
        else:
            cols = self.data_columns

        if ft is None:
            ft = self.get_contrasts_table(
                subtract_mean=kwargs.get('subtract_mean', False)
            ).loc[:, cols]
        # make sure we don't modify the input ft
        ft: pd.DataFrame = ft.copy()

        # multiply contrasts of comps with sign seed (comps with high summer
        # seasonality should only have positive signs, winter comps only
        # negative after multiplication)
        ft = ft.multiply(np.sign(self.feature_table.seed), axis=0)
        # weigh contrasts by quality
        if weighted:
            ft = self.get_weighted(
                ft,
                use_L_contrasts=kwargs.get('use_L_contrasts', True)
            )
            weights = ft.weights
            ft = ft.loc[:, cols]
            if norm_weights:
                scaling = weights.sum()
                ft /= scaling

        if exclude_low_success:
            ft_succ = self.successes.loc[:, cols].copy()
            if mult_n:
                # multiply with ratio of successful layers
                r_succs = (
                        (ft_succ > self.n_successes_required).sum(axis=0)
                        / ft_succ.shape[0]
                )
                ft = ft.multiply(r_succs, axis=1)
            else:
                ft[ft_succ < self.n_successes_required] = np.nan

        # take median
        if norm_weights:
            seasonalities: pd.Series = ft.sum(axis=0)
        else:
            seasonalities: pd.Series = ft.median(axis=0)
        return seasonalities

    def scale_data(
            self,
            norm_mode: str,
            data: np.ndarray | pd.DataFrame,
            errors: np.ndarray | pd.DataFrame | None = None,
            y_bounds: tuple[int | float, int | float] | None = None
    ):
        modes = ['normal_distribution', 'upper_lower', 'contrast', 'none']
        assert norm_mode in modes, \
            f"choose one of {modes}, not {norm_mode=}"

        # std = 1, mean = 0
        if norm_mode.lower() == 'normal_distribution':
            if y_bounds is None:
                y_bounds = (-3, 3)
            data_scaled = StandardScaler().fit_transform(data)
        # bound between 0 and 1
        elif norm_mode.lower() == 'upper_lower':
            if y_bounds is None:
                y_bounds = (0, 1)
            data_scaled = rescale_values(
                data, new_min=y_bounds[0], new_max=y_bounds[1], axis=0
            )
        # bound between -1 and 1
        elif norm_mode.lower() == 'contrast':
            data_scaled = data.multiply(1 / data.abs().max(axis=0))
            y_bounds = (-1, 1)
        # dont scale
        elif norm_mode.lower() == 'none':
            if y_bounds is None:
                y_bounds = (data.min().min(), data.max().max())
            data_scaled = data
        else:
            raise KeyError('internal error')

        if errors is not None:
            pre_scale_mins: np.ndarray = np.array(np.nanmin(data, axis=0))
            pre_scale_maxs: np.ndarray = np.array(np.nanmax(data, axis=0))
            old_mins = (pre_scale_mins[None, :]
                        * np.ones(data.shape[0])[:, None])
            old_maxs = (pre_scale_maxs[None, :]
                        * np.ones(data.shape[0])[:, None])

            post_scale_mins: np.ndarray = np.array(np.nanmin(data_scaled, axis=0))
            post_scale_maxs: np.ndarray = np.array(np.nanmax(data_scaled, axis=0))
            new_mins = (post_scale_mins[None, :]
                        * np.ones(data.shape[0])[:, None])
            new_maxs = (post_scale_maxs[None, :]
                        * np.ones(data.shape[0])[:, None])

            # only apply the scaling, not the shifting
            scales = (new_maxs - new_mins) / (old_maxs - old_mins)
            errors_scaled = scales * errors
        else:
            errors_scaled = None

        # make sure comps_scaled is dataframe (e.g. StandardScalar does not return dataframe)
        if not isinstance(data_scaled, pd.DataFrame):
            data_scaled = pd.DataFrame(
                data=data_scaled, columns=data.columns, index=data.index
            )
        return data_scaled, y_bounds, errors_scaled

    def get_sign_weighted_table(
            self,
            feature_table: pd.DataFrame | None = None,
            use_L_contrasts=True,
            **kwargs
    ) -> pd.DataFrame:
        """Return table of weighted sign correlations."""
        if feature_table is None:
            feature_table = self.feature_table

        # get the weights
        w = self.get_weighted(
            feature_table,
            use_L_contrasts=use_L_contrasts,
            **kwargs
        ).weights
        N_w = len(w)

        def swc(a, b):
            """Calculate weighted sign correlation of a and b."""
            return np.sum((np.sign(a) == np.sign(b)) * w) / N_w * 2 - 1

        return feature_table.corr(method=swc)

    def get_sign_corr_table(
            self, feature_table: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return table of sign correlations."""
        def sc(a: np.ndarray, b: np.ndarray) -> float:
            """Calculate sign correlation of a and b."""
            return np.mean(np.sign(a) == np.sign(b)) * 2 - 1

        if feature_table is None:
            feature_table = self.feature_table

        return feature_table.corr(method=sc)

    def get_corr_with_grayscale(
            self,
            method: str | Callable = 'pearson',
            feature_table: pd.DataFrame | None = None,
            cols: Iterable | None = None,
            contrast: bool = False,
            weighted: bool = False,
            **kwargs
    ) -> pd.Series:
        """
        Calculate the correlation with the grayscale values for each compound.

        Parameters
        ----------
        method: str | Callable, optional
            Correlation method passed on to the pandas corrwith method. Default
            is "pearson"
        feature_table: pd.DataFrame, optional
            Feature table for which to calcualte the correlations. If not
            provided, will use the instances feature_table attribute. Please
            note that if you want to calculate the correlation of contrasts,
            you can either provide the feature table of contrasts and set
            contrast to False or provide the original values and set contrast
            to True
        cols: Iterable, optional
            Columns for which to calculate the correlations. But make sure that
            "L" is included, it will be attempted to take the L values from the
            class attribute, but this is less save.
        contrast: bool, optional
            Wether to calculate contrasts from the feature table. The default
            is False.
        weighted: bool, optional
            Wether to weigh the feature table (applied after contrasts). The
            default is False
        kwargs: Any
            Additional keywords for get_weights

        Returns
        -------
        corr_with_L: pd.Series
            The correlations with the L values for each compound.
        """
        if cols is None:
            cols: set = set(self.data_columns) | set('L')
        if feature_table is None:
            feature_table: pd.DataFrame = self.feature_table
        feature_table: pd.DataFrame = feature_table.copy().loc[:, cols]

        if 'L' not in feature_table.columns:
            logger.warning('"L" not in feature table, attempting to fetch it'
                           'from the class attribute')
            assert feature_table.shape[0] == self.feature_table.shape[0], \
                ('Trying to add L from feature table failed because input feature'
                 'table does not have the same length. Please make sure the '
                 'input feature table has the grayscale values')
            feature_table.loc[:, 'L'] = self.feature_table.L

        if contrast:
            feature_table: pd.DataFrame = self.get_contrasts_table(
                feature_table=feature_table
            )

        if weighted:
            feature_table = self.get_weighted(
                feature_table=feature_table,
                use_L_contrasts=kwargs.get('use_L_contrasts', True)
            )

        L: pd.Series = feature_table.L

        corr_with_L = feature_table.loc[:, cols].corrwith(L, method=method)

        return corr_with_L

    def power(
            self,
            targets: list[str] | None = None,
            plts: bool = False
    ) -> pd.DataFrame:
        if targets is None:
            targets = self.data_columns
        t: pd.Series[float] = self.feature_table.age
        N_points: int = len(t)
        ys: pd.DataFrame = self.feature_table.loc[:, targets]
        ys.fillna(0, inplace=True)
        ys: np.ndarray[int] = detrend(ys, axis=0)
        weights: np.ndarray[float] = blackman(N_points).reshape((N_points,))
        ys = (ys.T * weights.T).T

        res: list[tuple] = [LombScargle(t, ys[:, idx]).autopower()
                            for idx in range(len(targets))]
        frequencies: list[np.ndarray] = [r[0] for r in res]
        powers: list[np.ndarray] = [r[1] for r in res]
        # f_w, spec_w = LombScargle(t, y_w).autopower()
        # power
        # p = np.abs(spec) ** 2
        # p_w = np.abs(spec_w) ** 2

        if plts:
            fig, axs = plt.subplots(nrows=2)
            for idx, target in enumerate(targets):
                y = ys[:, idx]
                axs[0].plot(t, y, label=target)
                # axs[0].plot(t, y_d, label='detrended')
                # axs[0].plot(t, y_w, label='blackman')

                f = frequencies[idx]
                power = powers[idx]
                axs[1].plot(f, power, label=target)
                # axs[1].plot(f_w, p_w)

            axs[0].legend()
            axs[0].set_xlabel('Age in yrs b2k')
            axs[0].set_ylabel('Tapered time series')

            axs[1].legend()
            axs[1].set_xlabel('Frequency in 1 / yrs')
            axs[1].set_ylabel('Power')
            axs[1].vlines(1,
                          ymin=0,
                          ymax=np.max(np.array(powers)),
                          color='black',
                          linestyle='--')
            plt.show()

        assert all(np.allclose(frequencies[0], f) for f in frequencies)

        df = pd.DataFrame(dict(zip(targets, powers)))

        df['f'] = frequencies[0]

        return df

    def split_at_depth(self, depth: float | int) -> tuple[Self, Self]:
        """
        Split object at depth and return parts as time_series objects.

        depth: float.
            Depth at which to split in cm
        """
        assert 'depth' in self.feature_table.columns, \
            'Cannot split without depth column'
        min_depth = self.feature_table.depth.min()
        max_depth = self.feature_table.depth.min()
        # convert seed pixel coordinate to depth and depth to age
        x = self.feature_table.x_ROI.abs()
        # seed = 0 corresponds to min_depth
        # seed.max corresponds to max_depth (roughly)
        depths = self.feature_table.depth
        idx_T = np.argwhere(depths > depth)[0][0]
        path = self.path_folder
        if hasattr(self, 'd_folder'):
            path = os.path.join(path, self.d_folder)

        TSu = self.__class__(path, self.n_successes_required)
        TSl = TimeSeries(path, self.n_successes_required)
        for attr in [
                'feature_table_zone_averages',
                'feature_table_zone_standard_deviations',
                'feature_table_zone_successes'
        ]:
            if hasattr(self, attr):
                ftu = self.__getattribute__(attr).iloc[:idx_T, :]
                TSu.__setattr__(attr, ftu)
                ftl = self.__getattribute__(attr).iloc[idx_T:, :]
                TSl.__setattr__(attr, ftl)
        return TSu, TSl

    def get_stats(self) -> pd.DataFrame:
        cols = self.data_columns
        features = [
            'seasonality',
            'av', 'v_I_std', 'h_I_std',
            'contrast_med', 'contrast_std'
        ]
        df = pd.DataFrame(
            data=np.empty((len(features), len(cols)), dtype=float),
            columns=cols,
            index=features
        )

        df.loc['seasonality', cols] = self.get_seasonalities()
        df.loc['av', cols] = self.feature_table.loc[:, cols].mean(axis=0)
        # std of average intensities
        df.loc['v_I_std', cols] = self.feature_table.loc[
                                  :, cols
                                  ].std(axis=0)
        # median std is horizontal spread
        df.loc['h_I_std', cols] = self.deviations.loc[
                                  :, cols
                                  ].median(axis=0)
        # median abs contrast
        df.loc['contrast_med', cols] = self.get_contrasts_table().loc[
                                       :, cols
                                       ].abs().median(axis=0)
        # spread of contrasts --> std of contrasts
        df.loc['contrast_std', cols] = self.get_contrasts_table().loc[
                                       :, cols
                                       ].abs().std(axis=0)

        return df

    def plot_comp(
            self,
            comps: float | str | Iterable[float | str],
            color_seasons: bool = False,
            exclude_layers_low_successes: bool = False,
            errors: bool = True,
            title: str | None = None,
            colors: list | None = None,
            names: list | None = None,
            correct_tic: bool | Iterable[bool] = False,
            norm_mode: str = 'normal_distribution',
            contrasts: bool = False,
            annotate_l_correlations: bool = False,
            **kwargs
    ) -> None:
        """
        Plot a specific compound or list of compounds

        Parameters
        ----------
        comps: float | str | Iterable[float | str]
            The compound(s) to plot.
        color_seasons: bool
            If True, will add colored zones for the seasons (alpha value
            correlates to the quality). The default is False.
        exclude_layers_low_successes: bool, optional
            If True, will discard values below the required amount of successes.
            The default is False
        title: str, optional
            The figure title.
        norm_mode: str, optional
            Will scale values for visibility and comparability. See scale_data.
            The default values is "normal_distribution".
        errors: bool, optional
            Will shade the areas between interval spanned by center +/-
            standard error. The default is True.
        contrasts: bool, optional
            If this is set to True, will plot the contrasts. Default is False.
        annotate_l_correlations: bool, optional
            Whether to add correlation scores. Only available if contrasts is
            True.
        """
        def _filter_plot_data(comp_):
            if exclude_layers_low_successes and (comp_ in self.successes.columns):
                mask_valid = (self.successes.loc[:, comp_]
                              >= self.n_successes_required)
            else:
                mask_valid = np.ones_like(t, dtype=bool)
            _t = t[mask_valid]
            _values = data_scaled.loc[mask_valid, comp_]
            if errors_scaled is not None:
                _error = errors_scaled.loc[mask_valid, comp_]
            else:
                _error = None

            return mask_valid, _t, _values, _error

        def _add_seasons_coloring():
            # season_to_color = {-1: 'darkkhaki', 1: 'lightgoldenrodyellow'}
            season_to_color = {-1: 'blue', 1: 'red'}

            shades = self.feature_table.quality.to_numpy()
            shades[(shades < 0) | np.isnan(shades)] = 0
            shades = shades ** (1 / 3)
            shades = rescale_values(shades, new_min=0, new_max=1)
            seeds = (np.sign(self.feature_table.seed) * t).to_numpy()[mask_any]
            seasons = np.sign(seeds)[:-1]
            bounds = np.abs(seeds)[:-1] + np.diff(np.abs(seeds)) / 2
            bounds = np.insert(bounds, [0, -1], [t.min(), t.max()])

            for idx in range(seasons.shape[0]):
                c_index = seasons[idx]
                # skip nan vals
                if c_index not in season_to_color:
                    continue

                axs.axvspan(
                    bounds[idx],
                    bounds[idx + 1],
                    facecolor=season_to_color[c_index],
                    alpha=shades[idx],
                    edgecolor='none',
                    zorder=-1
                )

        def _add_yd_transition():
            if (t.min() < YD_transition) and (YD_transition < t.max()):
                logger.info('Pl-H transition in slice!')
                axs.vlines(YD_transition,
                           y_bounds[0],
                           y_bounds[1],
                           linestyles='solid',
                           alpha=.75,
                           label='Pl-H boundary',
                           color='black',
                           linewidth=2)

        if annotate_l_correlations and (not contrasts):
            logger.warning(
                'Cannot add correlations if contrasts is set to False.'
            )
            annotate_l_correlations = False
        if contrasts and correct_tic:
            logger.warning(
                'Correcting TIC for contrasts does not make sense because '
                'contrast values are scaled by neighbouring values'
            )

        t: pd.Series = self.age

        # put comps in list if it is only one
        if isinstance(comps, float | str):
            comps = [comps]
        n_comps: int = len(comps)

        if colors is not None:
            assert len(colors) == n_comps, \
                f'expected {n_comps=} colors, got {len(colors)}'
        else:
            colors = [f'C{idx + 2}' for idx in range(n_comps)]

        # find closest mz for comps in feature table
        comps: list = [
            self.get_closest_mz(comp)
            if (comp not in self.feature_table.columns)
            else comp
            for comp in comps
        ]

        if names is not None:
            assert len(names) == n_comps, \
                f'expected {n_comps=} names, got {len(names)}'
        else:
            names = comps

        if ('L' not in comps) and annotate_l_correlations:
            comps.append('L')

        # get data for relevant columns
        if hasattr(correct_tic, '__iter__'):
            assert len(correct_tic) == n_comps, 'Provide a value for each compound'
            data = self.feature_table.loc[:, comps].copy()
            data_scaled = self.get_feature_table_tic_corrected(**kwargs).loc[:, comps].copy()
            for comp, correct in zip(comps, correct_tic):
                if not correct:
                    continue
                data.loc[:, comp] = data_scaled.loc[:, comp]
        if correct_tic:
            data = self.get_feature_table_tic_corrected(**kwargs).loc[:, comps].copy()
        else:
            data = self.feature_table.loc[:, comps].copy()

        if errors:
            if contrasts:
                errors_data: pd.DataFrame = self._get_contrast_errors().loc[:, comps]
            else:
                errors_data: pd.DataFrame = self.errors.loc[:, comps]
            if correct_tic:
                errors_data = errors_data.divide(self._get_tic_scales(), axis=0)
        else:
            errors_data = None

        # contrasts will be returned from feature_table_zone_averages as it is already in there
        if contrasts:
            logger.info(f'getting contrasts for {data.columns}')
            data = self.get_contrasts_table(feature_table=data)

        data_scaled, y_bounds, errors_scaled = self.scale_data(
            norm_mode=norm_mode,
            data=data,
            errors=errors_data,
            y_bounds=kwargs.get('y_bounds')
        )

        if annotate_l_correlations:
            # split off L
            L: pd.Series = data_scaled.L
            data_scaled.drop(columns='L', inplace=True)
            comps.remove('L')

            # calculate weights
            w: pd.Series = (self.feature_table.contrast *
                            np.sign(self.feature_table.homogeneity))
            w[w < 0] = 0

            seas: pd.Series = self.get_seasonalities(cols=comps)

        fig, axs = plt.subplots(figsize=(10, 2))

        # mask to keep track of which layers have a compound with enough successful spectra
        mask_any = np.zeros_like(t, dtype=bool)
        for idx, comp in enumerate(comps):
            mask_comp, t_plot, values_plot, error_plot = _filter_plot_data(comp)

            if annotate_l_correlations:
                l_values: pd.Series = L.loc[mask_comp]
                # get corr with L
                rho = l_values.corr(values_plot, method='pearson')
                # get corr of signs
                s = sign_corr(l_values, values_plot)
                ws = sign_weighted_corr(l_values, values_plot, w[mask_comp])
                sea = seas[comp]
                label: str = (fr'{names[idx]} ($\rho_L$={rho:.2f}, $f_L=${s:.2f}, '
                              fr'$w_L=${ws:.2f}, $seas=${sea:.3f})')
            else:
                label: str = names[idx]

            if errors:
                axs.fill_between(t_plot,
                                 values_plot - error_plot,
                                 values_plot + error_plot,
                                 color=colors[idx],
                                 alpha=.5,
                                 zorder=-.5)
            axs.plot(
                t_plot,
                values_plot,
                label=label,
                color=colors[idx]
            )
            # add successful layers to mask
            mask_any |= mask_comp

        if color_seasons:
            _add_seasons_coloring()

        if annotate_l_correlations:
            s_l = sign_corr(
                self.feature_table.contrast[mask_any],
                self.feature_table.seed[mask_any]
            )

            ws_l = sign_weighted_corr(
                self.feature_table.contrast[mask_any],
                self.feature_table.seed[mask_any],
                w[mask_any]
            )
            label_l = fr'L ($f_s=${s_l:.2f}, $w_s=${ws_l:.2f})'

            if errors:
                L_errors = errors_scaled.loc[mask_any, 'L']
                axs.fill_between(t[mask_any],
                                 L[mask_any] - L_errors,
                                 L[mask_any] + L_errors,
                                 color='k',
                                 alpha=.5,
                                 zorder=-.5)
            axs.plot(
                t[mask_any],
                L[mask_any],
                label=label_l,
                color='k',
                alpha=.5
            )

        # add vertical line to mark the transition
        _add_yd_transition()

        # axs.set_xlim((t.min(), t.max()))
        axs.set_ylim(y_bounds)
        axs.set_xlabel('age (yr B2K)')
        axs.set_ylabel(f'{"scaled" if norm_mode != "none" else ""} '
                       f'{"contrasts" if contrasts else "intensities"}')
        if title is None:
            title = (f'excluded layers with less than '
                     f'{self.n_successes_required}: '
                     f'{exclude_layers_low_successes}, '
                     f'scaled mode: {norm_mode}')
        fig.suptitle(title)
        axs.grid(True)
        axs.legend()

        fig.tight_layout()
        plt.show()


class MultiSectionTimeSeries(TimeSeries):
    def __init__(self, time_series: list[TimeSeries], path_folder=None, **kwargs):
        if path_folder is None:
            try:
                path_folder = os.path.commonpath([ts.path_folder for ts in time_series])
            except ValueError as e:
                logger.warning(e)
                logger.warning(
                    'cannot set the path_folder, please provide on '
                    'initialization if you want to make use of save and load'
                )
        super().__init__(path_folder=path_folder, **kwargs)
        self._set_df(time_series)

    def _set_df(self, time_series: list[TimeSeries]):
        dfs = [ts.feature_table for ts in time_series]
        self._feature_table = combine_feature_tables(dfs)

        dfs = [ts.deviations for ts in time_series]
        self._feature_table_standard_deviations = combine_feature_tables(dfs)

        dfs = [ts.successes for ts in time_series]
        self._feature_table_successes = combine_feature_tables(dfs)


if __name__ == '__main__':
    pass
