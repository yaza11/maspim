from data.combine_feature_tables import combine_feature_tables
from util.cClass import Convinience, return_existing, verbose_function
from res.constants import elements, mC37_2, YD_transition, contrasts_scaling
from util.manage_obj_saves import class_to_attributes, Data_nondata_columns
from imaging.util.coordinate_transformations import rescale_values
from data.file_helpers import get_d_folder

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from sklearn.preprocessing import StandardScaler


class TimeSeries(Convinience):
    def __init__(self, path_folder: str) -> None:
        """Initialize."""
        self.plts = False
        self.verbose = False
        
        self.path_folder = path_folder

    @verbose_function
    def load(self):
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        path_d_folder = os.path.join(
            self.path_folder, 
            get_d_folder(self.path_folder)
        )
        with open(os.path.join(path_d_folder, name), 'rb') as f:
            obj = pickle.load(f)
        self.__dict__ |= obj.__dict__
        
    @verbose_function
    def save(self):
        """Save the object to disc."""
        dict_backup = self.__dict__.copy()
        keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
        existent_attributes = list(self.__dict__.keys())
        verbose = self.verbose
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)
        if verbose:
            print(f'saving image object with {self.__dict__.keys()}')
        name = str(self.__class__).split('.')[-1][:-2] + '.pickle'
        path_d_folder = os.path.join(
            self.path_folder, get_d_folder(self.path_folder)
        )
        with open(os.path.join(path_d_folder, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.__dict__ = dict_backup

    @verbose_function
    def set_time_series_tables(
            self, correct_zeros=False, Data_obj: object = None, **kwargs) -> None:
        """Get a feature table with zone wise averages."""
        raise NotImplementedError('Depricated, use Project and set_msi_time_series')
        if Data_obj is None:
            if self._data_type == 'msi':
                from cMSI import MSI
                Data_obj = MSI(self._section, self._window)
            elif self._data_type == 'xrf':
                from cXRF import XRF
                Data_obj = XRF(self._section, self._window)
            else:
                raise KeyError()
            Data_obj.verbose = self.verbose
            Data_obj.load(**kwargs)
        # check if all attributes are in feature table
        for c in Data_nondata_columns:
            assert c in Data_obj.get_nondata_columns(), f'{c} not in feature table.'

        # append L and x_ROI to features
        columns_feature_table = np.append(Data_obj.get_data_columns(), ['x_ROI', 'y_ROI', 'L'])

        # calculate the seedwise averages, stds and nonzeros
        ft_seeds_avg, ft_seeds_std, ft_seeds_success = Data_obj.processing_zone_wise_average(
            zones_key='seed',
            columns=columns_feature_table,
            correct_zeros=correct_zeros,
            calc_std=True,
            **kwargs
        )
        ft_seeds_avg = ft_seeds_avg.fillna(0)
        del Data_obj

        # add quality criteria
        from imaging.main.cImage import ImageClassified
        IClassified = ImageClassified(self._section, self._window)
        IClassified.load()
        cols_quals = ['homogeneity', 'continuity', 'contrast', 'quality']
        quals = IClassified.params_laminae_simplified.loc[:, ['seed', 'color'] + cols_quals].copy()
        del IClassified

        # sign the seeds in the quality dataframe
        mask_c = quals['color'] == 'light'
        quals.loc[mask_c, 'color'] = 1
        quals.loc[~mask_c, 'color'] = -1
        quals['seed'] *= quals['color']
        quals = quals.drop(columns='color')
        quals['seed'] = quals.seed.astype(int)
        quals = quals.set_index('seed')
        # join the qualities to the averages table
        ft_seeds_avg = ft_seeds_avg.join(quals, how='left')
        # insert infty for every column in success table that is not there yet
        missing_cols = set(ft_seeds_avg.columns).difference(
            set(ft_seeds_success.columns)
        )
        for col in missing_cols:
            ft_seeds_success[col] = np.infty

        # plot the qualities
        if self.plts:
            plt.figure()
            plt.plot(quals.index, quals.quality, '+', label='qual')
            plt.plot(ft_seeds_avg.index, ft_seeds_avg.quality, 'x', label='ft')
            plt.legend()
            plt.xlabel('seed')
            plt.ylabel('quality')
            plt.title('every x should have a +')
            plt.show()

        # need to insert the x_ROI from avg
        ft_seeds_std['spread_x_ROI'] = ft_seeds_std.x_ROI.copy()
        ft_seeds_success['N_total'] = ft_seeds_success.x_ROI.copy()
        ft_seeds_std['x_ROI'] = ft_seeds_avg.x_ROI.copy()
        ft_seeds_success['x_ROI'] = ft_seeds_avg.x_ROI.copy()
        # sort by depth
        ft_seeds_avg = ft_seeds_avg.sort_values(by='x_ROI')
        ft_seeds_std = ft_seeds_std.sort_values(by='x_ROI')
        ft_seeds_success = ft_seeds_success.sort_values(by='x_ROI')
        # drop index (=seed) into dataframe
        ft_seeds_avg.index.names = ['seed']
        ft_seeds_std.index.names = ['seed']
        ft_seeds_success.index.names = ['seed']
        # reset index
        ft_seeds_avg.reset_index(inplace=True)
        ft_seeds_avg['seed'] = ft_seeds_avg.seed.astype(int)
        ft_seeds_std.reset_index(inplace=True)
        ft_seeds_std['seed'] = ft_seeds_std.seed.astype(int)
        ft_seeds_success.reset_index(inplace=True)
        ft_seeds_success['seed'] = ft_seeds_success.seed.astype(int)

        self.feature_table = ft_seeds_avg
        self.feature_table_standard_deviations = ft_seeds_std
        self.feature_table_successes = ft_seeds_success

    def combine_duplicate_seed(self, weighted=False):
        """Combining layers with same seeds, information about quality is lost."""
        if weighted:
            raise NotImplementedError(
                'there is an issue with counts for quality criteria, check \
that before using this option'
            )
            seeds = self.feature_table.seed.copy()
            # take weighted average of rows that have the same xROI
            #   mult every comp in each layer by its n
            cols = self.get_data_columns() + \
                ['L', 'x_ROI', 'quality', 'homogeneity', 'continuity',
                 'contrast', 'quality'
                 ]
            ns = self.feature_table_successes.loc[:, cols]
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
            wmean = self.feature_table\
                .groupby('seed')\
                .mean()\
                .fillna(0)\
                .sort_values(by='x_ROI')\
                .reset_index()
            ns = self.feature_table_successes\
                .groupby('seed')\
                .mean()\
                .sort_values(by='x_ROI')\
                .reset_index()

        return wmean, ns

    @return_existing('feature_table_zone_averages')
    def get_feature_table(self):
        """Call the set function if attribute does not exist and return it."""
        assert hasattr(self, 'feature_table'), 'set the feature table first'
        return self.feature_table

    @return_existing('feature_table_zone_standard_deviations')
    def get_feature_table_standard_deviations(self):
        """Call the set function if attribute does not exist and return it."""
        assert hasattr(self, 'feature_table'), 'set the feature table first'
        return self.feature_table_standard_deviations

    @return_existing('feature_table_zone_successes')
    def get_feature_table_zone_successes(self):
        """Call the set function if attribute does not exist and return it."""
        assert hasattr(self, 'feature_table'), 'set the feature table first'
        return self.feature_table_successes


    def get_feature_table_standard_errors(self):
        assert hasattr(self, 'feature_table'), 'set the feature table first'
        if hasattr(self, 'feature_table_standard_errors'):
            return self.feature_table_standard_errors
        # standard error: sigma / sqrt(n)
        self.feature_table_standard_errors = self.feature_table_standard_deviations.loc[:, self.sget_data_columns()].div(
            np.sqrt(self.feature_table_successes.loc[:, 'N_total']), axis='rows'
        )
        return self.feature_table_standard_errors

    def get_data_columns(self):
        columns = self.get_feature_table().columns
        columns_valid = []
        columns_xrf = [col for col in columns if
                       col in list(elements.Abbreviation)]
        columns_msi = [col for col in columns if str(
            col).replace('.', '', 1).isdigit()]
        columns_valid = columns_xrf + columns_msi
        return columns_valid

    @return_existing('data_columns')
    def sget_data_columns(self) -> np.ndarray[float | str]:
        """Get columns with MSI/XRF measurments in feature table."""
        columns_valid = self.get_data_columns()
        self.data_columns = np.array(columns_valid)
        return self.data_columns

    def get_weighted(
            self,
            feature_table: pd.DataFrame,
            use_L_contrasts: bool,
            **kwargs
    ) -> pd.DataFrame:
        """
        Return average intensities weighted by quality of layers.

        Parameters
        ----------
        feature_table : pd.DataFrame
            The feature table in which to add the weights.
        use_L_contrasts : bool
            If True, will use contrasts signed by homogeinity, otherwise the
            geometric mean of the quality.
        **kwargs : dict
            Will be ignored.

        Returns
        -------
        data_weighted : pd.DataFrame
            The feature table weighted by the quality.

        """
        # get quality column
        if not use_L_contrasts:
            if self.verbose:
                print('using geometric mean of quality')
            q = self.get_feature_table().quality
            # calc geometric mean
            q = q.abs().pow(1 / 3)
        # use contrasts of grayscale
        else:
            if self.verbose:
                print('using signed contrasts for quality')
            # try to take criteria from feature table
            if ('contrast' in feature_table.columns) and \
                    ('homogeneity' in feature_table.columns):
                if self.verbose:
                    print('taking contrast, hom from passed ft.')
                c = feature_table.contrast.copy()
                h = feature_table.homogeneity.copy()
            else:
                if self.verbose:
                    print('taking contrast, hom from get-fcts.')
                c = self.get_feature_table().contrast
                h = self.get_feature_table().homogeneity
            q = c * np.sign(h) * contrasts_scaling

        # set qs below 0 to 0
        mask_less_than_zero = q < 0
        q[mask_less_than_zero] = 0

        # weigh data by geometric mean of quality
        data_weighted = feature_table.multiply(q, axis=0)
        # add weights columns
        data_weighted['weights'] = q
        return data_weighted

    def get_contrasts_table(
            self,
            feature_table: pd.DataFrame | None = None,
            subtract_mean=False
    ) -> pd.DataFrame:
        """
        Return the contrasts for each layer in the averages table.

        Parameters
        ----------
        feature_table : pd.DataFrame | None, optional
            The feature table for which to calculate the conrasts.
            The default is None.
            This will default to get_feature_table_zone_averages
        columns : Iterable | None, optional
            T. The default is None.

        Returns
        -------
        ft_contrast : pd.DataFrame
            The contrasts for each component and layer.
            Reflecting boundary conditions will be applied.
            x, x_ROI, seed will be copied over from get_feature_table_zone_averages,
            contrast will from get_feature_table_zone_averages will be renamed to L
        """
        if feature_table is None:
            feature_table = self.get_feature_table()

        columns = feature_table.columns

        # contrast calculation
        brightnesses = feature_table.to_numpy()
        if subtract_mean:
            brightnesses = np.subtract(brightnesses, brightnesses.mean(axis=0))
        # add boundary
        brightnesses_bound = np.pad(brightnesses, ((1, 1), (0, 0)), mode='reflect')
        # define slices
        slice_center = np.index_exp[1:-1, :]
        slice_up = np.index_exp[:-2, :]
        slice_down = np.index_exp[2:, :]
        # get neighbour, center values
        neighbour_up = brightnesses_bound[slice_up]
        neighbour_down = brightnesses_bound[slice_down]
        neighbours = (neighbour_up + neighbour_down) / 2
        center = brightnesses_bound[slice_center]
        # calc contrast (prevent division by zero)
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
        ft_contrast['x'] = self.get_feature_table().x.copy()
        ft_contrast['x_ROI'] = self.get_feature_table().x_ROI.copy()
        ft_contrast['seed'] = self.get_feature_table().seed.copy()
        ft_contrast['L'] = self.get_feature_table().contrast.copy()

        return ft_contrast

    def get_envelope(self, comp: str) -> tuple[np.ndarray]:
        """
        Return envelopes in the contrast signal for a given compound.

        This function finds local extrma and linearly interpolates values for
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
        v = self.get_contrasts_table().loc[:, comp]
        idxs_local_mins = (np.diff(np.sign(np.diff(v))) > 0).nonzero()[0] + 1
        idxs_local_maxs = (np.diff(np.sign(np.diff(v))) < 0).nonzero()[0] + 1

        # interpolate to all indices
        idxs = np.arange(len(v), dtype=int)
        envelope_min = np.interp(idxs, idxs_local_mins, v[idxs_local_mins])
        envelope_max = np.interp(idxs, idxs_local_maxs, v[idxs_local_maxs])
        return envelope_min, envelope_max

    def get_seasonality(self, comp, window_size, dt=.5):
        comp = self.get_closest_mz(comp)
        v = self.get_contrasts_table().loc[:, comp]
        t = self.age_scale()
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

        if self.plts:
            plt.plot(t, v, label=str(comp))
            plt.plot(t, e_min, label='envelope', color='darkblue')
            plt.plot(t, e_max, color='darkblue')
            plt.plot(t_i, s_i, label='amp.')
            plt.plot(t_, s_, label=f'smoothed amp. ({window_size * dt:.1f} yrs)')

            if (t.min() < YD_transition) and (YD_transition < t.max()):
                if self.verbose:
                    print('Pl-H transition in slice!')
                # add vertical line to mark the transition
                plt.vlines(YD_transition, -1, 1, linestyles='solid', alpha=.75, label='Pl-H boundary', color='black', linewidth=2)

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
            ft=None,
            weighted=True,
            exclude_low_success=True,
            mult_N=True,
            norm_weights=False,
            subtract_mean=False,
            n_successes_required=10
    ):
        if ft is None:
            ft = self.get_contrasts_table(subtract_mean=subtract_mean).loc[:, self.get_data_columns()]
            cols = self.get_data_columns()
        else:
            cols = ft.columns
        # multiply contrasts of comp with sign seed (comp with high summer
        # seasonality should only have positive signs, winter comp only negative after multiplication)
        ft = ft.multiply(np.sign(self.get_feature_table().seed), axis=0)
        # weigh contrasts by quality
        if weighted:
            ft = self.get_weighted(ft, use_L_contrasts=True)
            weights = ft.weights
            ft = ft.loc[:, cols]
            if norm_weights:
                scaling = weights.sum()
                ft /= scaling

        if exclude_low_success:
            ft_succ = self.get_feature_table().loc[:, cols].copy()
            ft[ft_succ < n_successes_required] = np.nan

        # multiply with ratio of successful layers
        if exclude_low_success and mult_N:
            r_succs = (ft_succ > n_successes_required).sum(axis=0) / ft_succ.shape[0]
            ft = ft.multiply(r_succs, axis=1)

        # take median
        if norm_weights:
            seasonalities = ft.sum(axis=0)
        else:
            seasonalities = ft.median(axis=0)
        return seasonalities

    def scale_data(self, norm_mode, data, y_bounds=None):
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
            raise KeyError(f"choose one of 'normal_distribution', \
'upper_lower', 'none', 'contrast' for norm_mode, not {norm_mode=}")

        # make sure comps_scaled is dataframe (e.g. StandardScalar does not return dataframe)
        if not isinstance(data_scaled, pd.DataFrame):
            data_scaled = pd.DataFrame(
                data=data_scaled, columns=data.columns, index=data.index
            )
        return data_scaled, y_bounds

    def plt_comp(
            self,
            comps: float | str | Iterable[float | str],
            color_seasons: bool = False,
            exclude_layers_low_successes: bool = False,
            title: str | None = None,
            norm_mode: str = 'normal_distribution',
            y_bounds: tuple[float, float] = None,  # plotting and scaling
            plt_contrasts: bool = False,
            **kwargs
    ) -> None:
        """Plot a specific compound or list of compounds"""
        t = self.age_scale()
        # put comp in list if it is only one
        if isinstance(comps, float | str):
            comps = [comps]
        assert type(comps) is list, 'pass comps as list type'

        # find closest mz for comps in feature table
        comps: list = [self.get_closest_mz(comp)
                       if (comp not in self.get_feature_table().columns)
                       else comp
                       for comp in comps]

        # get data for relevant columns
        data = self.get_feature_table().loc[:, comps].copy()

        # contrasts will be returned from feature_table_zone_averages as it is already in there
        if plt_contrasts:
            if self.verbose:
                print(f'getting contrasts for {data.columns}')
            data = self.get_contrasts_table(feature_table=data)

        data_scaled, y_bounds = self.scale_data(
            norm_mode=norm_mode, data=data, y_bounds=y_bounds)

        plt.figure(figsize=(10, 2))

        # mask to keep track of which layers have a compound with enough successful spectra
        mask_any = np.zeros_like(t, dtype=bool)
        for idx, comp in enumerate(comps):
            if exclude_layers_low_successes and (comp in self.get_feature_table_successes().columns):
                mask_enough_successes = self.get_feature_table_successes()\
                    .loc[:, comp] >= n_successes_required
            else:
                mask_enough_successes = np.ones_like(t, dtype=bool)
            v = data.loc[mask_enough_successes, comp]

            plt.plot(
                t[mask_enough_successes],
                data_scaled.loc[mask_enough_successes, comp],
                label=fr'{comp}',
                color=f'C{idx+2}'
            )
            # add successful layers to mask
            mask_any |= mask_enough_successes

        # season_to_color = {-1: 'darkkhaki', 1: 'lightgoldenrodyellow'}
        season_to_color = {-1: 'blue', 1: 'red'}
        if color_seasons:
            shades = self.get_feature_table().quality.to_numpy()
            shades[shades < 0] = 0
            shades = shades ** (1 / 3)
            shades = rescale_values(shades, new_min=0, new_max=1)
            seeds = (np.sign(self.get_feature_table().seed) * t).to_numpy()[mask_any]
            seasons = np.sign(seeds)[:-1]
            bounds = np.abs(seeds)[:-1] + np.diff(np.abs(seeds)) / 2
            bounds = np.insert(bounds, [0, -1], [t.min(), t.max()])
            ax = plt.gca()

            for idx in range(seasons.shape[0]):
                ax.axvspan(bounds[idx], bounds[idx + 1], facecolor=season_to_color[seasons[idx]], alpha=shades[idx], edgecolor='none', zorder=-1)

        # add vertical line to mark the transition
        if (t.min() < YD_transition) and (YD_transition < t.max()):
            if self.verbose:
                print('Pl-H transition in slice!')
            plt.vlines(YD_transition, y_bounds[0], y_bounds[1], linestyles='solid', alpha=.75, label='Pl-H boundary', color='black', linewidth=2)

        plt.xlim((t.min(), t.max()))
        plt.ylim(y_bounds)
        plt.xlabel('age (yr B2K)')
        plt.ylabel('scaled intensity')
        if title is None:
            title = f'excluded layers with less than {n_successes_required}: {exclude_layers_low_successes}, scaled mode: {norm_mode}'
            if plt_contrasts:
                title += ', contrasts'
        plt.title(title)
        plt.grid('on')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plt_against_grayscale(
            self,
            comps: float | str | Iterable[float | str],
            color_seasons: bool = False,
            exclude_layers_low_successes: bool = False,
            title: str | None = None,
            norm_mode: str = 'normal_distribution',
            y_bounds: tuple[float, float] = None,  # plotting and scaling
            plt_contrasts: bool = False,
            annotate_correlations=True,
            hold=False,
            ** kwargs
    ) -> None:
        """Plot a specific compound or list of compounds against grayscale."""
        t = self.age_scale()
        # put comp in list if it is only one
        if isinstance(comps, float | str):
            comps = [comps]
        assert type(comps) is list, 'pass comps as list type'

        # find closest mz for comps in feature table
        comps: list = [self.get_closest_mz(comp)
                       if (comp not in self.get_feature_table().columns)
                       else comp
                       for comp in comps]

        # make sure L is in comps
        if 'L' not in comps:
            comps.append('L')

        # get data for relevant columns
        data = self.get_feature_table().loc[:, comps].copy()

        # contrasts will be returned from feature_table_zone_averages as it is already in there
        if plt_contrasts:
            if self.verbose:
                print(f'getting contrasts for {data.columns}')
            data = self.get_contrasts_table(feature_table=data)

        data_scaled, y_bounds = self.scale_data(
            norm_mode=norm_mode, data=data, y_bounds=y_bounds)

        # split off L
        L = data_scaled.L
        data_scaled = data_scaled.drop(columns='L')
        comps.remove('L')

        plt.figure(figsize=(10, 2))

        # weights to use for the weighted sign correlation calculation
        if plt_contrasts and annotate_correlations:
            w = contrasts_scaling * \
                self.get_feature_table().contrast * \
                np.sign(self.get_feature_table().homogeneity)
            w[w < 0] = 0

        # mask to keep track of which layers have a compound with enough successful spectra
        mask_any = np.zeros_like(t, dtype=bool)
        for idx, comp in enumerate(comps):
            if exclude_layers_low_successes and (comp in self.get_feature_table_successes().columns):
                mask_enough_successes = self.get_feature_table_successes()\
                    .loc[:, comp] >= n_successes_required
            else:
                mask_enough_successes = np.ones_like(t, dtype=bool)
            v = data.loc[mask_enough_successes, comp]
            if plt_contrasts and annotate_correlations:
                # get corr with L
                rho = data.L.loc[mask_enough_successes].corr(v, method='pearson')
                # get corr of signs
                s = self.sign_corr(
                    data.L.loc[mask_enough_successes], v)

                ws = self.sign_weighted_corr(
                    data.L.loc[mask_enough_successes],
                    v,
                    w[mask_enough_successes]
                )
                seas = self.get_seasonalities()[comp]
                label = fr'{comp} ($\rho_L$={rho:.2f}, $f_L=${s:.2f}, $w_L=${ws:.2f}, $seas=${seas:.3f})'
            else:
                label = fr'{comp}'

            plt.plot(
                t[mask_enough_successes],
                data_scaled.loc[mask_enough_successes, comp],
                label=label,
                color=f'C{idx+2}'
            )
            # add successful layers to mask
            mask_any |= mask_enough_successes

        if plt_contrasts:
            s = self.sign_corr(
                self.get_feature_table().contrast[mask_any],
                self.get_feature_table().seed[mask_any]
            )

            ws = self.sign_weighted_corr(
                self.get_feature_table().contrast[mask_any],
                self.get_feature_table().seed[mask_any],
                w[mask_any]
            )
            label = fr'L ($f_s=${s:.2f}, $w_s=${ws:.2f})'
        else:
            label = 'L'

        plt.plot(
            t[mask_any],
            L[mask_any],
            label=label,
            color='k',
            alpha=.5
        )

        # season_to_color = {-1: 'darkkhaki', 1: 'lightgoldenrodyellow'}
        season_to_color = {-1: 'blue', 1: 'red'}
        if color_seasons:
            shades = self.get_feature_table().quality.to_numpy()
            shades[shades < 0] = 0
            shades = shades ** (1 / 3)
            shades = rescale_values(shades, new_min=0, new_max=1)
            seeds = (np.sign(self.get_feature_table().seed) * t).to_numpy()[mask_any]
            seasons = np.sign(seeds)[:-1]
            bounds = np.abs(seeds)[:-1] + np.diff(np.abs(seeds)) / 2
            bounds = np.insert(bounds, [0, -1], [t.min(), t.max()])
            ax = plt.gca()

            for idx in range(seasons.shape[0]):
                ax.axvspan(bounds[idx], bounds[idx + 1], facecolor=season_to_color[seasons[idx]], alpha=shades[idx], edgecolor='none', zorder=-1)

        # add vertical line to mark the transition
        if (t.min() < YD_transition) and (YD_transition < t.max()):
            if self.verbose:
                print('Pl-H transition in slice!')
            plt.vlines(YD_transition, y_bounds[0], y_bounds[1], linestyles='solid', alpha=.75, label='Pl-H boundary', color='black', linewidth=2)

        plt.xlim((t.min(), t.max()))
        plt.ylim(y_bounds)
        plt.xlabel('age (yr B2K)')
        plt.ylabel('scaled intensity')
        if title is None:
            title = f'excluded layers with less than {n_successes_required}: {exclude_layers_low_successes}, scaled mode: {norm_mode}'
            if plt_contrasts:
                title += ', contrasts'
        plt.title(title)
        plt.grid('on')
        plt.legend()

        plt.tight_layout()
        if not hold:
            plt.show()

    def plt_top(self, series_scores: pd.Series, N_top=5, title='', **kwargs):
        # sorted lowest to highest
        series_scores = series_scores.sort_values()
        self.plt_against_grayscale(list(series_scores.index[:N_top]), title=f'Top {N_top} dark' + title, **kwargs)
        self.plt_against_grayscale(list(series_scores.index[-N_top:][::-1]), title=f'Top {N_top} light' + title, **kwargs)

    def sign_corr(self, a: Iterable, b: Iterable) -> float:
        """
        Return the fraction of like signs over overall signs.

        The value is normed to be between -1 and 1 where 1 is returned, if all
        signs match and -1 if all signs oppose. Nans will be ignored.
        a and b must have the same length.

        Parameters
        ----------
        a : Iterable
            DESCRIPTION.
        b : Iterable
            DESCRIPTION.

        Returns
        -------
        float
            number of same signs over number of entries.

        """
        assert len(a) == len(b), 'a and b must have same length'
        r = np.nanmean(np.sign(a) == np.sign(b)) * 2 - 1
        return r

    def sign_weighted_corr(self, a: Iterable, b: Iterable, w: Iterable) -> float:
        """
        Return the fraction of like signs over overall signs.

        The value is normed to be between -1 and 1 where 1 is returned, if all
        signs match and -1 if all signs oppose. Nans will be ignored.
        a and b must have the same length.

        Parameters
        ----------
        a : Iterable
            DESCRIPTION.
        b : Iterable
            DESCRIPTION.
        w : Iterable
            The weights for each dimension

        Returns
        -------
        float
            number of same signs over number of entries weighted.

        """
        assert len(a) == len(b), 'a and b must have same length'
        assert len(a) == len(w), 'weights must have same length as a and b'
        assert np.min(w) >= 0, 'weights should be bigger than 0'

        r = np.sum((np.sign(a) == np.sign(b)) * w) / len(w) * 2 - 1
        return r

    def get_sign_weighted_table(
            self, feature_table: pd.DataFrame | None = None, use_L_contrasts=True, **kwargs
    ) -> pd.DataFrame:
        """Return table of weighted sign correlations."""
        if feature_table is None:
            feature_table = self.get_feature_table_zone_averages()

        # get the weights
        w = self.get_weighted(feature_table, use_L_contrasts=use_L_contrasts, **kwargs).weights
        N_w = len(w)

        def swc(a, b):
            """Calculat weighted sign correlation of a and b."""
            return np.sum((np.sign(a) == np.sign(b)) * w) / N_w * 2 - 1

        return feature_table.corr(method=swc)

    def get_sign_table(
            self, feature_table: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return table of sign correlations."""
        if feature_table is None:
            feature_table = self.get_feature_table()

        def sc(a, b):
            """Calculate sign correlation of a and b."""
            return np.mean(np.sign(a) == np.sign(b)) * 2 - 1

        return feature_table.corr(method=sc)

    def get_corr_with_grayscale(
            self, method='pearson', comps=None, contrast=False, weighted=False
    ) -> pd.Series:
        if weighted:
            ft = self.sget_data_quality_weighted()
        if contrast:
            if weighted:
                ft = self.get_contrasts_table(feature_table=ft)
            else:
                ft = self.get_contrasts_table()
        else:
            ft = self.get_feature_table()
        L = ft.L

        if comps is None:
            comps = self.sget_data_columns()
        corr_with_L = ft.loc[:, comps].corrwith(L, method=method)
        return corr_with_L

    def correct_distortion(self):
        raise NotImplemented('Depricated')
        """Correct x_ROI, y_ROI by image transformation."""
        if self._window == transformation_target:
            return

        # set global flag to make sure transformation is not applied twice
        if self.check_attribute_exists('applied_depth_correction') and self.applied_depth_correction:
            if self.verbose:
                print('depth correction has already been applied')
            return

        from imaging.main.cTransformation import apply_stretching, ImageTransformation
        from imaging.main.cImage import ImageROI
        IT = ImageTransformation(self._section, self._window, transformation_target)
        IT.load_transformed()
        # apply stretching to seed and x_ROI
        # stretching function assumes input vec spans entire section
        IR = ImageROI(self._section, self._window)
        IR.load()
        # x_ROI's to transform
        x_ROI = self.feature_table.x_ROI.copy()
        y_ROI = self.feature_table.y_ROI.copy()
        # original vector
        # x_ROI_i = np.arange(0, IR.xywh_ROI[-2], 1)
        # x_ROI_T =
        d, i = IT.apply_mapping_to_points(np.c_[x_ROI, y_ROI])
        y_ROI_T, x_ROI_T = np.unravel_index(
            i, IT.X_transformed.shape)

        self.feature_table['x_ROI'] = x_ROI_T
        self.feature_table['y_ROI'] = y_ROI_T

        self.feature_table_standard_deviations['x_ROI'] = x_ROI_T
        self.feature_table_standard_deviations['y_ROI'] = y_ROI_T

        self.feature_table_successes['x_ROI'] = x_ROI_T
        self.feature_table_successes['y_ROI'] = y_ROI_T

        # set flag
        self.applied_depth_correction = True

        if self.plts:
            plt.scatter(x_ROI, y_ROI)
            plt.title('scatter before')
            plt.show()
            plt.scatter(x_ROI_T, y_ROI_T)
            plt.title('scatter after')
            plt.show()
            plt.plot(x_ROI - x_ROI_T, label=r'$\Delta x_\mathrm{ROI}$')
            plt.plot(y_ROI - y_ROI_T, label=r'$\Delta y_\mathrm{ROI}$')
            plt.plot(np.sqrt((x_ROI - x_ROI_T) ** 2 + (y_ROI - y_ROI_T) ** 2), label='distance')
            plt.plot(d, label=r'err')
            plt.legend()
            plt.title('coordinates before - after')
            plt.show()

    def set_age_scale(self, ages):
        self.ages = ages

    def age_scale(self):
        assert hasattr(self, 'ages'), 'set ages with set_ages first'
        return self.ages

    def split_at_depth(self, depth) -> tuple[object]:
        raise NotImplementedError('Depricated')
        """Split object at depth and return parts as TS objects.

        depth: float.
            Depth at which to split in cm"""
        min_depth = self._section[0]  # --> m
        max_depth = self._section[1]  # --> m
        # convert seed pixel coordinate to depth and depth to age
        x = self.feature_table.x_ROI.abs()
        # seed = 0 corresponds to min_depth
        # seed.max corresponds to max_depth (roughly)
        depths = rescale_values(x, new_min=min_depth, new_max=max_depth, old_min=x.min(), old_max=x.max())
        idx_T = np.argwhere(depths > depth)[0][0]
        TSu = TimeSeries((self._section[0], depth), self._window)
        TSl = TimeSeries((depth, self._section[1]), self._window)
        for attr in ['feature_table_zone_averages', 'feature_table_zone_standard_deviations', 'feature_table_zone_successes', 'feature_table_zone_averages_clean', 'feature_table_zone_successes_clean']:
            if hasattr(self, attr):
                ftu = self.__getattribute__(attr).iloc[:idx_T, :]
                TSu.__setattr__(attr, ftu)
                ftl = self.__getattribute__(attr).iloc[idx_T:, :]
                TSl.__setattr__(attr, ftl)
        return TSu, TSl

    def get_stats(self):
        cols = self.get_data_columns()
        features = [
            'seasonality',
            'av', 'v_I_std', 'h_I_std',
            'contrast_med', 'contrast_std'
        ]
        df = pd.DataFrame(data=np.empty((len(features), len(cols)), dtype=float), columns=cols, index=features)

        df.loc['seasonality', cols] = self.get_seasonalities()
        df.loc['av', cols] = self.feature_table.loc[:, cols].mean(axis=0)
        # std of average intensities
        df.loc['v_I_std', cols] = self.feature_table.loc[
            :, cols
        ].std(axis=0)
        # median std is horizontal spread
        df.loc['h_I_std', cols] = self.feature_table_standard_deviations.loc[
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


class MultiSectionTimeSeries(TimeSeries):
    def __init__(self, time_series: list[TimeSeries]):
        self.plts = False
        self.verbose = False

        self._set_df(time_series)

    def _set_df(self, time_series: list[TimeSeries]):
        dfs = [ts.feature_table for ts in time_series]
        self.feature_table = combine_feature_tables(dfs)

        dfs = [ts.feature_table_standard_deviations for ts in time_series]
        self.feature_table_standard_deviations = combine_feature_tables(dfs)

        dfs = [ts.feature_table_successes for ts in time_series]
        self.feature_table_successes = combine_feature_tables(dfs)

def cpr_corrs():
    section = (490, 495)
    window = 'Alkenones'
    TS = TimeSeries(section, window)
    TS.plts = False
    TS.verbose = True
    # TS.sget_time_series_table()
    TS.load()
    TS.sget_contrasts_table()
    # correlations with grayscale
    L = TS.feature_table_zone_averages.L
    corr_with_L_p = TS.get_feature_table_zone_averages().loc[:, TS.sget_data_columns()].corrwith(L, method='pearson')
    TS.plt_top(corr_with_L_p, color_seasons=True, N_top=5, title=' corr with L (pearson)')

    corr_with_L_s = TS.get_feature_table_zone_averages().loc[:, TS.sget_data_columns()].corrwith(L, method='spearman')
    TS.plt_top(corr_with_L_s, color_seasons=True, N_top=5, title=' corr with L (spearman)')

    corr_with_L_k = TS.get_feature_table_zone_averages().loc[:, TS.sget_data_columns()].corrwith(L, method='kendall')
    TS.plt_top(corr_with_L_k, color_seasons=True, N_top=5, title=' corr with L (kendall)')

    # correlations with quality
    Q = TS.feature_table_zone_averages.quality
    corr_with_Q = TS.get_feature_table_zone_averages().loc[:, TS.sget_data_columns()].corrwith(Q, method='pearson')
    TS.plt_top(corr_with_Q, color_seasons=True, N_top=5, title=' corr with Q')

    # correlation of contrasts with contrast
    C = TS.feature_table_zone_averages.contrast
    corr_C_with_C = TS.sget_contrasts_table().loc[
        :, TS.sget_data_columns()
    ].corrwith(C, method='pearson')
    TS.plt_top(corr_C_with_C, color_seasons=True, N_top=5, title=' corr contrasts with contrast L')


def combine_sections(sections: list, window: str):
    """Combine time series of multiple sections."""
    from imaging.main.cImage import ImageProbe
    combined_section = (sections[0][0], sections[-1][-1])
    TS_combined = TimeSeries(combined_section, window)
    avs = []
    stds = []
    sucs = []
    x_ROI_offset = 0
    for section in sections:
        # add offset to x_ROI
        TS = TimeSeries(section, window)
        TS.set_time_series_tables(use_common_mzs=True)
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


if __name__ == '__main__':
    pass
