"""Gather meta-features for all compounds."""
from data.cMSI import MSI
from data.cXRF import XRF
from timeSeries.cTimeSeries import TimeSeries
from res.constants import YD_transition_depth, windows_all, dict_labels

from data.annotating import Homologous
from util.manage_obj_saves import class_to_attributes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


import pykrev as pk


def rankings_name_to_label(name):
    if name not in dict_labels:
        return name
    return dict_labels[name]


def PCA_biplot(
        score: pd.DataFrame,
        coeffs, labels=None, var=None, categories=None, add_labels=True, add_eigenvectors=True, hold=False, **kwargs):
    # coordinates in PCA coordinate system
    xs = score.iloc[:, 0]
    ys = score.iloc[:, 1]
    n = coeffs.shape[1]  # number of criteria
    # scale vectors to unit length
    range_x = xs.max() - xs.min()
    range_y = ys.max() - ys.min()
    xs /= range_x
    ys /= range_y
    # plot compounds in new frame of reference
    if categories is None:
        plt.scatter(xs, ys, alpha=.8)
    else:
        assert len(categories) == len(xs), f'entries in categories and points does not match ({len(categories)} != {len(xs)})'

        # colormap = np.array([f'C{i}' for i in range(len(np.unique(categories)))])
        sc = plt.scatter(xs, ys, alpha=.8, c=categories)
        plt.colorbar(sc)
    # add labels to point
    if add_labels:
        if labels is None:
            labels = score.index
        for i in range(score.shape[0]):
            plt.annotate(
                labels[i], score.iloc[i] + .01, alpha=0.5, zorder=-1, size=5)

    if add_eigenvectors:
        # plot arrows for the criteria
        for i in range(n):
            coeff_x = coeffs.iloc[0, i]
            coeff_y = coeffs.iloc[1, i]
            plt.arrow(
                0, 0, coeff_x, coeff_y, color='r', alpha=0.5
            )
            plt.arrow(
                0, 0, -coeff_x, -coeff_y, color='r', alpha=0.5,
                linestyle=':'
            )
            txt = rankings_name_to_label(coeffs.columns[i])
            plt.text(coeff_x * 1.15, coeff_y * 1.15, txt,
                     color='g', ha='center', va='center')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    if var is None:
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
    else:
        plt.xlabel(f'PC 1 ({var[0]:.1%})')
        plt.ylabel(f'PC 2 ({var[1]:.1%})')
    plt.axis('equal')
    plt.grid()
    if not hold:
        plt.show()


class MetaFeatures:
    def __init__(self, windows=None):
        # features in table
        self.features = [  # from calculate_light_dark_rankings
            'av_intensity', 'av_intensity_light', 'av_intensity_dark', 'KL_div_light',
            'KL_div_dark', 'intensity_div', 'density_nonzero', 'KL_div', 'score',
            'corr_L', 'corr_classification'
        ] + [  # from time series
            'seasonality', 'av', 'v_I_std', 'h_I_std',
            'contrast_med', 'contrast_std'
        ] + [  # from pykrev
            'DBE', 'AI', 'NOSC', 'mz_k', 'dmz_k'
        ]

        # features used for clustering
        self.features_clustering = ['seasonality', 'KL_div']

        if windows is None:
            windows = windows_all.copy()
        self.windows = windows
        self.windows.sort()
        self._window = '_'.join(self.windows)

        self.plts = False
        self.verbose = False
        self._section = (490, 510)

    def load(self):
        from util.manage_class_imports import load_obj
        """Actions to performe when object was loaded from disc."""
        if __name__ == '__main__':
            raise RuntimeError('Cannot load obj from file where it is defined.')
        self.__dict__ = load_obj(
            self._section, self._window, self.__class__.__name__
        ).__dict__
        self.plts = False
        self.verbose = False

    def save(self):
        from util.manage_class_imports import save_obj
        if __name__ == '__main__':
            raise RuntimeError('Cannot save object from the file in which it is defined.')
        # delete all attributes that are not flagged as relevant
        existent_attributes = list(self.__dict__.keys())
        keep_attributes = set(existent_attributes) & class_to_attributes(self)
        # create temporary storage in case user wants to continue after save
        verbose = self.verbose
        plts = self.plts
        for attribute in existent_attributes:
            if attribute not in keep_attributes:
                self.__delattr__(attribute)

        if verbose:
            print(f'saving {self.__class__} object with {self.__dict__.keys()}')
        save_obj(obj=self)
        self.verbose = verbose
        self.plts = plts

    def set_all_masses(self):
        # windows = ('Alkenones', 'GDGT', 'xrf')
        cmpds = []
        section = (490, 495)
        for window in self.windows:
            print(f'adding comps for {window}')
            m = MSI(section, window)
            if window == 'xrf':
                m = XRF(section, window)
                m.load()
            elif window in ('FA', 'Alkenones'):
                m = MSI(section, window)
                m.load()
            else:
                m = MSI(section, window)
                m.load(use_common_mzs=True)
            cmpds += list(m.get_data_columns().astype(str))
            del m

        self.cmpds = cmpds

    def get_split_data_tables_for_window(self, window: str = None):
        if window is None:
            window = self.windows[0]

        if window == 'FA':
            section = (490, 510)
        else:
            section = (490, 510)
        if window == 'xrf':
            m = XRF(section, window)
        else:
            m = MSI(section, window)
        m.load()
        u, l = m.split_at_depth(YD_transition_depth)
        return u, l

    def get_split_time_series_for_window(self, window: str = None):
        if window is None:
            window = self.windows[0]

        if window == 'FA':
            section = (490, 510)
        else:
            section = (490, 510)
        m = TimeSeries(section, window)
        m.load()
        u, l = m.split_at_depth(YD_transition_depth)
        return u, l

    def initiate_meta_feature_table(self):
        values = np.empty((len(self.features), len(self.cmpds)), dtype=float)

        self.df = pd.DataFrame(data=values, columns=self.cmpds, index=self.features)

    def set_data_obj_attrs(self, window=None):
        # fill with attributes from dataclass
        u = self.get_split_data_tables_for_window(window)[0]

        ru = u.calculate_lightdark_rankings(
            use_intensities=True, use_successes=True,
            use_KL_div=True, calc_corrs=True
        ).T

        self.df.loc[ru.index, ru.columns] = ru

        # calculate nonzero median intensity
        self.df.loc['av_intensity', list(u.get_data_columns())] = u.get_data_mean()

    def add_NMF_loadings(self, k, window=None, **kwargs):
        u = self.get_split_data_tables_for_window(window)[0]
        u.analyzing_NMF(k=k, **kwargs)
        for i in range(k):
            self.df.loc[f'nmf{i}', :] = u.H[i, :]

    def calculate_TOM_NMF_weights(self, k, beta=12, window=None, **kwargs):
        u = self.get_split_data_tables_for_window(window)[0]
        u.analyzing_NMF(k=k, **kwargs)
        u.plt_NMF(k)

        H = u.H

        H_df = pd.DataFrame(data=H, columns=u.get_data_columns())
        r = H_df.corr()
        diss = (r + 1) / 2
        # calculate TOM
        A = diss ** beta
        L = np.dot(A, A)
        k = np.sum(A, axis=0)
        d = len(k)
        tile = np.tile(k, (d, 1))
        K = np.min(np.stack((tile, tile.T), axis=2), axis=2)
        TOM = (L + A) / (K + 1 - A)
        np.fill_diagonal(TOM.to_numpy(), 1)

        TOM_df = pd.DataFrame(data=TOM, columns=H_df.columns, index=H_df.columns)
        self.TOM_NMF_weights = TOM_df

    def set_time_obj_attrs(self, window=None):
        # fill with attributes from timeseries
        u = self.get_split_time_series_for_window(window)[0]

        dfu = u.get_stats()
        self.df.loc[dfu.index, dfu.columns] = dfu

    def set_formulas(self):
        # get composition for each compound
        from data.annotating import dataBase
        formulas = np.empty(len(self.cmpds), dtype=object)
        for idx, comp in enumerate(self.cmpds):
            formula = dataBase.find(comp)
            if formula.replace('.', '').isnumeric():
                formulas[idx] = str(formula)
            else:
                formulas[idx] = formula

        self.formulas = dict(zip(self.cmpds, formulas))

    def get_pk_tup(self, mask_valid=None):
        if mask_valid is None:
            mask_valid = np.ones(len(self.cmpds), dtype=bool)
        mstup = pk.msTuple(
            list(np.array(list(self.formulas.values()))[mask_valid]),
            self.df.loc['av_intensity', :].to_numpy()[mask_valid].copy(),
            np.array(self.cmpds)[mask_valid].astype(float).copy()
        )
        return mstup

    def add_from_pykrev(self):
        """Return data in right format for pykrev."""

        # exclude masses without formula
        mask_invalid = np.array([
            f.replace('.', '').isnumeric()
            for f in self.formulas.values()
        ])
        mask_valid = ~mask_invalid

        mstup = self.get_pk_tup(mask_valid)
        mstup_all = self.get_pk_tup()

        # compounds without formulas get a value of 1
        self.df.loc['DBE', mask_valid] = pk.double_bond_equivalent(mstup)
        self.df.loc['AI', mask_valid] = pk.aromaticity_index(mstup, index_type='rAI')
        self.df.loc['NOSC', mask_valid] = pk.nominal_oxidation_state(
            mstup
        )
        # element counts
        elements_list_dict: list[dict] = pk.element_counts(mstup)
        elements_df = pd.DataFrame(elements_list_dict).T
        elements_df.columns = mstup.mz.astype(str)
        for name, entries in elements_df.iterrows():
            self.df.loc[name, entries.index] = entries
        # element ratios
        elementRatios_list_dict = pk.element_ratios(mstup, ratios=['HC', 'OH', 'OC', 'NC'])
        elementRatios_df = pd.DataFrame(elementRatios_list_dict).T
        elementRatios_df.columns = mstup.mz.astype(str)
        for name, entries in elementRatios_df.iterrows():
            self.df.loc[name, entries.index] = entries

        mz_k, dmz_k = pk.kendrick_mass_defect(mstup_all)
        self.df.loc['mz_k'] = mz_k  # kendrick mass
        self.df.loc['dmz_k'] = dmz_k  # kendrick mass defect

        self.df.loc['DBE', mask_invalid] = np.nan
        self.df.loc['AI', mask_invalid] = np.nan
        self.df.loc['NOSC', mask_invalid] = np.nan

    def pk_plots(self, mask_interest=None):
        if mask_interest is None:
            mask_interest = np.ones(len(self.cmpds), dtype=bool)

        # compounds without formula
        mask_has_formula = ~np.array([
            f.replace('.', '').isnumeric()
            for f in self.formulas.values()
        ])

        mask_valid = mask_interest & mask_has_formula

        # mz of interest with formula
        mstup = self.get_pk_tup(mask_valid)
        mstup_all = self.get_pk_tup()

        # Van Krevelen diagrams
        # plt.figure()
        # dbe = self.df.loc['DBE', mask_valid].copy()
        # pk.van_krevelen_plot(mstup_valid, y_ratio='HC', c=dbe, s=7, cmap='plasma')
        # cbar = plt.colorbar()  # add a colour bar
        # cbar.set_label('Double bond equivalence')
        # plt.show()

        plt.figure()
        pk.van_krevelen_plot(mstup, y_ratio='HC', s=7)
        plt.colorbar().set_label('Kernel Density')
        plt.show()

        plt.figure()
        pk.van_krevelen_plot(mstup, y_ratio='NC', s=7)
        plt.show()

        # plt.figure()
        # fig, ax, d_index = pk.van_krevelen_histogram(
        #     mstup_valid, bins=[10, 10], cmap='viridis')
        # cbar = plt.colorbar()
        # cbar.set_label('Counts')
        # plt.show()

        # plt.figure()
        # fig, ax, d_index = pk.van_krevelen_histogram(
        #     mstup_valid, bins=[np.linspace(0, 1, 5), np.linspace(0, 2, 5)], cmap='cividis')
        # cbar = plt.colorbar()
        # cbar.set_label('Counts')
        # plt.show()

        # kendrick mass defect plot
        plt.figure()
        fig, ax, (kendrickMass, kendrickMassDefect) = pk.kendrick_mass_defect_plot(
            mstup_all, base='CH2', rounding='even', s=3)
        # plt.colorbar().set_label('Aromaticity Index')
        plt.show()

        # atomic class plots
        plt.figure()
        element = 'O'
        fig, ax, (mean, median, sigma) = pk.atomic_class_plot(
            mstup_valid,
            element=element,
            color='c',
            summary_statistics=True,
            bins=range(
                int(self.df.loc[element].min()),
                int(self.df.loc[element].max() + 1)
            )
        )
        plt.show()

        plt.figure()
        element = 'N'
        fig, ax, (mean, median, sigma) = pk.atomic_class_plot(
            mstup,
            element=element,
            color='c',
            summary_statistics=True,
            bins=range(
                int(self.df.loc[element].min()),
                int(self.df.loc[element].max() + 1)
            )
        )
        plt.show()

        # compounds class plots
        plt.figure()
        pk.compound_class_plot(mstup, color='g', method='MSCC')
        plt.show()
        # mass histogram
        plt.figure()
        fig, ax, (mean, median, sigma) = pk.mass_histogram(
            mstup,
            method='monoisotopic',
            bin_width=20,
            summary_statistics=True,
            color='blue',
            alpha=0.5,
            kde=True,
            kde_color='blue',
            density=False
        )
        plt.xlabel('Monoisotopic atomic mass (Da)')
        plt.show()

    def overview_plts(self, labels_class='label'):
        # intensities vs mass defects
        plt.figure()
        plt.scatter(self.df.loc['dmz_k'], self.df.loc['av'], s=7, c=self.df.loc[labels_class])
        plt.xlabel('Kendrick Mass Defect')
        plt.ylabel('Intensity')
        plt.colorbar().set_label('class label')
        plt.title('Intensities vs Kendrick Mass Defects')
        plt.show()
        # KMD vs KM
        plt.figure()
        plt.scatter(self.df.loc['mz_k'], self.df.loc['dmz_k'], s=7, c=self.df.loc[labels_class], alpha=.7)
        plt.xlabel('Kendrick Mass')
        plt.ylabel('Kendrick Mass defect')
        plt.colorbar().set_label('class label')
        plt.title('Kendrick Mass vs Kendrick Mass Defects')
        plt.show()
        # histograms O, N, S, P
        elmts = ["O", "N", "S", "P"]
        for elmt in elmts:
            Ns = self.df.loc[elmt]
            bins = range(int(Ns.min()), int(Ns.max() + 1))
            for label in np.unique(self.df.loc[labels_class]):
                mask_c = self.df.loc[labels_class] == label
                plt.hist(Ns[mask_c], bins=bins, label=int(label), color=f'C{int(label)}', alpha=.5)
            plt.title(f'{elmt} element abundances')
            plt.legend()
            plt.show()
        # 3D surface x: H/C, y: O/C, z: N/C
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        markers = ['o', 'v', 's', 'D', '^', 'p', '*', '1', '8']
        for label in np.unique(self.df.loc[labels_class]):
            mask_c = self.df.loc[labels_class] == label
            C = self.df.loc['C', mask_c]
            H = self.df.loc['H', mask_c]
            O = self.df.loc['O', mask_c]
            N = self.df.loc['N', mask_c]
            x = H / C
            y = O / C
            z = N / C
            ax.scatter(x, y, z, marker=markers[int(label) % len(markers)], label=int(label))
        ax.set_xlabel('H/C')
        ax.set_ylabel('O/C')
        ax.set_zlabel('N/C')
        plt.legend()
        plt.show()

        mask_valid = ~self.df.loc['DBE'].isna()
        self.features_clustering = ['HC', 'NC', 'OC']
        self.biplot(categories=self.df.loc[labels_class, mask_valid])

    def set_all_attrs(self):
        self.initiate_meta_feature_table()
        for window in self.windows:
            print(f'setting attributes for {window}')
            self.set_time_obj_attrs(window)
            self.set_data_obj_attrs(window)
        self.set_formulas()
        self.add_from_pykrev()

    def get_ft_clustering(self, only_valid=True):
        df = self.df.loc[self.features_clustering, :].copy().T

        invalid_comps = self.df.loc[self.features_clustering, :].isna().any(axis=0)
        if only_valid and invalid_comps.any():
            print(f'dropping {invalid_comps.sum()} compounds that contain nan values')
            df = df.loc[~invalid_comps, :]
        return df

    def biplot(self, df=None, **kwargs):
        if df is None:
            df = self.get_ft_clustering()

        columns = df.columns
        X = StandardScaler().fit_transform(df)
        # do PCA:
        pca = PCA(n_components=2)
        # compounds in new frame of reference
        pcs = pd.DataFrame(
            data=pca.fit_transform(X),
            columns=[f'PC {i + 1}' for i in range(pca.n_components)],
            index=df.index
        )
        # get coefficients of criteria
        coeffs = pd.DataFrame(data=pca.components_, columns=columns)
        PCA_biplot(pcs, coeffs, var=pca.explained_variance_ratio_, **kwargs)

    def cluster(self, method, **kwargs):
        """Cluster compounds based on features in features_clustering."""
        df = self.get_ft_clustering()

        from sklearn.preprocessing import StandardScaler
        from sklearn import metrics
        if method == 'DBSCAN':
            from sklearn.cluster import DBSCAN

            X = StandardScaler().fit_transform(df)

            db = DBSCAN(**kwargs).fit(X)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)

        if method == 'KMeans':
            from sklearn.cluster import KMeans
            X = StandardScaler().fit_transform(df)
            silhouette_score_best: float = -10
            for i in range(2, 6):
                kmeans = KMeans(n_clusters=i).fit(X)
                labels = kmeans.labels_
                silhouette_score = metrics.silhouette_score(X, labels)
                if silhouette_score > silhouette_score_best:
                    silhouette_score_best = silhouette_score
                    kmeans_best = kmeans
            labels = kmeans_best.labels_
            self.centers = kmeans_best.cluster_centers_

        print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

        add_eigenvectors = len(self.features_clustering) < 10
        self.biplot(df, categories=labels, add_labels=False, add_eigenvectors=add_eigenvectors, **kwargs)

        cmpds_labeled = df.index.astype(str)
        self.df.loc['label', cmpds_labeled] = labels

    def performe_TOM_NMF_clustering(self, k, cluster_method='KMeans', **kwargs_cluster):
        features = self.features.copy()
        self.features_clustering = self.cmpds
        self.features = self.cmpds
        self.initiate_meta_feature_table()

        self.calculate_TOM_NMF_weights(k=k)
        self.df = self.TOM_NMF_weights

        self.cluster(cluster_method, **kwargs_cluster)
        labels = self.df.loc['label'].copy()
        self.features = features
        self.set_all_attrs()
        self.df.loc['label_nmf', :] = labels

    def get_group(self, label=None, labels_class='label'):
        if label is not None:
            agg = self.df.T.groupby(by=labels_class)
            agg = agg.get_group(label)
        else:
            agg = self.df.T.copy()
        return agg

    def get_compounds_in_cluster(self, label, **kwargs):
        return list(self.get_group(label, **kwargs).index)

    def test_significane_MWU(self, labels_class, category, label1, label2=None, alpha=.05, verbose=True):
        values_group1 = self.get_group(label1, labels_class).loc[:, category]
        if label2 is not None:
            values_group2 = self.get_group(label2, labels_class).loc[:, category]
        else:
            mask_not_group1 = self.df.loc[labels_class] != label1
            values_group2 = self.df.loc[category, mask_not_group1]
        res = stats.mannwhitneyu(values_group1, values_group2, nan_policy='omit')
        if (res.pvalue < alpha) and verbose:
            print(f'values in group 1 with label {label1} are significantly different from values in group 2 with labels {label2} (pval={res.pvalue:.3e} < {alpha}=alpha)')
        elif verbose:
            print(f'values in group 1 with label {label1} are NOT significantly different from values in group 2 with labels {label2} (pval={res.pvalue:.3e} >= {alpha}=alpha)')
        return res

    def test_significane_T(self, labels_class, category, label1, label2=None, alpha=.05, verbose=True):
        values_group1 = self.get_group(label1, labels_class).loc[:, category]
        if label2 is not None:
            values_group2 = self.get_group(label2, labels_class).loc[:, category]
        else:
            mask_not_group1 = self.df.loc[labels_class] != label1
            values_group2 = self.df.loc[category, mask_not_group1]
        res = stats.ttest_ind(values_group1, values_group2, nan_policy='omit')
        if (res.pvalue < alpha) and verbose:
            print(f'values in group 1 with label {label1} are significantly different from values in group 2 with labels {label2} (pval={res.pvalue:.3e} < {alpha}=alpha)')
        elif verbose:
            print(f'values in group 1 with label {label1} are NOT significantly different from values in group 2 with labels {label2} (pval={res.pvalue:.3e} >= {alpha}=alpha)')
        return res

    def table_pvals(self, labels_class, alpha_normal=0.05):
        """Return table with pvals for each category and cluster."""
        labels = np.unique(self.df.loc[labels_class, :])
        criteria = self.df.index.tolist()
        criteria.remove('label')
        criteria.remove('label_nmf')

        # 0 --> no, 1 --> yes, None --> test
        dict_is_normal = [0]

        pvals_table = pd.DataFrame(data=np.empty((len(criteria), len(labels)), dtype=float), columns=labels, index=criteria)
        for criterium in criteria:
            # check whether values are normal distributed
            res = stats.normaltest(self.df.loc['av', :], nan_policy='omit')
            if res.pvalue > alpha_normal:  # likely normal distributed
                for label in labels:
                    pvals_table.loc[criterium, label] = self.test_significane_against(labels_class, criterium, label, verbose=False).pvalue
            else:
                for label in labels:
                    pvals_table.loc[criterium, label] = self.test_significane_against(labels_class, criterium, label, verbose=False).pvalue
        return pvals_table

    def get_summary(self, label=None, what=['means', 'std'], **kwargs):
        agg = self.get_group(label, **kwargs)
        # members = agg.index
        # print(f'{len(members)} compounds ({len(members) / len(self.cmpds):.1%}) in cluster {label}.')

        out = {}
        if 'means' in what:
            means = agg.mean()
            out['mean'] = means
            print('means:', means)
        if 'std' in what:
            stds = agg.std()
            out['std'] = stds
            print('std:', stds)

        # medians = agg.median(axis=1)
        # mins = agg.min(axis=1)
        # maxs = agg.max(axis=1)
        # print('medians:', medians)
        # print('mins:', mins)
        # print('maxs:', maxs)
        return out


if __name__ == '__main__':
    windows = ['FA', 'Alkenones']
    self = MetaFeatures(windows=['FA'])
    self.set_all_masses()

    self.performe_TOM_NMF_clustering(k=5)
    out = self.get_summary()

    # look for homologs in clusters
    # for l in set(self.df.loc['label']):
    hs = Homologous(self.cmpds)
    hs.find_series()
    print(hs.series)

    # self.set_all_attrs()
    # self.features_clustering = ['seasonality', 'density_nonzero']
    # self.cluster('KMeans')

    # out = self.get_summary()

    # features = self.features.copy()
    # self.features_clustering = self.cmpds
    # self.features = self.cmpds
    # self.initiate_meta_feature_table()

    # self.calculate_TOM_NMF_weights(k=5)
    # self.df = self.TOM_NMF_weights

    # self.cluster('KMeans')

    # self.features = features
    # self.set_all_attrs()
    # self.summary_for_clusters()
