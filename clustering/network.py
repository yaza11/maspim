# -*- coding: utf-8 -*-
from imaging.util.coordinate_transformations import rescale_values
from res.constants import sections_all, max_deviation_mz

from data.annotating import AllAnotations
from timeSeries.cTimeSeries import TimeSeries
from data.cDataClass import Data

from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
try:
    from PyWGCNA import WGCNA as wna
except ModuleNotFoundError:
    def wna(*args, **kwargs):
        print('PyWGCNA not installed!')
import anndata as ad
import matplotlib.pyplot as plt

# all_annotations = AllAnotations(mz_precision=max_deviation_mz)
all_annotations = None


def mass_to_name(mass):
    if str(mass).replace('.', '').isnumeric():
        name = all_annotations.get_for(mass).name()
        return name
    return str(mass)


def mass_to_color(mass):
    if 'L' in mass:
        return 'gray'
    elif all_annotations.get_for(mass).has_child():
        return 'orange'
    return 'blue'


@dataclass
class GraphAttributes:
    labels: dict[str, str]
    colors: np.ndarray[str]
    intensities: np.ndarray[float]
    log_intensities: np.ndarray[float]
    node_sizes: np.ndarray[float]
    edge_widths: np.ndarray[float]


def set_node_attributes(G, corr, df=None):
    if df is not None:
        abs_means = df.abs().mean(axis=0)

    N: int = len(G.nodes)
    labels: np.ndarray[str] = np.empty(N, dtype=object)
    colors: np.ndarray[str] = np.empty(N, dtype=object)
    intensities: np.ndarray[float] = np.empty(N, dtype=float)
    log_intensities: np.ndarray[float] = np.empty(N, dtype=float)
    # set attributes by iterating over nodes
    nx.set_node_attributes(G, [], name='label')
    nx.set_node_attributes(G, [], name='color')
    nx.set_node_attributes(G, [], name='intensity')
    for idx, (node, data) in enumerate(G.nodes(data=True)):
        label = mass_to_name(node)
        data['label'] = label
        labels[idx] = label
        color = mass_to_color(node)
        data['color'] = color
        colors[idx] = color
        if df is not None:
            abs_mean = abs_means.loc[node]
            intensities[idx] = abs_mean
            log_intensity = np.log10(abs_mean) + 3
        else:
            log_intensity = 1
        data['intensity'] = log_intensity
        log_intensities[idx] = log_intensity

    if df is not None:
        node_sizes = rescale_values(log_intensities, 100, 400)
    else:
        node_sizes = 400
        intensities = 1
        log_intensities = 1

    # iterate over edges
    N_edges: int = len(G.edges)
    edge_widths: np.ndarray[float] = np.empty(N_edges, dtype=float)
    nx.set_edge_attributes(G, [], name='r')
    for idx, (node_a, node_b, data) in enumerate(G.edges(data=True)):
        ew = corr.loc[node_a, node_b]
        data['r'] = ew
        edge_widths[idx] = ew

    attrs = GraphAttributes(
        labels=dict(zip(G.nodes, labels)),
        colors=colors,
        intensities=intensities,
        log_intensities=log_intensities,
        node_sizes=node_sizes,
        edge_widths=edge_widths
    )

    return G, attrs


def create_network_from_r_table(
        corr: pd.DataFrame = None,
        df: pd.DataFrame = None,
        title: str = '',
        plts=True,
        ** kwargs
) -> nx.graph.Graph:
    assert (corr is not None) or (df is not None), 'pass either a corr or feature table'

    if corr is None:
        corr = df.corr()

    if 'thr' not in kwargs:
        # find lowest threshold for which L and -L are in
        thr = np.min([corr['L'].sort_values()[-2], corr['-L'].sort_values()[-2]])
    else:
        thr = kwargs['thr']

    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2', 'r']
    links = links.sort_values(by='r', ascending=False).reset_index(drop=True)
    links = links.loc[links['var1'] != links['var2']]

    def graph_from_links(links, thr):
        tol = 1e-12
        links_filtered = links.loc[links['r'] >= thr - tol]

        G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')

        return G

    G = graph_from_links(links, thr)
    if ('thr' in kwargs) or nx.has_path(G, 'L', '-L'):
        pass
    else:
        fac_thr = .99
        while not nx.has_path(G, 'L', '-L'):
            # increment thr
            thr *= fac_thr
            G = graph_from_links(links, thr)
        thr /= fac_thr

    G = graph_from_links(links, thr)

    G, attrs = set_node_attributes(G, corr, df)

    N = len(G.nodes)
    p = -1 / 5
    pos = nx.spring_layout(G, seed=1337, k=N ** p)  # default k: -1/2

    if plts:
        fig, ax = plt.subplots()
        plt.title(title + f', {thr=:.2e}')

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=attrs.labels,
            node_color=attrs.colors,
            node_size=attrs.node_sizes,
            edge_color='black',
            width=rescale_values(attrs.edge_widths, 1, 5),
            font_size=8,
            font_color='white',
            alpha=.7

        )
        # ax.set_facecolor('cyan')
        fig.set_facecolor('peru')

        plt.show()

    return G, attrs


def get_TS(section, window):
    TS = TimeSeries(section, window)
    if window == 'combined':
        TS.load()
    elif section not in sections_all:
        from timeSeries.cTimeSeries import combine_sections
        l = section[0]
        u = section[1]
        assert (l % 5 == 0) and (u % 5 == 0)
        sections = [(l + 5 * i, l + 5 * (i + 1)) for i in range((u - l) // 5)]
        TS = combine_sections(sections, window)
    else:
        TS.set_time_series_tables(use_common_mzs=True)
    TS.verbose = True
    TS.feature_table_zone_averages, TS.feature_table_zone_successes = TS.combine_duplicate_seed()
    return TS


def bicor_df(df: pd.DataFrame):
    # first, calculate col-wise xtild
    # then calc scalar product of all possible pairs
    medx = df.median(axis=0)
    MADs = (df - medx).abs().median(axis=0)
    x = df - medx
    u = x / (9 * MADs)

    w = (1 - u ** 2) ** 2
    I = (1 - u.abs()) > 0
    w *= I
    k = np.sqrt(((x * w) ** 2).sum(axis=0))
    xt = x * w / k
    xt = xt.fillna(0)
    corr = xt.T @ xt
    return corr


def get_ft(TS):
    ftc = TS.get_contrasts_table()
    # ftc = TS.feature_table_zone_averages

    # reset the quality criteria
    ftc['contrast'] = TS.feature_table_zone_averages.contrast
    ftc['homogeneity'] = TS.feature_table_zone_averages.homogeneity
    ftc['quality'] = TS.feature_table_zone_averages.quality

    # this is not the vector that creates the -L corr but is necessary for
    # mean (to calc node size)
    ftc['-L'] = TS.feature_table_zone_averages.L
    return ftc


def get_corr_table(TS, corr_method, include_L=True):
    cols = list(set(TS.get_data_columns()) & set(TS.feature_table_zone_averages.columns))
    if include_L:
        cols.append('L')
    ftc = get_ft(TS)

    # calculate correlations
    if corr_method == 'wsc':
        corr = TS.get_sign_weighted_table(ftc, use_L_contrasts=True).loc[
            cols, cols
        ]
    elif corr_method == 'sc':
        corr = TS.get_sign_table(ftc).loc[cols, cols]
    elif corr_method == 'bicor':
        # exclude zeros from correlation
        mask_zeros = ftc == 0
        ftc[mask_zeros] = np.nan
        corr = bicor_df(ftc.loc[:, cols])

    else:
        # exclude zeros from correlation
        mask_zeros = ftc == 0
        ftc[mask_zeros] = np.nan

        corr = ftc.loc[:, cols].corr(
            min_periods=10,
            method=corr_method
        )

    if include_L:
        # add -L
        corr['-L'] = -corr.L
        corr.loc['-L'] = corr['-L'].tolist() + [1.]
    else:
        corr = corr.loc[cols, cols]
    # rescale corr values to be in [0, 1]
    corr = (corr + 1) / 2
    corr[corr > 1] = 1
    corr[corr < 0] = 0
    return corr


def network_for_timeSeries(section, window, corr_method='pearson', **kwargs):
    TS = get_TS(section, window)
    ftc = get_ft(TS)
    corr = get_corr_table(TS, corr_method)
    # raise to power of beta
    if 'beta' in kwargs:
        corr = corr ** kwargs['beta']

    title = f'network for {TS._section} {TS._window} with {corr_method=} and \n {kwargs}'

    G, attrs = create_network_from_r_table(corr=corr, df=ftc, title=title, **kwargs)

    return G, corr, attrs, TS


def get_anndata(geneExp: pd.DataFrame) -> ad.AnnData:
    """Create anndata obj from dataframe."""
    expressionList = geneExp
    geneInfo = pd.DataFrame(index=expressionList.columns)
    sampleInfo = pd.DataFrame(index=expressionList.index)
    anndata = ad.AnnData(X=expressionList, obs=sampleInfo, var=geneInfo)
    return anndata


def get_WGCNA(TS, corr_method='bicor', **kwargs):
    fts = get_ft(TS)
    # corr = get_corr_table(TS, corr_method, include_L=False)
    fts = fts.fillna(0)
    geneExp: pd.DataFrame = fts.copy()
    cols = list(set(TS.get_data_columns()) & set(TS.feature_table_zone_averages.columns))
    geneExp = geneExp.loc[:, cols]

    anndata = get_anndata(geneExp)

    w = wna(anndata=anndata, **kwargs)

    return w


def perform_WGCNA(w: wna, **kwargs):
    # w.preprocess()

    w.findModules(**kwargs)
    return w


def make_plts(w):
    # # create network
    modules = w.datExpr.var.moduleColors.unique().tolist()
    w.CalculateSignedKME()
    w.CoexpressionModulePlot(
        modules=modules,
        numGenes=100,
        numConnections=1000,
        minTOM=0,
        file_name="all"
    )


class TS_WGCNA:
    def __init__(self, TS: TimeSeries):
        self.TS = TS

    def calc_WGCNA(self, **wgcna_kwargs):
        self.w = get_WGCNA(self.TS, **wgcna_kwargs)
        self.w = perform_WGCNA(self.w)

        self.eigen_compounds = self.w.MEs.reset_index(drop=True)
        # labels of each compound
        clusts = self.w.datExpr.var['moduleColors']
        # overlap
        self.TOM = self.w.TOM

        labels = clusts.to_numpy()
        names = clusts.index
        ulabels = np.unique(labels)
        cs = []
        for l in ulabels:
            cs.append(names[labels == l])
        cs = dict(zip(ulabels, cs))
        self.clusters = cs

    def summary(self):

        eigens = self.eigen_compounds.copy()
        eigens.index = self.TS.feature_table_zone_averages.index

        seas_eigens = self.TS.get_seasonalities(
            eigens, weighted=True, exclude_low_success=False, mult_N=False
        )

        for l in self.clusters:
            print(f'{l}, N: {len(self.clusters[l])}, seas eig: {seas_eigens["ME" + l]:.3f}')

    def compare_to_grayscale(self):
        # read marker "gene" list
        # marker gene table should have genes as the row index --> compounds as row index
        # a column called moduleColors
        pass


class NMF_WGCNA:
    def __init__(self, DC: Data):
        assert hasattr(DC, 'H')
        self.DC = DC

    def calc_WGCNA(self, **wgcna_kwargs):
        H_df = pd.dataFrame
        self.w = get_WGCNA(H_df, **wgcna_kwargs)
        self.w = perform_WGCNA(self.w)

        self.eigen_compounds = self.w.MEs.reset_index(drop=True)
        # labels of each compound
        clusts = self.w.datExpr.var['moduleColors']
        # overlap
        self.TOM = self.w.TOM

        labels = clusts.to_numpy()
        names = clusts.index
        ulabels = np.unique(labels)
        cs = []
        for l in ulabels:
            cs.append(names[labels == l])
        cs = dict(zip(ulabels, cs))
        self.clusters = cs

    def summary(self):

        eigens = self.eigen_compounds.copy()
        eigens.index = self.TS.feature_table_zone_averages.index

        seas_eigens = self.TS.get_seasonalities(
            eigens, weighted=True, exclude_low_success=False, mult_N=False
        )

        for l in self.clusters:
            print(f'{l}, N: {len(self.clusters[l])}, seas eig: {seas_eigens["ME" + l]:.3f}')

    def compare_to_grayscale(self):
        # read marker "gene" list
        # marker gene table should have genes as the row index --> compounds as row index
        # a column called moduleColors
        pass


if __name__ == '__main__':
    # section = sections_all[0]

    section = (490, 510)
    TS = TimeSeries(section, 'FA')
    TS.load()
    # TS = get_FA_TSa()
    # if False:
    #     succ_layers = (
    #         TS.feature_table_zone_successes[TS.get_data_columns()] >
    #         n_successes_required
    #     ).sum(axis=0)
    #     cutoff = np.max([succ_layers.median(), n_successes_required])
    #     # filter out compounds with success rate lower than the median
    #     mask_drop = succ_layers < succ_layers.quantile(.75)
    #     drop_cols = mask_drop.loc[mask_drop == True].index.tolist()
    # TS.feature_table_zone_averages = TS.feature_table_zone_averages.drop(columns=drop_cols)

    wgcna = TS_WGCNA(TS)
    # wgcna.calc_WGCNA(MEDissThres=0, powers=[12])
    wgcna.calc_WGCNA()

    # with open(r'E:\Master_Thesis\TS_WGCNA_objects\FA_before_mz388-481.pickle', 'wb') as f:
    #     pickle.dump(wgcna, f, pickle.HIGHEST_PROTOCOL)

    # with open(r'E:\Master_Thesis\TS_WGCNA_objects\FA_before_mz388-481.pickle', 'rb') as f:
    #     wgcna = pickle.load(f)

    # wgcna.summary()

    # w, corr = get_WGCNA(TS, minModuleSize=30)
    # w = perform_WGCNA(w, corr)

    # eigen_compounds = w.MEs.reset_index(drop=True)
    # # labels of each compound
    # clusts = w.datExpr.var['moduleColors']
    # # overlap
    # TOM = w.TOM

    # # calculate seasonality scores of eigengenes
    # eigens = wgcna.eigen_compounds.copy()
    # eigens.index = TS.feature_table_zone_averages.index

    # seas_eigens = TS.get_seasonalities(
    #     eigens, weighted=True, exclude_low_success=False, mult_N=False
    # )
    # print(seas_eigens)

    # labels = clusts.to_numpy()
    # names = clusts.index
    # ulabels = np.unique(labels)
    # cs = []
    # for l in ulabels:
    #     cs.append(names[labels == l])
    # cs = dict(zip(ulabels, cs))
    # for k, v in cs.items():
    #     print(v, seas_eigens[f'ME{k}'])

    # create overview plot of clusters with
    # seasonality of each eigencomp
    # corr of each member to eigencomp
    # seasonality of each comp

    # G, corr, attrs, TS = network_for_timeSeries(
    #     section,
    #     window,
    #     corr_method='pearson',
    # )
    # nx.write_graphml(G, rf"D:\My Drive\Master Thesis\networks\{window}_{section[0]}-{section[1]}.graphml")
    pass
