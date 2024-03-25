# -*- coding: utf-8 -*-
from data.cMSI import MSI
from clustering.network import set_node_attributes
from res.directory_paths import folder_thesis_images

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

low_res = 100


def create_network_from_r_table(
        corr: pd.DataFrame = None,
        df: pd.DataFrame = None,
        title: str = '',
        hold=False,
        **kwargs
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
    if ("thr" in kwargs) or nx.has_path(G, 'L', '-L'):
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

    fig, ax = plt.subplots()
    plt.title(title + f', {thr=:.2e}')

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        with_labels=False,
        # labels=attrs.labels,
        # node_color=attrs.colors,
        node_size=attrs.node_sizes,
        edge_color='black',
        # width=rescale_values(attrs.edge_widths, 1, 5),
        font_size=8,
        font_color='white',
        alpha=.7

    )
    # ax.set_facecolor('cyan')
    fig.set_facecolor('peru')

    if not hold:
        plt.show()

    return G, attrs


def get_weights_NMF_TOM(DC, beta=12):
    DC.analyzing_NMF(k=5)
    H = DC.H

    H_df = pd.DataFrame(data=H, columns=DC.get_data_columns())
    r = H_df.corr()
    plt.imshow(r)
    diss = (r + 1) / 2
    # calculate TOM
    A = diss ** beta
    L = np.dot(A, A)
    k = np.sum(A, axis=0)
    d = len(k)
    tile = np.tile(k, (d, 1))
    K = np.min(np.stack((tile, tile.T), axis=2), axis=2)
    weights = (L + A) / (K + 1 - A)
    np.fill_diagonal(weights.to_numpy(), 1)

    weights_df = pd.DataFrame(data=weights, columns=H_df.columns, index=H_df.columns)
    return weights_df


if __name__ == '__main__':
    section = (490, 510)
    window = 'FA'

    title = 'Network based on TOM of NMF'
    thr = .4

    DC = MSI(section, window)
    DC.load()
    weights_df = get_weights_NMF_TOM(DC)
    G, attrs = create_network_from_r_table(weights_df, DC.get_data(), thr=thr, title=title)
    plt.savefig(os.path.join(
        folder_thesis_images,
        rf'{DC._window}_{DC.get_section_formatted()[1]}_NMF_TOM_k=5'),
        dpi=low_res
    )

    N = len(G.nodes)
    p = -1 / 8
    pos = nx.spring_layout(G, seed=1337, k=N ** p)  # default k: -1/2

    # %%
    fig, ax = plt.subplots()
    plt.title(title + f', {thr=:.2e}')

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        with_labels=False,
        # labels=attrs.labels,
        # node_color=attrs.colors,
        node_size=attrs.node_sizes / 10,
        edge_color='black',
        # width=rescale_values(attrs.edge_widths, 1, 5),
        font_size=8,
        font_color='white',
        alpha=.7

    )

    plt.savefig(
        os.path.join(
            folder_thesis_images,
            rf'{DC._window}_{DC.get_section_formatted()[1]}_NMF_TOM_k=5_network'
        ),
        dpi=low_res
    )
    plt.show()

    # fig.set_facecolor('peru')

    # nx.write_graphml(G, rf"D:\My Drive\Master Thesis\networks\NMF_TOM\{DC._window}_{DC.get_section_formatted()[1]}_NMF_TOM_k=5.graphml")
