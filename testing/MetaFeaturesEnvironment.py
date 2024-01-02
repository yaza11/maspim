# -*- coding: utf-8 -*-
from cMetaFeatures import MetaFeatures
from cMetaFeatures import *

import os
import pykrev as pk
import matplotlib.pyplot as plt
import matplotlib


self = MetaFeatures(windows=['FA'])
self.load()

labels_class_to_type = {
    'label': 'seasonality/success rate',
    'label_nmf': 'co-localization'
}

# self.biplot(self.df.loc[['av', 'contrast_med'], :].T, categories=self.df.loc['label', :].tolist(), add_labels=False)

# %% mean and std tables for s and NT clusters
labels_nmf = self.df.loc['label_nmf']
labels_sd = self.df.loc['label']

lu_nmf = np.unique(labels_nmf).astype(int)
lu_sd = np.unique(labels_sd).astype(int)

means = {}
stds = {}
for l in lu_sd:
    o = self.get_summary(label=l, labels_class='label_nmf')
    means[l] = o['mean']
    stds[l] = o['std']


def format_v(m, s):
    return f'${m:.3f} \pm {s:.3f}$'


means_df = pd.DataFrame(means)
stds_df = pd.DataFrame(stds)


res = pd.DataFrame(data=np.empty_like(means_df, dtype=object))
res.index = means_df.index
for i in range(means_df.shape[0]):
    for j in range(means_df.shape[1]):
        m = means_df.iloc[i, j]
        s = stds_df.iloc[i, j]
        res.iloc[i, j] = format_v(m, s)


print(res.to_latex(escape=False))

# %%
if 0:
    labels_nmf = self.df.loc['label_nmf']
    labels_sd = self.df.loc['label']

    # unique labels
    lu_nmf = np.unique(labels_nmf)
    lu_sd = np.unique(labels_sd)

    # for each nmf cluster, distribution of sd clusters
    dist = pd.DataFrame(data=np.zeros((len(lu_sd), len(lu_nmf)), dtype=int),
                        columns=lu_nmf,
                        index=lu_sd)
    for label_nmf in lu_nmf:
        mz_nmf = set(labels_nmf.loc[labels_nmf == label_nmf].index)
        for label_sd in lu_sd:
            mz_sd = set(labels_sd.loc[labels_sd == label_sd].index)
            dist.at[label_sd, label_nmf] = len(mz_nmf & mz_sd)

# %%
if 0:
    window = 'FA'
    labels_class = 'label_nmf'

    for label in np.unique(self.df.loc['label']):
        mzs = self.get_compounds_in_cluster(label=label, labels_class=labels_class)
        mask_valid = ~self.df.loc['DBE'].isna()
        mask_valid &= self.df.columns.isin(mzs)

        mstup = self.get_pk_tup(mask_valid)

        plt.figure()
        pk.compound_class_plot(mstup, color='g', method='MSCC')
        plt.title(f'Compound classes in cluster {int(label)} of {labels_class_to_type[labels_class]} ')
        plt.show()

# %%
# self.overview_plts(labels_class=labels_class)

if 0:
    KM, KMD = pk.kendrick_mass_defect((None, None, self.df.columns.to_numpy().astype(float)), base='CH2')
    df = pd.DataFrame({'KM': KM, 'KMD': KMD})
    # line in cluster 1
    idx1 = 1309
    idx2 = 1537
    p1 = df.loc[idx1, :].tolist()
    p2 = df.loc[idx2, :].tolist()
    slope1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # line in cluster 2
    # (388.84753, .1524779)
    # (402.94146, .058533)
    idx3 = 43
    idx4 = 413
    p3 = df.loc[idx3, :].tolist()
    p4 = df.loc[idx4, :].tolist()
    slope2 = (p4[1] - p3[1]) / (p4[0] - p3[0])


def f_to_tup(formula):
    return ([formula], None, None)


def mz_to_tup(mz):
    return (None, None, np.array([mz]))


def KMD_by_swapping(f_a, f_b):
    """Calc mass defect added by replacing a with b."""
    m_a = pk.calculate_mass(f_to_tup(f_a))
    m_b = pk.calculate_mass(f_to_tup(f_b))

    KM_a, KMD_a = pk.kendrick_mass_defect(mz_to_tup(m_a))
    KM_b, KMD_b = pk.kendrick_mass_defect(mz_to_tup(m_b))
    return (KM_a - KM_b) % 1


def KMD(f_a):
    m_a = pk.calculate_mass(f_to_tup(f_a))
    KM_a, KMD_a = pk.kendrick_mass_defect(mz_to_tup(m_a))
    return KMD_a


if 0:
    reactions = [
        'H', 'H2', 'O', 'H2O', 'OH', 'CO', 'N2', 'NO2',
        'CH2', 'CH3', 'C2H2', 'C2H3'
    ]

    df_KMDs = pd.DataFrame(data=np.zeros((len(reactions), len(reactions)), dtype=float), columns=reactions, index=reactions)
    for ra in reactions:
        for rb in reactions:
            kmd_ = KMD_by_swapping(ra, rb) % 1
            if kmd_ > .5:
                kmd_ -= 1
            df_KMDs.loc[ra, rb] = kmd_

    m_CH2 = pk.calculate_mass(f_to_tup('CH2'))
    m_N = pk.calculate_mass(f_to_tup('N'))

    KM_CH2, KMD_CH2 = pk.kendrick_mass_defect(mz_to_tup(m_CH2))
    KM_N, KMD_N = pk.kendrick_mass_defect(mz_to_tup(m_N))

    # color based on hs
    hs = Homologous(self.cmpds)
    hs.find_series()

# %%
# color_reaction = 'CH2'
# self.df.loc['hs'] = 0
# cmpds = set(self.cmpds)
# for idx, (k, v) in enumerate(hs.series.items()):
#     reaction = k[0]
#     if reaction != color_reaction:
#         continue
#     if idx == 100:
#         for c in v:
#             if (c := str(c)) in cmpds:
#                 self.df.loc['hs', c] += idx
#     # for c in v:
#     #     if (c := str(c)) in cmpds:
#     #         self.df.loc['hs', c] += idx
# %%
if 0:
    plt.figure()
    plt.scatter(KM, KMD, s=7, c=self.df.loc[labels_class], alpha=.7)
    plt.plot(df.iloc[[idx1, idx2], 0], df.iloc[[idx1, idx2], 1])
    plt.plot(df.iloc[[idx3, idx4], 0], df.iloc[[idx3, idx4], 1])
    plt.xlabel('Kendrick Mass')
    plt.ylabel('Kendrick Mass defect')
    plt.colorbar().set_label('class label')
    plt.title('Kendrick Mass vs Kendrick Mass Defects')
    plt.show()

# %% elemental ratios

if 0:
    C = self.df.loc['C', :]
    H = self.df.loc['H', :]
    O = self.df.loc['O', :]
    N = self.df.loc['N', :]
    HC = H / C
    OC = O / C
    NC = N / C

    fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col', layout='constrained')

    cmap = matplotlib.colormaps['viridis']

    d = {'HC': HC, 'OC': OC, 'NC': NC}
    dl = {'HC': 'H/C', 'OC': 'O/C', 'NC': 'N/C'}

    for k, labels_class in enumerate(('label', 'label_nmf')):
        labels = np.unique(self.df.loc[labels_class])
        j_max = len(labels)
        for i, plt_type in enumerate(('HC/OC', 'HC/NC', 'OC/NC')):
            xr, yr = plt_type.split('/')
            xs = d[xr]
            ys = d[yr]
            for j, label in enumerate(labels):
                mask_c = self.df.loc[labels_class] == label
                x = xs[mask_c]
                y = ys[mask_c]
                if (i == 0) and (k == 0):
                    label = int(label)
                else:
                    label = None
                axs[k, i].scatter(x, y,
                                  c=cmap(j / (j_max - 1)),
                                  label=label,
                                  alpha=.5,
                                  )
            axs[k, i].set_ylabel(dl[yr])
            if k == 1:
                axs[k, i].set_xlabel(dl[xr])
    fig.legend()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# markers = ['o', 'v', 's', 'D', '^', 'p', '*', '1', '8']
# cmap = matplotlib.colormaps['viridis']
# labels = np.unique(self.df.loc[labels_class])
# i_max = len(labels)
# padding = .5

# xs = H / C
# ys = O / C
# zs = N / C

# for i, label in enumerate(labels):
#     mask_c = self.df.loc[labels_class] == label
#     x = xs[mask_c]
#     y = ys[mask_c]
#     z = zs[mask_c]
#     ax.scatter(x, y, z,
#                 # marker=markers[int(label) % len(markers)],
#                 c=cmap(i / (i_max - 1)),
#                 label=int(label),
#                 alpha=.5
#     )
#     ax.scatter(y, z, xs.min() - padding + i / i_max * padding / 2, zdir='x', c=cmap(i / (i_max - 1)), alpha=1)
#     ax.scatter(x, z, ys.max() + padding - i / i_max * padding / 2, zdir='y', c=cmap(i / (i_max - 1)), alpha=1)
#     ax.scatter(x, y, zs.min() - padding + i / i_max * padding / 2, zdir='z', c=cmap(i / (i_max - 1)), alpha=1)
# ax.set_xlabel('H/C')
# ax.set_ylabel('O/C')
# ax.set_zlabel('N/C')

# ax.set_zlim(zs.min() - padding, zs.max() + padding)
# ax.set_ylim(ys.min() - padding, ys.max()+ padding)
# ax.set_xlim(xs.min() - padding, xs.max() + padding)
# plt.legend()
# plt.show()

# mask_valid = ~self.df.loc['DBE'].isna()
# self.features_clustering = ['HC', 'NC', 'OC']
# self.biplot(categories=self.df.loc[labels_class, mask_valid], add_labels=False)


# %% boxplot double box equavalents
y_crit = 'DBE'
x = self.df.loc['seasonality']
y = self.df.loc[y_crit]
c = self.df.loc['label_nmf']

# viridis_cmap = plt.get_cmap('viridis')
# norm = plt.Normalize(min(c), max(c))
# plt.scatter(x, y, c=c.tolist(), cmap=viridis_cmap, norm=norm)
# plt.show()

xs = [x[c == ci].to_numpy() for ci in set(c)]
ys = [y[(c == ci) & (~y.isna())].to_numpy() for ci in set(c)]
plt.boxplot(ys)
plt.xlabel('label NT-cluster')
plt.ylabel(y_crit)
plt.show()