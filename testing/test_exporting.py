import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from res.constants import mC37_2, mC37_3
from data.cMSI import MSI
from exporting.from_mcf.cSpectrum import Spectra, ClusteringManager
from exporting.from_mcf.rtms_communicator import ReadBrukerMCF
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from typing import Iterable

# d_folder = "D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 FA.i/2018_08_27 Cariaco 490-495 FA.d"
# d_folder = "D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 GDGT.i/2018_08_27 Cariaco 490-495 GDGT.d"
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TGA3_5-10_Test2_5/SBB_TGA3_5-10_Test2_5.d"  # geht nicht
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_4/13012023_SBB_TG3A_05-10_Test2_4.d"  # geht
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_3/13012023_SBB_TG3A_05-10_Test2_3.d"  # geht nicht
# d_folder ="//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_2/13012023_SBB_TG3A_05-10_Test2_2.d"  # geht
# d_folder = '//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_1/13012023_SBB_TG3A_05-10_Test2.d'  # geht
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2/13012023_SBB_TG3A_05-10_Test2.d"  # geht
d_folder = "D:/Promotion/Test data/13012023_SBB_TG3A_05-10_Test2.d"
# d_folder = 'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d'
con = ReadBrukerMCF(d_folder)
con.create_reader()
con.create_indices()
con.set_meta_data()
con.set_QTOF_window()

# mm: ClusteringManager = ClusteringManager(con)
# mm.indices = list(range(1, 2000))
# mm.set_clusters(method='random', N_chunks=4)
# mm.plt_cluster_distribution()
# spec_c = mm.get_spectra(con)
spec = Spectra(path_d_folder=d_folder, initiate=False)
# spec = Spectra(path_d_folder=d_folder, reader=con)
# con.mzs = spec.mzs
# spec.full(con)
spec.load()

# spec.detect_side_peaks(max_distance=.01, max_relative_height=.3, plts=True)

# spec.filter_peaks(SNR_threshold=1, remove_sidepeaks=False, plts=True)

# spectra1 = Spectra(reader=con, indices=con.indices[:len(con.indices)//2])
# spectra2 = Spectra(reader=con, indices=con.indices[len(con.indices)//2:])

# mz_lim: tuple[float] = (552.0, 552.5)
# spectra1.add_all_spectra(con)
# spectra1.subtract_baseline(window_size=.05, plts=False)
# spectra1.set_peaks(prominence=.01)
# spectra1.set_kernels(use_bigaussian=False, fine_tune=True)
# # spectra1.plt_summed(mz_limits=mz_lim)
# spectra1.bin_spectra(con, integrate_peaks=True)
# spectra1.binned_spectra_to_df(con)

# spectra2.add_all_spectra(con)
# spectra2.subtract_baseline(window_size=.05, plts=False)
# spectra2.set_peaks(prominence=.01)
# spectra2.set_kernels(use_bigaussian=False, fine_tune=True)
# # spectra2.plt_summed(mz_limits=mz_lim)
# spectra2.bin_spectra(con, integrate_peaks=True)
# spectra2.binned_spectra_to_df(con)

# spectra_c = spectra1.combine_with(spectra2)
# spectra_c.plt_summed(mz_limits=mz_lim)

# spectra = Spectra(reader=con)
# spectra.add_all_spectra(con)
# spectra.subtract_baseline(window_size=.05, plts=False)
# spectra.set_peaks(prominence=.01)
# spectra.set_kernels(use_bigaussian=False, fine_tune=True)
# spectra.plt_summed(mz_limits=mz_lim)
# spectra.bin_spectra(con, integrate_peaks=True)
# spectra.binned_spectra_to_df(con)

# s = Spectra(path_d_folder=d_folder, initiate=False)
# s.load()

# targets = [mC37_2, mC37_3]

# s.set_targets(targets, tolerances = None, method='area_overlap')
# # s.set_kernels()
# s.bin_spectra(con)
# s.plt_summed()


def test_against_export():
    d_folder = 'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d'
    spectra = Spectra(load=True, path_d_folder=d_folder)

    df = pd.read_csv(
        r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d/exported_summed_peaks.txt',
        sep='\t')

    plt.figure()
    plt.plot(
        spectra.mzs, spectra.intensities / len(spectra.indices), label='average'
    )
    plt.stem(
        spectra.kernel_params[:, 0], spectra.kernel_params[:, 1] / len(spectra.indices),
        markerfmt='',
        linefmt='b-',
        label='spectra'
    )
    mask = (df['m/z'] >= spectra.mzs.min()) & (df['m/z'] <= spectra.mzs.max())
    plt.stem(
        df.loc[mask, 'm/z'] + .001,
        df.loc[mask, 'I'],
        markerfmt='',
        linefmt='r-',
        label='export'
    )
    plt.legend()
    # plt.xlim((558.2, 558.37))
    plt.xlabel('m/z in Da')
    plt.ylabel('av Intensity')
    plt.title('exported shifted by +1 mDa for visibility')
    plt.show()

# test_against_export()
# spectra.indices=list(range(1, 101))


# spectra.save()

# msi = MSI(d_folder)
# msi.set_feature_table_from_spectra(spectra)
# msi.set_distance_pixels()

# offsets = np.array([spectra.xcorr(con.get_spectrum(idx), plts=True, max_mass_offset=3e-3)
#            for idx in spectra.indices])

def test_reconstruction_spec(idx, mz_lim=(556.1, 556.4)):
    kernels = spectra._get_kernels(norm_mode='height')

    idx_spec = spectra.indices[idx]
    spec = con.get_spectrum(idx_spec)
    spec.resample(spectra.mzs)
    areas = spectra.line_spectra[idx, :]

    if spectra.binning_by == 'area':
        Hs = spectra.H_from_area(areas, spectra.kernel_params[:, 2])
    else:
        Hs = spectra.line_spectra[idx, :]

    plt.figure()
    plt.plot(spec.mzs, spec.intensities, 'red', label='original')
    plt.plot(spec.mzs, (kernels * Hs).sum(axis=1), 'C0--', label='reconstruced')
    plt.stem(spectra.kernel_params[:, 0], Hs, markerfmt='', linefmt='C0-', label='H rec')
    if hasattr(spectra, 'noise_level'):
        plt.plot(spectra.mzs, spectra.noise_level, 'C2', label='noise level')
    plt.xlim(mz_lim)
    mask = (spec.mzs >= mz_lim[0]) & (spec.mzs <= mz_lim[1])
    plt.ylim((0, spec.intensities[mask].max()))
    plt.xlabel('m/z in Da')
    plt.ylabel('Intensity')
    plt.legend()


# test_reconstruction_spec(4)

def plt_C37s(spectra, is2=True):
    # find peaks
    x = spectra.mzs
    y = spectra.intensities
    median = np.median(y)
    peaks, properties = find_peaks(y, prominence=.1 * median, width=3)
    # draw peaks

    plt.plot(x, y, color='C0')
    plt.vlines(x[peaks], ymin=y[peaks] - properties['prominences'], ymax=y[peaks], colors='C1')
    plt.hlines(
        y=properties["width_heights"],
        xmin=x[(properties["left_ips"] + .5).astype(int)],
        xmax=x[(properties["right_ips"] + .5).astype(int)],
        color="C1"
    )
    # vlines for C37's (Na+ adduct)
    mz_C37_2 = 553.53188756536
    mz_C37_3 = 551.51623750122
    plt.vlines([mz_C37_3, mz_C37_2], ymin=0, ymax=y.max(), colors='red')
    if is2:
        xlim = (mz_C37_2 - .1, mz_C37_2 + .1)
    else:
        xlim = (mz_C37_3 - .1, mz_C37_3 + .1)
    mask = (x >= xlim[0]) & (x <= xlim[1])
    plt.xlim(xlim)
    plt.ylim((0, y[mask].max()))


def ion_imgs_UK(spectra):
    mz_C37_2 = 553.53188756536
    mz_C37_3 = 551.51623750122

    # find closest masses
    mzs = spectra.feature_table.drop(columns=['R', 'x', 'y']).columns.astype(float)
    mz2_idx = np.argmin(np.abs(mzs - mz_C37_2))

    plt.imshow(spectra.feature_table.pivot(
        index='x', columns='y', values=spectra.feature_table.columns[mz2_idx])
    )
    plt.show()

    mz3_idx = np.argmin(np.abs(mzs - mz_C37_3))

    plt.imshow(spectra.feature_table.pivot(
        index='x', columns='y', values=spectra.feature_table.columns[mz3_idx])
    )
    plt.show()
