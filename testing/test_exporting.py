import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from exporting_mcf.rtms_communicator import ReadBrukerMCF, Spectra

# d_folder = "D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 FA.i/2018_08_27 Cariaco 490-495 FA.d"
# d_folder = "D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 GDGT.i/2018_08_27 Cariaco 490-495 GDGT.d"
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TGA3_5-10_Test2_5/SBB_TGA3_5-10_Test2_5.d"  # geht nicht
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_4/13012023_SBB_TG3A_05-10_Test2_4.d"  # geht
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_3/13012023_SBB_TG3A_05-10_Test2_3.d"  # geht nicht
# d_folder ="//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_2/13012023_SBB_TG3A_05-10_Test2_2.d"  # geht
# d_folder = '//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2_1/13012023_SBB_TG3A_05-10_Test2.d'  # geht
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TG3A_05-10_Test2/13012023_SBB_TG3A_05-10_Test2.d"  # geht
# d_folder = "D:/Promotion/Test data/13012023_SBB_TG3A_05-10_Test2.d"
# d_folder = 'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d'
# con = ReadBrukerMCF(d_folder)
# con.create_reader()
# con.create_indices()

# spectra = Spectra(reader=con)
spectra = Spectra(load=True, path_d_folder=d_folder)
# spectra.add_all_spectra(con)

# spectra.set_peaks()

# spectra.set_kernels()

# spectra.plt_kernels()

# spectra.bin_spectra(con)

# spectra.binned_spectra_to_df(con)

img = spectra.feature_table.pivot(
    index='x', 
    columns='y', 
    values = spectra.feature_table.columns[0]
)
plt.imshow(img)

# spectra.save()

# %% sqlite parser
# from exporting.parser import parse_sqlite

# from exporting.parser import parse_acqumethod

# acqumethod = parse_acqumethod(r'D:/Promotion/Test data/13012023_SBB_TG3A_05-10_Test2.d/13012022_1M_150-2000_Q1554+-20_L35-500-700_200um_fid.m/apexAcquisition.method')

# xy, mzs, intensities,snrs = parse_sqlite(r'D:/Promotion/Test data/13012023_SBB_TG3A_05-10_Test2.d/peaks.sqlite')
# # xy, mzs, intensities,snrs = parse_sqlite(r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 GDGT.i/2018_08_27 Cariaco 490-495 GDGT.d/peaks.sqlite')

# # %% compare
# plt.stem(
#     spectra.feature_table.drop(columns=['R', 'x', 'y']).columns.to_numpy().astype(float), 
#     spectra.feature_table.drop(columns=['R', 'x', 'y']).iloc[-1, :] / np.max(spectra.feature_table.drop(columns=['R', 'x', 'y']).iloc[-1, :]),
#     markerfmt='', linefmt='C1-'
# )
# plt.stem(mzs[-1], intensities[-1] / np.max(intensities[-1]), markerfmt='', linefmt='C0-')

# plt.xlim(spectra.limits)

# %%
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
        color = "C1"
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
    

# center bigaussian around peaks
# mz_C37_2 = 553.53188756536
# window = (mz_C37_2 - .1, mz_C37_2 + .1)

# x = spectra.mzs
# y = spectra.intensities
# peaks = spectra.peaks
# properties = spectra.properties

# x_idx = np.argmin(np.abs(x - mz_C37_2))
# peak_idx = np.argmin(np.abs(peaks - x_idx))
# # update x_idx
# x_idx = peaks[peak_idx]

# mask = (x >= window[0]) & (x <= window[1])
# plt.plot(x[mask], y[mask])
# plt.vlines(x[x_idx], ymin=y[x_idx] - properties['prominences'][peak_idx], ymax=y[x_idx], colors='C1')
# plt.hlines(
#     y=properties["width_heights"][peak_idx], 
#     xmin=x[(properties["left_ips"][peak_idx] + .5).astype(int)],
#     xmax=x[(properties["right_ips"][peak_idx] + .5).astype(int)], 
#     color = "C1"
# )
# plt.vlines(mz_C37_2, ymin=y[mask].min(), ymax=y[mask].max(), linestyles='--', colors='C2')

# %% cross-correlation to find offset
def xcorr(a, b, maxlags=1):
    """
    Calculate crosscorrelation for a and b within a certain lag window.

    Parameters
    ----------
    a : Iterable
        DESCRIPTION.
    b : Iterable
        DESCRIPTION.
    maxlags : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    lags : TYPE
        DESCRIPTION.
    corrs : TYPE
        DESCRIPTION.

    """
    assert len(a) == len(b), 'inputs must have same length'
    N = len(b)

    b = np.pad(b.copy(), maxlags)    
    lags = np.arange(-maxlags, maxlags+1, dtype=int)
    corrs = np.array([
        np.correlate(
            a, 
            b[maxlags + lag:maxlags + lag + N], 
            mode='valid'
        )[0] for lag in lags
    ])
    return lags, corrs

def find_offset(a, b, mz, maxdifference: float, plts=False):
    assert len(np.unique(np.diff(mz))) == 1, \
        'masses must be equally spaced, interpolate if necessary'
    # convert difference to lag
    dmz = mz[1] - mz[0]
    maxlags = int(maxdifference / dmz) + 1
    lags, corrs = xcorr(a, b, maxlags)
    idx = np.argmax(corrs)
    lag = lags[idx]
    offset = dmz * lag
    if plts:
        plt.figure()
        plt.plot(dmz * lags, corrs)
        plt.xlabel('offset in Da')
        plt.ylabel('crosscorrelation')
        plt.show()
    
    return offset

# a = spec1.intensities.copy()
# b = spec2.intensities.copy()

# lags, corrs = xcorr(spec1.intensities, spec2.intensities, int(.1/.0001))
# offset_from_a_to_b = find_offset(spec1.intensities, spec2.intensities, spec1.mzs, .1, plts=True)

# plt.figure()
# mask = (spec1.mzs > 1310) & (spec1.mzs < 1314.4)
# plt.plot(spec1.mzs[mask], spec1.intensities[mask], label='target')
# plt.plot(spec2.mzs[mask], spec2.intensities[mask], label='source')
# plt.plot(spec2.mzs[mask] + offset_from_a_to_b, spec2.intensities[mask], label='shifted')
# plt.legend()
# plt.show()

# %% compare profile, line
# import matplotlib.pyplot as plt
# window = (1314.2, 1314.24)
# mask = (spec1.mzs > window[0]) & (spec1.mzs < window[1])
# plt.plot(spec1.mzs[mask], spec1.intensities[mask], '-')

# mask = (mzs[0] > window[0]) & (mzs[0] < window[1])
# plt.stem(mzs[0][mask], intensities[0][mask], markerfmt='', linefmt='C1-', label='target')

# mask = (mzs[1] > window[0]) & (mzs[1] < window[1])
# plt.stem(mzs[1][mask], intensities[1][mask], markerfmt='', linefmt='C2-', label='source')

# mask = (mzs[1] > window[0]) & (mzs[1] < window[1])
# plt.stem(mzs[1][mask] + offset_from_a_to_b, intensities[1][mask], markerfmt='', linefmt='C3-', label='shifted')
# plt.legend()
# plt.show()

# import xmltodict
# import json
# f = r'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 GDGT.i/2018_08_27 Cariaco 490-495 GDGT.d/LDI pos 600_2000_256k_CASI_1310_1330 2018_08_20.m/apexAcquisition.method'

# with open(f, 'r') as xml_file:
#     xml_data = xml_file.read()

# # Convert XML to Python dictionary
# xml_dict = xmltodict.parse(xml_data)

# # Convert Python dictionary to JSON string
# json_data = json.dumps(xml_dict, indent=2)
