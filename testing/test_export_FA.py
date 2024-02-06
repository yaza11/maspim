import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from exporting_mcf.rtms_communicator import ReadBrukerMCF

# d_folder = "D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 FA.i/2018_08_27 Cariaco 490-495 FA.d"
# d_folder = "D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 GDGT.i/2018_08_27 Cariaco 490-495 GDGT.d"
# d_folder = "//10.111.38.201/scratch/Jannis/FTdata/oldlaser/13012023_SBB_TGA3_5-10_Test2_5/SBB_TGA3_5-10_Test2_5.d"
d_folder = 'D:/Cariaco Data for Weimin/505-510cm/2018_08_30 Cariaco 505-510 FA.i/2018_08_30 Cariaco 505-510 FA B.d'
con = ReadBrukerMCF(d_folder)
con.create_reader()
con.create_indices()
con.create_spots()

# %%
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

limits=(388, 481)

# %%
# plt.figure()
spec1 = con.get_spectrum(int(con.indices[0]), limits=limits)    
spec1.resample()
mzs = spec1.mzs
spec1intensities = spec1.intensities
offsets = []
for index in con.indices[:200]:
    spec = con.get_spectrum(int(index), limits=limits)    
    offset = find_offset(spec1intensities, spec.intensities, mzs, maxdifference=.1)
    offsets.append(offset)
    # plt.plot(spec.mzs, spec.intensities)
# plt.show()
