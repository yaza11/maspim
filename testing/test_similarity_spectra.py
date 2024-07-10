import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exporting.from_mcf.cSpectrum import Spectra
from exporting.sqlite_mcf_communicator.hdf5Handler import hdf5Handler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_d_folder = "D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d"

reader = hdf5Handler(path_d_folder)

# spectra = Spectra(reader=reader)

def spectra_similarities(reader, plts=False):
    II = reader.read()['intensities']
    # normalize such that the max intensity of each spec is 1
    means = II.mean(axis = 1)
    # add new axis
    means = means[:, None]
    II = II / means
    # covariance matrix
    covs = II @ II.T
    if plts:
        covs_plt = covs.copy()
        # normalize
        cov_max = covs.max()
        covs /= cov_max
        
        clip_val = np.quantile(covs, .95)
        covs[covs > clip_val] = clip_val
        plt.imshow(covs, interpolation="none")
        plt.show()
    return covs



