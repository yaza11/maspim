# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:24:01 2024

@author: Yannick Zander
"""

from exporting.sqlite_mcf_communicator.sql_to_mcf import get_sql_files

import numpy as np

path_d_folder = r'D:\Cariaco Data for Weimin\505-510cm\2018_08_30 Cariaco 505-510 alkenones.i\2018_08_30 Cariaco 505-510 alkenones.d'


files = get_sql_files(path_d_folder)

mzs = files['mzs']
intensities = files['intensities']
fwhm = files['fwhm']

N_spectra, N_points = mzs.shape

# this takes forever, dont use!!!
def dissimilarity(mzs_a, mzs_b, intensities_a, intensities_b):
    N_points_a = mzs_a.shape[0]
    collector = 0
    for i in range(N_points_a):
        # for every mz in a find the closest b and take the difference in intensities
        mz_a, intensity_a = mzs_a[i], intensities_a[i]
        idx = np.argmin(np.abs(mz_a - mzs_b)) 
        collector += np.abs(
            mz_a * intensity_a - mzs_b[idx] * intensities_b[idx]
        )
    return collector

N_spectra = 10

dissimiliarities = np.zeros((N_spectra, N_spectra))

for i in range(N_spectra):
    mzs_i = mzs[i]
    intensities_i = intensities[i]
    for j in range(N_spectra):
        dissimiliarities[i, j] = dissimilarity(
            mzs_a=mzs_i, 
            mzs_b=mzs[j], 
            intensities_a=intensities_i, 
            intensities_b=intensities[j]
        )




        