from scipy.constants import physical_constants

m_e = physical_constants['electron mass in u'][0]

# masses
mNa_p = 22.989770 - m_e

##### FA window #################################################################
# steranes
mass_C28_sterane_Na_p = mC28 = 386.391250 + mNa_p
mass_C29_sterane_Na_p = mC29 = 400.406900 + mNa_p

# stanol, stenol
mass_C29_stanol_Na_p = mC29stanol = 416.401815 + mNa_p
mass_C29_stenol_Na_p = mC29stenol = 414.386165 + mNa_p

# fatty acids
mC24FA = mass_C24_FA_Na_p = 368.365430 + mNa_p  # CH3(CH2)22CO2H
mC26FA = mass_C26_FA_Na_p = 396.396730 + mNa_p  # CH3(CH2)24CO2H
mC28FA = mass_C28_FA_Na_p = 424.428030 + mNa_p  # CH3(CH2)26CO2H
mC30FA = mass_C30_FA_Na_p = 452.459330 + mNa_p  # CH3(CH2)28CO2H

##### Alkenone window #################################################################
# Na+ C37:2 mass: 553.53188756536
# Na+ C37:3 mass: 551.51623750122
mass_C37_2_Na_p = mC37_2 = 553.53188756536
mass_C37_3_Na_p = mC37_3 = 551.51623750122

mass_pyropheophorbide_a_Na_p = mPyro = 534.263091 + mNa_p

##### GDGT window #################################################################
# https://en.wikipedia.org/wiki/TEX86#/media/File:Molecular_structures_and_HPLC_detection_of_GDGTs.jpg
mGDGT0 = mass_GDGT_0_Na_p = 1301.315390 + mNa_p
mGDGT1 = mass_GDGT_1_Na_p = 1299.299740 + mNa_p
mGDGT2 = mass_GDGT_2_Na_p = 1297.284090 + mNa_p
mGDGT3 = mass_GDGT_3_Na_p = 1295.268440 + mNa_p
mCren_p = mass_cren_prime_Na_p = 1291.237140 + mNa_p

# outside mass window
# mass_GDGT_I = 1006.986740
# mass_GDGT_II = 1021.002390
# mass_GDGT_III = 1035.018040

