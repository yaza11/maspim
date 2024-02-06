import pandas as pd
import os

elements = pd.read_csv(os.path.join(os.path.dirname(__file__), 'elements.txt'),
                       sep='\t',
                       names=['Z', 'Abbreviation'],
                       index_col=None,
                       usecols=[0, 1])

max_deviation_mz = 3e-3  # mDa
# thr for peak detection in feature table creation
peak_threshold_ref_peaks_msi_raw_feature_table = .1

# distance between data points in meter for each datatype
distance_pixels_msi = 100e-6  # meter
distance_pixels_xrf = 50e-6  # meter
distance_pixels = {'msi': distance_pixels_msi,
                   'xrf': distance_pixels_xrf,
                   'combined': distance_pixels_msi}


def window_to_mass_window(window: str) -> tuple[int]:
    d = {'alkenones': (548, 560),
         'fa': (388, 481),
         'gdgt': (1310, 1330),
         'xrf': (None, None)}
    return d[window.lower()]


windows_all = ['xrf', 'FA', 'GDGT', 'Alkenones']
# sections_all = [(490, 495), (495, 500), (500, 505), (505, 510), (510, 515), (515, 520)]
sections_all = [(490, 495), (495, 500), (500, 505), (505, 510)]
# the standard target for correcting distortions
# (all other mass windows will be mapped to this one)
transformation_target = 'Alkenones'


def window_to_type(window: str) -> str:
    window_l = window.lower()
    if window_l in ('fa', 'gdgt', 'alkenones', 'msi'):
        return 'msi'
    elif window_l in ('xrf'):
        return 'xrf'
    elif window_l == 'combined':
        return 'combined'
    elif window_l == 'none':
        return 'none'
    else:
        raise KeyError(f'{window=} is not a valid window')


# for labeling plots and such
dict_labels = {'density_nonzero': r'$\sigma_\bar{0}$',
               'intensity_div': r'$\Delta I$',
               'av_intensity_light': r'$<I^\mathrm{l}>$',
               'av_intensity_dark': r'$<I^\mathrm{d}>$',
               'intensity_div': r'$\Delta I$',
               'corr_classification': r'$r_\mathrm{C}$',
               'corr_classification_s': r'$r_\mathrm{C}$',
               'corr_L': r'$r_\mathrm{L}$',
               'corr_av_L': r'$r_\bar{\mathrm{L}}$',
               'corr_av_C': r'$r_\bar{\mathrm{C}}$',
               'KL_div': r'$D_\mathrm{KL}$',
               'KL_div_light': r'$D_\mathrm{KL}^\mathrm{l}$',
               'KL_div_dark': r'$D_\mathrm{KL}^\mathrm{d}$',
               'score': r'$s$',
               'seasonalities_median': r'$|c|$',
               'av': r'$<I_\mathrm{layer}>$',
               'contrast_med': r'$c_\text{median}$'}

key_light_pixels = 255
key_dark_pixels = 127
key_hole_pixels = 0


# number of successful spectra required to be meaningful
n_successes_required = 10

# Na+ C37:2 mass: 553.53188756536
# Na+ C37:3 mass: 551.51623750122
mass_C37_2_Na_p = mC37_2 = 553.53188756536
mass_C37_3_Na_p = mC37_3 = 551.51623750122

YD_transition = 11_673  # yr b2k
YD_transition_depth = 504.285  # cm

# scale the contrast criterion to 1 across all sections and windows
# calculated in ImageEnvironment with get_all_contrasts()
# for
# ['xrf', 'FA', 'GDGT', 'Alkenones'],
# [(490, 495), (495, 500), (500, 505), (505, 510)]
contrasts_scaling = 1 / 0.03833409461067907
