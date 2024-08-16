import pandas as pd
import os

elements = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'elements.txt'),
    sep='\t',
    names=['Z', 'Abbreviation'],
    index_col=None,
    usecols=[0, 1]
)

DEFAULT_MASS_TOLERANCE = 3e-3  # mDa

# for labeling plots and such
dict_labels = {'density_nonzero': r'$\sigma_\bar{0}$',
               'intensity_div': r'$\Delta I$',
               'av_intensity_light': r'$<I^\mathrm{l}>$',
               'av_intensity_dark': r'$<I^\mathrm{d}>$',
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

key_light_pixels: int = 255
key_dark_pixels: int = 127
key_hole_pixels: int = 0

YD_transition = 11_673  # yr b2k
YD_transition_depth = 504.285  # cm

