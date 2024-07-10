from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
from exporting.from_mcf.rtms_communicator import Spectra

d_folder = 'D:/Cariaco Data for Weimin/490-495cm/2018_08_27 Cariaco 490-495 alkenones.i/2018_08_27 Cariaco 490-495 alkenones.d'

# spec = con.get_spectrum(4, limits=(548, 560))
# coords_triples = np.load('cariaco_490-495_spec4_peaks.npy')
# coords_triples = np.load('cariaco_490-495_peaks_summed.npy')
spec = Spectra(path_d_folder=d_folder, initiate=False)
spec.load()

# fig, ax = plt.subplots()
# ax.plot(spec.mzs, spec.intensities)
# ax.scatter(
#     coords_triples[::3], 
#     [
#           spec.intensities[np.argmin(np.abs(spec.mzs - mz))]
#           for mz in coords_triples[::3]
#     ],
#     c='r'         
# )
# ax.scatter(
#     coords_triples[1::3], 
#     [
#           spec.intensities[np.argmin(np.abs(spec.mzs - mz))]
#           for mz in coords_triples[1::3]
#     ],
#     c='g'         
# )
# ax.scatter(
#     coords_triples[2::3], 
#     [
#           spec.intensities[np.argmin(np.abs(spec.mzs - mz))]
#           for mz in coords_triples[2::3]
#     ],
#     c='r'         
# )

# coords_peaks = []

# def onclick(event):
#     global coords_peaks
#     if event.button == 3:
#         coords_peaks.append(event.xdata)
    

# cid = fig.canvas.mpl_connect('button_press_event', onclick)

mzs_peaks = np.load(r'G:/Meine Ablage/Promotion/msi_workflow/testing/ringing.npy')
ints_peaks = np.array([
    spec.intensities[np.argmin(np.abs(spec.mzs - mz))] for mz in mzs_peaks
])

def sinc(x, x_c=0, period=1, amplitude=1):
    return amplitude * np.sinc((x - x_c) * period)

def sinc_squared(x, x_c=0, period=1, amplitude=1):
    return sinc(x, x_c, period, np.sqrt(amplitude)) ** 2

def sinc_abs(x, x_c, period, amplitude):
    return np.abs(sinc(x, x_c, period, amplitude))

x_lim = (552-.025, 552+.025)
x_c = 552.
period = 1 / .005
amplitude = ints_peaks[2]

x = np.linspace(x_lim[0], x_lim[1], 1000)
y = sinc_abs(x, x_c, period, amplitude)

mask = (spec.mzs >= x_lim[0]) & (spec.mzs <= x_lim[1])
spec_mzs = spec.mzs[mask]
spec_intensities = spec.intensities[mask]

# params, *_ = curve_fit(
#     sinc_abs, xdata=spec_mzs, ydata=spec_intensities, p0=(x_c, period, amplitude)
# )
# y_fit = sinc_squared(x, *params)

fig, ax = plt.subplots()
ax.plot(spec_mzs, spec_intensities, label='original')
ax.plot(x, y, label='intial guess')
# ax.plot(x, y_fit, label='fit')
ax.scatter(mzs_peaks, ints_peaks)
ax.set_xlabel('m/z in Da')
ax.set_ylabel('Intensity')

plt.legend()
plt.show()