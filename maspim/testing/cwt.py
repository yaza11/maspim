import pywt
import numpy as np
import matplotlib.pyplot as plt


# %%
def cmor(t, bandwidth, center_frequency, t_offset=0, scale=1):
    t = (t - t_offset) / scale
    return 1 / np.sqrt(np.pi * bandwidth) * np.exp(-t ** 2 / bandwidth) * np.exp(
        2j * np.pi * center_frequency * t)


def calculate_cone_of_influence(scales):
    cone_of_influence: np.ndarray = np.ceil(np.sqrt(2) * scales).astype(np.int64)
    return cone_of_influence

sample_interval = .1
bandwidth = 20  # yrs, width of the wavelet
center_frequency = 1  # 1 / yrs, frequency of modulated signal, max frequency is center_frequency / sample_interval, so no reason to go below 1
scales = np.geomspace(1, 1024, num=128)

ages = np.arange(1, 50 + sample_interval, sample_interval)  # yrs
ages_center = (ages.max() - ages.min()) / 2 + ages.min()
freqs_source = np.array([1, 2.5]) * 2 * np.pi
intensities = np.sin(freqs_source[0] * ages) + .5 * np.sin(freqs_source[1] * ages) * (ages > ages_center)

var = np.var(intensities, ddof=1)


# plt.plot(ages, np.abs(cmor(ages, bandwidth, center_frequency, t_offset=ages_center)), 'r',
#          label='abs(cmor) with B={bandwidth}, C={center_frequency}'.format(bandwidth=bandwidth,
#                                                                            center_frequency=center_frequency))
# plt.plot(ages, np.real(cmor(ages, bandwidth, center_frequency, t_offset=ages_center)), label='real(cmor)')
# plt.plot(ages, np.imag(cmor(ages, bandwidth, center_frequency, t_offset=ages_center)), label='imag(cmor)')
for scale in [scales.min(), np.median(scales), scales.max()]:
    plt.plot(ages, np.real(cmor(ages, bandwidth, center_frequency, t_offset=ages_center, scale=scale)),
             label='real(cmor) with scale={scale:.1f}'.format(scale=scale))


plt.plot(ages, intensities, label='signal')
plt.xlabel("Time (yrs)")
plt.ylabel("Intensity")
plt.legend()

plt.show()

# %%
wavelet = f"cmor{bandwidth}-{center_frequency}"
# logarithmic scale for scales, as suggested by Torrence & Compo:
cwtmatr, freqs = pywt.cwt(intensities, scales, wavelet, sampling_period=sample_interval)
# absolute take absolute value of complex result
# cwtmatr = np.abs(cwtmatr[:-1, :-1])


cone_of_influence = calculate_cone_of_influence(scales)
N_ages = len(ages)
idxs_l = np.array([c for c in cone_of_influence if c < N_ages // 2])
idxs_r = (N_ages - idxs_l)[::-1]
idxs = np.hstack([idxs_l, idxs_r])
cone_of_influence = ages[idxs]
cone_freqs_l = freqs[:len(idxs_l)]
cone_freqs_r = freqs[:len(idxs_l)][::-1]
cone_freqs = np.hstack([cone_freqs_l, cone_freqs_r])

# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1)
pcm = axs[0].pcolormesh(ages, freqs, np.abs(cwtmatr))
axs[0].set_yscale("log")
axs[0].set_xlabel("Age (yrs b2k)")
axs[0].set_ylabel("Frequency (1 / yr)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
axs[0].hlines(freqs_source / 2 / np.pi, ages.min(), ages.max(), linestyles="--", color="r", alpha=.5)
axs[0].plot(cone_of_influence, cone_freqs, 'g')

fig.colorbar(pcm, ax=axs[0])

# plot fourier fit for comparison
yf = np.fft.rfft(intensities)
xf = np.fft.rfftfreq(len(intensities), sample_interval)
# axs[1].semilogx(xf, np.abs(yf))
axs[1].plot(xf, np.abs(yf))
axs[1].vlines(freqs_source / 2 / np.pi, 0, np.max(np.abs(yf)), linestyles="--", color="r")
axs[1].set_xlabel("Frequency (1 / yr)")
axs[1].set_title("Fourier Transform")

plt.tight_layout()

plt.show()
