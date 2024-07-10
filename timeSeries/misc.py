"""Module for testing continous-wavelet transformation."""
from timeSeries.cProxy import UK37
from timeSeries.cTimeSeries import TimeSeries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal.windows import blackman

import pycwt as wavelet
from pycwt.helpers import find

def get_dat(
        ts: TimeSeries,
        target: str,
        use_laminae_age: bool,
        use_light_dark_both: str,
        dt: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Get data corresponding to light or dark layers."""
    mask = (ts.feature_table.seed != 0) & (~ts.feature_table[target].isna())

    # only light layers
    if use_light_dark_both == 'light':
        mask &= ts.feature_table.seed > 0
        dt *= 2
    elif use_light_dark_both == 'dark':
        mask &= ts.feature_table.seed < 0
        dt *= 2
    elif use_light_dark_both == 'both':
        pass
    else:
        raise KeyError()

    # assuming we found all laminae and classified them correctly
    if use_laminae_age:
        dat = ts.feature_table.loc[mask, target].to_numpy()
        N = len(dat)
        t = np.arange(N) * dt + round(ts.feature_table.age.min())
    else:
        df: pd.DataFrame = ts.feature_table.loc[mask, [target, 'age']].copy()
        df['age_rounded'] = dt * np.around(df.age / dt)
        # average duplicates
        df = df.groupby('age', as_index=False).mean()
        # add missing values
        ages: np.ndarray[float] = np.arange(
            df.age_rounded.min(),
            df.age_rounded.max() + dt,
            dt
        )

        dat: np.ndarray[float] = np.interp(ages, df['age_rounded'], df[target])

        t = ages

    return t, dat, dt


def plt_cwt(
        ts: TimeSeries,
        target: str,
        use_laminae_age: bool,
        use_light_dark_both: str,
        title: str | None = None,
        unit='1',
        period1: int = 2,
        period2: int = 8,
        dt: float = 1
):
    t, dat, dt = get_dat(ts=ts, target=target, use_laminae_age=use_laminae_age, use_light_dark_both=use_light_dark_both, dt=dt)
    t0 = t.min()
    N = len(dat)

    if title is None:
        title = 'UK37 proxy' if target == 'ratio' else target
    label = title
    units = '1'

    p = np.polyfit(t - t0, dat, 1)
    dat_notrend = dat - np.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset

    fig, axs = plt.subplots(nrows=2)

    weights = blackman(len(dat_notrend))

    fft_tapered = np.fft.fft(dat_notrend * weights)
    fft = np.fft.fft(dat_notrend)
    freq = np.fft.fftfreq(t.shape[-1], dt)

    axs[0].plot(t, dat_notrend * weights)
    axs[0].set_xlabel('age in yrs b2k')
    axs[0].set_ylabel('UK37p val')

    axs[1].stem(freq, np.abs(fft_tapered) ** 2, markerfmt='')
    axs[1].set_xscale('log')
    axs[1].set_ylabel('Power')
    axs[1].set_xlabel('Frequency in yrs')

    fig.tight_layout()
    plt.show()

    # FFT and LombScargle
    fig, axs = plt.subplots(nrows=2)

    mask = (1 / freq >= period1) & (1 / freq <= period2)

    ax = axs[0]
    ax.stem(freq, np.abs(fft) ** 2, markerfmt='')
    ax.stem(freq[mask], np.abs(fft[mask]) ** 2, markerfmt='', linefmt='r')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency in 1 / yrs')
    ax.set_ylabel('Power')

    ax = axs[1]
    dat_hat = np.fft.ifft(fft * mask.astype(float))
    ax.plot(t, dat_notrend, '--')
    ax.plot(t, dat_hat)
    ax.set_xlabel('age in yrs b2k')
    ax.set_ylabel(label)
    ax.set_title(f'Time series in {period1}-{period2} period band')
    plt.show()

    ts_light = ts.copy()
    ts_light.feature_table = pd.DataFrame({'age': t, target: dat})
    ts_light.power(targets=[target], plts=True)

    omega0 = 6
    mother = wavelet.Morlet(omega0)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    # J = 7 / dj  # Seven powers of two with dj sub-octaves
    J = 7 / dj
    alpha, _, _ = wavelet.ar1(dat_norm)  # Lag-1 autocorrelation for red noise

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                          mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(
        var, dt, scales, 1, alpha,
        significance_level=0.95, dof=dof,
        wavelet=mother
    )

    sel = find((period >= period1) & (period < period2))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
    scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                                 significance_level=0.95,
                                                 dof=[scales[sel[0]],
                                                      scales[sel[-1]]],
                                                 wavelet=mother)

    # Prepare the figure
    plt.close('all')
    plt.ioff()
    figprops = dict(figsize=(11, 8), dpi=72)
    fig = plt.figure(**figprops)

    age_model = 'relative' if use_laminae_age else 'absolute'
    suptitle = f'CWT with {age_model} age model, {use_light_dark_both} laminae\n\n'

    # First sub-plot, the original time series anomaly and inverse wavelet
    # fit.
    ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    ax.plot(t, np.real(iwave), '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax.plot(t, dat - dat.mean(), 'k', linewidth=1.5)
    ax.set_title(suptitle + 'a) {}'.format(title))
    ax.set_ylabel(r'{} [{}]'.format(label, units))

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
                extend='both', cmap=plt.cm.viridis)
    extent = [t.min(), t.max(), 0, max(period)]
    bx.contour(
        t,
        np.log2(period),
        sig95,
        [-99, 1],
        colors='k',
        linewidths=2,
        extent=extent
    )
    bx.fill(
        np.concatenate([
            t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt
        ]),
        np.concatenate([
            np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]
        ]),
        'k',
        alpha=0.3,
        hatch='x'
    )
    bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
    bx.set_ylabel('Period (years)')
    #
    Yticks = 2 ** np.arange(
        np.ceil(
            np.log2(period.min())
        ),
        np.ceil(
            np.log2(period.max())
        )
    )
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)

    # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
    # noise spectra. Note that period scale is logarithmic.
    cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
    cx.plot(glbl_signif, np.log2(period), 'k--')
    cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
    cx.plot(var * fft_power, np.log2(1. / fftfreqs), '-', color='#cccccc',
            linewidth=1.)
    cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    cx.set_title('c) Global Wavelet Spectrum')
    cx.set_xlabel(r'Power [({})^2]'.format(units))
    cx.set_xlim([0, glbl_power.max() * var])
    cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.set_yticks(np.log2(Yticks))
    cx.set_yticklabels(Yticks)
    plt.setp(cx.get_yticklabels(), visible=False)

    # Fourth sub-plot, the scale averaged wavelet spectrum.
    dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
    dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
    dx.plot(t, scale_avg, 'k-', linewidth=1.5)
    dx.set_title('d) {}--{} year scale-averaged power'.format(period1, period2))
    dx.set_xlabel('Time (year)')
    dx.set_ylabel(r'Average variance [{}]'.format(units))
    ax.set_xlim([t.min(), t.max()])

    plt.show()
