import os
import sys

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.transform import rotate
from scipy.signal import convolve
from typing import Iterable
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


from imaging.util.Image_processing import filter_with_mask_by_rescaling_weights


def morlet(x: float | np.ndarray[float], sigma: float = 1.) -> complex | np.ndarray[complex]:
    c0 = 1 / np.sqrt(1 + np.exp(- sigma ** 2) - 2 * np.exp(- 3 / 4 * sigma ** 2))
    c = c0 / np.pi ** (1 / 4)
    k = np.exp(- 1 / 2 * sigma ** 2)
    psi = c * np.exp(- 1 / 2 * x ** 2) * (np.exp(1j * sigma * x) - k)
    return psi


def gabor(x, y, lam, theta, psi, sigma, gamma):
    """
    https://en.wikipedia.org/wiki/Gabor_filter

    In this equation,
    lambda represents the wavelength of the sinusoidal factor,
    theta represents the orientation of the normal to the parallel stripes of a Gabor function,
    psi is the phase offset,
    sigma is the sigma/standard deviation of the Gaussian envelope and
    gamma is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function.
    """
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = -x * np.sin(theta) + y * np.cos(theta)
    frac = (xp ** 2 + (gamma * yp) ** 2) / (2 * sigma ** 2)
    phase = 2 * np.pi * xp / lam + psi

    gaussian = np.exp(-frac)
    sine = np.exp(1j * phase)
    g = gaussian * sine
    return g


def mask_circ(side_length: int) -> np.ndarray[bool]:
    a = np.linspace(-1, 1, side_length, endpoint=True)
    xx, yy = np.meshgrid(a, a)
    m = xx ** 2 + yy ** 2 <= 1
    return m

def get_circle_kernel(angle, profile):
    w = profile.shape[0]
    pad = round(w / 2 * (np.sqrt(2) - 1))
    footprint = np.ones((w, w)) * profile[None, :]
    footprint = np.pad(footprint, ((pad, pad), (pad, pad)))
    kernel = rotate(footprint, -angle * 180 / np.pi)
    # zero mean
    mask = kernel != 0
    mean = kernel.mean()
    kernel[mask] -= mean
    # unit vol
    k = np.abs(kernel).mean()
    kernel /= k
    return kernel


def get_kernels(
        *,
        n_angles: int = 32,
        image_shape: tuple[int, ...] | None = None,
        max_width: int | None = None,
        n_widths: int = 16
) -> tuple[list[np.ndarray[float]], list[dict]]:
    """
    Create kernels which have Morlet wavelets as profile and have a circular footprint.

    The Morlet wavelets are taken from -pi to pi with sigma = 1, which gives one center
    peak with two smaller side peaks for the center-kernels and one up and one down pea
    for the edge peaks with both profiles vanishing at the bounds.

    Parameters
    ----------
    n_angles : int
        Number of angles between plus and minus 45 degrees for which to get kernels.
        The default is 32.
    diams : Iterable[int], optional
        The diameters of the footprints. If not provided, will be estimated from the image shape
        with n_diams values on a log scale from 5 to image_width
    image_shape : tuple, optional
        The shape of the image on which to apply the kernels. Must be provided, if diams is
        not specified.
    n_diams : int, optional
        number of diameters on the log scale. The default is 16. Will not be used if diams is
        provided.

    Returns
    -------
    kernels, params: tuple[list[np.ndarray[float]], list[dict]]
        The center and edge kernels together with a list of parameters.
    """
    assert (image_shape is not None)
    # angles varying between + and -45 deg
    angles = np.linspace(-np.pi / 4, np.pi / 4, n_angles, endpoint=True)

    if max_width is None:
        max_width = image_shape[0] // 5
    min_width = 15

    # log scale
    # widths = np.logspace(np.log2(min_width), np.log2(max_width), base=2, num=n_widths)
    widths = np.linspace(min_width, max_width, n_widths)
    print(f'kernel periods between {widths[0] * 2:.0f} and {widths[-1] * 2:.0f} pixels')
    
    nx = round(max_width * 2)
    x = np.linspace(-max_width, max_width, nx, endpoint=True)

    kernels_diam = []
    params = []

    for width in widths:
        kernels = []

        profile_center = np.zeros_like(x, dtype=int)
        profile_center[np.abs(x) <= width] = -1
        profile_center[np.abs(x) <= width / 2] = 1

        profile_edge = np.zeros_like(x, dtype=int)
        profile_edge[(x <= width) & (x > 0)] = 1
        profile_edge[(x >= -width) & (x < 0)] = -1

        for angle in angles:
            params_dict = dict(lam=width, angle=angle)
            kernel_center = get_circle_kernel(angle, profile_center)
            kernel_edge = get_circle_kernel(angle, profile_edge)

            kernels.append(kernel_center)
            params.append(params_dict | dict(phase='peak'))

            kernels.append(kernel_edge)
            params.append(params_dict | dict(phase='edge'))

        # stack along new dimension
        kernels_diam.append(
            np.stack(
                kernels,
                axis=-1
            )
        )

    return kernels_diam, params


def xxget_kernels(
        *,
        n_angles: int = 32,
        image_shape: tuple[int, ...] | None = None,
        max_width: int | None = None,
        n_lams: int = 16
) -> tuple[list[np.ndarray[float]], list[dict]]:
    """
    Create kernels which have Morlet wavelets as profile and have a circular footprint.

    The Morlet wavelets are taken from -pi to pi with sigma = 1, which gives one center
    peak with two smaller side peaks for the center-kernels and one up and one down pea
    for the edge peaks with both profiles vanishing at the bounds.

    Parameters
    ----------
    n_angles : int
        Number of angles between plus and minus 45 degrees for which to get kernels.
        The default is 32.
    diams : Iterable[int], optional
        The diameters of the footprints. If not provided, will be estimated from the image shape
        with n_diams values on a log scale from 5 to image_width
    image_shape : tuple, optional
        The shape of the image on which to apply the kernels. Must be provided, if diams is
        not specified.
    n_diams : int, optional
        number of diameters on the log scale. The default is 16. Will not be used if diams is
        provided.

    Returns
    -------
    kernels, params: tuple[list[np.ndarray[float]], list[dict]]
        The center and edge kernels together with a list of parameters.
    """
    assert (image_shape is not None)
    # angles varying between + and -45 deg
    angles = np.linspace(-np.pi / 4, np.pi / 4, n_angles, endpoint=True)

    if max_width is None:
        max_width = image_shape[0] // 5

    widths = np.linspace(max_width / 10, max_width, n_lams, endpoint=True)
    lams = 2 * widths

    kernels_diam = []
    params = []

    for lam, width in zip(lams, widths):
        kernels = []
        nx = int(np.ceil(2 * width))
        x: np.ndarray[float] = np.linspace(-3 / 4 * lam, 3 / 4 * lam, nx, endpoint=True)
        sigma = lam / 2.43798745675  # numerical value tuned to zero mean for large widths

        xx, yy = np.meshgrid(x, x)
        for angle in angles:
            kernel: np.ndarray[complex] = gabor(
                xx,
                yy,
                lam=lam,
                theta=angle,
                psi=0,
                sigma=sigma,
                gamma=0
            )
            kernel *= mask_circ(nx)
            # try to balance benefit from more points with
            # randomness from too few points
            kernel /= (nx / 2) ** 2 * np.pi
            kernel_center = kernel.real
            kernel_edge = kernel.imag

            # profile: np.ndarray[complex] = morlet(x, lam)
            # profile_center: np.ndarray[float] = profile.real
            # profile_edge: np.ndarray[float] = profile.imag

            params_dict = dict(lam=lam, angle=angle)
            # kernels.append(get_circle_kernel(angle, profile_center))
            kernels.append(kernel_center)
            params.append(params_dict | dict(phase='peak'))

            # kernels.append(get_circle_kernel(angle, profile_edge))
            kernels.append(kernel_edge)
            params.append(params_dict | dict(phase='edge'))

            # same information as taking negative inverse of convolution
            # kernels.append(get_circle_kernel(angle, -profile_center))
            # params.append(params_dict | dict(phase='trough'))
            #
            # kernels.append(get_circle_kernel(angle, -profile_edge))
            # params.append(params_dict | dict(phase='falling'))

            # stack along new dimension
        kernels_diam.append(
            np.stack(
                kernels,
                axis=-1
            )
        )

    return kernels_diam, params


def test_image(theta=0) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.linspace(0, np.pi, 200)
    xx, yy = np.meshgrid(x, y)
    n = 10
    zz = np.sin(xx * np.cos(theta) * n + yy * np.sin(theta) * n)
    print(f'period: {400 / n} pixels')

    return zz


def test_real():
    from PIL import Image
    from PIL.ImageOps import autocontrast
    from skimage.filters import threshold_otsu
    
    img = Image.open('C:/Users/Yannick Zander/Downloads/2020_03_23_Cariaco_535-540cm_ROI.bmp').convert('L')
    # img = np.array(Image.open('C:/Users/Yannick Zander/Downloads/S0363b_Cariaco_525-530cm_100um_0001_roi.tif').convert('L'))
    img = np.array(img)

    thr = threshold_otsu(img)
    bg = img < thr

    # img = img.astype(float) - img[~bg].mean()
    # img /= img.max()
    # img[bg] = 0
    img = img.astype(float)
    img -= img[~bg].min()
    img *= 2 / img[~bg].max()
    img -= img[~bg].mean()
    img[bg] = 0
    return img, ~bg


def test_cl():
    import pickle
    from skimage.transform import resize
    file = r'F:/535-540cm/2020_03_23_Cariaco_535-540cm_Alkenones.i/ImageClassified.pickle'
    with open(file, 'rb') as f:
        d = pickle.load(f)
    
    img = d._image_classification.astype(int)
    img[img == 255] = 1
    img[img == 127] = -1
    img[img == 0] = -1
    
    k = 500 / img.shape[0]
    img_shape = round(k * img.shape[0]), round(k * img.shape[1])
    img = resize(img, img_shape)
    
    return img


def add_negative(res, params):
    res = np.dstack((res, -res))
    mparams = []
    for d in params:
        d_new = d.copy()
        d_new['phase'] = 'trough' if d['phase'] == 'peak' else 'medge'
        mparams.append(d_new)
    params = params + mparams
    return res, params


def get_res(img, kernels, params):
    n_diam = kernels[0].shape[-1]
    img_diam = np.repeat(img[:, :, np.newaxis], n_diam, axis=-1)
    # mask_diam = np.repeat(mask.astype(float)[:, :, np.newaxis], n_diam, axis=-1)

    res = []
    for kernel_diam in tqdm(kernels, desc='calculating convolutions'):
        # res.append(convolve(img_diam, kernel_diam, mode='same'))
        r = convolve(img_diam, kernel_diam, mode='same')
        # w = convolve(mask_diam, kernel_diam, mode='same')
        # valid = w != 0
        # r[valid] /= w[valid]
        # r *= mask_diam
        res.append(r)
    res = np.dstack(res)

    res, params = add_negative(res, params)

    params_df = pd.DataFrame(params)

    return res, params_df


def get_angles(res, params_df):
    # max
    f_max = np.argmax(res, axis=-1)
    angles = params_df.loc[f_max.ravel(), 'angle'].to_numpy()
    return angles.reshape(f_max.shape)
    
    # weighted sum
    # weights = conv values
    # res_ = np.abs(res[:, :, :res.shape[-1]//2])
    # sums = res_.sum(axis=-1)
    # angles = params_df.loc[:params_df.shape[0] // 2 - 1, 'angle'].to_numpy()
    # angles_ = angles[None, None, :]
    
    # # broadcast
    # weighted_mean = (res_ * angles_).sum(axis=-1) / sums

    # return weighted_mean


def get_phases(res, params_df):
    f_max = np.argmax(res, axis=-1)
    phases = params_df.loc[f_max.ravel(), 'phase'].to_numpy()

    mapper = {'peak': 2, 'trough': -2, 'edge': 1, 'medge': -1}

    p_new = np.array([mapper[p] for p in phases])
    return p_new.reshape(f_max.shape)

    # res_ = np.abs(res[:, :, :res.shape[-1]//2])
    # sums = res_.sum(axis=-1)
    # phases = params_df.loc[:params_df.shape[0] // 2 - 1, 'phase'].to_numpy()

    # mapper = {'peak': 2, 'trough': -2, 'edge': 1, 'medge': -1}
    # p_new = np.array([mapper[p] for p in phases])

    # phases_ = p_new[None, None, :]

    # # broadcast
    # weighted_mean = (res_ * phases_).sum(axis=-1) / sums

    # return weighted_mean


def get_widths(res, params_df):
    f_max = np.argmax(res, axis=-1)

    lams = params_df.loc[f_max.ravel(), 'lam'].to_numpy()
    lams = lams.reshape(f_max.shape)

    return lams
    
    # res_ = np.abs(res[:, :, :res.shape[-1]//2])
    # sums = res_.sum(axis=-1)
    # widths = params_df.loc[:params_df.shape[0] // 2 - 1, 'lam'].to_numpy()
    # widths_ = widths[None, None, :]
    
    # # broadcast
    # weighted_mean = (res_ * widths_).sum(axis=-1) / sums

    # return weighted_mean


def get_res_params_for(res, params_df, phase=None, lam=None, angle=None):
    if lam is not None:
        lams = np.unique(params_df.lam)
        lam = lams[np.argmin(np.abs(lam - lams))]
        lam_mask = (params_df.lam == lam).to_numpy()
    else:
        lam_mask = np.ones(params_df.shape[0], dtype=bool)
    if phase is not None:
        phase_mask = (params_df.phase == phase).to_numpy()
    else:
        phase_mask = np.ones(params_df.shape[0], dtype=bool)
    if angle is not None:
        angle_mask = (params_df.angle == angle).to_numpy()
    else:
        angle_mask = np.ones(params_df.shape[0], dtype=bool)

    mask = lam_mask & phase_mask & angle_mask

    params_df_sub = params_df.loc[mask, :]
    params_df_sub.reset_index(inplace=True, drop=True)

    r = res[:, :, mask]

    return r, params_df_sub


def plt_kernels(kernels):
    ks_ = []
    for ks in kernels:
        k_ = np.hstack([ks[:, :, i] / np.abs(ks[:, :, i]).max() for i in range(ks.shape[-1])])
        ks_.append(k_)
    k = np.vstack(ks_)
    plt.imshow(k)
    plt.show()

def plt_gabor(width):
    lam = 2 * width
    print()
    x = np.linspace(-3 / 4 * lam, 3 / 4 * lam, int(np.ceil(2 * width)), endpoint=True)
    # sigma = lam / (2.4379704)
    sigma = lam / 2.43798745675
    osz = np.cos(2 * np.pi / lam * x)
    gauss = np.exp(-x ** 2 / 2 / sigma ** 2)

    plt.plot(x, osz)
    plt.plot(x, gauss)
    plt.plot(x, osz * gauss)
    plt.grid('on')
    plt.vlines([-width / 2, width / 2], osz.min(), osz.max(), 'k')
    plt.show()

    print(np.trapz(osz * gauss, x))


def plt_gabor_overview():
    kernels, params = get_kernels(n_angles=3, image_shape=(512, 512), n_lams=5)

    fig, axs = plt.subplots(nrows=2, ncols=5, sharex='col', sharey='col')

    idx90_p = 2
    idx90_e = 3

    for i in range(5):
        kernel_peak = kernels[i][:, :, idx90_p]
        kernel_edge = kernels[i][:, :, idx90_e]

        k = kernel_peak.shape[0]
        hk = k / 2
        hki = int(hk)
        x = range(k)

        axu, axd = axs[:, i]
        axu.imshow(kernel_peak)
        axu.plot(  # along x axis
            x,
            kernel_peak[hki, :] * hk + hk
        )
        axu.plot(
            kernel_peak[:, hki] * hk + hk,
            x
        )

        axd.imshow(kernel_edge)
        axd.plot(
            x,
            kernel_edge[hki, :] * hk + hk
        )
        axd.plot(
            kernel_edge[:, hki] * hk + hk,
            x
        )
        param = params[6 * i + 2]
        axu.set_title(fr'$\lambda$: {param["lam"]:.1f}')

    plt.tight_layout()
    plt.show()


def plt_kernel_on_img(img, kernels, krnl_idx):
    kernel = kernels[-1][:, :, krnl_idx].copy()
    # rescale
    kernel = kernel.astype(float) * img.max() / kernel.max()

    img_k = img.copy()
    img_k[
        img.shape[0] // 2 - kernel.shape[0] // 2:img.shape[0] // 2 - kernel.shape[0] // 2 + kernel.shape[0],
        img.shape[1] // 3 - kernel.shape[1] // 2:img.shape[1] // 3 - kernel.shape[1] // 2 + kernel.shape[1]
    ] = kernel

    kernel = kernels[0][:, :, krnl_idx].copy()
    # rescale
    kernel = kernel.astype(float) * img.max() / kernel.max()

    img_k[
        img.shape[0] // 2 - kernel.shape[0] // 2:img.shape[0] // 2 - kernel.shape[0] // 2 + kernel.shape[0],
        img.shape[1] * 2 // 3 - kernel.shape[1] // 2:img.shape[1] * 2 // 3 - kernel.shape[1] // 2 + kernel.shape[1]
    ] = kernel

    plt.imshow(img_k)
    plt.title('Input image with biggest and smallest kernel (rescaled for visibility)')
    plt.show()
    


def plt_quiver(img, angles, mask=None, lams=None, phases=None, every=20):
    if mask is None:
        mask = np.ones_like(img, dtype=bool)
    if lams is None:
        lams = 1
    
    u = np.cos(angles) * lams
    v = np.sin(angles) * lams

    u[~mask] = 0
    v[~mask] = 0

    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    idxs = np.index_exp[::every, ::every]
    
    plt.imshow(img)
    if phases is not None:
        plt.quiver(x[idxs], y[idxs], u[idxs], v[idxs], phases[idxs], angles='xy')
        plt.legend()
    else:
        plt.quiver(x[idxs], y[idxs], u[idxs], v[idxs], angles='xy')
    plt.show()


def plt_phases(phases):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(phases, cmap='hsv')
    fig.colorbar(im, cax=cax)
    plt.title('Phases')
    plt.show()
    
    
def plt_widths(lams):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(lams)
    fig.colorbar(im, cax=cax)
    plt.title('Widths')
    plt.show()
    
    
def plt_angles(angles):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(angles * 180 / np.pi)
    fig.colorbar(im, cax=cax)
    ax.set_title('Angles in degree')
    plt.show()


if __name__ == "__main__":
    img, mask = test_real()
    # angle = -0.409292
    # print(f'expected angle: {angle * 180 / np.pi:.1f}')
    # img = test_image(theta=angle)
    # plt.imshow(img)
    # img = test_cl()

    kernels, params = get_kernels(image_shape=img.shape, n_angles=36, n_widths=16)
    
    res, params_df = get_res(img, kernels, params)
    # %% filter
    # res_, params_df_ = get_res_params_for(res, params_df)

    # %% plot
    # angles = get_angles(res_, params_df_)
    # lams = get_widths(res_, params_df_)
    # phases = get_phases(res_, params_df_)

    # plt.imshow(img, cmap='coolwarm')
    # plt.title('preprocessed input image')
    # plt.show()
    
    # plt_kernels(kernels)
    # plt_kernel_on_img(img, kernels, krnl_idx=36)
    # plt_quiver(img, angles, lams=None, every=5)
    # plt_widths(lams)
    # plt_angles(angles)
    # plt_phases(phases)
    
    # plt.imshow(np.dstack((
    #     lams / lams.max(),
    #     (angles - angles.min()) / (angles - angles.min()).max(),
    #     (phases + 2) / 4
    # )))
    # plt.show()
    