from cImage import ImageProbe, ImageROI, ImageClassified
from cImage import full_initialization_standard_params, full_initalization_section
from Image_plotting import plt_cv2_image, plt_overview
from Image_convert_types import infere_mode, swap_RB
from cTransformation import ImageTransformation
import constants
from constants import sections_all, windows_all, key_dark_pixels, key_light_pixels, key_hole_pixels
from cXRF import XRF
from cMSI import MSI

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def SiFe_ratio():
    window = 'XRF'
    for section in sections_all:
        # get Si and Fe img
        X = XRF(section, window)
        X.load()
        Si_img = X.get_comp_as_img('Si')
        Fe_img = X.get_comp_as_img('Fe')

        # scale by median
        Si_img /= np.median(Si_img[Si_img > 0])
        Fe_img /= np.median(Fe_img[Fe_img > 0])

        # plt_cv2_image(Si_img, 'Si')
        # plt_cv2_image(Fe_img, 'Fe')

        brightness = (Si_img - Fe_img) / (Si_img + Fe_img)

        plt_cv2_image(brightness, 'Si/Fe idx ' + str(section))
        plt_cv2_image(X.get_comp_as_img('L'), 'grayscale ' + str(section))


def full_initi_all():
    for section in constants.sections_all:
        full_initalization_section(section=section, plts=True, verbose=True)


def all_images():
    fig_L, axs_L = plt.subplots(
        nrows=len(windows_all), ncols=len(sections_all),
        figsize=(15, 8), layout='constrained'
    )
    fig_c, axs_c = plt.subplots(
        nrows=len(windows_all), ncols=len(sections_all),
        figsize=(15, 8), layout='constrained'
    )
    fig_s, axs_s = plt.subplots(
        nrows=len(windows_all), ncols=len(sections_all),
        figsize=(15, 8), layout='constrained'
    )

    for j, section in enumerate(sections_all):
        for i, window in enumerate(windows_all):
            print(section, window)
            I = ImageClassified(section, window)
            I.load()
            image = I.sget_image_original()
            if infere_mode(image, image_type=I.current_image_type) != 'L':
                image = swap_RB(image.copy())
            axs_L[i, j].imshow(image, interpolation='none')
            axs_L[i, j].set_title(f'{section[0]}-{section[1]} cm, {window}')
            axs_L[i, j].set_axis_off()

            axs_c[i, j].imshow(I.sget_image_classification(), interpolation='none')
            axs_c[i, j].set_title(f'{section[0]}-{section[1]} cm, {window}')
            axs_c[i, j].set_axis_off()

            axs_s[i, j].imshow(I.get_image_simplified_classification(), interpolation='none')
            axs_s[i, j].set_title(f'{section[0]}-{section[1]} cm, {window}')
            axs_s[i, j].set_axis_off()

    fig_L.suptitle('grayscales')
    # fig_L.tight_layout()
    fig_L.show()

    fig_c.suptitle('classifications')
    # fig_c.tight_layout()
    fig_c.show()

    fig_s.suptitle('simplified classifications')
    # fig_s.tight_layout()
    fig_s.show()


def get_all_contrasts():
    # norm median across all images to 1
    norm_overall = -np.infty
    xs = np.empty((len(windows_all), len(sections_all)), dtype=object)
    cs = np.empty((len(windows_all), len(sections_all)), dtype=object)
    for j, section in enumerate(sections_all):
        for i, window in enumerate(windows_all):
            print(section, window)
            I = ImageClassified(section, window)
            I.load()
            c = I.params_laminae_simplified.contrast * np.sign(I.params_laminae_simplified.homogeneity)
            c[c < 0] = 0
            x = I.params_laminae_simplified.seed
            med = np.median(c)
            if med > norm_overall:
                norm_overall = med
            xs[i, j] = x
            cs[i, j] = c
    return xs, cs, norm_overall


def all_contrasts(xs, cs, norm_overall):
    fig, axs = plt.subplots(nrows=len(windows_all), ncols=len(sections_all), figsize=(15, 8), sharey=True)
    for j, section in enumerate(sections_all):
        for i, window in enumerate(windows_all):
            print(section, window)
            axs[i, j].plot(xs[i, j], cs[i, j] / norm_overall)
            axs[i, j].set_title(
                f'{section}, {window}')
            axs[i, j].grid('on')
            axs[i, j].set_xlim((0, xs[i, j].max()))
            c_m = np.median(cs[i, j])
            print(c_m)
            axs[i, j].axhline(c_m / norm_overall, color='k', alpha=.75, label=f'median: {c_m:.2f} --> {c_m / norm_overall:.1f} (scaled)')
            axs[i, j].legend()

    fig.suptitle(fr'contrast $\cdot$ sign(homogeneity) scaled to the highest median across all images ({norm_overall:.2f})')
    fig.tight_layout()
    fig.show()


def all_transformations(hold=False):
    from constants import sections_all, windows_all
    window_to = 'Alkenones'
    windows_all.remove(window_to)
    fig = plt.figure(figsize=(15, 8))
    idx = 0
    for i, window in enumerate(windows_all):
        for j, section in enumerate(sections_all):
            idx += 1
            print(section, window, idx)
            IT = ImageTransformation(section=section, window_from=window, window_to=window_to)
            IT.load()
            IT.plts = True
            ax = plt.subplot(len(windows_all), len(sections_all), idx)
            IT.plt_final(hold=True)
            del IT
            ax.set_title(f'{section[0]}-{section[1]} cm, {window}')
            ax.set_axis_off()
    fig.tight_layout()
    if not hold:
        plt.show()
    else:
        return fig


def test_image_flow():
    from skimage.registration import optical_flow_tvl1, optical_flow_ilk
    from skimage.transform import resize, downscale_local_mean
    from skimage.transform import warp
    section = (490, 495)
    window0 = 'Alkenones'
    window1 = 'FA'
    I0 = ImageROI(section, window0)
    I1 = ImageROI(section, window1)
    I0.load()
    I1.load()
    # target
    image0 = downscale_local_mean(I0.sget_image_grayscale(), 2)
    # moving
    image1 = resize(I1.sget_image_grayscale(), image0.shape)
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(image0.T)
    axs[0].set_title('target')
    axs[1].imshow(image1.T)
    axs[1].set_title('moving')

    v, u = optical_flow_tvl1(image0, image1)

    norm = np.sqrt(u ** 2 + v ** 2)

    # --- Display
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    # --- Sequence image sample

    ax0.imshow(image0, cmap='gray')
    ax0.set_title("Sequence image sample")
    ax0.set_axis_off()

    # --- Quiver plot arguments

    nvec = 20  # Number of vectors to be displayed along each image dimension
    nl, nc = image0.shape
    step = max(nl // nvec, nc // nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    ax1.imshow(norm)
    ax1.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax1.set_title("Optical flow magnitude and vector field")
    ax1.set_axis_off()
    fig.tight_layout()

    plt.show()

    nr, nc = image0.shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')

    image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]),
                       mode='edge')

    plt.imshow(image1_warp)


def test_SIFT():
    from skimage.feature import match_descriptors, plot_matches, SIFT
    from skimage.transform import resize, downscale_local_mean

    section = (490, 495)
    window0 = 'Alkenones'
    window1 = 'FA'
    I0 = ImageROI(section, window0)
    I1 = ImageROI(section, window1)
    I0.load()
    I1.load()
    # target
    img1 = downscale_local_mean(I0.sget_image_grayscale(), 16)
    # moving
    img2 = downscale_local_mean(I1.sget_image_grayscale(), 16)

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(img1)
    axs[0].set_title('target')
    axs[1].imshow(img2)
    axs[1].set_title('moving')

    plt.show()

    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 8))

    plt.gray()

    plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12, only_matches=True)
    ax[0].axis('off')
    ax[0].set_title("Target and moving image\n"
                    "(all keypoints and matches)")

    plot_matches(ax[1], img1, img2, keypoints1, keypoints2, matches12[::15],
                 only_matches=True)
    ax[1].axis('off')
    ax[1].set_title("Target and moving image\n"
                    "(subset of matches for visibility)")

    plt.tight_layout()
    plt.show()

# xs, cs, norm_overall = get_all_contrasts()
# all_contrasts(xs, cs, norm_overall)


if __name__ == '__main__':
    # ta = []
    # na = []
    # for j, section in enumerate(sections_all):
    #     for i, window in enumerate(windows_all):
    #         print(section, window)
    #         I = ImageClassified(section, window)
    #         I.load()
    #         image_s = I.get_image_simplified_classification()
    #         image_c = I.sget_image_classification()
    #         assigned = image_s > 0
    #         ta_ = assigned.ravel().mean()

    #         nonholes_assigned = assigned.ravel().sum() / (image_c > 0).ravel().sum()
    #         ta.append(ta_)
    #         na.append(nonholes_assigned)

    #         print('total assigned:', ta_)
    #         print('nonhole assigned:', nonholes_assigned)

    # print(np.mean(ta), np.std(ta))
    # print(np.mean(na), np.std(na))

    # IROI = ImageROI((490, 495), 'Alkenones')
    # IROI.plts = True
    # IROI.verbose = True

    # def ensure_odd(x):
    #     if x % 2 == 0:
    #         x += 1
    #     return x

    # # %%
    # size = ensure_odd(
    #     int(IROI.get_average_width_yearly_cycle() * 20))
    # print(size)
    # IROI.get_classification_adaptive_mean(
    #     image_gray=IROI.sget_image_grayscale(),
    #     kernel_size_adaptive=size,
    #     estimate_kernel_size_from_age_model=False
    # )

    # test_image_flow()

    # sections_all = sections_all + [(510, 515), (515, 520)]
    all_images()
    # xs, cs, norm = get_all_contrasts()
    # all_contrasts(xs, cs, norm)
    # all_transformations()
    # from constants import elements
    # windows = ['FA']
    # for window in windows:
    #     for section in sections_all:
    #         I = ImageProbe(section, window)
    #         I.load()
    #         print(I.xywh_ROI)
    # D = MSI(section, 'FA')
    # D.load()
    # D = XRF(section)
    # D.load()
    pass
