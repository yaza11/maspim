import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.filters.rank import entropy


def I_dist(x):
    # assumed distribution
    return 1 / x


def stripes(X, Y, width=2):
    """
    Stripes with given width.

    Parameters
    ----------
    X : 2D array
        x-values.
    Y : 2D array
       y-values.
    width : float, optional
        width of stripe (dark and light area). The default is 2.

    Returns
    -------
    2D array
        surface.

    """
    return (X % width - width / 2 >= 0)


def estimate_prob(I, log=True, bin_edges=None):
    if bin_edges is not None:
        counts, bin_edges = np.histogram(I, bins=bin_edges)
    elif log:
        I_nonzero = I[I != 0]
        I_sorted = np.sort(I_nonzero)
        I_sorted_log = np.log(I_sorted)
        counts, bin_edges = np.histogram(I_sorted_log, bins=255)
        # add the zeros in
        bin_edges = np.exp(bin_edges)
        bin_edges = np.insert(bin_edges, 0, 0)
        counts = np.insert(counts, 0, I.size - I_nonzero.size)
    else:
        counts, bin_edges = np.histogram(I, bins=256)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    # divide bin counts by total counts for probability
    area = np.trapz(counts, bin_centers)
    prob = counts / area
    # check if integral over prob is 1
    return prob, bin_edges


def run(plts=False, verbose=False, enhance_factor=0):
    x = np.linspace(0, 500, 500)
    y = np.linspace(0, 50, 50)

    X, Y = np.meshgrid(x, y)

    I = I_dist(np.random.normal(loc=30, scale=5, size=X.shape))  # random image
    mask_random = np.random.random(X.shape) > .98
    I[mask_random] = 0  # to resemble the SNR cutoff
    I[I < 0] = 0

    classification = ((stripes(X, Y, width=5) + 1) * 127.5).astype(int)

    # add higher intensity to light pixels
    if enhance_factor:
        mask_enhance_light = (classification == 255) & (I != 0)
        # np.random.random(X.shape)[mask_enhance_light]
        I[mask_enhance_light] *= enhance_factor

    if plts:
        plt.imshow(I, interpolation='none', vmax=np.quantile(I, .95))
        plt.show()

    I_light = I[classification == 255]
    I_dark = I[classification == 127]

    # estimate the probability distribution in the entire image
    prob, bin_edges = estimate_prob(I, log=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    # estimate the probability distribution in the light/dark pixels of the image
    # using the same bin_edges
    prob_light, _ = estimate_prob(I_light, bin_edges=bin_edges)
    prob_dark, _ = estimate_prob(I_dark, bin_edges=bin_edges)

    if plts:
        # plt.loglog(bin_centers, prob, label='pdf img')
        plt.plot(bin_centers, prob, label='pdf img')
        plt.plot(bin_centers, prob_light, label='pdf light')
        plt.plot(bin_centers, prob_dark, label='pdf dark')
        plt.legend()
        plt.xlabel('intensity')
        plt.ylabel('probability density')
        plt.show()

    S_img = scipy.stats.entropy(prob)
    S_light = scipy.stats.entropy(prob_light)
    S_dark = scipy.stats.entropy(prob_dark)

    D_light = scipy.stats.entropy(prob_light, prob)
    D_dark = scipy.stats.entropy(prob_dark, prob)

    # calculate entropy, relative entropy
    if verbose:
        print(f'shanon entropy image: {S_img:.4f}')
        print(f'shanon entropy light: {S_light:.4f}')
        print(f'shanon entropy dark: {S_dark:.4f}')
        print(
            f'Kullback-Leibler divergence for light pixels {D_light:.4f}')
        print(
            f'Kullback-Leibler divergence for dark pixels {D_dark:.4f}')

    return S_img, S_light, S_dark, D_light, D_dark


def multi_run(N_runs=1000, enhance_factor=0):
    S_imgs = np.zeros(N_runs)
    S_lights = np.zeros(N_runs)
    S_darks = np.zeros(N_runs)
    D_lights = np.zeros(N_runs)
    D_darks = np.zeros(N_runs)

    for i in range(N_runs):
        S_img, S_light, S_dark, D_light, D_dark = run(
            enhance_factor=enhance_factor)
        S_imgs[i] = S_img
        S_lights[i] = S_light
        S_darks[i] = S_dark
        D_lights[i] = D_light
        D_darks[i] = D_dark

    # show variables in boxplot
    plt.boxplot([S_imgs, S_lights, S_darks, D_lights, D_darks],
                labels=['S_imgs', 'S_lights', 'S_darks', 'D_lights', 'D_darks'])
    plt.title(f'boxplot for {N_runs=} and {enhance_factor=}')
    print(f'S_imgs mean: {S_imgs.mean():.4f}, s.d.: {S_imgs.std():.4f}')
    print(f'S_lights mean: {S_lights.mean():.4f}, s.d.: {S_lights.std():.4f}')
    print(f'S_darks mean: {S_darks.mean():.4f}, s.d.: {S_darks.std():.4f}')
    print(f'D_lights mean: {D_lights.mean():.4f}, s.d.: {D_lights.std():.4f}')
    print(f'D_darks mean: {D_darks.mean():.4f}, s.d.: {D_darks.std():.4f}')


if __name__ == '__main__':
    # run(plts=True, verbose=True, enhance_factor=1.1)
    # multi_run(1000, enhance_factor=1.1)

    p = np.array([0, 1/4, 1/4, 1/2])
    q = np.array([0, 1/2, 1/4, 1/4])
    print(scipy.stats.entropy(p, q))
