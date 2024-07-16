"""Module associated with finding the sample region."""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

from src.imaging.util.coordinate_transformations import kartesian_to_polar, polar_to_kartesian


def calculate_directionality_PCA(contours):
    """
    Estimate the directionality of multiple contours by comparing their angles.

    Parameters
    ----------
    contours : array with dimensions (len(contour), 1, 2)
        array containing the points defining the contour.

    Returns
    -------
    weighted_angles_mean: float
        The general direction of the contours.
    abs_cosine_similarity: float
        The similarity of the angles calculated by taking the mean of the
        absolute value for each pair of contour.
    mean_mean: tuple(int, int)
        The mean of contour means.

    """
    # https://docs.opencv.org/4.x/d1/dee/tutorial_introduction_to_pca.html
    N_contours = len(contours)
    angles = np.empty(N_contours, dtype=float)
    means = np.empty(N_contours, dtype=object)
    explained_variances = np.empty(N_contours, dtype=float)
    eigvec1 = np.empty((N_contours, 2), dtype=float)
    # eigvec2 = np.empty((len(contours), 2), dtype=float)
    eigval1 = np.empty(N_contours, dtype=float)
    # eigval2 = np.empty(len(contours), dtype=float)
    for idx, contour in enumerate(contours):
        sz = len(contour)
        contour_reduced_dim = np.empty((sz, 2), dtype=np.float64)
        contour_reduced_dim[:, 0] = contour[:, 0, 0]
        contour_reduced_dim[:, 1] = contour[:, 0, 1]
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(
            contour_reduced_dim, mean=mean)
        # first eigenvector (first row) corresponds to transformed axis with
        # highest variance
        angles[idx] = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
        # store eigenvecs
        eigvec1[idx, :] = eigenvectors[0, :]
        # eigvec2[idx, :] = eigenvectors[1, :]
        eigval1[idx] = eigenvalues[0]
        # eigval2[idx] = eigenvalues[1]
        # scale such that ratio si between 0 and 1
        # (since lowest value would otherwise be .5)
        if np.sum(eigenvalues) != 0:
            explained_variances[idx] = eigenvalues[0] / np.sum(eigenvalues)
        else:
            print(contour_reduced_dim.shape, eigenvalues)
            explained_variances[idx] = 0
        means[idx] = mean

    # calculate scalar product of all eigenvectors
    abs_cosine_similarity = 0
    if N_contours == 1:
        return angles[0], 1 - explained_variances[0] ** 2, means[0]
    for idx, ev1 in enumerate(eigvec1):
        for ev2 in eigvec1[(idx + 1):]:
            abs_cosine_similarity += np.abs(ev1 @ ev2.T)
    # scale by number of elements
    abs_cosine_similarity /= N_contours * (N_contours + 1) / 2
    # take inverse
    abs_cosine_similarity = 1 - abs_cosine_similarity
    mean_mean = np.mean(means, axis=0)
    weighted_angles_mean = None
    return weighted_angles_mean, abs_cosine_similarity, mean_mean


def calculate_directionality_moments(image):
    """
    Calculate the angle, minor and major axes for an image from moments.

    Parameters
    ----------
    image : array
        The image for which to determine the angle.

    Returns
    -------
    theta : float
        Angle.
    major_axis : float
        The main direction.
    minor_axis : float
        perpendicular to major axis.

    """
    # for binary image
    # following
    # http://raphael.candelier.fr/?blog=Image%20Moments
    # and
    # https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html
    # image_binary = get_binary(image)
    # convert to type matching that of a contour
    # pseudo_contour = np.expand_dims(np.argwhere(test_img != 0)[
    #                                 :, :2], 1).astype(np.int32)
    M = cv2.moments(image[:, :, 0])
    M00 = M['m00']
    x_bar = M['m10'] / M00
    y_bar = M['m01'] / M00
    mu20_prime = M['m20'] / M00 - x_bar ** 2
    mu11_prime = M['m11'] / M00 - x_bar * y_bar
    mu02_prime = M['m02'] / M00 - y_bar ** 2
    # if mu20_prime - mu02_prime == 0:
    #     print('mu20 = mu02')
    #     theta = np.pi / 4
    # else:
    theta = .5 * np.arctan2(2 * mu11_prime,
                            (mu20_prime - mu02_prime)) % (2 * np.pi)
    inner_sqrt = np.sqrt(4 * mu11_prime ** 2 + (mu20_prime - mu02_prime) ** 2)
    major_axis = np.sqrt(8 * (mu20_prime + mu02_prime + inner_sqrt))
    minor_axis = np.sqrt(8 * (mu20_prime + mu02_prime - inner_sqrt))
    return theta, major_axis, minor_axis


def contour_to_xy(contour):
    """
    Get x and y values for pixels in contour.

    Parameters
    ----------
    contour : array of shape (len, 1, 2)
        The contour.

    Returns
    -------
    v_x : array of ints
        The x-indexes of the contour.
    v_y : array of ints
        The y-indexes of the contour.

    """
    v_x = contour[:, 0, 0]
    v_y = contour[:, 0, 1]
    return v_x, v_y


def star_domain_contour(contour, center_point, plts=False, smoothing_size=5):
    """
    Turn contour into star domain by taking the largest radius for same angles.

    Parameters
    ----------
    contour : array
        The contour to turn into star domain.
    center_point : tuple(float, float)
        The (x, y) point from which to calculate the radii and angles of the
        contour.
    plts : bool, optional
        Plot the resulting contour if True. The default is False.
    smoothing_size : int, optional
        Take the largest of smoothing_size values for each radius to remove
        binning artefacts. The default is 5.

    Returns
    -------
    contour: array[int, int, int]
        The transformed contour.

    """
    v_x, v_y = contour_to_xy(contour)
    v_x -= center_point[0]
    v_y -= center_point[1]
    r, phi = kartesian_to_polar(v_x, v_y)

    sorting = np.argsort(phi)
    phi_sorted = phi[sorting]
    r_sorted = r[sorting]

    # contour is not guarantied to by convex, thus there may be multiple radii for
    # same angle
    # solution: bin the radii by angles and the the largest value
    # bin size is 2 pi / number of points in contour
    N_phis = len(phi_sorted)
    dphi = 2 * np.pi / N_phis
    # round the phis to nearest multiple of dphi
    phi_rounded = (phi_sorted / dphi + .5).astype(int) * dphi
    # for radii corresponding to same angles take the largest one
    phi_unique, idxs_unique, counts = np.unique(
        phi_rounded, return_index=True, return_counts=True)
    r_largest = np.zeros_like(phi_unique)
    for idx, (phi, idx_first, count) in enumerate(zip(phi_unique, idxs_unique, counts)):
        rs = r_sorted[idx_first:idx_first + count]
        r_largest[idx] = np.max(rs)

    # run maximum filter
    r_max = scipy.ndimage.maximum_filter(
        r_largest, size=smoothing_size, mode='wrap')
    # resample to regular intervals
    phi_regular = np.linspace(0, 2 * np.pi, N_phis)
    r_regular = np.interp(
        phi_regular, phi_unique, r_max, period=2 * np.pi)

    v_x_prime, v_y_prime = polar_to_kartesian(r_regular, phi_regular)

    if plts:
        plt.figure()
        plt.plot(v_x, v_y)
        plt.plot(v_x_prime, v_y_prime)
        plt.plot((0, 0))
        plt.axis('equal')
        plt.show()

    v_x_prime += center_point[0]
    v_y_prime += center_point[1]

    contour_out = np.zeros((N_phis, 1, 2), dtype=float)
    contour_out[:, 0, 0] = v_x_prime
    contour_out[:, 0, 1] = v_y_prime
    return contour_out.astype(int)
