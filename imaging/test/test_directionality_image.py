import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import skimage


def arr_to_grayscale_img(arr):
    """
    Convert an array to cv2-grayscale image.

    Parameters
    ----------
    arr : np.ndarray
        Array to convert.

    Returns
    -------
    grayImage : array
        cv2 grayscale image.

    """
    uint_img = (arr / arr.max() * 255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    return grayImage


def rotate(x, y, theta, func, **kwargs):
    """
    Calculate the surface of rotated shape.

    Parameters
    ----------
    x : vector
        x-values.
    y : vector
        y-values.
    theta : float
        angle of rotation.
    func : function
        function to calculate surface.
    **kwargs : dict
        optional arguments for func.

    Returns
    -------
    2D array
        array where pixels belonging to surface are 1, otherwise 0.

    """
    X, Y = np.meshgrid(x, y)
    new_X = X * np.cos(theta) + Y * np.sin(theta)
    new_Y = -X * np.sin(theta) + Y * np.cos(theta)
    return func(new_X, new_Y, **kwargs)


def rotate_image(image, angle):
    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(
        image_center, angle * 2 * np.pi / 360, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def ellipse(X, Y, a=5, b=1):
    """
    Ellipse equation.

    Parameters
    ----------
    X : 2D array
        x-values.
    Y : 2D array
        y-values.
    a : float, optional
        major axis. The default is 5.
    b : float, optional
        minor axis. The default is 1.

    Returns
    -------
    2D array
        surface of ellipse.

    """
    return X ** 2 / a ** 2 + Y ** 2 / b ** 2 <= 1


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


def cosine_grid(X, Y, frequency=None, cycles=1):
    if frequency is None:
        T = X.max() - X.min() + np.diff(x)[0]
        f = 1 / T * cycles
        omega = f * 2 * np.pi
        print(f'cosine grid with {omega=:.3f}')
    return np.cos(X * omega)


def rectangle(X, Y, width=5, height=1):
    """
    Rectangle with given width and height.

    Parameters
    ----------
    X : 2D array
        x-values.
    Y : 2D array
       y-values.
    width : TYPE, optional
        DESCRIPTION. The default is 5.
    height : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return (X < width / 2) & (X > -width / 2) & (Y < height / 2) & (Y > -height / 2)


def stripes_in_rect(X, Y, width_rect=5, height_rect=1, width_stripes=1):
    rect = rectangle(X, Y, width_rect, height_rect)
    strip = stripes(X, Y, width_stripes)
    return rect & strip


def single_image(func, theta, **kwargs):
    x = np.linspace(-6, 6, 1000)
    y = np.linspace(-6, 6, 1000)
    arr = rotate(x, y, theta, func, **kwargs)
    image = arr_to_grayscale_img(arr)
    return image


def test_shape(func, **kwargs):
    x = np.linspace(-6, 6, 1000)
    y = np.linspace(-6, 6, 1000)
    thetas = np.arange(0, np.pi, np.pi / 40)
    err = np.empty_like(thetas)
    for idx, theta in enumerate(thetas):
        arr = rotate(x, y, theta, func, **kwargs)
        image = arr_to_grayscale_img(arr)
        theta_calc, l, w = calculate_directionality_moments(image)
        print(theta * 180 / np.pi, theta_calc * 180 /
              np.pi, (theta - theta_calc) * 180 / np.pi, l / w)
        err[idx] = theta - theta_calc

    plt.plot(thetas * 180 / np.pi, err * 180 / np.pi, '-o')
    plt.xlabel('orientation in deg')
    plt.ylabel('deviation of calculated angle')


def test_ratio():
    widths_stripes = np.linspace(.1, 5, 30)
    ratios = np.empty_like(widths_stripes)

    for idx, width_stripes in enumerate(widths_stripes):
        image = single_image(stripes_in_rect, np.pi/6,
                             width_stripes=width_stripes, height_rect=5)
        contours = get_contours(image, filter_by_size=0)
        angle, mean, ratio = calculate_directionality_PCA(contours)
        plt.figure()
        plt.imshow(image)
        plt.title(str(ratio))
        plt.show()
        ratios[idx] = ratio

    plt.figure()
    plt.plot(widths_stripes, ratios)
    plt.show()


def calc_fft2d(img, angle_in=0):
    plt.figure(figsize=(15, 10))

    # calculate transect
    d_image = calc_length_diag(x, y, angle=angle_in)
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    # center around 0
    x_c = x.min() + x_range / 2
    y_c = y.min() + y_range / 2
    xt1 = np.cos(angle_in) * d_image / 2 + x_c
    yt1 = np.sin(angle_in) * d_image / 2 + y_c
    xt2 = np.cos(angle_in + np.pi) * d_image / 2 + x_c
    yt2 = np.sin(angle_in + np.pi) * d_image / 2 + y_c
    # plt original and transect
    plt.subplot(221)
    plt.imshow(img, interpolation='None', origin='lower',
               extent=(x.min(), x.max(), y.min(), y.max()))
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('original')
    # plot the line of direction
    plt.plot([xt1, xt2], [yt1, yt2], '-ro', label='direction in', alpha=.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # find corresponding pixel coordinates
    idx2 = [np.argmin(np.abs(y - yt1)), np.argmin(np.abs(x - xt1))]
    idx1 = [np.argmin(np.abs(y - yt2)), np.argmin(np.abs(x - xt2))]
    transect = skimage.measure.profile_line(
        img, idx1, idx2)
    plt.subplot(222)

    plt.plot(np.linspace(0, 1, len(transect)), transect, '-')
    plt.xlabel('t')
    plt.title('Data along transect \n' +
              fr'from $p_1$=({x[idx1[1]]:.2f}, {y[idx1[0]]:.2f}) to ' +
              fr'$p_2$=({x[idx2[1]]:.2f}, {y[idx2[0]]:.2f})')

    # calculate fft and frequencies
    res = fft.fftshift(fft.fft2(img - np.mean(img)))
    f_x = fft.fftshift(fft.fftfreq(img.shape[1], d=dx))
    f_y = fft.fftshift(fft.fftfreq(img.shape[0], d=dy))
    f_X, f_Y = np.meshgrid(f_x, f_y)
    f_R = np.sqrt(f_X ** 2 + f_Y ** 2)
    f_Theta = np.arctan2(f_Y, f_X)

    # get maximum frequency
    idx_max = np.unravel_index(np.argmax(np.abs(res)), res.shape)
    f_x_max = f_x[idx_max[1]]
    f_y_max = f_y[idx_max[0]]
    f_max = np.sqrt(f_x_max ** 2 + f_y_max ** 2)
    omega_max = 2 * np.pi * f_max
    T_max = 1 / f_max
    angle_max = np.arctan2(f_y_max, f_x_max)
    d_max = calc_length_diag(f_x, f_y, angle=angle_max)

    # plot fft and transect
    plt.subplot(223)
    plt.imshow(np.abs(res), interpolation='None', origin='lower',
               extent=(f_x.min(), f_x.max(), f_y.min(), f_y.max()))
    plt.xlabel(r'$f_x$')
    plt.ylabel(r'$f_y$')
    plt.title('fft result')
    # plot estimated direction
    f_xt1 = np.cos(angle_max) * d_max / 2
    f_yt1 = np.sin(angle_max) * d_max / 2
    f_xt2 = np.cos(angle_max + np.pi) * d_max / 2
    f_yt2 = np.sin(angle_max + np.pi) * d_max / 2
    plt.plot([f_xt1, f_xt2], [f_yt1, f_yt2], '-ro',
             label='direction max', alpha=.5)
    plt.xlim((f_x.min(), f_x.max()))
    plt.ylim((f_y.min(), f_y.max()))
    plt.legend()

    d_predict = calc_length_diag(np.append(x, x.max() + dx),
                                 np.append(y, y.max() + dy), angle_max)
    n_predict = np.abs(d_predict * f_max)

    # find pixel coordinates
    f_idx2 = [np.argmin(np.abs(f_y - f_yt1)), np.argmin(np.abs(f_x - f_xt1))]
    f_idx1 = [np.argmin(np.abs(f_y - f_yt2)), np.argmin(np.abs(f_x - f_xt2))]
    res_transect = skimage.measure.profile_line(
        np.abs(res), f_idx1, f_idx2)

    plt.subplot(224)

    plt.stem(np.linspace(0, 1, len(res_transect)), res_transect, markerfmt='')
    plt.hlines(N / 2, xmin=0, xmax=1,
               linestyle='--', color='black', label='N / 2')
    # plt.vlines([cycles / N_x], ymin=0, ymax=N / 2, linestyle=':',
    #            color='orange', label=rf'$\omega_0=\pm${cycles / N_x:.3f}')
    # plt.vlines([-cycles / N_x], ymin=0, ymax=N / 2, linestyle=':',
    #            color='orange')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xlabel('t')
    plt.title('Frequencies along transect \n' +
              fr'from $f_1$=({f_x[f_idx1[1]]:.2f}, {f_y[f_idx1[0]]:.2f}) to ' +
              fr'$f_2$=({f_x[f_idx2[1]]:.2f}, {f_y[f_idx2[0]]:.2f})')
    plt.show()
    print(f'found dominant cycle length with {T_max:.1f}.\n' +
          f'estm. {f_max=:.3f} (should be {nu:.3f}), \n' +
          f'estm. {omega_max=:.3f}) (should be {omega:.3f}), \n' +
          f'predicting {n_predict:.1f} cycles (should be {cycles}), \n' +
          f'with an angle of {angle_max % np.pi:.2f} \
rad = {angle_max * 180 / np.pi % 180:.0f} \
deg (should be {angle:.2f} rad = {angle * 180 / np.pi:.0f} deg)')

    return res


def calc_length_diag(x, y, angle):
    width = x.max() - x.min()
    height = y.max() - y.min()
    # critical angle for calculation is the diagonal
    angle_crit = np.arctan2(height, width)
    if (np.abs(angle) < angle_crit) or (np.abs(angle) > np.pi / 2 - angle_crit):
        return width / np.cos(angle)
    return height / np.sin(angle)


test_classification = False
test_cosine = False

if __name__ == '__main__':
    test_classification = False
    test_cosine = False


if test_classification:
    # img = np.load('OOP/c.npy')
    # img[img == 0] = round(255 / 2 + 127 / 2)
    img = cv2.imread(
        r"C:\Users\yanni\OneDrive\Master_Thesis\regions_of_interest\region_msi.png", cv2.IMREAD_GRAYSCALE)

    N_y, N_x = img.shape
    N = N_x * N_y
    angle = 0
    # x = np.linspace(0, 2 * np.pi, N_x, endpoint=False)
    # y = np.linspace(0, 2 * np.pi, N_y, endpoint=False)
    x = np.arange(N_x)
    y = np.arange(N_y)
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    # laminated in y-direction
    cycles = 64
    nu = cycles / calc_length_diag(
        np.append(x, x.max() + dx), np.append(y, y.max() + dy), angle)
    omega = 2 * np.pi * nu

    mask = 1

    res = calc_fft2d(img * mask, angle)


if test_cosine:
    n = 2000
    N_x = n * 5
    N_y = n
    N = N_x * N_y
    # x = np.linspace(0, 2 * np.pi, N_x, endpoint=False)
    # y = np.linspace(0, 2 * np.pi, N_y, endpoint=False)
    x = np.arange(N_x)
    y = np.arange(N_y)
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    # laminated in y-direction
    cycles = 64

    angle = 5 * np.pi / 180
    nu = cycles / calc_length_diag(
        np.append(x, x.max() + dx), np.append(y, y.max() + dy), angle)
    omega = 2 * np.pi * nu

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)
    Theta_T = Theta - angle
    XT = R * np.cos(Theta_T)
    YT = R * np.sin(Theta_T)
    # img = (stripes(X, Y, width=width)).astype(float)
    # img = rotate(x, y, angle, stripes, width=width)
    # img = cosine_grid(X)
    # img = rotate(x, y, angle, cosine_grid, cycles=cycles)
    img = np.cos(XT * omega)
    x_c = (x.min() + x.max()) / 2
    y_c = (y.min() + y.max()) / 2
    mask = ((X - x_c) ** 2 + (Y - y_c) **
            2) <= np.min([x_c - x.min(), y_c - y.max()]) ** 2
    mask = 1

    res = calc_fft2d(img * mask, angle)
