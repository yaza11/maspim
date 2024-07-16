"""
This module is still under development. The idea here is to find parameters
of polynomials in the x- and y-direction to find offsets between two images.
"""
import skimage
import numpy as np
import matplotlib.pyplot as plt
import logging

from typing import Callable
from scipy import optimize

from msi_workflow.imaging.register.helpers import interpolate_shifts, get_transect_indices


logger = logging.getLogger(__name__)


def apply_stretching(arr: np.ndarray, *coeffs: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the stretching fit to a vector or array.

    Parameters
    ----------
    arr : np.ndarray
        function values to be stretched
    coeffs: list
        Coefficients of polynomial describing the stretching starting with the highest degree

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The array with applied stretching and the applied (point-wise) distortions.
    """
    arr_dim: int = arr.ndim
    assert arr_dim in (1, 2), f'provided array must be 1d or 2d but is {arr_dim}.'

    # dimension in x direction
    N_x: int = arr.shape[-1]
    # arr holding x pixel coords
    x: np.ndarray = np.ones_like(arr) * np.arange(N_x, dtype=float)
    # center
    x_middle: float = (x.max() - x.min()) / 2 + x.min()
    x -= x_middle
    # scale (--> -1/2 to 1/2)
    x /= N_x
    # get func
    func: Callable = stretching_func(*coeffs)
    # get func vals
    y: np.ndarray = func(x)
    # stretch
    x_stretched: np.ndarray = flatten_func(x, y)
    # squeeze
    x_rescaled: np.ndarray = rescale_x(x, x_stretched)
    # interpolate values on new grid
    if arr.ndim == 1:
        arr_stretched: np.ndarray = np.interp(x, x_rescaled, arr)
    elif arr.ndim == 2:
        N_y: int = arr.shape[0]
        arr_stretched: np.ndarray = np.zeros_like(arr)
        for idx_y in range(N_y):
            arr_stretched[idx_y, :] = np.interp(
                x[idx_y, :], x_rescaled[idx_y, :], arr[idx_y, :])
    else:
        raise ValueError(
            f"provided array must have at least one entry, you passed {arr}"
        )
    return arr_stretched, x_rescaled * N_x + x_middle


def apply_vshifts(image, u, target_shape) -> np.ndarray:
    nr, nc, *_ = target_shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')

    source: np.ndarray = skimage.transform.warp(
        image, np.array([row_coords, col_coords + u]),
        mode='edge'
    )
    source /= source.max()
    
    return source


def get_target(shape: tuple[int, int]) -> np.ndarray:
    target = np.zeros(shape)

    x = np.linspace(-.5, .5, target.shape[1])
    y = np.arange(target.shape[0])
    X, Y = np.meshgrid(x, y)

    target = np.cos(X * 8 * np.pi) + np.cos(X * 8 * 5 * np.pi + np.random.random() * 2) + np.cos(X * 4 * np.pi + np.random.random() * 2) > 0
    # target = np.cos(X * 8 * np.pi) + np.cos(X * 8 * 5 * np.pi + np.random.random() * 2) + np.cos(X * 4 * np.pi + np.random.random() * 2)

    return target.astype(int)
    # return target


def get_source(target: np.ndarray, *, n_transects, params_in, degree) -> np.ndarray:
    # source = apply_vshifts(target, shift_matrix)
    source = warp_with_params(image=target, params=params_in, target_shape=target.shape, n_transects=n_transects, degree=degree)

    return source


def metric(target, source) -> float:
    return np.mean(np.abs(source - target) ** 2)


def vmetric(target, source) -> float:
    return (source - target).ravel()


def fit_shifts_local(
        *,
        target: np.ndarray,
        source: np.ndarray,
        n_transects: int,
        degree: int,
        **kwargs
):
    """
    Find the shift function to fit the source to the target.
    
    This does the fit for each transect separately.
    """
    def warp(params) -> np.ndarray:
        # calculate the shift vectors from the provided parameters
        shifts: np.ndarray = np.polyval(params, x=x)

        image_warped: np.ndarray = skimage.transform.warp(
            source_section, np.array([row_coords, col_coords + shifts]),
            mode='edge'
        )
        # return image_warped[pad:-pad, pad:-pad]
        return image_warped

    def loss_func(params: np.ndarray) -> float:
        return metric(target_section, warp(params))
    
    def vloss_func(params):
        return vmetric(target_section, warp(params))

    n_params_poly = degree + 1

    # ensure images are normalized
    target: np.ndarray[float] = target.copy().astype(float) / target.max()
    source: np.ndarray[float] = source.copy().astype(float) / source.max()

    params0 = np.zeros(n_params_poly)
    
    coeff_matrix: np.ndarray[float] = np.zeros((n_transects, n_params_poly))
    
    # pad = round(source.shape[1] / 2)
    # x_pad = .25
    
    for i_transect in range(n_transects):
        target_section = target[get_transect_indices(i_transect, target.shape, n_transects)]
        source_section = source[get_transect_indices(i_transect, source.shape, n_transects)]
        
        # source_section = np.pad(source_section, ((pad, pad), (pad, pad)))
        
        nr, nc, *_ = source_section.shape
        # x = np.linspace(-.5 - x_pad, .5 + x_pad, nc)
        x = np.linspace(-.5, .5, nc)
        row_coords, col_coords = np.meshgrid(
            np.arange(nr), np.arange(nc),
            indexing='ij'
        )
        
        res = optimize.basinhopping(
            loss_func,
            x0=params0,
            # niter=500,
            minimizer_kwargs={'method': 'CG'},
            **kwargs
        )
        
        logger.info(res.message)
        
        coeff_matrix[i_transect, :] = res.x
        
    return coeff_matrix


def warp_with_params(image, *, params, target_shape, n_transects, degree) -> np.ndarray:
    if params.ndim == 1:
        params = params.reshape((n_transects, degree + 1))
    x = np.linspace(-.5, .5, image.shape[1])
    shifts: list[np.ndarray] = [
            np.polyval(params[i, :], x=x) for i in range(n_transects)
    ]
    shifts: np.ndarray = interpolate_shifts(shifts, target_shape, n_transects)

    image_warped = apply_vshifts(image, shifts, target_shape)
    return image_warped


def fit_shifts_global(
        *, target, source, n_transects, degree, **kwargs
) -> optimize.OptimizeResult:
    def warp(params) -> np.ndarray:
        params = params.reshape((n_transects, -1))
        shifts: list[np.ndarray] = [
            np.polyval(params[i, :], x=x) for i in range(n_transects)
        ]
        shifts: np.ndarray = interpolate_shifts(shifts, target.shape, n_transects)

        image_warped: np.ndarray = skimage.transform.warp(
            source, np.array([row_coords, col_coords + shifts]),
            mode='edge'
        )
        return image_warped

    def loss_func(params):
        return metric(target, warp(params))
    
    def vloss_func(params):
        return vmetric(target, warp(params))

    n_params_poly = degree + 1

    # ensure images are normalized
    target = target.copy().astype(float) / target.max()
    source = source.copy().astype(float) / source.max()

    nr, nc, *_ = target.shape
    x = np.linspace(-5., .5, nc)
    row_coords, col_coords = np.meshgrid(
        np.arange(nr), np.arange(nc),
        indexing='ij'
    )
    
    params0 = np.zeros((n_transects, n_params_poly)).ravel()
    bounds = [(-20, 20)] * len(params0)

    logger.info(
        f"starting optimization using direct, initial loss: {loss_func(params0)}"
    )
    res: optimize.OptimizeResult = optimize.direct(
        loss_func,
        bounds=bounds,
        maxfun=int(1e5),
        maxiter=int(1e5),
        vol_tol=0,
        f_min=0,
        locally_biased=True,
        **kwargs
    )
    logger.info(f"done optimizing, final loss: {res.fun}")
        
    return res


def fit_shifts_1D(target: np.ndarray, source: np.ndarray, degree: int, **kwargs):
    def loss_func(params):
        return metric(target, apply_stretching(source, params))

    n_params_poly: int = degree + 1
    params0 = np.zeros(n_params_poly)

    logger.info(
        f"starting optimization using direct, initial loss: {loss_func(params0)}"
    )
    res: optimize.OptimizeResult = optimize.basinhopping(
        loss_func,
        x0=params0,
        # niter=500,
        minimizer_kwargs={'method': 'CG'},
        **kwargs
    )
    logger.info(f"done optimizing, final loss: {res.fun}")

    return res


def test_inverse(target, source, params_in):
    params_back = -params_in

    target_inv_inv = warp_with_params(source, params_back)
    
    plt.imshow(target)
    plt.show()
    plt.imshow(source)
    plt.show()
    plt.imshow(target_inv_inv)
    plt.show()


def plt_res(res, *, target, source, n_transects, degree, params_in):
    if type(res) is optimize.OptimizeResult:
        print(res.message)
        params = res.x.reshape(n_transects, degree + 1)
    else:
        params = res
        
    x = np.linspace(-.5, .5, target.shape[1])

    source_warped = warp_with_params(
        source,
        params=params.ravel(),
        target_shape=target.shape,
        n_transects=n_transects,
        degree=degree
    )

    plt.imshow(source_warped)
    ax2 = plt.gca().twinx()
    for i, (row, row_in) in enumerate(zip(params, -params_in)):
        ax2.plot(np.polyval(row, x) + i * 10, 'r')
        ax2.plot(-np.polyval(row_in, x) + i * 10, 'b')
    plt.legend(['estimated', 'true'])
    plt.show()



def test():
    # np.random.seed(0)
    n_transects: int = 3
    degree: int = 4
    degree_fit = 4
    
    # target = skimage.data.cat().mean(axis=-1)
    target: np.ndarray = get_target((60, 80))
    
    f: float = target.shape[1] / 15
    # f = 1
    params_in: np.ndarray = np.array([
        (np.random.random(degree + 1) - .5) * f for _ in range(n_transects)
    ])
    
    # params_in: np.ndarray = np.array([
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0]
    # ])
    
    source: np.ndarray = get_source(target, n_transects=n_transects, params_in=params_in, degree=degree)
    
    # test_inverse(target, source, params_in)
    
    plt.imshow(target)
    plt.show()
    plt.imshow(source)
    plt.show()
    
    res = fit_shifts_global(target=target, source=source, n_transects=n_transects, degree=degree_fit)
    # res = fit_shifts_local(target=target, source=source, n_transects=n_transects, degree=degree_fit)
    
    
    plt_res(res, target=target, source=source, n_transects=n_transects, degree=degree_fit, params_in=params_in)
    
    print(-params_in)
    # print(res)
    params_fit = res.x.reshape(n_transects, degree_fit + 1)
    print(params_fit)



if __name__ == '__main__':
    test()


def stretching_func(*coeffs: float) -> np.poly1d:
    """
    Function defining deformations in an image along the x-direction

    Parameters
    ----------
    coeffs: float
        Coefficients defining the polynomial.

    Returns
    -------
    A callable that yields y values for given x values according to the \
    provided coefficients.
    """
    return np.poly1d(c_or_r=coeffs)


def flatten_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Flatten a function by mapping arclength to x.

    Parameters
    ----------
    x: np.ndarray
        x-values (vector, array-like) of a function
    y: np.ndarray
        y-values (vector, array-like) of a function

    Returns
    -------
    x_stretched: np.ndarray
        x-values if pieces of curve would be stacked behind each other.
    """
    dx: np.ndarray = np.diff(x, axis=-1)
    dy: np.ndarray = np.diff(y, axis=-1)
    x_stretched: np.ndarray = np.cumsum(np.sqrt(dx ** 2 + dy ** 2), axis=-1)
    # zeropad to the left by 1
    if len(x.shape) == 1:
        pads: tuple = (1, 0)
    else:
        pads: tuple = ((0, 0), (1, 0))
    return np.pad(x_stretched, pads, 'constant')


def rescale_x(x: np.ndarray, x_stretched: np.ndarray) -> np.ndarray:
    """
    Scale the stretched out x to be in the bounds of original x.

    After the flattening the x-values will likely exceed the original bounds.
    This function scales x-values equally such that the range of the stretched
    x-values is the same as x.

    Parameters
    ----------
    x: np.ndarray
        Original x-values.
    x_stretched: np.ndarray
        Stretched x-values.

    Returns
    -------
    x_rescaled: np.ndarray
        Rescaled stretched x-values, such that x.min() and x.max() are the
         same as x_rescaled.min() and x_rescaled.max()
    """
    range_x: float = x.max() - x.min()
    range_x_stretched: float = x_stretched.max() - x_stretched.min()
    x_rescaled: np.ndarray = (
            (x_stretched - x_stretched.min()) / range_x_stretched * range_x + x.min()
    )
    return x_rescaled
