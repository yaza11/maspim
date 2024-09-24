""""""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio

from io import BytesIO
from typing import Iterable
from scipy.optimize import minimize
from tqdm import tqdm


def distorted_rect(
        width: int, height: float, coeffs: Iterable[float], plts: bool = False
) -> np.ndarray[bool]:
    """
    Mask for distorted rectangle in square region.

    coeffs are expected to be in relative coordinates (coordinates that match
    those of x in [-1, 1]).

    Parameters
    ----------
    width : int
        The lateral extension of the rectangle. Function returns transposed image.
    height : float
        The thickness of the layer in pixels.
    coeffs : Iterable[float]
        The coefficients of the polynomial in relative coordinates.
    plts : bool, optional
        Will plot region, if True. The default is False.

    Returns
    -------
    region : np.ndarray[bool]
        The region of the distorted rectangle where pixels inside region are True.

    """
    x: np.ndarray[float] = np.linspace(-1, 1, width, dtype=float)

    Y: np.ndarray[float] = x[:, None] * np.ones(width)[None, :]
    # polynomial values
    y: np.ndarray[float] = np.poly1d(coeffs)(x)
    # center curve around center of mass
    y -= y.mean()
    #
    y_shift: float = height / width / 2
    y_upper: np.ndarray[float] = y + y_shift
    y_lower: np.ndarray[float] = y - y_shift

    region: np.ndarray[bool] = (Y > y_lower) & (Y <= y_upper)
    region: np.ndarray[bool] = region.T

    if plts:
        plt.imshow(region)
        plt.plot(width / 2, width / 2, 'ro')
        plt.title(f'coeffs={[np.round(coeff, 3) for coeff in coeffs]}, {height=:.1f}')
        plt.show()

    return region


def animate_optimization(x0s: list, path_file: str, image_classification, seed, is_dark):
    width = image_classification.shape[0]

    # List to store images in memory
    images = []
    for idx, x0 in enumerate(x0s):
        *coeffs, height = x0

        coeffs4: list[float] = [0.] * 4
        n_missing: int = 4 - len(coeffs)
        for i, coeff in enumerate(coeffs):
            coeffs4[i + n_missing] = coeff
        a, b, c, d = coeffs4

        region: np.ndarray[bool] = distorted_rect(width, height, (a, b, c, d))
        image_region: np.ndarray = image_classification[:, seed:seed + width]
        plt.figure(figsize=(6, 6))
        plt.imshow(image_region + (1 - is_dark * 2) * 255 * region)
        plt.plot(width / 2, width / 2, 'ro')
        plt.plot(width / 2 + d, width / 2, 'bo')
        plt.title(f'{a=:.3f}, {b=:.3f}, {c=:.3f} {d=:.3f} {height=:.1f}')

        # Save figure to an in-memory buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer to the beginning
        images.append(imageio.imread(buf))  # Read the image from buffer

        # Close the plot to avoid memory issues
        plt.close()

    # Create a GIF from in-memory images
    imageio.mimsave(path_file, images, duration=0.5, loop=0)


def find_layer(
        image_classification: np.ndarray[int | float],
        seed: int | float,
        height0: float,
        color: str,
        fixate_height: bool = False,
        key_light: int = 255,
        key_dark: int = 127,
        max_slope: float = .1,
        x0: np.ndarray[float] | None = None,
        degree: int | None = None,
        bounds: tuple[tuple[float, float], ...] | None = None,
        return_steps: bool = False,
        plts: bool = False,
        plts_region: bool = False,
        **kwargs
) -> list[float | bool] | tuple[list, list]:
    """
    Find parameters for distorted rectangle around seed in classified image.

    Parameters
    ----------
    image_classification : np.ndarray[float]
        The classified image (light, dark, hole).
    seed : int | float
        Start depth of centerpoint of rectangle.
    height0 : float
        Initial guess for width of layer in pixels.
    fixate_height: bool, optional
            If this is set to True, will not optimize the heights but instead
            use height0 values.
    color : str
        "light" or "dark", the color of the layer.
    key_light : int, optional
        Value assigned to light pixels in classification. The default is 255.
    key_dark : int, optional
        Value assigned to dark pixels in classification. The default is 127.
    max_slope : float, optional
        Maximum allowed slope of layers (adding abs value of coefficients). 
        The default is .1.
    x0 : tuple, optional
        Initial guess for parameters of rectangle. The default is None (no distortion).
    bounds : tuple, optional
        Bounds for parameters of rectangle. The default is None (will be infered from max slope).
    plts : bool, optional
        Whether to plot results. The default is False.
    plts_region : bool, optional
        Whether to plot distorted rects during optimization. The default is False.
        Setting this to true will take much longer.

    Returns
    -------
    params : list
        Parameters of best distorted rectangle for seed.

    """
    def metric(x0_: np.ndarray) -> float:
        """Evaluation function for parameters of rectangle."""
        *coeffs_, height_ = x0_
        x0s.append(x0_)

        if height_ < 0:
            return -1

        layer_region: np.ndarray[bool] = distorted_rect(
            width, height_, coeffs_, plts=plts_region)
        image_region_: np.ndarray = image_classification[:, seed:seed + width]

        vision_layer: np.ndarray = image_region_ * layer_region
        num_light_layer: int = np.sum(vision_layer == key_light)
        num_dark_layer: int = np.sum(vision_layer == key_dark)
        num_layer: int = num_light_layer + num_dark_layer + 1
        advantage_layer: float = (
                (num_light_layer - num_dark_layer * ratio_total) / num_layer
        )
        density: float = num_light_layer / num_layer
        density: float = is_dark - density
        score_layer: float = advantage_layer * density * height_ / width
        return score_layer

    def metric_constant_height(x0_: np.ndarray) -> float:
        x0_new = np.append(x0_, [max(height0, 1)])
        return metric(x0_new)

    assert (degree is not None) or (x0 is not None)

    x0s = []

    if degree is None:
        # subtract 1 because need n+1 parameters to define polynomial of degree
        # n
        # and another one because one of the parameters in x0 is the height
        degree: int = len(x0) - 2
    elif x0 is None:
        x0: list[float] = [0.] * (degree + 1)
        if not fixate_height:
            x0.append(height0)
    else:
        raise ValueError('internal error')

    # in this case there is nothing left to optimize
    if (degree == 0) and fixate_height:
        return [0, 0, 0, 0, height0, True]

    if (key_light not in image_classification) or (key_dark not in image_classification):
        raise ValueError(f'Specify keys (values) for light and dark '
                         f'classification. Either {key_light=} or {key_dark=} '
                         f'is not in image.')

    if color.lower() not in ('light', 'dark'):
        raise KeyError(
            f'color has to be either "light" or "dark", not {color}.')

    is_dark: float = float(color == 'dark')

    width: int = image_classification.shape[0]

    # independent of layer shape
    num_light_total: int = np.sum(image_classification == key_light)
    num_dark_total: int = np.sum(image_classification == key_dark)
    ratio_total: float = num_light_total / num_dark_total

    if bounds is None:
        bounds: tuple[tuple[float, float], ...] = tuple(
            (-max_slope / deg / 3, max_slope / deg / 3) for deg in range(degree, 0, -1)
        ) + (
            (-height0 / 2 / width, height0 / 2 / width),  # d
        )
        if not fixate_height:
            bounds += ((height0 / 2, height0 * 2),)  # thickness layer

    score0: float = metric(x0)

    params = minimize(
        metric_constant_height if fixate_height else metric,  # function to minimize
        x0=x0,  # start values
        method=kwargs.pop('method', 'Nelder-Mead'),
        bounds=bounds
    )

    # get values
    if fixate_height:
        coeffs = params.x
        height = height0
    else:
        *coeffs, height = params.x
    coeffs4: list[float] = [0.] * 4
    n_missing: int = 4 - len(coeffs)
    for i, coeff in enumerate(coeffs):
        coeffs4[i + n_missing] = coeff
    a, b, c, d = coeffs4

    if plts:
        region: np.ndarray[bool] = distorted_rect(width, height, (a, b, c, d))
        image_region: np.ndarray = image_classification[:, seed:seed + width]
        plt.imshow(image_region + (1 - is_dark * 2) * 255 * region)
        plt.plot(width / 2, width / 2, 'ro')
        plt.plot(width / 2 + d, width / 2, 'bo')
        plt.title(f'{a=:.3f}, {b=:.3f}, {c=:.3f} {d=:.3f} {height=:.1f}\n\
converged: {params.success}, score: {params.fun:.3f} (from {score0:.3f})')
        plt.show()

    ret = [a, b, c, d, height] + [params.success]

    if not return_steps:
        return ret
    return ret, x0s


def find_layers(
        image_classification: np.ndarray,
        seeds: np.ndarray[float | int],
        height0: np.ndarray[float] | float,
        color: str,
        degree: int = 3,
        plts: bool = False,
        **kwargs
) -> pd.DataFrame:
    """
    Find distorted rectangles for all seeds.

    Parameters
    ----------
    image_classification : np.ndarray
        Classified image (light, dark, hole) for finding rectangles.
    seeds : Iterable[float | int]
        depths of layers to be searched.
    height0 : Iterable[float]
        Initial guesses for thicknesses of layers.
    color : str
        Color of layers.
    degree: int, optional
        Degree of the distortion polynomial. Default is 3. Higher degrees are
        theoretically possible but not supported.
    plts : bool, optional
        Whether to create result plot. The default is False.
    **kwargs : dict
        Optional keyword arguments. Will be ignored

    Returns
    -------
    df_out : pd.DataFrame
        Result dataFrame holding information about the seeds, parameters of 
        optimized rectangle, success of optimizer and color.

    """
    assert 0 <= degree <= 3, f'degree must be between 0 and 3, not {degree}'

    N: int = len(seeds)
    columns_df: list[str] = [
        'seed', 'a', 'b', 'c', 'd', 'height', 'success', 'color'
    ]
    df_out: pd.DataFrame = pd.DataFrame(
        data=np.zeros((N, len(columns_df)), dtype=object),
        columns=columns_df)
    df_out['seed'] = seeds
    df_out['color'] = color

    width: int = image_classification.shape[0]
    half_width: int = (width + 1) // 2

    image_classification_pad: np.ndarray[int] = np.pad(
        image_classification,
        ((0, 0), (half_width, half_width))
    )
    image_result: np.ndarray[int] = image_classification_pad.copy().astype(np.int32)

    if color in ('black', 'dark'):
        sign_l: int = -1
    else:
        sign_l: int = 1

    if not hasattr(height0, '__iter__'):
        height0: np.ndarray[float] = np.ones(N) * height0

    for idx in tqdm(range(0, N), desc='searching parameters for laminae'):
        out: list[float | bool] = find_layer(
            image_classification_pad,
            seeds[idx],
            height0[idx],
            color,
            plts=plts,
            degree=degree,
            **kwargs
        )

        df_out.iloc[idx, 1:-1] = out

        if plts:
            region = distorted_rect(width, out[-2], out[:-2])

            image_result[:, seeds[idx]: seeds[idx] + width] += \
                sign_l * 127 * region.astype(np.int32)

            plt.imshow(
                image_result[:, half_width:-half_width],
                vmin=-127, vmax=127 + 255,
                interpolation='none'
            )
            plt.title(f'{idx + 1} out of {N}')
            plt.show()

    return df_out


if __name__ == '__main__':
    pass
