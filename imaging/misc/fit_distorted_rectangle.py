import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Iterable
from scipy.optimize import minimize


def distorted_rect(
        width: int, height: float, coeffs: Iterable[float], plts: bool = False
) -> np.array:
    """
    Mask for distorted rectangle in square region.

    coeffs are expected to be in relative coordinates (coordinates that match
    those of x in [-1, 1]).

    Parameters
    ----------
    width : int
        The lateral extension of the rectangle. Function returns transposed image.
    height : float
        The thickness of the layer.
    coeffs : Iterable[float]
        The coefficients of the polynomial in relative coordinates.
    plts : bool, optional
        Will plot region, if True. The default is False.

    Returns
    -------
    region : np.ndarray[bool]
        The region of the distorted rectangle where pixels inside region are True.

    """
    x = np.linspace(-1, 1, width, dtype=float)

    Y = np.repeat(x, width).reshape((width, width))
    y = np.poly1d(coeffs)(x)
    y += coeffs[-1] - y.mean()
    y_shift = height / width / 2
    y_upper = y + y_shift
    y_lower = y - y_shift

    region = (Y > y_lower) & (Y <= y_upper)
    region = region.T

    if plts:
        plt.imshow(region)
        plt.plot(width / 2, width / 2, 'ro')
        plt.title(f'coeffs={[np.round(coeff, 3) for coeff in coeffs]}, {height=:.1f}')
        plt.show()

    return region


def find_layer(image_classification, seed, height0, color,
               key_light=255, key_dark=127, max_slope=.1, x0=None, bounds=None,
               plts=False, plts_region=False, **kwargs):
    """
    Find parameters for distorted rectangle around seed in classified image.

    Parameters
    ----------
    image_classification : np.ndarray[float]
        The classified image (light, dark, hole).
    seed : int | float
        Start depth of centerpoint of rectangle.
    height0 : float
        Initial guess for width of layer (width of peak).
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
    **kwargs : dict
        Additional keyword arguments. Will be ignored.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    parms : list
        Parameters of best distorted rectangle for seed.

    """

    def metric(x0):
        a, b, c, d, height = x0
        layer_region = distorted_rect(
            width, height, (a, b, c, d), plts=plts_region)
        image_region = image_classification[:, seed:seed + width]

        vision_layer = image_region * layer_region
        num_light_layer = np.sum(vision_layer == key_light)
        num_dark_layer = np.sum(vision_layer == key_dark)
        num_layer = num_light_layer + num_dark_layer + 1
        advantage_layer = (num_light_layer - num_dark_layer * ratio_total) / \
                          num_layer

        density = num_light_layer / num_layer
        density = is_dark - density
        score_layer = advantage_layer * density * height / width
        return score_layer

    if (key_light not in image_classification) or (key_dark not in image_classification):
        raise ValueError(f'Specify keys (values) for light and dark \
classification. Either {key_light=} or {key_dark=} is not in image.')

    if color.lower() not in ('light', 'dark'):
        raise KeyError(
            f'color has to be either "light" or "dark", not {color}.')

    is_dark = (color == 'dark')

    width = image_classification.shape[0]

    num_light_total = np.sum(image_classification == key_light)
    num_dark_total = np.sum(image_classification == key_dark)
    ratio_total = num_light_total / num_dark_total

    if x0 is None:
        x0 = [0, 0, 0, 0, height0]

    if bounds is None:
        bounds = ((-max_slope / 9, max_slope / 9),  # a * x**3
                  (-max_slope / 6, max_slope / 6),  # b * x**2
                  (-max_slope / 3, max_slope / 3),  # c * x
                  (-height0 / 2 / width, height0 / 2 / width),  # d
                  (height0 / 2, height0 * 2))  # thickness layer

    score0 = metric(x0)

    params = minimize(
        metric,  # function to minimize
        x0=x0,  # start values
        method='COBYLA',
        bounds=bounds
    )

    # get values
    a, b, c, d, height = params.x

    if plts:
        region = distorted_rect(width, height, (a, b, c, d))
        image_region = image_classification[:, seed:seed + width]
        plt.imshow(image_region + (1 - is_dark * 2) * 255 * region)
        plt.plot(width / 2, width / 2, 'ro')
        plt.plot(width / 2 + d, width / 2, 'bo')
        plt.title(f'{a=:.3f}, {b=:.3f}, {c=:.3f} {d=:.3f} {height=:.1f}\n\
converged: {params.success}, score: {params.fun:.3f} (from {score0:.3f})')
        plt.show()

    return list(params.x) + [params.success]


# depricated
# def find_middle_layer(
#         image_classification: np.ndarray,
#         seed: int,
#         row_upper: pd.Series,
#         row_lower: pd.Series,
#         height0: float,
#         color: str,
#         plts=False,
#         **kwargs
# ):


def find_layers(
        image_classification, seeds, height0, color,
        plts=False, **kwargs):
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
    color : Iterable[str]
        Colors of layers.
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
    N = len(seeds)
    columns_df = ['seed', 'a', 'b', 'c', 'd', 'height', 'success', 'color']
    df_out = pd.DataFrame(
        data=np.zeros((N, len(columns_df)), dtype=object),
        columns=columns_df)
    df_out['seed'] = seeds
    df_out['color'] = color

    width = image_classification.shape[0]
    half_width = (width + 1) // 2

    image_classification_pad = np.pad(
        image_classification,
        ((0, 0), (half_width, half_width))
    )
    image_result = image_classification_pad.copy().astype(np.int32)

    if color in ('black', 'dark'):
        sign_l = -1
    else:
        sign_l = 1

    height0 = np.ones(N) * height0

    time0 = time.time()
    print_interval = 10 ** (np.around(np.log10(N), 0) - 2)
    print(f'searching parameters for {N} {color} laminae ...')
    for idx in range(0, N):
        out = find_layer(
            image_classification_pad, seeds[idx], height0[idx], color, plts=plts, **kwargs
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

        if idx % print_interval == 0:
            time_now = time.time()
            time_elapsed = time_now - time0
            predict = time_elapsed * N / (idx + 1)  # s
            left = predict - time_elapsed
            left_min, left_sec = divmod(left, 60)
            print(end='\x1b[2K')
            print(f'estimated time left: {str(int(left_min)) + " min" if left_min != 0 else ""} {left_sec:.1f} sec',
                  end='\r')
    print()

    return df_out


def distorted_slice(params: np.ndarray, widths: Iterable, image_shape: tuple[int, int]) -> np.ndarray:
    """

    Parameters
    ----------
    params: matrix containing the parameters for each horizon.
        Vertical entries correspond to parameters describing one horizon
        horizontal entries correspond to the different horizons.
        The first entry in each row corresponds to the highest potence of the polynomial
    widths: iterable describing the width of each layer in pixels

    Returns
    -------

    """
    assert params.shape[1] == len(widths), 'The number of horizons must match the number of provided widths'

    # construct image
    # build stripes first
    image = np.cumsum(widths)
    ...

    # apply polytransform


def find_distorted_slice(image, n_transects=3):
    """
    Rather than fitting laminae by laminae, describe the laminated sediment with a backbone function

    Returns
    -------
    image: np.ndarray
        classified or grayscale image
    """


if __name__ == '__main__':
    # import manage_class_imports
    # # get the classification image
    # image_object = manage_class_imports.get_obj((490, 495), 'Alkenones', 'Image')

    # i_c, seeds = image_object.laminae_seeds()

    # image_object.plts = True
    # yearly_thickness = image_object.average_width_yearly_cycle()
    # width = i_c.shape[0]
    # half_width = (width + 1) // 2

    # height0 = yearly_thickness / 2  # --> expected thickness

    # c_new = np.pad(i_c, ((0, 0), (half_width, half_width)))
    # c_layered = c_new.copy().astype(np.uint32)

    # df_out = pd.DataFrame(
    #     data=np.zeros((len(seeds), 7), dtype=object),
    #     columns=['seed', 'a', 'b', 'c', 'd', 'height', 'success'])
    # for idx in range(0, len(seeds)):
    #     out = find_layer(c_new, seeds[idx], height0,
    #                      plts=True, max_slope=.1)
    #     df_out.iloc[idx, :] = [seeds[idx], out['coeffs'][0], out['coeffs'][1],
    #                            out['coeffs'][2], out['coeffs'][3], out['height'], out['params'].success]
    #     # df_out.to_csv('distorted_rect_params.csv')

    #     region = distorted_rect(width, out['height'], out['coeffs'])

    #     c_layered[:, seeds[idx]: seeds[idx]
    #               + width] += 127 * region.astype(np.uint32)

    #     plt.imshow(c_layered[:, half_width:-half_width], vmax=127 + 255)
    #     plt.title(f'{idx+1} out of {len(seeds)}')
    #     plt.show()

    pass

# plt.imshow(c_new)
# plt.plot(half_width + seeds[idx], half_width, 'ro')
# plt.show()


# vision_layer = image_region * region
# plt.imshow(vision_layer)
# plt.show()

# plt.figure()
# for idx in range(len(seeds)):
#     c_new[:, half_width + (seeds[idx] - half_width):half_width +
#           (seeds[idx] + half_width)] += 127 * r
# plt.imshow(c_new, vmax=255 + 127)
# plt.plot(half_width + seeds, half_width * np.ones_like(seeds), 'ro')
# plt.show()
