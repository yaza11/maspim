"""
Find the transformation from one distorted image to the other.
"""
from imaging.main.cImage import ImageROI
from util.cClass import Convinience
from imaging.util.Image_plotting import plt_cv2_image, plt_contours
from imaging.util.Image_helpers import find_points_in_grid
from imaging.util.coordinate_transformations import kartesian_to_polar
from util.manage_class_imports import get_params, save_params
from res.constants import transformation_target

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage

from scipy.optimize import minimize, basinhopping


def apply_to_uint8(array, func, verbose=False, **kwargs):
    if (dtype_initial := array.dtype) != np.uint8:
        if verbose:
            print('transforming img to uint8 to rescale and scaling back \
afterwards, this will introduce roundoff errors')
        offset_initial = array.min()
        # shift min val to 0
        array -= offset_initial
        # scale to 255
        factor_initial = array.max() / 255
        array /= factor_initial
        # convert to uint8
        array = array.astype(np.uint8)
        # apply func
        array_ret = func(array, **kwargs)
        # transform back
        array_ret = array_ret.astype(dtype_initial)
        array_ret *= factor_initial
        array_ret += offset_initial
        return array_ret
    return func(array, **kwargs)


def rescale_image(image_to_scale: np.ndarray, image_blueprint: np.ndarray,
                  verbose=False, plts=False) -> np.ndarray:
    """Scales input such that images have same resolution along axis."""
    if (np.argmax(image_blueprint.shape) != 1) or \
            (np.argmax(image_to_scale.shape) != 1):
        raise ValueError('Images are expected to be oriented horizontally.')
    # make the images have the same number of pixels along the long side
    desired_pixels_x = image_blueprint.shape[1]
    inferred_pixels_y = round(
        image_blueprint.shape[1]
        / image_to_scale.shape[1]
        * image_to_scale.shape[0])
    dsize = (desired_pixels_x, inferred_pixels_y)
    image_resized = apply_to_uint8(
        image_to_scale, cv2.resize,
        dsize=dsize, interpolation=cv2.INTER_AREA, verbose=verbose)

    if plts:
        plt_cv2_image(image_to_scale, title='image before scaling')
        plt_cv2_image(image_resized, title='image after scaling')
        plt_cv2_image(image_blueprint, title='target image')

    return image_resized


def find_epsilon(contour, number_desired_sides=4, verbose=False, plts=True):
    """Find epsilon that turns contour into desired n-gon."""
    epsilon = 0
    number_sides = contour.shape[0]
    len_contour = cv2.arcLength(contour, closed=True)
    if verbose:
        print('Searching epsilon!')
    while number_sides != number_desired_sides:
        if verbose:
            print(f'Current sides: {number_sides}, current epsilon: {epsilon}')
        if number_sides < number_desired_sides:
            diff = number_desired_sides - number_sides
            epsilon *= (1 - 10 ** (- diff))
        else:
            epsilon += len_contour / 10
        approx = cv2.approxPolyDP(
            contour, epsilon, closed=True)
        number_sides = approx.shape[0]
        if plts:
            plt.figure()
            plt.plot(approx[:, 0, 0], approx[:, 0, 1])
            plt.title(
                f'Current sides: {number_sides}, current epsilon: {epsilon}')
            plt.show()
    return approx, epsilon


def prepare_corners(ROI_corners, method, plts=False, verbose=False):
    """Find transformation that matches corners."""
    # format: [[x_topleft, y_topleft],
    #          [x_bottomleft, y_bottomleft],
    #          [x_bottomright, y_bottomright],
    #          [x_topright, y_topright]]
    if method == 'epsilon':
        ROI_corners = np.array([ROI_corners[i, 0, :]
                                for i in range(4)], dtype=np.float32)
    # for rot_rect points may be outside of ROI, scale all points by factor of
    # 1 / 2
    elif method == 'rot_rect':
        ROI_corners = ROI_corners.astype(float)
        if plts:
            plt.figure()
            plt.plot(ROI_corners[:, 0], ROI_corners[:, 1],
                     label='unsorted', linestyle='--')

        center = ROI_corners.mean(axis=0)
        ROI_corners -= center
        ROI_corners /= 2
        ROI_corners += center
        ROI_corners = ROI_corners.astype(np.float32)

        if plts:
            plt.plot(
                ROI_corners[:, 0], ROI_corners[:, 1],
                label='after scaling')

        # make sure corners are in the right order
        x = ROI_corners[:, 0]
        y = ROI_corners[:, 1]
        x_c = x.copy() - ROI_corners[:, 0].mean()
        y_c = y.copy() - ROI_corners[:, 1].mean()

        r, phi = kartesian_to_polar(x_c, y_c)
        o = np.argsort(phi)
        ROI_corners[:, 0] = x[o]
        ROI_corners[:, 1] = y[o]
    else:
        raise KeyError()

    if plts:
        plt.plot(ROI_corners[:, 0], ROI_corners[:, 1],
                 label='after sorting', linestyle='--')
        plt.axis('equal')
        plt.title('after sorting corners')
        plt.grid('on')
        plt.legend()
        plt.show()

    return ROI_corners


def stretching_func(a, b, c, d):
    """Return the model function by which the probe is stretched."""
    return lambda x: a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x


def flatten_func(x, y):
    """Calculate the arc length of the curve."""
    dx = np.diff(x, axis=-1)
    dy = np.diff(y, axis=-1)
    x_stretched = np.cumsum(np.sqrt(dx ** 2 + dy ** 2), axis=-1)
    # zeropad to the left by 1
    if len(x.shape) == 1:
        pads = (1, 0)
    else:
        pads = ((0, 0), (1, 0))
    return np.pad(x_stretched, pads, 'constant')


def rescale_x(x, x_stretched):
    """Scale the stretched out x to be in the bounds of original x."""
    range_x = x.max() - x.min()
    range_x_stretched = x_stretched.max() - x_stretched.min()
    x_rescaled = (x_stretched - x_stretched.min()) / \
        range_x_stretched * range_x + x.min()
    return x_rescaled


def apply_stretching(a: float, b: float, c: float, d: float, arr: np.ndarray):
    """Apply the stretching transform to a vector or array."""
    # dimension in x direction
    N_x = arr.shape[-1]
    # arr holding x pixel coords
    x = np.ones_like(arr) * np.arange(N_x, dtype=float)
    # center
    x_middle = (x.max() - x.min()) / 2 + x.min()
    x -= x_middle
    # scale (--> -1/2 to 1/2)
    x /= N_x
    # get func
    func = stretching_func(a, b, c, d)
    # get func vals
    y = func(x)
    # stretch
    x_stretched = flatten_func(x, y)
    # squeeze
    x_rescaled = rescale_x(x, x_stretched)
    # interpolate values on new grid
    if len(arr.shape) == 1:
        arr_stretched = np.interp(x, x_rescaled, arr)
    elif len(arr.shape) == 2:
        N_y = arr.shape[0]
        arr_stretched = np.zeros_like(arr)
        for idx_y in range(N_y):
            arr_stretched[idx_y, :] = np.interp(
                x[idx_y, :], x_rescaled[idx_y, :], arr[idx_y, :])
    return arr_stretched, x_rescaled * N_x + x_middle


def evaluate_fit(x0, vec_to_transform, vec2):
    """Calculate error (negative correlation) between target and object."""
    if len(vec2) != len(vec_to_transform):
        raise ValueError('vectors must be of same shape!')
    a, b, c, d = list(x0)
    vec_transformed, _ = apply_stretching(a, b, c, d, vec_to_transform)
    df = pd.DataFrame(
        data=np.vstack([vec2, vec_transformed]).T,
        columns=['vec2', 'vec_t'])
    corr = df.corr().loc['vec2', 'vec_t']
    return -corr


def perform_optimization_for_conditions(
        mean_light_from, mean_light_to, x0, method, verbose=False):
    params = minimize(
        evaluate_fit,
        x0=x0,
        args=(mean_light_from, mean_light_to),
        method=method)
    if verbose:
        print(f'optimization successful: {params.success}')
    # get params
    x = params.x
    err = evaluate_fit(
        x, mean_light_from, mean_light_to)
    if verbose:
        print(f'finished with correlation of {-err} and params {x=} \
and {method=})')
    return x, err


def get_shift_vectors(x00, shift_vals, try_shift_vals_dense):
    # only along coordinate axes
    shift_vals = np.concatenate([-shift_vals, shift_vals[::-1]])

    # shifts will be
    # [shift_vals[0], 0, 0, 0]
    # [0, shift_vals[0], 0, 0]
    # ...
    # [shift_vals[1], 0, 0, 0]
    # ...

    # try all possible combinations of offsets
    if try_shift_vals_dense:
        raise NotImplementedError()
    else:
        x0s = []
        for s_idx, shift in enumerate(shift_vals):
            for x_idx in range(len(x00)):
                x = [x_ for x_ in x00]
                x[x_idx] += shift
                x0s.append(x)

    return x0s


def find_stretching_function(
        mean_light_from, mean_light_to,
        methods=['CG'],
        shift_vals=np.array([1, 2, 0.5, .25]),
        try_shift_vals_dense=False,
        x00=np.array([0, 0, 0, 1]),
        plts=False, verbose=False, **kwargs):
    """Find the stretching function from one image to the other."""

    # if there is no saved file, find the parameters
    if verbose:
        print('finding stretching model')

    x0s = get_shift_vectors(x00, shift_vals, try_shift_vals_dense)

    # initiate error
    err00 = evaluate_fit(
        x00, mean_light_from, mean_light_to)
    # take the best result from multiple runs
    x_best = x00
    err_best = err00
    for idx, x0 in enumerate(x0s):
        for jdx, method in enumerate(methods):
            if verbose:
                print(f'optimization with x0={x0} and {method=} \
(run {idx * len(methods) + jdx} of {len(x0s) * len(methods)})')
            x, err = perform_optimization_for_conditions(
                mean_light_from, mean_light_to, x0, method, verbose=verbose)
            if err < err_best:
                err_best = err
                x_best = x
                method_best = method
    x = x_best
    err = err_best
    return {'x': x, 'err': err, 'err00': err00, 'method': method_best}


def find_stretching_function_basin(
        mean_light_from,
        mean_light_to,
        x00=np.array([0, 0, 0, 1])
):
    def corr_with(vec):
        vecm = vec - np.mean(vec)
        return to_scaled @ vecm / np.sqrt(np.sum(vecm ** 2))

    def evaluate_fit_pad(x0):
        """Calculate error (negative correlation) between target and object."""
        #a, b, c, d, zeropad_l, zeropad_r = list(x0)
        a, b, c, d, = list(x0)
        # zeropad_l = (np.min([np.max([0, zeropad_l]), 1]) * width_pad_max).astype(int)
        # zeropad_r = (np.min([np.max([0, zeropad_r]), 1]) * width_pad_max).astype(int)
        vec_transformed, _ = apply_stretching(a, b, c, d, mean_light_from)
        # vec_transformed = np.pad(vec_transformed, (zeropad_l, zeropad_r))
        return -corr_with(vec_transformed)

    # scaling to mean 0 and var 1
    # therefore this has to be calculated only once
    to_scaled = (mean_light_to - mean_light_to.mean()).T / np.sqrt(np.sum((mean_light_to - mean_light_to.mean())**2))
    err00 = evaluate_fit_pad(x00)
    print(f'starting optimization with loss {err00:.3f}')
    params = basinhopping(
        evaluate_fit_pad,
        x0=x00,
        # stepsize=.05,
        disp=True,
        minimizer_kwargs={'method': 'L-BFGS-B'}
    )

    print(f'optimization successful: {params.success}')

    # get params
    x = params.x
    err = evaluate_fit_pad(x)
    print(f'finished with correlation of {-err} and params {x=}')
    return {'x': x, 'err': err, 'err00': err00, 'method': 'basin_hopping'}


def plt_two_channel(image_red, image_cyan, title='', hold=False):
    image_compare = np.stack(
        [image_red / image_red.max(),
         image_cyan / image_cyan.max(),
         image_cyan / image_cyan.max()]
    ).swapaxes(0, 2).swapaxes(0, 1)
    if (not hold) and (hold != ' on'):
        plt.figure()
    plt.imshow(image_compare)
    plt.title(title)
    if (not hold) and (hold != ' on'):
        plt.show()


class ImageTransformation(Convinience):
    """Contains functions acting on object level and performs trafo."""

    def __init__(
            self, section, window_from, window_to):
        """Initializer."""

        self._section = section
        self._window_from = window_from
        self._window_to = window_to

        self.plts = False
        self.verbose = False

    def perform_steps(self, **kwargs):
        self.set_img_objs()
        # rescale ROI of from_obj, classify laminae
        self.prepare_ROIs_from_image_obj()

        # match the rotated extents of the probes and transform light laminae
        self.match_bounding_boxes()

        # apply the transformation to the laminae classification
        self.warpPerspective_laminae()
        self.get_stretching_function(**kwargs)

        self.save_stretching_function(**kwargs)
        self.create_transformed()
        self.save_transformed(**kwargs)
        self.create_mapping()
        self.save_mapping(**kwargs)
        self.plt_final()

    def load(self):
        self.set_img_objs()
        self.prepare_ROIs_from_image_obj()
        self.match_bounding_boxes()
        self.warpPerspective_laminae()

        self.get_stretching_function()
        self.load_transformed()
        self.load_mapping()
        self.plt_final()

    def set_img_objs(self):
        # load objects
        self.from_obj = ImageROI(section=self._section, window=self._window_from)
        self.from_obj.load()

        self.to_obj = ImageROI(section=self._section, window=self._window_to)
        self.to_obj.load()

        # set objects verbose and plt attributes
        for i_obj in [self.from_obj, self.to_obj]:
            i_obj.plts = self.plts
            i_obj.verbose = self.verbose

    def handle_rescaling(self):
        """Match the scales along the horizontal axis."""
        if self.verbose:
            print('rescaling img')

        # rescale the laminae so they dont have to be redefined
        self.from_obj.image_classification = rescale_image(
            self.from_obj.image_classification, self.image_to
        )
        self.from_obj._image_original = rescale_image(
            self.from_obj.sget_image_original(), self.image_to, verbose=self.verbose, plts=self.plts)

    def prepare_ROIs_from_image_obj(self):
        """Make sure the horizontal resolution of from_obj matches to_obj."""
        assert self.from_obj.check_attribute_exists('image_classification')
        assert self.to_obj.check_attribute_exists('image_classification')
        self.image_to = self.to_obj.sget_image_original()
        # rescale from obj
        self.handle_rescaling()
        # get the images
        self.image_from = self.from_obj.sget_image_original()

    def get_quadrilateral(self, key, method):
        """Get the bounding rectangle of the simplified probe area."""
        if key == 'from':
            ROI_obj = self.from_obj
        elif key == 'to':
            ROI_obj = self.to_obj

        # simplify the binary image by stepping up size of median filter
        image_simplified = ROI_obj.sget_simplified_image()
        # get the largest contour (hopefully the only one)
        probe_contour = ROI_obj.get_main_contour(
            image_binary=image_simplified,
            method='take_largest')
        if method == 'epsilon':
            # tweak epsilon until contour has 4 points
            probe_contour_simplified, _ = find_epsilon(
                probe_contour, plts=False, verbose=self.verbose)
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        elif method == 'rot_rect':
            rect = cv2.minAreaRect(probe_contour)
            box = cv2.boxPoints(rect)
            probe_contour_simplified = np.intp(box)
        else:
            raise NotImplementedError
        if self.plts:
            plt_contours([probe_contour_simplified], image_simplified)
        return probe_contour_simplified

    def match_bounding_boxes(self, method='rot_rect'):
        ROI_from_corners = self.get_quadrilateral(
            'from', method=method)
        ROI_to_corners = self.get_quadrilateral(
            'to', method=method)
        # sort corners or bring in right shape, depending on method
        ROI_from_corners = prepare_corners(
            ROI_from_corners, method, plts=self.plts, verbose=self.verbose)
        ROI_to_corners = prepare_corners(
            ROI_to_corners, method, plts=self.plts, verbose=self.verbose)

        # apply warp perspective
        if self.verbose:
            print('warping perspective by matching corners of quadrilaterals')

        self.M = cv2.getPerspectiveTransform(ROI_from_corners, ROI_to_corners)

        # transformation step
        ROI_from_matched_box = cv2.warpPerspective(
            self.image_from, self.M, dsize=(self.image_to.shape[1], self.image_to.shape[0]))

        if self.plts:
            plt_cv2_image(self.image_from, title='ROI before warp perspective')
            plt_cv2_image(ROI_from_matched_box,
                          title='ROI after warp perspective')

        self.image_from = ROI_from_matched_box

    def optical_flow(self):
        self.set_img_objs()
        self.prepare_ROIs_from_image_obj()
        self.match_bounding_boxes()

        if len(self.image_to.shape) == 3:
            img_target = skimage.color.rgb2gray(self.image_to)
        else:
            img_target = self.image_to
        if len(self.image_from.shape) == 3:
            img_moving = skimage.color.rgb2gray(self.image_from)
        else:
            img_moving = self.image_from

        img_target = img_target.astype(float) / img_target.max()
        img_moving = img_moving.astype(float) / img_moving.max()

        # calculate flow field for target and source
        print('calculating dense flow (this may take a while)')
        self.flow = skimage.registration.optical_flow_tvl1(img_target, img_moving)

        if self.plts:
            self.plt_flow()

    def plt_flow(self, nvec=50):
        if len(self.image_to.shape) == 3:
            image0 = skimage.color.rgb2gray(self.image_to)
        else:
            image0 = self.image_to
        if len(self.image_from.shape) == 3:
            image1 = skimage.color.rgb2gray(self.image_from)
        else:
            image1 = self.image_from

        image0 = image0.astype(float) / image0.max()
        image1 = image1.astype(float) / image1.max()

        v, u = self.flow

        nr, nc = image0.shape

        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                             indexing='ij')

        image1_warp = skimage.transform.warp(
            image1, np.array([row_coords + v, col_coords + u]),
            mode='edge'
        )

        # build an RGB image with the unregistered sequence
        seq_im = np.zeros((nr, nc, 3))
        seq_im[..., 0] = image1
        seq_im[..., 1] = image0
        seq_im[..., 2] = image0

        # build an RGB image with the registered sequence
        reg_im = np.zeros((nr, nc, 3))
        reg_im[..., 0] = image1_warp
        reg_im[..., 1] = image0
        reg_im[..., 2] = image0

        # --- Show the result

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 6))

        ax0.imshow(seq_im)
        ax0.set_title("Unregistered sequence")
        ax0.set_axis_off()

        ax1.imshow(reg_im)
        ax1.set_title("Registered sequence")
        ax1.set_axis_off()

        # --- Compute flow magnitude
        norm = np.sqrt(u ** 2 + v ** 2)

        # --- Quiver plot arguments

        step = max(nr // nvec, nc // nvec)

        y, x = np.mgrid[:nr:step, :nc:step]
        u_ = u[::step, ::step]
        v_ = v[::step, ::step]

        ax2.imshow(norm)
        ax2.quiver(x, y, u_, v_, color='r', units='dots',
                   angles='xy', scale_units='xy', lw=3)
        ax2.set_title("Optical flow magnitude and vector field")
        ax2.set_axis_off()

        fig.tight_layout()

        plt.show()

    def warpPerspective_laminae(self):
        # transform the light lamine in the from obj used for the next
        image_classification = cv2.warpPerspective(
            self.from_obj.image_classification,
            self.M,
            dsize=(self.image_to.shape[1], self.image_to.shape[0]))

        if self.plts:
            plt_cv2_image(self.from_obj.image_classification,
                          title='light pixels of scaled image')
            plt_cv2_image(image_classification,
                          title='light pixels of perspective transformed image')

        self.from_obj.image_classification = image_classification

    def get_stretching_function(self, overwrite=False, **kwargs):
        # try to load params from memory
        if ((params := get_params(self.from_obj, self.to_obj, 'x')) is not None) \
                and (not overwrite):
            self.dict_stretching = {'x': params,
                                    'err': np.nan, 'err00': np.nan, 'method': 'loaded'}
        # define 1d laminae averages
        if self.plts or (params is None) or overwrite:
            i_from = self.from_obj.image_classification.copy().astype(float)
            i_to = self.to_obj.image_classification.copy().astype(float)
            # i_from[self.from_obj.image_classification < 127 / 2] = np.nan
            # i_to[self.to_obj.image_classification < 127 / 2] = np.nan

            mean_light_from = np.nanmean(i_from, axis=0)
            mean_light_to = np.nanmean(i_to, axis=0)

        if (params is None) or overwrite:
            # allow zeropadding of two layers
            # pad = round(self.to_obj.get_average_width_yearly_cycle() * 2)
            # mean_light_to = np.pad(mean_light_to, (pad, pad))
            self.dict_stretching = find_stretching_function_basin(
                mean_light_from, mean_light_to, **kwargs)
            params = self.dict_stretching['x']

        if self.plts:
            # a, b, c, d, l, r = params
            a, b, c, d = params
            mean_light_from_transformed, x_distorted = apply_stretching(
                a, b, c, d, mean_light_from)
            # mean_light_from_transformed = np.pad(mean_light_from_transformed, (pad, pad))

            # plot showing the average number of light pixels per column before and
            #   after transformation
            plt.figure()
            plt.plot(mean_light_to, label='to', color='blue')
            plt.plot(mean_light_from, label='from',
                     color='orange', linestyle='--')
            plt.plot(mean_light_from_transformed, label='from transformed',
                     color='red', linestyle='--')
            plt.title(
                r'$f=ax^4 + bx^3 + cx^2 + dx$' +
                f'\n with {a=:.2e}, {b=:.2e}, {c=:.2e}, {d=:.2e} \n' +
                f' corr={-self.dict_stretching["err"]:.3f} (corr0={-self.dict_stretching["err00"]:.3f}) \
(with method {self.dict_stretching["method"]})')
            plt.tight_layout()
            plt.legend()
            plt.show()

            # plot showing the distortions
            plt.figure()
            x = np.arange(len(x_distorted))
            dx = x_distorted - x
            plt.plot(x / x.max(), label='x')
            plt.plot(x_distorted / x_distorted.max(), label='distorted x')
            plt.plot(dx / np.abs(dx).max(),
                     label=f'deviation of distorted x from x (max={int(np.abs(dx).max())})')
            x_func = np.linspace(-.5, .5, len(x))
            y_func = stretching_func(a, b, c, d)(x_func)
            dy_func = (4 * a * x_func ** 3 + 3 * b *
                       x_func ** 2 + 2 * c * x_func + d)
            plt.plot(y_func / y_func.max(), label='deformation function')
            plt.plot(dy_func / dy_func.max(), label='derivative')
            plt.plot(np.diff(x_distorted) / np.diff(x),
                     label='stretching factors')
            plt.legend()
            plt.grid('on')
            plt.show()

            # plot showing the light laminae overlayed
            ROI_from_laminae_distorted = self.from_obj.image_classification
            ROI_to_laminae = self.to_obj.image_classification
            plt_two_channel(ROI_to_laminae, ROI_from_laminae_distorted, title='light laminae of \n\
target (red), objective perspective transformed (cyan)')

        return self.dict_stretching

    def save_stretching_function(self, overwrite=False, **kwargs):
        if self.dict_stretching is None:
            self.dict_stretching = self.get_stretching_function(
                overwrite=overwrite)
        x = self.dict_stretching['x']
        save_params(self.from_obj, self.to_obj, x, 'x', overwrite=overwrite)

    def create_transformed(self):
        # apply
        # 1. rescale
        # 2. warpPerspective
        # 3. stretching to (continous) pixel coordinates of from_obj
        I = ImageROI(self._section, self._window_from)
        I.load()
        y = np.arange(I.xywh_ROI[-1])
        x = np.arange(I.xywh_ROI[-2])
        assert x.shape[0] > y.shape[0]
        X, Y = np.meshgrid(x, y)

        # convert to 2 uint8 arrays
        X_mod = (X % 256).astype(np.uint8)
        X_fac = (X // 256).astype(np.uint8)

        Y_mod = (Y % 256).astype(np.uint8)
        Y_fac = (Y // 256).astype(np.uint8)

        # rescale
        transformed = []
        for I in [X_mod, X_fac, Y_mod, Y_fac]:
            I = rescale_image(I, self.image_to)
            transformed.append(I)
        X_mod, X_fac, Y_mod, Y_fac = transformed
        X = (X_fac * 256 + X_mod).astype(float)
        Y = (Y_fac * 256 + Y_mod).astype(float)
        # warp perspective
        X = cv2.warpPerspective(
            X, self.M, dsize=(self.image_to.shape[1], self.image_to.shape[0]))
        Y = cv2.warpPerspective(
            Y, self.M, dsize=(self.image_to.shape[1], self.image_to.shape[0]))

        # stretching
        # a, b, c, d, l, r = self.dict_stretching['x']
        a, b, c, d = self.dict_stretching['x']
        X, _ = apply_stretching(a, b, c, d, X)
        Y, _ = apply_stretching(a, b, c, d, Y)

        # X = np.pad(X, ((0, 0), (l, r)))
        # Y = np.pad(Y, ((0, 0), (l, r)))

        X[X == 0] = np.nan
        Y[Y == 0] = np.nan

        self.X_transformed = X
        self.Y_transformed = Y

        return X, Y

    def save_transformed(self, overwrite=False, **kwargs):
        # save X field
        save_params(self.from_obj, self.to_obj, self.X_transformed,
                    'X_transformed', overwrite)
        # save Y field
        save_params(self.from_obj, self.to_obj, self.Y_transformed,
                    'Y_transformed', overwrite)

    def load_transformed(self):
        if not self.check_attribute_exists('from_obj'):
            self.from_obj = ImageROI(section=self._section, window=self._window_from)
            self.from_obj.load()
        if not self.check_attribute_exists('to_obj'):
            self.to_obj = ImageROI(section=self._section, window=self._window_to)
            self.to_obj.load()
        self.X_transformed = get_params(
            self.from_obj, self.to_obj, 'X_transformed')
        self.Y_transformed = get_params(
            self.from_obj, self.to_obj, 'Y_transformed')

    def get_transformed(self):
        self.load_transformed()
        if (self.X_transformed is None) or (self.Y_transformed is None):
            self.perform_steps()
        return self.X_transformed, self.Y_transformed

    def create_mapping(self):
        # create mask that links each pixel in transformed image to original
        #   imageI = ImageROI(self._section, self._window_from)
        I = ImageROI(self._section, self._window_from)
        I.load()
        y = np.arange(I.xywh_ROI[-1])
        x = np.arange(I.xywh_ROI[-2])
        assert x.shape[0] > y.shape[0]
        X, Y = np.meshgrid(x, y)
        # filter out nans

        # transformed points
        p_transformed = np.c_[
            self.X_transformed.ravel(), self.Y_transformed.ravel()]

        # set invalid pixels to fairly low value since kdtree can't handle nans
        p_transformed[np.isnan(p_transformed)] = -1e6

        # grid points
        p_grid = np.c_[X.ravel(), Y.ravel()]

        distances, idxs = find_points_in_grid(p_grid, p_transformed, distance_upper_bound=1e3)
        # idxs corresponds to the points in p_grid
        # kdtree asigns N to values not found
        N = X.ravel().shape[0]
        mask_valid = idxs < N

        idxs_valid = idxs[mask_valid]

        X_mapped = np.zeros_like(idxs, dtype=float)
        Y_mapped = np.zeros_like(idxs, dtype=float)
        X_mapped[mask_valid] = p_grid[idxs_valid, 0]
        Y_mapped[mask_valid] = p_grid[idxs_valid, 1]
        X_mapped[~mask_valid] = np.nan
        Y_mapped[~mask_valid] = np.nan

        self.X_mapped = X_mapped.reshape(
            self.X_transformed.shape)
        self.Y_mapped = Y_mapped.reshape(
            self.Y_transformed.shape)

        if self.plts:
            plt.imshow(self.X_mapped)
            plt.show()
            plt.imshow(self.Y_mapped)
            plt.show()

    def save_mapping(self, overwrite=False, **kwargs):
        # save X field
        save_params(self.from_obj, self.to_obj, self.X_mapped,
                    'X_mapped', overwrite)
        # save Y field
        save_params(self.from_obj, self.to_obj, self.Y_mapped,
                    'Y_mapped', overwrite)

    def load_mapping(self):
        if self.from_obj is None:
            self.from_obj = ImageROI(section=self._section, window=self._window_from)
            self.from_obj.load()
        if self.to_obj is None:
            self.to_obj = ImageROI(section=self._section, window=self._window_from)
            self.to_obj.load()
        self.X_mapped = get_params(
            self.from_obj, self.to_obj, 'X_mapped')
        self.Y_mapped = get_params(
            self.from_obj, self.to_obj, 'Y_mapped')

    def get_mapping(self):
        self.load_mapping()
        if (self.X_mapped is None) or (self.Y_mapped is None):
            self.perform_steps()
        return self.X_mapped, self.Y_mapped

    def apply_mapping_to_image(self, image_from):
        """Transform image in from_context to to_context."""
        mask_invalid = (np.isnan(self.Y_mapped)) | (
            np.isnan(self.X_mapped))
        mask_valid = ~mask_invalid.ravel()

        image_mapped = np.zeros_like(mask_valid, dtype=image_from.dtype)
        image_mapped[mask_valid] = image_from[
            self.Y_mapped.ravel()[mask_valid].astype(int),
            self.X_mapped.ravel()[mask_valid].astype(int)]
        image_mapped = image_mapped.reshape(self.X_mapped.shape)

        if self.plts:
            plt_cv2_image(image_mapped)

        return image_mapped

    def apply_mapping_to_points(self, points):
        """For each point (=row in points) transform the (x_from, y_from)."""
        X_grid = self.X_transformed.ravel()
        Y_grid = self.Y_transformed.ravel()
        mask_invalid = np.isnan(X_grid) | np.isnan(Y_grid)
        # set invalid points to very low values
        X_grid[mask_invalid] = -1e9
        Y_grid[mask_invalid] = -1e9
        grid_points = np.c_[X_grid, Y_grid]
        distances, idxs = find_points_in_grid(grid_points, points, distance_upper_bound=1e3)

        return distances, idxs

    def plt_final(self, **kwargs):
        if self.plts is False:
            return
        self.plts = False
        from_orig = ImageROI(self._section, self._window_from)
        from_orig.load()
        image_from_laminae_distorted = self.apply_mapping_to_image(
            from_orig.image_classification)
        image_to_laminae = self.to_obj.image_classification
        plt_two_channel(
            image_to_laminae, image_from_laminae_distorted,
            title=f'classification in {self.from_obj._section} of \n\
target ({self.to_obj._window}, red) and fitted ({self.from_obj._window}, cyan)',
            **kwargs
        )
        self.plts = True


def create_all_transformations():
    from constants import sections_all, windows_all
    window_to = transformation_target
    windows_all.remove(window_to)
    print(f'creating transformations for sections {sections_all} from {windows_all} to {window_to}')
    for section in sections_all:
        for window in windows_all:
            print(section, window)
            self = ImageTransformation(section=section, window_from=window, window_to=window_to)
            self.perform_steps()
            self.plts = True
            self.plt_final()


if __name__ == '__main__':
    section = (500, 505)
    window_from = 'FA'
    window_to = 'Alkenones'
    self = ImageTransformation(section=section, window_from=window_from, window_to=window_to)
    self.plts = False
    self.load()
    
    from defence import create_gif
    
    figs_warp = []
    # images for blending transformation
    for alpha in np.linspace(0, 1, 21, endpoint=1):
        image_classification = cv2.warpPerspective(
            self.from_obj.sget_image_grayscale(),
            self.M * alpha + np.identity(3) * (1 - alpha),
            dsize=(self.image_to.shape[1], self.image_to.shape[0]))
        
        fig = plt_cv2_image(
            np.stack([
                image_classification,
                image_classification, 
                self.to_obj.sget_image_grayscale()
            ], axis=-1), 
            no_ticks=True, 
            title=f'blending factor: {alpha:.1f}',
            hold = True
        )
        figs_warp.append(fig)
    
    create_gif(figs_warp, 'warping.gif', duration=0.25)
    for fig in figs_warp:
        fig.close() 
    
    a, b, c, d = self.dict_stretching['x']
    I = [0, 0, 0, 1]
    figs_stretch = []
    for alpha in np.linspace(0, 1, 21, endpoint=1):
        a_ = a * alpha
        b_ = b * alpha
        c_ = c * alpha
        d_ = d * alpha + (1 - alpha)
        image_transformed = apply_stretching(a_, b_, c_, d_, self.from_obj.image_classification)[0]
        
        fig = plt_cv2_image(
            np.stack([
                image_transformed,
                image_transformed, 
                self.to_obj.image_classification
            ], axis=-1), 
            no_ticks=True, 
            title=f'blending factor: {alpha:.1f}',
            hold=True
        )
        figs_stretch.append(fig)
        
    create_gif(figs_stretch, 'stretching.gif', duration=0.25)
    for fig in figs_warp:
        fig.close() 
    
    # self.plt_final() 
    # self.create_mapping()
