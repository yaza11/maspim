"""Module for finding transformations between sample regions."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import skimage.registration

from scipy.optimize import basinhopping
from typing import Callable, Any, Iterable, Sequence
from skimage.transform import warp, PiecewiseAffineTransform

from msi_workflow.imaging.main import ImageSample, ImageROI
from msi_workflow.imaging.register.finding_global_shift import apply_stretching, stretching_func
from msi_workflow.imaging.register.helpers import apply_displacement, Mapper
from msi_workflow.imaging.register.descriptor import Descriptor
from msi_workflow.imaging.util.coordinate_transformations import (
    kartesian_to_polar, polar_to_kartesian, rescale_values)
from msi_workflow.imaging.util.image_plotting import plt_contours, plt_cv2_image
from msi_workflow.res.constants import key_light_pixels, key_dark_pixels, key_hole_pixels

logger = logging.getLogger(__name__)


def apply_to_uint8(
        array: np.ndarray, func: Callable[[np.ndarray[np.uint8], ...], Any], *args: Any, **kwargs: Any
) -> Any:
    """
    Apply a function to converted image.

    If the input array is not of type uint 8 already, it will be converted,
    preserving the value range and afterward converted back. This may be
    inaccurate for arrays with small spacings between values. It is assumed
    that func takes the array as first argument

    Parameters
    ----------
    array : np.ndarray
        Input array.
    func: Callable[[np.ndarray[np.uint8], ...], Any]
        Function to apply that requires a uint8 array as input.
    args: Any
        Arguments to be passed on to func.
    kwargs: Any
        Keyword arguments to be passed on to func.

    Returns
    -------
    array_ret: np.ndarray
        The return value of func.
    """
    if (dtype_initial := array.dtype) != np.uint8:
        offset_initial = array.min()
        # shift min val to 0
        array -= offset_initial
        # scale to 255
        factor_initial = array.max() / 255
        array /= factor_initial
        # convert to uint8
        array = array.astype(np.uint8)
        # apply func
        array_ret = func(array, *args, **kwargs)
        # fit back
        array_ret = array_ret.astype(dtype_initial)
        array_ret *= factor_initial
        array_ret += offset_initial
        return array_ret
    return func(array, *args, **kwargs)


def rescale_image_axis(
        source: np.ndarray,
        target: np.ndarray,
        plts: bool = False
) -> np.ndarray:
    """
    Scales input such that images have same resolution along the first axis.

    Maintains the aspect ratio of the target image.

    Parameters
    ----------
    source: np.ndarray
        The image serving as reference for the shape.
    target: np.ndarray
        The image to be rescaled.
    plts: bool, optional
        Plot the input and rescaled images. The default is False.
    """
    if (np.argmax(source.shape) != 1) or \
            (np.argmax(target.shape) != 1):
        raise ValueError('Images are expected to be oriented horizontally.')
    # make the images have the same number of pixels along the long side
    desired_pixels_x: int = target.shape[1]
    inferred_pixels_y: int = round(
        target.shape[1]
        / source.shape[1]
        * source.shape[0]
    )
    dsize: tuple[int, int] = (desired_pixels_x, inferred_pixels_y)
    image_resized: np.ndarray[np.uint8] = apply_to_uint8(
        source, cv2.resize,
        dsize=dsize, interpolation=cv2.INTER_AREA)

    if plts:
        plt_cv2_image(source, title='image before scaling')
        plt_cv2_image(image_resized, title='image after scaling')
        plt_cv2_image(target, title='target image')

    return image_resized


def find_line_contour_intersect(
        contour: np.ndarray[int],
        holes: list[np.ndarray[int], np.ndarray[int]],
        hole_side: str,
        image_shape: tuple[int, ...]
) -> np.ndarray[int]:
    """
    Find the intersection point of the line with the sample area by projecting
    down/up from the first hole.

    Parameters
    ----------
    contour: np.ndarray[int]
        A contour.
    holes: list[np.ndarray[int], np.ndarray[int]]
        Position of holes (e.g. result from find_holes).
    hole_side: str
        Location of holes (either 'top' or 'bottom').
    image_shape: tuple[int, ...]
        Shape of the image in which the contour lives.
    """
    # use opposite half of that where holes were searched
    if hole_side == 'bottom':
        mask_section: np.ndarray[bool] = contour[:, 0, 1] < image_shape[0] / 2
    else:
        mask_section: np.ndarray[bool] = contour[:, 0, 1] > image_shape[0] / 2
    # extract the values of the contour in the depth-direction
    x_contour: np.ndarray[int] = contour[mask_section, 0, 0]

    # find idxs where contour intersects line
    x_hole: int = holes[0][1]
    idx_projected: int = np.argmin(np.abs(x_contour - x_hole))

    projected_point: np.ndarray[int] = contour[mask_section, 0, :][idx_projected]
    return projected_point[::-1]


def find_corners_contour(contour: np.ndarray[int]) -> list[np.ndarray[int]]:
    """
    Find corner points spanning a rectangle around a contour.

    Parameters
    ----------
    contour: np.ndarray[int]
        The contour.

    Returns
    -------
    points: list[np.ndarray[int]]
        4 Entries for upper left, upper right, and lower left, lower right in
        format (y, x).
    """
    xmin = contour[:, 0, 0].min()
    xmax = contour[:, 0, 0].max()
    ymin = contour[:, 0, 1].min()
    ymax = contour[:, 0, 1].max()
    points = [
        np.array([ymin, xmin]),
        np.array([ymin, xmax]),
        np.array([ymax, xmin]),
        np.array([ymax, xmax])
    ]
    return points


def sort_corners(corners: np.ndarray[int]) -> np.ndarray:
    """
    Bring an array of corners in anti-clockwise order.
    
    format = [
        [x_topleft, y_topleft],
        [x_bottomleft, y_bottomleft],
        [x_bottomright, y_bottomright],
        [x_topright, y_topright]
    ]

    Parameters
    ----------
    corners: np.ndarray[int]
        An array of corners.

    Returns
    -------
    np.ndarray[int]
        Corners sorted in anti-clockwise order.
    """
    # make sure corners are in the right order
    x: int = corners[:, 0]
    y: int = corners[:, 1]
    c: float = np.mean(corners, axis=0)
    x_c: float = x.copy() - c[0]
    y_c: float = y.copy() - c[1]

    r, phi = kartesian_to_polar(x_c, y_c)
    # scale down to make sure points are inside image
    r /= 2
    # sort
    o: np.ndarray[int] = np.argsort(phi)
    r, phi = r[o], phi[o]
    x, y = polar_to_kartesian(r, phi)
    corners[:, 0] = x + c[0]
    corners[:, 1] = y + c[1]
    return corners.astype(np.float32)


def find_stretching_function_basin(
        horizon_source: np.ndarray,
        horizon_target: np.ndarray,
        x00: Iterable | None = None,
        deg: int | None = None
) -> dict[str, Any]:
    """
    Find the stretching function for two horizons by optimizing.

    This function tries to find the global optimum for the stretching function
    by employing the basin hopping algorithm.

    Parameters
    ----------
    horizon_source : np.ndarray
        The vector to fit
    horizon_target : np.ndarray
        The vector to aim for.
    x00 : Iterable, optional
        Initial values for the parameters of the stretching function.
    deg : int, optional
        The degree of the polynomial. Either x00 or deg must be provided.

    Returns
    -------
    dict[str, Any]
        Output dict specifying the solution vector, inital and final error
        as well as the method.

    """
    def corr_with(vec: np.ndarray) -> np.ndarray:
        """Correlation of input vector with target."""
        vecm = vec - np.mean(vec)
        return target_scaled @ vecm / np.sqrt(np.sum(vecm ** 2))

    def evaluate_fit_pad(x0: Iterable) -> float:
        """Calculate error (negative correlation) between target and object."""
        # a, b, c, d, zeropad_l, zeropad_r = list(x0)
        # zeropad_l = (np.min([np.max([0, zeropad_l]), 1]) * width_pad_max).astype(int)
        # zeropad_r = (np.min([np.max([0, zeropad_r]), 1]) * width_pad_max).astype(int)
        vec_transformed, _ = apply_stretching(horizon_source, *x0)
        # vec_transformed = np.pad(vec_transformed, (zeropad_l, zeropad_r))
        return -corr_with(vec_transformed)

    def evaluate_fit_area(x0: Iterable) -> float:
        """Loss is the area between curves (L1 norm)"""
        vec_transformed, _ = apply_stretching(horizon_source, *x0)

        return np.sum(np.abs(vec_transformed - horizon_target) ** 2)

    assert len(horizon_source) == len(horizon_target), \
        'source and target must have the same length'
    assert (x00 is not None) or (deg is not None), \
        'provide either start values "x00" or the degree of the polynomial "deg"'

    if x00 is None:
        x00: np.ndarray[float] = np.zeros(deg)
        x00[-2] = 1

    # scaling to mean 0 and var 1
    # therefore this has to be calculated only once
    target_scaled = (
        (horizon_target - horizon_target.mean()).T
        / np.sqrt(np.sum((horizon_target - horizon_target.mean()) ** 2))
    )
    err00: float = evaluate_fit_pad(x00)
    logger.info(f'starting optimization with loss {err00:.3f}')
    params = basinhopping(
        # evaluate_fit_pad,
        evaluate_fit_area,
        x0=x00,
        # stepsize=.05,
        disp=False,
        minimizer_kwargs={'method': 'L-BFGS-B'}
    )

    logger.info(f'optimization successful: {params.success}')

    # get params
    x = params.x
    err: float = evaluate_fit_pad(x)
    logger.info(f'finished with correlation of {-err} and params {x=}')
    return {'x': x, 'err': err, 'err00': err00, 'method': 'basin_hopping'}


def find_stretching_function_transect(
        transect_source: np.ndarray,
        transect_target: np.ndarray,
        x00: Iterable | None = None,
        deg: int | None = None
) -> dict[str, Any]:
    """
    Find the stretching function for two horizons by optimizing.

    This function tries to find the global optimum for the stretching function
    by employing the basin hopping algorithm.

    Parameters
    ----------
    transect_source : np.ndarray
        The 2D array to fit
    transect_target : np.ndarray
        The 2D array to aim for.
    x00 : Iterable, optional
        Initial values for the parameters of the stretching function.
    deg : int, optional
        The degree of the polynomial. Either x00 or deg must be provided.

    Returns
    -------
    dict[str, Any]
        Output dict specifying the solution vector, inital and final error
        as well as the method.

    """

    def corr_with(vec: np.ndarray) -> np.ndarray:
        """Correlation of input vector with target."""
        vecm = vec - np.mean(vec)
        return target_scaled @ vecm / np.sqrt(np.sum(vecm ** 2))

    def evaluate_fit_area(x0: Iterable) -> float:
        """Loss is the area between curves"""
        transect_transformed, _ = apply_stretching(transect_source, *x0)

        return np.sum(np.abs(transect_transformed - transect_target)) / area

    assert transect_source.shape == transect_target.shape, \
        f'source and target must have the same shape but have shapes {transect_source.shape} and {transect_target.shape}'
    assert (x00 is not None) or (deg is not None), \
        'provide either start values "x00" or the degree of the polynomial "deg"'

    area: int = transect_target.shape[0] * transect_source.shape[1]

    if x00 is None:
        x00: np.ndarray = np.zeros(deg)
        x00[-2] = 1

    # scaling to mean 0 and var 1
    # therefore this has to be calculated only once
    target_scaled: np.ndarray = (
            (transect_target - transect_target.mean()).T
            / np.sqrt(np.sum((transect_target - transect_target.mean()) ** 2))
    )
    err00: float = evaluate_fit_area(x00)
    logger.info(f'starting optimization with loss {err00:.3f}')
    params = basinhopping(
        # evaluate_fit_pad,
        evaluate_fit_area,
        x0=x00,
        # stepsize=.05,
        disp=False,
        minimizer_kwargs={'method': 'L-BFGS-B'}
    )

    logger.info(f'optimization successful: {params.success}')

    # get params
    x = params.x
    err: float = evaluate_fit_area(x)
    logger.info(f'finished with correlation of {-err} and params {x=}')
    return {'x': x, 'err': err, 'err00': err00, 'method': 'basin_hopping'}


class Transformation:
    """
    Find transformation from source (image to be transformed) to
    destination (target to be matched).

    Multiple transformation types with multiple stages are supported. This
    object is still under development. Currently saving and loading parameters
    is not possible.

    Example Usage
    -------------
    >>> from msi_workflow import Transformation, ImageROI
    >>> ir1 = ImageROI.from_disk('some/path/to/ImageROI.pickle')
    >>> ir2 = ImageROI.from_disk('some/other/path/to/ImageROI.pickle')
    >>> t = Transformation(ir1, ir2, 'light', 'light')

    The first step is usally some coarser transformation, e.g.
    >>> t.estimate('bounding_box')
    or
    >>> t.estimate('punchholes')
    Results can be visualized
    >>> t.plot_fit()

    Also, a fine-tuning step can be added
    >>> t.estimate('image_flow')
    or
    >>> t.estimate('laminae')

    Images with the same shape as the source image can be fitted using
    >>> img = np.random.random(ir1.image.shape)
    >>> t.fit(img)
    """

    def __init__(
            self,
            source: np.ndarray[int | float] | ImageSample | ImageROI,
            target: np.ndarray[int | float] | ImageSample | ImageROI | None,
            source_obj_color: str | None = None,
            target_obj_color: str | None = None
    ) -> None:
        """
        Initializer.

        Parameters
        ----------
        source: np.ndarray[int | float] | ImageSample | ImageROI
            The source image (image to be transformed).
        target: np.ndarray[int | float] | ImageSample | ImageROI
            The target image (image that will be used as reference).
        source_obj_color: str, optional
            Color of the sample in the source image (light or dark).
        target_obj_color: str, optional
            Color of the sample in the target image (light or dark).

        """
        self._factors_rescaling: tuple[float, float] = 1, 1

        if target is not None:
            self.target: ImageROI = self._handle_input(
                target, target_obj_color, rescale=False
            )
            self.target_shape: tuple[int, ...] = self.target.image.shape
        else:
            self.target_shape = source.shape if isinstance(source, np.ndarray) else source.image.shape

        self.source: ImageROI = self._handle_input(
            source, source_obj_color, rescale=True
        )
        self.source_shape: tuple[int, ...] = self.source.image.shape

        self.trafos: list = []
        self.trafo_types: list = []

    def _handle_rescaling(
            self, image_roi: ImageROI, keep_ratio: bool, **kwargs
    ) -> ImageROI:
        """
        Match the shape of an image to the target and turn into _image_roi.

        Parameters
        ----------
        image_roi : ImageROI
            input object for which to match the shape
        keep_ratio : bool
            whether to keep the ratio of the input image

        Returns
        -------
        image_roi_new : ImageROI
            a new ImageROI instance with the desired image shape
        """
        input_h, input_w = image_roi.image.shape[:2]
        if keep_ratio:
            img_rescaled: np.ndarray = rescale_image_axis(
                image_roi.image, self.target.image, **kwargs)
        else:
            img_rescaled: np.ndarray = cv2.resize(
                image_roi.image,
                dsize=(self.target_shape[1], self.target_shape[0]),
                interpolation=cv2.INTER_AREA
            )

        image_roi_new: ImageROI = ImageROI(image=img_rescaled, obj_color=image_roi.obj_color)
        if hasattr(image_roi, "age_span"):
            image_roi_new.age_span = image_roi.age_span

        output_h, output_w = image_roi_new.image.shape[:2]
        # multiply unscaled input by this factor to account for rescaling
        self._factors_rescaling = output_h / input_h, output_w / input_w
        return image_roi_new

    def _handle_input(
            self,
            obj: np.ndarray[int | float] | ImageSample | ImageROI,
            obj_color: str | None,
            rescale: bool
    ) -> ImageROI:
        """Make sure inputs have an object color and initialize ImageROI."""
        if isinstance(obj, ImageROI):  # nothing to do
            pass
        elif isinstance(obj, str):  # file of image passed, create new obj
            obj: ImageSample = ImageSample(path_image_file=obj)
        elif isinstance(obj, np.ndarray):  # image passed, create new obj
            assert obj_color is not None, \
                'if an array is used as input, the obj_color must be specified'
            obj: ImageROI = ImageROI(image=obj, obj_color=obj_color)
        if type(obj) is ImageSample:  # create ImageROI obj from ImageSample
            obj: ImageROI = ImageROI(
                image=obj.image_sample_area,
                obj_color=obj.obj_color
            )

        # match horizontal scale
        if rescale:
            obj: ImageROI = self._handle_rescaling(obj, keep_ratio=False)

        return obj

    def _infer_side(
            self,
            points: list[np.ndarray[int], np.ndarray[int]],
            is_source: bool
    ) -> str:
        """
        Infer from the points coordinates and image shape if the punch-holes
        are positioned on top or bottom.

        Parameters
        ----------
        points : list[np.ndarray[int], np.ndarray[int]]
            Positions of the punch-holes.
        is_source : bool
            Whether to determine this for source or targe.

        Returns
        -------
        side: str
            Top or bottom, depending on the y-coordinates.
        """
        shape: tuple[int, ...] = self.source_shape if is_source else self.target_shape
        height: int = shape[0]
        points_y: list[float | int] = [p[0] for p in points]
        # points above middle
        if np.mean(points_y) > height / 2:
            return 'bottom'
        return 'top'

    def _get_classified(self, simplify: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Get the classified images of source and target."""
        target: ImageROI = self.target.copy()
        source: ImageROI = self.source.copy()
        if simplify:
            target.set_classification_adaptive_mean(
                remove_outliers=True,
                remove_small_areas=True,
            )
            source.set_classification_adaptive_mean(
                remove_outliers=True,
                remove_small_areas=True,
            )
        return target.image_classification, source.image_classification

    def get_transformed_source(self) -> ImageROI:
        source_image_new = self.fit()

        source_new: ImageROI = ImageROI(
            image=source_image_new, obj_color=self.source.obj_color
        )
        if hasattr(self.source, 'age_span'):
            source_new.age_span = self.source.age_span

        return source_new

    def _transform_from_punchholes(
            self,
            points_source: list[np.ndarray[int], np.ndarray[int]],  # xray
            points_target: list[np.ndarray[int], np.ndarray[int]],  # data
            is_piecewise: bool,
            contour_source: np.ndarray[int] = None,
            contour_target: np.ndarray[int] = None,
            is_rescaled: bool = False
    ) -> None:
        """
        Stretch and resize ROIs to match.

        Using the positions of the punch-holes (and the projected intersection
        with the contour) perform an affine (or piece-wise affine)
        transformation.

        Parameters
        ----------
        points_source: list[np.ndarray[int], np.ndarray[int]]
            Punch-holes in the source.
        points_target: list[np.ndarray[int], np.ndarray[int]]
            Punch-holes in the target.
        is_piecewise: bool
            Whether to use piece-wise affine or regular affine transformation.
            For the piece-wise affine transformation points defining the bounding
            box will be used in addition.
        contour_source: np.ndarray[int], optional
            The contour of the sample in the source image. If not provided,
            will be inferred.
        contour_target: np.ndarray[int],
            The contour of the sample in the source image. If not provided,
            will be inferred.
        is_rescaled: bool, optional
            Whether the input points and contours of the source are rescaled
            according to the input handling of the initializer. Default is False
        """
        if contour_source is None:
            contour_source: np.ndarray[int] = self.source.main_contour.copy()
        elif not is_rescaled:
            contour_source[:, 0, 1] = (
                    contour_source[:, 0, 1].astype(float)
                    * self._factors_rescaling[0]
            )
            contour_source[:, 0, 0] = (
                    contour_source[:, 0, 0].astype(float)
                    * self._factors_rescaling[1]
            )
        if contour_target is None:
            contour_target: np.ndarray[int] = self.target.main_contour

        if not is_rescaled:
            points_source: list[np.ndarray[int]] = [
                np.array((
                    p[0] * self._factors_rescaling[0],
                    p[1] * self._factors_rescaling[1]
                )) for p in points_source
            ]

        side_source: str = self._infer_side(points_source, True)
        side_target: str = self._infer_side(points_target, False)

        if side_source != side_target:
            logger.info(
                f'punch-holes are located at {side_source} for source and ' +
                f'at {side_target} for target, flipping X-Ray'
            )
            side_source = side_target
            # define warping that flips to and bottom
            h: int = self.source_shape[0] - 1  # index starts at 0
            contour_source[:, 0, 1] = h - contour_source[:, 0, 1]
            # format [array([y, x]), array([y, x])]
            points_source: list[np.ndarray[int]] = [
                np.array(([h - p[0], p[1]])) for p in points_source
            ]

            self.trafos.append(np.flipud)
            self.trafo_types.append('flip_ud')

        # append points from line-contour intersection
        points_source.append(
            find_line_contour_intersect(
                contour=contour_source,
                holes=points_source,
                hole_side=side_source,
                image_shape=self.source_shape
            )
        )
        points_target.append(
            find_line_contour_intersect(
                contour=contour_target,
                holes=points_target,
                hole_side=side_target,
                image_shape=self.target_shape
            )
        )

        if is_piecewise:
            # piecewise affine
            # add corners of contour so that entire image is transformed
            points_source += find_corners_contour(contour_source)
            points_target += find_corners_contour(contour_target)

            pwc: PiecewiseAffineTransform = PiecewiseAffineTransform()
            # swap xy of points
            src: np.ndarray = np.array(points_source)[:, ::-1]
            dst: np.ndarray = np.array(points_target)[:, ::-1]
            # msi_workflow and dst swapped??
            pwc.estimate(dst, src)
            M: PiecewiseAffineTransform = pwc
            transform_type: Any = PiecewiseAffineTransform
        else:
            # perform affine transformation
            M: np.ndarray[float] = cv2.getAffineTransform(
                np.array(points_source).astype(np.float32)[:, ::-1],
                np.array(points_target).astype(np.float32)[:, ::-1]
            )  # source: xray, target: msi
            transform_type = cv2.getAffineTransform
        self.trafos.append(M)
        self.trafo_types.append(transform_type)

    def _transform_from_bounding_box(
            self, plts: bool = False
    ) -> None:
        """Projective fit between corners of bounding boxes."""

        def get_points(obj: ImageROI) -> np.ndarray[int]:
            """Retrieve points defining the bounding box (rotated rect)."""
            rect: tuple[Sequence[float], Sequence[int], float] = \
                cv2.minAreaRect(obj.main_contour)
            points: np.ndarray[float] = cv2.boxPoints(rect)
            # sort point anticlockwise
            points = sort_corners(points)
            return points

        points_source: np.ndarray[np.float32] = get_points(self.source)
        points_target: np.ndarray[np.float32] = get_points(self.target)

        M: np.ndarray[float] = cv2.getPerspectiveTransform(
            points_source, points_target
        )

        if plts:
            plt_contours([points_source], self.source.image)
            plt_contours([points_target], self.target.image)

        self.trafos.append(M)
        self.trafo_types.append('perspective')

    def _transform_from_tilt_correct(
            self,
            plts: bool = False,
            nx_pixels_downscaled: int = 512,
            **kwargs
    ):
        source_new: ImageROI = self.get_transformed_source()
        source_image: np.ndarray = source_new.image

        downscale_factor: float = min((
            nx_pixels_downscaled / source_image.shape[1],
            1
        ))
        downscaled_shape = (
            round(source_image.shape[0] * downscale_factor),
            round(source_image.shape[1] * downscale_factor)
        )
        image_downscaled: np.ndarray = skimage.transform.resize(
            source_image, downscaled_shape
        )
        logger.info(
            f'initializing descriptor with image of shape '
            f'{image_downscaled.shape} (instead of {source_image.shape})'
        )
        d = Descriptor(image=image_downscaled, **kwargs)

        d.set_conv()
        d.fit(**kwargs)

        def apply_tilt_correction(image: np.ndarray) -> np.ndarray:
            return d.transform(image)

        self.trafos.append(apply_tilt_correction)
        self.trafo_types.append('tilt_correction')

        if plts:
            d.plot_kernels()
            d.plot_kernel_on_img()
            d.plot_parameter_images()
            d.plot_quiver()
            d.plot_corrected()

    def _transform_from_image_flow(
            self,
            plts: bool = False,
            use_classified: bool = False,
            **kwargs
    ) -> None:
        """
        Transform source to target with dense image flow.

        Parameters
        ----------
        plts: bool, optional
            Plot the flow field and transformed image. Default is False.
        use_classified: bool, optional
            Whether to use the classified or grayscale images to estimate the
            flow. The default is to use grayscale.
        kwargs: dict
            Optional keyword arguments for skimage.registration.optical_flow_tvl1
        """
        if use_classified:
            source, target = self._get_classified(**kwargs)
            source = source.astype(float)
            target = target.astype(float)
        else:
            source = self.source.image_grayscale.copy().astype(float)
            target = self.target.image_grayscale.copy().astype(float)

        # normalize images to 1
        source /= source.max()
        target /= target.max()

        logger.info('calculating dense flow ...')
        u, v = skimage.registration.optical_flow_tvl1(target, source)

        nr, nc, *_ = target.shape

        row_coords, col_coords = np.meshgrid(
            np.arange(nr),
            np.arange(nc),
            indexing='ij'
        )

        def apply_flow_shift(image: np.ndarray) -> np.ndarray:
            """Closure for applying flow field to arbitrary images."""
            image_warped = skimage.transform.warp(
                image, np.array([row_coords + v, col_coords + u]),
                mode='edge'
            )
            return image_warped

        self.trafos.append(apply_flow_shift)
        self.trafo_types.append('image_flow')

        if plts:
            self.plt_flow(source, target, u, v)

    def _transform_from_laminae(
            self,
            plts: bool = False,
            degree: int = 5,
            **kwargs
    ) -> None:
        """
        Fine-tuning step to match laminae by finding stretching function along
        multiple transects.

        Parameters
        ----------
        n_transects: int, optional
            Number of transects on which the stretching function will be
            evaluated. Defaults to 5.
        plts: bool, optional
            If True, will plot the deformed images and stretching functions.
            Default is False.
        deg: int, optional
            Degree of the stretching function. The default is 5.
        local: bool, optional
            Whether to perform local or global optimization. The default is True.
        """
        def calc_light_dark_ratio(img: np.ndarray) -> np.ndarray[float]:
            """Takes an image, turns it into a time series"""
            lights: np.ndarray[float] = (img == key_light_pixels).sum(axis=0).astype(float)
            darks: np.ndarray[float] = (img == key_dark_pixels).sum(axis=0).astype(float)
            holes: np.ndarray[float] = (img == key_hole_pixels).sum(axis=0).astype(float)
            both: np.ndarray[float] = lights + darks
            ratio: np.ndarray[float] = np.divide(
                lights,
                np.ones_like(lights) * img.shape[0],
                out=np.zeros_like(lights),
                where=(lights != 0)
            )

            # ratio = np.around(ratio, 0)
            return ratio

        def pop_params(params):
            if isinstance(transect_params, dict):
                return params['x']
            else:
                return params

        def calc_shift_vectors(transect_params: dict):
            """Calculate shift vectors from transect_prams."""
            logger.info("Calculating shift vectors...")

            x_dim: int = target_ic.shape[1]
            xs: np.ndarray = np.arange(x_dim)
            _, x_warped = apply_stretching(xs.copy(), *pop_params(transect_params))
            # reshape
            u = np.ones(target_ic.shape[0])[:, None] * (x_warped - xs)[None, :]
            v: np.ndarray = np.zeros_like(u)

            return u, v

        def laminae_flow(source: np.ndarray) -> np.ndarray:
            """
            Cubic interpolation between transects.

            This function is used to save the scope for transforming future images.
            """
            nr, nc, *_ = target_ic.shape

            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                                 indexing='ij')

            image_warped: np.ndarray = skimage.transform.warp(
                source, np.array([row_coords + v, col_coords + u]),
                mode='edge'
            )

            return image_warped

        # new source image
        source_new: ImageROI = self.get_transformed_source()

        # use classified images for brightness function
        target: ImageROI = self.target.copy()
        target.set_classification_adaptive_mean()
        source_new.set_classification_adaptive_mean()

        target_ic: np.ndarray = target.image_classification
        source_ic: np.ndarray = source_new.image_classification

        target_ld = calc_light_dark_ratio(target_ic)
        source_ld = calc_light_dark_ratio(source_ic)
        transect_params: dict = find_stretching_function_basin(
            target_ld, source_ld, deg=degree
        )
        u, v = calc_shift_vectors(transect_params)

        self.trafos.append(laminae_flow)
        self.trafo_types.append('laminae')

        if plts:
            self.plt_flow(
                self.source.image_grayscale, self.target.image_grayscale, u, v
            )

            fig, axs_ = plt.subplots(
                nrows=1,
                ncols=2,
                sharex=True,
                layout='constrained',
                figsize=(7, 3 * 1)
            )
            fig.suptitle(
                r'$f=\sum_i^n a_i x^i$ with $n=$' + f' {degree}'
            )

            params = transect_params
            source_horizons = calc_light_dark_ratio(target_ic)
            target_horizons = calc_light_dark_ratio(source_ic)
            source_horizons_warped, x_warped = apply_stretching(
                source_horizons, *pop_params(params)
            )
            x0 = pop_params(params)

            # plot showing the average number of light pixels per column before and
            #   after transformation
            axs_[0].plot(target_horizons, label='target', color='blue')
            axs_[0].plot(
                source_horizons,
                label='source',
                color='orange',
                linestyle='--'
            )
            axs_[0].plot(
                source_horizons_warped,
                label='warped source',
                color='red',
                linestyle='--'
            )
            if type(params) is list:
                axs_[0].set_title(
                    f' loss={params["err"]:.3f} (loss0={params["err00"]:.3f})'
                )

            # plot showing the distortions
            x = np.arange(len(x_warped))
            dx = x_warped - x
            x_func = np.linspace(-.5, .5, len(x))
            y_func = stretching_func(*x0)(x_func)
            dy_func = np.polyder(stretching_func(*x0))(x_func)
            axs_[1].plot(x / x.max(), label='x')
            axs_[1].plot(x_warped / x_warped.max(), label='distorted x')
            axs_[1].plot(dx / np.abs(dx).max(),
                         label=f'deviation of distorted x from x (max={int(np.abs(dx).max())})')
            axs_[1].plot(y_func / y_func.max(), label='deformation function')
            axs_[1].plot(dy_func / np.nanmax(dy_func), label='derivative')
            axs_[1].plot(np.diff(x_warped) / np.diff(x),
                         label='stretching factors')

            axs_[1].grid(True)

            ax0 = axs_[0]
            ax1 = axs_[1]
            ax0.legend(bbox_to_anchor=(0, 0), loc='upper left')
            ax1.legend(bbox_to_anchor=(0, 0), loc='upper left')
            plt.show()

    def estimate(self, method: str, *args, **kwargs):
        """Estimate the specified transformation and add it to the stack."""
        methods: tuple[str, ...] = (
            'punchholes', 'bounding_box', 'image_flow', 'tilt', 'laminae'
        )
        assert method in methods, f'method must be in {methods}, not {method}'

        if method == 'punchholes':
            self._transform_from_punchholes(*args, **kwargs)
        elif method == 'bounding_box':
            self._transform_from_bounding_box(*args, **kwargs)
        elif method == 'image_flow':
            self._transform_from_image_flow(*args, **kwargs)
        elif method == 'tilt':
            self._transform_from_tilt_correct(*args, **kwargs)
        elif method == 'laminae':
            self._transform_from_laminae(*args, **kwargs)
        else:
            raise NotImplementedError()

    def fit(self, img: np.ndarray = None) -> np.ndarray:
        """Apply all transformations on the stack to an input image."""
        def apply_(img, M, transform_type):
            if transform_type == 'perspective':
                warped = cv2.warpPerspective(
                    img, M, dsize=(self.target_shape[1], self.target_shape[0])
                )
            elif transform_type == cv2.getAffineTransform:
                warped = cv2.warpAffine(
                    img, M, dsize=(self.target_shape[1], self.target_shape[0])
                )
            elif transform_type == PiecewiseAffineTransform:
                warped = warp(img, M, output_shape=self.target_shape[:2])
            elif transform_type in ('laminae', 'image_flow', 'flip_ud', 'tilt_correction'):
                warped = M(img)
            else:
                raise NotImplementedError(
                    f'Unknown transformation type {transform_type}'
                )
            return warped

        if img is None:
            img = self.source.image_grayscale
        img = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))

        # apply the appropriate fit
        for M, transform_type in zip(self.trafos, self.trafo_types):
            logger.info(f'applying fit: {transform_type}')
            img = apply_(img, M, transform_type)

        return img

    def to_mapper(
            self,
            image_shape: tuple[int, ...] | None = None,
            path_folder: str | None = None,
            tag: str | None = None
    ) -> Mapper:
        """Turn the transformations into a Mapper object."""
        assert (source_shape := self.source.image_grayscale.shape) == self.fit().shape, \
            'Mapper only defined for transformations that do not change the size'
        if tag is None:
            tag = '_AND_'.join(self.trafo_types)
        if image_shape is None:
            image_shape = source_shape
        mapper = Mapper(
            image_shape=image_shape,
            path_folder=path_folder,
            tag=tag
        )
        mapper.add_UV(trafo=self.fit, is_uint8=False)

        return mapper

    def plot_fit(
            self,
            img: np.ndarray | None = None,
            use_classified: bool = False,
            simplify: bool = True,
            hold: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """
        Plot the transformed image for the estimated transformations.

        Parameters
        ----------
        img : np.ndarray, optional
            The image to fit with the same shape as the source image.
            If not provided, the source image will be used.
        use_classified: bool, optional
            If True, will plot the classified images. The default is False.
        simplify: bool, optional
            If True, will use the simplified classified images. The default is
            False.
        hold: bool, optional
            If True, will return fig and axs of the plot. The default is False.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            if hold is set to True.
        """
        if use_classified:
            target, orig = self._get_classified(simplify=simplify)
            warped: np.ndarray = self.fit(orig)
        else:
            orig = self.source.image_grayscale
            warped: np.ndarray = self.fit(img)
            target = self.target.image_grayscale

        orig = rescale_values(orig, 0, 1)
        warped = rescale_values(warped, 0, 1)
        target = rescale_values(target, 0, 1)

        img3d = np.stack(
            [
                orig,
                orig / 2 + target / 2,
                target
            ], axis=-1
        )

        img3d_warped = np.stack(
            [
                warped,
                warped / 2 + target / 2,
                target
            ], axis=-1
        )

        img2d = np.abs(orig - target)

        img2d_warped = np.abs(warped - target)

        area = orig.shape[0] * orig.shape[1]

        fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')
        axs[0, 0].imshow(img3d, interpolation='none')
        axs[0, 0].set_axis_off()
        axs[0, 0].set_title(f'Before (teal=target)')

        axs[1, 0].imshow(img3d_warped, interpolation='none')
        axs[1, 0].set_axis_off()
        axs[1, 0].set_title(f'Warped (orange=source)')

        axs[0, 1].imshow(img2d, interpolation='none')
        axs[0, 1].set_axis_off()
        axs[0, 1].set_title(f'Average distance before ({img2d.sum() / area:.2%})')

        axs[1, 1].imshow(img2d_warped, interpolation='none')
        axs[1, 1].set_axis_off()
        axs[1, 1].set_title(f'Average distance after ({img2d_warped.sum() / area:.2%})')

        if not hold:
            plt.show()
        else:
            return fig, axs

    @staticmethod
    def plt_flow(
            source: np.ndarray,
            target: np.ndarray,
            u: np.ndarray,
            v: np.ndarray,
            nvec: int = 50
    ) -> None:
        """
        Plot the flow field for a source and target as well as the flow field.

        Parameters
        ----------
        source : np.ndarray
            The source image
        target : np.ndarray
            The target image
        u: np.ndarray
            x-component of flow field
        v: np.ndarray
            y-component of flow field
        nvec: int
            Maximum number vectors in any direction.
        """
        source = source.copy().astype(float) / source.max()
        target = target.copy().astype(float) / target.max()
        nr, nc, *_ = target.shape

        source_warped = apply_displacement(u, v, source)

        # build an RGB image with the unregistered sequence
        seq_im = np.zeros((nr, nc, 3))
        seq_im[..., 0] = source
        seq_im[..., 1] = target
        seq_im[..., 2] = target

        # build an RGB image with the registered sequence
        reg_im = np.zeros((nr, nc, 3))
        reg_im[..., 0] = source_warped
        reg_im[..., 1] = target
        reg_im[..., 2] = target

        # --- Show the result

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 6))

        ax0.imshow(seq_im)
        ax0.set_title("Before")
        ax0.set_axis_off()

        ax1.imshow(reg_im)
        ax1.set_title("After")
        ax1.set_axis_off()

        # --- Compute flow magnitude
        norm = np.sqrt(u ** 2 + v ** 2)

        # --- Quiver plot
        if nvec > 0:
            step = max(nr // nvec, nc // nvec)

            y, x = np.mgrid[:nr:step, :nc:step]
            u_ = u[::step, ::step] / norm[::step, ::step]
            v_ = v[::step, ::step] / norm[::step, ::step]

            ax2.imshow(norm)
            ax2.quiver(x, y, u_, v_, color='r', units='dots',
                       angles='xy', scale_units='xy', lw=3)

        ax2.set_title("Optical flow magnitudes and directions")
        ax2.set_axis_off()

        fig.tight_layout()

        plt.show()
