"""Feature descriptor based on rectangular filters."""
from functools import cached_property

import matplotlib
import numpy as np
import logging

from typing import Callable, Self
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.filters import rank
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from scipy.signal import fftconvolve
from scipy.interpolate import griddata, CubicSpline

from maspim.imaging.register.helpers import Mapper, apply_displacement
from maspim.imaging.util.coordinate_transformations import rescale_values
from maspim.imaging.util.image_convert_types import ensure_image_is_gray
from maspim.util.convenience import check_attr

logger = logging.getLogger(__name__)


def gabor(nx, width, theta, psi, sigma=.5, gamma=.5):
    """
    https://en.wikipedia.org/wiki/Gabor_filter

    In this equation,
    lambda represents the wavelength of the sinusoidal factor,
    theta represents the orientation of the normal to the parallel stripes
    of a Gabor function, psi is the phase offset, sigma is the sigma/standard
    deviation of the Gaussian envelope and gamma is the spatial aspect ratio,
    and specifies the ellipticity of the support of the Gabor function.
    """
    lam = 4 * width / nx
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, y)
    # rotate
    Xp = X * np.cos(theta) + Y * np.sin(theta)
    Yp = -X * np.sin(theta) + Y * np.cos(theta)
    # decay
    frac = (Xp ** 2 + (gamma * Yp) ** 2) / (2 * sigma ** 2)
    gaussian = np.exp(-frac)
    # oscillation
    phase = 2 * np.pi * Xp / lam + psi
    osz = np.cos(phase)

    g = gaussian * osz
    return g


def rect(nx_rect: int, nx: int, phase: float) -> np.ndarray[float]:
    """
    Rectangular function with length nx_rect.

    The function returns a vector of length nx, where nx_rect points are
    part of the 'wave'. The phase controls the position of the positive hump.
    A phase of 0 corresponds to a wave being 1 in the first half and -1 in the
    second half (so, corresponds to a blocky sin wave).

                   __________
                  |          |
                  |          |
        __________|          |           __________
                             |          |
                             |          |
                             |__________|

    Parameters
    ----------
    nx_rect : int
        Number of points being either -1 or 1 (nx_rect corresponds to the length
        of the rectangular wave).
    nx: int
        Length of the return vector. Should be greater than or equal to nx_rect.
    phase : float
        The phase of the wave
    """
    assert nx_rect <= nx, \
        f"nx_rect must be smaller than nx (got {nx_rect} and {nx})"
    # phase = phase % (2 * np.pi)
    x_rect: np.ndarray[float] = (np.linspace(
        0, 2*np.pi, nx_rect
    ) - phase) % (2 * np.pi)
    # fill everything with -1 to begin with
    y_rect: np.ndarray[float] = np.full_like(x_rect, -1)
    # positive phase
    y_rect[x_rect < np.pi] = 1

    # initialize out vector
    y: np.ndarray[float] = np.zeros(nx)
    start_idx: int = round(nx / 2 - nx_rect / 2)
    # fill with
    y[start_idx:start_idx + nx_rect] = y_rect
    return y


def get_mask(image: np.ndarray) -> np.ndarray[bool]:
    """Return mask with foreground pixels."""
    image = ensure_image_is_gray(image)
    thr = threshold_otsu(image)
    return image > thr


class Descriptor:
    """
    Class for describing an image in terms of its angle, phase and wavelength
    by using structure kernels.

    Example Usage
    -------------
    Initialization
    >>> from maspim import Descriptor
    >>> import skimage
    >>> image = skimage.data.brick()
    A mask is not suitable for this problem
    >>> mask = np.ones(image.shape, dtype=bool)
    It is advisable to increase the number kernels to increase the resolution
    >>> descriptor = Descriptor(image=image, mask=mask, n_sizes=16, n_phases=16, n_angles=16)
    Estimate filer with best fitting angle, phase and width for each pixel
    >>> descriptor.set_conv_chunk()
    Visualize the resutls
    >>> descriptor.plot_parameter_images()
    Set the tilt corrections
    >>> descriptor.fit()
    And view the result
    >>> descriptor.plot_corrected()
    """
    # inputs
    kernel_types: list[str] = ['rect', 'gabor']
    kernel_type: str | None = None

    image: np.ndarray | None = None
    mask: np.ndarray[bool] | None = None

    n_angles: int | None = None
    angles: np.ndarray[float] | None = None

    max_period: int | None = None
    min_period: int | None = None
    n_sizes: int | None = None
    widths: np.ndarray[float] | None = None

    n_phases: int | None = None
    phases: np.ndarray[float] | None = None

    nx: int | None = None
    pad: int | None = None

    # inbetween results
    vals: np.ndarray[float] | None = None
    image_angles: np.ndarray[float] | None = None
    image_phases: np.ndarray[float] | None = None
    image_widths: np.ndarray[float] | None = None

    _stream_lines: list[matplotlib.path.Path] | None = None
    _x_seeds: np.ndarray[float]

    # results
    _points_shift: np.ndarray[float] | None = None
    _points_shift_inverse: np.ndarray[float] | None = None
    _shifts: np.ndarray[float] | None = None
    _shifts_inverse: np.ndarray[float] | None = None


    def __init__(
            self,
            image: np.ndarray,
            mask: np.ndarray | None = None,
            use_mask: bool = False,
            n_angles: int = 32,
            n_sizes: int = 8,
            n_phases: int = 8,
            max_period: int | float = None,
            min_period: int | float = None,
            kernel_type: str = 'rect',
            **_
    ):
        """
        Initialization

        Parameters
        ----------
        image : np.ndarray
            Input image (Multi- or single-channel).
        mask : np.ndarray, optional
            Mask with same shape as image where foreground pixels are > 0.
        use_mask: bool, optional
            Whether to use a mask to exclude background pixels. The default is
            False.
        n_angles : int, optional
            The number of angles for which to create kernels (angles will be
            evenly distributed in [0, pi)). The default is 8
        n_sizes : int, optional
            The number of widths for which to create kernels (will be evenly
            distributed between 5 and max_period). The default is 8.
        n_phases : int, optional
            The number of phases for which to create kernels (evenly distributed
            between [0, 2 * pi)).
        min_period: int | float, optional
            The size of the smallest kernel. If this value is <= 1, this value
            is assumed to be in terms of the image dimension, so for example
            a value of 0.01 (which is the default behavior) results in a min_period
            of round(0.01 * N) for an N x M image with N < M. If this value is
            not provided, it will be taken to be 1/10 the max_period, but at
            least 5.
        max_period: int | float, optional
            The size of the largest kernel. If this value is <= 1, this value
            is assumed to be in terms of the image dimension, so for example
            a value of 0.1 (which is the default behavior) results in a max_period
            of round(0.1 * N) for an N x M image with N < M.
        kernel_type : str, optional
            The type of kernel to be used. Currently, options are 'rect' (default)
             and 'gabor'.
        """
        if (mask is not None) and (use_mask is False):
            logger.warning(
                'a mask was provided, but use_mask is set to False. This will '
                'ignore the mask. In order to use the mask, set use_mask=True'
            )

        if mask is None:
            if use_mask:
                mask: np.ndarray = get_mask(image)
            else:
                mask: np.ndarray = np.ones(image.shape[:2], dtype=bool)

        assert mask.shape[:2] == image.shape[:2], \
            "Mask and image must have the same shape along dim 1 and 2"
        assert kernel_type in self.kernel_types, \
            f'Unknown kernel type: {kernel_type}, must be one in {self.kernel_types}'
        assert np.min(image.shape[:2]) >= 5, \
            'image must have at least 5 pixels in each direction'

        if max_period is None:
            max_period: float = .1
        if max_period < 1:
            max_period: int = round(np.min(image.shape[:2]) * max_period)
        # make sure this value is at least 5
        self.max_period: int = max([5, max_period])

        if min_period is None:
            min_period: float = round(max_period / 10)
        elif min_period < 1:
            min_period: int = max([5, round(np.min(image.shape[:2]) * min_period)])
        # make sure this value is at least 5
        self.min_period: int = max([5, min_period])

        assert min_period <= max_period

        self.kernel_type: str = kernel_type
        self.image: np.ndarray = image
        self.mask: np.ndarray[bool] = mask > 0
        self.n_angles: int = n_angles
        self.n_sizes: int = n_sizes
        self.n_phases: int = n_phases
        # account for rotating square out of original footprint
        self.nx: int = np.ceil(self.max_period * np.sqrt(2)).astype(int)
        self.pad = self.nx - self.max_period

        if self.n_sizes > 1:
            self.widths: np.ndarray[int] = np.linspace(
                self.min_period, self.max_period, self.n_sizes, dtype=int
            )
        else:
            self.widths: np.ndarray[int] = np.array(
                [self.max_period], dtype=int
            )
        # equally spaced between 0 and 180 degrees
        self.angles: np.ndarray[float] = np.linspace(
            0, np.pi, self.n_angles, endpoint=False
        )
        self.phases: np.ndarray[float] = np.linspace(
            0, 2 * np.pi, self.n_phases, endpoint=False
        )

    @classmethod
    def from_descriptor(cls, other: Self, **kwargs) -> Self:
        """
        Alternative constructor to bypass setting the parameter images.

        This is the recommended way for working downstream with tilt corrected
        images.
        """
        assert check_attr(other, '_shifts'), 'call fit on descriptor first'
        # since this is constructed from angle corrected image, we are only
        # interested in one angle
        kwargs['n_angles'] = kwargs.get('n_angles', 1)
        kwargs['n_sizes'] = kwargs.get('n_sizes', 16)
        kwargs['n_phases'] = kwargs.get('n_phases', 16)

        image_corrected = other.transform(other.image)
        new = cls(image=image_corrected, **kwargs)
        return new

    @cached_property
    def image_processed(self) -> np.ndarray[float]:
        """
        Preprocess image to be between -1 and 1 and set background pixels to 0.
        """
        # single channel
        image: np.ndarray = ensure_image_is_gray(self.image)
        foreground: np.ndarray[bool] = self.mask
        image: np.ndarray[float] = image.astype(float)
        # shift to min = 0
        image -= image[foreground].min()
        # equalize
        # image *= 255 / image[foreground].max()
        # image = image.astype(np.uint8)
        # image = skimage.exposure.equalize_hist(image, mask=self.mask).astype(float)
        # set max to 2
        image *= 2 / image.max()
        # range -1 to 1
        image -= 1
        image[~foreground] = image[foreground].mean()
        # image *= self._get_taper()
        return image

    def _profile_to_kernel(
            self, profile: np.ndarray, angle: float
    ) -> np.ndarray[float]:
        """
        Turn a profile into a kernel.

        Kernels are scaled to zero-mean and unit volume.

        Parameters
        ----------
        profile: np.ndarray
            1D array specifying the amplitudes of the kernel.
        angle: float
            rotation of the kernel in rad counter-clockwise

        Returns
        -------
        np.ndarray
            2D array extending the profile perpendicular to the vector pointing
            in the direction of the angle.
        """
        # pad profile such that footprint stays within square when rotated
        top_bot_pad: np.ndarray[float] = np.zeros_like(profile)
        top_bot_pad[self.pad // 2:-self.pad // 2] = 1
        footprint: np.ndarray[float] = (
                np.ones((self.nx, self.nx)) *
                profile[None, :] *
                top_bot_pad[:, None]
        )
        kernel: np.ndarray[float] = rotate(footprint, -angle * 180 / np.pi)
        # zero mean
        mask = kernel != 0
        mean = kernel[mask].mean()
        kernel[mask] -= mean
        # unit variance
        k = np.sqrt(np.abs(kernel ** 2).sum())
        if k > 0:
            kernel /= k
        return kernel

    def _get_kernel_rect(
            self, width: int, angle: float, phase: float
    ) -> np.ndarray[float]:
        """Get kernel with specified width, angle and phase."""
        profile: np.ndarray[float] = rect(width, self.nx, phase)
        return self._profile_to_kernel(profile, angle)

    def _get_kernel_gabor(
            self, width: int, angle: float, phase: float
    ) -> np.ndarray[float]:
        kernel: np.ndarray[float] = gabor(
            nx=self.nx, width=width, theta=angle, psi=phase
        )
        # unit vol
        k = np.sqrt(np.abs(kernel ** 2).sum())
        kernel /= k
        return kernel

    @property
    def get_kernel(self) -> Callable:
        """Convenience function for fetching the right kernel function."""
        if self.kernel_type == 'gabor':
            return self._get_kernel_gabor
        elif self.kernel_type == 'rect':
            return self._get_kernel_rect
        else:
            raise NotImplementedError

    def set_conv(self):
        """
        Find ideal parameters for each pixel by convolving image with all kernels.
        """
        shape: tuple[int, int] = self.image_processed.shape
        self.vals: np.ndarray[float] = np.zeros(shape, dtype=float)
        self.image_widths: np.ndarray[int] = np.zeros(shape, dtype=int)
        self.image_angles: np.ndarray[float] = np.zeros(shape, dtype=float)
        self.image_phases: np.ndarray[float] = np.zeros(shape, dtype=float)

        for width in tqdm(self.widths, desc='searching parameters', total=self.n_sizes):
            for phase in self.phases:
                for angle in self.angles:
                    kernel: np.ndarray[float] = self.get_kernel(
                        width=width, phase=phase, angle=angle
                    )
                    conv: np.ndarray[float] = fftconvolve(
                        self.image_processed,
                        kernel[::-1, ::-1],
                        mode='same',
                        axes=(0, 1)
                    )

                    # update res and idxs
                    mask: np.ndarray[bool] = conv > self.vals

                    self.vals[mask] = conv[mask]
                    self.image_widths[mask] = width
                    self.image_angles[mask] = angle
                    self.image_phases[mask] = phase

        # shift to be between -pi and pi
        self.image_angles[self.image_angles > np.pi / 2] -= np.pi

    def set_conv_chunk(self):
        """
        Same as set_conv, but stacking kernels with the same width and phase.
        Not much faster than set_conv (~25 %).
        """
        if self.n_angles == 1:
            logger.info('Only one angle, calling set_conv implicitly')
            self.set_conv()
            return

        shape: tuple[int, int] = self.image_processed.shape
        self.vals: np.ndarray[float] = np.zeros(shape, dtype=float)
        self.image_widths: np.ndarray[int] = np.zeros(shape, dtype=int)
        self.image_angles: np.ndarray[float] = np.zeros(shape, dtype=float)
        self.image_phases: np.ndarray[float] = np.zeros(shape, dtype=float)

        d_angle: float = np.diff(self.angles)[0]
        image_stack: np.ndarray[float] = np.repeat(
            self.image_processed[:, :, np.newaxis], self.n_angles, axis=-1
        )
        for width in tqdm(self.widths, desc='searching parameters', total=self.n_sizes):
            for phase in self.phases:
                kernels = np.dstack([self.get_kernel(
                    width=width, phase=phase, angle=angle) for angle in self.angles
                ])
                conv: np.ndarray[float] = fftconvolve(
                    image_stack,
                    kernels[::-1, ::-1, :],
                    mode='same',
                    axes=(0, 1)
                )

                # update res and idxs
                idxs: np.ndarray[int] = np.argmax(conv, axis=2)
                # use d_angle and indices to calculate angle
                angles: np.ndarray[float] = idxs * d_angle + self.angles[0]

                vals: np.ndarray[float] = np.max(conv, axis=2)
                # only update if value is higher than previous
                mask: np.ndarray[bool] = vals > self.vals

                self.vals[mask] = vals[mask]
                self.image_widths[mask] = width
                self.image_angles[mask] = angles[mask]
                self.image_phases[mask] = phase

        # shift to be between -pi and pi
        self.image_angles[self.image_angles > np.pi / 2] -= np.pi

    def _get_legal_angle_mask(
            self,
            angle_min: float = -np.pi / 4,
            angle_max: float = np.pi / 4,
            mute_bounds: float = 0,
            **_
    ) -> np.ndarray[bool]:
        """
        Set angles outside bounds and in mask to 0.

        Parameters
        ----------
        angle_min: float, optional
            The minimum allowed angle. Defaults to -pi / 4 (-45 degrees)
        angle_max: float, optional
            The maximum allowed angle. Defaults to pi / 4 (45 degrees)
        mute_bounds: stripes at top and bottom set to 0. Defaults to 0.

        Returns
        -------
        mask_valid: np.ndarray[bool]
            Mask where angles fulfilling all criteria are set to True.
        """
        assert (mute_bounds < .5) and (mute_bounds >= 0), \
            f'mute_bounds must be between 0 and 1/2, got {mute_bounds}'

        image_angles: np.ndarray[float] = self.image_angles.copy()
        mask_valid: np.ndarray[bool] = (
            (image_angles >= angle_min)
            & (image_angles <= angle_max)
            & self.mask
        )

        if mute_bounds > 0:
            height: int = image_angles.shape[0]
            muted_bounds: np.ndarray[float] = np.ones((height, 1))
            idx_mute: int = round(height * mute_bounds)
            muted_bounds[:idx_mute] = 0
            muted_bounds[height - idx_mute:] = 0

            mask_valid: np.ndarray[bool] = (mask_valid * muted_bounds).astype(bool)
        return mask_valid

    def _get_filtered_angles(
            self,
            n_footprint: int = 1,
            interpolate: bool = False,
            plts: bool = False,
            **kwargs
    ) -> np.ndarray[float]:
        """
        Smooth and/ or interpolate angles.

        Parameters
        ----------
        n_footprint: int, optional
            The size of the footprint of the modal filter. The default is 1
            which results in not applying the modal filter.
        interpolate: bool, optional
            Whether to interpolate invalid values. The default is False. If
            interpolate is set to False, invalid pixels will be set to 0.
        plts: bool, optional
            Whether to plot the angle image before and after processing.
        kwargs: Number, optional
            keywords for _get_legal_angle_mask
        """
        image_angles: np.ndarray[float] = self.image_angles.copy()
        image_angles[image_angles > np.pi / 2] -= np.pi
        mask_invalid: np.ndarray[bool] = ~self._get_legal_angle_mask(**kwargs)

        if n_footprint != 1:
            # make sure images are between 0 and 180 degrees
            image_angles[image_angles < 0] += np.pi
            # rank filters need integer images, 256 bins should be enough to cover
            # angles from 0 to 180 degrees with enough precision
            angles_new: np.ndarray[np.uint8] = np.round(
                np.rad2deg(image_angles)
            ).astype(np.uint8)

            angles_new: np.ndarray[np.uint8] = np.deg2rad(rank.modal(
                angles_new,
                footprint=np.ones((n_footprint, n_footprint)),
                mask=~mask_invalid
            ))
            # overwrite invalid values
            angles_new[mask_invalid] = image_angles[mask_invalid]
            # convert images back to -90 to 90 degrees
            angles_new[angles_new > np.pi / 2] -= np.pi
        else:
            angles_new = image_angles

        if interpolate:
            y, x = np.indices(angles_new.shape)

            # Get the coordinates of the known data points
            known_points: np.ndarray[int] = np.array(
                (y[~mask_invalid], x[~mask_invalid])
            ).T
            known_values: np.ndarray[int] = angles_new[~mask_invalid]

            # Get the coordinates of the missing data points
            missing_points: np.ndarray[int] = np.array(
                (y[mask_invalid], x[mask_invalid])
            ).T

            # Perform the interpolation
            interpolated_values: np.ndarray[float] = griddata(
                known_points,
                known_values,
                missing_points,
                method='linear',
                fill_value=0  # fill nan values with 0
            )

            # Fill the missing values with the interpolated data
            angles_new[mask_invalid] = interpolated_values
        else:
            angles_new[mask_invalid] = 0

        if plts:
            plt.imshow(self.image_angles)
            plt.title('Angles before filtering')
            plt.show()

            plt.imshow(angles_new)
            plt.title('Angles after filtering')
            plt.show()

        return angles_new

    def _get_sparse_angles(
            self,
            ny_cells: int = 10,
            plts: bool = False,
            **kwargs
    ):
        """
        Compute the weighted angles in cells.

        Angles will be binned into cells on a primary and dual grid (dual grid
        is shifted by 1/2 cell-size in x and y direction). The weighted mean
        angles and positions are calculated for each cell. Subsequently, values
        of angles are interpolated between weighted means. Overall this has the
        effect of smoothing the angles. The parameter ny_cells controls the
        area of effect of the smoothing: smaller values mean that the image will
        be divided into fewer, bigger cells. Hence, the smoothing effect will
        be bigger. This algorithm is inspired by 'Histogram of Gradients'.

        Parameters
        ----------
        ny_cells: int, optional
            The number of cells in the y-direction in the primary grid. The total
            number (taking the dual grid into account) is 2 * ny_cells - 1.
        plts: bool, optional
            If True, will plot the weights and interpolated angles. The default
            is False.
        """
        def weighted_sum(indices) -> tuple[float, float, float]:
            weights_sum: float = weights[indices].sum()
            if weights_sum == 0:
                return 0., 0., 0.
            rw_: float = weighted_values[indices].sum() / weights_sum
            xw_: float = weighted_x[indices].sum() / weights_sum
            yw_: float = weighted_y[indices].sum() / weights_sum
            return rw_, xw_, yw_

        values = self._get_filtered_angles(plts=plts, **kwargs)
        if ny_cells == 1:
            return values

        mask_valid: np.ndarray[bool] = self._get_legal_angle_mask(**kwargs)
        # use the values of the convolution with kernels as a proxy for the
        # quality of the lamination
        # scale to be between 0 and 1
        weights = self.vals.copy()
        weights -= weights.min()
        weights /= weights.max()
        weights *= mask_valid.astype(float)

        # coordinates of grid nodes
        yy, xx = np.indices(values.shape)

        # precompute weighted qualities and coordinates
        weighted_values: np.ndarray[float] = values * weights
        weighted_y: np.ndarray[float] = yy * weights
        weighted_x: np.ndarray[float] = xx * weights

        # iterate over cells
        h, w = values.shape
        # pixels per cell in x- and y-direction
        c: int = round(h / ny_cells)
        # number of pixels by which the dual grid is shifted
        dual_shift: int = round(c / 2)
        # number of cells in x-direction
        nx_cells: int = int(w / c)
        n_cells: tuple[int, int] = (ny_cells, nx_cells)
        n_cells_dual: tuple[int, int] = (ny_cells - 1, nx_cells - 1)

        # container for weighted values
        cell_values_primary: np.ndarray[float] = np.zeros(n_cells)
        ys: np.ndarray[float] = np.zeros(n_cells)
        xs: np.ndarray[float] = np.zeros(n_cells)

        cell_values_dual: np.ndarray[float] = np.zeros(n_cells_dual)
        ys_dual: np.ndarray[float] = np.zeros(n_cells_dual)
        xs_dual: np.ndarray[float] = np.zeros(n_cells_dual)

        # pick weighted mean
        for i in range(ny_cells):
            for j in range(nx_cells):
                idxs = np.index_exp[
                       i * c:(i + 1) * c,
                       j * c:(j + 1) * c
                ]
                cw, xw, yw = weighted_sum(idxs)
                cell_values_primary[i, j] = cw
                xs[i, j] = xw
                ys[i, j] = yw

                # dual grid is one smaller
                if (i == ny_cells - 1) or (j == nx_cells - 1):
                    continue
                idxs_dual = np.index_exp[
                            i * c + dual_shift:(i + 1) * c + dual_shift,
                            j * c + dual_shift:(j + 1) + dual_shift
                ]
                cw, xw, yw = weighted_sum(idxs_dual)
                cell_values_dual[i, j] = cw
                xs_dual[i, j] = xw
                ys_dual[i, j] = yw

        # Get the coordinates of the known data points
        known_points = np.array((
            np.concatenate((xs.ravel(), xs_dual.ravel())),
            np.concatenate((ys.ravel(), ys_dual.ravel())),
        )).T
        known_values = np.concatenate(
            (cell_values_primary.ravel(), cell_values_dual.ravel())
        )

        # Perform the interpolation
        interpolated_values = griddata(
            known_points,
            known_values,
            (xx, yy),
            method='linear',
            fill_value=0  # fill nan values with 0
        )

        if plts:
            plt.imshow(weights)
            plt.scatter(known_points[:, 0], known_points[:, 1], s=.1, c='r')
            plt.title('Quality')
            plt.show()

            plt.imshow(interpolated_values)
            plt.scatter(known_points[:, 0], known_points[:, 1], s=.1, c='r')
            plt.title('Interpolated angles')
            plt.show()

        return interpolated_values

    def _set_stream_lines(
            self, n_seeds: int = 100, **kwargs
    ) -> None:
        angles: np.ndarray[float] = self._get_sparse_angles(**kwargs)
        h, w = angles.shape
        # x and y components of flow field
        U: np.ndarray[float] = np.cos(angles)
        V: np.ndarray[float] = np.sin(angles)
        Y, X = np.indices(U.shape)
        x_seeds: np.ndarray[float] = np.linspace(0, w, n_seeds + 2)[1:-1]
        y_seeds: np.ndarray[float] = np.full_like(x_seeds, h // 2, dtype=int)
        seed_points: np.ndarray[float] = np.stack((x_seeds, y_seeds), axis=1)

        plt.imshow(self.image_processed)
        res: matplotlib.streamplot.StreamplotSet = plt.streamplot(
            X, Y, -V, U,  # use slopes perpendicular to angles for flow field
            broken_streamlines=False,
            start_points=seed_points
        )
        plt.close()
        # extract paths
        lines: matplotlib.collections.LineCollection = res.lines
        paths: list[matplotlib.path.Path] = lines.get_paths()

        self._x_seeds: np.ndarray[float] = x_seeds
        self._stream_lines: list[matplotlib.path.Path] = paths

    def fit(
            self,
            ny_cells: int = 10,
            **kwargs
    ):
        """
        Use angles to un-tilt the image.

        The angles define the tilt at each location. Now we would like to
        find a mapping that takes every pixel to a new location such that the
        resulting image is distortion free. The issue is that the tilts are
        not directly correlated to the distortions since deformations accumulate.

        This is done by integrating paths from the flow field at certain seed
        points distributed along the middle of the x-axis.

        Parameters
        ----------
        ny_cells: int, optional
            Number of cells in vertical direction used to pool angles. The
            actual number is about twice of this due to using staggered grids.
        """
        if not check_attr(self, 'vals', True):
            self.set_conv()

        h, w = self.image.shape[:2]
        # post-processed angles
        if not check_attr(self, 'x_seeds'):
            self._set_stream_lines(ny_cells=ny_cells, **kwargs)

        # construct array of shifts from nodes of streamlines
        n_nodes: int = max(ny_cells * 2, 10)
        shifts: np.ndarray[float] = np.zeros((n_nodes, len(self._x_seeds)))

        # stem points for interpolation
        y_ip: np.ndarray[float] = np.linspace(0, h, n_nodes)
        for i, streamline in tqdm(
                enumerate(self._stream_lines),
                total=len(self._stream_lines),
                desc='setting deformation field from streamlines'
        ):
            xs = streamline.vertices[:, 0]
            ys = streamline.vertices[:, 1]
            # drop duplicates
            ys, uidxs = np.unique(ys, return_index=True)
            xs = xs[uidxs]
            # interpolate
            cs: CubicSpline = CubicSpline(ys, xs)
            x_ip: np.ndarray = cs(y_ip)
            # store shifts relative to seed
            shifts[:, i] = x_ip - self._x_seeds[i]

        # grid of shift vectors
        X_shift, Y_shift = np.meshgrid(self._x_seeds, y_ip)

        # interpolate shift vectors between nodes
        points_shift: np.ndarray[float] = np.stack(
            (X_shift.ravel(), Y_shift.ravel()),
            axis=1
        )
        # placed on vertical lines, pointing at stream-lines
        self._points_shift: np.ndarray[float] = points_shift
        self._shifts: np.ndarray[float] = shifts
        # placed on stream-lines, pointing at vertical lines
        self._points_shift_inverse: np.ndarray[float] = np.stack(
            (X_shift.ravel() + shifts.ravel(), Y_shift.ravel()),
            axis=1
        )
        self._shifts_inverse: np.ndarray[float] = -shifts

    def _grid_upscale(
            self, m: np.ndarray, shape_out: tuple[int, ...], is_inverse: bool, **kwargs
    ) -> np.ndarray:
        """Interpolate grid values to match shape of input."""
        Y, X = np.indices(shape_out[:2])
        scale_factor: float = shape_out[0] / self.image.shape[0]

        assert np.abs(shape_out[0] / shape_out[1] - self.image.shape[0] / self.image.shape[1]) < .01, \
            (f'the image to be transformed should maintain roughly the same '
             f'aspect ratio, but found ratios {shape_out[0] / shape_out[1]:.2f} and '
             f'{self.image.shape[0] / self.image.shape[1]:.2f}, please use the same'
             f'downscaling factor along each axis.')

        points = self._points_shift_inverse.copy() if is_inverse else self._points_shift.copy()
        points *= scale_factor

        m_ip: np.ndarray[float] = griddata(
            points, m.ravel() * scale_factor, (X, Y), **kwargs
        )
        m_ip[np.isnan(m_ip)] = 0

        return m_ip

    def get_shift_matrix(self, shape: tuple[int, ...], **kwargs) -> np.ndarray:
        """Interpolate shift values to match shape of input."""
        return self._grid_upscale(self._shifts, shape, is_inverse=False, **kwargs)

    def _get_inverse_shift_matrix(self, shape: tuple[int, ...], **kwargs) -> np.ndarray:
        return self._grid_upscale(self._shifts, shape, is_inverse=True, **kwargs)

    def transform(
            self,
            image: np.ndarray | None = None,
            is_inverse: bool = False
    ) -> np.ndarray:
        """
        Apply the shift map to an image.

        fit needs to be called before this.

        Parameters
        ----------
        image : np.ndarray, optional
            An image to fit. Must have the same shape as the input image
            along the first two axes. Defaults to using the preprocessed image.

        Returns
        -------
        np.ndarray
            The warped (tilt corrected) image.
        """
        if image is None:
            image: np.ndarray = self.image_processed

        logger.info(
            f'transforming image with shape {image.shape} '
            f'and dtype {image.dtype}'
        )
        assert check_attr(self, '_shifts'), 'call fit first'

        if is_inverse:
            u = self._get_inverse_shift_matrix(image.shape[:2], method='cubic')
        else:
            u = self.get_shift_matrix(image.shape[:2], method='cubic')
        v = np.zeros_like(u)

        warped = apply_displacement(u, v, image, preserve_range=True)
        warped = warped.astype(image.dtype)

        return warped

    def fit_transform(
            self, image: np.ndarray | None = None, **kwargs
    ) -> np.ndarray:
        """Find and apply a transformation."""
        self.fit(**kwargs)
        return self.transform(image)

    def get_mapper(self, image_shape_original: tuple | None = None, **kwargs) -> Mapper:
        if image_shape_original is None:
            image_shape_original = self.image.shape[:2]
        mapper = Mapper(
            image_shape=image_shape_original,
            tag='tilt_correction',
            **kwargs
        )
        mapper.add_UV(U=self._get_inverse_shift_matrix(image_shape_original))

        return mapper

    def plot_kernels(self, n: int = 5):
        widths = self.widths[np.unique(np.around(
            np.linspace(0, self.n_sizes - 1, n)
        ).astype(int))]
        angles = self.angles[np.unique(np.around(
            np.linspace(0, self.n_angles - 1, n)
        ).astype(int))]
        phases = self.phases[np.unique(np.around(
            np.linspace(0, self.n_phases - 1, n)
        ).astype(int))]

        for width in widths:
            col_stack = []
            for phase in phases:
                row_stack = []
                for angle in angles:
                    row_stack.append(self.get_kernel(width, angle, phase))
                row = np.hstack(row_stack)
                col_stack.append(row)
            col = np.vstack(col_stack)
            plt.imshow(col, cmap='viridis')
            plt.show()

    def plot_kernel_on_img(self):
        img = self.image_processed.copy()
        img_k = img.copy()

        kernel = self.get_kernel(self.widths[0], 0, 0)
        kernel = rescale_values(kernel.astype(float), -1, 1, )

        img_k[:kernel.shape[0], :kernel.shape[1]] = kernel

        kernel = self.get_kernel(self.widths[-1], 0, 0)
        kernel = rescale_values(kernel.astype(float), -1, 1)
        img_k[-kernel.shape[0]:, -kernel.shape[1]:] = kernel

        plt.imshow(img_k)
        plt.title(
            'Input image with biggest and smallest kernel (rescaled for visibility)'
        )
        plt.show()

    def plot_parameter_images(self):
        if not check_attr(self, 'vals', True):
            self.set_conv()
        fix, ((tl, tr), (bl, br)) = plt.subplots(
            2, 2, sharex=True, sharey=True
        )

        tl.imshow(self.image_processed)
        tl.set_title("Processed input image")
        tr.imshow(self.image_widths)
        tr.set_title("Widths")
        bl.imshow(self.image_phases, cmap='hsv')
        bl.set_title("Phases")
        br.imshow(self.image_angles, cmap='hsv')
        br.set_title("Angles")
        plt.show()

    def plot_quiver(self, n_row: int = 50, u=None, v=None, fig=None, ax=None):
        h, w = self.image.shape[:2]
        if (u is None) or (v is None):
            u = np.cos(self.image_angles)
            v = np.sin(self.image_angles)

        x, y = np.meshgrid(np.arange(w), np.arange(h))

        every: int = round(h / n_row)
        idxs = np.index_exp[::every, ::every]

        if plt_res := (fig is None):
            fig, ax = plt.subplots()

        ax.imshow(self.image)
        ax.quiver(
            x[idxs],
            y[idxs],
            u[idxs],
            v[idxs],
            angles='xy',
            pivot='middle',
            color='white'
            # headaxislength=0, headlength=0
        )
        if plt_res:
            plt.show()
        else:
            return fig, ax

    def plot_corrected(self, image: np.ndarray | None = None):
        assert check_attr(self, '_shifts'), 'call fit first'
        if image is None:
            image = self.image_processed
        assert image.shape[:2] == self.image.shape[:2]

        h, w = self.image.shape[:2]

        fig, (ax1, ax2) = plt.subplots(nrows=2, layout='constrained')

        # plot streamlines
        ax1.imshow(image)
        for line in self._stream_lines:
            ax1.plot(
                line.vertices[:, 0],
                line.vertices[:, 1],
                linewidth=.15,
                color='k'
            )
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
        ax1.axis("off")
        ax1.set_title('Streamlines on input image')

        warped: np.ndarray = self.transform(image)
        ax2.imshow(warped)
        ax2.vlines(self._x_seeds, 0, h, colors='k', linewidth=.15)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)
        ax2.axis("off")
        ax2.set_title('Tilt corrected image')


def test_rect():
    phis = np.linspace(0, np.pi, 20)
    for phi in phis:
        plt.plot(rect(1000, 2000, phi), '--')
        plt.plot(rect(1000, 2000, phi + np.pi), '--')
        plt.title(f'phase: {phi * 180 / np.pi:.0f}')
        plt.show()
