import os
from typing import Callable, Self

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline, bisplev
from skimage.data import checkerboard
from skimage.transform import warp, resize
import logging

from maspim.util import Convenience
from maspim.util.convenience import check_attr

logger = logging.getLogger(__name__)


def get_transect_indices(i_transect: int, image_shape: tuple[int, ...], n_transects: int) -> slice:
    """
    Obtain indices of i-th transect.
    
    :param i_transect: Index of transect for which to obtain indices
    :param image_shape: shape of the image
    :param n_transects: number of transects
    
    Returns
    -------
    slice: indices of transects
        2d slice object
    """
    unit = image_shape[0] / n_transects
    indices: slice = slice(int(unit * i_transect), int(unit * (1 + i_transect)))
    return indices

def get_transect(image: np.ndarray, i_transect: int, n_transects):
    indices = get_transect_indices(i_transect, image.shape, n_transects)
    return image[indices, :]


def extrapolate_bispline_top_bottom(f: RectBivariateSpline, yp, xi, yi) -> np.ndarray:
    """
    Extrapolate values outside yp span of irregular grid points.
    
    RectBivariateSpline fills values outside the span with constant values. This function replaces 
    them with the linear interpolation.
    
    Parameters
    ----------
    f: RectBivariateSpline
        Initialized RectBivariateSpline object 
    yp: y-grid points at which z-values are known
    xi: x-grid points at which to interpolate
    yi: y-grid points at which to interpolate

    Returns
    -------
    Z: interpolated and extrapolated z-values
    """
    # xp, yp: points spanning the irregular grid for which z values exist
    # xi, yi: points spanning the regular grid
    Z = f(xi, yi)
    tck = f.tck + f.degrees

    # get boundary values
    # (linear interpolation)
    y_upper = bisplev(xi, yi[-1], tck)
    y_lower = bisplev(xi, yi[0], tck)
    dy_upper = bisplev(xi, yi[-1], tck, dy=1)
    dy_lower = -bisplev(xi, yi[0], tck, dy=1)

    mask_upper = yi > yp.max()
    n_upper = mask_upper.sum()
    upper_vals = np.cumsum(np.repeat(dy_upper, n_upper, axis=-1), axis=1) + y_upper
    Z[:, mask_upper] = upper_vals

    mask_lower = yi < yp.min()
    n_lower = mask_lower.sum()
    lower_vals = np.cumsum(np.repeat(dy_lower, n_lower, axis=-1), axis=1)[:, ::-1] + y_lower
    Z[:, mask_lower] = lower_vals

    return Z


def interpolate_shifts(shifts: list[np.ndarray] | np.ndarray, image_shape: tuple[int, ...], n_transects) -> np.ndarray:
    """
    Interpolate shift vectors inbetween known transects.

    Extrapolates values linearly outside area between the transects. This function makes use of scipy's
    RectBivariateSpline interpolation function but extrapolates values linearly instead of filling with
    constant values

    Parameters
    ----------
    shifts: list[np.ndarray]: shift values along transects
    image_shape: shape of the target image
    n_transects: number of transects

    Returns
    -------

    """
    # unevenly spaced grid
    xp = np.arange(image_shape[1])
    yp = np.linspace(
        0,
        1,
        n_transects + 2,
        endpoint=True
    )[1:-1] * image_shape[0]
    ZP = np.asarray(shifts).T

    # evenly spaced grid
    xi = xp.copy()
    yi = np.arange(image_shape[0])
    f = RectBivariateSpline(xp, yp, ZP, kx=1, ky=3 if n_transects > 3 else n_transects - 1)

    Z = extrapolate_bispline_top_bottom(f=f, yp=yp, xi=xi, yi=yi)

    return Z.T


def _apply_displacement_single_channel(u, v, image, **kwargs):
    """Apply flow fields to a signle channel image."""
    assert image.ndim == 2
    assert u.shape == v.shape == image.shape

    nr, nc = image.shape
    row_coords, col_coords = np.meshgrid(
        np.arange(nr), np.arange(nc), indexing='ij'
    )

    warped = warp(
        image,
        np.array([row_coords + v, col_coords + u]),
        mode='edge',
        **kwargs
    )

    return warped

def apply_displacement(u: np.ndarray, v: np.ndarray, image: np.ndarray, **kwargs):
    """Apply flow fields to a (possibly multichannel) image."""
    assert image.ndim in (2, 3), 'image must be either 2 or 3 dimensional'
    assert u.shape == v.shape == image.shape[:2]

    if image.ndim == 2:
        return _apply_displacement_single_channel(u, v, image, **kwargs)
    warped = np.zeros(image.shape)
    for channel in range(image.shape[2]):
        warped[:, :, channel] = _apply_displacement_single_channel(
            u, v, image[:, :, channel], **kwargs
        )
    if kwargs.get('preserve_range'):
        warped = warped.astype(image.dtype)

    return warped


class Mapper(Convenience):
    """
    Object to store shift matrices U and V and apply them to Images/ Feature tables.
    """
    _save_attrs: set[str] = {
        '_Us',
        '_Vs',
        '_image_shape',
        '_tag'
    }

    def __init__(
            self,
            image_shape: tuple[int, ...] | None = None,
            path_folder: str | None = None,
            tag: str | None = None
    ) -> None:
        """
        Initialize the object

        Parameters
        ----------
        image_shape : tuple[int, ...], optional
            The shape of the image to be transformed. Must be specified if
            mapper is not loaded.
        path_folder: str, optional
            The path to the folder to load or save an instance to or from disk.
        tag: str, optional
            Tags specifying the type of transformation. Will be appended to the
            file name (necessary if multiple transformations are stored in the
            same folder, as is usually the case).
        """
        self._image_shape: tuple[int, int] | None = (
            image_shape[:2]
            if image_shape is not None
            else None
        )
        self._Us: list[np.ndarray[int]] = []
        self._Vs: list[np.ndarray[int]] = []
        self.path_folder: str | None = path_folder
        self._tag: str = tag if tag is not None else ''

    def __add__(self, other: Self) -> Self:
        assert self._image_shape == other._image_shape
        new: Self = self.__class__(self._image_shape, self.path_folder, self._tag)
        new._Us = self._Us + other._Us
        new._Vs = self._Vs + other._Vs

        return new

    def _pre_save(self):
        """
        Compress transformations before saving.

        This saves a bit of disk space. No reason to save all the inbetween
        results.
        """
        self._stack_UV()

    def get_XY(self) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """Get coordinate matrices of right shape."""
        Y, X = np.indices(self._image_shape)
        return X, Y

    def add_UV(
            self,
            *args,
            trafo: Callable | None = None,
            U: np.ndarray[float] | None = None,
            V: np.ndarray[float] | None = None,
            **kwargs
    ) -> None:
        """Add transformation fields either from a function or by providing the
        point-wise shifts in the x- and y-direction (U and V respectively).
        """
        assert (trafo is not None) or (U is not None) or (V is not None), \
            "Must provide either trafo or U or V."

        if trafo is not None:
            U, V = self.get_UV(trafo, *args, **kwargs)
        elif U is None:
            U: np.ndarray[float] = np.zeros(self._image_shape)
        elif V is None:
            V: np.ndarray[float] = np.zeros(self._image_shape)

        assert U.shape == self._image_shape
        assert V.shape == self._image_shape

        self._Us.append(U)
        self._Vs.append(V)

    def get_UV(self, trafo: Callable, *args, is_uint8: bool = False, **kwargs):
        """Apply a transformation to coordinate matrices to get shift matrices."""
        X, Y = self.get_XY()
        if is_uint8:  # assuming mappings are linear we can split up transformation
            # TODO: fix this, if it is needed
            raise NotImplementedError('this is not working properly')
            m: int = 256
            Xm, Xr = np.divmod(X, m)  # multiplicity and remainder
            Ym, Yr = np.divmod(Y, m)

            XTm = trafo(Xm.astype(np.uint8), *args, *kwargs)
            XTr = trafo(Xr.astype(np.uint8), *args, *kwargs)
            YTm = trafo(Ym.astype(np.uint8), *args, *kwargs)
            YTr = trafo(Yr.astype(np.uint8), *args, *kwargs)

            XT = XTm * m + XTr
            YT = YTm * m + YTr
        else:
            XT = trafo(X.astype(np.float32), *kwargs)
            YT = trafo(Y.astype(np.float32), *kwargs)

        U = XT - X
        V = YT - Y

        return U, V

    def _get_combined_UV(self):
        """Chain the displacements."""
        X, Y = self.get_XY()
        XT, YT = X.copy(), Y.copy()
        for U, V in zip(self._Us, self._Vs):
            XT = apply_displacement(U, V, XT, preserve_range=True)
            YT = apply_displacement(U, V, YT, preserve_range=True)

        U = XT - X
        V = YT - Y

        return U, V

    def _stack_UV(self):
        """Stack displacements."""
        U, V = self._get_combined_UV()
        self._Us = [U]
        self._Vs = [V]

    def fit(self, image: np.ndarray, **kwargs) -> np.ndarray:
        assert image.ndim in (2, 3), 'Image must be 2D or 3D'
        assert image.shape[:2] == self._image_shape[:2], \
            f'Expected image of shape {self._image_shape[:2]}, got {image.shape[:2]}'
        if (n_trafos := len(self._Us)) == 0:
            return image
        elif n_trafos == 1:
            U, V = self._Us[0], self._Vs[0]
        else:
            U, V = self._get_combined_UV()
        return apply_displacement(U, V, image, **kwargs)

    def get_transformed_coords(self):
        X, Y = self.get_XY()
        XT = self.fit(X.astype(float), preserve_range=True)
        YT = self.fit(Y.astype(float), preserve_range=True)
        return XT, YT

    def plot_overview(self, ny: int = 50):
        img = resize(checkerboard(), self._image_shape)
        warped = self.fit(img)

        every = round(self._image_shape[0] / ny)
        indices = np.index_exp[::every, ::every]

        U, V = self._get_combined_UV()
        X, Y = self.get_XY()

        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        ax0.imshow(img)
        ax0.quiver(X[indices], Y[indices], U[indices], V[indices], angles='xy')
        ax0.set_title('original with deformation vectors')
        ax1.imshow(warped)
        ax1.set_title('transformed')
        plt.show()


