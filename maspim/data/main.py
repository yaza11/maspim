"""Superclass for cMSI, cXRF, cXRay."""
import os

import cv2
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from typing import Callable, Self
from collections.abc import Iterable

from scipy.interpolate import RectBivariateSpline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from tqdm import tqdm

from maspim.data.helpers import plot_comp, plt_comps
from maspim.imaging.util.coordinate_transformations import rescale_values
from maspim.res.constants import elements
from maspim.res.constants import dict_labels
from maspim.util.convenience import Convenience, check_attr
from maspim.imaging.util.image_plotting import plt_cv2_image


logger = logging.getLogger(__name__)


def estimate_probability_distribution(
        I, bin_edges: list[float] | None = None
) -> tuple[np.ndarray[float], list[float], np.ndarray[float]]:
    """
    For an image I calculate the probability distribution.

    Intensity are binnend into 256 bins with bin edges beeing distributed on a
    log scale to account for possible large intensity variations. For the log
    distribution, zeros will be counted separately. The returen pd is scaled
    to have area 1.

    Parameters
    ----------
    I : ndarray
        The image for which to calculate the prob dist.
    bin_edges : list[float] | None, optional
        Option to specify the bin_edges. The default is None.

    Returns
    -------
    prob : 1D array[float]
        The probabilty distribution.
    bin_edges : list[float]
        The bin edges used for binning the intensites.
    bin_centers : np.ndarray[float]
        The centers of bins

    """
    if bin_edges is not None:
        prob, bin_edges = np.histogram(I, bins=bin_edges, density=True)
    else:
        prob, bin_edges = np.histogram(I, bins='auto', density=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    return prob, bin_edges, bin_centers


def rankings_name_to_label(name):
    """Assign labels to features."""
    if name not in dict_labels:
        return name
    return dict_labels[name]


def PCA_biplot(
        score: pd.DataFrame,
        coeffs: pd.DataFrame,
        var: None | Iterable = None,
        title: str = '',
        hold: bool = False,
        add_annotations: bool = True
) -> None | tuple[plt.Axes, plt.Figure]:
    """
    Creat a biplot out of PCA results.

    Parameters
    ----------
    score : pd.DataFrame
        A pandas DataFrame containing the scores of each compound in the PCA
        result. Each row represents a compound, and each column represents a principal component score.
        The scores indicate the position of each compound in the PCA coordinate system.
    coeffs : pd.DataFrame
        A pandas DataFrame containing the coefficients or loadings of the principal components.
        Each row corresponds to a principal component, and each column represents a variable or feature
        in the original data space. These coefficients indicate the contribution of each variable to the
        principal components.
    var : None or Iterable, optional
        Optional iterable containing variance explained by PC1 and PC2.
    title : str, optional
        Optional title for the biplot.
    hold : bool, optional
        Whether to hold the plot for further modifications (default is False).
    add_annotations : bool, optional
        Whether to add annotations to the points (default is True).

    Returns
    -------
    None or tuple[plt.Axes, plt.Figure]
        If `hold` is False, returns None. If `hold` is True, returns a tuple
        containing the matplotlib Figure and Axes objects.
    """
    # coordinates in PCA coordinate system
    xs = score.iloc[:, 0]
    ys = score.iloc[:, 1]
    n = coeffs.shape[1]  # number of criteria
    # scale vectors to unit length
    range_x = xs.max() - xs.min()
    range_y = ys.max() - ys.min()
    xs /= range_x
    ys /= range_y
    # plot compounds in new frame of reference
    fig, axs = plt.subplots()
    plt.scatter(xs, ys, alpha=.8)
    # add labels to point
    if add_annotations:
        for i in range(score.shape[0]):
            plt.annotate(
                score.index[i], score.iloc[i] + .01, alpha=0.5, zorder=-1, size=5)

    # plot arrows for the criteria
    for i in range(n):
        coeff_x = coeffs.iloc[0, i]
        coeff_y = coeffs.iloc[1, i]
        plt.arrow(
            0, 0, coeff_x, coeff_y, color='r', alpha=0.5
        )
        plt.arrow(
            0, 0, -coeff_x, -coeff_y, color='r', alpha=0.5,
            linestyle=':'
        )
        txt = rankings_name_to_label(coeffs.columns[i])
        plt.text(coeff_x * 1.15, coeff_y * 1.15, txt,
                 color='g', ha='center', va='center')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    if var is None:
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
    else:
        plt.xlabel(f'PC 1 ({var[0]:.1%})')
        plt.ylabel(f'PC 2 ({var[1]:.1%})')
    plt.axis('equal')
    plt.grid()
    plt.title(title)
    if not hold:
        plt.show()
    else:
        return fig, axs


class DataBaseClass:
    _feature_table: pd.DataFrame | None

    @property
    def feature_table(self) -> pd.DataFrame:
        """Feature table is read-only"""
        assert check_attr(self, '_feature_table')
        return self._feature_table

    def _get_tic_scales(self,
                        column_tic: str = 'tic_window',
                        method: str = 'median',
                        **_) -> pd.Series:
        assert method in ['median', 'mean', 'portion'], f'{method=} not valid'
        assert (check_attr(self, '_feature_table')
                and (column_tic in self.feature_table)),\
            f'{column_tic=} not in ft, make sure that ft is defined'

        tics: pd.Series = self.feature_table.loc[:, column_tic]
        if method == 'median':
            scales: pd.Series = tics / tics.median()
        elif method == 'mean':
            scales: pd.Series = tics / tics.mean()
        elif method == 'portion':
            scales: pd.Series = rescale_values(tics, 0, 1)
        else:
            raise NotImplementedError(f'internal error: {method=} not defined')
        return scales

    def get_feature_table_tic_corrected(
            self,
            columns: Iterable | None = None,
            **kwargs
    ) -> pd.DataFrame:
        if columns is None:
            columns = self.columns

        scales = self._get_tic_scales(**kwargs)
        feature_table: pd.DataFrame = self.feature_table\
                                          .loc[:, columns]\
                                          .divide(scales, axis=0)
        return feature_table

    @property
    def data_columns(self) -> pd.Series:
        """Overwritten by children."""
        raise NotImplementedError()

    @property
    def data(self) -> pd.DataFrame:
        return self.feature_table.loc[:, self.data_columns]

    @property
    def nondata_columns(self) -> pd.Series:
        """Overwritten by children."""
        raise NotImplementedError()

    @property
    def nondata(self) -> pd.DataFrame:
        return self.feature_table.loc[:, self.nondata_columns]

    @property
    def xy(self) -> pd.DataFrame:
        return self.feature_table.loc[:, ['x', 'y']]

    @property
    def x(self) -> pd.Series:
        return self.feature_table.x

    @property
    def y(self) -> pd.Series:
        return self.feature_table.y

    @property
    def R(self) -> pd.Series:
        return self.feature_table.R

    @property
    def columns(self) -> pd.Index:
        return self.feature_table.columns


# need to inherit from DataBaseClass first, otherwise feature_table will be
# taken from Convenience
class Data(DataBaseClass, Convenience):
    """
    Class to manipulate and analyze MSI or XRF data.

    For application take a look at the MSI or XRF class. The Data class is not
    ment to be used directly.
    """
    path_folder: str | None = None
    d_folder: str | None = None

    _distance_pixels: int | float | None = None

    tilt_correction_applied: bool = False

    _pca_xy: pd.DataFrame | None = None
    _pca: PCA | None = None
    _pcs: np.ndarray[float] | None = None
    _pca_columns: pd.Series | None = None

    _nmf_columns: Iterable | None = None
    _nmf_xy: pd.DataFrame | None = None
    _W: np.ndarray[float] | None = None
    _H: np.ndarray[float] | None = None

    _kmeans: KMeans | None = None

    _save_attrs: set[str] = {
        'distance_pixels',
        '_feature_table',  # could be processed, so not necessarily redundant information
        'depth_section',
        'age_span',
        'tilt_correction_applied'
    }

    @property
    def path_d_folder(self):
        """Overwritten by children"""
        raise NotImplementedError()

    def _get_data_columns_xrf(self) -> np.ndarray[str]:
        """Return columns corresponding to data."""
        columns: pd.Index = self.columns

        # data columns are elements
        columns_valid: list[bool] = [
            col
            for col in columns
            if col in list(elements.Abbreviation)
        ]

        data_columns = np.array(columns_valid)
        return data_columns

    def _get_data_columns_msi(self) -> np.ndarray[str]:
        """Return columns corresponding to data."""
        columns = self.feature_table.columns
        # only return cols with masses
        columns_valid: list[str] = [
            col for col in columns if str(col).replace(
                '.', '', 1
            ).isdigit()
        ]

        data_columns: np.ndarray[str] = np.array(columns_valid)
        return data_columns

    @property
    def data_columns(self) -> pd.Series:
        return pd.Series(np.concatenate([
            self._get_data_columns_msi(), self._get_data_columns_xrf()
        ]))

    @property
    def nondata_columns(self) -> pd.Series:
        """Return all columns of the feature table that are not data."""
        # cols to check against
        dcols: set = set(self.data_columns)
        nondata_columns: set = set(self.columns) - dcols
        return pd.Series(nondata_columns)

    def get_data_for_columns(self, columns: Iterable[str | float]) -> pd.DataFrame:
        """Return part of the feature table specified by columns."""
        return self.feature_table.loc[:, columns]

    @property
    def data_mean(self) -> pd.Series:
        """Get the mean values for each feature in the feature table."""
        return self.data.mean(axis=0)

    def add_graylevel_from_data_frame(self, overwrite: bool = False) -> None:
        """
        Apply grayscale conversion in feature table.

        Using the R, G and B channel inside the feature table, add grayscale values.
        """
        assert check_attr(self, 'feature_table'), \
            'need to set a feature table'
        # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
        if ('L' in self.feature_table.columns) and (not overwrite):
            return
        self._feature_table['L'] = self.feature_table.apply(
            lambda row:
            round(0.299 * row.R + 0.587 * row.G + 0.114 * row.B),
            axis=1
        )

    def pixels_get_photo_ROI_to_ROI(
            self,
            data_ROI_xywh: tuple[int, ...],
            photo_ROI_xywh: tuple[int, ...],
            image_ROI_xywh: tuple[int, ...]
    ) -> None:
        """
        Add x_ROI, y_ROI columns.
        """
        # pixel coords of data, set by set_photo_roi
        (xd, yd, wd, hd) = data_ROI_xywh  # data units
        # corner and dimensions of data ROI in original image
        (xp, yp, wp, hp) = photo_ROI_xywh  # photo units
        # region of ROI defined by find_sample_region in pixel coords of
        # original image
        (xr, yr, wr, hr) = image_ROI_xywh
        # fit the pixel coords in the feature table to ROI pixel
        # coordinates by
        #   1. get values
        x_ft = self.x.to_numpy().astype(float)
        y_ft = self.y.to_numpy().astype(float)
        #   2. shift to 0
        x_ft -= xd
        y_ft -= yd
        #   3. scale so that coordinates are in terms of photo
        x_ft *= wp / wd
        y_ft *= hp / hd
        # --> coordinates in terms of photo_ROI
        #   4. shift accordingly in detected ROI
        # first add the corners defined in photo_ROI, then subtract the
        # offset of the detected ROI
        # --> pixel coordinates in photo-units
        x_ft += xp
        y_ft += yp
        # shift relative to origin of image ROI
        x_ft -= xr
        y_ft -= yr
        # append to feature_table, so now each (x, y) has an according
        # (x_ROI, y_ROI) corresponding to pixels in the ROI
        self._feature_table['x_ROI'] = np.around(x_ft).astype(int)
        self._feature_table['y_ROI'] = np.around(y_ft).astype(int)

    def add_attribute_from_image(
            self,
            image: np.ndarray,
            column_name: str,
            median: bool = False,
            coordinate_name: str = '_ROI',
            fill_value=0,
            plts: bool = False,
            **_
    ) -> None:
        """
        Add a column to the feature table from an image (ROI of sample)

        Parameters
        ----------
        image : np.ndarray
            The image to add to the feature table, must be grayscale and in the
            region of interest.
        column_name : str
            Name of the new column in the dataframe.
        median : bool, optional
            Whether to average out values around measurement points.
            The default is False.

        Returns
        -------
        None.

        """
        x_col = f'x{coordinate_name}'
        y_col = f'y{coordinate_name}'
        assert image.ndim == 2, 'image must be single channel'
        if median:
            # footprint of area to average out for classification
            length_median: int = int(
                np.median(np.diff(self.feature_table.loc[:, x_col])) / 2
            )
            if not length_median % 2:
                length_median += 1
            # for each point in feature table, take the median of an
            # dpixels/2 x dpixels/2 area to classify point
            image = cv2.medianBlur(image, length_median)

        # use interpolator:
        # we know values at p_roi_image
        # need values at x_roi_table, y_roi_table
        y_roi_image, x_roi_image = np.indices(image.shape, sparse=True)
        # grid in feature table may be irregular
        y_roi_table = self.feature_table.loc[:, y_col]
        x_roi_table = self.feature_table.loc[:, x_col]

        interpolator: RectBivariateSpline = RectBivariateSpline(
            x_roi_image, y_roi_image, image.T
        )

        interpolated = interpolator(x_roi_table, y_roi_table, grid=False)
        # for each point in feature table, find closest in image
        self._feature_table[column_name] = interpolated

        if plts:
            idxs = np.c_[
                self.feature_table.loc[:, y_col],
                self.feature_table.loc[:, x_col]
            ]
            plt.figure()
            plt.imshow(image)
            plt.plot(idxs[:, 1], idxs[:, 0], '-')
            plt.show()

            # plt_cv2_image(image_zeropad,'zeropaded')

            if median:
                plt_cv2_image(
                    image,
                    'classification after blurring according to data points'
                )

            # plt_cv2_image(image_zeropad, 'final classification')

            plt.figure()
            plt.imshow(
                self.feature_table.pivot(
                    index=y_col, columns=x_col, values=column_name),
                cmap='rainbow')
            plt.title('classification in feature table')
            plt.show()

    def combine_with(self, other: Self) -> None:
        """Combine two data objects based on the depth column."""

        def both_have(attr: str, obj1=self, obj2=other) -> bool:
            return check_attr(obj1, attr) & check_attr(obj2, attr)

        assert type(self) is type(other), 'objects must be of the same type'
        assert 'depth' in self.feature_table.columns, \
            'found no depth column in first object'
        assert 'depth' in other.feature_table.columns, \
            'found no depth column in second object'
        assert self.feature_table.depth.max() <= other.feature_table.depth.min(), \
            'first object must have smaller depths and depth intervals cannot overlap'

        # determine the new folder
        paths: list[str] = [self.path_folder, other.path_folder]
        new_path: str = os.path.commonpath(paths)
        logger.info(f'found common path: {new_path}')

        self_df: pd.DataFrame = self.feature_table.copy()
        other_df: pd.DataFrame = other.feature_table.copy()

        # set min to 0
        other_df.loc[:, 'x'] -= other_df.x.min()
        # shift by last value of this df
        other_df.loc[:, 'x'] += self_df.x.max()
        if both_have('x_ROI', self_df, other_df):
            # set min to 0
            other_df.loc[:, 'x_ROI'] -= other_df.x_ROI.min()
            # shift by last value of this df
            other_df.loc[:, 'x_ROI'] += self_df.x_ROI.max()

        new_df: pd.DataFrame = pd.concat(
            [self_df, other_df],
            axis=0
        ).reset_index(drop=True)
        self._feature_table = new_df

    def split_at_depth(self, depth: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split feature table at depth [cm] and return upper and lower section."""
        assert 'depth' in self.columns

        idxs_u = np.argwhere(self.feature_table.depth < depth)[:, 0]
        idxs_l = np.argwhere(self.feature_table.depth >= depth)[:, 0]

        Du = self.feature_table.iloc[idxs_u, :]
        Dl = self.feature_table.loc[idxs_l, :]
        return Du, Dl

    def estimate_pca(self, columns=None, th_pca=.95, exclude_holes=False):
        """Perform PCA analysis on the ion images."""
        if columns is None:
            columns: pd.Series = self.data_columns
        data: pd.DataFrame = self.get_data_for_columns(columns)
        # exclude rows containing only nans, mask is True for valid rows
        mask_nonnans: pd.Series = ~np.isnan(data).all(axis=1)
        # exclude holes, mask is True for valid rows
        if exclude_holes:
            mask_nonholes: pd.Series = (self.feature_table.classification != 0)
        else:
            mask_nonholes: np.ndarray[bool] = np.ones_like(mask_nonnans, dtype=bool)
        mask_valid_rows: pd.Series = mask_nonnans & mask_nonholes
        data_valid: pd.DataFrame = data.loc[mask_valid_rows, :].copy()
        # fill remaining nans with zeros
        data_valid: pd.DataFrame = data_valid.fillna(0)
        self._pca_xy: pd.DataFrame = self.xy.loc[mask_valid_rows, :]
        FT_s = StandardScaler().fit_transform(data_valid)
        # do PCA:
        self._pca: PCA = PCA(n_components=th_pca)
        self._pcs: np.ndarray[float] = self._pca.fit_transform(FT_s)
        self._pca_columns: pd.Series = columns

    @property
    def distance_pixels(self) -> int | float:
        assert check_attr(self, '_distance_pixels')
        return self._distance_pixels

    def estimate_nmf(
            self,
            k: int,
            columns: Iterable[float | str] | None = None,
            exclude_holes: bool = False,
            hole_column: str = 'valid',
            use_repeated_NMF: bool = False,
            N_rep: int = 30,
            max_iter: int = 10_000,
            return_summary: bool = False
    ) -> (
            tuple[np.ndarray[float], np.ndarray[float]] |
            tuple[np.ndarray[float], np.ndarray[float] | type]
    ):
        """Perform NMF analysis on the ion images"""
        if columns is None:
            columns: pd.Series = self.data_columns
        self._nmf_columns: Iterable = columns
        data: pd.DataFrame = self.get_data_for_columns(columns)
        # exclude rows containing only nans, mask is True for valid rows
        mask_nonnans: pd.Series = ~np.isnan(data).all(axis=1)
        # exclude holes, mask is True for valid rows
        if exclude_holes:
            assert hole_column in self.columns, \
                f'specified {hole_column} is not in the feature table'
            mask_nonholes: pd.Series = (self.feature_table.loc[:, hole_column] != 0)
        else:
            mask_nonholes: np.ndarray[bool] = np.ones_like(mask_nonnans, dtype=bool)
        mask_valid_rows: pd.Series = mask_nonnans & mask_nonholes
        data_valid: pd.DataFrame = data.loc[mask_valid_rows, :].copy()
        if np.any(data_valid.to_numpy() < 0):
            logger.warning(f'Found values smaller than 0 in NMF: '
                           f'{np.min(np.min(data_valid))}. '
                           f'Clipping negative values to 0.')
            data_valid[data_valid < 0] = 0
        # fill remaining nans with zeros
        data_valid: pd.DataFrame = data_valid.fillna(0)
        self._nmf_xy: pd.DataFrame = self.xy.loc[mask_valid_rows, :]
        feature_table_scaled = MaxAbsScaler().fit_transform(data_valid)

        if use_repeated_NMF:
            from mfe.feature import repeated_nmf
            S: type = repeated_nmf(feature_table_scaled, k, N_rep, max_iter=max_iter)
            self._W: np.ndarray[float] = S.matrix_w_accum
            self._H: np.ndarray[float] = S.matrix_h_accum
        else:
            model: NMF = NMF(n_components=k, max_iter=max_iter, init='nndsvd')

            self._W: np.ndarray[float] = model.fit_transform(feature_table_scaled)
            self._H: np.ndarray[float] = model.components_
        # store result in dict to be accessible by other functions such as
        # plot_nmf
        if return_summary and use_repeated_NMF:
            return self._W, self._H, S
        return self._W, self._H

    def nmf_find_k(
            self,
            k_min: int = 2,
            k_max: int = 10,
            dk: int = 1,
            plts: bool = True,
            **kwargs
    ) -> list[type]:
        """Helper method to determine the inner dimensions k for the NMF."""
        k_vec: np.ndarray[int] = np.arange(k_min, k_max + dk, dk)
        summary: list[type] = []
        k_plt: list[int] = []

        if plts:
            plt.figure()
        for k in k_vec:
            _, _, s = self.estimate_nmf(
                k, return_summary=True, use_repeated_NMF=True, **kwargs
            )
            summary.append(s)
            k_plt.append(k)
            if plts:
                coph: list[float] = [
                    summary[i].cophcor
                    for i in range(len(k_plt))
                ]
                plt.figure()
                plt.plot(k_plt, coph,
                         'o-',
                         label='Cophenetic correlation',
                         linewidth=2)
                plt.legend(loc='upper center',
                           bbox_to_anchor=(0.5, -0.05),
                           ncol=3,
                           numpoints=1)
                plt.xlabel('number of inner dimensions k')
                plt.ylabel('cophenetic correlation coefficient')
                plt.show()
        return summary

    def estimate_kmeans(self, n_clusters: int, columns: Iterable | None = None, **kwargs):
        """Perform k-means clustering on the ion images."""
        if columns is None:
            columns = self.data_columns
        kmeans: KMeans = KMeans(n_clusters=n_clusters, **kwargs) \
            .fit(self.get_data_for_columns(columns))
        self._feature_table[f'kmeans'] = kmeans.labels_
        self._kmeans: KMeans = kmeans

    def get_sparsed(self, th_nonzero: float, plts=False):
        """
        Drop compounds that fall below the threshold.

        The threshold determines what fraction of pixels has to be successful
        in order to be kept. For example, setting th_nonzero to .1 will drop
        all compounds that have less than 10% successful spectra (meaning more
        than 90% of the pixels have a value of 0).
        """
        # sparsity before:
        logger.info('sparsity before:',
                    sum((self.data == 0).astype(int).sum()) /
                    self.data.size)

        N = self.data.shape[0]
        sparsed_feature_table = self.data.copy()
        ratio_nonzero = sparsed_feature_table.astype(bool).sum(axis=0) / N

        if plts:
            plt.hist(ratio_nonzero, bins=len(self.data_columns) // 10)
            plt.xlabel('ratio not zero')
            plt.ylabel('count')
            plt.show()

        logger.info(
            'percent compounds removed: '
            + str((1 - (ratio_nonzero >= th_nonzero).sum() / len(ratio_nonzero)) * 100))
        # set rows for compounds that fall below threshold to 0
        sparsed_feature_table.loc[:, ratio_nonzero < th_nonzero] = 0

        # sparsity after:
        FT_c = sparsed_feature_table.loc[
               :, self.data_columns]
        logger.info(
            f'sparsity after: {sum((FT_c == 0).astype(int).sum()) / FT_c.size}'
        )

        # drop all zero columns
        logger.info(
            f'dropping columns with too many zeros: '
            f'{sparsed_feature_table.shape[1] - np.sum((sparsed_feature_table != 0).any(axis=0)):.2f}'
        )
        sparsed_feature_table = \
            sparsed_feature_table.loc[:, (sparsed_feature_table != 0).any(axis=0)]

        sparsed_feature_table[['x', 'y']] = self.xy
        sparsed_feature_table = sparsed_feature_table.sort_values(
            by=['y', 'x']).reset_index(drop=True)

        return sparsed_feature_table

    def get_filled_feature_table(
            self,
            data: pd.DataFrame = None,
            fill_value: float = 0
    ) -> tuple[pd.DataFrame, tuple[int, int, int], pd.Index]:
        """
        Data pixels may not fill a rectangular region.

        Applying this function assures that all pixels within a rectangular
        region are inside the feature table.
        """
        if data is None:
            data: pd.DataFrame = self.data.copy()

        columns: list = list(data.columns)
        # add p = (x, y) (will be dropped again later)
        data['p'] = self.xy.apply(lambda row: (row.x, row.y), axis=1)

        # convert the data in ft to stacked image
        x_unique: np.ndarray[int] = self.x.unique()
        y_unique: np.ndarray[int] = self.y.unique()
        # determine dimensions of the cube
        Nx: int = len(x_unique)
        Ny: int = len(y_unique)
        Ndims: int = len(columns)
        shape_filled: tuple[int, int, int] = (Nx, Ny, Ndims)

        # data may not be in rectangular region, those x, y will be missing
        # add zero columns for those pixels and put them in a mask
        x_all, y_all = np.meshgrid(x_unique, y_unique)  # all x and y values
        p: set = set(zip(x_all.ravel(), y_all.ravel()))  # all points
        p_missing: list = list(p.difference(set(data.p)))  # missing points
        df_zero_pixels: pd.DataFrame = pd.DataFrame(
            data=np.full((len(p_missing), Ndims), np.nan),
            columns=columns
        )
        df_zero_pixels['p'] = p_missing
        data_filled: pd.DataFrame = pd.concat([data, df_zero_pixels]) \
            .sort_values(by=['p']) \
            .reset_index(drop=True)
        # idx_inserted_pixels = [data_filled.index[data_filled.p == val][0]
        #                        for val in p_missing]
        idx_inserted_pixels: pd.Index = data_filled[
            data_filled['p'].isin(p_missing)
        ].index
        data_filled['x'] = data_filled.apply(lambda row: row.p[0], axis=1)
        data_filled['y'] = data_filled.apply(lambda row: row.p[1], axis=1)
        data_filled.drop(columns=['p'], inplace=True)
        # fill nans
        data_filled.fillna(fill_value, inplace=True)

        return data_filled, shape_filled, idx_inserted_pixels

    def get_data_cube(self, data: pd.DataFrame | None = None) -> np.ndarray:
        if data is None:
            data: pd.DataFrame = self.data
        data_rect, shape_filled, _ = self.get_filled_feature_table(data)

        Nx, Ny, Ndims = shape_filled

        # reshape
        data_cube: np.ndarray[float] = data_rect.to_numpy().reshape(
            (Nx, Ny, Ndims + 2)  # +2 because x and y are not taken into Ndims
        )

        return data_cube

    def perform_on_stacked_image(
            self, data: pd.DataFrame, func: Callable, **kwargs
    ) -> pd.DataFrame:
        """
        Apply a function to a stacked view of the feature table.

        Certain filters are quicker to apply on a multi-channel image.
        """

        # assumes that feature table contains pixels on rectangular grid,
        # no nans
        if ('x' not in data.columns) or ('y' not in data.columns):
            raise ValueError('data must contain x and y column')
        if data.shape[0] != len(data.x.unique()) * len(data.y.unique()):
            raise ValueError('pixels in feature table must be a \
    rectangular grid. You may have to use get_filled_feature_table')
        if data.isna().any(axis=None):
            raise ValueError('feature table cannot contain nans')

        # convert to cube
        columns = list(data.columns)

        # convert the data in ft to stacked image
        x_unique = self.feature_table.x.unique()
        y_unique = self.feature_table.y.unique()
        Nx = len(x_unique)
        Ny = len(y_unique)
        # exclude x and y column
        Ndims = len(columns) - 2

        # reshape
        data_cube = data.to_numpy().reshape((Nx, Ny, Ndims + 2))
        # apply function
        data_cube_processed = func(data_cube, **kwargs)

        # put back into feature table
        # copy over x, y
        data_cube_processed[:, :, -2] = data_cube[:, :, -2]
        data_cube_processed[:, :, -1] = data_cube[:, :, -1]
        data_processed = data_cube_processed.reshape(Nx * Ny, Ndims + 2)
        processed_feature_table = pd.DataFrame(
            data=data_processed,
            columns=columns)
        return processed_feature_table

    def processing_perform_smoothing(
            self,
            columns: Iterable | None = None,
            kernel_size: int = 3,
            sigma: float | int = 1,
            use_mask_for_holes: bool = True,
            classification_column: str = 'classification',
            key_holes: int = 0
    ):
        """Iterate over features and apply smoothing function."""
        # https://answers.opencv.org/question/3031/smoothing-with-a-mask/ --> not well explained
        # https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image

        if columns is None:
            columns = self.data_columns

        data = self.get_data_for_columns(columns).copy()

        # pixels where data is nan
        mask_nans = data.isna()

        # fill values to complete rectangular grid
        logger.info('filling missing values')
        data_filled, shape_filled, idx_inserted_pixels = \
            self.get_filled_feature_table(data)
        Nx, Ny, Ndims = shape_filled

        # create mask for hole cells
        if use_mask_for_holes:
            logger.info('creating mask')
            # grap classification column, x and y
            df_mask = self.get_data_for_columns([
                classification_column, 'x', 'y']
            )
            # fill missing values in df
            df_mask, _, _ = self.get_filled_feature_table(df_mask)
            # identify hole cells
            mask_holes = (df_mask[classification_column]).to_numpy() == key_holes
            # create nonhole cell with mathcing shape
            # False --> missing value
            # True --> available value
            mask_nonholes = (~mask_holes.reshape(Nx, Ny)).astype(float)
            # extend to Ndims
            mask_nonholes = np.repeat(
                mask_nonholes[:, :, np.newaxis], Ndims + 2, axis=2)
        else:
            mask_nonholes = 1

        data_cube = data_filled.to_numpy().reshape((Nx, Ny, Ndims + 2))

        # apply gaussian blur
        logger.info('smooting cube')
        data_cube_smoothed = scipy.ndimage.filters.gaussian_filter(
            data_cube * mask_nonholes,
            sigma=sigma, radius=kernel_size, axes=(0, 1)
        )

        # calcualte weights
        if use_mask_for_holes:
            logger.info('calculating weights')
            weights = scipy.ndimage.filters.gaussian_filter(
                mask_nonholes, sigma=sigma, radius=kernel_size, axes=(0, 1)
            )
        else:
            weights = 1

        # normalize smoothed by weights
        mask_valid_weights = weights > 0
        data_cube_smoothed[mask_valid_weights] /= weights[mask_valid_weights]
        # set data in invalid pixels to 0
        data_cube_smoothed *= mask_nonholes

        # put back into feature table
        # copy over x, y
        logger.info('writing smoothed feature table')
        data_cube_smoothed[:, :, -2] = data_cube[:, :, -2]
        data_cube_smoothed[:, :, -1] = data_cube[:, :, -1]
        data_smoothed = data_cube_smoothed.reshape(Nx * Ny, Ndims + 2)
        smoothed_feature_table = pd.DataFrame(
            data=data_smoothed,
            columns=data_filled.columns)

        # drop columns that were put in artificially
        smoothed_feature_table = smoothed_feature_table.drop(
            idx_inserted_pixels)
        smoothed_feature_table = smoothed_feature_table.sort_values(
            by=['y', 'x'])
        smoothed_feature_table.x = smoothed_feature_table.x.astype(int)
        smoothed_feature_table.y = smoothed_feature_table.y.astype(int)
        smoothed_feature_table = smoothed_feature_table.reset_index(drop=True)

        # set back original nans
        smoothed_feature_table[mask_nans] = np.nan
        logger.info('done smoothing')

        self.smoothed_feature_table = smoothed_feature_table
        return self.smoothed_feature_table

    def processing_zone_wise_average(
            self,
            zones_key: str | float | None = None,
            astype=float,
            columns: Iterable | None = None,
            exclude_zeros: bool = False,
            correct_zeros: bool = False,
            calc_std: bool = False,
            **_
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Return dataframe of zone-wise averages.

        Nans will not be included in the calculation. Optionally also return
        the standard deviations.

        Parameters
        ----------
        zones_key : TYPE, optional
            Column in feature table that define the zones. Rows with the same
            value in the zones_key-column will be averaged. The returned
            feature table contains a row for each unique value in zones-key.
            The default is None and defaults to column-wise averages of the
            image.
        astype : dtype, optional
            Convert the values in the columns to a specific type. The default
            is float. This can be used to count certain quantites. For example,
            setting astype to bool will calculate the nonzero density.
        columns : Iterable | None, optional
            The columns to include in the calculation. The default is None.
            This defaults to the data_columns.
        exclude_zeros: bool, optional
            If this is True, only those pixels within each zone will be taken
            that are not zero.
        correct_zeros: bool, optional
            If this option is set to True, zeros will be shifted to slightly
            higher values for calculating the average to account for the
            uncertainty induced by the SNR.
        calc_std : bool, optional
            If this is True, calculate also the standard deviation in each zone.
            The default is False. In that case only the averaged feature table
            will be returned. If this is True, a tuple of dataframes will be
            returned.

        Returns
        -------
        feature_table_averages : pd.DataFrame
            The averaged values. Contains a row for each unique value in the
            zones-key column and a column for each item in the passed columns
            (or the data columns).
        feature_table_stds : pd.DataFrame
            The table of standard deviations for each zone for each comp.
            Will only be returned if calc_std is True.
        feature_table_Ns: pd.DataFrame
            The number of nonzero entries in each zone for each comp.
            Will only be returned if calc_std is True.

        """
        if columns is None:
            columns: pd.Series = self.data_columns
        # column-wise if no key is given
        if zones_key is None:
            zones_key: str = 'zones_row_wise'
            # set zones as x-val starting with 0
            self._feature_table.loc[:, zones_key] = self.x - self.x.min()

        data_table: pd.DataFrame = (self.get_data_for_columns(columns)
                                    .copy().
                                    astype(astype))

        # get zones
        zones: np.ndarray = np.unique(self.feature_table.loc[:, zones_key])
        # remove nan key
        mask_keys: np.ndarray[bool] = ~np.isnan(zones)
        zones: np.ndarray = zones[mask_keys]

        n_zones: int = len(zones)

        # calculate zone wise averages
        averages: np.ndarray = np.empty((n_zones, data_table.shape[1]))

        if calc_std:
            stds: np.ndarray = np.empty((n_zones, data_table.shape[1]))
            Ns: np.ndarray = np.empty((n_zones, data_table.shape[1]))
        average_x_idx: np.ndarray[float] = np.zeros_like(zones, dtype=float)

        zones_column: pd.Series = self.feature_table.loc[:, zones_key].copy()
        x_column = self.x.copy()
        if exclude_zeros and correct_zeros:
            raise KeyError('exclude zeros and correct zeros are mutually exclusive.')
        elif exclude_zeros:
            # nans will not be respected in mean and std
            data_table[data_table == 0] = np.nan
        elif correct_zeros:
            data_table = self.get_data_zeros_corrected().loc[:, columns]

        # iterate over zones
        for idx, zone in tqdm(
                enumerate(zones),
                total=n_zones,
                desc='averaging values for zones'
        ):
            # average of each component in that zone
            # pixel mask: where entry in zones_key matches key
            mask_pixels_in_zone: np.ndarray[bool] = zones_column == zone
            # sub dataframe of data_table only containing pixels in zone
            pixels_in_zone: pd.DataFrame = data_table.loc[mask_pixels_in_zone, :]
            # pd.Series, averages for each compound in that zone
            averages[idx, :] = pixels_in_zone.mean(axis=0, skipna=True)
            if calc_std:
                # sigma = sqrt(1/(N - 1) * sum(x_i - mu))
                stds[idx, :] = pixels_in_zone.std(axis=0, skipna=True, ddof=1)
                Ns[idx, :] = (pixels_in_zone > 0).sum(axis=0, skipna=True)
            # average depth
            average_x_idx[idx] = x_column.loc[mask_pixels_in_zone].mean()
        # put in feature_table
        feature_table_averages: pd.DataFrame = pd.DataFrame(
            data=averages,
            columns=columns,
            index=zones)
        # add the mean x-value to data frame
        feature_table_averages['x'] = average_x_idx
        if not calc_std:
            return feature_table_averages

        feature_table_stds: pd.DataFrame = pd.DataFrame(
            data=stds,
            columns=columns,
            index=zones)
        feature_table_stds['x'] = average_x_idx
        feature_table_Ns: pd.DataFrame = pd.DataFrame(
            data=Ns,
            columns=columns,
            index=zones)
        feature_table_Ns['x'] = average_x_idx

        return feature_table_averages, feature_table_stds, feature_table_Ns

    def get_data_zeros_corrected(self) -> pd.DataFrame:
        """
        Return copy of current ft with corrected zeros.

        Zeros are shifted by 1/2 the lowest value in each spectrum.
        Since 0 could be anything between 0 and the SNR, this is on average
        closer to the real value if equal likelyhood is assumed.

        Returns
        -------
        pd.DataFrame
            The feature table with shifted zeros.

        """
        # nonzero mask
        nonzero: pd.DataFrame = self.data > 0
        # the lowest nonzero value for each pixel
        mins: np.ndarray[float] = (self.feature_table[nonzero]
                                   .min(axis=1).to_numpy())

        def replace_zeros(col: pd.Series) -> np.ndarray[float]:
            col: np.ndarray[float] = col.to_numpy()
            mask_zero: np.ndarray[bool] = col == 0
            col[mask_zero] = mins[mask_zero] / 2
            return col

        ft: pd.DataFrame = self.feature_table.copy()
        # add 1/2 of min to each nonzero
        ft.apply(lambda col: replace_zeros(col), axis=0, inplace=True)
        return ft

    def plot_pca(self, n_top: int = 10) -> None:
        """
        Perform and plot PCA results.

        Parameters
        ----------
        n_top : int, optional
            Plots the top N eigen-images with the highest explained variance.
        """
        assert check_attr(self, '_pca')
        # calculate loadings
        loadings: pd.DataFrame = pd.DataFrame(
            self._pca.components_.T,
            columns=['PC' + str(i) for i in range(np.shape(self._pcs)[1])],
            index=self._pca_columns)
        pc_mode = self._pca_xy.copy()
        top_compounds: np.ndarray[float] = self._pca.explained_variance_ratio_.cumsum()

        plt.figure()
        plt.plot(top_compounds)
        plt.grid(True)
        plt.xlabel('modes')
        plt.ylabel('cumulative explained variance')
        plt.show()

        fig, axs = plt.subplots(
            nrows=n_top,
            ncols=2,
            sharex='col',
            layout='constrained'
        )
        # df with x, y like original FT
        for i in range(n_top):
            pc_mode.loc[:, i] = self._pcs[:, i]
            axs[i, 0].imshow(pc_mode.pivot(index='y', columns='x', values=i),
                             aspect='equal',
                             interpolation='none')
            labels = [
                float(loading)
                if str(loading).replace('.', '').isnumeric()
                else str(loading)
                for loading in loadings.index
            ]
            axs[i, 1].stem(labels,
                           loadings['PC' + str(i)],
                           markerfmt=' ')

    def plot_nmf(self, plot_log_scale: bool = False, hold: bool = False):
        """
        Perform and plot the results of NMF.

        Parameters
        ----------
        plot_log_scale : bool, optional
            If True, intensities of the images will be scaled logarithmically.
        hold : bool, optional
            If True, will return the fig and axs objects for further modification.
        """
        assert check_attr(self, '_W'), 'call estimate_nmf first'

        W, H = self._W.copy(), self._H.copy()
        k: int = W.shape[1]

        if plot_log_scale:
            W: np.ndarray[float] = np.log(W)
        # put in df
        W_df: pd.DataFrame = pd.DataFrame(W, index=self._nmf_xy.index)

        W_df[['x', 'y']] = self._nmf_xy

        fig, axs = plt.subplots(
            nrows=k,
            ncols=2,
            figsize=(10, 2 * k),
            sharex='col', layout='constrained'
        )

        x_vals: np.ndarray = np.array(self._nmf_columns)
        if all([x_val.replace('.', '').isnumeric() for x_val in x_vals]):
            x_vals = x_vals.astype(float)

        for i in range(k):
            values = W_df.pivot(index='y', columns='x', values=i)

            axs[i, 0].imshow(values,
                             aspect='equal',
                             interpolation='none')
            axs[i, 0].axes.get_xaxis().set_ticks([])
            axs[i, 0].axes.get_yaxis().set_ticks([])

            axs[i, 1].stem(x_vals,
                           H[i, :],
                           markerfmt=' ',
                           linefmt='blue')
            axs[i, 1].set_ylabel('Weights')
        axs[i, 1].set_xlabel('m/z in Da')

        if hold:
            return fig, axs
        plt.show()

    def plot_kmeans(self):
        assert check_attr(self, '_kmeans'), \
            'call estimate_kmeans first'
        fig, axs = plt.subplots(nrows=2)
        axs[0].imshow(
            self.feature_table.pivot(
                index='y', columns='x', values=f'kmeans'),
            interpolation='none',
            cmap='Set1')

        # plot corresponding centers
        colors: list[str] = ['red', 'orange', 'gray']
        centers = self._kmeans.cluster_centers_
        for center, color in zip(centers, colors):
            labels = self.data_columns.astype(float)
            axs[1].plot(labels, center, color=color)
        plt.show()

    def plot_comp(
            self,
            comp: str | float | int,
            correct_tic: bool = False,
            **kwargs
    ) -> None:
        """
        Plot the ion image of a compound or feature.

        Parameters
        ----------
        comp: str | float | int
            The compound to plot.
        correct_tic: bool, optional
            If this is set to True, the intensities of each pixel will be
            scaled by the tic values
        kwargs: dict, optional
            Additional keywords for plot_comp and get_comp_as_img
        """
        comp = self.get_closest_mz(comp, max_deviation=None)
        if check_attr(self, '_distance_pixels'):
            kwargs['distance_pixels'] = self.distance_pixels

        plot_comp(
            comp,
            data_frame=(self.get_feature_table_tic_corrected(**kwargs)
                        if correct_tic
                        else self.feature_table),
            **kwargs
        )

    def plot_intensity_histograms(
            self,
            cols: Iterable = None,
            legend: bool = False,
            plt_log_x: bool = False,
            plt_log_y: bool = False,
            ignore_zeros: bool = False
    ) -> None:
        """
        Plot the intensity distributions for certain compounds.

        This function is useful to verify wether compound abundances follow a certain distribution.

        Parameters
        ----------
        cols: list, optional
            The columns for which to create the plot. Defaults to all data columns.
        legend: bool, optional
            Whether to add a legend for the compounds. Defaults to True.
        plt_log_x: bool, optional
            Whether to make the x-axis logarithmic. Default is False.
        plt_log_y: bool, optional
            Whether to make the x-axis logarithmic. Default is False.
        ignore_zeros: bool, optional
            Whether to exclude zero values from the plot. The default is False.
        """
        if cols is None:
            cols: pd.Series = self.data_columns

        plt.figure()
        for col in tqdm(cols, total=len(cols), desc='plotting distributions'):
            # exclude nans
            mask_nonzero: pd.Series = self.feature_table[:, col] >= 0
            intensities: pd.Series = self.feature_table[mask_nonzero, col]
            prob, bin_edges, bin_centers = estimate_probability_distribution(
                intensities
            )
            if ignore_zeros:
                prob = prob[1:]
                bin_centers = bin_centers[1:]

            if plt_log_x and plt_log_y:
                plt.loglog(bin_centers, prob, label=str(col))
            elif plt_log_x:
                plt.semilogx(bin_centers, prob, label=str(col))
            elif plt_log_y:
                plt.semilogy(bin_centers, prob, label=str(col))
            else:
                plt.plot(bin_centers, prob, label=str(col))
            if legend:
                plt.legend()
        plt.show()


class ZoneAnalyzer(DataBaseClass):
    """
    Special functionality for laminated samples.

    Is inherited by Data class
    """
    def __init__(self, feature_table: pd.DataFrame | None = None):
        self._feature_table = feature_table

    def calculate_KL_div(
            self,
            col: str | int,
            column_classification: str = 'classification',
            plts: bool = False
    ) -> tuple[float]:
        """
        Calculate KL divergence for an image in the feature table.

        For an image with no seasonality the entropy within the light and dark
        pixels should be the same. Or in other words: the probability
        distribution of the light pixels should be the same as that for the
        whole image.

        Parameters
        ----------
        col : str | int
            The column in the feature table for which to calculate the KL div.

        Returns
        -------
        tuple[float]
            Entropy of image, entropy of light pixels, entropy of dark pixels,
            KL of light pixels, KL of dark pixels.

        """
        classification_image = self.feature_table.loc[:, column_classification]
        I = self.feature_table[:, col].loc[
            (classification_image != 0) &
            (self.feature_table[:, col] >= 0)]
        I_light = I.loc[classification_image == 255]
        I_dark = I.loc[classification_image == 127]
        # estimate the probability distribution in the entire image
        prob, bin_edges, bin_centers = estimate_probability_distribution(I)
        # estimate the probability distribution in the light/dark pixels of the image
        #   using the same bin_edges
        prob_light, _, _ = estimate_probability_distribution(
            I_light, bin_edges=bin_edges)
        prob_dark, _, _ = estimate_probability_distribution(
            I_dark, bin_edges=bin_edges)

        # calculate entropy, relative entropy
        d_light = scipy.stats.entropy(prob_light, prob)
        d_dark = scipy.stats.entropy(prob_dark, prob)

        logger.info(
            f'Kullback-Leibler divergence for light pixels {d_light:.4f}')
        logger.info(
            f'Kullback-Leibler divergence for dark pixels {d_dark:.4f}')

        if plts:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].loglog(bin_centers, prob, label='pdf img')
            axs[0].plot(bin_centers, prob_light, label='pdf light')
            axs[0].plot(bin_centers, prob_dark, label='pdf dark')
            axs[0].legend()
            axs[0].set_xlabel('intensity')
            axs[0].set_ylabel('probability density')

            axs[1].plot(bin_centers, prob, label='pdf img')
            axs[1].plot(bin_centers, prob_light, label='pdf light')
            axs[1].plot(bin_centers, prob_dark, label='pdf dark')
            axs[1].legend()
            axs[1].set_xlabel('intensity')
            # axs[1].set_ylabel('probability density')

            plt.show()

        return d_light, d_dark

    def calculate_lightdark_rankings(
            self,
            use_intensities: bool = True,
            use_successes: bool = False,
            use_KL_div: bool = False,
            scale: bool = False,
            columns: None | list[str] = None,
            classification_column: str = 'classification',
            calc_corrs: bool = False,
            **_
    ) -> pd.DataFrame:
        """
        Calculate rankings based on light and dark pixel intensities.

        Parameters
        ----------
        use_intensities : bool, optional
            Whether to use intensities for ranking calculation (default is True).
        use_successes : bool, optional
            Whether to consider successful spectra for ranking calculation
            (default is False).
        use_KL_div : bool, optional
            Whether to use Kullback-Leibler divergence for ranking calculation
            (default is False).
        scale : bool, optional
            Whether to scale the rankings (default is False).
        columns : None or list of str, optional
            List of column names to consider for ranking calculation. If None,
             uses all available columns (default is None).
        classification_column : str, optional
            Name of the column containing classification information (default
            is 'classification').
        calc_corrs : bool, optional
            Whether to calculate correlations (default is False).
        **_
            Additional keyword arguments (ignored).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated rankings.
        """
        # Check if at least one ranking method is selected
        if not any((use_intensities, use_successes, use_KL_div)):
            raise KeyError('Use at least one of the ranking methods.')

        # Set default columns if not provided
        if columns is None:
            columns = self.get_data_columns()

        # Initialize DataFrame for rankings
        self.rankings = pd.DataFrame(index=columns)

        # Calculate average intensities in light and dark pixels
        ft_averages = self.processing_zone_wise_average(
            zones_key=classification_column, columns=columns).drop(columns=['x'])
        light = ft_averages.loc[255]
        self.rankings['av_intensity_light'] = light
        dark = ft_averages.loc[127]
        self.rankings['av_intensity_dark'] = dark

        # Calculate ratios, densities, and KL divergence for ranking
        ratio_intensities = light / (light + dark) - .5
        if not use_intensities:
            ratio_intensities = np.sign(ratio_intensities)

        if use_successes:
            ft_nonzeros = self.processing_zone_wise_average(
                zones_key=classification_column, astype=bool, columns=columns).drop(columns=['x'])

            ratio_nonzeros = np.zeros(len(columns))
            mask_light = (ratio_intensities > 0).to_numpy()
            ratio_nonzeros[mask_light] = ft_nonzeros.loc[255].iloc[mask_light]
            ratio_nonzeros[~mask_light] = ft_nonzeros.loc[127]

    def plt_PCA_rankings(
            self,
            columns=None,
            sign_criteria=False,
            add_laminae_averages=False,
            **kwargs
    ):
        """
        Compute and plot compounds in PCA space.
        """
        # create biplot
        # https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
        # calculate PCA of scaled columns
        if columns is None:
            columns = self.rankings.columns
        rankings = self.rankings[columns].copy()

        if add_laminae_averages:
            raise NotImplementedError('Depricated')
            # from time_series.cTimeSeries import TimeSeries
            # TS = TimeSeries(self._section, self._window)
            # TS.load()
            # r_av_L = TS.get_corr_with_grayscale().copy()
            # r_av_C = TS.get_corr_with_grayscale(contrast=True).copy()
            # del TS
            #
            # if not sign_criteria:
            #     r_av_L = np.abs(r_av_L)
            #     r_av_C = np.abs(r_av_C)
            # # add median seasonality
            # rankings['corr_av_L'] = r_av_L
            # rankings['corr_av_C'] = r_av_C
            # columns = list(columns) + ['corr_av_C', 'corr_av_L']

        if sign_criteria:
            signs = np.sign(self.rankings['intensity_div'])
            for c in ('KL_div', 'density_nonzero'):
                if c in columns:
                    rankings[c] *= signs

        X = StandardScaler().fit_transform(rankings)
        # do PCA:
        pca = PCA(n_components=2)
        # compounds in new frame of reference
        pcs = pd.DataFrame(
            data=pca.fit_transform(X),
            columns=[f'PC {i + 1}' for i in range(pca.n_components)],
            index=rankings.index
        )
        # get coefficients of criteria
        coeffs = pd.DataFrame(data=pca.components_, columns=columns)
        PCA_biplot(pcs, coeffs, var=pca.explained_variance_ratio_, **kwargs)

    def plt_overview_rankings(self, cols=None, **kwargs):
        """Plot rakings of different criteria."""
        if cols is None:
            cols = self.get_data_columns()
        # calculate the light, dark and ratio
        ft_averages = self.processing_zone_wise_average(
            zones_key='classification').drop(columns='x')
        light_average = ft_averages.iloc[2]
        dark_average = ft_averages.iloc[1]
        ratio_intensities = light_average / (light_average + dark_average) - .5
        sign_intensities = np.sign(ratio_intensities)

        # count the nonzeros in light, dark
        ft_nonzeros = self.processing_zone_wise_average(
            zones_key='classification', dtype=bool).drop(columns='x')
        nonzeros_light = ft_nonzeros.iloc[2]
        nonzeros_dark = ft_nonzeros.iloc[1]

        # calculate the divergences
        KL_divs = pd.DataFrame(data=np.empty(
            (len(cols), 2)), index=cols, columns=['KL_light', 'KL_dark'])

        for idx, col in enumerate(cols):
            KL_divs.loc[col, :] = self.calculate_KL_div(
                col, **kwargs)

        KL_divs_av = (KL_divs['KL_light'] + KL_divs['KL_dark']) / 2

        scaled_intensity = ratio_intensities / \
                           np.max(np.abs(ratio_intensities))
        scaled_KL_div = KL_divs_av / np.max(KL_divs_av)
        scaled_nonzeros_light = nonzeros_light / np.max(nonzeros_light)
        scaled_nonzeros_dark = nonzeros_dark / np.max(nonzeros_dark)

        scaled_score = scaled_intensity * scaled_KL_div * scaled_nonzeros_light
        scaled_score_dark = scaled_intensity * scaled_KL_div * scaled_nonzeros_dark
        scaled_score_light = scaled_intensity * scaled_KL_div * scaled_nonzeros_light
        # plot the scores
        # o = np.argsort(KL_divs_av * sign_intensities).to_numpy()
        o = np.argsort(scaled_score).to_numpy()
        x = np.array(cols)[o]
        # plt.plot(x, ratio_intensities.to_numpy()[o], label='intensity div', color='blue')
        # plt.plot(x, nonzeros_light.to_numpy()[o], label='density nonzeros light', alpha=.5)
        # plt.plot(x, nonzeros_dark.to_numpy()[o], label='density nonzeros dark', alpha=.5)
        # plt.plot(x, (KL_divs['KL_light'] * sign_intensities).to_numpy()[o], label='KL div light', alpha=.5)
        # plt.plot(x, (KL_divs['KL_dark'] * sign_intensities).to_numpy()[o], label='KL div dark', alpha=.5)
        # plt.plot(x, (KL_divs_av * sign_intensities).to_numpy()[o], label='KL div av', color='red')
        plt.plot(x, scaled_KL_div.to_numpy()[o], label='scaled KL div')
        plt.plot(x, scaled_intensity.to_numpy()[o], label='scaled intensity')
        plt.plot(x, scaled_nonzeros_light.to_numpy()[
            o], label='scaled nonzero light density')
        plt.plot(x, scaled_nonzeros_dark.to_numpy()[
            o], label='scaled nonzero dark density')
        plt.plot(x, scaled_score_light.to_numpy()[o], label='score light')
        plt.plot(x, scaled_score_dark.to_numpy()[o], label='score dark')
        plt.legend()
        plt.xticks([])

    def plt_top_comps_laminated(
            self, columns=None, light_or_dark='light', N_top=10, hold=False, **kwargs):
        """Calculate and plot av. intensity in light, dark and hole pixels."""
        if columns is None:
            columns = self.get_data_columns()
        rankings = self.calculate_lightdark_rankings(columns=columns, **kwargs)

        sorting = rankings.score.sort_values()
        print(sorting)
        if light_or_dark == 'light':
            cols_top = sorting.index[-N_top:].tolist()[::-1]
            rankings_top = sorting[-N_top:].tolist()[::-1]
        elif light_or_dark == 'dark':
            cols_top = sorting.index[:N_top].tolist()
            rankings_top = sorting[:N_top].tolist()
        else:
            raise KeyError('light_or_dark must be one of "light", "dark".')
        print(cols_top)
        print(rankings_top)
        o = plt_comps(self.feature_table, cols_top,
                      suptitle=f'top {N_top} in {light_or_dark} layers',
                      titles=[f's.: {ranking:.3f}' for ranking in rankings_top],
                      hold=hold,
                      **kwargs)
        if not hold:
            plt.show()
        else:
            return o

    def plt_contrasts(self, q=.5):
        ft_contrast = self.calculate_contrasts_simplified_laminae()

        x = ft_contrast['x']

        seasonalities = np.abs(ft_contrast.drop(columns='x'))
        cols = seasonalities.columns
        plt.stem(cols.astype(float),
                 seasonalities.quantile(q=.90, axis=0).to_numpy(),
                 markerfmt='',
                 label='q 90',
                 linefmt='C3')
        plt.stem(cols.astype(float),
                 seasonalities.quantile(q=.75, axis=0).to_numpy(),
                 markerfmt='',
                 label='q 75',
                 linefmt='C2')
        plt.stem(cols.astype(float),
                 seasonalities.mean(axis=0).to_numpy(),
                 markerfmt='',
                 label='av',
                 linefmt='C1')
        plt.stem(cols.astype(float),
                 seasonalities.median(axis=0).to_numpy(),
                 markerfmt='',
                 label='med',
                 linefmt='C0', basefmt='k')
        plt.legend()
        plt.xlabel('mz')
        plt.ylabel('contrast')
        plt.title('magnitude of average, median, 75th, 90th percentile')
        plt.show()

        y = seasonalities.quantile(q=q, axis=0)
        o = np.argsort(y.to_numpy())[::-1]
        cols = cols[o]
        plt_comps(
            self.feature_table, cols=cols[:10], remove_holes=True,
            suptitle=f'comps with highest {round(q * 100)}th quantile seasonality in \
{self._window} window'
        )
