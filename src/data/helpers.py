import scipy
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from skimage.transform import warp
from scipy.interpolate import LinearNDInterpolator, griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Iterable
from tqdm import tqdm

from src.imaging.util.coordinate_transformations import rescale_values
from src.res.constants import elements


logger = logging.getLogger(__name__)


def get_comp_as_img(
        data_frame,
        comp: str | int | float,
        exclude_holes: bool = True,
        classification_column: str = None,
        key_hole_pixels: int | bool | str = 0,
        flip: bool = False,
        idx_x: str = 'x_ROI',
        idx_y: str = 'y_ROI'
) -> [np.ndarray, str, str]:
    """
    Return a component from the feature table as an image by pivoting.

    Parameters
    ----------
    comp: str | int | float
        The compound or feature to fetch.
    exclude_holes: bool, optional
        Whether to set holes to nan. The default is True.
    classification_column: str, optional
        The column specifying holes. If not specified, holes will not be excluded.
    key_hole_pixels: int | bool | str, optional
        The key defining holes. The default is 0.
    flip: bool, optional
        Whether to flip the image over its main diagonal. The default is False.

    Returns
    -------
    img_mz: np.ndarray
        The feature image (without holes, depending on the inputs).
    """
    if (idx_x not in data_frame.columns) and (idx_x != 'x'):
        logger.warning(
            f'did not find {idx_x} in {data_frame.columns}, ' +
            f'attempting to find "x" and "y"'
        )
        idx_x: str = 'x'
        idx_y: str = 'y'

    assert idx_x in data_frame.columns, f'did not find {idx_x} in {data_frame.columns}'
    assert idx_y in data_frame.columns, f'did not find {idx_y} in {data_frame.columns}'

    if flip:
        idx_x, idx_y = idx_y, idx_x

    if (
            (classification_column is not None) and
            (classification_column not in data_frame.columns)
    ):
        logger.warning(
            f'did not find the column {classification_column} in '
            f'the feature table classifying the holes, so not excluding pixels'
        )
        exclude_holes: bool = False

    img_mz: np.ndarray[float] = data_frame.pivot(
        index=idx_x, columns=idx_y, values=comp).to_numpy().astype(float)
    if exclude_holes and (classification_column is not None):
        mask_holes: np.ndarray[bool] = data_frame.pivot(
            index=idx_x, columns=idx_y, values=classification_column
        ).to_numpy() == key_hole_pixels
        img_mz[mask_holes] = np.nan
    return img_mz, idx_x, idx_y


def plot_comp_on_image(
        comp: str | float | int,
        background_image: np.ndarray,
        data_frame: pd.DataFrame,
        *,
        title: str | None = None,
        clip_at: float = .95,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        hold: bool = False,
        **kwargs
):
    assert 'x_ROI' in data_frame.columns
    assert 'y_ROI' in data_frame.columns

    # get ion image
    img_mz, *_ = get_comp_as_img(data_frame, comp, **kwargs)

    if clip_at is None:
        comp_is_numeric = str(comp).replace(".", "").isnumeric()
        comp_is_element = str(comp) in elements.Abbreviation
        comp_is_data = comp_is_element or comp_is_numeric
        clip_at = .95 if comp_is_data else None
    if clip_at is not None:
        vmax = np.nanquantile(img_mz, clip_at)
        logger.info(f'clipping values to {int(clip_at * 100)} percentile')
    else:
        vmax = np.nanmax(img_mz)
        logger.info(f'not clipping values')

    img_mz[img_mz > vmax] = vmax
    # img_mz[img_mz < np.nanquantile(img_mz, .25)] = 0

    # pad to fill out entire image
    x_ROI, *_ = get_comp_as_img(data_frame, 'x_ROI', **kwargs)
    y_ROI, *_ = get_comp_as_img(data_frame, 'y_ROI', **kwargs)
    values = img_mz.ravel()
    x = x_ROI.ravel()
    y = y_ROI.ravel()
    valid = (~np.isnan(values)) & (~np.isnan(x)) & (~np.isnan(y))
    values = values[valid]
    x = x[valid]
    y = y[valid]

    points = np.c_[x, y]
    h_ROI, w_ROI = background_image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w_ROI),
        np.arange(h_ROI)
    )
    # rescale to match resolution of background image
    img_mz = griddata(
        points, values, (grid_x, grid_y), method='linear', fill_value=0
    )
    img_mz = rescale_values(img_mz, 0, 1)
    background_image = rescale_values(background_image, 0, .5)

    if fig is None:
        fig, ax = plt.subplots()
    # plot background image
    ax.imshow(background_image)
    im = ax.imshow(img_mz, alpha=np.sqrt(img_mz), cmap='inferno')
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)

    # divider = make_axes_locatable(ax)
    # cax_pos = ('right' if (portait_mode := (img_mz.shape[0] > img_mz.shape[1]))
    #            else 'bottom')
    # cax = divider.append_axes(cax_pos, size="20%", pad=0.05)
    #
    # ticks = [0, vmax]
    # ticklabels = ['0', '{:.0e}'.format(vmax)]
    # print(ticks, ticklabels)
    # cbar = plt.colorbar(im, cax=cax, ticks=ticks, location=cax_pos)
    # if portait_mode:
    #     cbar.ax.set_yticklabels(ticklabels)
    #     cbar.ax.set_ylabel('Intensity', rotation=270)
    # else:
    #     cbar.ax.set_xticklabels(ticklabels)
    #     cbar.ax.set_xlabel('Intensity', rotation=0)

    if not hold:
        plt.show()
    return fig, ax


def plot_comp(
        comp: str | float | int,
        *,
        data_frame: pd.DataFrame | None = None,
        img_mz: np.ndarray | None = None,
        title: str | None = None,
        save_png: str | None = None,
        flip: bool = False,
        clip_at: float | None = None,
        SNR_scale: bool = False,
        N_labels: int = 5,
        y_tick_precision: int = 0,
        distance_pixels = None,
        hold: bool = False,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        **kwargs
):
    """
    Plot the ion image of a compound or feature.

    Parameters
    ----------
    comp: str | float | int
        The compound to plot.
    title: str | None, optional
        The title of the figure. If not provided, the compound name will be used.
    save_png: str | None, optional
        path and file name to save the plot. If not provided, the image will
        not be saved.
    flip: bool, optional
        Whether to flip the image over its main diagonal.
    clip_at: float | None, optional
        By default values are clipped to the 95th percentile for data features.
        Nondata features are by default not clipped. Providing a value will
        clip the data at the given quantile (valid values are (0, 1]).
    SNR_scale: bool, optional
        This changes the ticks behavior of the intensity scale. If this value
        is set to True, 4 ticks will be used with the lowest one always being 0.
        If False, only 2 ticks will be used (with the lower one being the
        minimum value of the ion image).
    N_labels: int, optional
        The number of ticks on the depth scale. The default is 5.
    y_tick_precision: int, optional,
        The number decimals of the depth scale. The default is 0.
    kwargs: dict, optional
        keywords for get_comp_as_img
    """
    if img_mz is None:
        assert data_frame is not None
        assert comp in data_frame.columns
    img_mz, idx_x, idx_y = get_comp_as_img(
        data_frame=data_frame, comp=comp, flip=flip, **kwargs
    )

    # clip values above vmax
    if clip_at is None:
        comp_is_numeric = str(comp).replace(".", "").isnumeric()
        comp_is_element = str(comp) in elements.Abbreviation
        comp_is_data = comp_is_element or comp_is_numeric
        clip_at = .95 if comp_is_data else None
    if clip_at is not None:
        vmax = np.nanquantile(img_mz, clip_at)
        logger.info(f'clipping values to {int(clip_at * 100)} percentile')
    else:
        vmax = np.nanmax(img_mz)
        logger.info(f'not clipping values')

    if fig is None:
        assert ax is None
        fig, ax = plt.subplots()
    else:
        hold = True

    im = plt.imshow(img_mz,
                    aspect='equal',
                    interpolation='none',
                    vmax=vmax)

    if title is None:
        title = f'{comp}'

    y_tick_positions = np.linspace(
        start=0,
        stop=img_mz.shape[0],
        num=N_labels,
        endpoint=True
    )
    ax.set_ylabel(r'depth (cm)')
    if 'depth' in data_frame.columns:
        y_tick_labels = np.linspace(
            start=data_frame.depth.min(),
            stop=data_frame.depth.max(),
            num=N_labels,
            endpoint=True
        )
    elif distance_pixels is not None:
        pixel_to_depth = distance_pixels * 1e-4  # cm
        y_tick_labels = y_tick_positions * pixel_to_depth
    else:
        if flip:
            col_d = idx_x
        else:
            col_d = idx_y
        y_tick_labels = np.linspace(
            data_frame[col_d].min(),
            data_frame[col_d].max(),
            N_labels,
            endpoint=True
        )
        ax.set_ylabel(r'pixel index y_ROI')
    y_tick_labels = np.round(y_tick_labels, y_tick_precision)
    if y_tick_precision <= 0:
        y_tick_labels = y_tick_labels.astype(int)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False
    )

    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax_pos = ('right' if (portait_mode := (img_mz.shape[0] > img_mz.shape[1]))
               else 'bottom')
    cax = divider.append_axes(cax_pos, size="20%", pad=0.05)
    # SNR ratios
    if SNR_scale:
        ticks = [0, vmax / 3, 2 * vmax / 3, vmax]
        ticklabels = [
            '0',
            f'{np.around(vmax / 3, 1)}',
            f'{np.around(2 * vmax / 3, 1)}',
            f'>{np.around(vmax, 1)}'
        ]
    else:
        ticks = [0, vmax]
        ticklabels = ['0', '{:.0e}'.format(vmax)]
    cbar = plt.colorbar(im, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.set_ylabel('Intensity', rotation=270 if portait_mode else 0)
    if save_png is not None:
        plt.savefig(save_png, dpi=300)
    if hold:
        return fig, ax
    plt.show()


def plt_comps(
        df,
        cols,
        suptitle='',
        titles=None,
        remove_holes=False,
        interpolate_zeros=False,
        figsize=(12, 5),
        hold=False,
        **_
):
    """Plot multiple compounds in feature table."""
    fig, axs = plt.subplots(nrows=1, ncols=len(cols), sharex=True, sharey=True,
                            figsize=figsize)

    C = df.pivot(index='x_ROI', columns='y_ROI', values='classification')
    for i, col in enumerate(cols):
        img = df.pivot(index='x_ROI', columns='y_ROI', values=col).copy()
        # set hole pixels to nan
        if interpolate_zeros:
            # valid grid points are not holes and not outside the data ROI
            mask_grid = (df.classification != 0) & (df[col] >= 0)
            gridX = df.x.loc[mask_grid]
            gridY = df.y.loc[mask_grid]

            # valid points are not zero (and not nan)
            mask_valid = (df[col] > 0)
            points_x = df.x.loc[mask_valid]
            points_y = df.y.loc[mask_valid]
            values = df[col].loc[mask_valid]
            # interpolate the missing values
            img_interpolated = scipy.interpolate.griddata(
                (points_x, points_y), values, (gridX, gridY))
            # turn into image shape
            img_full = np.zeros_like(mask_grid, dtype=float)
            img_full[~mask_grid] = np.nan
            img_full[mask_grid] = img_interpolated
            img = pd.DataFrame(
                data=np.vstack([img_full, df.x, df.y]).T,
                columns=['I', 'x_ROI', 'y_ROI']) \
                .pivot(index='x_ROI', columns='y_ROI', values='I')

        if remove_holes:
            img[C == 0] = np.nan

        axs[i].imshow(img,
                      aspect='equal',
                      interpolation='none',
                      vmax=df[col].quantile(.95))
        if titles is not None:
            title = f'{col}\n{titles[i]}'
        else:
            title = str(col)
        axs[i].set_title(title)
    # plt.tight_layout()
    plt.suptitle(suptitle)
    if not hold:
        plt.show()
    else:
        return fig, axs


def _get_grid(x: Iterable, y: Iterable, dx=1, dy=1) -> list[np.ndarray]:
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    x_min, x_max = np.floor(x.min() / dx) * dx, np.ceil(x.max() / dx) * dx
    y_min, y_max = np.floor(y.min() / dy) * dy, np.ceil(y.max() / dy) * dy
    x_new: np.ndarray[float] = np.arange(x_min, x_max + dx, dx)
    y_new: np.ndarray[float] = np.arange(y_min, y_max + dy, dy)

    # Create the regular grid
    return np.meshgrid(x_new, y_new)


def transform_feature_table(
        df: pd.DataFrame,
        *,
        p_ROI_T: pd.DataFrame | None = None,
        x_ROI_T: pd.Series | None = None,
        y_ROI_T: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Transform data in feature table according to transformed coordinates.

    It is assumed that x_ROI and y_ROI are columns in the feature table.
    Transformed coordinates have to be either in the data frame
    (as x_ROI_T and y_ROI_T) or
    """
    def get_comp_as_arr(comp: str | float | int) -> np.ndarray:
        return df.pivot(index='y_ROI', columns='x_ROI', values=comp).to_numpy()

    assert 'x_ROI' in df.columns
    assert 'y_ROI' in df.columns
    assert (
        (('x_ROI_T' in df.columns) and ('y_ROI_T' in df.columns)) or
        (p_ROI_T is not None) or
        ((x_ROI_T is not None) and (y_ROI_T is not None))
    ), 'If x_ROI_T and y_ROI_T are not in the df'

    # get x_ROI_T, y_ROI_T
    if p_ROI_T is not None:
        x_ROI_T: pd.Series = p_ROI_T.x_ROI_T
        y_ROI_T: pd.Series = p_ROI_T.y_ROI_T
    if x_ROI_T is not None:
        df['x_ROI_T'] = x_ROI_T
        df['y_ROI_T'] = y_ROI_T

    # get coordinate matrices
    X: np.ndarray[float] = get_comp_as_arr('x_ROI')
    Y: np.ndarray[float] = get_comp_as_arr('y_ROI')
    XT: np.ndarray[float] = get_comp_as_arr('x_ROI_T')
    YT: np.ndarray[float] = get_comp_as_arr('y_ROI_T')
    # calculate shift matrices
    # need to scale by differences for warp
    dX: np.ndarray[float] = np.ones_like(X)
    dX[:, 1:] = np.diff(X, axis=1)  # first values don't need to be scaled
    dY: np.ndarray[float] = np.ones_like(Y)
    dY[1:, :] = np.diff(Y, axis=0)  # first values don't need to be scaled
    U: np.ndarray[float] = (XT - X) / dX
    V: np.ndarray[float] = (YT - Y) / dY
    df.drop(columns=['x_ROI_T', 'y_ROI_T'], inplace=True)

    nr, nc = U.shape
    row_coords, col_coords = np.meshgrid(
        np.arange(nr), np.arange(nc), indexing='ij'
    )

    df_new = pd.DataFrame(
        data=np.zeros((nr * nc, len(df.columns))),
        columns=df.columns
    )
    for comp in tqdm(
            df.drop(columns=['x_ROI', 'y_ROI']).columns,
            total=df_new.shape[1] - 2,
            desc='warping ion images'
    ):
        image = get_comp_as_arr(comp)
        warped = warp(
            image,
            np.array([row_coords + V, col_coords + U]),
            mode='edge',
            preserve_range=True
        )
        df_new.loc[:, comp] = warped.ravel()
    df_new.loc[:, 'x_ROI'] = X.ravel()
    df_new.loc[:, 'y_ROI'] = Y.ravel()
    mask_nan = np.isnan(df_new.x_ROI) | np.isnan(df_new.y_ROI)
    return df_new.loc[~mask_nan, :]


def __transform_feature_table(
        df: pd.DataFrame,
        *,
        p_ROI_T: pd.DataFrame | None = None,
        x_ROI_T: pd.Series | None = None,
        y_ROI_T: pd.Series | None = None,
        dx: int | float | None = None,
        dy: int | float | None = None
) -> pd.DataFrame:
    """
    Transform data in feature table according to transformed coordinates.
    This wrong, but not sure why, dont want to delete yet

    It is assumed that x_ROI and y_ROI are columns in the feature table.
    Transformed coordinates have to be either in the data frame
    (as x_ROI_T and y_ROI_T) or
    """
    def get_d(vals: pd.Series) -> int | float:
        ds = vals.diff().iloc[1:]  # exclude first nan val
        for d in ds:  # pick the smallest value bigger than 0
            if d > 0:
                return d
        else:
            raise ValueError(f'found invalid spacings in x_ROI {ds} (x values should be increasing)')

    assert 'x_ROI' in df.columns
    assert 'y_ROI' in df.columns
    assert (
        (('x_ROI_T' in df.columns) and ('y_ROI_T' in df.columns)) or
        (p_ROI_T is not None) or
        ((x_ROI_T is not None) and (y_ROI_T is not None))
    ), 'If x_ROI_T and y_ROI_T are not in the df'

    x_ROI: pd.Series = df.x_ROI.copy()
    y_ROI: pd.Series = df.y_ROI.copy()
    # df.drop(columns=['x_ROI', 'y_ROI'], inplace=True)

    if dx is None:
        dx = get_d(x_ROI)
    if dy is None:
        dy = get_d(y_ROI)

    # get x_ROI_T, y_ROI_T
    if p_ROI_T is not None:
        x_ROI_T: pd.Series = p_ROI_T.x_ROI_T
        y_ROI_T: pd.Series = p_ROI_T.y_ROI_T
    elif 'x_ROI_T' in df.columns:
        x_ROI_T: pd.Series = df.x_ROI_T.copy()
        y_ROI_T: pd.Series = df.y_ROI_T.copy()
        df.drop(columns=['x_ROI_T', 'y_ROI_T'], inplace=True)
    # exclude nans
    mask_nan = x_ROI_T.isna() | y_ROI_T.isna()
    x_ROI_T = x_ROI_T[~mask_nan]
    y_ROI_T = y_ROI_T[~mask_nan]

    # map x_ROI, y_ROI to x_ROI_T, y_ROI_T in a way that conserves local averages
    # p_ROI_T are the points where values are known, we want them on p_ROI

    # arrays of shape (n_points_y, n_points_x)
    grid_x, grid_y = _get_grid(x_ROI_T, y_ROI_T, dx=dx, dy=dy)
    n_points = grid_x.shape[0] * grid_x.shape[1]

    # array of shape (n_points_y * n_points_x, n_columns)
    values: np.ndarray[float] = df.loc[~mask_nan, :].values
    n_columns = values.shape[1]

    # Interpolate data for each value column
    interp: LinearNDInterpolator = LinearNDInterpolator(
        list(zip(x_ROI_T, y_ROI_T)), values
    )
    # array of shape (n_points_y, n_points_x, n_columns)
    grid_z: np.ndarray[float] = interp(grid_x, grid_y)

    # Create a new DataFrame with grid coordinates
    new_df: pd.DataFrame = pd.DataFrame({'x_ROI': grid_x.ravel(), 'y_ROI': grid_y.ravel()})

    # Add the interpolated value columns to the DataFrame
    interpolated_df: pd.DataFrame = pd.DataFrame(
        grid_z.reshape((n_points, n_columns)),
        columns=df.columns
    )

    return pd.concat([new_df, interpolated_df], axis=1)




