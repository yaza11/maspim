import scipy
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from skimage.transform import warp
from scipy.interpolate import LinearNDInterpolator, griddata
from typing import Iterable
from tqdm import tqdm

from msi_workflow.imaging.util.coordinate_transformations import rescale_values
from msi_workflow.res.constants import elements

logger = logging.getLogger(__name__)


def get_comp_as_img(
        data_frame,
        comp: str | int | float,
        exclude_holes: bool = True,
        classification_column: str = None,
        key_hole_pixels: int | bool | str = 0,
        flip: bool = False,
        idx_x: str = 'x_ROI',
        idx_y: str = 'y_ROI',
        **_
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
        logger.info(
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
        logger.info(
            f'did not find the column {classification_column} in '
            f'the feature table classifying the holes, so not excluding pixels'
        )
        exclude_holes: bool = False

    img_mz: np.ndarray[float] = data_frame.pivot(
        index=idx_y, columns=idx_x, values=comp).to_numpy().astype(float)
    if exclude_holes and (classification_column is not None):
        mask_holes: np.ndarray[bool] = data_frame.pivot(
            index=idx_y, columns=idx_x, values=classification_column
        ).to_numpy() == key_hole_pixels
        img_mz[mask_holes] = np.nan
    return img_mz, idx_x, idx_y


def clip_image(
        image: np.ndarray,
        comp: str | float | int,
        clip_above_percentile: float = .95,
        clip_below_percentile: float = .0,
        **_
) -> tuple[np.ndarray, float, float]:
    """Return copy of image where values below and above specified percentiles are clipped."""
    image_clipped = image.copy()
    if clip_above_percentile is None:
        comp_is_numeric = str(comp).replace(".", "").isnumeric()
        comp_is_element = str(comp) in elements.Abbreviation
        comp_is_data = comp_is_element or comp_is_numeric
        clip_above_percentile = .95 if comp_is_data else None
    if clip_above_percentile is not None:
        vmax = np.nanquantile(image, clip_above_percentile)
        logger.info(f'clipping values to {clip_above_percentile:.0%} percentile')
    else:
        vmax = np.nanmax(image)
        logger.info(f'not clipping values')

    if clip_below_percentile is None:
        comp_is_numeric = str(comp).replace(".", "").isnumeric()
        comp_is_element = str(comp) in elements.Abbreviation
        comp_is_data = comp_is_element or comp_is_numeric
        clip_below_percentile = 0 if comp_is_data else None
    if clip_below_percentile is not None:
        vmin = np.nanquantile(image, clip_below_percentile)
        logger.info(f'clipping values to {clip_below_percentile:.0%} percentile')
    else:
        vmin = np.nanmin(image)
        logger.info(f'not clipping values')

    image_clipped[image > vmax] = vmax
    image_clipped[image < vmin] = vmin

    return image_clipped, vmin, vmax



def plot_comp_on_image(
        comp: str | float | int,
        background_image: np.ndarray,
        data_frame: pd.DataFrame,
        *,
        title: str | None = None,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        hold: bool = False,
        cmap='inferno',
        **kwargs
):
    assert 'x_ROI' in data_frame.columns
    assert 'y_ROI' in data_frame.columns

    # get ion image
    img_mz, *_ = get_comp_as_img(data_frame, comp, **kwargs)
    # apply clipping
    img_mz, *_ = clip_image(img_mz, comp=comp, **kwargs)

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
    im = ax.imshow(img_mz, alpha=np.sqrt(img_mz), cmap=cmap)
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
        SNR_scale: bool = False,
        N_labels: int = 5,
        tick_precision: int = 0,
        distance_pixels = None,
        hold: bool = False,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        ticks_on_longer_axis: bool = True,
        **kwargs
) -> tuple[plt.Axes, plt.Axes] | None:
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
    SNR_scale: bool, optional
        This changes the ticks behavior of the intensity scale. If this value
        is set to True, 4 ticks will be used with the lowest one always being 0.
        If False, only 2 ticks will be used (with the lower one being the
        minimum value of the ion image).
    N_labels: int, optional
        The number of ticks on the depth scale. The default is 5.
    tick_precision: int, optional,
        The number decimals of the depth scale. The default is 0.
    kwargs: dict, optional
        keywords for get_comp_as_img
    """
    if img_mz is None:
        assert data_frame is not None, 'if no image is provided, provide a dataframe'
        assert comp in data_frame.columns, \
            f'{comp=} is not in the dataframe with columns {data_frame.columns}'
    img_mz, idx_x, idx_y = get_comp_as_img(
        data_frame=data_frame, comp=comp, flip=flip, **kwargs
    )

    img_clipped, vmin, vmax = clip_image(img_mz, comp=comp, **kwargs)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = vmin
    if 'vmax' not in kwargs:
        kwargs['vmax'] = vmax

    if fig is None:
        assert ax is None
        fig, ax = plt.subplots()
    else:
        hold = True

    # chose the y or x axis, depending on ticks_on_longer_axis
    tick_axis: int = (np.argmax(img_clipped.shape)
                 if ticks_on_longer_axis
                 else np.argmin(img_clipped.shape))
    tick_axis: str = ['y', 'x'][tick_axis]


    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'equal'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'

    keys_imshow = (
        'cmap norm aspect interpolation alpha vmin vmax origin extent '
        'interpolation_stage filternorm filterrad resample url data'
    ).split()
    kwargs_imshow = {k: v for k, v in kwargs.items() if k in keys_imshow}

    im = plt.imshow(
        img_clipped,
        **kwargs_imshow
    )

    if title is None:
        title = f'{comp}'

    tick_positions = np.linspace(
        start=0,
        stop=img_clipped.shape[0] if tick_axis == 'y' else img_clipped.shape[1],
        num=N_labels,
        endpoint=True
    )

    if tick_axis == 'y':
        ax.set_ylabel(r'depth (cm)')
    else:
        ax.set_xlabel(r'depth (cm)')

    if 'depth' in data_frame.columns:
        tick_labels = np.linspace(
            start=data_frame.depth.min(),
            stop=data_frame.depth.max(),
            num=N_labels,
            endpoint=True
        )
    elif distance_pixels is not None:
        pixel_to_depth = distance_pixels * 1e-4  # cm
        tick_labels = tick_positions * pixel_to_depth
    else:
        if flip:
            col_d = idx_x
        else:
            col_d = idx_y
        tick_labels = np.linspace(
            data_frame[col_d].min(),
            data_frame[col_d].max(),
            N_labels,
            endpoint=True
        )
        if tick_axis == 'y':
            ax.set_ylabel('pixel index y_ROI')
        else:
            ax.set_xlabel('pixel index y_ROI')
    tick_labels = np.round(tick_labels, tick_precision)
    if tick_precision <= 0:
        tick_labels = tick_labels.astype(int)

    if tick_axis == 'y':
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_xticks([])
    else:
        # move x ticks to top
        # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks([])

    ax.set_title(title)

    # decide where to put the colorbar
    portait_mode = img_mz.shape[0] > img_mz.shape[1]

    # SNR ratios
    if SNR_scale:
        ticks = [vmin, vmin + (vmax - vmin) / 3, vmin + (vmax - vmin) * 2 / 3, vmax]
        ticklabels = [str(np.around(t, 1)) for t in ticks]
    else:
        ticks = [vmin, vmax]
        i = 0
        min_label = max_label = ''
        while min_label == max_label:
            min_label = f'{vmin:.{i}e}'
            max_label = f'{vmax:.{i}e}'
            i += 1
        ticklabels = [min_label, max_label]
    if vmin > np.nanmin(img_mz):
        ticklabels[0] = r'$\leq$' + ticklabels[0]
    if vmax < np.nanmax(img_mz):
        ticklabels[-1] = r'$\geq$' + ticklabels[-1]
    cbar = fig.colorbar(
        im,
        orientation='vertical' if portait_mode else 'horizontal',
        shrink=.8
    )
    cbar.set_ticks(ticks=ticks, labels=ticklabels)

    if portait_mode:
        cbar.ax.set_yticklabels(ticklabels)
        cbar.ax.set_ylabel('Intensity', rotation=270, fontsize='10')
    else:
        cbar.ax.set_xticklabels(ticklabels)
        cbar.ax.set_xlabel('Intensity', fontsize='10')
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




