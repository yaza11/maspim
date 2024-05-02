from imaging.util.Image_convert_types import swap_RB, infere_mode
from imaging.util.coordinate_transformations import rescale_values

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import cv2


def plt_cv2_image(
        image: np.ndarray, title: str | None = None,
        cmap: str | None = None, hold: bool | str = False,
        ax: plt.Axes | None = None, fig: plt.Figure | None = None,
        save_png=None, dpi=300,
        no_ticks=False,
        **kwargs
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot the array.

    Parameters
    ----------
    image : np.ndarray
        image to plot.
    title : str | None, optional
        The default is None.
    cmap : str | None, optional
        Colormap. For grayscale image None defaults to 'gray'. The default is None.
    hold : bool, optional
        If True, will not show the plot but return the figure.
        The default is False.
    **kwargs : TYPE
        Optional parameters to pass to imshow.

    Returns
    -------
    fig : plt.figure()
    """

    # cv2 uses BGR instead of RGB so swap for image
    if ax is not None:
        assert fig is not None, "If axes is passed, also provide a fig"
        hold = True
    else:
        fig, ax = plt.subplots(layout="constrained")
    if (cmode := infere_mode(image)) != 'L':
        image = swap_RB(image.copy())
        ax.imshow(image, interpolation='None')
    elif (cmode == 'L') and (cmap is None):
        cmap = 'gray'
        ax.imshow(image, interpolation='None', cmap=cmap, **kwargs)

    if title is not None:
        ax.set_title(title)
    if no_ticks:
        ax.set_axis_off()
    else:
        ax.set_xlabel(r'Pixel coordinates in $x$-direction')
        ax.set_ylabel(r'Pixel coordinates in $y$-direction')
    if (hold == 'off') or (not hold):
        plt.show()
        if save_png is not None:
            fig.savefig(save_png, dpi=dpi)
    else:
        return fig, ax


def plt_contours(
        contours: np.ndarray, image: np.ndarray,
        title: str | None = None,
        hold=False,
        **kwargs
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot contours on an image.

    Parameters
    ----------
    contours : np.ndarray
        DESCRIPTION.
    image : np.ndarray
        DESCRIPTION.
    title : str | None, optional
        Title of figure. The default is None.

    Returns
    -------
    None
    """
    canvas = image.copy()
    canvas = rescale_values(canvas, 0, 255).astype(np.uint8)
    for contour in contours:
        if len(contour) > 0:
            cv2.drawContours(canvas, [contour.astype(np.int32)], 0, 127, np.max([
                np.min(canvas.shape[:2]) // 20, 1]))
    fig, ax = plt_cv2_image(canvas, title=title, hold=True, **kwargs)
    if hold:
        return fig, ax
    else:
        plt.show()


def plt_rect_on_image(
        image: np.ndarray,
        box_params: dict[str, int | float],
        save_png: str | None = None,
        dpi: int = 300,
        hold: bool | str = False,
        **kwargs: dict
) -> tuple[plt.Figure, plt.Axes] | None:
    canvas = image.copy()
    fig, ax = plt_cv2_image(canvas, hold=True, **kwargs)
    rect = patches.Rectangle(
        xy=box_params['point_topleft'],  # start point
        width=box_params['w'],
        height=box_params['h'],
        edgecolor='r',
        fill=False
        # linewidth=np.min(image.shape[:2]) // 100
    )

    ax.add_patch(rect)
    if (hold == 'off') or (not hold):
        plt.show()
        if save_png is not None:
            fig.savefig(save_png, dpi=dpi)
    else:
        return fig, ax


if __name__ == '__main__':
    import skimage

    plt_cv2_image(swap_RB(skimage.data.astronaut()))
    plt_cv2_image(skimage.data.brick())
