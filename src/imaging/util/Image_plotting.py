"""Module for plotting (cv2) images."""
import matplotlib.pyplot as plt
import numpy as np
import cv2

from typing import Iterable, Any
from matplotlib import patches

from src.imaging.util.Image_convert_types import swap_RB, infer_mode
from src.imaging.util.coordinate_transformations import rescale_values


def plt_cv2_image(
        image: np.ndarray,
        title: str | None = None,
        cmap: str | None = None,
        hold: bool | str = False,
        ax: plt.Axes | None = None,
        fig: plt.Figure | None = None,
        save_png: str | None = None,
        dpi: int = 300,
        no_ticks: bool = False,
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
    ax : plt.Axes, optional
        An axes object. If not provided, a new one will be initialized.
        If provided, fig must also be provided and hold will be set to True.
    fig : plt.Figure, optional
        A figure object. If not provided, a new one will be initialized.
    save_png : str, optional
        A path to save the image to disk. Default is None.
    dpi : int, optional
        The resolution of the saved image. Defaults to 300.
    no_ticks : bool, optional
        If True, will not label the x- and y-axis. Default is False.
    **kwargs : Any
        Optional parameters to pass to imshow.

    Returns
    -------
    fig : plt.Figure
        The figure if hold is True.
    ax: plt.Axes
        The axes if hold is True.
    """

    # cv2 uses BGR instead of RGB so swap for image
    if ax is not None:
        assert fig is not None, "If axes is passed, also provide a fig"
        hold = True
    else:
        fig, ax = plt.subplots(layout="constrained")
    if (cmode := infer_mode(image)) != 'L':
        image = swap_RB(image.copy())
        ax.imshow(image, interpolation='None')
    elif (cmode == 'L') and (cmap is None):
        cmap = 'gray'
        ax.imshow(image, interpolation='None', cmap=cmap, **kwargs)
    else:
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
        contours: Iterable[np.ndarray[int]],
        image: np.ndarray,
        title: str | None = None,
        hold: bool = False,
        save_png: str | None = None,
        dpi: int = 300,
        **kwargs
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot contours on an image.

    Parameters
    ----------
    contours : Iterable[np.ndarray[int]]
        Contours to plot.
    image : np.ndarray
        The image which will be used as background.
    title : str | None, optional
        Title of figure. The default is None.
    hold : bool, optional
        Whether to return fig and ax or plot the figure. Default is False.
    save_png: str | None, optional
        Path to save the figure. Default is None.
    dpi : int, optional
        Resolution of saved image. Default is 300.
    kwargs: Any
        Additional keywords for plt_cv2_image.

    Returns
    -------
    fig : plt.Figure
        The figure if hold is True.
    ax: plt.Axes
        The axes if hold is True.
    """
    canvas: np.ndarray = image.copy()
    canvas: np.ndarray = rescale_values(
        canvas, 0, 255
    ).astype(np.uint8)
    for contour in contours:
        if len(contour) > 0:
            color = 127 if image.ndim == 2 else (0, 0, 255)
            cv2.drawContours(
                canvas,
                [contour.astype(np.int32)],
                0,
                color,
                np.max([np.min(canvas.shape[:2]) // 20, 1])
            )
    fig, ax = plt_cv2_image(canvas, title=title, hold=True, **kwargs)
    if hold:
        return fig, ax
    else:
        if save_png is not None:
            fig.savefig(save_png, dpi=dpi)
        plt.show()


def plt_rect_on_image(
        image: np.ndarray,
        box_params: dict[str, Any],
        save_png: str | None = None,
        dpi: int = 300,
        hold: bool | str = False,
        **kwargs: dict
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot a rectangle on top of an image.

    Parameters
    ----------
    image : np.ndarray
        The image which to use as canvas.
    box_params : dict[str, Any]
        A dictionary specifying the top-left point, width and height of the box.
    save_png: str, optional
        The path to save the image. Default is None which will not save the image.
    dpi: int, optional
        Resoluttion of saved image.
    hold: bool, optional
        Whether to return fig and ax.
    kwargs: Any
        Additional keywords for plt_cv2_image()

    Returns
    -------
    fig : plt.Figure
        The figure if hold is True.
    ax: plt.Axes
        The axes if hold is True.
    """
    canvas: np.ndarray = image.copy()
    fig, ax = plt_cv2_image(canvas, hold=True, **kwargs)
    rect = patches.Rectangle(
        xy=box_params['point_topleft'],  # start point
        width=box_params['w'],
        height=box_params['h'],
        edgecolor='r',
        fill=False,
        linewidth=2.5
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
