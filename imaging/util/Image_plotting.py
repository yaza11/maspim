from imaging.util.Image_convert_types import swap_RB, infere_mode

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import cv2


def plt_cv2_image(image: np.ndarray, title: str | None = None,
                  cmap: str | None = None, hold: bool | str = False,
                  save_png=None, dpi=300,
                  no_ticks=False, **kwargs
                  ):
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
    fig = plt.figure()
    ax = plt.gca()
    if (cmode := infere_mode(image)) != 'L':
        image = swap_RB(image.copy())
        plt.imshow(image, interpolation='None')
    elif (cmode == 'L') and (cmap is None):
        cmap = 'gray'
        plt.imshow(image, interpolation='None', cmap=cmap, **kwargs)

    if title is not None:
        plt.title(title)
    if no_ticks:
        ax.set_axis_off()
    else:
        plt.xlabel(r'Pixel coordinates in $x$-direction')
        plt.ylabel(r'Pixel coordinates in $y$-direction')
    if (hold == 'off') or (not hold):
        plt.show()
        if save_png is not None:
            plt.savefig(save_png, dpi=dpi)
    else:
        return fig


def plt_contours(
        contours: np.ndarray, image: np.ndarray,
        title: str | None = None,
        hold=False,
        **kwargs
) -> None:
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
    for contour in contours:
        cv2.drawContours(canvas, [contour], 0, 127, np.max([
            np.min(canvas.shape[:2]) // 20, 1]))
    fig = plt_cv2_image(canvas, title=title, hold=hold, **kwargs)
    if hold:
        return fig


def plt_rect_on_image(image, box_params, save_png=None, dpi=300, hold: bool = False, **kwargs):
    canvas = image.copy()
    fig = plt_cv2_image(canvas, hold=True, **kwargs)
    rect = patches.Rectangle(
        xy=box_params['point_topleft'],  # start point
        width=box_params['w'],
        height=box_params['h'],
        edgecolor='r',
        fill=False
        # linewidth=np.min(image.shape[:2]) // 100
    )

    ax = fig.gca()
    ax.add_patch(rect)
    if (hold == 'off') or (not hold):
        plt.show()
        if save_png is not None:
            plt.savefig(save_png, dpi=dpi)
    else:
        return fig


def plt_overview(section, window):
    from imaging.main.cImage import ImageProbe, ImageClassified
    I = ImageProbe(section, window)
    I.load()
    plt_cv2_image(I.sget_image_original(), title='original image')
    del I

    I = ImageClassified(section, window)
    I.load()
    plt_cv2_image(I.sget_mask_foreground(), title='foreground pixels')
    plt_cv2_image(I.sget_image_original(), title='ROI image')
    plt_cv2_image(I.sget_image_classification(), title='light and dark pixels')
    plt_cv2_image(I.get_image_simplified_classification(), title='simplified classification')
    del I


if __name__ == '__main__':
    import skimage
    plt_cv2_image(swap_RB(skimage.data.astronaut()))
    plt_cv2_image(skimage.data.brick())
