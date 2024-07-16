"""Functions for handling boxes."""
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging

from typing import Any

from src.imaging.util.Image_convert_types import ensure_image_is_gray
from src.imaging.util.Image_plotting import plt_cv2_image

logger = logging.getLogger('msi_workflow' + __name__)


def region_in_box(
        *,
        image: np.ndarray,
        center_box: tuple[int | float, int | float] | None = None,
        box_ratio_x: float | None = None,
        box_ratio_y: float | None = None,
        point_topleft: tuple[int, int] | None = None,
        point_bottomright: tuple[int, int] | None = None,
        x: int | None = None,
        y: int | None = None,
        w: int | None = None,
        h: int | None = None,
        plts: bool = False
) -> dict[str, Any]:
    """
    Infer remaining box parameters.

    Provide either all of (center_box, box_ratio_x, box_ratio_y),
    (point_topleft, point_bottomright), or (x, y, w, h).

    Parameters
    ----------
    image : np.ndarray
        Input image in which the box is defined.
    center_box: tuple[int | float, int | float], optional
        The center of the box as a tuple defining the x and y pixel coordinates.
    box_ratio_x: float, optional
        The extent of the box in the x-direction in terms of total width
        centered around center_box[0].
    box_ratio_y: float, optional
        The extent of the box in the y-direction in terms of total width
        centered around center_box[1].
    point_topleft: tuple[int | float, int | float], optional
        The top left point in the bounding box. First value corresponds to
        x, second to y pixel coordinate.
    point_bottomright: tuple[int | float, int | float], optional
        The bottom right point in the bounding box. First value corresponds to
        x, second to y pixel coordinate.
    x: int, optional
        Start value of box in x-direction.
    y: int, optional
        Start value of box in y-direction.
    w: int, optional
        Width of box in x-direction in pixels.
    h: int, optional
        Height of box in y-direction in pixels.

    Returns
    -------
    dict[str, Any]
        Other calculated box parameters as well as the image region inside the
        box and a mask array where values inside the box are true.
    """
    # points in format (x_idx, y_idx)
    if len(image.shape) != 2:
        image: np.ndarray = ensure_image_is_gray(image)
        logger.warning(
            'region_in_box expects grayscale image, converted input to grayscale'
        )
        logger.warning(image.shape)

    height, width = image.shape

    if all([ # box specified by center and ratios
            param is not None for param in
            (center_box, box_ratio_x, box_ratio_y)
    ]):
        center_box: np.ndarray[int] = np.around(center_box, 0).astype(int)
        x, y = center_box - (
            np.array([round(width * box_ratio_x / 2),
            round(height * box_ratio_y / 2)], dtype=int)
        )
        x_max, y_max = center_box + (
            np.array([round(width * box_ratio_x / 2),
            round(height * box_ratio_y / 2)], dtype=int)
        )
        # clip values
        y: int = np.max([y, 0])
        x: int = np.max([x, 0])
        y_max: int = np.min([y_max, height])
        x_max: int = np.min([x_max, width])

        point_topleft: np.ndarray[int] = np.array([x, y])
        point_bottomright: np.ndarray[int] = np.array([x_max, y_max])

        w: int = x_max - x
        h: int = y_max - y
    elif all([ # box specified by corners
            param is not None for param in
            (point_topleft, point_bottomright)
    ]):
        point_topleft: np.ndarray[int] = np.array(point_topleft, dtype=int)
        point_bottomright: np.ndarray[int] = np.array(point_bottomright, dtype=int)
        # get indexes
        x, y = point_topleft
        x_max, y_max = point_bottomright

        w: int = x_max - x
        h: int = y_max - y

        box_ratio_x: float = float(w) / width
        box_ratio_y: float = float(h) / height
        center_box: np.ndarray[int] = np.array(
            [round(x + float(w) / 2), round(y + float(h) / 2)],
            dtype=int
        )
    # box specified by topleft and size
    elif all([
            param is not None for param in (x, y, w, h)
    ]):
        box_ratio_x: float = float(w) / width
        box_ratio_y: float = float(h) / height

        y_max: int = y + h
        x_max: int = x + w

        point_topleft: np.ndarray[int] = np.array([x, y])
        point_bottomright: np.ndarray[int] = np.array([x_max, y_max])

        center_box: np.ndarray[int] = np.array(
            [round(x + float(w) / 2), round(y + float(h) / 2)],
            dtype=int
        )
    else:
        raise KeyError('Specify at least some of the parameters.')

    # get mask for pixels in box
    mask_box: np.ndarray[np.uint8] = np.zeros(image.shape, dtype=np.uint8)
    mask_box[y:y_max, x:x_max] = 255

    image_box: np.ndarray = image[y:y_max, x:x_max].copy()

    out_keys: list[str] = [
        'center_box',
        'box_ratio_x',
        'box_ratio_y',
        'point_topleft',
        'point_bottomright',
        'x', 'y', 'w', 'h',
        'mask_box',
        'image_box'
    ]
    out_vals: list = [
        center_box,
        box_ratio_x,
        box_ratio_y,
        point_topleft,
        point_bottomright,
        x, y, w, h,
        mask_box,
        image_box
    ]

    if plts:
        canvas = image.copy()
        cv2.rectangle(canvas, point_topleft,
                      point_bottomright, 127, height // 20)
        cv2.circle(
            canvas,
            (center_box),
            radius=height // 20,
            color=127,
            thickness=-1
        )
        plt_cv2_image(canvas)

    return dict(zip(out_keys, out_vals))


def get_mask_box_from_ratios(
        image_shape: tuple[int, ...],
        center_box: tuple,
        box_ratio_x: float,
        box_ratio_y: float
) -> np.ndarray[np.uint8]:
    """
    Create a mask object for an image where the region of the box contains 255.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Shape of the image.
    center_box : tuple
        Point defining the center of the box (x, y).
    box_ratio_x : float
        Coverage of the box relative to entire image in x-direction.
    box_ratio_y : float
        Coverage of the box relative to entire image in x-direction.

    Returns
    -------
    mask_box : np.ndarray[np.uint8]
        The mask array.

    """
    height, width = image_shape
    x, y = center_box - \
           np.array([round(width * box_ratio_x / 2),
                     round(height * box_ratio_y / 2)])
    x_max, y_max = center_box + \
                   np.array([round(width * box_ratio_x / 2),
                             round(height * box_ratio_y / 2)])

    # clip values
    y: int = np.max([y, 0])
    x: int = np.max([x, 0])
    y_max: int = np.min([y_max, height])
    x_max: int = np.min([x_max, width])

    box_idx_x, box_idx_y = np.meshgrid(
        np.arange(x, x_max), np.arange(y, y_max))
    # create the mask
    mask_box: np.ndarray[np.uint8] = np.zeros(image_shape, dtype=np.uint8)
    mask_box[box_idx_y, box_idx_x] = 255

    return mask_box


def get_mean_intensity_box(
        image: np.ndarray,
        center_box: tuple | None = None,
        box_ratio_x: float = .5,
        box_ratio_y: float = .5
) -> tuple[float, float]:
    """
    For given ratios and center calculate the difference between the
    average pixel values of the inside and outside the box.

    Parameters
    ----------
    image : np.ndarray
        Source image. The default is None.
    center_box : tuple, optional
        Pixel coordiantes of the center of the box. The default is None.
    box_ratio_x : float
        between 0 and 1, optional
        horizontal extent of the box in terms of box_length / image_length.
        The default is .5.
    box_ratio_y : float
        between 0 and 1, optional
        horizontal extent of the box in terms of box_length / image_length.
        The default is .5.

    Returns
    -------
    float, float
        mean pixel value of the pixles inside the box and outside the box.

    """
    height, width = image.shape

    # default is to take the center of the image
    if center_box is None:
        center_box = np.array(
            [round(width / 2), round(height / 2)], dtype=int)

    mask_box = get_mask_box_from_ratios(
        image.shape, center_box, box_ratio_x, box_ratio_y
    ).astype(bool)

    mean_box: float = image[mask_box].mean()
    mean_rest: float = image[~mask_box].mean()

    return mean_box, mean_rest


def get_ROI_in_image(image: np.ndarray, xywh_ROI: tuple[int, ...]) -> np.ndarray:
    """Return section if image specified by top-left, width and height."""
    x, y, w, h = xywh_ROI
    return image[y:y + h, x:x + w]


def test_region_in_box():
    from skimage.data import brick
    import matplotlib.patches as patches  # for drawing box
    image = brick()  # grayscale image with 512 x 512 pixels
    # define box from center point and extent
    params_center_extent = region_in_box(image=image, center_box=(image.shape[0] // 2, image.shape[1] // 2),
                                         box_ratio_x=.5, box_ratio_y=.5)
    # define box from point in top-left and bottom-right
    params_corners = region_in_box(image=image, point_topleft=(128, 128), point_bottomright=(384, 384))
    # define box from top-left corner, width and height
    params_xywh = region_in_box(image=image, x=128, y=128, w=256, h=256)
    # stack images, result should be grayscale (if parameters are the same)
    image_res = np.stack(
        [params['image_box']
         for params in (params_center_extent, params_corners, params_xywh)],
        axis=-1
    )

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(image)
    # draw boxes
    for params, c in zip((params_center_extent, params_corners, params_xywh), ('r', 'g', 'b')):
        box = patches.Rectangle(xy=(params['x'], params['y']),
                                width=params['w'], height=params['h'],
                                linewidth=2, edgecolor=c, facecolor='none', alpha=.3)
        axs[0].add_patch(box)

    axs[1].imshow(image_res)
    plt.show()


def test_mask_box_from_ratios():
    from skimage.data import brick
    image = brick()
    mask = get_mask_box_from_ratios(image.shape, center_box=(128, 256), box_ratio_x=.25, box_ratio_y=.75)

    plt.figure()
    plt.imshow(mask)
    plt.show()


def test_get_mean_intensity_box():
    from skimage.data import brick
    # image = brick()
    image = np.zeros((256, 256))
    params_box = dict(center_box=(128, 128), box_ratio_x=.1, box_ratio_y=.1)
    mask = get_mask_box_from_ratios(image.shape, **params_box).astype(bool)
    image[mask] = 1
    box, rest = get_mean_intensity_box(image, **params_box)

    plt.figure()
    plt.imshow(image)
    plt.show()

    print(f'{box=}, {rest=}')
    print('should be box=1, rest=0')


def test_get_ROI_in_image():
    from skimage.data import brick
    image = brick()
    image_section = get_ROI_in_image(image, xywh_ROI=(128, 128, 32, 32))

    plt.figure()
    plt.imshow(image_section)
    plt.show()


def test_all_fcts():
    test_region_in_box()
    test_mask_box_from_ratios()
    test_get_mean_intensity_box()
    test_get_ROI_in_image()


if __name__ == '__main__':
    # test_mask_box_from_ratios()
    test_get_mean_intensity_box()
