from imaging.util.Image_convert_types import ensure_image_is_gray
from imaging.util.Image_plotting import plt_cv2_image

import scipy
import matplotlib.pyplot as plt
import skimage
import cv2
import numpy as np
import logging

logger = logging.getLogger('msi_workflow' + __name__)


def region_in_box(
        image=None,
        center_box=None, box_ratio_x=None, box_ratio_y=None,
        point_topleft=None, point_bottomright=None,
        x=None, y=None, w=None, h=None, plts=False):
    """
    Return copy of image in box-region.

    Specify either (center_box, box_ratio_x, box_ratio_y),
        (point_topleft, point_bottomright) or
        (x, y, w, h)

    Parameters
    ----------
    image : array, optional
        DESCRIPTION. The default is None.
    center_box : tuple(int: x, int: y), optional
        DESCRIPTION. The default is None.
    box_ratio_x : float, optional
        DESCRIPTION. The default is None.
    box_ratio_y : float, optional
        DESCRIPTION. The default is None.
    point_topleft : tuple(int: x, int: y), optional
        DESCRIPTION. The default is None.
    point_bottomright : tuple(int: x, int: y), optional
        DESCRIPTION. The default is None.
    x : int, optional
        DESCRIPTION. The default is None.
    y : int, optional
        DESCRIPTION. The default is None.
    w : int, optional
        DESCRIPTION. The default is None.
    h : int, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    KeyError
        If wrong combination of parameters has been specified.

    Returns
    -------
    dict
        keys: variabel names and image region in box.

    """
    # points in format (x_idx, y_idx)
    if len(image.shape) != 2:
        image = ensure_image_is_gray(image)
        logger.warning('region_in_box expects grayscale image, converted input to \
grayscale')
        logger.warning(image.shape)

    height, width = image.shape

    # box specified by center and ratios
    if all([param is not None for param in
            (center_box, box_ratio_x, box_ratio_y)]):
        center_box = (np.array(center_box) + .5).astype(int)
        x, y = center_box - \
               np.array([round(width * box_ratio_x / 2),
                         round(height * box_ratio_y / 2)], dtype=int)
        x_max, y_max = center_box + \
                       np.array([round(width * box_ratio_x / 2),
                                 round(height * box_ratio_y / 2)], dtype=int)

        # clip values
        y = np.max([y, 0])
        x = np.max([x, 0])
        y_max = np.min([y_max, height])
        x_max = np.min([x_max, width])

        point_topleft = np.array([x, y])
        point_bottomright = np.array([x_max, y_max])

        w = x_max - x
        h = y_max - y

    # box specified by corners
    elif all([param is not None for param in
              (point_topleft, point_bottomright)]):
        point_topleft = np.array(point_topleft, dtype=int)
        point_bottomright = np.array(point_bottomright, dtype=int)
        # get indexes
        x, y = point_topleft
        x_max, y_max = point_bottomright

        w = x_max - x
        h = y_max - y

        box_ratio_x = float(w) / width
        box_ratio_y = float(h) / height
        center_box = np.array([round(x + float(w) / 2),
                               round(y + float(h) / 2)], dtype=int)

    # box specified by topleft and size
    elif all([param is not None for param in (x, y, w, h)]):
        box_ratio_x = float(w) / width
        box_ratio_y = float(h) / height

        y_max = y + h
        x_max = x + w

        point_topleft = np.array([x, y])
        point_bottomright = np.array([x_max, y_max])

        center_box = np.array([round(x + float(w) / 2),
                               round(y + float(h) / 2)], dtype=int)
    else:
        raise KeyError('Specify at least some of the parameters.')

    # get mask for pixels in box
    mask_box = np.zeros(image.shape, dtype=np.uint8)
    mask_box[y:y_max, x:x_max] = 255

    image_box = image[y:y_max, x:x_max].copy()

    out_keys = ['center_box', 'box_ratio_x', 'box_ratio_y',
                'point_topleft', 'point_bottomright',
                'x', 'y', 'w', 'h',
                'mask_box', 'image_box']
    out_vals = [center_box, box_ratio_x, box_ratio_y,
                point_topleft, point_bottomright,
                x, y, w, h,
                mask_box, image_box]

    if plts:
        canvas = image.copy()
        cv2.rectangle(canvas, point_topleft,
                      point_bottomright, 127, height // 20)
        cv2.circle(canvas, (center_box), radius=height //
                                                20, color=127, thickness=-1)
        plt_cv2_image(canvas)

    return dict(zip(out_keys, out_vals))


def get_mask_box_from_ratios(image_shape, center_box, box_ratio_x, box_ratio_y):
    """
    Create a mask object for an image where the region of the box contains 1.

    Parameters
    ----------
    image_shape : TYPE
        DESCRIPTION.
    center_box : TYPE
        DESCRIPTION.
    box_ratio_x : TYPE
        DESCRIPTION.
    box_ratio_y : TYPE
        DESCRIPTION.

    Returns
    -------
    mask_box : TYPE
        DESCRIPTION.

    """
    height, width = image_shape
    x, y = center_box - \
           np.array([round(width * box_ratio_x / 2),
                     round(height * box_ratio_y / 2)])
    x_max, y_max = center_box + \
                   np.array([round(width * box_ratio_x / 2),
                             round(height * box_ratio_y / 2)])

    # clip values
    y = np.max([y, 0])
    x = np.max([x, 0])
    y_max = np.min([y_max, height])
    x_max = np.min([x_max, width])

    box_idx_x, box_idx_y = np.meshgrid(
        np.arange(x, x_max), np.arange(y, y_max))
    # create the mask
    mask_box = np.zeros(image_shape, dtype=np.uint8)
    mask_box[box_idx_y, box_idx_x] = 255

    return mask_box


def get_mean_intensity_box(image, center_box=None, box_ratio_x=.5, box_ratio_y=.5) -> tuple[float, float]:
    """
    For given ratios and center calculate the difference between the
    average pixel values of the inside and outside of the box.

    Parameters
    ----------
    image : cv2 image | None, optional
        Source image. The default is None.
    center_box : tuple(int, int), optional
        Pixel coordiantes of the center of the box. The default is None.
    box_ratio_x : float between 0 and 1, optional
        horizontal extent of the box in terms of box_length / image_length.
        The default is .5.
    box_ratio_y : float between 0 and 1, optional
        horizontal extent of the box in terms of box_length / image_length.
        The default is .5.
    plts : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    float, float, tuple(tuple(int, int), tuple(int, int))
        mean pixel value of the pixles inside the box, outside the box and
        the bounding points of the box.

    """
    height, width = image.shape

    # default is to take the center of the image
    if center_box is None:
        center_box = np.array(
            [round(width / 2), round(height / 2)], dtype=int)

    mask_box = get_mask_box_from_ratios(
        image.shape, center_box, box_ratio_x, box_ratio_y
    ).astype(bool)

    mean_box = image[mask_box].mean()
    mean_rest = image[~mask_box].mean()

    return mean_box, mean_rest


def get_ROI_in_image(image: np.ndarray, xywh_ROI: tuple[int]) -> np.ndarray:
    """Return section if image specified by top-left, width and height."""
    x, y, w, h = xywh_ROI
    return image[y:y + h, x:x + w]


def test_region_in_box():
    from skimage.data import brick
    import matplotlib.patches as patches  # for drawing box
    image = brick()  # grayscale image with 512 x 512 pixels
    # define box from center point and extent
    params_center_extent = region_in_box(
        image,
        center_box=(image.shape[0] // 2, image.shape[1] // 2),
        box_ratio_x=.5,
        box_ratio_y=.5
    )
    # define box from point in top-left and bottom-right
    params_corners = region_in_box(
        image,
        point_bottomright=(384, 384),
        point_topleft=(128, 128)
    )
    # define box from top-left corner, width and height
    params_xywh = region_in_box(
        image,
        x=128,
        y=128,
        w=256,
        h=256
    )
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
