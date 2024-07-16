"""Module for finding squre-shaped holes at the boundary of samples."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import logging

from src.imaging.util.Image_convert_types import ensure_image_is_gray

logger = logging.getLogger("msi_workflow." + __name__)

# assumed extent of the punched holes
SCALE_HOLE: float = 1 / 5  # cm

# side on which the holes are positioned
SIDES = ('top', 'bottom')


def find_holes_side(
        image: np.ndarray,
        side: str,
        obj_color: str,
        depth_section: float = 5,
        plts: bool = False
) -> tuple[list[np.ndarray[int]], float, float]:
    """
    Identify punch-holes in an image at the specified side

    Use template matching to find the position of areas that are
    hat-shaped.

    Parameters
    ----------
    image : np.ndarray
        Image in which to look for the holes. The algorithm works on binary
        images, but the function converts color images to grayscale
        and grayscale to binary, if necessary.
    side : str
        The side at which to look
    obj_color: str
        The color of the sample material 
        ("light" if lighter than background, "dark" otherwise).
    depth_section: float
        The horizontal (depth-direction) extent of the image in cm
    plts: bool
        Option for plotting inbetween results

    Returns
    -------
    tuple
        The identified holes (center points), the score and the sidelength of the holes.

    """
    assert side in SIDES, f'side must be one of {SIDES} not {side}'
    # splitting image into top and bottom section
    width_sector: float = .5
    # ensure binary image
    if len(image.shape) != 2:
        image: np.ndarray = ensure_image_is_gray(image)
    if len(np.unique(image)) > 2:
        logger.info('Converting provided image into binary')
        image: np.ndarray[np.uint8] = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    h, w = image.shape
    # search only in the bottom or top half of the image, depending on where
    # the holes are
    if side == 'top':
        image_section: np.ndarray[np.uint8] = image[:int(h * width_sector + .5), :]
    elif side == 'bottom':
        image_section: np.ndarray[np.uint8] = image[-int(h * width_sector + .5):, :]
    else:
        raise NotImplementedError('internal error')
    # search (square-shaped) holes at approx 1 / 3 and 2 / 3
    size: int = round(SCALE_HOLE / depth_section * w)
    size_k: int = size * 3

    # fill value used outside the image extent (same as background)
    if obj_color == 'dark':
        fill_value: int = np.unique(image)[1]
    else:
        fill_value: int = np.unique(image)[0]
    # could be np.uint8
    fill_value: int = int(fill_value)
    image_padded = cv2.copyMakeBorder(
        src=image_section,
        top=size_k, bottom=size_k, left=size_k, right=size_k,
        borderType=cv2.BORDER_CONSTANT,
        value=fill_value
    )
    # convolve with hat-shaped filter
    # filter is build up of chunks (with size of the hole) like
    # [[ 0 -1  0],
    #  [-1  3 -1]
    #  [ 1  1  1]]
    # and flipped vertically if holes are positioned at the top
    kernel_chunk: np.ndarray[float] = np.ones((size, size))
    kernel: np.ndarray[float] = np.block([
        [0 * kernel_chunk, -kernel_chunk, 0 * kernel_chunk],
        [-kernel_chunk, 3 * kernel_chunk, -kernel_chunk],
        [kernel_chunk, kernel_chunk, kernel_chunk]
    ])
    if side == 'top':
        kernel = kernel[::-1, :]

    # convolve with kernel
    image_square_filtered: np.ndarray[float] = cv2.filter2D(
        image_padded.astype(float), ddepth=-1, kernel=kernel.astype(float)
    )

    # crop to size of initial image
    image_square_filtered: np.ndarray[float] = image_square_filtered[
        size_k:-size_k, size_k:-size_k
    ]

    # pick max (for obj color dark) in upper half and lower half
    image_left: np.ndarray = image_square_filtered[:, :w // 2].copy()
    image_right: np.ndarray = image_square_filtered[:, w // 2:].copy()
    if obj_color == 'dark':
        hole_left: int = np.argmax(image_left)
        hole_right: int = np.argmax(image_right)
    else:
        hole_left: int = np.argmin(image_left)
        hole_right: int = np.argmin(image_right)
    score: float = (
        image_left.ravel()[hole_left] +
        image_right.ravel()[hole_right]
    )
    hole_left: np.ndarray[int] = np.array(np.unravel_index(hole_left, image_left.shape))
    hole_right: np.ndarray[int] = np.array(np.unravel_index(hole_right, image_right.shape))
    hole_right += np.array([0, w // 2])
    # add vertical offset for cropped off part
    if side == 'bottom':
        hole_left += np.array([int((1 - width_sector) * h), 0])
        hole_right += np.array([int((1 - width_sector) * h), 0])

    extent: float = SCALE_HOLE / depth_section * w / 2

    # some overview plots
    if plts:
        fig, axs = plt.subplots(nrows=3)

        axs[0].imshow(image_padded)
        axs[0].set_title('padded image')

        axs[1].imshow(kernel)
        axs[1].set_title('kernel')

        axs[2].set_title('filtered image')
        axs[2].imshow(image_square_filtered)

        plt.figure()
        plt.imshow(image)
        # red line at border of sector
        if side == 'top':
            y: float = h * width_sector
        else:
            y: float = h - h * width_sector
        plt.hlines(y, 0, w, colors='red')

        # squares around found center holes
        y1, x1 = hole_left - extent
        y2, x2 = hole_right - extent
        ax: plt.Axes = plt.gca()
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                2 * extent,
                2 * extent,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x2, y2),
                2 * extent,
                2 * extent,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
        )
        plt.title('identified holes')
        plt.tight_layout()
        plt.show()

    return [hole_left, hole_right], score, 2 * extent


def find_holes(
        image: np.ndarray,
        obj_color: str,
        side: str | None = None,
        depth_section: float = 5,
        plts=False
) -> tuple[list[np.ndarray[int]], float]:
    """Run find_holes_side for each side and pick the one with a higher score."""
    # run on provided side
    if side is not None:
        points, _, size = find_holes_side(
            image,
            side=side,
            obj_color=obj_color,
            depth_section=depth_section,
            plts=plts
        )
        return points, size

    # run on both sides
    points_b, score_b, size = find_holes_side(
        image,
        side='bottom',
        obj_color=obj_color,
        depth_section=depth_section,
        plts=plts
    )
    points_t, score_t, size = find_holes_side(
        image,
        side='top',
        obj_color=obj_color,
        depth_section=depth_section,
        plts=plts
    )

    # pick higher values
    points = [points_b, points_t]
    scores = [score_b, score_t]

    if obj_color == 'dark':  # pick higher score
        idx = np.argmax(scores)
    else:
        idx = np.argmin(scores)
    logger.info(f'found holes at {"bottom" if idx == 0 else "top"}')
    return points[idx], size


if __name__ == '__main__':
    pass
