import logging

from imaging.util.Image_convert_types import ensure_image_is_gray

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# assumed extent of the punched holes
scale_hole = 1 / 5  # cm

# side on which the holes are positioned
sides = ('top', 'bottom')

logger = logging.getLogger("msi_workflow." + __name__)


def find_holes_side(
        image: np.ndarray,
        side: str,
        obj_color: str,
        depth_section: float = 5,
        plts=False
) -> tuple[list[np.ndarray[int]], float, float]:
    """
    Identify punchholes in an image at the specified side

    Parameters
    ----------
    image : np.ndarray
        Image in which to look for the holes. The algorithm works on binary
        images, but the function converts color images to grayscale
        and grayscale to binary, if necessary.
    side : str
        The side at which to look
    width_sector: float
        The width of the section of the image in which to look for the holes
        in terms of the image shape (value between 0 and 1).
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
    assert side in sides, f'side must be one of {sides} not {side}'
    width_sector = .5
    if len(image.shape) != 2:
        image = ensure_image_is_gray(image)
    if len(np.unique(image)) > 2:
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = image.shape
    # search only in the bottom or top half of the image, depending on where
    # the holes are
    if side == 'top':
        image_section = image[:int(h * width_sector + .5), :]
    elif side == 'bottom':
        image_section = image[-int(h * width_sector + .5):, :]
    # search (square-shaped) holes at approx 1 / 3 and 2 / 3
    size = int(scale_hole / depth_section * w + .5)
    size_k = size * 3

    # fill value used outside the image extent (same as background)
    if obj_color == 'dark':
        fill_value = 255
    else:
        fill_value = 0
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
    kernel_chunk = np.ones((size, size))
    kernel = np.block([
        [0 * kernel_chunk, -kernel_chunk, 0 * kernel_chunk],
        [-kernel_chunk, 3 * kernel_chunk, -kernel_chunk],
        [kernel_chunk, kernel_chunk, kernel_chunk]
    ])
    if side == 'top':
        kernel = kernel[::-1, :]

    # convolve with kernel
    image_square_filtered = cv2.filter2D(
        image_padded.astype(float), ddepth=-1, kernel=kernel.astype(float)
    )

    # crop to size of initial image
    image_square_filtered = image_square_filtered[size_k:-size_k, size_k:-size_k]

    # pick max (for obj color dark) in upper half and lower half
    image_left = image_square_filtered[:, :w // 2].copy()
    image_right = image_square_filtered[:, w // 2:].copy()
    if obj_color == 'dark':
        hole_left = np.argmax(image_left)
        hole_right = np.argmax(image_right)
    else:
        hole_left = np.argmin(image_left)
        hole_right = np.argmin(image_right)
    score = image_left.ravel()[hole_left] + image_right.ravel()[hole_right]
    hole_left = np.array(np.unravel_index(hole_left, image_left.shape))
    hole_right = np.array(np.unravel_index(hole_right, image_right.shape))
    hole_right += np.array([0, w // 2])
    # add vertical offset for cropped off part
    if side == 'bottom':
        hole_left += np.array([int((1 - width_sector) * h), 0])
        hole_right += np.array([int((1 - width_sector) * h), 0])

    extent = scale_hole / depth_section * w / 2

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
            y = h * width_sector
        else:
            y = h - h * width_sector
        plt.hlines(y, 0, w, colors='red')

        # squares around found center holes
        y1, x1 = hole_left - extent
        y2, x2 = hole_right - extent
        ax = plt.gca()
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
    if side is not None:
        points, _, size = find_holes_side(
            image,
            side=side,
            obj_color=obj_color,
            depth_section=depth_section,
            plts=plts
        )
        return points, size

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

    print(size)

    points = [points_b, points_t]
    scores = [score_b, score_t]

    if obj_color == 'dark':  # pick higher score
        idx = np.argmax(scores)
    else:
        idx = np.argmin(scores)
    logger.info(f'found holes at {"bottom" if idx == 0 else "top"}')
    return points[idx], size


# %%
if __name__ == '__main__':
    pass
