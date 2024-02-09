from imaging.util.Image_convert_types import ensure_image_is_gray

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import data
import cv2

scale_hole = 1 / 5  # cm

sides = ('top', 'bottom')

def find_holes(image: np.ndarray, side: str, obj_color: str, width_sector=.5, depth_section=5, plts=False) -> tuple:
    """
    Identify punchholes in an image at the specified side

    Parameters
    ----------
    image : np.ndarray
        Image in which to look for the holes.
    side : str
        The side at which to look
    width_sector: float
        The width of the section of the image in which to look for the holes
        in terms of the image shape (value between 0 and 1).
    obj_color: str
        The color of the sample material 
        ("light" if lighter than background, "dark" otherwise).
    

    Returns
    -------
    tuple
        The identified holes (center points.

    """
    
    assert side in sides, f'side must be one of {sides} not {side}' 
    if len(image.shape) != 2:
        image = ensure_image_is_gray(image)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    h, w = image.shape
    if side == 'top':
        image_section = image[:int(h * width_sector + .5), :]
    elif side == 'bottom':
        image_section = image[-int(h * width_sector + .5):, :]
    # elif side == 'left':
    #     image_section = image[:, :int(w * width_sector + .5)]
    # else:  # side == 'right
    #     image_section = image[:, -int(w * width_sector + .5):]
    size = int(scale_hole / depth_section * w + .5)
        
    # search (square-shaped) holes at approx 1 / 3 and 2 / 3
    # convolve with square shaped kernel
    if obj_color == 'dark':
        fill_value = 0
    else:
        fill_value = 255    
        
    image_padded = cv2.copyMakeBorder(
        src=image_section, 
        top=size, bottom=size, left=size, right=size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=fill_value
    )
    image_square_filtered = cv2.blur(image_padded, (size, size))
    
    if plts:
        plt.figure()
        plt.imshow(image_square_filtered)
        plt.show()
        
    image_square_filtered = image_square_filtered[size:-size, size:-size]
    
    # pick max in upper half and lower half
    image_left = image_square_filtered[:, :w//2]
    image_right = image_square_filtered[:, w//2:]
    if obj_color == 'dark':
        hole_left = np.argmax(image_left)
        hole_right = np.argmax(image_right)
    else:
        hole_left = np.argmin(image_left)
        hole_right = np.argmin(image_right)
    hole_left = np.unravel_index(hole_left, image_left.shape)
    hole_right = np.unravel_index(hole_right, image_right.shape)
    hole_right += np.array([0, w // 2])
    # add vertical offset for cropped off part
    if side == 'bottom':
        hole_left += np.array([int((1 - width_sector) * h), 0])
        hole_right += np.array([int((1 - width_sector) * h), 0])
        
    if plts:
        plt.figure()
        plt.imshow(image)
        if side == 'top':
            y = h * width_sector
        else:
            y = h - h * width_sector
        plt.hlines(y, 0, w, colors='red')
        plt.scatter((hole_left[1], hole_right[1]), (hole_left[0], hole_right[0]))
        extent = scale_hole / depth_section * w / 2
        y1, x1 = hole_left - extent 
        y2, x2 = hole_right - extent
        ax = plt.gca()
        ax.add_patch(
            patches.Rectangle((x1, y1), 2 * extent, 2 * extent, linewidth=2, edgecolor='red', facecolor='none')
        )
        ax.add_patch(
            patches.Rectangle((x2, y2), 2 * extent, 2 * extent, linewidth=2, edgecolor='red', facecolor='none')
        )
        plt.show()
        
    return hole_left, hole_right

# %%
if __name__ == '__main__':
    pass