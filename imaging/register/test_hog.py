from skimage.feature import hog
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
img = np.array(Image.open('C:/Users/Yannick Zander/Downloads/2020_03_23_Cariaco_535-540cm_ROI.bmp').convert('L'))


def get_pixels_per_cell(img_shape, n_cells_y = 50, gamma=1.):
    # gamma: ellipsisity of blocks: 
    #   gamma > 1: blocks elongated in x-direction
    #   gamma < 1: blocks elongated in y-direction

    ny, nx = img_shape[:2]

    y_pixels_per_cell = ny / n_cells_y
    x_pixels_per_cell = y_pixels_per_cell * gamma

    return round(y_pixels_per_cell), round(x_pixels_per_cell)


orientations = 4
pixels_per_cell = get_pixels_per_cell(img.shape, n_cells_y=9)

out, hog_image = hog(
    img, 
    orientations=orientations, 
    visualize=True, 
    feature_vector=False, 
    pixels_per_cell=pixels_per_cell,
    cells_per_block=(1, 1),  # default: 3x3,
    transform_sqrt=False
)

# out contains (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient)
# get main direction for each block

# blocks and directions
directions = out.sum(axis=(2, 3))

angles = np.pi * (np.arange(orientations) + 0.5) / orientations

tile = np.zeros((min(pixels_per_cell)) * 2)
s = tile.shape[0]
c = s / 2
ny, nx = directions.shape[:2]


plt.imshow(img)
for i in range(ny):
    for j in range(nx):
        for k, angle in enumerate(angles):
            v = directions[i, j, k] / 10
            xc = s * (j % nx) + c
            yc = s * i + c
            x1 = xc + np.cos(angle) * c
            x2 = xc - np.cos(angle) * c
            y1 = yc + np.sin(angle) * c
            y2 = yc - np.sin(angle) * c

            plt.plot((x1, x2), (y1, y2), c=[v] * 3)
plt.show()


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(angles, directions[3, 7])
# plt.show()

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
hog_image_rescaled[hog_image_rescaled == 0] = np.nan

plt.figure()
plt.axis('off')
plt.imshow(img)
plt.imshow(hog_image_rescaled, cmap='gray')
# plt.set_title('Histogram of Oriented Gradients')
plt.show()
