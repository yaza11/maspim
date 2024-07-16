# from imaging.register.helpers import get_transect_indices
# from imaging.register.main import Transformation

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import rank

# im = np.asarray(Image.open(r'C:/Users/Yannick Zander/Nextcloud2/Promotion/msi_workflow/imaging/register/image_classified.png'))


im2 = np.array(Image.open(r'C:/Users/Yannick Zander/Nextcloud2/Promotion/msi_workflow/imaging/register/2020_03_23_Cariaco_535-540cm_Fullerite_0001.bmp').convert('L'))

_, mask = cv2.threshold(im2, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

proc = rank.median(
    image=im2,
    footprint=np.ones((10, 10)),
    mask=mask
)

proc2 = im2.copy()
mask_ = np.abs(im2 - proc) > 10
proc2[mask_] = proc[mask_]

# plt.imshow(im2, cmap='gray')
# plt.show()
# plt.imshow(mask)
# plt.show()
# plt.imshow(proc, cmap='gray')
# plt.show()

# plt.imshow(proc2, cmap='gray')
# plt.show()

n_transects = 5

transect_ys = np.linspace(0, im2.shape[0], n_transects + 2, endpoint=True)[1:-1]
transect_width = np.diff(transect_ys)[0]

plt.imshow(im2)
plt.hlines(transect_ys, 0, im2.shape[1], colors='r')
for transect_y in transect_ys:
    plt.plot((1 - im2[int(transect_y), :] / 255) * transect_width + transect_y - transect_width / 2, color='k')


plt.imshow(proc)
plt.hlines(transect_ys, 0, im2.shape[1], colors='r')
for transect_y in transect_ys:
    plt.plot((1 - proc[int(transect_y), :] / 255) * transect_width + transect_y - transect_width / 2, color='k')
