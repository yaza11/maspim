import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import skimage

from scipy.signal import fftconvolve

from imaging.register.tilt_descriptor import Descriptor, rect
from imaging.align_net.synthetic_images import get_image_pair

def test_real():
    from PIL import Image
    # from PIL.ImageOps import autocontrast
    from skimage.filters import threshold_otsu

    img = np.array(Image.open('C:/Users/Yannick Zander/Downloads/2020_03_23_Cariaco_535-540cm_ROI.bmp').convert('L'))
    # img = np.array(Image.open('C:/Users/Yannick Zander/Downloads/S0363b_Cariaco_525-530cm_100um_0001_roi.tif').convert('L'))
    # img = np.array(Image.open('C:/Users/Yannick Zander/Downloads/2018_08_27 Cariaco 490-495 alkenone_0002.tif').convert('L'))

    return img


def test_synthetic():
    """Synthetic image"""
    image, mask = get_image_pair((200, 500))
    return image, mask


def test_cl():
    """Classified image"""
    import pickle
    from skimage.transform import resize
    file = r'F:/535-540cm/2020_03_23_Cariaco_535-540cm_Alkenones.i/ImageClassified.pickle'
    with open(file, 'rb') as f:
        d = pickle.load(f)

    img = d._image_classification.astype(int)
    img[img == 255] = 1
    img[img == 127] = -1
    # img[img == 0] = -1

    # force width of 500 pixels
    k = 500 / img.shape[0]
    img_shape = round(k * img.shape[0]), round(k * img.shape[1])
    img = resize(img, img_shape)

    mask = img == 0

    return img, mask


def test_image(theta=np.pi / 4) -> np.ndarray:
    """Sine"""
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.linspace(0, np.pi, 200)
    xx, yy = np.meshgrid(x, y)
    n = 10
    zz = np.sin(xx * np.cos(theta) * n + yy * np.sin(theta) * n)
    print(f'period: {400 / n} pixels')

    return zz

def test_image2(theta=0):
    """Stripes"""
    img = test_image(theta)
    return (img > 0).astype(float) * 2 - 1


def test_image3():
    """Horse"""
    from skimage.data import horse
    kernel = 1 - horse().astype(float)

    img = np.pad(1 - horse().astype(float), ((200, 150), (120, 220)))
    return img, kernel


def test_image4():
    """Cross"""
    dim = 400
    width = dim//20
    img = np.zeros((dim, dim)) - 1
    
    img[:, dim//2-width:dim//2+width] = 1
    img[dim//2-width:dim//2+width, :] = 1
    
    return img


def test_image5():
    """Rings"""
    def ring(r1, r2):
        x = np.arange(dim) - dim / 2
        xx, yy = np.meshgrid(x, x)
        circle = (xx ** 2 + yy ** 2 < r1 ** 2) & (xx ** 2 + yy ** 2 > r2 ** 2)
        return circle
    
    dim = 400
    img = np.full((dim, dim), 0)
    # img += ring(dim / 2, dim/2 - dim / 10)
    img += ring(dim / 3, dim/ 3 - dim / 20)
    # img += ring(dim / 4, dim/ 4 - dim / 30)
    return img

# import skimage
# image = skimage.data.brick()
image = test_real()
image = np.repeat(image[:, :, None], repeats=3, axis=-1)

# image, mask = test_synthetic()
# image, mask = test_cl()
# image = test_image(theta=-np.pi/8)
# image, kernel = test_image3()
# image = test_image4()
# image = test_image5()

mask = None
# mask = np.ones(image.shape[:2], dtype=bool)

# plt.imshow(image)
# plt.show()

downscale_factor: float = min((400 / image.shape[1], 1))
downscaled_shape = (
    round(image.shape[0] * downscale_factor),
    round(image.shape[1] * downscale_factor)
)
image_downscaled: np.ndarray = skimage.transform.resize(
    image, downscaled_shape
)

d = Descriptor(
    image=image_downscaled,
    mask=mask,
    n_sizes=8,
    n_angles=32,
    n_phases=8,
    max_period=.1
)

# plt.imshow(image)
# plt.show()
# plt.imshow(d.image_processed)

# d.plot_kernels()
# d.plot_kernel_on_img()
# d.plot_parameter_images()
# d.plot_quiver(40)

d.set_conv_chunk()
d.fit()

plt.imshow(d._get_inverse_shift_matrix(image.shape))
plt.show()
plt.imshow(d._get_shift_matrix(image.shape))
plt.show()

warped = d.transform(image)
plt.imshow(warped)
plt.show()

iwarped = d.transform(warped, is_inverse=True)
plt.imshow(iwarped)
plt.show()


