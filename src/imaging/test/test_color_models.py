from cMSI import MSI
from Image_convert_types import convert, swap_RB
from Image_plotting import plt_cv2_image

import matplotlib.pyplot as plt
import cv2

#  get ROI photo
self = MSI((490, 495), 'Alkenones')
self.load()
BGR = convert('PIL', 'cv', self.sget_photo_ROI())

plt_cv2_image(BGR, 'data ROI RGB')

color_models = ['RGB', 'XYZ', 'YCrCb', 'HSV', 'HLS', 'Lab', 'Luv']

for model in color_models:
    cvt = eval(f'cv2.COLOR_BGR2{model}')
    I3 = cv2.cvtColor(BGR, cvt)
    labels = [letter for letter in model]
    if model == 'YCrCb':
        labels = ['Y', 'Cr', 'Cb']

    for channel in range(3):
        plt.subplot(3, 1, channel + 1)
        slice_img = I3[:, :, channel]
        plt.imshow(slice_img)
        plt.title(f'{labels[channel]}-channel (min={slice_img.min()}, max={slice_img.max()})')
    plt.show()
