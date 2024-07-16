import os
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

from msi_workflow.imaging.register.finding_global_shift import warp_with_params, interpolate_shifts



def wave_pattern_easy(img_shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    img_height, img_width = img_shape
    # superimposed sines
    n_periods = [5, 7, 20, 25]
    amplitudes = [100, 100, 20, 20]
    wave = np.zeros(img_width)
    x = np.linspace(0, 1, img_width)
    for period, amplitude in zip(n_periods, amplitudes):
        dperiod = np.random.uniform(low=-.5, high=.5)
        phase = np.random.uniform(low=-np.pi, high=np.pi)
        damplitude = np.random.uniform(low=.8, high=1.2)
        wave += damplitude * amplitude * np.cos(2 * np.pi * (period + dperiod) * x + phase)
    # rescale values to be between 50 and 255
    wave = (wave - wave.min()) / (wave.max() - wave.min()) * (255 - 50) + 50
    wave = wave ** .5

    img = wave[None, :] * np.ones(img_height)[:, None]
    return img

def smooth(vec: np.ndarray, kernel: int = 7) -> np.ndarray:
    return np.convolve(vec, np.ones(kernel) / kernel, mode='same')

def outside_missing_mask(img_shape):
    # set values missing at the outside
    img_height, img_width = img_shape
    
    x = np.arange(img_width)
    y = np.arange(img_height)
    
    X, Y = np.meshgrid(x, y)
    
    # random walk
    mask = np.ones(img_shape, dtype=bool)
    
    top_bound = img_height * np.random.uniform(low=0, high=.1) + np.cumsum(np.random.uniform(low=-1, high=1, size=img_width))
    bot_bound = img_height * np.random.uniform(low=.9, high=1) + np.cumsum(np.random.uniform(low=-1, high=1, size=img_width))
    
    left_bound = np.cumsum(np.random.uniform(low=-1, high=1, size=img_height))
    right_bound = img_width + np.random.uniform(low=.9, high=1) + np.cumsum(np.random.uniform(low=-1, high=1, size=img_height))
    
    top_bound = smooth(top_bound)
    bot_bound = smooth(bot_bound)
    left_bound = smooth(left_bound)
    right_bound = smooth(right_bound)
    
    top = top_bound[None, :] * np.ones(img_height)[:, None]
    bot = bot_bound[None, :] * np.ones(img_height)[:, None]
    
    left = left_bound[:, None] * np.ones(img_width)[None, :]
    right = right_bound[:, None] * np.ones(img_width)[None, :]
    
    mask[top > Y] = False
    mask[bot < Y] = False
    mask[left > X] = False
    mask[right < X] = False
    
    return mask


def fractures(img_shape):
    # fractures accure predominantly vertical to depths
    img_height, img_width = img_shape
    
    # number features
    n_fractures = int(np.random.uniform(10, 1000))
    # n_fractures = 500
    # center points
    xc = np.random.uniform(0, img_width, n_fractures)
    yc = np.random.uniform(0, img_height, n_fractures)
    # extents
    ry = img_height * np.random.beta(1, 10, size=n_fractures)
    rx = img_width / 80 * np.random.beta(1, 5, size=1000)

    mask = np.ones(img_shape, dtype=bool)
    
    x = np.arange(img_width)
    y = np.arange(img_height)
    
    X, Y = np.meshgrid(x, y)
        
    for i in range(n_fractures):
        mask[(X - xc[i]) ** 2 / rx[i] ** 2 + (Y - yc[i]) ** 2 / ry[i] ** 2 <= 1] = False
    
    # add 0 to 5 missing intervals
    n_missing = round(np.random.uniform(0, 5))
    for m in range(n_missing):
        height = img_width * np.random.uniform(1 / 100, 1 / 50)
        xc = img_width * np.random.uniform(.2, .8)
        mask[(X > (xc - height)) & (X < (xc + height))] = False
    
    return mask


def warped_image(image):
    n_transects: int = 4
    degree: int = 4
    
    f: float = image.shape[1] / 50
    # f = 1
    params_in: np.ndarray = np.array([
        (np.random.random(degree + 1) - .5) * f for _ in range(n_transects)
    ])

    image_warped = warp_with_params(image=image, params=params_in, target_shape=image.shape, n_transects=n_transects, degree=degree)
    
    x = np.linspace(-.5, .5, image.shape[1])
    shifts: list[np.ndarray] = [
            np.polyval(params_in[i, :], x=x) for i in range(n_transects)
    ]
    shifts: np.ndarray = interpolate_shifts(shifts, image.shape, n_transects)
            
    return image_warped, shifts

def training_pair(img_shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    mask = outside_missing_mask(img_shape) & fractures(img_shape)
    wave = wave_pattern_easy(img_shape).astype(float)
    wave /= wave.max()
    
    expected = wave * mask.astype(int)

    warped, flow = warped_image(expected)

    # zeropad to make square shaped
    # expected = np.pad(expected, ((0, img_shape[1] - img_shape[0]), (0, 0)))
    # warped = np.pad(warped, ((0, img_shape[1] - img_shape[0]), (0, 0)))
    #
    # expected = (resize(expected, (512, 512)) * 255).astype(np.uint8)
    # warped = (resize(warped, (512, 512)) * 255).astype(np.uint8)

    # expected = (expected * 255).astype(np.uint8)
    warped = (warped * 255).astype(np.uint8)
    flow = np.around(flow, 0).astype(int)

    return warped, flow


def get_image_pair(img_shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    mask = outside_missing_mask(img_shape) & fractures(img_shape)
    wave = wave_pattern_easy(img_shape).astype(float)
    wave /= wave.max()

    expected = wave * mask.astype(int)

    warped, _ = warped_image(expected)
    warped = (warped * 255).astype(np.uint8)

    warped_mask = warped_image(mask.astype(float))[0]
    warped_mask = warped_mask > 0

    return warped, warped_mask


def create_training_data(N: int, folder: str):
    for i in range(N):
        print(f"{i+1} out of {N}")
        warped, flow = training_pair(img_shape)
        # e = Image.fromarray(e)
        np.save(os.path.join(folder, 'masks', f'{i:05}_mask.npy'), flow)
        warped = Image.fromarray(warped)
        warped.save(os.path.join(folder, 'imgs', f'{i:05}.png'))



if __name__ == '__main__':
    img_height, img_width = 200, 800
    # img_height, img_width = 512, 512
    img_shape = (img_height, img_width)
    
    warped, flow = training_pair(img_shape)
    
    fig, axs = plt.subplots(nrows = 2, sharex=True)
    
    axs[0].imshow(warped)
    axs[0].set_title('Input image')
    
    axs[1].imshow(flow)
    axs[1].set_title('Expected output')
    
    plt.show()

    create_training_data(N=1000, folder=r'C:\Users\Yannick Zander\Downloads\Pytorch-UNet\data')
