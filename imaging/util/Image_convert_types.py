import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt


def swap_RB(arr):
    arr = arr.copy()
    # swap first and last channel
    # store X channel
    X = arr[:, :, 2].copy()
    # put B in last channel
    arr[:, :, 2] = arr[:, :, 0]
    # put R in first channel
    arr[:, :, 0] = X
    return arr


def infere_mode(image, image_type='cv'):
    """
    Infere channel mode (L, RGB, BGR) form image shape and
    current_image_type.
    """
    # check if grayscale
    if len(image.shape) == 2:
        return 'L'
    # all channels the same
    elif np.all(image[:, :, 0] == image[:, :, 1]) and \
            np.all(image[:, :, 1] == image[:, :, 2]):
        return 'L'
    # only one channel has not all 0
    elif sum([np.all(image[:, :, i] == 0) for i in range(3)]) > 1:
        return 'L'
    elif image_type.lower() in ('cv', 'cv2', 'opencv'):
        return 'BGR'
    elif image_type.lower() in ('pil', 'np', 'numpy', 'pillow'):
        return 'RGB'
    print('infering channel mode inconclusive, set image_type')
    return None


def np_to_PIL(arr):
    arr = (arr / arr.max() * 255).astype(np.uint8)
    return PIL.Image.fromarray(arr, mode=infere_mode(arr))


def np_to_cv(arr, mode_in='BGR'):
    arr = (arr / arr.max() * 255).astype(np.uint8)
    if (infere_mode(arr) == 'L') or (mode_in == 'BGR'):
        return arr
    elif mode_in == 'RGB':
        return swap_RB(arr)


def PIL_to_np(image, mode_out='RGB'):
    if mode_out in ('RGB', 'L'):
        return np.asarray(image)
    elif mode_out == 'BGR':
        return swap_RB(np.asarray(image))
    else:
        return KeyError('output mode has not been implemented, choose one of RGB, BGR, L')


def cv_to_np(image, mode_out='BGR'):
    if mode_out == 'BGR':
        return np.asarray(image)
    elif mode_out == 'RGB':
        return swap_RB(np.asarray(image))


def PIL_to_cv(image):
    if image.mode == 'L':
        arr = PIL_to_np(image, mode_out='L')
    else:
        arr = PIL_to_np(image, mode_out='BGR')
    return np_to_cv(arr)


def cv_to_PIL(image):
    arr = cv_to_np(image, mode_out='RGB')
    return np_to_PIL(arr)


def convert(type_in: str, type_out: str,
            image: object) -> object:
    type_in = type_in.lower()
    if type_in == 'cv2':
        type_in = 'cv'
    elif type_in == 'pil':
        type_in = type_in.upper()
    type_out = type_out.lower()
    if type_out == 'cv2':
        type_out = 'cv'
    elif type_out == 'pil':
        type_out = type_out.upper()
    
    types = ('np', 'cv', 'PIL')
    if (type_in not in types) or (type_out not in types):
        raise ValueError(f'types must be in {types}.')
    elif type_in == type_out:
        return image
    return eval(f'{type_in}_to_{type_out}(image)')


def ensure_image_is_gray(image):
    """
    Return image in grayscale.

    Parameters
    ----------
    image : TYPE, optional
        DESCRIPTION. The default is None.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    # image has three unique channels but channel not specified
    if (channel_mode := infere_mode(image)) != 'L':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image mode is L but image has three channels
    elif (channel_mode == 'L') and (len(image.shape) == 3):
        image = image[:, :, 0]
    # in the last remaining case (image is grayscale and has shape (y, x))
    # image remains unchanged
    return image


if __name__ == '__main__':
    example_img_path = r'C:\Users\yanni\OneDrive\Pictures\Picture1.png'
    PIL_img = PIL.Image.open(example_img_path).convert('RGB')
    PIL_img.show()
    cv_img = cv2.imread(example_img_path)
    cv2.imshow('image', cv_img)
    cv2.waitKey(0)

    PIL_to_cv_img = PIL_to_cv(PIL_img)
    cv2.imshow('image', PIL_to_cv_img)
    cv2.waitKey(0)

    cv_to_PIL_img = cv_to_PIL(cv_img)
    cv_to_PIL_img.show()

    # have to set mode_out to RGB for conversion from cv to np
    cv_to_np_img = cv_to_np(cv_img, mode_out='RGB')
    plt.imshow(cv_to_np_img)
