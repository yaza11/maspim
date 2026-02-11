import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw

def get_mask_convex_hull_points(image_shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """

    :param image_shape:
    :param points: in shape (n_points, 2)
    :return:
    """
    hull = ConvexHull(points)
    img = PIL_Image.new('L', image_shape[::-1], 0)
    PIL_ImageDraw.Draw(img).polygon(list(points[hull.vertices].ravel()), outline=1, fill=1)
    return np.array(img).astype(bool)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image_shape = (1000, 5000)
    n_points = 6
    x = (np.random.random(n_points) * image_shape[1])
    y = (np.random.random(n_points) * image_shape[0])
    points = np.c_[x, y]

    mask = get_mask_convex_hull_points(image_shape, points)

    plt.imshow(mask)
    plt.scatter(x, y)
    plt.show()

