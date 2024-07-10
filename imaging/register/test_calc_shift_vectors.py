from helpers import interpolate_shifts, get_transect_indices

import time
import numpy as np
import matplotlib.pyplot as plt


def interpolate_shifts_slow(shifts: list[np.ndarray], image_shape: tuple[int, ...], n_transects: int):
    ys = np.linspace(
        0,
        1,
        n_transects + 2,
        endpoint=True
    )[1:-1] * image_shape[0]

    y_grid = np.arange(image_shape[0])

    shift_matrix = np.zeros(image_shape[:2])
    shifts = np.array(shifts)

    for col in range(image_shape[1]):
        func = np.poly1d(np.polyfit(x=ys, y=shifts[:, col], deg=n_transects - 1))
        shift_matrix[:, col] = func(y_grid)

    return shift_matrix


n_transects = 4

# target = skimage.data.cat().mean(axis=-1)
target = np.zeros((3000, 4510))

x = np.linspace(-.5, .5, target.shape[1])
y = np.arange(target.shape[0])
X, Y = np.meshgrid(x, y)

target = np.cos(X * 8 * np.pi) ** 2

# plt.imshow(target)
# plt.show()

# shifts = [np.polyval(np.random.random(4) - .5, x) * 10 for _ in range(n_transects)]
shifts = [
    np.polyval([1, -2, 0], x) * 10,
    np.polyval([0, 0, 0], x) * 10,
    np.polyval([0, 0, 0], x) * 10,
    np.polyval([-1, 0, 0], x) * 10
]
shift_matrix = np.vstack(
    [shifts[i][None, :] * np.ones_like(X[get_transect_indices(i, target.shape, n_transects)])
     for i in range(n_transects)]
)

plt.figure()
plt.imshow(shift_matrix, aspect='auto')
ax2 = plt.gca().twinx()
for i in range(n_transects):
    ax2.plot(shifts[i] + i * 5, 'r')
plt.grid(True)
plt.show()

t0 = time.time()
shift_matrix_i = interpolate_shifts(shifts, target.shape, n_transects)
t1 = time.time()
print(f'fast method took {t1 - t0:.2f} seconds')
shift_matrix_is = interpolate_shifts_slow(shifts, target.shape, n_transects)
t2 = time.time()
print(f'slow method took {t2 - t1:.2f} seconds')

plt.imshow(shift_matrix_i)
plt.show()

# nr, nc, *_ = target.shape
# row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
#                                      indexing='ij')

# source: np.ndarray = skimage.fit.warp(
#     target, np.array([row_coords, col_coords + shift_matrix]),
#     mode='edge'
# )

# plt.imshow(source)

# u = np.zeros_like(target)
