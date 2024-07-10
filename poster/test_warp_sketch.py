import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from skimage import data
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

img1 = data.cat()
img_size = img1.shape

x_distortion = np.sin(np.linspace(0, 2 * np.pi, img_size[1]))[:, None]
y_distortion = np.cos(np.linspace(0, 2 * np.pi, img_size[0]))[None, :]
distortion = x_distortion * y_distortion

# Create 3D grid
x = np.arange(img_size[1])
y = np.arange(img_size[0])
X, Y = np.meshgrid(x, y)

# Plot the distorted image in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, distortion.T, facecolors=plt.cm.gray(img1.mean(axis=-1) / 255))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
