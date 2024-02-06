import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

data = np.random.random(20 * 3).reshape(20, 3)
xp = np.random.random(20) * 2 - 1
yp = np.random.random(20) - .5

points = np.vstack([xp, yp]).T

t = griddata(points, data, (X, Y), method='linear')
t = t.reshape(
    t.shape[0] * t.shape[1],
    t.shape[2])
xt = X.reshape(X.shape[0] * X.shape[1])
yt = Y.reshape(Y.shape[0] * Y.shape[1])
row_mask = ~np.isnan(t).all(axis=1)

t_f = t[row_mask, :]
xt = xt[row_mask]
yt = yt[row_mask]

plt.imshow(t, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
plt.scatter(xp, yp)
plt.show()
