import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.helpers import transform_feature_table
from imaging.register.helpers import Mapper

# Example data frame with multiple value columns
from skimage.data import checkerboard, horse
from skimage.transform import swirl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# d = checkerboard()
# y, x = np.indices(d.shape)
# xT = swirl(x, strength=2, radius=120, preserve_range=True)
# yT = swirl(y, strength=2, radius=120, preserve_range=True)
# dT = swirl(d, strength=2, radius=120, preserve_range=True)
# df: pd.DataFrame = pd.DataFrame({
#     'x_ROI': x.ravel(),
#     'y_ROI': y.ravel(),
#     'checkerboard': d.ravel(),
#     'swirl': dT.ravel(),
#     'x_ROI_T': xT.ravel(),
#     'y_ROI_T': yT.ravel()
# })
# data_columns: list[str] = 'checkerboard swirl'.split()

# new_df = transform_feature_table(df.copy())

# plt.imshow(df.pivot(index='x_ROI', columns='y_ROI', values='swirl'))
# plt.show()

# plt.imshow(new_df.pivot(index='x_ROI', columns='y_ROI', values='swirl'))
# plt.show()

d = horse()
Y,X = np.indices(d.shape)
XT = X.copy()
YT = np.flipud(Y)

U = XT - X
V = YT - Y

mapper = Mapper(d.shape)
mapper.add_UV(U=U, V=V)

plt.imshow(d)
plt.quiver(X[::50, ::50], Y[::50, ::50], U[::50, ::50], V[::50, ::50])
plt.show()

plt.imshow(mapper.fit(d))
plt.show()

xTp, yTp = mapper.get_transformed_coords()



# folder = r'C:\Users\Yannick Zander\Promotion\Cariaco MSI 2024\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i'

# mapper = Mapper((-1, -1), folder, 'xray')
# mapper.load()


# X, Y = mapper.get_XY()
# U = mapper._Us[0]
# V = mapper._Vs[0]

# plt.quiver(X[::50, ::50], Y[::50, ::50], U[::50, ::50], V[::50, ::50])