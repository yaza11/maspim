from util.cClass import return_existing
from data.cDataClass import Data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import NMF

class MSI(Data):
    def __init__(
            self,
            path_d_folder: str,
            distance_pixels = None
            
    ):
        self.path_d_folder = path_d_folder
        
        if distance_pixels is not None:
            self.distance_pixels = distance_pixels

        self.plts = False
        self.verbose = False

    def set_distance_pixels(self):
        # TODO: parse mis file
        pass
    
    def get_data_columns(self):
        if ('feature_table' not in self.__dict__) or (self.feature_table is None):
            return None
        columns = self.feature_table.columns
        # only return cols with masses
        columns_valid = [col for col in columns if str(
            col).replace('.', '', 1).isdigit()]
        
        data_columns = np.array(columns_valid)
        return data_columns
    
    def plt_img_from_feature_table(self):
        plt.figure()
        if 'RGB' not in self.feature_table.columns:
            self.combine_photo_feature_table()
        img_FT = np.array(
            [np.array(self.feature_table.pivot(
                columns='y', index='x', values=c))
             for c in ('R', 'G', 'B')],
            dtype=np.uint8).T
        plt.imshow(img_FT, interpolation='None')
        plt.show()

    def plt_NMF_photo(self, k, use_repeated_NMF=False, N_rep=30,
                      return_summary=False):
        if 'RGB' not in self.feature_table.columns:
            self.combine_photo_msi()
        FT_s = MaxAbsScaler().fit_transform(
            self.feature_table[['R', 'G', 'B']])

        if use_repeated_NMF:
            from mfe.feature import repeated_nmf
            S = repeated_nmf(FT_s, k, N_rep, max_iter=100_000, init='random')
            W = S.matrix_w_accum
            H = S.matrix_h_accum
        else:
            model = NMF(n_components=k, max_iter=100_000, init='nndsvd')

            W = model.fit_transform(FT_s)
            H = model.components_
        self.plt_NMF(k=k, W=W, H=H)


def test_features():
    o = MSI('490-495', 'Alkenones')
    o.plt_photo()
    o.load_feature_table()
    o.plt_NMF(k=3)
    o.plt_PCA()
    o.plt_kmeans(n_clusters=3)
    o.plt_img_from_feature_table()


if __name__ == '__main__':
    DC = MSI((505, 510), 'FA')
    DC.load()
    DC.plts = True
    DC.sget_photo_ROI()
