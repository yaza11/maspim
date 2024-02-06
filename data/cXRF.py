from data.cDataClass import Data
import imaging.util.Image_convert_types
import res.directory_paths as directory_paths
from res.constants import elements
from util.cClass import return_existing

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table, search_peak_th

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


def handle_video_file(folder, file_name):
    file_path = os.path.join(folder, file_name)
    gray = pd.read_csv(file_path, sep=';', header=None)
    N_y, N_x = gray.shape
    X, Y = np.meshgrid(range(N_x), range(N_y))

    v_x = X.ravel()
    v_y = Y.ravel()
    v_gray = gray.to_numpy().ravel()
    return [v_gray, v_x, v_y]


def txt_to_vec(folder, file_name):
    file_path = os.path.join(folder, file_name)
    df = pd.read_csv(file_path, sep=';', header=None)
    return df.to_numpy().ravel()


class XRF(Data):
    dir_saves: str = directory_paths.absolute_path_to_saves
    dir_pics: str = os.path.join(dir_saves, 'Python Pictures')
    dir_read_data = directory_paths.absolute_path_to_xrf_data
    dir_data = dir_read_data

    def __init__(self, section, window=None, default_file_type='S'):
        # window will be ignored, just so call signature is consistent with MSI
        super().__init__(section, 'xrf')
        self._data_type = 'xrf'
        self._window = 'xrf'  # to be consistent with MSI format
        self.type_to_columns = {self._window: self.get_data_columns()}
        self.default_file_type = default_file_type
        # set the data path, measurement name, folder
        self.find_measurement()

    def find_measurement(self):
        # find the folder containing the interval
        section_min = self._section[0]
        section_max = self._section[1]
        section_length = section_max - section_min
        for folder in os.listdir(self.dir_read_data):
            measurement_number, rest = folder.split('Cariaco')
            interval = rest.split('cm')[0].strip(' _')
            interval_min = int(interval.split('-')[0])
            interval_max = int(interval.split('-')[1])
            if (section_min >= interval_min) and (section_max <= interval_max):
                self.folder = folder
                self.measurement_number = measurement_number.strip(' _')
                self.measurement_idx = (
                    section_min - interval_min) // section_length
                self.measurement_suffix = chr(ord('a') + self.measurement_idx)
                self.measurement_name = self.measurement_number + self.measurement_suffix
                break

        # section 480 to 510 is nested in extra layer
        if self._section_str in ('490-495', '495-500', '500-505', '505-510'):
            self.folder = os.path.join(
                self.folder, f'{self.measurement_name}_{self._section_str}cm')
        self.path_data = os.path.join(self.dir_read_data, self.folder)

    @return_existing('img_original')
    def sget_photo(self):
        if self.img_original is not None:
            return self.img_original
        path = self.path_data
        for file in os.listdir(path):
            if (self._section_str in file) and ('Mosaic.tif' in file):
                self.img_path_original = file
                self.img_original = PIL_Image.open(os.path.join(path, file))
                break
        return self.img_original

    @return_existing('photo_ROI_probe')
    def sget_photo_ROI(self):
        file_name = f'{self.measurement_name}_Video 1.txt'
        v_gray, v_x, v_y = handle_video_file(self.path_data, file_name)
        df = pd.DataFrame({'graylevel': v_gray, 'x': v_x, 'y': v_y})
        img = df.pivot(index='y', columns='x', values='graylevel').to_numpy()
        img = Image_convert_types.convert('np', 'PIL', img)
        self.photo_ROI_probe = img
        return self.photo_ROI_probe

    def get_name_feature_table_file(self, key=None) -> str:
        if key is None:
            key = self.default_file_type
        return f'feature_table_raw_{self._section_str}_{self._window}_{key}.csv'

    def get_data_frames_from_txt(self, file_types=None):
        if file_types is None:
            file_types = [self.default_file_type]
        # find all relevant files
        files = os.listdir(self.path_data)
        print(files)

        feature_tables = {}
        for file_type in file_types:
            vecs = []
            keys = []

            # file names of data are in format
            # [S | D][measurement_number][measurement_key]_[element | Video 1].[txt]
            # example: D0343c_Mg.txt
            for file in files:
                name, suffix = file.split('.')
                # exclude files that have something added to element name
                # (likely modified file)
                if len(name.split('_')) == 2:
                    measurement, element = name.split('_')
                    prefix = measurement[0]
                    if ((prefix == file_type) and (measurement[1:] == self.measurement_name[1:]) and ((len(element) <= 3) or ('Video' in element)) and (suffix == 'txt')):
                        if 'Video' in element:
                            vecs.extend(
                                handle_video_file(self.path_data, file))
                            keys.extend(['graylevel', 'x', 'y'])
                        else:
                            vecs.append(txt_to_vec(self.path_data, file))
                            keys.append(element)
            # combine to feature_table
            FT = pd.DataFrame(data=np.vstack(vecs).T, columns=keys)

            # create column that allows unique identification of pixels
            FT['xy'] = list(zip(FT['x'], FT['y']))

            name_feature_table = self.get_name_feature_table_file(file_type)
            path_feature_table = os.path.join(
                self.dir_saves, 'raw_feature_tables', self._section_str,
                self._window, name_feature_table)

            FT.to_csv(path_feature_table)
            feature_tables[file_type] = FT

        return feature_tables
    
    def get_data_columns(self):
        if ('feature_table' not in self.__dict__) or (self.feature_table is None):
            return None
        columns = self.feature_table.columns
        
        
        # data columns are elements
        columns_valid = [col for col in columns if
                         col in list(elements.Abbreviation)]
    
        data_columns = np.array(columns_valid)
        return data_columns

    def plt_img_from_feature_table(self):
        plt.figure()
        if 'graylevel' not in self.feature_table.columns:
            self.combine_photo_feature_table()
        img_FT = np.array(
            [np.array(self.feature_table.pivot(
                columns='y', index='x', values=c))
             for c in ('R', 'G', 'B')],
            dtype=np.uint8).T
        plt.imshow(img_FT, interpolation='None')
        plt.show()


def test_features():
    o = XRF('490-495', 'S')
    # o.plt_photo()
    o.load_feature_table()
    # o.plt_NMF(k=3)
    # o.plt_PCA()
    o.plt_kmeans(n_clusters=3)
    o.plt_img_from_feature_table()


if __name__ == '__main__':
    # test_features()
    p = XRF('490-495', 'S')
    p.load()
    p.plt_comp('L')
