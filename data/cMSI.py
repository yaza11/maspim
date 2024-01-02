from util.cClass import return_existing
import res.directory_paths as directory_paths
from res.constants import (peak_threshold_ref_peaks_msi_raw_feature_table, 
                       window_to_mass_window)
from data.cDataClass import Data

from mfe.from_txt import (msi_from_txt, get_ref_peaks, create_feature_table, 
                          search_peak_th)

from typing import Iterable
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw
PIL_Image.MAX_IMAGE_PIXELS = None


class MSI(Data):
    dir_saves: str = directory_paths.absolute_path_to_saves
    dir_data: str = os.path.join(dir_saves, 'Data')
    dir_pics: str = os.path.join(dir_saves, 'Python Pictures')
    dir_read_data = directory_paths.absolute_path_to_msi_data

    # class for section FT

    def __init__(
            self,
            section,
            window,
            mass_window=None,
            peak_th_ref_peak_default=peak_threshold_ref_peaks_msi_raw_feature_table
    ):
        super().__init__(section=section, window=window, data_type='msi')

        self._mass_window = self.get_mass_window(mass_window)
        # define the peak_th used for the object
        self.peak_th_ref_peak = peak_th_ref_peak_default

        self.plts = False
        self.verbose = False

    def get_mass_window(self, mass_window):
        """
        Set mass window of object.

        This Python function sets the mass window for a given object.
        If the mass_window parameter is provided, it updates the mass window
        value. If mass_window is None, it returns the mass window value
        associated with the current window type.

        Parameters
        ----------
        mass_window : TYPE
            The mass window value to be set.

        Returns
        -------
        TYPE
            If mass_window is None, the function returns the mass window
            value associated with the current window type.
            Otherwise, it returns the updated mass_window value.

        """
        if mass_window is None:
            return window_to_mass_window(self._window)
        return mass_window

    @return_existing('dict_file_paths')
    def sget_directory_paths(self) -> dict[str, str]:
        """
        Get directory paths for disc files.

        Returns
        -------
        dict
            A dictionary containing the following directory paths:
            'dir_raw_data': The directory path for raw data.
            'mis_path': The directory path for MIS data.
            'path_txt_in': The directory path for input text data.
            'path_txt_out': The directory path for output text data.

        """
        for dot_i in os.listdir(os.path.join(
                self.dir_read_data, f'{self._section_str}cm')):
            if (self._section_str in dot_i) and (self._window.lower() in dot_i.lower()):
                name_folder_data = dot_i
        mass_window_str = self.get_section_formatted(self._mass_window)[1]
        self.dict_file_paths: dict[str, str] = {
            'dir_raw_data': os.path.join(
                self.dir_read_data,
                f'{self._section_str}cm',
                name_folder_data
            ),
            'mis_path': self.find_mis_file(name_folder_data),
            'path_txt_in': os.path.join(
                self.dir_data, self._section_str, self._window,
                f'{self._section_str}_{self._window}_mw{mass_window_str}.txt'
            ),
            'path_txt_out': os.path.join(
                self.dir_data, self._section_str, self._window,
                f'{self._section_str}_{self._window}_mw{mass_window_str}_dots.txt'
            )
        }
        return self.dict_file_paths

    def find_mis_file(self, name_folder_data):
        # path to folder with mis file
        path = os.path.join(
            self.dir_read_data,
            f'{self._section_str}cm',
            name_folder_data
        )
        # how the file should be named
        target_file = name_folder_data[:-1] + 'mis'
        best = directory_paths.find_best_match_file(path, target_file)
        if (best != target_file) and self.verbose:
            print(f'could not find {target_file}, using {best}')
        return os.path.join(path, best)

    @property
    def path_txt_in(self):
        return self.sget_directory_paths()['path_txt_in']

    @property
    def path_txt_out(self):
        return self.sget_directory_paths()['path_txt_out']

    def write_txt_out(self):
        """
        Convert commas to dots in txt-file.

        This Python function writes data from an input text file to an output
        text file. If the output file does not already exist, it creates a
        new file and replaces commas with dots in each line of the input file
        before writing to the output file. If the output file already exists,
        it prints a message indicating that the file already exists.

        Returns
        -------
        None.

        """
        if not os.path.exists(self.path_txt_out):
            print('writing new txt file')
            # replace comma by dot in each line
            with open(self.path_txt_in, 'r') as file_in,\
                    open(self.path_txt_out, 'w') as file_out:
                for line in file_in:
                    file_out.write(line.replace(',', '.'))
        else:
            print('out file already exists')

    def search_peak_th(self, peak_th_ref_peaks):
        """
        Search peak threshold for msi data.

        Plot different parameters important for evaluating the peak threshold
        for creating the featuer table.

        Parameters
        ----------
        peak_th_ref_peaks : list[float]
            list of peak_th to check.

        Returns
        -------
        None.

        """
        # make sure commas are converted to dots
        self.write_txt_out()
        spectra = msi_from_txt(self.path_txt_out)
        print('finished msi_from_txt')

        dict_out = search_peak_th(spectra, peak_th_ref_peaks)

        # plot result
        x = np.array(peak_th_ref_peaks)
        for metric, values in dict_out.items():
            if metric == 'n_ref':
                values = [len(value) for value in values]
            print(metric, values)
            plt.plot(x, values / np.max(values), '-o',
                     label=f'{metric}, max={np.max(values):.2f}')
        plt.xlabel('peak th')
        plt.ylabel('metric relative to max')
        plt.legend()
        plt.title(
            f'section: {self._section_str}, window: {self._window},\n \
mass_window: {self._mass_window}')
        plt.show()

    def get_data_frames_from_txt(
            self,
            peak_th_ref_peaks: float,
            ref_peaks: None | Iterable[float] = None
    ) -> pd.DataFrame:
        """
        For given peak ths create feature tables from txt.

        Parameters
        ----------
        peak_th_ref_peaks : list[float]
            the thresholds for which the feature tables will be created.

        Returns
        -------
        feature_tables : dict(float, pd.DataFrame)
            keys: peak_th, values: feature_tables to peak_th.

        """
        # make sure commas are converted to dots
        self.write_txt_out()
        spectra = msi_from_txt(self.path_txt_out)
        print('finished msi_from_txt')
        if ref_peaks is None:
            ref_peaks = get_ref_peaks(spectra, peak_th=peak_th_ref_peaks)

        feature_tables = {}
        for peak_th in peak_th_ref_peaks:
            feature_table, _ = create_feature_table(
                spectra,
                ref_peaks[peak_th],
                normalization='median')

            feature_table = feature_table.sort_values(by=['y', 'x'])

            # file path to where feature_table will be saved
            name_feature_table = self.get_name_feature_table_file(peak_th)
            path_feature_table = os.path.join(
                self.dir_saves, 'raw_feature_tables', self._section_str,
                self._window, name_feature_table)

            feature_table.to_csv(path_feature_table)
            feature_tables[peak_th] = feature_table
        return feature_tables

    def get_name_feature_table_file(self, key=None) -> str:
        if key is None:
            key = self.peak_th_ref_peak
        return f'feature_table_raw_{self._section_str}_{self._window}_\
{str(key).replace(".", "dot")}.csv'

    def search_keys_in_xml(self, keys):
        # iniate list of lists for values
        out_dict = {key: [] for key in keys}
        # open xml
        with open(self.sget_directory_paths()['mis_path']) as xml:
            # parse through lines
            for line in xml:
                line = line.replace('/', '')
                # search for keys
                for key in keys:
                    key_xml = f'<{key}>'
                    if key_xml in line:
                        value = line.split(key_xml)[1]
                        out_dict[key].append(value)
        for key, value in out_dict.items():
            if len(value) == 1:
                out_dict[key] = value[0]
        return out_dict

    @return_existing('img_original')
    def sget_photo(self):
        mis_dict = self.search_keys_in_xml(['ImageFile'])
        self.img_path_original = os.path.join(
            self.sget_directory_paths()['dir_raw_data'], mis_dict['ImageFile'])
        self.img_original = PIL_Image.open(self.img_path_original)
        return self.img_original

    @return_existing('photo_ROI_probe')
    def sget_photo_ROI(self, match_pxls=True):
        # search the mis file for the point data and image file
        mis_dict = self.search_keys_in_xml(['Point'])
        img = self.sget_photo()

        points_mis = mis_dict['Point']
        points = []
        # get the points of the defined area
        for point in points_mis:
            p = (int(point.split(',')[0]), int(point.split(',')[1]))
            points.append(p)

        if self.plts:
            img_rect = img.copy()
            draw = PIL_ImageDraw.Draw(img_rect)
            draw.polygon(points, outline=(255, 0, 0), width=round(np.min(img_rect._size) / 50))
            plt.figure()
            plt.imshow(img_rect, interpolation='None')
            plt.show()

        # get the extent of the image
        points_x = [p[0] for p in points]
        points_y = [p[1] for p in points]

        x_min_area = np.min(points_x)
        x_max_area = np.max(points_x)
        y_min_area = np.min(points_y)
        y_max_area = np.max(points_y)

        # get extent of data points in feature table
        x_min_FT = self.current_feature_table.x.min()
        x_max_FT = self.current_feature_table.x.max()
        y_min_FT = self.current_feature_table.y.min()
        y_max_FT = self.current_feature_table.y.max()

        # resize region in photo to match data points
        if match_pxls:
            img_resized = img.resize(
                (x_max_FT - x_min_FT + 1, y_max_FT - y_min_FT + 1),  # new number of pixels
                box=(x_min_area, y_min_area, x_max_area, y_max_area),  # area of photo
                resample=PIL_Image.Resampling.LANCZOS  # supposed to be best
            )
        else:
            img_resized = img.crop(
                (x_min_area, y_min_area, x_max_area, y_max_area))
        # xywh of data ROI in original image, photo units
        xp = x_min_area
        yp = y_min_area
        wp = x_max_area - x_min_area
        hp = y_max_area - y_min_area
        # xywh of data, data units
        xd = x_min_FT
        yd = y_min_FT
        wd = x_max_FT - x_min_FT
        hd = y_max_FT - y_min_FT

        self.photo_ROI_xywh = (xp, yp, wp, hp)  # photo units
        self.data_ROI_xywh = (xd, yd, wd, hd)  # data units
        self.photo_ROI_probe = img_resized
        return self.photo_ROI_probe

    def power_spectrum_mz(self, mz):
        from scipy.signal import detrend, blackman
        from astropy.timeseries import LombScargle
        feature_table_averages_layers = self.zone_wise_average()
        t = feature_table_averages_layers.age
        y = feature_table_averages_layers[mz]
        y_d = detrend(y)
        y_w = y_d * blackman(len(t))

        self.plt_mz_img(mz)
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(t, y, label='original')
        axs[0].plot(t, y_d, label='detrended')
        axs[0].plot(t, y_w, label='blackman')
        axs[0].legend()

        f, spec = LombScargle(t, y_d).autopower()
        f_w, spec_w = LombScargle(t, y_w).autopower()
        # power
        p = np.abs(spec) ** 2
        p_w = np.abs(spec_w) ** 2

        axs[1].plot(f, p)
        axs[1].plot(f_w, p_w)
        axs[1].vlines(1, ymin=0, ymax=np.max(p), color='black', linestyle='--')
        plt.show()

    def plt_img_from_feature_table(self):
        plt.figure()
        if 'RGB' not in self.current_feature_table.columns:
            self.combine_photo_feature_table()
        img_FT = np.array(
            [np.array(self.current_feature_table.pivot(
                columns='y', index='x', values=c))
             for c in ('R', 'G', 'B')],
            dtype=np.uint8).T
        plt.imshow(img_FT, interpolation='None')
        plt.show()

    def plt_NMF_photo(self, k, use_repeated_NMF=False, N_rep=30,
                      return_summary=False):
        if 'RGB' not in self.current_feature_table.columns:
            self.combine_photo_msi()
        FT_s = MaxAbsScaler().fit_transform(
            self.current_feature_table[['R', 'G', 'B']])

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
