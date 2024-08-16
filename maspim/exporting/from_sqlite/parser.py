import os
import re
import sqlite3
import struct

import numpy as np
import pandas as pd

from maspim.exporting.from_sqlite.func import _find_closest_peak, estimateThreshold


def parse_sqlite(peaks_sqlite_path, save=True, save_path=None, keep_all=False):
    """
    parse the peaks.sqlite file to get the m/z values and intensities for all spectra
    :param save_path: the folder to save the m/z values and intensities for all spectra, if None, save in the same folder as the peaks.sqlite file
    :param save: if True, save the m/z values and intensities for all spectra
    :param peaks_sqlite_path:the folder containing the peaks.sqlite file
    :return: xy, mzs, intensities, snr

    code example:
    .. code-block:: python
        xy, mzs, intensities, snr = parse_sqlite('path/to/peaks.sqlite', save=True, save_path='path/to/save')
    """

    # if the path is a peaks.sqlite file, get the folder
    if peaks_sqlite_path.endswith('peaks.sqlite'):
        peaks_sqlite_path = os.path.dirname(peaks_sqlite_path)

    conn = sqlite3.connect(os.path.join(peaks_sqlite_path, 'peaks.sqlite'))
    df = pd.read_sql_query(
        "SELECT XIndexPos,YIndexPos,PeakMzValues,PeakIntensityValues,NumPeaks,PeakSnrValues, PeakFwhmValues from Spectra", conn)
    raw_sql = np.empty(df.shape[0], dtype=[('id', 'O'), ('x', 'O'), ('y', 'O'), ('peak_mz', 'O'), ('peak_int', 'O'),
                                      ('peak_snr', 'O'), ('peak_fwhm', 'O')])
    for num in range(df.shape[0]):
        mzs = np.array(list(struct.unpack('d' * df['NumPeaks'][num], df['PeakMzValues'][num])))
        intensities = np.array(list(struct.unpack('f' * df['NumPeaks'][num], df['PeakIntensityValues'][num])))
        snr = np.array(list(struct.unpack('f' * df['NumPeaks'][num], df['PeakSnrValues'][num])))
        fwhm = np.array(list(struct.unpack('f' * df['NumPeaks'][num], df['PeakFwhmValues'][num])))

        if not keep_all:
            threshold = estimateThreshold(fwhm, mzs)
            real_peaks = fwhm / mzs ** 2 > threshold
            # make the intensities of the peaks that are not real to be 0
            intensities = intensities * real_peaks
            snr = snr * real_peaks
        raw_sql[num] = np.array(
            [(num + 1,
              df['XIndexPos'][num],
              df['YIndexPos'][num],
              np.array(mzs, dtype='d'),
              np.array(intensities, dtype='f'),
              np.array(snr, dtype='f'),
              np.array(fwhm, dtype='f'))],
            dtype=[('id', 'O'), ('x', 'O'), ('y', 'O'), ('peak_mz', 'O'), ('peak_int', 'O'), ('peak_snr', 'O'),
                   ('peak_fwhm', 'O')])


    # get the m/z values and intensities for all spectra
    mzs = np.vstack(raw_sql['peak_mz'])
    intensities = np.vstack(raw_sql['peak_int'])
    xy = np.vstack((raw_sql['x'], raw_sql['y'])).T
    snr = np.vstack(raw_sql['peak_snr'])
    fwhm = np.vstack(raw_sql['peak_fwhm'])

    if save:
        if save_path is None:
            save_path = peaks_sqlite_path
        # save the m/z values and intensities for all spectra in the specified folder
        np.save(os.path.join(save_path, 'xy.npy'), xy)
        np.save(os.path.join(save_path, 'mzs.npy'), mzs)
        np.save(os.path.join(save_path, 'intensities.npy'), intensities)
        np.save(os.path.join(save_path, 'snr.npy'), snr)
        np.save(os.path.join(save_path, 'fwhm.npy'), fwhm)

    return xy, mzs, intensities, snr, fwhm


def parse_acqumethod(path):
    """
    parse the acquisition method file to get the measurement parameters
    :param path: the path to the acquisition method file
    :return: a dictionary containing the measurement parameters

    code example:
    .. code-block:: python
        params = parse_acqumethod('path/to/acqumethod')
    """
    # get all the lines between <param name = *> and </param>
    with open(path, 'r') as f:
        lines = f.readlines()
    # keep the lines between <paramlist> and </paramlist>
    lines = lines[lines.index('<paramlist>\n') + 1:lines.index('</paramlist>\n')]
    params = {}
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    lines = (''.join(lines)).split('</param>')
    lines = [line for line in lines if line.startswith('<param name="')]

    for line in lines:
        # get the parameter name
        name = re.findall(r'name="(.*)"><value', line)[0]
        # get the number of values by counting the number of </value> tags
        num_val = len(re.findall('</value>', line))
        if num_val == 1:
            # if there is only one value, get the value
            try:
                val = float(re.findall(r'(-?\d+(?:\.\d+)?)</value>', line)[0])
            except IndexError:
                # if it is not a number, get the string
                val = re.findall(r'<value>(.*?)</value>', line)[0]
        else:
            raise NotImplementedError('num_val > 1')
        params[name] = val
    return params


def extract_mzs(target_mz, xy, mzs, intensities, snrs, tol=0.01, min_int=10000, min_snr=0):
    """
    extract the target m/z values and intensities for all spectra
    :param target_mz: the m/z values to be extracted, either a list of m/z values or a single m/z value
    :param xy:
    :param mzs:
    :param intensities:
    :param snr:
    :return: a dataframe containing the target m/z values and intensities for all spectra, and the ppm error for each
    target m/z value, as well as the x and y coordinates

    code example:
    .. code-block:: python
        df = extract_mzs(target_mz, xy, mzs, intensities, snrs, tol=0.01, min_int=10000, min_snr=0)
    """
    if isinstance(target_mz, float):
        mz = np.zeros(xy.shape[0])
        intensity = np.zeros(xy.shape[0])
        snr = np.zeros(xy.shape[0])
        for i in range(xy.shape[0]):
            mz[i], intensity[i], snr[i] = _find_closest_peak(target_mz, tol, mzs[i], intensities[i], snrs[i],
                                                             min_int=min_int, min_snr=min_snr)
    # TODO: a lazy solution, probably need to be improved for speed
    elif isinstance(target_mz, list):
        mz = np.zeros((xy.shape[0], len(target_mz)))
        intensity = np.zeros((xy.shape[0], len(target_mz)))
        snr = np.zeros((xy.shape[0], len(target_mz)))
        for i in range(xy.shape[0]):
            for j in range(len(target_mz)):
                mz[i, j], intensity[i, j], snr[i, j] = _find_closest_peak(target_mz[j], tol, mzs[i], intensities[i],
                                                                          snrs[i], min_int=min_int, min_snr=min_snr)

    else:
        raise NotImplementedError('target_mz must be either a float or a list of floats')
    # compile everything into a dataframe
    intensity_df = pd.DataFrame(intensity, columns=['intensity_'+str(mz) for mz in target_mz])
    intensity_df['x'] = xy[:, 0]
    intensity_df['y'] = xy[:, 1]

    quality_df = pd.DataFrame(snr, columns=['snr_'+str(mz) for mz in target_mz])
    quality_df['x'] = xy[:, 0]
    quality_df['y'] = xy[:, 1]
    # attach the quality df
    for i in range(len(target_mz)):
        quality_df['ppm_'+str(target_mz[i])] = (mz[:, i] - target_mz[i]) / target_mz[i] * 1e6

    df = pd.merge(intensity_df, quality_df, on=['x', 'y'])
    # move the x and y columns to the front
    cols = df.columns.tolist()
    col = cols.pop(cols.index('x'))
    cols.insert(0, col)
    col = cols.pop(cols.index('y'))
    cols.insert(1, col)
    df = df[cols]
    return df


if __name__ == '__main__':
    raise NotImplementedError('This is not a script')


