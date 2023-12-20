import os
import re
import sqlite3
import struct

import numpy as np
import pandas as pd

from src.func import _find_closest_peak


def parse_sqlite(peaks_sqlite_path, save=True):
    """
    parse the peaks.sqlite file to get the m/z values and intensities for all spectra
    :param peaks_sqlite_path:the folder containing the peaks.sqlite file
    :return:
    """
    # i
    if peaks_sqlite_path.endswith('peaks.sqlite'):
        peaks_sqlite_path = os.path.dirname(peaks_sqlite_path)

    conn = sqlite3.connect(os.path.join(peaks_sqlite_path, 'peaks.sqlite'))
    c = conn.cursor()
    df = pd.read_sql_query(
        "SELECT XIndexPos,YIndexPos,PeakMzValues,PeakIntensityValues,NumPeaks,PeakSnrValues from Spectra", conn)
    XX = np.empty(df.shape[0], dtype=[('id', 'O'), ('x', 'O'), ('y', 'O'), ('peak_mz', 'O'), ('peak_int', 'O'),
                                      ('peak_snr', 'O')])
    for num in range(df.shape[0]):
        if np.floor(num / 1000) == num / 1000:
            print(num, ',', end="\r", flush=True)
        mzs = list(struct.unpack('d' * df['NumPeaks'][num], df['PeakMzValues'][num]))
        intensities = list(struct.unpack('f' * df['NumPeaks'][num], df['PeakIntensityValues'][num]))
        snr = list(struct.unpack('f' * df['NumPeaks'][num], df['PeakSnrValues'][num]))
        XX[num] = np.array(
            [(num + 1,
              df['XIndexPos'][num],
              df['YIndexPos'][num],
              np.array(mzs, dtype='d'),
              np.array(intensities, dtype='f'),
              np.array(snr, dtype='f'))],
            dtype=[('id', 'O'), ('x', 'O'), ('y', 'O'), ('peak_mz', 'O'), ('peak_int', 'O'), ('peak_snr', 'O')])
    # get the m/z values and intensities for all spectra
    mzs = np.vstack(XX['peak_mz'])
    intensities = np.vstack(XX['peak_int'])
    xy = np.vstack((XX['x'], XX['y'])).T
    snr = np.vstack(XX['peak_snr'])

    if save:
        # save the m/z values and intensities for all spectra in the same folder
        np.save(os.path.join(os.path.dirname(peaks_sqlite_path), 'xy.npy'), xy)
        np.save(os.path.join(os.path.dirname(peaks_sqlite_path), 'mzs.npy'), mzs)
        np.save(os.path.join(os.path.dirname(peaks_sqlite_path), 'intensities.npy'), intensities)
        np.save(os.path.join(os.path.dirname(peaks_sqlite_path), 'snr.npy'), snr)

    return xy, mzs, intensities, snr


def parse_acqumethod(path):
    """
    parse the acquisition method file to get the measurement parameters
    :param path:
    :return:
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
    :param target_mz:
    :param xy:
    :param mzs:
    :param intensities:
    :param snr:
    :return:
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


