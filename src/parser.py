import re
import sqlite3
import struct

import numpy as np
import pandas as pd


def parse_sqlite(peaks_sqlite_path):
    """
    parse the peaks.sqlite file to get the m/z values and intensities for all spectra
    :param peaks_sqlite_path:
    :return:
    """
    conn = sqlite3.connect(peaks_sqlite_path)
    c = conn.cursor()
    df = pd.read_sql_query(
        "SELECT XIndexPos,YIndexPos,PeakMzValues,PeakIntensityValues,NumPeaks from Spectra", conn)
    XX = np.empty(df.shape[0], dtype=[('id', 'O'), ('x', 'O'), ('y', 'O'), ('peak_mz', 'O'), ('peak_int', 'O')])
    for num in range(df.shape[0]):
        if np.floor(num / 1000) == num / 1000:
            print(num, ',', end="\r", flush=True)
        mzs = list(struct.unpack('d' * df['NumPeaks'][num], df['PeakMzValues'][num]))
        intensities = list(struct.unpack('f' * df['NumPeaks'][num], df['PeakIntensityValues'][num]))
        XX[num] = np.array(
            [(num + 1,
              df['XIndexPos'][num],
              df['YIndexPos'][num],
              np.array(mzs, dtype='d'),
              np.array(intensities, dtype='f'))],
            dtype=[('id', 'O'), ('x', 'O'), ('y', 'O'), ('peak_mz', 'O'), ('peak_int', 'O')])
    # get the m/z values and intensities for all spectra
    mzs = np.vstack(XX['peak_mz'])
    intensities = np.vstack(XX['peak_int'])
    xy = np.vstack((XX['x'], XX['y'])).T

    # save the m/z values and intensities for all spectra
    np.save('../test/xy.npy', xy)
    np.save('../test/mzs.npy', mzs)
    np.save('../test/intensities.npy', intensities)

    return xy, mzs, intensities


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


if __name__ == '__main__':
    # xy, mzs, intensities = parse_sqlite('/Users/weimin/Downloads/peaks.sqlite')
    apex0 = parse_acqumethod('/Users/weimin/Downloads/25-30_apexAcquisition.method')
    apex1 = parse_acqumethod('/Users/weimin/Downloads/20-25_apexAcquisition.method')

    # compare the two acquisition methods and find the difference using pretty print
    for key in apex1.keys():
        if apex0[key] != apex1[key]:
            print(key, apex0[key], apex1[key])
