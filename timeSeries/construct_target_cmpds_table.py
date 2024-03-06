import logging
import os.path
import re
import sqlite3

import numpy as np
import pandas as pd

from exporting.legacy.parser import extract_mzs


def get_mz_int_depth(DA_txt_path, db_path, target_cmpds=None):
    # parse the spectrum file name from the path
    spec_file_name = os.path.basename(DA_txt_path).replace('.txt', '')
    # extract the target compounds from exported_txt_path
    df = extract_mzs(target_cmpds, DA_txt_path, tol=0.01, min_snr=1)

    # connect to the sqlite database
    conn = sqlite3.connect(db_path)
    # create a view using the spec_id from both tables, spec_file_name from table metadata, spot_name from metadata, and
    # xray_array and linescan_array from table transformation
    conn.execute('''
    CREATE VIEW IF NOT EXISTS dataview AS
    SELECT metadata.spec_id, metadata.spec_file_name, metadata.spot_name, transformation.xray_array, transformation.linescan_array
    FROM metadata
    INNER JOIN transformation
    ON metadata.spec_id = transformation.spec_id
    ''')

    # get the spot_name, xray_array, and linescan_array from the view, where spec_file_name is the same as the one from the
    # exported_txt_path
    query = f'''
    SELECT spot_name, xray_array, linescan_array
    FROM dataview
    WHERE spec_file_name = '{spec_file_name}'
    '''
    coords = conn.execute(query).fetchall()[0]
    spot_names = coords[0].split(',')
    # in every spot_names, only preserve string 'R(0-9)+X(0-9)+Y(0-9)+'
    spot_names = [re.findall(r'R\d+X\d+Y\d+', spot_name)[0] for spot_name in spot_names]
    spot_names = pd.DataFrame(spot_names, columns=['spot_name'])
    xray_array = np.frombuffer(coords[1], dtype=np.float64).reshape(-1, 2)
    xray_array = pd.DataFrame(xray_array, columns=['px', 'py'])
    linescan_array = np.frombuffer(coords[2], dtype=np.float64).reshape(-1, 2)
    linescan_array = pd.DataFrame(linescan_array[:, 0], columns=['d'])
    # merge all the dataframes
    coords = pd.concat([spot_names, xray_array, linescan_array], axis=1)
    # joint the coords and df on 'spot_name'
    df = pd.merge(coords, df, on='spot_name')
    # only keep the successful spectrum, where both GDGT_0 and GDGT_5 are present
    logging.debug(f"Successful rate: {df.dropna().shape[0] / df.shape[0]:.2f}")
    return df


def find_optimal_interval_for_irregular_data(depth, min_n_samples=40, alpha=0.1):
    # Ensure the data is sorted
    sorted_data = sorted(depth)

    # Calculate differences between consecutive data points
    differences = [sorted_data[i + 1] - sorted_data[i] for i in range(len(sorted_data) - 1)]

    # Start with the smallest non-zero difference as the initial interval size
    initial_interval_size = min(diff for diff in differences if diff > 0)

    # Function to count points per interval starting from a specific interval size
    def count_points_per_interval(interval_size):
        intervals_count = 0
        start_index = 0
        while start_index < len(sorted_data):
            end_value = sorted_data[start_index] + interval_size
            current_index = start_index
            while current_index < len(sorted_data) and sorted_data[current_index] <= end_value:
                current_index += 1
            if current_index - start_index >= min_n_samples:  # At least 40 points in the interval
                intervals_count += 1
                start_index = current_index
            else:  # Not enough points in the interval, break the loop
                return 0  # Indicates that this interval size is too small
        return intervals_count

    # Increment the interval size until finding the optimal size that includes at least 40 points per interval
    optimal_interval_size = initial_interval_size
    while True:
        intervals_count = count_points_per_interval(optimal_interval_size)
        if intervals_count > 0:  # Found an interval size that works
            break
        optimal_interval_size *= (1 + alpha)  # Increase the interval size by alpha and try again

    return optimal_interval_size


# get the chunks of the depth array and return the start and end index of each chunk
def get_chunks(depth, interval_size):
    # ensure the depth is sorted
    depth = np.array(depth)
    assert all(depth[i] <= depth[i + 1] for i in range(len(depth) - 1)), "The depth array is not sorted"
    start_index = 0
    chunks = []
    while start_index < len(depth):
        end_value = depth[start_index] + interval_size
        current_index = start_index
        while current_index < len(depth) and depth[current_index] <= end_value:
            current_index += 1
        chunks.append((start_index, current_index))
        start_index = current_index
    return chunks


def to_1d(df, chunks, how: str):
    # get the mean of the intensities in each chunk
    df_1d = []
    for chunk in chunks:
        start_index, end_index = chunk
        data = df.iloc[start_index:end_index]
        # perform the calculation according to how string
        df_1d.append(eval(how))
    return df_1d


if __name__ == "__main__":
    target_cmpds = {
        'GDGT_0': 1324.3046,
        'GDGT_5': 1314.2264
    }
    exported_txt_path = '/Users/weimin/Downloads/MV0811-14TC_5-10_Q1_1320_w40_75DR.d.d.txt'
    sqlite_db_path = '/Users/weimin/Downloads/metadata.db'
    df = get_mz_int_depth(exported_txt_path, sqlite_db_path, target_cmpds)
    df = df.dropna()
    df = df.sort_values(by='d')
    optimal_interval_size = find_optimal_interval_for_irregular_data(df['d'])
    logging.debug(f"Optimal interval size: {optimal_interval_size}cm")
    chunks = get_chunks(df['d'], optimal_interval_size)
    # get the mean depth of each chunk
    depth_1d = to_1d(df, chunks, "data['d'].mean()")
    ratio_1d = to_1d(df, chunks, "data['Int_GDGT_5'].sum() / (data['Int_GDGT_5'].sum() + data['Int_GDGT_0'].sum())")
    df_1d = pd.DataFrame({'d': depth_1d, 'ratio': ratio_1d})

