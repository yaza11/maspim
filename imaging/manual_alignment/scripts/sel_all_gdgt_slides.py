# put all 0-19cm gdgt slides into one folder

import sqlite3
import shutil
import pandas as pd

# connect to the database
db_path = '/Users/weimin/Projects/msi_workflow/imaging/manual_alignment/data/metadata.db'
conn = sqlite3.connect(db_path)

# select the msi_img_file_name and Q1 and start_depth columns from the metadata table, where Q1 is 1320 and
# start_depth is between 0 and 15
query = 'SELECT msi_img_file_name, Q1, start_depth FROM metadata WHERE Q1 = 1320 AND start_depth BETWEEN 0 AND 15'
df = pd.read_sql_query(query, conn)

# copy all the images to the same folder
src_folder = '/Users/weimin/Projects/msi_workflow/imaging/manual_alignment/data/msi_img'
dst_folder = '/Users/weimin/Downloads/GDGT0_19'

for i, row in df.iterrows():
    src = f'{src_folder}/{row.msi_img_file_name}'
    dst = f'{dst_folder}/{row.msi_img_file_name}'
    shutil.copy(src, dst)

