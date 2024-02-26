# a script to parse the columns of the metadata sqlite file

import sqlite3
import pandas as pd

# connect to the sqlite database
db_path = '/Users/weimin/Projects/msi_workflow/imaging/manual_alignment/data/metadata.db'
conn = sqlite3.connect(db_path)

# select the spec_file_name column from the metadata table
query = 'SELECT spec_file_name FROM metadata'
df = pd.read_sql_query(query, conn)

# parse the Q1 mass from the spec_file_name, very likely after Q1_ and before _
df['Q1'] = df['spec_file_name'].str.extract(r'Q1_(\d+)')
print(df.Q1.unique())
# find where Q1 is missing
print(df[df.Q1.isnull()])
# find where Q1 is 522
print(df[df.Q1 == '522'])
# it's wrong, should be 552, replace it
df.loc[df.Q1 == '522', 'Q1'] = '552'
# find where Q1 is 21320
print(df[df.Q1 == '21320'])
# it's wrong, should be 1320, replace it
df.loc[df.Q1 == '21320', 'Q1'] = '1320'
df = df.astype({'Q1': int})
print(df.Q1.unique())


# parse the starting depth from the spec_file_name, very likely two numbers connected by -, excluding MV0811-14TC
df['start_depth'] = df['spec_file_name'].str.replace('MV0811-14TC', '').str.extract(r'(\d+)-\d+')
df['start_depth'] = df['start_depth'].astype(int)
# print the unique start_depth, sorted
print(sorted(df.start_depth.unique()))
# looks fine

# put the values back to the metadata table, by updating the start_depth and Q1 columns
# create the start_depth and Q1 columns in the metadata table
query = 'ALTER TABLE metadata ADD COLUMN start_depth INTEGER'
conn.execute(query)
query = 'ALTER TABLE metadata ADD COLUMN Q1 INTEGER'
conn.execute(query)

for i, row in df.iterrows():
    query = f'UPDATE metadata SET start_depth = {row.start_depth}, Q1 = {row.Q1} WHERE spec_file_name = "{row.spec_file_name}"'
    conn.execute(query)

# commit the changes and close the connection
conn.commit()
conn.close()

# done