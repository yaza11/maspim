import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt

from data.cMSI import MSI

path_d_folder = r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i\2018_08_27 Cariaco 490-495 alkenones.d'

msi = MSI(path_d_folder)
msi.load()

mz_C37_2 = 553.53188756536
mz_C37_3 = 551.51623750122

msi.plot_comp(mz_C37_2)
msi.plot_comp(mz_C37_3)

ts = msi.processing_zone_wise_average(
    zones_key='depth', 
    columns=['551.5174', '553.5323', 'age']
)

ts_lam = msi.processing_zone_wise_average(
    zones_key='classification_s', 
    columns=['551.5174', '553.5323', 'age']
)
ts_lam = ts_lam.sort_values(by='age').reset_index()

ts_lam_ex = msi.processing_zone_wise_average(
    zones_key='classification_se', 
    columns=['551.5174', '553.5323', 'age']
)
ts_lam_ex = ts_lam_ex.sort_values(by='age').reset_index()

ts['Uk37'] = ts['553.5323'] / (ts['553.5323'] + ts['551.5174'])
ts_lam['Uk37'] = ts_lam['553.5323'] / (ts_lam['553.5323'] + ts_lam['551.5174'])
ts_lam_ex['Uk37'] = ts_lam_ex['553.5323'] / (ts_lam_ex['553.5323'] + ts_lam_ex['551.5174'])

df = pd.read_csv(r'C:/Users/yanni/Downloads/MD03-2621_UK37_SST_BAYSPLINE.tab', sep='\t')
df['Age'] *= 1000
# %%
mask = (df.Age >= ts.age.min()) & (df.Age <= ts.age.max())

plt.plot(ts['age'], ts['Uk37'], label='depth-wise')
plt.plot(ts_lam['age'], ts_lam['Uk37'], label='laminae-wise')
plt.plot(ts_lam_ex['age'], ts_lam_ex['Uk37'], label='expanded laminae-wise')
plt.plot(df.loc[mask, 'Age'], df.loc[mask, "UK37"], label='reference')
plt.xlabel('age in yr b2k')
plt.ylabel("UK'37")
plt.legend()
plt.show()
