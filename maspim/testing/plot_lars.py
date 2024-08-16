import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maspim.project.main import get_project
from maspim.res.compound_masses import mC37_2, mPyro, mGDGT0, mCren_p

import logging

logging.basicConfig(level=logging.INFO)


path_folder = r'D:\plotting 515-520\alkenone'
da_file = r'D:/plotting 515-520/alkenone/2019_07_31_Cariaco_515-520cm_60um_alkenone.txt'

p = get_project(is_MSI=True, path_folder=path_folder)
p.set_image_handler()
p.set_image_sample()
p.set_image_roi()

comps = [mC37_2]
titles = ['C37:2']
for comp, title in zip(comps, titles):
    p.plot_comp(
        comp=comp, 
        title=title, 
        da_export_file=da_file, 
        source='da_export',
        plot_on_background=True
    )


# %%
path_folder = r'D:\plotting 515-520\GDGT'
da_file = r'D:/plotting 515-520/GDGT/2019_07_31_Cariaco_515-520cm_60um_GDGT.txt'
mis_file = r'2019_07_31_Cariaco_515-520cm_60um.GDGT.mis'

p = get_project(is_MSI=True, path_folder=path_folder, mis_file=mis_file)
p.set_image_handler()
p.set_image_sample()
p.set_image_roi()


comps = [mGDGT0, mCren_p]
titles = ['GDGT-0', 'GDGT-5']
for comp, title in zip(comps, titles):
    p.plot_comp(
        comp=comp, 
        title=title, 
        da_export_file=da_file, 
        source='da_export',
        plot_on_background=True
    )
