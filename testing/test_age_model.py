import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cAgeModel import AgeModel

path_file1 = r'G:/Meine Ablage/Master Thesis/AgeModel/480_510_MSI_age_model_mm_yr.txt'

AM1 = AgeModel(path_file1, sep='\t', index_col=False)
AM1.add_depth_offset(4800)

path_file2 = r'G:/Meine Ablage/Master Thesis/AgeModel/510_540_MSI_age_model_mm_yr.txt'

AM2 = AgeModel(path_file2, sep='\t', index_col=False)
AM2.add_depth_offset(5100)

AM = AM1 + AM2
print(AM)

AM.save(r'D:\Cariaco Data for Weimin\490-495cm\2018_08_27 Cariaco 490-495 alkenones.i\2018_08_27 Cariaco 490-495 alkenones.d')


am = AgeModel(r'G:\Meine Ablage\Master Thesis\AgeModel')
