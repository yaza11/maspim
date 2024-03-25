import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cAgeModel import AgeModel

path_age_model = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\Age Model\480_510_MSI_age_model_mm_yr.txt'

AM1 = AgeModel(path_age_model, sep='\t', index_col=False)
AM1.add_depth_offset(4800)

# AM1.save(r"C:\Users\Yannick Zander\Promotion\Cariaco 2024\Age Model")