import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cAgeModel import AgeModel

path_age_model_480_510 = r'C:/Users/Yannick Zander/Promotion/Cariaco 2024/Age Model/480-510/480_510_MSI_age_model_mm_yr.txt'
path_age_model_510_540 = r'C:/Users/Yannick Zander/Promotion/Cariaco 2024/Age Model/510-540/510_540_MSI_age_model_mm_yr.txt'

age_model_480_510 = AgeModel(path_age_model_480_510, depth_offset=4800, conversion_to_cm=1 / 10, sep='\t', index_col=False)
age_model_510_540 = AgeModel(path_age_model_510_540, depth_offset=5100, conversion_to_cm=1 / 10, sep='\t', index_col=False)

age_model = age_model_480_510 + age_model_510_540


age_model.save(r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\Age Model')