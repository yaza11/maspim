import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cProject import Project
from data.file_helpers import get_folder_structure, get_mis_file

folder = r'C:\Users\Yannick Zander\Promotion\Cariaco 2024\490-495cm\2018_08_27 Cariaco 490-495 GDGT.i'


p = Project(is_MSI=True, path_folder=folder)

print(p.__dict__)


