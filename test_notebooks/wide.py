from matplotlib import pyplot as plt

from maspim import get_project, ProjectMSI
from maspim.res.compound_masses import mC37_2

path_folder = r'C:\Users\Yannick Zander\Downloads\2018_08_02_1002E_130-135cm_DHB_Alkenones\2018_08_02_1002E_130-135cm_DHB_Alkenones.i'


p: ProjectMSI = get_project(path_folder=path_folder, is_MSI=True)

p.print_overview()

