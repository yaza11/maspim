import os
import socket
from textdistance import hamming as textdist

use_disc = 0
is_NB = socket.gethostname() == 'DESKTOP-51884LJ'

if use_disc and is_NB:
    disc = 'E:'
elif use_disc:
    disc = 'L:'
elif is_NB:
    disc = 'G:/Meine Ablage/Master Thesis'
else:
    disc = 'D:/My Drive/Master Thesis'

if is_NB:
    folder_thesis_images = r'G:\Meine Ablage\Master Thesis\Images Thesis\Software development'
else:
    folder_thesis_images = r'D:\My Drive\Master Thesis\Tex\Software development\images'

if use_disc:
    relative_path_to_saves = 'Master_Thesis'
    relative_path_to_msi_data = 'Cariaco Data for Weimin'
    relative_path_to_xrf_data = r'Cariaco line scan Xray\uXRF slices'
    relative_path_to_age_model = r'Promotion\msi_workflow\exampleData'

    absolute_path_to_saves = os.path.join(disc, relative_path_to_saves)

    absolute_path_to_msi_data = os.path.join(disc,
                                             relative_path_to_msi_data)
    absolute_path_to_xrf_data = os.path.join(disc,
                                             relative_path_to_xrf_data)
    absolute_path_to_age_model = os.path.join(
        disc, relative_path_to_saves, relative_path_to_age_model)

    file_dataBase = os.path.join(disc, relative_path_to_saves, r'DataBase/MB_export_tolerance_new=3mDa.csv')
else:
    relative_path_to_msi_data = 'MSI'
    relative_path_to_xrf_data = r'XRF'
    relative_path_to_age_model = r'AgeModel'

    absolute_path_to_saves = disc

    absolute_path_to_msi_data = os.path.join(disc,
                                             relative_path_to_msi_data)
    absolute_path_to_xrf_data = os.path.join(disc,
                                             relative_path_to_xrf_data)
    absolute_path_to_age_model = os.path.join(
        disc, relative_path_to_age_model)

    file_dataBase = os.path.join(disc, r'DataBase/MB_export_tolerance=3mDa.csv')

# age model
file_480_to_510 = os.path.join(
    absolute_path_to_age_model, '480_510_MSI_age_model_mm_yr.txt')
file_510_to_540 = os.path.join(
    absolute_path_to_age_model, '510_540_MSI_age_model_mm_yr.txt')

window_to_folder = {'Alkenones': 'Alkenones',
                    'FA': 'FA',
                    'GDGT': 'GDGT',
                    'xrf': 'XRF',
                    'combined': 'combined'}

window_to_data_type = {'msi': 'msi',
                       'Alkenones': 'msi',
                       'FA': 'msi',
                       'GDGT': 'msi',
                       'xrf': 'xrf',
                       'combined': 'combined'}


def get_section_formatted(section):
    # convert to tuple with (top, bottom) in cm as ints
    if isinstance(section, str):
        d1, d2 = section.split('-')
        section = (d1, d2)
    if isinstance(section, tuple):
        d1 = int(section[0])
        d2 = int(section[1])
    # make sure d2 > d1
    if d1 > d2:
        d = d2
        d2 = d1
        d1 = d
    return (d1, d2), f'{d1}-{d2}'


def get_file_path(section: tuple[int, int], window: str, obj: str | object) -> str:
    # get according obj_type if not provided as str
    obj_str = str(type(obj)).split(' ')[-1].split('.')[-1].strip("'>")
    # obj provided as str (one of Image, Data, XRF, )
    if obj_str == 'str':
        if obj.lower() in ('xrf', 'data', 'msi', 'dataclass'):
            obj_type = 'DataClass'
        elif obj.lower() == 'image':
            obj_type = 'Image'
        else:
            raise ValueError("if obj is provided as string, it has to be one \
of 'xrf', 'data', 'msi', 'dataclass' or 'image'.")
    elif obj_str == 'Image':
        obj_type = 'Image'
    elif obj_str in ('XRF', 'MSI'):
        obj_type = 'DataClass'
    else:
        raise
    folder_obj_type = obj_type + '_objects'
    section_str = f'{section[0]}-{section[1]}'
    folder_path_file = os.path.join(
        absolute_path_to_saves, folder_obj_type, section_str, window_to_folder[window])
    name = f'{window_to_data_type[window]}_obj_{section_str}cm_{window}.pkl'
    file = os.path.join(folder_path_file, name)
    return file


def find_best_match_file(directory, template):
    template_suffix = os.path.splitext(template)[1]
    best_match = None
    best_distance = float('inf')

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_suffix = os.path.splitext(filename)[1]
            if file_suffix == template_suffix:
                distance = textdist(filename, template)
                if distance < best_distance:
                    best_distance = distance
                    best_match = filename

    return best_match
