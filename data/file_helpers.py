import os

def get_folder_structure(path):
    # Initialize the result dictionary with folder 
    # name, type, and an empty list for children 
    result = {
        'name': os.path.basename(path), 
        'type': 'folder', 
        'children': []
    } 
  
    # Check if the path is a directory 
    if not os.path.isdir(path): 
        return result 
  
    # Iterate over the entries in the directory 
    for entry in os.listdir(path): 
       # Create the full path for the current entry 
        entry_path = os.path.join(path, entry) 
  
        # If the entry is a directory, recursively call the function 
        if os.path.isdir(entry_path): 
            result['children'].append(get_folder_structure(entry_path)) 
        # If the entry is a file, create a dictionary with name and type 
        else: 
            result['children'].append({'name': entry, 'type': 'file'}) 
  
    return result 


def find_files(folder_structure: dict[str, dict | str], *names, by_suffix=False):
    children = folder_structure['children']
    matches = {}
    for child in children:
        name = child['name']
        if by_suffix:
            name = name.split('.')[-1] if '.' in name else ''
        if name in names:
            matches[name] = child['name']
    return matches


def get_mis_file(path_folder):
    folder_structure = get_folder_structure(path_folder)
    return find_files(folder_structure, 'mis', by_suffix=True)['mis']


def get_d_folder(path_folder):
    folder_structure = get_folder_structure(path_folder)
    return find_files(folder_structure, 'd', by_suffix=True)['d']


def search_keys_in_xml(path_mis_file, keys):
    # iniate list of lists for values
    out_dict = {key: [] for key in keys}
    # open xml
    with open(path_mis_file) as xml:
        # parse through lines
        for line in xml:
            line = line.replace('/', '')
            # search for keys
            for key in keys:
                key_xml = f'<{key}>'
                if key_xml in line:
                    value = line.split(key_xml)[1]
                    out_dict[key].append(value)
    for key, value in out_dict.items():
        if len(value) == 1:
            out_dict[key] = value[0]
    return out_dict


def get_image_file(path_folder):
    path_mis_file = os.path.join(path_folder, get_mis_file(path_folder))
    return search_keys_in_xml(path_mis_file, ['ImageFile'])['ImageFile']
