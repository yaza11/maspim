# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:48:21 2024

@author: Yannick Zander
"""
import os
import json

    
def get_teaching_points(
        path_image_file: str,
        path_json: str,
        not_found: str = 'error'
):
    """
    Obtain the teaching points matching punch holes from MSI photo to X-ray.

    Parse entries in a json file for the image. Use pair_tp_str to find pairs.


    """
    with open(path_json, 'r') as f:
        d = json.load(f)

    has_pairings: bool = 'pair_tp_str' in d
    assert has_pairings, 'Please set the teaching points manually with msiAlign'

    for img in d['items']:  # look for the path_image_file specified within json
        if 'img_path' not in img:  # E.g. vertical lines
            continue
        try:
            if os.path.samefile(img['img_path'], path_image_file):
                # this is what we are looking for
                break
        except FileNotFoundError:  # just for testing
            # TODO: make this throw an error, if not testing
            if img['img_path'] == path_image_file:
                break
    else:  # target file not found in json
        if not_found == 'ignore':
            return
        elif not_found == 'error':
            raise KeyError(f'Could not find {path_image_file} in the json.')
        else:
            raise NotImplementedError()

    # fetch the teaching points
    x: list[float] = []
    y: list[float] = []
    label: list[int] = []
    for point in img['teaching_points'].values():
        x.append(point[0])
        y.append(point[1])
        label.append(point[-1])
    
    return x, y, label


def get_teaching_point_pairings_dict(pairings: str) -> dict[int, int]:
    """
    turn the mapping into a dict

    mapping looks something like this
        0 5
        1 6
        ...
    where points with labels (0, 5), (1, 6), ... are linked
    """
    mapping: dict[int, int] = {
        int(e.split()[0]): int(e.split()[1])
        for e in pairings.split('\n')
    }
    # check for duplicate labels
    assert (len(list(mapping.keys()) + list(mapping.values()))
            == len(set(mapping.keys()) | set(mapping.values()))), \
        'Found duplicate labels, make sure there are no duplicates'

    return mapping


if __name__ == '__main__':
    x, y, l = get_teaching_points(
        path_image_file=r"/Users/weimin/Projects/projects/SBB14TC/PAH/msiAlign/PAH0_19/MV0811-14TC_0-5E136_0000.tif", 
        path_json=r'C:/Users/Yannick Zander/Downloads/pah0_19 (1).json'
    )
