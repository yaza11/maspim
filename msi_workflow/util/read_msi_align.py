# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:48:21 2024

@author: Yannick Zander
"""
import os
import json

    
def get_teaching_points(path_image_file: str, path_json: str, not_found='error'):
    with open(path_json, 'r') as f:
        d = json.load(f)

    for img in d['items']:
        if 'img_path' not in img:  # E.g. vertical lines
            continue
        try:
            if os.path.samefile(img['img_path'], path_image_file):
                break
        except FileNotFoundError:  # just for testing
            if img['img_path'] == path_image_file:
                break
    else:
        if not_found == 'ignore':
            return
        elif not_found == 'error':
            raise KeyError(f'Could not find {path_image_file} in the json.')
        else:
            raise NotImplementedError()
        
    has_pairings: bool = 'pair_tp_str' in d
    x: list[float] = []
    y: list[float] = []
    label: list[int] = []
    for point in img['teaching_points'].values():
        x.append(point[0])
        y.append(point[1])
        if has_pairings:
            label.append(point[-1])
    
    return x, y, label


if __name__ == '__main__':
    x, y, l = get_teaching_points(
        path_image_file=r"/Users/weimin/Projects/projects/SBB14TC/PAH/msiAlign/PAH0_19/MV0811-14TC_0-5E136_0000.tif", 
        path_json=r'C:/Users/Yannick Zander/Downloads/pah0_19 (1).json'
    )
