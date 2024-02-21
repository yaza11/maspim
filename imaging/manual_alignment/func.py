import re
import numpy as np

def get_msi_rect_from_imaginginfo(xml_file):
    with open(xml_file, 'r') as f:
        xml = f.read()
    # get all the value between <spotName> and </spotName>
    spot_name = re.findall(r'<spotName>(.*?)</spotName>', xml)
    # parse the spot name to get x and y
    spot_name = [re.findall(r'X(\d+)Y(\d+)', s) for s in spot_name]
    # flatten the list
    spot_name = [item for sublist in spot_name for item in sublist]
    # convert to int
    spot_name = [(int(x), int(y)) for x, y in spot_name]
    # convert to array
    spot_name = np.array(spot_name)
    # get the coordinates of the four corners
    x_min, y_min = spot_name.min(axis=0)
    x_max, y_max = spot_name.max(axis=0)

    return [x_min, y_min, x_max, y_max]


def get_image_file_from_mis(mis_file):
    with open(mis_file, 'r') as f:
        mis = f.read()
    # get the value between <ImageFile> and </ImageFile>
    image_file = re.findall(r'<ImageFile>(.*?)</ImageFile>', mis)
    return image_file[0]


def get_px_rect_from_mis(mis_file):
    with open(mis_file, 'r') as f:
        mis = f.read()
    # get all the value between <Point> and </Point>
    roi_bound = re.findall(r'<Point>(.*?)</Point>', mis)
    roi_bound = [s.split(',') for s in roi_bound]
    roi_bound = [(int(x), int(y)) for x, y in roi_bound]
    roi_bound = np.array(roi_bound)
    # get a rectangle shape from the points
    x_min, y_min = roi_bound.min(axis=0)
    x_max, y_max = roi_bound.max(axis=0)
    return [x_min, y_min, x_max, y_max]


if __name__ == "__main__":
    pass

