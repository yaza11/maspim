import re

import numpy as np


def sort_points_clockwise(points):
    """
    Sort three points in a clockwise order.
    """
    # get the centroid of the points
    centroid = np.mean(points, axis=0)
    # get the angle of the points with respect to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    # sort the points based on the angles
    points = points[np.argsort(angles)]
    return points

def get_msi_rect_from_imaginginfo(xml_file, return_spot_name=False):
    with open(xml_file, 'r') as f:
        xml = f.read()
    # get all the value between <spotName> and </spotName>
    spot_name = re.findall(r'<spotName>(.*?)</spotName>', xml)
    if return_spot_name:
        spot_name_str = spot_name.copy()
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

    if return_spot_name:
        return [x_min, y_min, x_max, y_max], spot_name_str
    else:
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


class CorSolver:
    """
    A class that takes tie points from two coordinate system (i.e., one from Fleximaging, the other from
    xray measurement) and solve the transformation between these two systems.
    """

    def __init__(self):
        self.translation_vector = None
        self.transformation_matrix = None

    def _reset(self):
        if hasattr(self, 'transformation_matrix'):
            del self.transformation_matrix
            del self.translation_vector

    def fit(self, src_tri, dst_tri):
        """
        Parameters:
        --------
            src_tri: coordinates of the source triangle in the source coordinate system
            dst_tri:  coordinates of the target triangle in the target coordinate system
        """
        self._reset()
        return self.partial_fit(src_tri, dst_tri)

    def partial_fit(self, src_tri, dst_tri):
        """
        solve the affine transformation matrix between FlexImage coordinates and X_ray coordinates
        https://stackoverflow.com/questions/56166088/how-to-find-affine-transformation-matrix-between-two-sets-of-3d-points
        """
        len_source = len(src_tri)
        basis = np.vstack([np.transpose(src_tri), np.ones(len_source)])
        diagonal = 1.0 / np.linalg.det(basis)

        def entry(r_val, d_val):
            return np.linalg.det(np.delete(np.vstack([r_val, basis]), (d_val + 1), axis=0))

        m_matrix = [[(-1) ** i * diagonal * entry(R, i) for i in range(len_source)] for R in
                    np.transpose(dst_tri)]
        # pylint: disable=unbalanced-tuple-unpacking
        a_matrix, t_val = np.hsplit(np.array(m_matrix), [len_source - 1])
        t_val = np.transpose(t_val)[0]
        self.transformation_matrix = a_matrix
        self.translation_vector = t_val
        return self

    def transform(self, src_coordinate):
        """
        Parameters:
        --------
            src_coordinate: the source coordinates that needs be transformed
        """
        dst_coordinate = src_coordinate.dot(self.transformation_matrix.T) + self.translation_vector
        return np.round(dst_coordinate, 0)




if __name__ == "__main__":
    pass

