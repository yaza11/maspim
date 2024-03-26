from imaging.main.cImage import ImageSample, ImageROI
from imaging.util.coordinate_transformations import kartesian_to_polar, polar_to_kartesian
from imaging.util.Image_plotting import plt_contours

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import (
    warp, AffineTransform, ProjectiveTransform, PiecewiseAffineTransform
)


def find_line_contour_intersect(
        contour: np.ndarray[int], 
        holes: list[int], 
        hole_side: str, 
        image_shape: np.ndarray
) -> np.ndarray[int]:
    """
    Find the intersection of the line with the sample area by projecting 
    down/up from the first hole.
    """
    # use opposite half of that where holes were searched
    if hole_side == 'bottom':
        mask_section: np.ndarray[bool] = contour[:, 0, 1] < image_shape[0] / 2
    else:
        mask_section: np.ndarray[bool] = contour[:, 0, 1] > image_shape[0] / 2
    # extract the values of the contour in the depth-direction
    x_contour: np.ndarray[int] = contour[mask_section, 0, 0]

    # find idxs where contour intersects line
    x_hole: int = holes[0][1]
    idx_projected: int = np.argmin(np.abs(x_contour - x_hole))

    projected_point: np.ndarray[int] = contour[mask_section, 0, :][idx_projected]
    return projected_point[::-1]


def find_corners_contour(contour):
    xmin = contour[:, 0, 0].min()
    xmax = contour[:, 0, 0].max()
    ymin= contour[:, 0, 1].min()
    ymax = contour[:, 0, 1].max()
    points = [
        np.array([ymin, xmin]), 
        np.array([ymin, xmax]), 
        np.array([ymax, xmin]), 
        np.array([ymax, xmax])
    ]
    return points


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """
    Bring an array of corners in anti-clockwise order.
    
    format = [
        [x_topleft, y_topleft],
        [x_bottomleft, y_bottomleft],
        [x_bottomright, y_bottomright],
        [x_topright, y_topright]
    ]
    """
    # make sure corners are in the right order
    x = corners[:, 0]
    y = corners[:, 1]
    c = np.mean(corners, axis = 0)
    x_c = x.copy() - c[0]
    y_c = y.copy() - c[1]

    r, phi = kartesian_to_polar(x_c, y_c)
    # scale down to make sure points are inside image
    r /= 2
    # sort
    o = np.argsort(phi)
    r, phi = r[o], phi[o]
    x, y = polar_to_kartesian(r, phi)
    corners[:, 0] = x + c[0]
    corners[:, 1] = y + c[1]
    return corners.astype(np.float32)

class Transformation(ImageSample):
    """Find transformation from source (image to be transformed) to 
    destination (target to be matched)."""
    def __init__(
            self, 
            source: np.ndarray[int | float] | ImageSample | ImageROI, 
            target: np.ndarray[int | float] | ImageSample | ImageROI,
    ) -> None:
        self.source: ImageROI = self._handle_input(source)
        self.target: ImageROI = self._handle_input(target)
        
        self.source_shape: tuple[int] = self.source.sget_image_original().shape
        self.target_shape: tuple[int] = self.target.sget_image_original().shape

        self.trafos: list = []
        self.trafo_types: list = []
        
    def _handle_input(
            self, obj: np.ndarray[int | float] | ImageSample | ImageROI
    ) -> ImageROI:
        if isinstance(obj, ImageROI):  # nothing to do
            return obj
        elif isinstance(obj, str):  # file of image passed, create new obj
            obj: ImageSample = ImageSample(path_image_file=obj)
        elif isinstance(obj, np.ndarray):  # image passed, create new obj
            obj: ImageROI = ImageROI(image=obj)
        if type(obj) == ImageSample:  # create ImageROI obj from ImageSample
            obj: ImageROI = ImageROI(
                image=obj.sget_sample_area()[0],
                obj_color=obj.obj_color
            )
        return obj
    
    def _transform_from_punchholes(
            self, 
            points_source: list[int], # xray
            points_target: list[int], # data
            is_piecewise: bool,
            hole_side: str,
            contour_source: np.ndarray[int] = None,
            contour_target: np.ndarray[int] = None,
    ) -> np.ndarray[float] | PiecewiseAffineTransform:
        """Stretch and resize ROI to match"""
        if contour_source is None:
            contour_source = self.source.sget_main_contour()
        if contour_target is None:
            contour_target = self.target.sget_main_contour()
        
        # append points from line-contour intersection
        points_source.append(
            find_line_contour_intersect(
                contour=contour_source, 
                holes=points_source, 
                hole_side=hole_side,
                image_shape=self.source_shape
            )
        )
        points_target.append(
            find_line_contour_intersect(
                contour=contour_target,
                holes=points_target, 
                hole_side=hole_side,
                image_shape=self.target_shape
            )
        )
        
        if is_piecewise:
            # piecewise affine
            # add corners of contour so that entire image is transformed
            points_source += find_corners_contour(contour_source)
            points_target += find_corners_contour(contour_target)
            
            pwc = PiecewiseAffineTransform()
            # swap xy of points
            src = np.array(points_source)[:, ::-1]
            dst = np.array(points_target)[:, ::-1]
            # src and dst swapped??
            pwc.estimate(dst, src)
            M: PiecewiseAffineTransform = pwc
            transform_type = PiecewiseAffineTransform
        else:
            # perform affine transformation
            M: np.ndarray[float] = cv2.getAffineTransform(
                np.array(points_source).astype(np.float32)[:, ::-1], 
                np.array(points_target).astype(np.float32)[:, ::-1]
            )  # source: xray, target: msi
            transform_type = cv2.getAffineTransform
        self.trafos.append(M)
        self.trafo_types.append(transform_type)
        return M
    
    def _transform_from_bounding_box(
            self, plts: bool = False
    ) -> np.ndarray[float]:
        """Projective transform between corners of bounding boxes."""
        def get_points(obj: ImageROI) -> np.ndarray[int]:
            rect: tuple[float] = cv2.minAreaRect(obj.sget_main_contour())
            points: np.ndarray[float] = cv2.boxPoints(rect)
            # sort point anticlockwise
            points = sort_corners(points)
            return points
    
        points_source: np.ndarray[np.float32] = get_points(self.source)
        points_target: np.ndarray[np.float32] = get_points(self.target)
        
        M = cv2.getPerspectiveTransform(points_source, points_target)
        transform_type = cv2.getPerspectiveTransform
        
        if plts:
            plt_contours([points_source], self.source.sget_image_original())
            plt_contours([points_target], self.target.sget_image_original())
        
        self.trafos.append(M)
        self.trafo_types.append(transform_type)

        return M
    
    def _transform_from_image_flow(self):
        # TODO: <--
        ...
        raise NotImplementedError()

    def _transform_from_laminae(self):
        # TODO: <--
        ...
        raise NotImplementedError()
    
    def estimate(self, method: str, *args, **kwargs):
        methods = ('punchholes', 'bounding_box', 'image_flow', 'laminae')
        assert method in methods, f'method must be in {methods}, not {method}'

        if method == 'punchholes':
            self._transform_from_punchholes(*args, **kwargs)
        elif method == 'bounding_box':
            self._transform_from_bounding_box(*args, **kwargs)
        elif method == 'image_flow':
            self._transform_from_image_flow(*args, **kwargs)
        elif method == 'laminae':
            self._transform_from_laminae(*args, **kwargs)
        else:
            raise NotImplementedError()
    
    def fit(self, img: np.ndarray = None) -> np.ndarray:
        def apply_(img, M, transform_type):
            if transform_type == cv2.getPerspectiveTransform:
                warped = cv2.warpPerspective(
                    img, M, dsize=(self.target_shape[1], self.target_shape[0])
                )
            elif transform_type == cv2.getAffineTransform:
                warped = cv2.warpAffine(
                    img, M, dsize=(self.target_shape[1], self.target_shape[0])
                )
            elif transform_type == PiecewiseAffineTransform:
                print(img.shape, self.target_shape)
                warped = warp(img, M, output_shape=self.target_shape[:2])
            else: 
                raise NotImplementedError()
            return warped
    
        if img is None:
            img = self.source.sget_image_original()
        
        # apply the appropriate transform
        for M, transform_type in zip(self.trafos, self.trafo_types):
            img = apply_(img, M, transform_type)

        return img