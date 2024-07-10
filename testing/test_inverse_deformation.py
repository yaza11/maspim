# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:02:30 2024

@author: Yannick Zander
"""
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle, vortex
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

import SimpleITK as sitk


def apply_displacement(u, v, image):
    assert image.ndim == 2
    assert u.shape == v.shape == image.shape
    
    nr, nc = image.shape
    
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    
    warped = warp(image, np.array([row_coords + v, col_coords + u]), mode='edge')
    
    return warped
    

def get_inverted_displacement(
        u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use SimpleITK to invert the displacement fields."""
    assert u.shape == v.shape

    nr, nc = u.shape

    # Create the displacement field array
    displacement_field: np.ndarray = np.zeros((nr, nc, 2), dtype=np.float32)
    displacement_field[..., 0] = u
    displacement_field[..., 1] = v

    # Convert to SimpleITK format
    disp_field_sitk = sitk.GetImageFromArray(displacement_field, isVector=True)

    # Set origin and spacing for the displacement field
    disp_field_sitk.SetOrigin((0.0, 0.0))
    disp_field_sitk.SetSpacing((1.0, 1.0))

    # Compute the inverse displacement field
    deform_inv_sitk = sitk.InverseDisplacementField(
        disp_field_sitk,
        size=disp_field_sitk.GetSize(),
        outputOrigin=disp_field_sitk.GetOrigin(),
        outputSpacing=disp_field_sitk.GetSpacing()
    )

    # Convert the inverse displacement field back to a numpy array
    deform_inv = sitk.GetArrayFromImage(deform_inv_sitk)

    v_inv = deform_inv[..., 1]
    u_inv = deform_inv[..., 0]

    return u_inv, v_inv


# --- Load the sequence
image0, image1, disp = stereo_motorcycle()

# --- Convert the images to gray level: color is not supported.
image0 = rgb2gray(image0)
image1 = rgb2gray(image1)

# --- Compute the optical flow
v, u = optical_flow_ilk(image0, image1)

# --- Apply warping to image1 to get the warped image
warped = apply_displacement(u, v, image1)
u_inv, v_inv = get_inverted_displacement(u, v)
inv_warped = apply_displacement(u_inv, v_inv, warped)

# --- Plot the original and inversely warped images
plt.subplot(2, 2, 1)
plt.title('Original Image1')
plt.imshow(image1, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Warped Image')
plt.imshow(warped, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Inverse Warped Image')
plt.imshow(inv_warped, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Inverse flow Image')
plt.imshow(apply_displacement(-u, -v, warped), cmap='gray')
plt.show()
