import numpy as np
import nibabel as nib
from typing import Tuple, List, Optional
from PIL import Image
from scipy.ndimage import measurements
import torch
import matplotlib.pyplot as plt
import os



def load_nifti(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return volume and voxel spacing."""
    img = nib.load(filepath) 
    return img.get_fdata(), img.header.get_zooms()[:3]

def save_nifti(filepath: str, volume: np.ndarray, reference_file: str) -> None:
    """Save volume as NIfTI using reference file header."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ref_img = nib.load(reference_file)
    new_img = nib.Nifti1Image(volume, ref_img.affine, ref_img.header)
    nib.save(new_img, filepath) 

def pix2real(pixel_coords: np.ndarray, voxel_spacing: np.ndarray) -> np.ndarray:
    """Convert pixel/voxel coordinates to real-world coordinates.
    
    Args:
        pixel_coords: Array of shape (H, W, D) containing pixel/voxel coordinates
        voxel_spacing: Array of shape (3,) containing voxel dimensions (x, y, z)
        
    Returns:
        Array of shape (H, W, D) with real-world coordinates
    """
    return pixel_coords * voxel_spacing

def create_hu_image(X, dicom_attributes):
    return X * dicom_attributes['rescale_slope'] + dicom_attributes['rescale_intercept']


def get_object_agatston(calc_object: np.ndarray, calc_pixel_count: int):
    """Applies standard categorization: https://radiopaedia.org/articles/agatston-score"""
    object_max = np.max(calc_object)
    object_agatston = 0
    if 130 <= object_max < 200:
        object_agatston = calc_pixel_count * 1
    elif 200 <= object_max < 300:
        object_agatston = calc_pixel_count * 2
    elif 300 <= object_max < 400:
        object_agatston = calc_pixel_count * 3
    elif object_max >= 400:
        object_agatston = calc_pixel_count * 4
    return object_agatston


def compute_total_agatston_score(X, Y, dicom_attributes, min_calc_object_pixels=1) -> int:

    total_score = 0
    
    N = X.shape[-1]
    for sliceIdx in range(N):
        
        mask = Y[:, :, sliceIdx]
        if np.sum(mask) == 0:
            continue
        slice_agatston = 0
        pixel_volume = (dicom_attributes['slice_thickness']
                        * dicom_attributes['pixel_spacing'][0]
                        * dicom_attributes['pixel_spacing'][1])/3

        hu_image = create_hu_image(X[:, :, sliceIdx], dicom_attributes)
        labeled_mask, num_labels = measurements.label(mask,
                                                    structure=np.ones((3, 3)))
        for calc_idx in range(1, num_labels + 1):
            label = np.zeros(mask.shape)
            label[labeled_mask == calc_idx] = 1
            calc_object = hu_image * label
           

            calc_pixel_count = np.sum(label)
           
            # Remove small calcified objects
            if calc_pixel_count <= min_calc_object_pixels:
                continue
            calc_volume = calc_pixel_count * pixel_volume
            object_agatston = round(get_object_agatston(calc_object, calc_volume))
            slice_agatston += object_agatston
        print(f'agagston score for slice {sliceIdx} is {slice_agatston}')
        total_score += slice_agatston
    print(f'total agagston score: {total_score}')    
    return total_score

def create_new_mask(original_mask, warped_scan_hu):
    """Create a new mask for calcium deposits using GPU acceleration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert inputs to tensors on GPU
    original_mask = torch.from_numpy(original_mask).to(device)
    warped_scan_hu = torch.from_numpy(warped_scan_hu).to(device)
    
    # Create structuring element for dilation
    structuring_element = torch.ones((15, 15, 15), dtype=torch.uint8, device=device)
    
    # Perform dilation on GPU
    padding = 7  # (15-1)//2
    padded_mask = torch.nn.functional.pad(
        original_mask.unsqueeze(0).unsqueeze(0).float(),
        (padding, padding, padding, padding, padding, padding),
        mode='constant',
        value=0
    )
    
    # Use max_pool3d for dilation
    expanded_mask = torch.nn.functional.max_pool3d(
        padded_mask,
        kernel_size=15,
        stride=1,
        padding=0
    ).squeeze(0).squeeze(0)
    
    # Apply HU threshold
    threshold = 130
    new_mask = (warped_scan_hu >= threshold) & (expanded_mask > 0)
    
    # Convert back to numpy and return
    return new_mask.cpu().numpy()