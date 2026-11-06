#!/usr/bin/env python3
"""
Miscellaneous helpers for training/validation
"""

import logging
from typing import Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def denormalize_to_hu(normalized_data: np.ndarray, hu_min: float = -200, hu_max: float = 800) -> np.ndarray:
    """
    Denormalize data from [0,1] range back to HU values.

    Args:
        normalized_data: Data in [0,1] range
        hu_min: Minimum HU value (default: -200)
        hu_max: Maximum HU value (default: 800)

    Returns:
        Data in HU values
    """
    hu_data = normalized_data * (hu_max - hu_min) + hu_min
    hu_data = np.clip(hu_data, hu_min, hu_max)

    return hu_data


def compute_agatston_score(patch_data: np.ndarray, voxel_size: Tuple[float, float, float] = (0.625, 0.625, 2.5), debug: bool = False) -> float:
    """
    Compute Agatston score for a 3D patch using the correct methodology.

    Args:
        patch_data: 3D array of HU values (slices, height, width)
        voxel_size: Voxel size in mm (x, y, z)
        debug: Whether to print debug information

    Returns:
        Agatston score
    """
    calcium_threshold = 130
    pixel_area = voxel_size[0] * voxel_size[1]

    if debug:
        logger.info(f"Pixel area: {pixel_area:.4f} mm^2")

    total_agatston_score = 0.0
    total_lesions = 0

    for slice_idx in range(patch_data.shape[0]):
        slice_data = patch_data[slice_idx, :, :]

        # Find calcium pixels
        calcium_mask = slice_data >= calcium_threshold

        if not np.any(calcium_mask):
            continue

        # Label connected components (lesions)
        labeled_lesions, num_lesions = ndimage.label(calcium_mask, structure=ndimage.generate_binary_structure(2, 2))

        if debug and num_lesions > 0:
            logger.info(f"Slice {slice_idx}: Found {num_lesions} lesions")

        for lesion_id in range(1, num_lesions + 1):
            # Create mask for this lesion
            lesion_mask = (labeled_lesions == lesion_id)

            # Calculate lesion area in mm^2
            num_calcium_pixels = np.sum(lesion_mask)
            lesion_area = num_calcium_pixels * pixel_area

            # Ignore lesions smaller than 1 mm^2
            if lesion_area < 1.0:
                if debug:
                    logger.info(f"  Lesion {lesion_id}: Skipped (area {lesion_area:.2f} mm^2 < 1.0 mm^2)")
                continue

            lesion_hu_values = slice_data[lesion_mask]
            max_hu = np.max(lesion_hu_values)

            # Assign density factor based on max HU
            if 130 <= max_hu < 200:
                density_factor = 1
            elif 200 <= max_hu < 300:
                density_factor = 2
            elif 300 <= max_hu < 400:
                density_factor = 3
            else:  
                density_factor = 4

            lesion_score = lesion_area * density_factor
            total_agatston_score += lesion_score
            total_lesions += 1

            if debug:
                logger.info(f"  Lesion {lesion_id}: Area={lesion_area:.2f} mm^2, Max HU={max_hu:.0f}, Density={density_factor}, Score={lesion_score:.2f}")

    if debug:
        logger.info(f"Total Agatston score: {total_agatston_score:.2f} (from {total_lesions} lesions)")

    return float(total_agatston_score)
