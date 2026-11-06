#!/usr/bin/env python3
import os
import glob
import h5py
import numpy as np
import torch
import torchio as tio
import nibabel as nib
from nibabel.processing import resample_from_to
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HU_WINDOW = (-200.0, 800.0)
HU_OFFSET = 1024.0


def normalize_to_hu_window(patch, input_is_raw: bool, hu_window: Tuple[float, float] = HU_WINDOW):
    """Normalize a patch to [0, 1] using the fixed HU window.

    input_is_raw=True  : patch is raw CT intensity [0, 4095]; convert to HU first
                         (subtract HU_OFFSET). Use for legacy raw-intensity volumes.
    input_is_raw=False : patch is already in HU (e.g. gated_nifti_hu volumes and
                         the motion-simulation recons); clip + scale directly.

    Returns (normalized_patch, norm_params); norm_params records the window used
    so the metadata matches the previous CTNormalizer output.
    """
    lo, hi = hu_window
    hu = (patch - HU_OFFSET) if input_is_raw else patch
    if isinstance(hu, torch.Tensor):
        clipped = torch.clamp(hu, lo, hi)
    else:
        clipped = np.clip(hu, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    norm_params = {
        'data_min': lo, 'data_max': hi,
        'hu_window_min': lo, 'hu_window_max': hi,
        'input_is_raw': bool(input_is_raw),
    }
    return normalized, norm_params


@dataclass
class PatchInfo:
    """Information about a single patch"""
    subject_id: str
    motion_type: str
    patch_idx: int
    patch_coords: Tuple[int, int, int]  # (x, y, z) coordinates of patch center
    original_shape: Tuple[int, int, int]  # Original image shape
    sampler_type: str  # 'random' or 'calcium'
    motion_corrupted_norm_params: Optional[Dict] = None
    motion_free_norm_params: Optional[Dict] = None

class RandomSampler:
    """Randomly samples patch centers from an image"""
    def __init__(self, patch_size: Tuple[int, int, int]):
        self.patch_size = patch_size
        
    def get_valid_centers(self, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get all valid center points for patches"""
        half_patch = tuple(s // 2 for s in self.patch_size)
        valid_centers = []
        
        for x in range(half_patch[0], image_shape[0] - half_patch[0]):
            for y in range(half_patch[1], image_shape[1] - half_patch[1]):
                for z in range(half_patch[2], image_shape[2] - half_patch[2]):
                    valid_centers.append((x, y, z))
                    
        return valid_centers
    
    def sample_centers(self, image_shape: Tuple[int, int, int], num_samples: int) -> List[Tuple[int, int, int]]:
        """Sample random centers for patches"""
        valid_centers = self.get_valid_centers(image_shape)
        if len(valid_centers) == 0:
            return []
            
        indices = np.random.choice(len(valid_centers), size=min(num_samples, len(valid_centers)), replace=False)
        return [valid_centers[i] for i in indices]

class CalciumSampler:
    """Samples patch centers from calcium regions in the mask"""
    def __init__(self, patch_size: Tuple[int, int, int]):
        self.patch_size = patch_size
        
    def get_valid_centers(self, mask: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get all valid center points where mask > 0, without offsets"""
        half_patch = tuple(s // 2 for s in self.patch_size)
        valid_centers = []
        
        try:
            calcium_coords = np.where(mask > 0)
            
            if len(calcium_coords) == 0 or len(calcium_coords[0]) == 0:
                logger.warning(f"No calcium coordinates found in mask with shape {mask.shape}")
                return valid_centers
            
            if len(calcium_coords) != 3:
                logger.warning(f"Expected 3D coordinates, got {len(calcium_coords)} dimensions")
                return valid_centers
            
            for x, y, z in zip(*calcium_coords):
                # Check if center would result in a valid patch
                if (x >= half_patch[0] and x < mask.shape[0] - half_patch[0] and
                    y >= half_patch[1] and y < mask.shape[1] - half_patch[1] and
                    z >= half_patch[2] and z < mask.shape[2] - half_patch[2]):
                    valid_centers.append((x, y, z))
        except Exception as e:
            logger.error(f"Error in get_valid_centers: {e}")
            logger.error(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            logger.error(f"Mask min: {mask.min()}, max: {mask.max()}")
            return valid_centers
       
        return valid_centers
    
    def sample_centers(self, mask: np.ndarray, num_samples: int, min_distance: int = 20) -> List[Tuple[int, int, int]]:
        """Sample centers from calcium regions with minimum distance constraint"""
        logger.info(f"CalciumSampler.sample_centers called with mask shape: {mask.shape}, num_samples: {num_samples}")
        valid_centers = self.get_valid_centers(mask)
        logger.info(f"get_valid_centers returned {len(valid_centers)} centers")
        if len(valid_centers) == 0:
            logger.warning("No valid centers found in mask")
            return []
        
        # If we have fewer centers than requested samples, return all available
        if len(valid_centers) <= num_samples:
            return valid_centers
        
        # Greedy selection with minimum distance constraint
        selected_centers = []
        remaining_centers = valid_centers.copy()
        
        # Start with a random center
        if remaining_centers:
            first_center = remaining_centers.pop(np.random.randint(len(remaining_centers)))
            logger.info(f"Selected first center: {first_center} (type: {type(first_center)})")
            selected_centers.append(first_center)
        
        # Iteratively select centers that are at least min_distance away from all selected centers
        for i in range(num_samples):
            while len(selected_centers) < num_samples and remaining_centers:
                # Find centers that are at least min_distance away from all selected centers
                valid_candidates = []
                
                for center in remaining_centers:
                    is_valid = True
                    for selected_center in selected_centers:
                        try:
                            # Calculate Euclidean distance (ignore z-axis)
                            # Ensure center and selected_center are tuples/lists with at least 2 elements
                            if len(center) < 2 or len(selected_center) < 2:
                                logger.warning(f"Invalid center format: center={center}, selected_center={selected_center}")
                                is_valid = False
                                break
                            
                            # Calculate Euclidean distance using only x and y coordinates (ignore z)
                            distance = np.sqrt((center[0] - selected_center[0]) ** 2 + (center[1] - selected_center[1]) ** 2)
                            if distance < min_distance:
                                is_valid = False
                                break
                        except Exception as e:
                            logger.error(f"Error calculating distance between {center} and {selected_center}: {e}")
                            is_valid = False
                            break
                    
                    if is_valid:
                        valid_candidates.append(center)
                
                if not valid_candidates:
                    break
                
                # Randomly select from valid candidates
                selected_center = valid_candidates[np.random.randint(len(valid_candidates))]
                selected_centers.append(selected_center)
                remaining_centers.remove(selected_center)
        
        return selected_centers

def extract_patch(
    image: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int]
) -> np.ndarray:
    """Extract a patch from an image centered at the given coordinates"""
    half_patch = tuple(s // 2 for s in patch_size)
    x, y, z = center
    
    x_start = x - half_patch[0]
    x_end = x + half_patch[0]
    y_start = y - half_patch[1]
    y_end = y + half_patch[1]
    z_start = z - half_patch[2]
    z_end = z + half_patch[2]
    
    return image[x_start:x_end, y_start:y_end, z_start:z_end]


def process_subject(
    motion_corrupted_file: str,
    motion_free_file: str,
    mask_file: str,
    output_dir: str,
    patch_size: Tuple[int, int, int] = (64, 64, 5),
    max_calcium_patches: int = 45,
    max_random_patches: int = 5,
    num_transforms: int = 4, 
    motion_id: Optional[int] = None,
    min_distance: int = 20,
    input_is_raw: bool = False

) -> Optional[str]:
    """Process a single subject and generate patches"""
    try:
        logger.info(f"Processing: {motion_corrupted_file}")

        # Resample the motion-free volume onto the mask grid so that
        # motion_corrupted, motion_free, and mask all share the same alignment.
        mask_nib = nib.load(mask_file)
        motion_free_nib = nib.load(motion_free_file)
        motion_free_aligned = resample_from_to(
            motion_free_nib, (mask_nib.shape, mask_nib.affine), order=1
        )

        motion_corrupted_img = tio.ScalarImage(motion_corrupted_file)

        # intensity values are HU by default, or raw [0, 4095] if --input_is_raw
        motion_corrupted_data = motion_corrupted_img.data.squeeze().numpy()
        motion_free_data = motion_free_aligned.get_fdata().astype(np.float32)
        mask_data = mask_nib.get_fdata().astype(np.float32)
        
        logger.info(f"Data shapes - motion_corrupted: {motion_corrupted_data.shape}, motion_free: {motion_free_data.shape}, mask: {mask_data.shape}")
        logger.info(f"Mask stats - min: {mask_data.min()}, max: {mask_data.max()}, non-zero: {np.count_nonzero(mask_data)}")
        
        random_sampler = RandomSampler(patch_size)
        calcium_sampler = CalciumSampler(patch_size)
        
        # Generate patches using random sampler
        logger.info(f"Sampling {max_random_patches} random patches and {max_calcium_patches} calcium patches")
        random_centers = random_sampler.sample_centers(motion_corrupted_data.shape, max_random_patches)
        logger.info(f"Found {len(random_centers)} random centers")
        
        try:
            logger.info("Starting calcium sampling...")
            calcium_centers = calcium_sampler.sample_centers(mask_data, max_calcium_patches, min_distance=min_distance)
            logger.info(f"Found {len(calcium_centers)} calcium centers")
        except Exception as e:
            logger.error(f"Error in calcium sampling: {e}")
            logger.error(f"Mask details - shape: {mask_data.shape}, dtype: {mask_data.dtype}")
            logger.error(f"Mask unique values: {np.unique(mask_data)}")
            raise e
        
        motion_corrupted_patches = []
        motion_free_patches = []
        patch_info_list = []
        
        # Process random patches
        for i, center in enumerate(random_centers):
            motion_corrupted_patch = extract_patch(motion_corrupted_data, center, patch_size)
            motion_free_patch = extract_patch(motion_free_data, center, patch_size)
            
            motion_corrupted_patch = torch.from_numpy(motion_corrupted_patch).float()
            motion_free_patch = torch.from_numpy(motion_free_patch).float()
            
            # Apply HU-window normalization with parameter tracking
            motion_corrupted_normalized, motion_corrupted_norm_params = normalize_to_hu_window(
                motion_corrupted_patch, input_is_raw
            )
            motion_free_normalized, motion_free_norm_params = normalize_to_hu_window(
                motion_free_patch, input_is_raw
            )
            
            motion_corrupted_patches.append(motion_corrupted_normalized)
            motion_free_patches.append(motion_free_normalized)
            
            # Store normalization parameters in patch info
            patch_info = PatchInfo(
                subject_id=extract_subject_id(motion_corrupted_file),
                motion_type=extract_motion_type(motion_corrupted_file),
                patch_idx=i,
                patch_coords=center,
                original_shape=motion_corrupted_data.shape,
                sampler_type='random'
            )
            
            # Add normalization parameters to patch info
            patch_info.motion_corrupted_norm_params = motion_corrupted_norm_params
            patch_info.motion_free_norm_params = motion_free_norm_params
            
            patch_info_list.append(patch_info)
        
        # Process calcium patches
        for i, center in enumerate(calcium_centers):
            motion_corrupted_patch = extract_patch(motion_corrupted_data, center, patch_size)
            motion_free_patch = extract_patch(motion_free_data, center, patch_size)
            
            motion_corrupted_patch = torch.from_numpy(motion_corrupted_patch).float()
            motion_free_patch = torch.from_numpy(motion_free_patch).float()

            # Apply HU-window normalization with parameter tracking
            motion_corrupted_normalized, motion_corrupted_norm_params = normalize_to_hu_window(
                motion_corrupted_patch, input_is_raw
            )
            motion_free_normalized, motion_free_norm_params = normalize_to_hu_window(
                motion_free_patch, input_is_raw
            )
            
            motion_corrupted_patches.append(motion_corrupted_normalized)
            motion_free_patches.append(motion_free_normalized)

            # Store normalization parameters in patch info
            patch_info = PatchInfo(
                subject_id=extract_subject_id(motion_corrupted_file),
                motion_type=extract_motion_type(motion_corrupted_file),
                patch_idx= i + max_random_patches,
                patch_coords=center,
                original_shape=motion_corrupted_data.shape,
                sampler_type='calcium'
            )
            
            # Add normalization parameters to patch info
            patch_info.motion_corrupted_norm_params = motion_corrupted_norm_params
            patch_info.motion_free_norm_params = motion_free_norm_params
            
            patch_info_list.append(patch_info)
        
        h5_path = save_patches_to_h5(
            motion_corrupted_patches,
            motion_free_patches,
            patch_info_list,
            output_dir,
            motion_id
        )
        
        return h5_path
        
    except Exception as e:
        logger.error(f"Error processing {motion_corrupted_file}: {str(e)}")
        return None

def preprocess_dataset(
    motion_corrupted_dir: str,
    motion_free_dir: str,
    mask_dir: str,
    output_dir: str,
    ignore_list_file: str,
    patch_size: Tuple[int, int, int] = (64, 64, 5),
    max_calcium_patches: int = 45,
    max_random_patches: int = 5,
    num_workers: int = 4,
    motion_free_only: bool = False,
    p_motion_free: float = 0.3,  # Probability of processing motion-free scans
    motion_id_start: int = 1,  # Starting index for motion IDs
    min_distance: int = 20,
    input_is_raw: bool = False

) -> List[str]:
    """Preprocess entire dataset using multiprocessing
    
    Args:
        motion_corrupted_dir: Directory containing motion-corrupted CT scans
        motion_free_dir: Directory containing motion-free CT scans
        mask_dir: Directory containing mask files
        output_dir: Directory to save patches
        ignore_list_file: Path to file containing list of subjects to ignore
        patch_size: Size of patches to extract
        max_calcium_patches: Maximum number of calcium patches to extract per subject
        max_random_patches: Maximum number of random patches to extract per subject
        num_workers: Number of worker processes
        motion_free_only: Whether to only process motion-free scans
        p_motion_free: Probability of processing motion-free scans (default: 0.3)
        motion_id_start: Starting index for motion IDs (default: 1)
        min_distance: Minimum distance between patch centers (default: 20)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read ignore list
    with open(ignore_list_file, 'r') as f:
        ignore_list = set(line.strip() for line in f)
   
    # Get all subject IDs from motion-corrupted directory structure
    subject_dirs = [d for d in os.listdir(motion_corrupted_dir) 
                   if os.path.isdir(os.path.join(motion_corrupted_dir, d))]
    
    # Create list of (motion_corrupted, motion_free, prob_map) tuples
    file_pairs = []
    for subject_id in subject_dirs:
        if subject_id in ignore_list:
            continue
            
        mask_file = os.path.join(mask_dir, f"mask_{subject_id}.nii.gz")
        
        if not os.path.exists(mask_file):
            logger.warning(f"Mask not found for {subject_id}")
            continue

        motion_free_file = os.path.join(motion_free_dir, f"{subject_id}.nii")
        if not os.path.exists(motion_free_file):
            logger.warning(f"Motion-free file not found for {subject_id}")
            continue
            
        if motion_free_only:
            # For motion-free only mode, randomly decide whether to process this subject
            if random.random() > p_motion_free:
                continue
           
            # Create a motion-free file pair
            output_file = os.path.join(output_dir, f"patient_{subject_id}_motionfree_patches.h5")
            print("output file", output_file)
        
            if os.path.exists(output_file):
                logger.info(f"Skipping {motion_free_file} - output file already exists: {output_file}")
                continue
        
            motion_corrupted_file = os.path.join(motion_corrupted_dir, subject_id, "reconstructions", "motionfree.nii.gz")
            # Use the motion-free file as both input and target
            file_pairs.append((motion_corrupted_file, motion_free_file, mask_file))
        
        else:
            reconstructions_dir = os.path.join(motion_corrupted_dir, subject_id, "reconstructions")
            if not os.path.exists(reconstructions_dir):
                logger.warning(f"Reconstructions directory not found for {subject_id}")
                continue
                
            motion_corrupted_files = sorted(os.listdir(reconstructions_dir))  # Sort for consistent ordering
            
            for motion_idx, motion_corrupted_file in enumerate(motion_corrupted_files):
                motion_id = motion_id_start + motion_idx  # Assign motion ID based on order
                
                # full path for motion corrupted file
                motion_corrupted_file = os.path.join(motion_corrupted_dir, subject_id, "reconstructions", motion_corrupted_file)
                output_file = os.path.join(output_dir, f"patient_{subject_id}_motion_{motion_id}_patches.h5")
                
                if os.path.exists(output_file):
                    logger.info(f"Skipping {motion_corrupted_file} - output file already exists: {output_file}")
                    continue
                    
                file_pairs.append((motion_corrupted_file, motion_free_file, mask_file, motion_id))
    
    logger.info(f"Found {len(file_pairs)} file pairs to process")
    
    results = []
    
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            # Create a list of arguments for each process
            process_args = []
            for file_tuple in file_pairs:
                if len(file_tuple) == 4:  # has motion_id
                    motion_corrupted_file, motion_free_file, mask_file, motion_id = file_tuple
                    process_args.append((motion_corrupted_file, motion_free_file, mask_file, output_dir, patch_size, max_calcium_patches, max_random_patches, 4, motion_id, min_distance, input_is_raw))
                else:  # motion_free_only case
                    motion_corrupted_file, motion_free_file, mask_file = file_tuple
                    process_args.append((motion_corrupted_file, motion_free_file, mask_file, output_dir, patch_size, max_calcium_patches, max_random_patches, 4, None, min_distance, input_is_raw))
            
            # Process in parallel with progress bar
            for result in tqdm(pool.starmap(process_subject, process_args), total=len(process_args), desc="Processing subjects"):
                if result is not None:
                    results.append(result)
    else:
        # Process sequentially
        for file_tuple in tqdm(file_pairs, desc="Processing subjects"):
            if len(file_tuple) == 4:  # Has motion_id
                motion_corrupted_file, motion_free_file, mask_file, motion_id = file_tuple
                result = process_subject(
                    motion_corrupted_file=motion_corrupted_file,
                    motion_free_file=motion_free_file,
                    mask_file=mask_file,
                    output_dir=output_dir,
                    patch_size=patch_size,
                    max_calcium_patches=max_calcium_patches,
                    max_random_patches=max_random_patches,
                    motion_id=motion_id,
                    min_distance=min_distance,
                    input_is_raw=input_is_raw
                )
            else:  # motion_free_only case
                motion_corrupted_file, motion_free_file, mask_file = file_tuple
                result = process_subject(
                    motion_corrupted_file=motion_corrupted_file,
                    motion_free_file=motion_free_file,
                    mask_file=mask_file,
                    output_dir=output_dir,
                    patch_size=patch_size,
                    max_calcium_patches=max_calcium_patches,
                    max_random_patches=max_random_patches,
                    motion_id=None,
                    input_is_raw=input_is_raw
                )
            if result is not None:
                results.append(result)
    
    return results

def save_patches_to_h5(
    motion_corrupted_patches: List[torch.Tensor],
    motion_free_patches: List[torch.Tensor],
    patch_info_list: List[PatchInfo],
    output_dir: str,
    motion_id: Optional[int] = None
) -> str:
    """
    Save patches to HDF5 file.
    
    Args:
        motion_corrupted_patches: List of motion-corrupted patches
        motion_free_patches: List of motion-free patches
        patch_info_list: List of patch information
        output_dir: Directory to save HDF5 file
        
    Returns:
        Path to saved HDF5 file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    subject_id = patch_info_list[0].subject_id
    motion_type = patch_info_list[0].motion_type
    
    # Use motion_id if provided, otherwise fall back to motion_type
    if motion_id is not None:
        h5_path = os.path.join(output_dir, f"patient_{subject_id}_motion_{motion_id}_patches.h5")
    elif motion_type != "motionfree":
        h5_path = os.path.join(output_dir, f"patient_{subject_id}_motion_{motion_type}_patches.h5")
    else:
        h5_path = os.path.join(output_dir, f"patient_{subject_id}_motionfree_patches.h5")
    
    if "motionfree" in h5_path:
        print(h5_path, "motionfree")
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset(
            'motion_corrupted',
            data=torch.stack(motion_corrupted_patches).numpy(),
            compression='gzip'
        )
        f.create_dataset(
            'motion_free',
            data=torch.stack(motion_free_patches).numpy(),
            compression='gzip'
        )
        
        metadata = {
            'patient_id': subject_id,
            'motion_id': motion_id if motion_id is not None else motion_type,
            'num_patches': len(patch_info_list),
            'patch_size': motion_corrupted_patches[0].shape,
            'original_shape': patch_info_list[0].original_shape,
            'patch_info': [
                {
                    'patch_idx': int(info.patch_idx), 
                    'patch_center': [int(x) for x in info.patch_coords],
                    'sampler_type': info.sampler_type,
                    'motion_corrupted_norm_params': info.motion_corrupted_norm_params,
                    'motion_free_norm_params': info.motion_free_norm_params
                }
                for info in patch_info_list
            ]
        }
        try:
            f.attrs['metadata'] = json.dumps(metadata)
        except Exception as e:
            logger.error(f"Error saving metadata for {h5_path}: {str(e)}")
            raise e
    
    return h5_path

def extract_subject_id(file_path):
    """
    Extract the subject ID from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Subject ID extracted from the file path
    """
    # Split the path into parts
    path_parts = file_path.split(os.sep)
    
    if "reconstructions" in path_parts:
        reconstructions_idx = path_parts.index("reconstructions")
        if reconstructions_idx > 0:
            subject_id = path_parts[reconstructions_idx - 1]
            if subject_id.isdigit() or subject_id.replace('_', '').isalnum():
                return subject_id
    
    # Fallback: look for any numeric part in the path
    for part in path_parts:
        if part.isdigit():
            return part
    
    # Final fallback: extract from filename
    basename = os.path.basename(file_path)
    if 'recon' in basename:
        return basename.split('.')[0].split('_')[-1]
    else:
        # this is original motion-free image
        return basename.split(".")[0]
    

def extract_motion_type(file_path):
    """
    Extract the motion type from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Motion type extracted from the file path
    """
    
    if "motionfree" in file_path:
        return "motionfree"
    # Fallback to filename
    filename = str(file_path).split('/')[-1]
    motion_type = filename.replace('recon_', '').replace('.nii.gz', '').split('_')[-1]
    return motion_type

if __name__ == "__main__":  
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CT scans into patches")
    parser.add_argument("--motion_corrupted_dir", type=str, required=True, help="Directory containing motion-corrupted CT scans")
    parser.add_argument("--motion_free_dir", type=str, required=True, help="Directory containing motion-free CT scans")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save patches")
    parser.add_argument("--ignore_list_file", type=str, required=True, help="Path to file containing list of subjects to ignore")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 16], help="Size of patches to extract")
    parser.add_argument("--max_calcium_patches", type=int, default=45, help="Maximum number of calcium patches to extract per subject")
    parser.add_argument("--max_random_patches", type=int, default=5, help="Maximum number of random patches to extract per subject")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--motion_free_only", type=bool, default=False, help="Generate only motion-free patches")
    parser.add_argument("--p_motion_free", type=float, default=0.3, help="Probability of processing motion-free scans")
    parser.add_argument("--motion_id_start", type=int, default=1, help="Starting index for motion IDs")
    parser.add_argument("--min_distance", type=int, default=20, help="Minimum distance between patch centers")
    parser.add_argument("--input_is_raw", action="store_true",
                        help="Input volumes are raw CT intensity [0,4095] (subtract 1024 to get HU). "
                             "Omit for HU inputs such as gated_nifti_hu / motion-simulation recons.")

    args = parser.parse_args()

    h5_files = preprocess_dataset(
        args.motion_corrupted_dir,
        args.motion_free_dir,
        args.mask_dir,
        args.output_dir,
        args.ignore_list_file,
        tuple(args.patch_size),
        args.max_calcium_patches,
        args.max_random_patches,
        args.num_workers,
        args.motion_free_only,
        args.p_motion_free,
        args.motion_id_start,
        args.min_distance,
        args.input_is_raw
    )
    
    logger.info(f"Preprocessing complete. Generated {len(h5_files)} HDF5 files.") 