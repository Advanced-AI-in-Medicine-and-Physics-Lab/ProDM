#!/usr/bin/env python3
"""
Script to create optimized validation set with only 6 patches per h5 file.
This reduces validation time by sampling only 3 random patches and 3 calcium patches
from each original validation file, with distance requirements for calcium patches.
"""

import os
import h5py
import numpy as np
import json
import random
import argparse
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HU window used to normalize the patches ([0,1] = [HU_MIN, HU_MAX]) and the
# calcium threshold the Agatston scorer uses. Random patches are only eligible
# if their ground-truth max HU is below this threshold, so the scorer can never
# assign them a nonzero Agatston score.
HU_MIN = -200.0
HU_MAX = 800.0
CALCIUM_HU_THRESHOLD = 130.0

def calculate_distance(center1: List[int], center2: List[int]) -> float:
    """Calculate Euclidean distance between two patch centers"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(center1, center2)))

def select_distant_calcium_patches(
    calcium_patches: List[Dict], 
    num_patches: int = 3, 
    min_distance: float = 50.0
) -> List[Dict]:
    """
    Select calcium patches that are sufficiently distant from each other.
    
    Args:
        calcium_patches: List of calcium patch metadata
        num_patches: Number of patches to select
        min_distance: Minimum distance between selected patches
    
    Returns:
        List of selected calcium patch metadata
    """
    if len(calcium_patches) <= num_patches:
        return calcium_patches
    
    selected = []
    available = calcium_patches.copy()

    first_patch = random.choice(available)
    selected.append(first_patch)
    available.remove(first_patch)
    
    while len(selected) < num_patches and available:
        # Find patches that are distant from all selected patches
        distant_patches = []
        for patch in available:
            is_distant = True
            for selected_patch in selected:
                distance = calculate_distance(
                    patch['patch_center'], 
                    selected_patch['patch_center']
                )
                if distance < min_distance:
                    is_distant = False
                    break
            
            if is_distant:
                distant_patches.append(patch)
        
        if distant_patches:
            chosen_patch = random.choice(distant_patches)
            selected.append(chosen_patch)
            available.remove(chosen_patch)
        else:
            # If no distant patches found, select the one with maximum minimum distance
            best_patch = None
            best_min_distance = 0
            
            for patch in available:
                min_distances = []
                for selected_patch in selected:
                    distance = calculate_distance(
                        patch['patch_center'], 
                        selected_patch['patch_center']
                    )
                    min_distances.append(distance)
                
                min_dist = min(min_distances)
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_patch = patch
            
            if best_patch:
                selected.append(best_patch)
                available.remove(best_patch)
            else:
                break
    
    return selected

def create_optimized_validation_file(
    input_file_path: str,
    output_file_path: str,
    num_random_patches: int = 3,
    num_calcium_patches: int = 3,
    min_calcium_distance: float = 50.0,
    show_progress: bool = True
) -> bool:
    """
    Create an optimized validation h5 file with only selected patches.
    
    Args:
        input_file_path: Path to original validation h5 file
        output_file_path: Path to output optimized h5 file
        num_random_patches: Number of random patches to include
        num_calcium_patches: Number of calcium patches to include
        min_calcium_distance: Minimum distance between calcium patches
        show_progress: Whether to show progress messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with h5py.File(input_file_path, 'r') as f_in:
            metadata_str = f_in.attrs['metadata']
            if isinstance(metadata_str, str):
                metadata: Dict[str, Any] = json.loads(metadata_str)
            else:
                metadata = metadata_str 
            
            # Separate patches by sampler type
            random_patches = []
            calcium_patches = []
            
            for i, patch_info in enumerate(metadata['patch_info']):  # type: ignore
                if patch_info['sampler_type'] == 'random':
                    # Only keep random patches whose ground truth has no calcium
                    # (max HU < threshold), so Agatston scoring returns exactly 0.
                    norm_max = float(np.max(f_in['motion_free'][i]))
                    max_hu = norm_max * (HU_MAX - HU_MIN) + HU_MIN
                    if max_hu >= CALCIUM_HU_THRESHOLD:
                        continue
                    random_patches.append({
                        'index': i,
                        'patch_center': patch_info['patch_center'],
                        'sampler_type': patch_info['sampler_type']
                    })
                elif patch_info['sampler_type'] == 'calcium':
                    calcium_patches.append({
                        'index': i,
                        'patch_center': patch_info['patch_center'],
                        'sampler_type': patch_info['sampler_type']
                    })
            
            if show_progress:
                logger.info(f"File {os.path.basename(input_file_path)}: "
                           f"{len(random_patches)} calcium-free random, {len(calcium_patches)} calcium patches")
            
            selected_random = random.sample(
                random_patches, 
                min(num_random_patches, len(random_patches))
            )
   
            selected_calcium = select_distant_calcium_patches(
                calcium_patches, 
                num_calcium_patches, 
                min_calcium_distance
            )
            
            # Combine selected patches
            selected_patches = selected_random + selected_calcium
            selected_indices = [patch['index'] for patch in selected_patches]
            
            if show_progress:
                logger.info(f"Selected {len(selected_random)} random and {len(selected_calcium)} calcium patches")
            
            with h5py.File(output_file_path, 'w') as f_out:
                sorted_indices = sorted(selected_indices)
                motion_corrupted_data = f_in['motion_corrupted'][sorted_indices] 
                motion_free_data = f_in['motion_free'][sorted_indices] 
                f_out.create_dataset('motion_corrupted', data=motion_corrupted_data, compression='gzip')
                f_out.create_dataset('motion_free', data=motion_free_data, compression='gzip')
                
                new_patch_info = []
                index_to_patch = {patch['index']: patch for patch in selected_patches}
                
                for i, original_idx in enumerate(sorted_indices):
                    patch = index_to_patch[original_idx]
                    new_patch_info.append({
                        'patch_idx': i,
                        'patch_center': patch['patch_center'],
                        'sampler_type': patch['sampler_type'],
                        'original_index': patch['index']  # Keep track of original index
                    })
                
                new_metadata = {
                    'patient_id': metadata['patient_id'],  
                    'motion_id': metadata['motion_id'],  
                    'num_patches': len(selected_patches),
                    'patch_size': metadata['patch_size'], 
                    'original_shape': metadata['original_shape'],  
                    'original_num_patches': metadata['num_patches'], 
                    'selected_indices': sorted_indices, 
                    'patch_info': new_patch_info
                }
                
                f_out.attrs['metadata'] = json.dumps(new_metadata)
            
            if show_progress:
                logger.info(f"Created optimized file: {output_file_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error processing {input_file_path}: {str(e)}")
        return False

def create_optimized_validation_set(
    input_dir: str,
    output_dir: str,
    num_random_patches: int = 3,
    num_calcium_patches: int = 3,
    min_calcium_distance: float = 50.0,
    show_progress: bool = True
) -> None:
    """
    Create optimized validation set from all h5 files in input directory.
    
    Args:
        input_dir: Directory containing original validation h5 files
        output_dir: Directory to save optimized h5 files
        num_random_patches: Number of random patches per file
        num_calcium_patches: Number of calcium patches per file
        min_calcium_distance: Minimum distance between calcium patches
        show_progress: Whether to show progress messages
    """
    os.makedirs(output_dir, exist_ok=True)
    
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    h5_files.sort()
    
    if show_progress:
        logger.info(f"Found {len(h5_files)} h5 files in {input_dir}")
    
    successful_files = 0
    total_files = len(h5_files)
    
    for h5_file in h5_files:
        input_path = os.path.join(input_dir, h5_file)
        output_path = os.path.join(output_dir, h5_file)
        
        if show_progress:
            logger.info(f"Processing {h5_file} ({successful_files + 1}/{total_files})")
        
        if create_optimized_validation_file(
            input_path, 
            output_path, 
            num_random_patches, 
            num_calcium_patches, 
            min_calcium_distance,
            show_progress
        ):
            successful_files += 1
    
    if show_progress:
        logger.info(f"Successfully created {successful_files}/{total_files} optimized files")
        logger.info(f"Optimized files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create optimized validation set")
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help='Directory containing original validation h5 files'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Directory to save optimized h5 files'
    )
    parser.add_argument(
        '--num_random', 
        type=int, 
        default=3,
        help='Number of random patches to include per file (default: 3)'
    )
    parser.add_argument(
        '--num_calcium', 
        type=int, 
        default=3,
        help='Number of calcium patches to include per file (default: 3)'
    )
    parser.add_argument(
        '--min_distance', 
        type=float, 
        default=50.0,
        help='Minimum distance between calcium patches (default: 50.0)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no_progress', 
        action='store_true',
        help='Disable progress messages and logging'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.no_progress:
        logging.getLogger().setLevel(logging.ERROR)
    
    if not args.no_progress:
        logger.info("Creating optimized validation set...")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Patches per file: {args.num_random} random + {args.num_calcium} calcium")
        logger.info(f"Minimum calcium distance: {args.min_distance}")
        logger.info(f"Random seed: {args.seed}")
    
    create_optimized_validation_set(
        args.input_dir,
        args.output_dir,
        args.num_random,
        args.num_calcium,
        args.min_distance,
        show_progress=not args.no_progress
    )
    
    if not args.no_progress:
        logger.info("Optimization complete!")

if __name__ == "__main__":
    main() 