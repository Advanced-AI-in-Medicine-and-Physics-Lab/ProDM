#!/usr/bin/env python3
"""
Simple evaluation script that directly loads H5 files like the working debug notebook.
No complex dataset classes, no preprocessing - just direct H5 file access.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import h5py
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random

# Make the vendored BBDM package importable regardless of the current directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
DEFAULT_CONFIG = os.path.join(_HERE, "configs", "bbdm_25d_config.yaml")

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def denorm_to_hu(normalized_data: np.ndarray) -> np.ndarray:
    """
    Denormalize data from [0, 1] to HU values.
    """
    hu_data = normalized_data * 1000 - 200
    return hu_data

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> BrownianBridgeModel:
    """Load the trained BBDM model"""
    logger.info(f"Loading model from {checkpoint_path}")

    config = OmegaConf.load(config_path)
    model = BrownianBridgeModel(config.model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Training epoch: {checkpoint.get('epoch', 'unknown')}")
    return model

def process_h5_file(file_path: str, model: BrownianBridgeModel, device: torch.device, 
                   output_dir: str, save_denormalized: bool = False, 
                   batch_size: int = 20, sampling_steps: int = 20) -> bool:
    """
    Process a single H5 file directly, just like the debug notebook.
    
    Args:
        file_path: Path to H5 file
        model: BBDM model
        device: Device to run inference on
        output_dir: Output directory
        save_denormalized: Whether to save denormalized data
        batch_size: Number of triplets to process in one batch
        sampling_steps: Number of sampling steps for BBDM (fewer = faster)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        with h5py.File(file_path, 'r') as f:
            motion_corrupted_patches = f['motion_corrupted'][:]
            motion_free_patches = f['motion_free'][:]
            
            logger.info(f"Loaded patches shape: {motion_corrupted_patches.shape}")
            
            predictions_list = []
            motion_corrupted_list = []
            motion_free_list = []
            
            all_triplets = []
            all_patch_indices = []
            all_slice_indices = []
            
            logger.info("Collecting all triplets...")
            for patch_idx in range(motion_corrupted_patches.shape[0]):
                patch_motion_corrupted = motion_corrupted_patches[patch_idx]  # (H, W, D)
                patch_motion_free = motion_free_patches[patch_idx]  # (H, W, D)
                
                center_slices = [6, 7, 8, 9, 10]
                num_slices = patch_motion_corrupted.shape[2]
                
                for slice_idx in center_slices:
                    if slice_idx < 1 or slice_idx >= num_slices - 1:
                        continue
                    
                    # Get triplet (3 adjacent slices)
                    motion_corrupted_triplet = patch_motion_corrupted[:, :, slice_idx-1:slice_idx+2]  # (H, W, 3)
                    motion_free_triplet = patch_motion_free[:, :, slice_idx-1:slice_idx+2]  # (H, W, 3)
                    
                    # Transpose to (3, H, W) format
                    motion_corrupted_triplet = motion_corrupted_triplet.transpose(2, 0, 1)  # (3, H, W)
                    motion_free_triplet = motion_free_triplet.transpose(2, 0, 1)  # (3, H, W)
                    
                    all_triplets.append((motion_corrupted_triplet, motion_free_triplet))
                    all_patch_indices.append(patch_idx)
                    all_slice_indices.append(slice_idx)
            
            logger.info(f"Collected {len(all_triplets)} triplets from {motion_corrupted_patches.shape[0]} patches")
            
            # Process triplets in batches
            num_triplets = len(all_triplets)
            patch_progress = tqdm(range(0, num_triplets, batch_size), 
                                desc=f"Processing {os.path.basename(file_path)}", 
                                leave=False)
            
            for batch_start in patch_progress:
                batch_end = min(batch_start + batch_size, num_triplets)
                current_batch_size = batch_end - batch_start
                
                if batch_start % (batch_size * 5) == 0:  # Log every 5 batches
                    logger.info(f"  Processing batch {batch_start//batch_size + 1}/{(num_triplets + batch_size - 1)//batch_size}")
                
                # Get current batch of triplets
                batch_triplets = all_triplets[batch_start:batch_end]
                batch_patch_indices = all_patch_indices[batch_start:batch_end]
                batch_slice_indices = all_slice_indices[batch_start:batch_end]
                
                motion_corrupted_batch = np.stack([triplet[0] for triplet in batch_triplets], axis=0)  # (N, 3, H, W)
                motion_free_batch = np.stack([triplet[1] for triplet in batch_triplets], axis=0)  # (N, 3, H, W)
                
                motion_corrupted_tensor = torch.from_numpy(motion_corrupted_batch).pin_memory().to(device, non_blocking=True)
                motion_free_tensor = torch.from_numpy(motion_free_batch).pin_memory().to(device, non_blocking=True)
                
                with torch.no_grad():
                    prediction = model.sample(motion_corrupted_tensor, show_progress=False)
                    # sample() returns the (N, 3, H, W) tensor directly; only unwrap
                    # if a (imgs, one_step_imgs) tuple/list is returned (sample_mid_step)
                    if isinstance(prediction, (tuple, list)):
                        prediction = prediction[0]

                prediction_np = prediction.cpu().numpy()  # (N, 3, H, W)
                motion_corrupted_np = motion_corrupted_tensor.cpu().numpy()  # (N, 3, H, W)
                motion_free_np = motion_free_tensor.cpu().numpy()  # (N, 3, H, W)
                
                del motion_corrupted_tensor, motion_free_tensor, prediction
                
                if batch_start % (batch_size * 3) == 0:
                    torch.cuda.empty_cache()
                
                if save_denormalized:
                    prediction_np = denorm_to_hu(prediction_np)
                    motion_corrupted_np = denorm_to_hu(motion_corrupted_np)
                    motion_free_np = denorm_to_hu(motion_free_np)
                
                # Store results for each triplet in the batch
                for i in range(current_batch_size):
                    patch_idx = batch_patch_indices[i]
                    slice_idx = batch_slice_indices[i]
                    
                    # Extract center slice from triplet (middle slice)
                    predicted_center_slice = prediction_np[i, 1, :, :]  # (H, W)
                    corrupted_center_slice = motion_corrupted_np[i, 1, :, :]  # (H, W)
                    free_center_slice = motion_free_np[i, 1, :, :]  # (H, W)
                    
                    # Store in lists to patchify later
                    predictions_list.append((patch_idx, slice_idx, predicted_center_slice))
                    motion_corrupted_list.append((patch_idx, slice_idx, corrupted_center_slice))
                    motion_free_list.append((patch_idx, slice_idx, free_center_slice))
            
            torch.cuda.empty_cache()
            
            logger.info("Organizing results by patch...")
            num_patches = motion_corrupted_patches.shape[0]
            patch_predictions = [[] for _ in range(num_patches)]
            patch_corrupted = [[] for _ in range(num_patches)]
            patch_free = [[] for _ in range(num_patches)]
            
            for patch_idx, slice_idx, data in predictions_list:
                patch_predictions[patch_idx].append((slice_idx, data))
            for patch_idx, slice_idx, data in motion_corrupted_list:
                patch_corrupted[patch_idx].append((slice_idx, data))
            for patch_idx, slice_idx, data in motion_free_list:
                patch_free[patch_idx].append((slice_idx, data))
            
            # Reconstruct patches from center slices
            final_predictions = []
            final_corrupted = []
            final_free = []
            
            for patch_idx in range(num_patches):
                if patch_predictions[patch_idx]:
                    # Sort by slice index and extract data
                    pred_slices = sorted(patch_predictions[patch_idx], key=lambda x: x[0])
                    corr_slices = sorted(patch_corrupted[patch_idx], key=lambda x: x[0])
                    free_slices = sorted(patch_free[patch_idx], key=lambda x: x[0])
                    
                    # Stack slices to form patch
                    predicted_patch = np.stack([data for _, data in pred_slices], axis=2)  # (H, W, num_slices)
                    corrupted_patch = np.stack([data for _, data in corr_slices], axis=2)  # (H, W, num_slices)
                    free_patch = np.stack([data for _, data in free_slices], axis=2)  # (H, W, num_slices)
                    
                    final_predictions.append(predicted_patch)
                    final_corrupted.append(corrupted_patch)
                    final_free.append(free_patch)
                else:
                    logger.warning(f"No valid triplets found for patch {patch_idx+1}")
                    # Create empty patch as fallback
                    final_predictions.append(np.zeros((64, 64, 5)))  # Assuming 64x64 patches
                    final_corrupted.append(np.zeros((64, 64, 5)))
                    final_free.append(np.zeros((64, 64, 5)))
        
        output_filename = os.path.basename(file_path).replace('.h5', '_simple_eval_results.h5')
        output_path = os.path.join(output_dir, output_filename)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset("predictions", data=np.stack(final_predictions, axis=0), compression="gzip")
            f.create_dataset("motion_corrupted", data=np.stack(final_corrupted, axis=0), compression="gzip")
            f.create_dataset("motion_free", data=np.stack(final_free, axis=0), compression="gzip")
            
            f.attrs['original_file'] = os.path.basename(file_path)
            f.attrs['num_patches'] = len(final_predictions)
            f.attrs['denormalized'] = save_denormalized
            f.attrs['denormalization_method'] = 'denorm_to_hu' if save_denormalized else 'none'
            f.attrs['batch_size'] = batch_size
            f.attrs['sampling_steps'] = sampling_steps
            f.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Saved results to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple BBDM evaluation - direct H5 file access")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str,
                        default=DEFAULT_CONFIG,
                        help='Path to model config (default: configs/bbdm_25d_config.yaml next to this script)')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation dataset dir of *_patches.h5 files (clean-HU patches)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device to use')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--save_denormalized', action='store_true',
                        help='Save denormalized data (HU values) instead of normalized data')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for processing triplets (higher = faster but more GPU memory)')
    parser.add_argument('--sampling_steps', type=int, default=20,
                        help='Number of sampling steps for BBDM (fewer = faster but potentially lower quality)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (FP16) for faster inference')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch for faster data loading')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = load_model(args.checkpoint_path, args.config_path, device)
    

    h5_files = [f for f in os.listdir(args.val_dir) if f.endswith('.h5')]
    if args.max_files is not None:
        h5_files = h5_files[:args.max_files]
    
    logger.info(f"Found {len(h5_files)} H5 files to process")
    
    successful_files = 0
    failed_files = 0
    start_time = time.time()
    
    for h5_file in tqdm(h5_files, desc="Processing files"):
        file_path = os.path.join(args.val_dir, h5_file)
        success = process_h5_file(file_path, model, device, args.output_dir, 
                                 args.save_denormalized, args.batch_size, args.sampling_steps)
        
        if success:
            successful_files += 1
        else:
            failed_files += 1
    
    total_time = time.time() - start_time
    avg_time_per_file = total_time / len(h5_files) if h5_files else 0
    
    logger.info(f"\n Evaluation complete!")
    logger.info(f"Successful files: {successful_files}")
    logger.info(f"Failed files: {failed_files}")
    logger.info(f"Total files processed: {len(h5_files)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per file: {avg_time_per_file:.2f} seconds")
    logger.info(f"Results saved to: {args.output_dir}")
    
if __name__ == "__main__":
    main()
