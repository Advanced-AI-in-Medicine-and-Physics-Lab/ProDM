#!/usr/bin/env python3
"""
Visualization utilities for monitoring training progress
"""

import os
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from data.patched_triplets_dataset import PatchedTripletsDataset_25D


def plot_central_slice_comparison(
    predictions_for_patches: List[np.ndarray],
    ground_truth_for_patches: List[np.ndarray],
    save_dir: str,
    epoch: int,
    use_wandb: bool = False,
):
    """
    Plot central slice predictions vs ground truth for each patch

    Args:
        predictions_for_patches: List of numpy arrays, each of shape (num_slices, height, width)
        ground_truth_for_patches: List of numpy arrays, each of shape (num_slices, height, width)
        save_dir: Directory to save plots
        epoch: Current epoch number
        use_wandb: Whether to log images to wandb
    """
    num_patches = len(predictions_for_patches)
    
    fig, axes = plt.subplots(num_patches, 2, figsize=(12, 5*num_patches + 10))
    if num_patches == 1:
        axes = axes.reshape(1, -1)

    for i, (predictions_for_patch, ground_truth_for_patch) in enumerate(zip(predictions_for_patches, ground_truth_for_patches)):
        num_slices = predictions_for_patch.shape[0]
        central_slice_idx = num_slices // 2  # Get the central slice index

        axes[i, 0].imshow(predictions_for_patch[central_slice_idx, :, :], cmap='gray')
        axes[i, 0].set_title(f'Prediction - Central Slice for Patch {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(ground_truth_for_patch[central_slice_idx, :, :], cmap='gray')
        axes[i, 1].set_title(f'Ground Truth - Central Slice for Patch {i+1}')
        axes[i, 1].axis('off')

    plt.suptitle(f'Epoch {epoch+1} - Central Slice Predictions vs Ground Truth - Validation Set')
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f'epoch_{epoch+1}', f'central_slice_comparison_epoch_{epoch+1}.png')
    plt.savefig(plot_path)

    if use_wandb:
        wandb.log({
            f"central_slice_comparison_epoch_{epoch+1}": wandb.Image(plot_path),
            "epoch": epoch + 1
        })

    plt.close()


def plot_single_slice(
    axes: np.ndarray,
    motion_corrupted: np.ndarray,
    prediction: np.ndarray,
    motion_free: np.ndarray,
    slice_idx: int,
    metadata: Dict,
    slice_name: str = ""
):
    """
    Plot a single slice for three views (motion corrupted, prediction, ground truth)

    Args:
        axes: matplotlib axes array (1, 3) for the three subplots
        motion_corrupted: motion corrupted volume (Z, Y, X)
        prediction: predicted volume (Z, Y, X)
        motion_free: ground truth volume (Z, Y, X)
        slice_idx: index of the slice to plot
        metadata: patch metadata
        slice_name: name to add to titles (e.g., "Middle", "Above", "Below")
    """
    # Motion Corrupted
    axes[0].imshow(motion_corrupted[slice_idx, :, :], cmap='gray')
    axes[0].set_title(f'Motion Corrupted {slice_name}')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(prediction[slice_idx, :, :], cmap='gray')
    axes[1].set_title(f'Prediction {slice_name}')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(motion_free[slice_idx, :, :], cmap='gray')
    axes[2].set_title(f'Ground Truth {slice_name}')
    axes[2].axis('off')

    # Add patch info to the first subplot
    axes[0].set_ylabel(f'Patient {metadata["patient_id"]}\nMotion {metadata["motion_id"]}\nPatch {metadata["patch_idx"]}')


def save_patch_prediction_25d(
    model: BrownianBridgeModel,
    dataset: PatchedTripletsDataset_25D,
    epoch: int,
    save_dir: str,
    device: torch.device,
    use_wandb: bool = False,
    selected_patches_indices: List[List[int]] = [],
    decoder: Optional[nn.Module] = None,
    show_progress: bool = True,
):
    """Save predictions for random calcium-sampled patches from validation data"""

    # selected_patches_indices is a list of list of indices associated with each slice being chosen
    all_predictions_for_patches = []
    all_ground_truth_for_patches = []
    for i, patch_indices in enumerate(selected_patches_indices):
        center_slice_predictions = []
        center_slice_ground_truth = []
        if show_progress:
            print(f"Processing patch {i+1} of {len(selected_patches_indices)}")
            print(f"Patch indices: {patch_indices}")

        for slice_idx, dataset_idx in enumerate(patch_indices):
            if isinstance(dataset_idx, list):
                if len(dataset_idx) == 1:
                    dataset_idx = dataset_idx[0] 
                else:
                    raise ValueError(f"Expected single integer in list, got {dataset_idx}")

            if show_progress:
                print(f"  Processing slice {slice_idx+1} of {len(patch_indices)} (dataset index: {dataset_idx})")


            motion_free, motion_corrupted, _ = dataset[dataset_idx]

            # motion_corrupted is the model input (to denoise); motion_free is the target
            motion_corrupted = motion_corrupted.unsqueeze(0).to(device, dtype=torch.float32)
            motion_free = motion_free.unsqueeze(0).to(device, dtype=torch.float32) 


            with torch.no_grad():
                prediction = model.sample(motion_corrupted, show_progress=show_progress)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


            # Unwrap prediction if it comes back as a tuple/list
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]

            if not isinstance(prediction, torch.Tensor):
                raise ValueError(f"Expected tensor prediction, got {type(prediction)}")

            prediction_np = prediction.squeeze().cpu().numpy()

            # Extract center slice from the triplet prediction
            if prediction_np.ndim == 3:  # (3, H, W) triplet
                center_slice = prediction_np[1, :, :] 
            else:
                center_slice = prediction_np

            center_slice_predictions.append(center_slice)

            # Extract center slice from ground truth motion_free
            motion_free_np = motion_free.squeeze().cpu().numpy()
            if motion_free_np.ndim == 3:  # (3, H, W) triplet
                center_slice_gt = motion_free_np[1, :, :]  
            else:
                center_slice_gt = motion_free_np 

            center_slice_ground_truth.append(center_slice_gt)

        # stack all center slice predictions for this patch into a single array
        # Shape (num_slices, height, width) for the selected slices
        predictions_for_patch = np.stack(center_slice_predictions, axis=0)
        ground_truth_for_patch = np.stack(center_slice_ground_truth, axis=0)
        all_predictions_for_patches.append(predictions_for_patch)
        all_ground_truth_for_patches.append(ground_truth_for_patch)

        # save the predicted patches as nifti files
        os.makedirs(os.path.join(save_dir, f'epoch_{epoch+1}'), exist_ok=True)
        nib.save(nib.Nifti1Image(predictions_for_patch, np.eye(4)),
                os.path.join(save_dir, f'epoch_{epoch+1}', f'prediction_patch_{i}.nii.gz'))
        nib.save(nib.Nifti1Image(ground_truth_for_patch, np.eye(4)),
                os.path.join(save_dir, f'epoch_{epoch+1}', f'ground_truth_patch_{i}.nii.gz'))

    plot_central_slice_comparison(all_predictions_for_patches, all_ground_truth_for_patches, save_dir, epoch, use_wandb=use_wandb)
