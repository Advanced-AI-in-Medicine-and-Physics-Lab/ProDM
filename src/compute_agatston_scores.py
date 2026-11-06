#!/usr/bin/env python3
"""
Agatston Score Computation and Evaluation Script

This script loads the evaluation results from H5 files and computes Agatston scores
for motion_corrupted, motion_free, and predicted patches. It evaluates:
1. Quantitative improvement of predictions vs motion_corrupted
2. Severity grading accuracy compared to ground truth
"""

import argparse
import torch
import os
import sys
import json
import numpy as np
import pandas as pd
import h5py
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import ndimage
from scipy.stats import pearsonr
from skimage import measure
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_severity_grade(agatston_score: float) -> str:
    """
    Get severity grade based on Agatston score.
    
    Args:
        agatston_score: Agatston score value
        
    Returns:
        Severity grade string
    """
    agatston_score = round(agatston_score)
    if agatston_score < 1:
        return "no_evidence"
    elif 1 <= agatston_score < 11:
        return "minimal"
    elif 11 <= agatston_score < 101:
        return "mild"
    elif 101 <= agatston_score < 401:
        return "moderate"
    else:
        return "severe"
        

def get_severity_grade_numeric(agatston_score: float) -> int:
    """
    Get numeric severity grade (0-4) based on Agatston score.
    
    Args:
        agatston_score: Agatston score value
        
    Returns:
        Numeric severity grade (0=no_evidence, 1=minimal, 2=mild, 3=moderate, 4=severe)
    """
    agatston_score = round(agatston_score)
    if agatston_score < 1:
        return 0
    elif 1 <= agatston_score < 11:
        return 1
    elif 11 <= agatston_score < 101:
        return 2
    elif 101 <= agatston_score < 401:
        return 3
    else:
        return 4

def compute_dice_loss_and_iou(patch1: np.ndarray, patch2: np.ndarray, threshold: float = 130.0) -> Tuple[float, float]:
    """
    Compute Dice loss and IoU for segmentation based on intensity threshold.
    
    Args:
        patch1: First 3D patch
        patch2: Second 3D patch
        threshold: Intensity threshold for segmentation
        
    Returns:
        Tuple of (dice_loss, iou)
    """
    # Create binary masks based on threshold
    mask1 = (patch1 >= threshold).astype(np.float32)
    mask2 = (patch2 >= threshold).astype(np.float32)
    
    # Compute Dice coefficient
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    if union > 0:
        dice = 2.0 * intersection / union
    else:
        dice = 1.0 if intersection == 0 else 0.0
    
    # Compute IoU
    if union > 0:
        iou = intersection / (union - intersection)
    else:
        iou = 1.0 if intersection == 0 else 0.0

    dice_loss = 1.0 - dice
    
    return dice_loss, iou


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
    
    total_agatston_score = 0.0
    total_lesions = 0

    for slice_idx in range(patch_data.shape[0]):
        slice_data = patch_data[slice_idx, :, :]
        
        calcium_mask = slice_data >= calcium_threshold
        
        if not np.any(calcium_mask):
            continue 
        
        # Label connected components (lesions)
        labeled_lesions, num_lesions = ndimage.label(calcium_mask, structure=ndimage.generate_binary_structure(2, 2))
        
        if debug and num_lesions > 0:
            logger.info(f"Slice {slice_idx}: Found {num_lesions} lesions")
        
        for lesion_id in range(1, num_lesions + 1):
            lesion_mask = (labeled_lesions == lesion_id)
            num_calcium_pixels = np.sum(lesion_mask)
            lesion_area = num_calcium_pixels * pixel_area * voxel_size[2]
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

def extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from filename.
    
    Args:
        filename: H5 filename (e.g., 'patient_107_motion_10_eval_results.h5')
        
    Returns:
        Patient ID string
    """
    # Extract patient ID from filename
    if 'patient_' in filename:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part == 'patient' and i + 1 < len(parts):
                return f"patient_{parts[i+1]}"
    return "unknown"

def extract_motion_id(filename: str) -> str:
    """
    Extract motion ID from filename.
    
    Args:
        filename: H5 filename (e.g., 'patient_107_motion_10_eval_results.h5')
        
    Returns:
        Motion ID string

    """
    # Extract motion ID from filename
    if 'motion_' in filename:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part == 'motion' and i + 1 < len(parts):
                return int(parts[i+1])
    return "unknown"

def get_voxel_size_for_patient(patient_id: str, voxel_size_map: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[float, float, float]:
    """
    Get voxel size for a specific patient.
    
    Args:
        patient_id: Patient identifier
        voxel_size_map: Dictionary mapping patient_id to voxel_size
        
    Returns:
        Voxel size tuple (x, y, z) in mm
    """
    if voxel_size_map and patient_id in voxel_size_map:
        return voxel_size_map[patient_id]
    return (0.625, 0.625, 2.5)


def load_and_compute_scores_simple(h5_file_path: str, voxel_size: Tuple[float, float, float] = (0.625, 0.625, 2.5), 
                                 hu_min: float = -200, hu_max: float = 800, 
                                 voxel_size_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
                                 debug: bool = False) -> List[Dict]:
    """
    Load H5 file with simple structure (direct arrays) and compute Agatston scores for all patches.

    Args:
        h5_file_path: Path to H5 file
        voxel_size: Voxel size in mm
        hu_min: Minimum HU value for denormalization
        hu_max: Maximum HU value for denormalization
        voxel_size_map: Dictionary mapping patient_id to voxel_size
        debug: Whether to print debug information
        
    Returns:
        List of dictionaries with scores and metadata
    """
    results = []
    
    with h5py.File(h5_file_path, 'r') as f:
        # Get file metadata
        original_file = f.attrs.get('original_file', 'unknown')
        num_patches = f.attrs.get('num_patches', 0)
        logger.info(f"Processing {h5_file_path}: {num_patches} total patches")

        # Extract patient ID and get patient-specific voxel size
        patient_id = extract_patient_id(os.path.basename(h5_file_path))
        patient_voxel_size = get_voxel_size_for_patient(patient_id, voxel_size_map)
        logger.info(f"Patient ID: {patient_id}, Voxel size: {patient_voxel_size}")
        
        # Load all data at once
        motion_corrupted_all = np.array(f['motion_corrupted'][()])
        motion_free_all = np.array(f['motion_free'][()])
        predictions_all = np.array(f['predictions'][()])
        
        # Check if data is denormalized (HU values)
        is_denormalized = f.attrs.get('denormalized', False)

        # Process every patch in the file
        for patch_idx in range(num_patches):
            # Process the patch
            result = _process_single_patch_simple(
                motion_corrupted_all[patch_idx],
                motion_free_all[patch_idx],
                predictions_all[patch_idx],
                patch_idx,
                h5_file_path,
                original_file,
                patient_id,
                patient_voxel_size,
                is_denormalized,
                hu_min,
                hu_max,
                debug
            )
            
            if result is not None:
                results.append(result)
    
    return results

def _process_single_patch_simple(motion_corrupted: np.ndarray, motion_free: np.ndarray, predictions: np.ndarray,
                                  patch_idx: int, h5_file_path: str, original_file: str, patient_id: str,
                                  patient_voxel_size: Tuple[float, float, float], is_denormalized: bool,
                                  hu_min: float, hu_max: float, debug: bool) -> Optional[Dict]:
    """
    Process a single patch and compute Agatston scores.
    
    Returns:
        Dictionary with results or None if processing fails
    """
    try:
        # Extract and transform patch data
        motion_corrupted = torch.from_numpy(motion_corrupted).permute(2, 0, 1).cpu().numpy()
        motion_free = torch.from_numpy(motion_free).permute(2, 0, 1).cpu().numpy()
        predictions = torch.from_numpy(predictions).permute(2, 0, 1).cpu().numpy()
        assert(motion_corrupted.shape[-1]==motion_corrupted.shape[-2])
        assert(motion_free.shape[-1]==motion_free.shape[-2])
        assert(predictions.shape[-1]==predictions.shape[-2])
      
        # Denormalize data if it's not already denormalized
        if not is_denormalized:
            logger.info(f"Denormalizing data in {h5_file_path} patch {patch_idx}")
            motion_corrupted = denormalize_to_hu(motion_corrupted, hu_min, hu_max)
            motion_free = denormalize_to_hu(motion_free, hu_min, hu_max)
            predictions = denormalize_to_hu(predictions, hu_min, hu_max)
        else:
            logger.info(f"Data in {h5_file_path} patch {patch_idx} is already denormalized")
        
        # Compute Agatston scores
        score_corrupted = compute_agatston_score(motion_corrupted, patient_voxel_size, debug=debug)
        score_free = compute_agatston_score(motion_free, patient_voxel_size, debug=debug)
        score_predicted = compute_agatston_score(predictions, patient_voxel_size, debug=debug)
        
        # Get severity grades
        grade_corrupted = get_severity_grade(score_corrupted)
        grade_free = get_severity_grade(score_free)
        grade_predicted = get_severity_grade(score_predicted)
        
        grade_corrupted_num = get_severity_grade_numeric(score_corrupted)
        grade_free_num = get_severity_grade_numeric(score_free)
        grade_predicted_num = get_severity_grade_numeric(score_predicted)
        
        # Calculate improvements
        improvement_abs = abs(score_corrupted - score_free) - abs(score_predicted - score_free)
        improvement_rel = (abs(score_corrupted - score_free) - abs(score_predicted - score_free)) / max(abs(score_corrupted - score_free), 1e-6) * 100
        
        # Calculate grade improvement
        grade_improvement = (grade_corrupted_num - grade_free_num) - (grade_predicted_num - grade_free_num)
        
        # Calculate new metric: successful correction of motion corruption
        # True when: motion_corrupted_grade != motion_free_grade AND motion_free_grade == predicted_grade
        successful_correction = (grade_corrupted_num != grade_free_num) and (grade_free_num == grade_predicted_num)
        
        # Calculate Dice loss and IoU for motion corrupted vs motion free
        dice_loss_corrupted, iou_corrupted = compute_dice_loss_and_iou(motion_corrupted, motion_free)
        
        # Calculate Dice loss and IoU for predicted vs motion free
        dice_loss_predicted, iou_predicted = compute_dice_loss_and_iou(predictions, motion_free)
        
        # Calculate absolute errors
        abs_error_corrupted_vs_free = abs(score_corrupted - score_free)
        abs_error_predicted_vs_free = abs(score_predicted - score_free)
        
        result = {
            'file_name': os.path.basename(h5_file_path),
            'original_file': original_file,
            'patient_id': patient_id,
            'voxel_size': patient_voxel_size,
            'patch_idx': patch_idx,
            'vol_idx': -1,  # Not available in simple format
            'slice_indices': [],  # Not available in simple format
            
            # Agatston scores
            'score_motion_corrupted': score_corrupted,
            'score_motion_free': score_free,
            'score_predicted': score_predicted,
            
            # Severity grades (text)
            'grade_motion_corrupted': grade_corrupted,
            'grade_motion_free': grade_free,
            'grade_predicted': grade_predicted,
            
            # Severity grades (numeric)
            'grade_motion_corrupted_num': grade_corrupted_num,
            'grade_motion_free_num': grade_free_num,
            'grade_predicted_num': grade_predicted_num,
            
            # Improvements
            'improvement_absolute': improvement_abs,
            'improvement_relative_percent': improvement_rel,
            'grade_improvement': grade_improvement,
            'successful_correction': successful_correction,
            
            # Dice loss and IoU metrics
            'dice_loss_corrupted_vs_free': dice_loss_corrupted,
            'iou_corrupted_vs_free': iou_corrupted,
            'dice_loss_predicted_vs_free': dice_loss_predicted,
            'iou_predicted_vs_free': iou_predicted,
            
            # Absolute errors
            'abs_error_corrupted_vs_free': abs_error_corrupted_vs_free,
            'abs_error_predicted_vs_free': abs_error_predicted_vs_free,
            
            # Data quality
            'is_denormalized': is_denormalized
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error computing scores for {h5_file_path} patch {patch_idx}: {e}")
        return None

def calculate_pearson_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Pearson correlations for calcium scores.
    
    Args:
        df: DataFrame with all results
        
    Returns:
        Dictionary with correlation values
    """
    correlations = {}
    
    # Calculate Pearson correlation between motion_corrupted and motion_free
    if len(df) > 1:
        corr_corrupted_free, p_val_corrupted_free = pearsonr(df['score_motion_corrupted'], df['score_motion_free'])
        correlations['pearson_corrupted_vs_free'] = corr_corrupted_free
        correlations['pearson_corrupted_vs_free_pvalue'] = p_val_corrupted_free
    else:
        correlations['pearson_corrupted_vs_free'] = 0.0
        correlations['pearson_corrupted_vs_free_pvalue'] = 1.0
    
    # Calculate Pearson correlation between predicted and motion_free
    if len(df) > 1:
        corr_predicted_free, p_val_predicted_free = pearsonr(df['score_predicted'], df['score_motion_free'])
        correlations['pearson_predicted_vs_free'] = corr_predicted_free
        correlations['pearson_predicted_vs_free_pvalue'] = p_val_predicted_free
    else:
        correlations['pearson_predicted_vs_free'] = 0.0
        correlations['pearson_predicted_vs_free_pvalue'] = 1.0
    
    return correlations

def compute_per_grade_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute per-severity grade accuracy, recall, precision, and F1 scores.
    
    Args:
        df: DataFrame with all results
        
    Returns:
        Dictionary with per-grade metrics for both motion corrupted and predicted
    """
    # Define severity grade labels
    severity_labels = ['0', '1-10', '11-100', '101-400', '>400']
    
    # Compute metrics for motion corrupted vs motion free
    precision_corrupted, recall_corrupted, f1_corrupted, support_corrupted = precision_recall_fscore_support(
        df['grade_motion_free_num'], df['grade_motion_corrupted_num'], 
        labels=[0, 1, 2, 3, 4], average=None, zero_division=0
    )
    
    # Compute metrics for predicted vs motion free
    precision_predicted, recall_predicted, f1_predicted, support_predicted = precision_recall_fscore_support(
        df['grade_motion_free_num'], df['grade_predicted_num'], 
        labels=[0, 1, 2, 3, 4], average=None, zero_division=0
    )
    
    # Create detailed per-grade metrics
    per_grade_metrics = {}
    
    for i, label in enumerate(severity_labels):
        per_grade_metrics[f'grade_{label}'] = {
            'motion_corrupted': {
                'precision': precision_corrupted[i],
                'recall': recall_corrupted[i],
                'f1': f1_corrupted[i],
                'support': support_corrupted[i]
            },
            'predicted': {
                'precision': precision_predicted[i],
                'recall': recall_predicted[i],
                'f1': f1_predicted[i],
                'support': support_predicted[i]
            }
        }
    
    # Compute overall macro averages
    macro_precision_corrupted, macro_recall_corrupted, macro_f1_corrupted, _ = precision_recall_fscore_support(
        df['grade_motion_free_num'], df['grade_motion_corrupted_num'], 
        average='macro', zero_division=0
    )
    
    macro_precision_predicted, macro_recall_predicted, macro_f1_predicted, _ = precision_recall_fscore_support(
        df['grade_motion_free_num'], df['grade_predicted_num'], 
        average='macro', zero_division=0
    )
    
    # Compute overall weighted averages
    weighted_precision_corrupted, weighted_recall_corrupted, weighted_f1_corrupted, _ = precision_recall_fscore_support(
        df['grade_motion_free_num'], df['grade_motion_corrupted_num'], 
        average='weighted', zero_division=0
    )
    
    weighted_precision_predicted, weighted_recall_predicted, weighted_f1_predicted, _ = precision_recall_fscore_support(
        df['grade_motion_free_num'], df['grade_predicted_num'], 
        average='weighted', zero_division=0
    )
    
    per_grade_metrics['overall'] = {
        'motion_corrupted': {
            'macro_precision': macro_precision_corrupted,
            'macro_recall': macro_recall_corrupted,
            'macro_f1': macro_f1_corrupted,
            'weighted_precision': weighted_precision_corrupted,
            'weighted_recall': weighted_recall_corrupted,
            'weighted_f1': weighted_f1_corrupted
        },
        'predicted': {
            'macro_precision': macro_precision_predicted,
            'macro_recall': macro_recall_predicted,
            'macro_f1': macro_f1_predicted,
            'weighted_precision': weighted_precision_predicted,
            'weighted_recall': weighted_recall_predicted,
            'weighted_f1': weighted_f1_predicted
        }
    }
    
    return per_grade_metrics

def create_evaluation_summary(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive evaluation summary and visualizations.
    
    Args:
        df: DataFrame with all results
        output_dir: Output directory for plots
    """
    logger.info("Creating evaluation summary...")
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Overall statistics
    total_patches = len(df)
    improved_patches = len(df[df['improvement_absolute'] > 0])
    worsened_patches = len(df[df['improvement_absolute'] < 0])
    no_change_patches = len(df[df['improvement_absolute'] == 0])
    
    # 2. Grade accuracy analysis
    grade_correct_corrupted = len(df[df['grade_motion_corrupted_num'] == df['grade_motion_free_num']])
    grade_correct_predicted = len(df[df['grade_predicted_num'] == df['grade_motion_free_num']])
    
    successful_corrections = len(df[df['successful_correction'] == True])
    motion_corrupted_patches = len(df[df['grade_motion_corrupted_num'] != df['grade_motion_free_num']])
    successful_correction_rate = (successful_corrections / motion_corrupted_patches * 100) if motion_corrupted_patches > 0 else 0
 
    correlations = calculate_pearson_correlations(df)
    per_grade_metrics = compute_per_grade_metrics(df)
    
    summary_stats = {
        'Total Patches': total_patches,
        'Improved Patches': improved_patches,
        'Worsened Patches': worsened_patches,
        'No Change Patches': no_change_patches,
        'Improvement Rate (%)': (improved_patches / total_patches * 100) if total_patches > 0 else 0,
        'Grade Accuracy - Motion Corrupted (%)': (grade_correct_corrupted / total_patches * 100) if total_patches > 0 else 0,
        'Grade Accuracy - Predicted (%)': (grade_correct_predicted / total_patches * 100) if total_patches > 0 else 0,
        'Motion Corrupted Patches': motion_corrupted_patches,
        'Successful Corrections': successful_corrections,
        'Successful Correction Rate (%)': successful_correction_rate,
        'Mean Absolute Improvement': df['improvement_absolute'].mean(),
        'Mean Relative Improvement (%)': df['improvement_relative_percent'].mean(),
        'Mean Grade Improvement': df['grade_improvement'].mean(),
        
        # Mean absolute errors with standard errors
        'Mean Absolute Error - Corrupted vs Free': df['abs_error_corrupted_vs_free'].mean(),
        'Mean Absolute Error - Predicted vs Free': df['abs_error_predicted_vs_free'].mean(),
        'SE Absolute Error - Corrupted vs Free': df['abs_error_corrupted_vs_free'].std() / np.sqrt(len(df)),
        'SE Absolute Error - Predicted vs Free': df['abs_error_predicted_vs_free'].std() / np.sqrt(len(df)),
        
        # Pearson correlations
        'Pearson Correlation - Corrupted vs Free': correlations['pearson_corrupted_vs_free'],
        'Pearson Correlation - Predicted vs Free': correlations['pearson_predicted_vs_free'],
        'Pearson Correlation P-value - Corrupted vs Free': correlations['pearson_corrupted_vs_free_pvalue'],
        'Pearson Correlation P-value - Predicted vs Free': correlations['pearson_predicted_vs_free_pvalue'],
        
        # Dice loss and IoU metrics with standard errors
        'Mean Dice Loss - Corrupted vs Free': df['dice_loss_corrupted_vs_free'].mean(),
        'Mean IoU - Corrupted vs Free': df['iou_corrupted_vs_free'].mean(),
        'Mean Dice Loss - Predicted vs Free': df['dice_loss_predicted_vs_free'].mean(),
        'Mean IoU - Predicted vs Free': df['iou_predicted_vs_free'].mean(),
        'SE Dice Loss - Corrupted vs Free': df['dice_loss_corrupted_vs_free'].std() / np.sqrt(len(df)),
        'SE IoU - Corrupted vs Free': df['iou_corrupted_vs_free'].std() / np.sqrt(len(df)),
        'SE Dice Loss - Predicted vs Free': df['dice_loss_predicted_vs_free'].std() / np.sqrt(len(df)),
        'SE IoU - Predicted vs Free': df['iou_predicted_vs_free'].std() / np.sqrt(len(df))
    }
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Agatston Score Evaluation Summary\n")
        f.write("==================================\n\n")
        
        for key, value in summary_stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.3f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write(f"\nDetailed Statistics:\n")
        f.write(f"==================\n")
        f.write(f"Score Statistics:\n")
        f.write(f"  Motion Corrupted - Mean: {df['score_motion_corrupted'].mean():.3f}, Std: {df['score_motion_corrupted'].std()/np.sqrt(len(df)):.3f}\n")
        f.write(f"  Motion Free - Mean: {df['score_motion_free'].mean():.3f}, Std: {df['score_motion_free'].std()/np.sqrt(len(df)):.3f}\n")
        f.write(f"  Predicted - Mean: {df['score_predicted'].mean():.3f}, Std: {df['score_predicted'].std()/np.sqrt(len(df)):.3f}\n")
        
        f.write(f"\nAbsolute Error Statistics:\n")
        f.write(f"  Mean Absolute Error - Corrupted vs Free: {df['abs_error_corrupted_vs_free'].mean():.3f} ± {df['abs_error_corrupted_vs_free'].std()/np.sqrt(len(df)):.3f}\n")
        f.write(f"  Mean Absolute Error - Predicted vs Free: {df['abs_error_predicted_vs_free'].mean():.3f} ± {df['abs_error_predicted_vs_free'].std()/np.sqrt(len(df)):.3f}\n")
        
        f.write(f"\nPearson Correlation Statistics:\n")
        f.write(f"  Correlation - Corrupted vs Free: {correlations['pearson_corrupted_vs_free']:.3f} (p={correlations['pearson_corrupted_vs_free_pvalue']:.3f})\n")
        f.write(f"  Correlation - Predicted vs Free: {correlations['pearson_predicted_vs_free']:.3f} (p={correlations['pearson_predicted_vs_free_pvalue']:.3f})\n")
        
        f.write(f"\nDice Loss and IoU Statistics:\n")
        f.write(f"  Dice Loss - Corrupted vs Free: {df['dice_loss_corrupted_vs_free'].mean():.3f} ± {df['dice_loss_corrupted_vs_free'].std()/np.sqrt(len(df)):.3f}\n")
        f.write(f"  IoU - Corrupted vs Free: {df['iou_corrupted_vs_free'].mean():.3f} ± {df['iou_corrupted_vs_free'].std()/np.sqrt(len(df)):.3f}\n")
        f.write(f"  Dice Loss - Predicted vs Free: {df['dice_loss_predicted_vs_free'].mean():.3f} ± {df['dice_loss_predicted_vs_free'].std()/np.sqrt(len(df)):.3f}\n")
        f.write(f"  IoU - Predicted vs Free: {df['iou_predicted_vs_free'].mean():.3f} ± {df['iou_predicted_vs_free'].std()/np.sqrt(len(df)):.3f}\n")
        
        # Calculate grade accuracy statistics with uncertainty
        grade_correct_corrupted = len(df[df['grade_motion_corrupted_num'] == df['grade_motion_free_num']])
        grade_correct_predicted = len(df[df['grade_predicted_num'] == df['grade_motion_free_num']])
        
        # Calculate proportions and standard errors for grade accuracy
        n = len(df)
        p_corrupted = grade_correct_corrupted / n
        p_predicted = grade_correct_predicted / n
        se_grade_corrupted = np.sqrt(p_corrupted * (1 - p_corrupted) / n)
        se_grade_predicted = np.sqrt(p_predicted * (1 - p_predicted) / n)
        
        f.write(f"\nGrade Accuracy Statistics:\n")
        f.write(f"  Grade Accuracy - Motion Corrupted: {p_corrupted*100:.3f}% ± {se_grade_corrupted*100:.3f}%\n")
        f.write(f"  Grade Accuracy - Predicted: {p_predicted*100:.3f}% ± {se_grade_predicted*100:.3f}%\n")
        
        # Write per-grade metrics
        f.write(f"\nPer-Severity Grade Metrics:\n")
        f.write(f"================================\n")
        
        severity_labels = ['0', '1-10', '11-100', '101-400', '>400']
        
        for label in severity_labels:
            grade_key = f'grade_{label}'
            if grade_key in per_grade_metrics:
                metrics = per_grade_metrics[grade_key]
                f.write(f"\nGrade {label}:\n")
                f.write(f"  Motion Corrupted - Precision: {metrics['motion_corrupted']['precision']:.3f}, Recall: {metrics['motion_corrupted']['recall']:.3f}, F1: {metrics['motion_corrupted']['f1']:.3f}, Support: {metrics['motion_corrupted']['support']}\n")
                f.write(f"  Predicted - Precision: {metrics['predicted']['precision']:.3f}, Recall: {metrics['predicted']['recall']:.3f}, F1: {metrics['predicted']['f1']:.3f}, Support: {metrics['predicted']['support']}\n")
        
        # Write overall metrics
        f.write(f"\nOverall Classification Metrics:\n")
        f.write(f"  Motion Corrupted - Macro F1: {per_grade_metrics['overall']['motion_corrupted']['macro_f1']:.3f}, Weighted F1: {per_grade_metrics['overall']['motion_corrupted']['weighted_f1']:.3f}\n")
        f.write(f"  Predicted - Macro F1: {per_grade_metrics['overall']['predicted']['macro_f1']:.3f}, Weighted F1: {per_grade_metrics['overall']['predicted']['weighted_f1']:.3f}\n")
    
    # 4. Create visualizations
    create_plots(df, plots_dir)
    
    logger.info(f"Evaluation summary saved to {summary_file}")
    logger.info(f"Plots saved to {plots_dir}")

def create_plots(df: pd.DataFrame, plots_dir: str) -> None:
    """
    Create evaluation plots.
    
    Args:
        df: DataFrame with results
        plots_dir: Directory to save plots
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Score comparison scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Motion Corrupted vs Motion Free
    axes[0].scatter(df['score_motion_free'], df['score_motion_corrupted'], alpha=0.6)
    axes[0].plot([0, df['score_motion_free'].max()], [0, df['score_motion_free'].max()], 'r--', alpha=0.8)
    axes[0].set_xlabel('Motion Free Score')
    axes[0].set_ylabel('Motion Corrupted Score')
    axes[0].set_title('Motion Corrupted vs Motion Free')
    axes[0].grid(True, alpha=0.3)
    
    # Predicted vs Motion Free
    axes[1].scatter(df['score_motion_free'], df['score_predicted'], alpha=0.6)
    axes[1].plot([0, df['score_motion_free'].max()], [0, df['score_motion_free'].max()], 'r--', alpha=0.8)
    axes[1].set_xlabel('Motion Free Score')
    axes[1].set_ylabel('Predicted Score')
    axes[1].set_title('Predicted vs Motion Free')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Improvement distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute improvement
    axes[0].hist(df['improvement_absolute'], bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[0].set_xlabel('Absolute Improvement')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Absolute Improvements')
    axes[0].grid(True, alpha=0.3)
    
    # Relative improvement
    axes[1].hist(df['improvement_relative_percent'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[1].set_xlabel('Relative Improvement (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Relative Improvements')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'improvement_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Grade accuracy comparison
    grade_accuracy = pd.DataFrame({
        'Method': ['Motion Corrupted', 'Predicted'],
        'Accuracy (%)': [
            len(df[df['grade_motion_corrupted_num'] == df['grade_motion_free_num']]) / len(df) * 100,
            len(df[df['grade_predicted_num'] == df['grade_motion_free_num']]) / len(df) * 100
        ]
    })
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(grade_accuracy['Method'], grade_accuracy['Accuracy (%)'])
    plt.ylabel('Grade Accuracy (%)')
    plt.title('Severity Grade Accuracy Comparison')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, grade_accuracy['Accuracy (%)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'grade_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Bland-Altman plot for predicted vs motion free
    plt.figure(figsize=(10, 8))
    
    mean_scores = (df['score_predicted'] + df['score_motion_free']) / 2
    diff_scores = df['score_predicted'] - df['score_motion_free']
    
    plt.scatter(mean_scores, diff_scores, alpha=0.6)
    plt.axhline(diff_scores.mean(), color='red', linestyle='--', alpha=0.8, label=f'Mean: {diff_scores.mean():.2f}')
    plt.axhline(diff_scores.mean() + 1.96*diff_scores.std(), color='red', linestyle=':', alpha=0.8, label=f'+1.96 SD: {diff_scores.mean() + 1.96*diff_scores.std():.2f}')
    plt.axhline(diff_scores.mean() - 1.96*diff_scores.std(), color='red', linestyle=':', alpha=0.8, label=f'-1.96 SD: {diff_scores.mean() - 1.96*diff_scores.std():.2f}')
    
    plt.xlabel('Mean Score (Predicted + Motion Free) / 2')
    plt.ylabel('Difference (Predicted - Motion Free)')
    plt.title('Bland-Altman Plot: Predicted vs Motion Free')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bland_altman.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Successful correction analysis
    motion_corrupted_patches = len(df[df['grade_motion_corrupted_num'] != df['grade_motion_free_num']])
    successful_corrections = len(df[df['successful_correction'] == True])
    failed_corrections = motion_corrupted_patches - successful_corrections
    
    if motion_corrupted_patches > 0:
        correction_data = pd.DataFrame({
            'Outcome': ['Successful Correction', 'Failed Correction'],
            'Count': [successful_corrections, failed_corrections],
            'Percentage': [
                (successful_corrections / motion_corrupted_patches) * 100,
                (failed_corrections / motion_corrupted_patches) * 100
            ]
        })
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(correction_data['Outcome'], correction_data['Count'], 
                      color=['green', 'red'], alpha=0.7)
        plt.ylabel('Number of Patches')
        plt.title(f'Motion Corruption Correction Analysis\n(Total Motion Corrupted Patches: {motion_corrupted_patches})')
        
        # Add value labels on bars
        for bar, count, percentage in zip(bars, correction_data['Count'], correction_data['Percentage']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{count} ({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'successful_correction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Correlation analysis plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Motion Corrupted vs Motion Free correlation
    axes[0].scatter(df['score_motion_free'], df['score_motion_corrupted'], alpha=0.6, s=20)
    axes[0].plot([0, df['score_motion_free'].max()], [0, df['score_motion_free'].max()], 'r--', alpha=0.8)
    axes[0].set_xlabel('Motion Free Score')
    axes[0].set_ylabel('Motion Corrupted Score')
    axes[0].set_title(f'Motion Corrupted vs Motion Free\n(r={df["score_motion_corrupted"].corr(df["score_motion_free"]):.3f})')
    axes[0].grid(True, alpha=0.3)
    
    # Predicted vs Motion Free correlation
    axes[1].scatter(df['score_motion_free'], df['score_predicted'], alpha=0.6, s=20)
    axes[1].plot([0, df['score_motion_free'].max()], [0, df['score_motion_free'].max()], 'r--', alpha=0.8)
    axes[1].set_xlabel('Motion Free Score')
    axes[1].set_ylabel('Predicted Score')
    axes[1].set_title(f'Predicted vs Motion Free\n(r={df["score_predicted"].corr(df["score_motion_free"]):.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Absolute error comparison
    plt.figure(figsize=(12, 6))
    
    # Create box plot for absolute errors
    error_data = pd.DataFrame({
        'Error Type': ['Corrupted vs Free'] * len(df) + ['Predicted vs Free'] * len(df),
        'Absolute Error': list(df['abs_error_corrupted_vs_free']) + list(df['abs_error_predicted_vs_free'])
    })
    
    sns.boxplot(data=error_data, x='Error Type', y='Absolute Error')
    plt.title('Absolute Error Comparison')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'absolute_error_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Confusion matrix for severity grades
    # Define severity grade labels
    severity_labels = ['0', '1-10', '11-100', '101-400', '>400']
    
    # Motion Corrupted vs Motion Free confusion matrix
    cm_corrupted = confusion_matrix(df['grade_motion_free_num'], df['grade_motion_corrupted_num'], labels=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_corrupted, annot=True, fmt='d', cmap='Greens', 
                xticklabels=severity_labels, yticklabels=severity_labels)
    plt.title('Motion Corrupted vs Motion Free\nSeverity Grade Confusion Matrix')
    plt.xlabel('Motion Corrupted Grade')
    plt.ylabel('Motion Free Grade (Ground Truth)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'motion_corrupted_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Predicted vs Motion Free confusion matrix
    cm_predicted = confusion_matrix(df['grade_motion_free_num'], df['grade_predicted_num'], labels=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_predicted, annot=True, fmt='d', cmap='Greens',
                xticklabels=severity_labels, yticklabels=severity_labels)
    plt.title('Predicted vs Motion Free\nSeverity Grade Confusion Matrix')
    plt.xlabel('Predicted Grade')
    plt.ylabel('Motion Free Grade (Ground Truth)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predicted_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute Agatston scores and evaluate BBDM results")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing H5 evaluation results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results and plots')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[0.625, 0.625, 2.5],
                        help='Voxel size in mm (x, y, z) (default: 0.625 0.625 2.5)')
    parser.add_argument('--hu_min', type=float, default=-200,
                        help='Minimum HU value for denormalization (default: -200)')
    parser.add_argument('--hu_max', type=float, default=800,
                        help='Maximum HU value for denormalization (default: 800)')
    parser.add_argument('--voxel_size_map', type=str, default=None,
                        help='JSON file containing patient_id to voxel_size mapping')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for Agatston score calculation')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip creating plots')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load voxel size map if provided
    voxel_size_map: Optional[Dict[str, Tuple[float, float, float]]] = None
    if args.voxel_size_map:
        try:
            with open(args.voxel_size_map, 'r') as f:
                voxel_size_map = json.load(f)
            logger.info(f"Loaded voxel size map from {args.voxel_size_map}")
        except FileNotFoundError:
            logger.warning(f"Voxel size map file not found at {args.voxel_size_map}. Using default voxel size.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {args.voxel_size_map}. Please check its format.")
            sys.exit(1)
    
    # Find all H5 files
    h5_files = list(Path(args.input_dir).glob('*_results.h5'))
    logger.info(f"Found {len(h5_files)} evaluation result files")
    
    if not h5_files:
        logger.error(f"No evaluation result files found in {args.input_dir}")
        return

    all_results = []
    for h5_file in tqdm(h5_files, desc="Processing H5 files"):
        try:
            results = load_and_compute_scores_simple(str(h5_file), tuple(args.voxel_size), args.hu_min, args.hu_max, voxel_size_map, args.debug)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Error processing {h5_file}: {e}")
            continue
    
    if not all_results:
        logger.error("No results to process!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results to CSV
    csv_file = os.path.join(args.output_dir, "agatston_scores.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Results saved to {csv_file}")
    
    # Create evaluation summary
    if not args.skip_plots:
        create_evaluation_summary(df, args.output_dir)
    
    # Print summary
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"Total patches processed: {len(df)}")
    logger.info(f"Improved patches: {len(df[df['improvement_absolute'] > 0])}")
    logger.info(f"Worsened patches: {len(df[df['improvement_absolute'] < 0])}")
    logger.info(f"Improvement rate: {len(df[df['improvement_absolute'] > 0]) / len(df) * 100:.3f}%")
    
    grade_correct_corrupted = len(df[df['grade_motion_corrupted_num'] == df['grade_motion_free_num']])
    grade_correct_predicted = len(df[df['grade_predicted_num'] == df['grade_motion_free_num']])
    
    logger.info(f"Grade accuracy - Motion corrupted: {grade_correct_corrupted / len(df) * 100:.3f}%")
    logger.info(f"Grade accuracy - Predicted: {grade_correct_predicted / len(df) * 100:.3f}%")
    
    # New metric: successful correction of motion corruption
    motion_corrupted_patches = len(df[df['grade_motion_corrupted_num'] != df['grade_motion_free_num']])
    successful_corrections = len(df[df['successful_correction'] == True])
    successful_correction_rate = (successful_corrections / motion_corrupted_patches * 100) if motion_corrupted_patches > 0 else 0
    
    logger.info(f"Motion corrupted patches: {motion_corrupted_patches}")
    logger.info(f"Successful corrections: {successful_corrections}")
    logger.info(f"Successful correction rate: {successful_correction_rate:.3f}%")

if __name__ == "__main__":
    main() 