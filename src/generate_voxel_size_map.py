#!/usr/bin/env python3
"""
Generate Voxel Size Mapping Script

This script extracts voxel size information from NIFTI files and creates a JSON mapping
from patient_id to voxel_size for use in Agatston score calculations.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import nibabel as nib
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_voxel_size_from_nifti(nifti_path: str) -> Optional[Tuple[float, float, float]]:
    """
    Extract voxel size from a NIFTI file.
    
    Args:
        nifti_path: Path to NIFTI file
        
    Returns:
        Tuple of (x_spacing, y_spacing, z_spacing) in mm, or None if failed
    """
    try:
        # Load NIFTI file
        img = nib.load(nifti_path)
        
        # Get voxel size from header
        voxel_size = img.header.get_zooms()
        
        if len(voxel_size) >= 3:
            # Return first 3 dimensions (x, y, z)
            return (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
        else:
            logger.warning(f"Insufficient voxel size dimensions in {nifti_path}: {voxel_size}")
            return None
        
    except Exception as e:
        logger.error(f"Error reading NIFTI file {nifti_path}: {e}")
        return None

def find_nifti_files(nifti_dir: str) -> Dict[str, str]:
    """
    Find NIFTI files organized by patient ID.
    
    Args:
        nifti_dir: Root directory containing NIFTI files named {patient_id}.nii
        
    Returns:
        Dictionary mapping patient_id to path of NIFTI file
    """
    patient_niftis = {}
    
    # Look for NIFTI files directly in the directory
    nifti_dir_path = Path(nifti_dir)
    
    # Look for files named {patient_id}.nii or {patient_id}.nii.gz
    for nifti_file in nifti_dir_path.glob("*.nii*"):
        # Extract patient ID from filename
        patient_id = extract_patient_id_from_filename(nifti_file.name)
        print(f"Found NIFTI file for {patient_id}: {nifti_file}")
        
        if patient_id:
            patient_niftis[patient_id] = str(nifti_file)
            logger.info(f"Found NIFTI file for {patient_id}: {nifti_file}")
        else:
            logger.warning(f"Could not extract patient ID from filename: {nifti_file.name}")
    
    return patient_niftis

def extract_patient_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract patient ID from filename.
    
    Args:
        filename: Filename (e.g., 'patient_107.nii', 'patient_107.nii.gz', or '237.nii')
        
    Returns:
        Patient ID string or None if not found
    """
    name_without_ext = filename.replace('.nii.gz', '').replace('.nii', '')
    if name_without_ext.startswith('patient_'):
        return name_without_ext
    if name_without_ext.isdigit():
        return f"patient_{name_without_ext}"
    return None

def generate_voxel_size_map_from_niftis(nifti_dir: str, output_file: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Generate voxel size mapping from NIFTI files.
    
    Args:
        nifti_dir: Directory containing patient NIFTI folders
        output_file: Output JSON file path
        
    Returns:
        Dictionary mapping patient_id to voxel_size
    """
    logger.info(f"Scanning NIFTI directory: {nifti_dir}")

    patient_niftis = find_nifti_files(nifti_dir)
    
    if not patient_niftis:
        logger.error("No patient NIFTI files found!")
        return {}
    
    voxel_size_map = {}
    
    for patient_id, nifti_path in tqdm(patient_niftis.items(), desc="Extracting voxel sizes"):
        voxel_size = extract_voxel_size_from_nifti(nifti_path)
        
        if voxel_size:
            voxel_size_map[patient_id] = voxel_size
            logger.info(f"{patient_id}: {voxel_size}")
        else:
            logger.warning(f"Failed to extract voxel size for {patient_id}")
    
    with open(output_file, 'w') as f:
        json.dump(voxel_size_map, f, indent=4)
    
    logger.info(f"Voxel size map saved to: {output_file}")
    logger.info(f"Generated mapping for {len(voxel_size_map)} patients")
    
    return voxel_size_map

def generate_voxel_size_map_from_h5_files(h5_dir: str, output_file: str, 
                                        default_voxel_size: Tuple[float, float, float] = (0.625, 0.625, 2.5)) -> Dict[str, Tuple[float, float, float]]:
    """
    Generate voxel size mapping from H5 filenames (for cases where NIFTI files are not available).
    
    Args:
        h5_dir: Directory containing H5 files
        output_file: Output JSON file path
        default_voxel_size: Default voxel size to use for all patients
        
    Returns:
        Dictionary mapping patient_id to voxel_size
    """
    logger.info(f"Scanning H5 directory: {h5_dir}")
    
    voxel_size_map = {}
    h5_files = list(Path(h5_dir).glob("*.h5"))
    
    for h5_file in h5_files:
        patient_id = extract_patient_id_from_filename(h5_file.name)
        
        if patient_id and patient_id not in voxel_size_map:
            voxel_size_map[patient_id] = default_voxel_size
            logger.info(f"{patient_id}: {default_voxel_size} (default)")
    
    with open(output_file, 'w') as f:
        json.dump(voxel_size_map, f, indent=4)
    
    logger.info(f"Voxel size map saved to: {output_file}")
    logger.info(f"Generated mapping for {len(voxel_size_map)} patients using default voxel size")
    
    return voxel_size_map

def main():
    parser = argparse.ArgumentParser(description="Generate voxel size mapping from NIFTI files or H5 files")
    parser.add_argument('--nifti_dir', type=str, default=None,
                        help='Directory containing patient NIFTI folders')
    parser.add_argument('--h5_dir', type=str, default=None,
                        help='Directory containing H5 files (for default voxel size mapping)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--default_voxel_size', type=float, nargs=3, default=[0.625, 0.625, 2.5],
                        help='Default voxel size in mm (x, y, z) for H5-based mapping')
    
    args = parser.parse_args()
    
    if not args.nifti_dir and not args.h5_dir:
        logger.error("Must specify either --nifti_dir or --h5_dir")
        sys.exit(1)
    
    if args.nifti_dir and args.h5_dir:
        logger.error("Cannot specify both --nifti_dir and --h5_dir")
        sys.exit(1)
    
    if args.nifti_dir:
        if not os.path.exists(args.nifti_dir):
            logger.error(f"NIFTI directory does not exist: {args.nifti_dir}")
            sys.exit(1)
        
        voxel_size_map = generate_voxel_size_map_from_niftis(args.nifti_dir, args.output_file)
        
    else: 
        if not os.path.exists(args.h5_dir):
            logger.error(f"H5 directory does not exist: {args.h5_dir}")
            sys.exit(1)
        
        voxel_size_map = generate_voxel_size_map_from_h5_files(
            args.h5_dir, args.output_file, tuple(args.default_voxel_size)
        )
    
    print(f"\nVoxel Size Mapping Summary:")
    print(f"==========================")
    print(f"Output file: {args.output_file}")
    print(f"Total patients: {len(voxel_size_map)}")
    print(f"\nPatient mappings:")
    for patient_id, voxel_size in voxel_size_map.items():
        print(f"  {patient_id}: {voxel_size}")
    
    if args.nifti_dir:
        print(f"\nNote: Voxel sizes extracted from NIFTI files in {args.nifti_dir}")
    else:
        print(f"\nNote: Using default voxel size {tuple(args.default_voxel_size)} for all patients")

if __name__ == "__main__":
    main() 