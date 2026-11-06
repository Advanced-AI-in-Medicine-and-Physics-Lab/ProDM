#!/usr/bin/env python3
"""
Voxel spacing utilities for CT image processing.
"""

import json

def load_voxel_spacing_map(voxel_size_map_path: str):
    """
    Load voxel spacing map from JSON file.
    
    Args:
        voxel_size_map_path: Path to the JSON file containing voxel spacing data
        
    Returns:
        Dictionary mapping patient IDs to voxel spacing tuples
    """
    try:
        with open(voxel_size_map_path, 'r') as f:
            voxel_map = json.load(f)
        
        # Convert lists to tuples for consistency
        voxel_map = {patient_id: tuple(spacing) for patient_id, spacing in voxel_map.items()}
        
        print(f"Loaded voxel spacing map with {len(voxel_map)} patients")
        return voxel_map
    except FileNotFoundError:
        print(f"Warning: Voxel spacing map not found at {voxel_size_map_path}")
        return {}
    except Exception as e:
        print(f"Error loading voxel spacing map: {e}")
        return {}

def get_spacing_for_patient(voxel_map, patient_id, default_spacing=(0.4, 0.4, 3.0)):
    """
    Get voxel spacing for a specific patient.
    
    Args:
        voxel_map: Dictionary mapping patient IDs to spacing
        patient_id: Patient ID to look up (can be string or list)
        default_spacing: Default spacing if patient not found
        
    Returns:
        Voxel spacing tuple (x, y, z)
    """
    if isinstance(patient_id, list):
        patient_id = patient_id[0] if patient_id else "unknown"

    patient_id = str(patient_id)
    
    return voxel_map.get(patient_id, default_spacing) 