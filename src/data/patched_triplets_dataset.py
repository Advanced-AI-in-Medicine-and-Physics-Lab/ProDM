#!/usr/bin/env python3
import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
import logging
import random
import threading
from contextlib import contextmanager
from collections import defaultdict
import torchvision.transforms as T
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatchedDataset(Dataset):
    """
    Dataset for loading preprocessed center slices from HDF5 files.
    This dataset assumes data has been preprocessed offline and only loads it.
    """
    
    def __init__(
        self,
        h5_dir: str,
        transform: Optional[T.Compose] = None,
        shuffle: bool = True,
        cache_size: int = 1000,  # Number of patches to cache
        prefetch_factor: int = 2  # Number of batches to prefetch
    ):
        """
        Args:
            h5_dir: Directory containing preprocessed HDF5 files
            transform: Optional transform to apply to patches
            shuffle: Whether to shuffle the dataset
            cache_size: Number of patches to keep in memory
            prefetch_factor: Number of batches to prefetch
        """
        self.h5_dir = h5_dir
        self.transform = transform
        self.shuffle = shuffle
        self.cache_size = cache_size
        self.prefetch_factor = prefetch_factor
        
        # Memory management
        self._file_handles = {}
        self._file_locks = {}
        self._lock = threading.Lock()
        self._patch_cache = {}
        self._cache_lock = threading.Lock()

        self.h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
        if not self.h5_files:
            raise ValueError(f"No HDF5 files found in {h5_dir}")
        
        self.metadata_list = []
        self.file_indices = [] 
        
        self._preload_metadata()
        
        if shuffle:
            self._shuffle_indices()
        
        logger.info(f"Loaded {len(self)} preprocessed patches from {len(self.h5_files)} HDF5 files")

    def _preload_metadata(self):
        for file_idx, h5_file in enumerate(self.h5_files):
            file_path = os.path.join(self.h5_dir, h5_file)
            with h5py.File(file_path, 'r') as f:
                if 'metadata' not in f.attrs:
                    continue
                
                metadata: Any = f.attrs['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                num_patches = metadata['num_patches']
                for patch_idx in range(num_patches):
                    patch_info = metadata['patch_info'][patch_idx]
                    metadata_idx = {
                        'patient_id': metadata['patient_id'],
                        'motion_id': metadata['motion_id'],
                        'patch_idx': patch_idx,
                        'patch_center': patch_info['patch_center'],
                        'sampler_type': patch_info['sampler_type'],
                        'file_path': file_path,
                        'file': h5_file,  
                        'vol_idx': patch_idx,  
                        'center_slices': patch_info.get('center_slices', [6, 7, 8, 9, 10, 11]),
                        'preprocessed': metadata.get('preprocessed', False)
                    }
                    self.metadata_list.append(metadata_idx)
                    self.file_indices.append((file_idx, patch_idx))

    def _shuffle_indices(self):
        """Shuffle dataset indices"""
        indices = np.arange(len(self.file_indices))
        np.random.shuffle(indices)
        self.file_indices = [self.file_indices[i] for i in indices]
        self.metadata_list = [self.metadata_list[i] for i in indices]

    @contextmanager
    def _get_file_handle(self, file_path: str):
        """Thread-safe file handle management"""
        try:
            with h5py.File(file_path, 'r') as f:
                yield f
        except Exception as e:
            print(f"Error opening file {file_path}: {e}")
            raise

    def _load_patch(self, file_path: str, patch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a preprocessed patch from file with caching"""
        cache_key = f"{file_path}_{patch_idx}"
        
        # Check cache first
        if cache_key in self._patch_cache:
            return self._patch_cache[cache_key]

        # Load from file if not in cache
        with self._get_file_handle(file_path) as f:
            motion_corrupted = torch.from_numpy(f['motion_corrupted'][patch_idx])
            motion_free = torch.from_numpy(f['motion_free'][patch_idx])

            # Cache the result
            if len(self._patch_cache) < self.cache_size:
                self._patch_cache[cache_key] = (motion_free, motion_corrupted)
            
            return motion_free, motion_corrupted

    def __len__(self) -> int:
        return len(self.file_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a preprocessed patch pair and its metadata
        
        Returns:
            Tuple of (motion_free_patch, motion_corrupted_patch, metadata)
        """
        _, patch_idx = self.file_indices[idx]
        metadata = self.metadata_list[idx]
        file_path = metadata['file_path']
        
        motion_free, motion_corrupted = self._load_patch(file_path, patch_idx)
        
        if self.transform is not None:
            motion_corrupted = self.transform(motion_corrupted)
            motion_free = self.transform(motion_free)
    
        motion_corrupted = motion_corrupted.float()
        motion_free = motion_free.float()
        
        return motion_free, motion_corrupted, metadata

    def __del__(self):
        """Clean up file handles when the dataset is deleted"""
        with self._lock:
            for handle in self._file_handles.values():
                try:
                    handle.close()
                except:
                    pass
            self._file_handles.clear()
            self._file_locks.clear()
            self._patch_cache.clear()

    def get_subject_patches(self, subject_id: str) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Get all patches for a specific subject"""
        subject_patches = []
        for idx, (file_idx, _) in enumerate(self.file_indices):
            h5_file = self.h5_files[file_idx]
            with h5py.File(os.path.join(self.h5_dir, h5_file), 'r') as f:
                metadata: Any = None
                if 'metadata' in f.attrs:
                    metadata = f.attrs['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                
                if metadata and metadata['patient_id'] == subject_id:
                    subject_patches.append(self[idx])
        
        return subject_patches

    def get_calcium_patches(self) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Get all patches containing calcified lesion"""
        calcium_patches = []
        for idx, metadata in enumerate(self.metadata_list):
            if metadata['sampler_type'] == 'calcium':
                calcium_patches.append(self[idx])
        return calcium_patches

    def get_random_patches(self) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Get all patches that were randomly sampled"""
        random_patches = []
        for idx, metadata in enumerate(self.metadata_list):
            if metadata['sampler_type'] == 'random':
                random_patches.append(self[idx])
        return random_patches

    def get_patches_by_motion_id(self, motion_id: int) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Get all patches for a specific motion ID"""
        motion_patches = []
        for idx, metadata in enumerate(self.metadata_list):
            if metadata['motion_id'] == motion_id:
                motion_patches.append(self[idx])
        return motion_patches

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_patches': len(self),
            'num_files': len(self.h5_files),
            'sampler_types': defaultdict(int),
            'motion_ids': defaultdict(int),
            'patient_ids': defaultdict(int)
        }
        
        for metadata in self.metadata_list:
            stats['sampler_types'][metadata['sampler_type']] += 1
            stats['motion_ids'][metadata['motion_id']] += 1
            stats['patient_ids'][metadata['patient_id']] += 1
        
        return dict(stats)


class PatchedTripletsDataset_25D(PatchedDataset):
    """
    Patched 2.5D dataset for loading preprocessed center slices, creating triplets
    formed of adjacent slices.
    """
    
    def __init__(
        self,
        h5_dir: str,
        transform: Optional[T.Compose] = None,
        shuffle: bool = True,
        triplet_depth: int = 3,
        cache_size: int = 1000,
        **kwargs
    ):
        """
        Args:
            h5_dir: Directory containing preprocessed HDF5 files
            transform: Optional transform to apply to patches
            shuffle: Whether to shuffle the dataset
            triplet_depth: Depth of triplets (should be 3)
            cache_size: Number of patches to keep in memory
        """
        self.triplet_depth = triplet_depth
        super().__init__(h5_dir, transform, shuffle, cache_size, **kwargs)
        
        self._create_triplets()
        
        logger.info(f"Created {len(self)} triplets from preprocessed center slices")

    def _create_triplets(self):
        """Create triplets from center slices"""
        new_metadata_list = []
        new_file_indices = []
        
        for idx, (file_idx, patch_idx) in enumerate(self.file_indices):
            metadata = self.metadata_list[idx]
            
            # Check if this is training data (6 center slices) or validation data (16 slices)
            if 'center_slices' in metadata and len(metadata.get('center_slices', [])) == 6:
                center_slices = metadata.get('center_slices', [6, 7, 8, 9, 10, 11])
                
                for triplet_idx in range(len(center_slices) - 2):
                    triplet_metadata = metadata.copy()
                    triplet_metadata['triplet_idx'] = triplet_idx
                    triplet_metadata['slice_indices'] = center_slices[triplet_idx:triplet_idx + 3]
                    
                    new_metadata_list.append(triplet_metadata)
                    new_file_indices.append((file_idx, patch_idx, triplet_idx))
            else:
                # For validation data, we'll create triplets around the center (slice 8)
                center_slice = 8  # Middle of 16 slices
                start_slice = max(0, center_slice - 1)
                end_slice = min(16, center_slice + 2)
                
                # Ensure we have 3 slices
                if end_slice - start_slice < 3:
                    if start_slice == 0:
                        end_slice = 3
                    else:
                        start_slice = end_slice - 3
                
                # Create a single triplet from the center region
                triplet_metadata = metadata.copy()
                triplet_metadata['triplet_idx'] = 0
                triplet_metadata['slice_indices'] = list(range(start_slice, end_slice))
                
                new_metadata_list.append(triplet_metadata)
                new_file_indices.append((file_idx, patch_idx, 0))
        
        self.metadata_list = new_metadata_list
        self.file_indices = new_file_indices

    def _load_triplet(self, file_path: str, patch_idx: int, triplet_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a triplet from preprocessed data"""
        cache_key = f"{file_path}_{patch_idx}_{triplet_idx}"

        if cache_key in self._patch_cache:
            return self._patch_cache[cache_key]

        with self._get_file_handle(file_path) as f:
            motion_corrupted = torch.from_numpy(f['motion_corrupted'][patch_idx])
            motion_free = torch.from_numpy(f['motion_free'][patch_idx])
            
            if motion_corrupted.dim() == 3:
                if motion_corrupted.shape[0] == 6:
                    triplet_corrupted = motion_corrupted[triplet_idx:triplet_idx + 3]  # (3, H, W)
                    triplet_free = motion_free[triplet_idx:triplet_idx + 3]  # (3, H, W)
                elif motion_corrupted.shape[-1] == 16:
                    motion_corrupted = motion_corrupted.permute(2, 0, 1)  # (16, H, W)
                    motion_free = motion_free.permute(2, 0, 1)  # (16, H, W)
                    
                    # Extract triplet from center slices (around slice 8)
                    center_slice = 8  # Middle of 16 slices
                    start_slice = max(0, center_slice - 1)
                    end_slice = min(16, center_slice + 2)
                    
                    # Ensure we have 3 slices
                    if end_slice - start_slice < 3:
                        if start_slice == 0:
                            end_slice = 3
                        else:
                            start_slice = end_slice - 3
                    
                    triplet_corrupted = motion_corrupted[start_slice:end_slice]  # (3, H, W)
                    triplet_free = motion_free[start_slice:end_slice]  # (3, H, W)
                else:
                    triplet_corrupted = motion_corrupted[triplet_idx:triplet_idx + 3]  # (3, H, W)
                    triplet_free = motion_free[triplet_idx:triplet_idx + 3]  # (3, H, W)
            else:
                logger.warning(f"Unexpected data dimensions: {motion_corrupted.shape}")
                triplet_corrupted = motion_corrupted[triplet_idx:triplet_idx + 3]  # (3, H, W)
                triplet_free = motion_free[triplet_idx:triplet_idx + 3]  # (3, H, W)
            
            if len(self._patch_cache) < self.cache_size:
                self._patch_cache[cache_key] = (triplet_free, triplet_corrupted)
            
            return triplet_free, triplet_corrupted

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a triplet and target with metadata (all values converted to strings)"""
        _, patch_idx, triplet_idx = self.file_indices[idx]
        metadata = self.metadata_list[idx]
        file_path = metadata['file_path']
        
        x, y = self._load_triplet(file_path, patch_idx, triplet_idx)
        
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        
        x = x.float()
        y = y.float()
        
        metadata_str = {k: str(v) for k, v in metadata.items()}
        
        return x, y, metadata_str
    
    def get_full_metadata(self, idx: int) -> Dict:
        """Get full metadata dictionary for a specific index"""
        return self.metadata_list[idx]
    
    def get_metadata_batch(self, indices: List[int]) -> List[Dict]:
        """Get full metadata for a batch of indices"""
        return [self.metadata_list[idx] for idx in indices]
    
    def get_all_indices_for_patch(self, file_name: str, vol_idx: int) -> List[int]:
        """Get all indices for a specific patch (compatibility method)"""
        patch_indices = []
        for idx, metadata in enumerate(self.metadata_list):
            # Check if this metadata matches the requested file and vol_idx
            if (metadata.get('file') == file_name and 
                metadata.get('vol_idx') == vol_idx):
                patch_indices.append(idx)
        return patch_indices
