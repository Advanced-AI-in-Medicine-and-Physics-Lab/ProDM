import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage
from enum import Enum
from skimage.transform import radon, iradon
import nibabel as nib

class MotionPattern(str, Enum):
    OSCILLATORY = "oscillatory"    # back-and-forth oscillation along a fixed direction
    LOCAL_JITTER = "local_jitter"  # multi-frequency small jitter about a point
    TRANSLATION = "translation"    # constant-velocity linear drift
    PIECEWISE = "piecewise"        # in-plane piecewise-linear drift

class CalciumMotionSimulator:
    def __init__(self, ct_scan: np.ndarray, voxel_spacing: np.ndarray, calcium_mask: np.ndarray):
        """Initialize the calcium motion simulator.
        
        Args:
            ct_scan: The input CT scan volume
            voxel_spacing: Voxel spacing in mm (dx, dy, dz)
            calcium_mask: Binary mask indicating calcium deposits
        """
        self.ct_scan = ct_scan.copy()
        self.voxel_spacing = voxel_spacing
        self.calcium_mask = calcium_mask.copy()
        
        # Store deposit information
        self.deposits = self._get_deposit_info()
        
    def _get_calcium_deposits(self) -> List[np.ndarray]:
        """Get all points in each deposit in voxel coordinates.
        
        Returns:
            List of arrays, each containing the points of one deposit
        """
        # find slices that contain calcium
        active_slices = []
        for z in range(self.calcium_mask.shape[2]):
            if np.any(self.calcium_mask[:, :, z]):
                active_slices.append(z)
        
        if not active_slices:
            return []
            
        # group together adjacent slices to form deposits (allowing for 1 slice gap)
        slice_groups = []
        current_group = [active_slices[0]]
        
        for i in range(1, len(active_slices)):
            if active_slices[i] - active_slices[i-1] <= 2:  # allow 1 slice gap
                current_group.append(active_slices[i])
            else:
                slice_groups.append(current_group)
                current_group = [active_slices[i]]
        slice_groups.append(current_group)
        
        # process each group as a single deposit
        deposits = []
        
        for slice_group in slice_groups:
            # generate a mask for this deposit group
            deposit_mask = np.zeros_like(self.calcium_mask, dtype=bool)
            for z in slice_group:
                deposit_mask[:, :, z] = self.calcium_mask[:, :, z]
            
            # extract all points in this deposit
            deposit_points = np.array(np.where(deposit_mask)).T
            
            if len(deposit_points) > 0:
                deposits.append(deposit_points)
        
        return deposits
        
    def _get_deposit_info(self) -> Dict[int, Dict]:
        """Extract information about each calcium deposit.
        
        Returns:
            Dictionary mapping deposit ID to its properties (points, intensity values, etc.)
        """
        deposits = {}
        deposit_points_list = self._get_calcium_deposits()
        
        for deposit_idx, points_voxel in enumerate(deposit_points_list, 1):
            # convert from voxel coordinates to real-world coordinates
            points_real = points_voxel * self.voxel_spacing
            intensities = self.ct_scan[points_voxel[:, 0], points_voxel[:, 1], points_voxel[:, 2]]
            
            deposits[deposit_idx] = {
                'points_real': points_real,  
                'points_voxel': points_voxel, 
                'intensities': intensities, 
                'center_real': np.mean(points_real, axis=0), 
                'center_voxel': np.mean(points_voxel, axis=0)  
            }
        
        return deposits
    
    def get_deposits(self) -> Dict[int, Dict]:
        """Get information about all calcium deposits.
        
        Returns:
            Dictionary containing information about each deposit
        """
        return self.deposits
    
    def inpaint_calcium(self, padding: int = 3) -> np.ndarray:
        """Remove calcium deposits from the CT scan and inpaint the regions slice by slice.
        
        Args:
            padding: Number of voxels to pad around each deposit for inpainting
            
        Returns:
            CT scan with calcium deposits removed and inpainted
        """
        # Create a copy of the CT scan
        inpainted = self.ct_scan.copy()
        
        # Process each slice that contains calcium
        for z in range(self.ct_scan.shape[2]):
            # Skip if no calcium in this slice
            if z > self.calcium_mask.shape[2] - 1 or not np.any(self.calcium_mask[:, :, z]):
                continue
                
            # Get 2D masks for this slice
            slice_mask = self.calcium_mask[:, :, z]
            
            # Dilate the mask in 2D
            dilated_mask = ndimage.binary_dilation(slice_mask, 
                                                 iterations=padding)
            
            # Get the slice data
            ct_slice = inpainted[:, :, z]
            
            # Find the bounding box of the dilated mask
            rows, cols = np.where(dilated_mask)
            if len(rows) == 0:
                continue
                
            # Get bounding box with padding
            min_row, max_row = np.min(rows) - padding, np.max(rows) + padding
            min_col, max_col = np.min(cols) - padding, np.max(cols) + padding
            
            # Ensure bounds are within slice
            min_row = max(0, min_row)
            max_row = min(ct_slice.shape[0] - 1, max_row)
            min_col = max(0, min_col)
            max_col = min(ct_slice.shape[1] - 1, max_col)
            
            # Extract region around calcium
            region = ct_slice[min_row:max_row+1, min_col:max_col+1]
            mask_region = dilated_mask[min_row:max_row+1, min_col:max_col+1]
            
            # Calculate mean of surrounding non-calcium voxels
            surrounding_values = region[~mask_region]
            if len(surrounding_values) > 0:
                mean_value = np.mean(surrounding_values)
                # Inpaint calcium region with mean value
                region[mask_region] = mean_value
                
                # Update the slice
                ct_slice[min_row:max_row+1, min_col:max_col+1] = region
                inpainted[:, :, z] = ct_slice
        
        return inpainted
    
    def advanced_inpaint_calcium(self, scale_factor: float = 0.5) -> np.ndarray:
        """Inpaint calcium deposits in the CT scan using a more advanced method by 1) inpaint; 2) forward projection;
        3) add poisson noise; 4) back projection.
        """
        # 1) inpaint
        inpainted = self.inpaint_calcium()
        # print("finished inpainting")
        
        # 2) forward projection
        sinogram = self.forward_project(inpainted)
        # print("finished forward projection")
        
        # 3) add poisson noise
        sinogram_noisy = self.add_poisson_noise(sinogram, scale_factor)
        # print("finished adding poisson noise")
        
        # 4) back projection
        inpainted_noisy = self.back_project(sinogram_noisy)
        # print("finished back projection")
        return inpainted_noisy
        
    def forward_project(self, ct_scan: np.ndarray, num_angles: int = 1024) -> np.ndarray:
        detector_size = int(np.ceil(np.sqrt(2) * max(ct_scan.shape[:2])))
        num_slices = ct_scan.shape[-1]
        angles = np.linspace(0, 180, num_angles, endpoint=False)
        sinograms = np.zeros((num_slices, detector_size, num_angles))
        for z in range(num_slices):
            sinogram = radon(ct_scan[:, :, z], theta=angles, circle=False)
            sinograms[z] = sinogram
        np.save("inpainted_sinogram.npy", sinograms)
        # sanity check against empty sinograms
        if np.all(sinograms == 0):
            raise ValueError("Sinogram is full of zeroes")
        
        return sinograms
    

    def add_poisson_noise(self, sinogram: np.ndarray, dose_scale: float = 1e-1) -> np.ndarray:
        """
        This is a helper for adding Poisson noise to the sinogram in the projection domain.
        
        Args:
            sinogram: The original sinogram array.
            dose_scale: Scaling factor for the Poisson noise.

        Returns:
            Noisy sinogram array.
        """
        
        transmission_scaled = dose_scale * sinogram
        noisy_sinogram_scaled = np.random.poisson(transmission_scaled).astype(np.float32)
        noisy_sinogram = noisy_sinogram_scaled / dose_scale
        if np.all(noisy_sinogram == 0):
            raise ValueError("Noisy sinogram is full of zeroes")
        return noisy_sinogram

    def back_project(self, noisy_sinogram: np.ndarray, num_angles: int = 1024) -> np.ndarray:
        noisy_sinogram = np.transpose(noisy_sinogram, (1, 2, 0))
        # back project each slice
        reconstructed_slices = []
        for z in range(noisy_sinogram.shape[-1]):
            # print(f"back projecting slice {z} out of {noisy_sinogram.shape[-1]}")
            reconstructed_slice = iradon(noisy_sinogram[:, :, z], theta=np.linspace(0, 180, num_angles), circle=False)
            reconstructed_slices.append(reconstructed_slice)
        reconstructed_ct = np.stack(reconstructed_slices, axis=-1)
        if np.all(reconstructed_ct == 0):
            raise ValueError("Reconstructed CT is full of zeroes")
        return reconstructed_ct
        


        
        
        
    
    def apply_displacements(self, displacements: np.ndarray, inpainted_ct: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply displacements to calcium deposits and return new CT scan.
        
        Args:
            displacements: Array of shape (num_deposits, 3) containing [dx, dy, dz]
                         displacements in mm for each deposit
            inpainted_ct: Optional pre-inpainted CT scan. If None, will use advanced_inpaint_calcium()
                         
        Returns:
            New CT scan with displaced calcium deposits
        """
        if len(displacements) != len(self.deposits):
            raise ValueError(f"Number of displacements ({len(displacements)}) must match "
                           f"number of deposits ({len(self.deposits)})")
        
        # Use provided inpainted CT or create new one
        new_ct = inpainted_ct.copy() if inpainted_ct is not None else self.advanced_inpaint_calcium(scale_factor=0.1)
        
        # For each deposit
        for (deposit_idx, deposit_info), displacement in zip(self.deposits.items(), displacements):
            # Convert displacement from mm to voxels
            displacement_voxels = displacement / self.voxel_spacing
            
            # Get original points and intensities
            points = deposit_info['points_voxel']
            intensities = deposit_info['intensities']
            
            # Apply displacement
            new_points = points + displacement_voxels
            
            # Round to nearest voxel and convert to integers
            new_points = np.round(new_points).astype(int)
            
            # Filter points that are within volume bounds
            valid_mask = np.all(
                (new_points >= 0) & (new_points < np.array(self.ct_scan.shape)),
                axis=1
            )
            new_points = new_points[valid_mask]
            intensities = intensities[valid_mask]
            
            # Place calcium values in new positions
            new_ct[new_points[:, 0], new_points[:, 1], new_points[:, 2]] = intensities
        
        return new_ct
    
    def generate_motion_tracks(self, num_angles: int,
                             motion_pattern: MotionPattern,
                             num_cycles: float = 1.0,
                             displacement_magnitude: float = 5.0,
                             displacement_ratios: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                             random_seed: Optional[int] = None, phase_shift: float = 0.0,
                             pw_theta1: float = 30.0, pw_theta2: float = 120.0,
                             pw_v1: float = 0.01, pw_v2: float = 0.2,
                             pw_t_star: int = 30, pw_dwell: int = 5,
                             pw_jitter_px: float = 0.0) -> np.ndarray:
        """Generate displacement tracks for each calcium deposit across projection angles.

        Single dispatch point for all motion patterns; returns an array of shape
        (num_angles, num_deposits, 3) fed to apply_displacements().

        Args:
            num_angles: Number of projection angles
            motion_pattern: OSCILLATORY | LOCAL_JITTER | TRANSLATION | PIECEWISE
            num_cycles: Complete motion cycles over the scan (OSCILLATORY/LOCAL_JITTER)
            displacement_magnitude: Peak displacement in mm (OSCILLATORY/LOCAL_JITTER/TRANSLATION)
            displacement_ratios: Relative (x, y, z) motion ratios
            random_seed: Optional random seed for reproducibility
            phase_shift: Phase shift (radians) for oscillation
            pw_theta1/pw_theta2: PIECEWISE phase directions (deg)
            pw_v1/pw_v2: PIECEWISE phase velocities (px/view)
            pw_t_star/pw_dwell: PIECEWISE change-view index / stationary views
            pw_jitter_px: PIECEWISE per-view, per-deposit Gaussian jitter (px std)
        """
        num_deposits = len(self.deposits)
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Normalize displacement ratios to max of 1
        ratios = np.array(displacement_ratios)
        ratios = ratios / np.max(ratios) 
        tracks = np.zeros((num_angles, num_deposits, 3))
        
        if motion_pattern == MotionPattern.OSCILLATORY:
            # Generate random but fixed directions for each deposit
            base_directions = np.random.randn(num_deposits, 3)
            # Normalize directions
            base_directions = base_directions / np.linalg.norm(base_directions, axis=1, keepdims=True)
            # scale by ratios
            base_directions *= ratios.reshape(1, 3)
            # re-normalize to maintain maximum displacement
            base_directions = base_directions / np.max(np.linalg.norm(base_directions, axis=1))
            
            # Generate displacements for each angle using smooth transition
            for angle_idx in range(num_angles):
                # Calculate time point in cardiac cycle [0,1]
                time_point = (angle_idx / num_angles) * num_cycles % 1.0
                
                # Use cosine centered around zero [-1, 1]
                x = 2 * np.pi * time_point
                time_factor = np.cos(x)  # Now oscillates between -1 and 1
                
                # Scale each direction by half the displacement magnitude and time factor
                # This makes the motion centered around the original position
                tracks[angle_idx] = base_directions * (displacement_magnitude/2) * time_factor
                
        elif motion_pattern == MotionPattern.LOCAL_JITTER:
            # generate multiple frequency components for local jittering motion
            base_freq = np.random.randn(num_deposits, 3) * ratios.reshape(1, 3)
            double_freq = np.random.randn(num_deposits, 3) * ratios.reshape(1, 3)
            triple_freq = np.random.randn(num_deposits, 3) * ratios.reshape(1, 3)
            phase_shifts = np.random.uniform(0, 2*np.pi, (num_deposits, 3))
            
            for angle_idx in range(num_angles):
                time_point = (angle_idx / num_angles) * num_cycles % 1.0
                x = 2 * np.pi * time_point
                
                # define basis of jittery motion components
                base_motion = np.cos(x + phase_shifts) * base_freq
                double_motion = 0.5 * np.cos(2*x + phase_shifts) * double_freq
                triple_motion = 0.25 * np.cos(3*x + phase_shifts) * triple_freq
                
                # combine motions and scale
                # scale by half the displacement magnitude to center around original position
                combined_motion = (base_motion + double_motion + triple_motion) * (displacement_magnitude/2)
                
                # Ensure maximum displacement is not exceeded
                norms = np.linalg.norm(combined_motion, axis=1)
                scale = np.minimum(1.0, displacement_magnitude/2 / norms)
                tracks[angle_idx] = combined_motion * scale[:, np.newaxis]


        elif motion_pattern == MotionPattern.TRANSLATION:
            # 3-D constant-velocity translation: each deposit drifts linearly in a
            # fixed random 3-D direction, from 0 to displacement_magnitude (mm).
            dirs = np.random.randn(num_deposits, 3)
            dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs *= ratios.reshape(1, 3)
            dirs = dirs / np.max(np.linalg.norm(dirs, axis=1))  # peak magnitude = 1
            for angle_idx in range(num_angles):
                t = angle_idx / num_angles
                tracks[angle_idx] = dirs * displacement_magnitude * t

        elif motion_pattern == MotionPattern.PIECEWISE:
            # 2-D in-plane piecewise-linear drift (z = 0). All deposits
            # share the path; pw_jitter_px adds per projection angle, per-deposit noise.
            u1 = _unit_vec_from_deg(pw_theta1)
            u2 = _unit_vec_from_deg(pw_theta2)
            for angle_idx in range(num_angles):
                view_idx = int((angle_idx / num_angles) * 180)  # view-index timing
                if view_idx < pw_t_star:
                    disp = view_idx * pw_v1 * u1
                elif view_idx < pw_t_star + pw_dwell:
                    disp = pw_t_star * pw_v1 * u1
                else:
                    disp = pw_t_star * pw_v1 * u1 + (view_idx - pw_t_star - pw_dwell) * pw_v2 * u2
                base = np.array([disp[1], disp[0]], dtype=np.float32)  # (y,x) -> (x,y)
                for d_idx in range(num_deposits):
                    d = base.copy()
                    if pw_jitter_px > 0:
                        d = d + np.random.normal(0.0, pw_jitter_px, size=2).astype(np.float32)
                    tracks[angle_idx, d_idx, 0] = d[0]
                    tracks[angle_idx, d_idx, 1] = d[1]

        return tracks
    
def _unit_vec_from_deg(deg: float) -> np.ndarray:
    """2-D unit vector [cos, sin] for an angle in degrees (used by PIECEWISE)."""
    th = np.radians(deg)
    return np.array([np.cos(th), np.sin(th)], dtype=np.float32)