#!/usr/bin/env python3
import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import nibabel as nib
from skimage.transform import radon, iradon

from calcium_simulator import CalciumMotionSimulator, MotionPattern
from utils import load_nifti

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "motion_config.json")


# ======================================================================================
# Unified motion profile interface
# ======================================================================================

class MotionProfile:
    """Base interface for a single named motion realization."""
    index: int
    name: str
    family: str

    def make_tracks(self, simulator: "CalciumMotionSimulator", num_angles: int,
                    seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError


@dataclass
class OscillationProfile(MotionProfile):
    """3-D sinusoidal oscillation: each deposit swings back and forth (cosine)
    along a fixed random 3-D direction. Differs by cycles-per-scan and amplitude."""
    index: int
    name: str
    cycles_range: Tuple[float, float]
    disp_range: Tuple[float, float]
    family: str = "oscillation"

    def make_tracks(self, simulator, num_angles, seed=None):
        return simulator.generate_motion_tracks(
            num_angles=num_angles,
            num_cycles=random.uniform(*self.cycles_range),
            motion_pattern=MotionPattern.OSCILLATORY,
            displacement_magnitude=random.uniform(*self.disp_range),
            displacement_ratios=get_random_displacement_ratios(),
            random_seed=seed,
            phase_shift=random.uniform(0, 2 * math.pi),
        )


@dataclass
class JitterProfile(MotionProfile):
    """3-D local jitter: multi-frequency, small, smooth random motion of each
    deposit about its original position (irregular, non-periodic)."""
    index: int
    name: str
    cycles_range: Tuple[float, float]
    disp_range: Tuple[float, float]
    family: str = "jitter"

    def make_tracks(self, simulator, num_angles, seed=None):
        return simulator.generate_motion_tracks(
            num_angles=num_angles,
            num_cycles=random.uniform(*self.cycles_range),
            motion_pattern=MotionPattern.LOCAL_JITTER,
            displacement_magnitude=random.uniform(*self.disp_range),
            displacement_ratios=get_random_displacement_ratios(),
            random_seed=seed,
            phase_shift=random.uniform(0, 2 * math.pi),
        )


@dataclass
class TranslationProfile(MotionProfile):
    """3-D constant-velocity translation: each deposit drifts linearly in a fixed
    random 3-D direction over the scan."""
    index: int
    name: str
    disp_range: Tuple[float, float]
    family: str = "translation"

    def make_tracks(self, simulator, num_angles, seed=None):
        return simulator.generate_motion_tracks(
            num_angles=num_angles,
            motion_pattern=MotionPattern.TRANSLATION,
            displacement_magnitude=random.uniform(*self.disp_range),
            displacement_ratios=get_random_displacement_ratios(),
            random_seed=seed,
        )


@dataclass
class PiecewiseProfile(MotionProfile):
    """2-D in-plane piecewise-linear ("lambda") drift: each deposit moves in
    direction theta1 at velocity v1 for t_star views, optionally dwells, then
    changes to direction theta2 at velocity v2. Each parameter may be a plain value
    or a {type: fixed|range|choice} spec, resolved per-realization at track time."""
    index: int
    name: str
    theta1: Any
    theta2: Any
    v1: Any
    v2: Any
    t_star: Any
    dwell: Any
    jitter_px: Any = 0.0
    family: str = "piecewise"

    @classmethod
    def from_config(cls, index: int, name: str, entry: Dict[str, Any]) -> "PiecewiseProfile":
        return cls(
            index=index, name=name,
            theta1=entry["piecewise_theta1"], theta2=entry["piecewise_theta2"],
            v1=entry["piecewise_v1"], v2=entry["piecewise_v2"],
            t_star=entry["piecewise_t_star"], dwell=entry["piecewise_dwell"],
            jitter_px=entry.get("jitter_px", 0.0),
        )

    def make_tracks(self, simulator, num_angles, seed=None):
        rng = np.random.default_rng(seed)
        jitter_px = sample_parameter(self.jitter_px, rng)
        theta1 = sample_parameter(self.theta1, rng)
        theta2 = sample_parameter(self.theta2, rng)
        v1 = sample_parameter(self.v1, rng)
        v2 = sample_parameter(self.v2, rng)
        t_star = sample_parameter(self.t_star, rng)
        dwell = sample_parameter(self.dwell, rng)
        return simulator.generate_motion_tracks(
            num_angles=num_angles,
            motion_pattern=MotionPattern.PIECEWISE,
            random_seed=seed,
            pw_theta1=theta1, pw_theta2=theta2,
            pw_v1=v1, pw_v2=v2,
            pw_t_star=t_star, pw_dwell=dwell,
            pw_jitter_px=jitter_px,
        )


# ======================================================================================
# Shared forward and inverse Radon logic
# ======================================================================================

def reconstruct_per_slice(sinogram: np.ndarray, theta: np.ndarray,
                          filter_name: str = "ramp") -> np.ndarray:
    """Filtered-back-projection per slice. sinogram: (num_slices, det, num_angles)."""
    num_slices = sinogram.shape[0]
    recon_slices = []
    for s in range(num_slices):
        # sinogram[s] is (detector, num_angles): each column is one projection
        # angle, which is exactly what iradon expects.
        sino_slice = sinogram[s]
        recon_slices.append(iradon(sino_slice, theta=theta, filter_name=filter_name, circle=False))
        if (s + 1) % 20 == 0:
            print(f"  reconstructed slice {s + 1}/{num_slices}")
    return np.stack(recon_slices, axis=2)


def simulate_one_motion(volume: np.ndarray, voxel_spacing: np.ndarray, calcium_path: str,
                        tracks: np.ndarray, theta: np.ndarray,
                        output_dir: Optional[str] = None, output_name: Optional[str] = None,
                        add_poisson_noise: bool = False, verbose: bool = True,
                        simulator: Optional["CalciumMotionSimulator"] = None) -> np.ndarray:
    """Shared forward/inverse engine for a single motion.

    Given pre-computed per-view deposit displacement `tracks` (num_angles,
    num_deposits, 3), inpaint the calcium, then for each view re-insert the
    displaced deposits and forward-project; finally reconstruct and (optionally)
    save as recon_<output_name>.nii.gz. Pass `simulator` to reuse one already
    built for track generation (avoids re-detecting deposits).
    """
    num_slices = volume.shape[2]
    num_angles = len(theta)
    detector_size = int(np.ceil(np.sqrt(2) * max(volume.shape[:2])))
    sinogram = np.zeros((num_slices, detector_size, num_angles))

    if simulator is None:
        # Fallback only: assumes the mask is already on the volume grid. The drivers
        # build the simulator from load_mask_on_volume_grid() and pass it in, which is
        # the affine-aligned path; prefer that for masks with a differing orientation.
        calcium_mask, _ = load_nifti(calcium_path)
        simulator = CalciumMotionSimulator(volume, voxel_spacing, calcium_mask)
    t0 = time.time()
    inpainted_ct = simulator.inpaint_calcium(padding=3)
    if verbose:
        print(f"  inpaint done in {time.time() - t0:.1f}s; "
              f"{tracks.shape[1]} deposits, {num_angles} views")

    for angle_idx, angle in enumerate(theta):
        warped = simulator.apply_displacements(tracks[angle_idx], inpainted_ct)
        for slice_idx in range(num_slices):
            proj = radon(warped[:, :, slice_idx], theta=[angle], circle=False)
            col = proj[:, 0]
            if add_poisson_noise:
                col = _poisson(col)
            sinogram[slice_idx, :, angle_idx] = col
        del warped
        if verbose and angle_idx % 60 == 0:
            print(f"  view {angle_idx + 1}/{num_angles}")

    if verbose:
        print("reconstructing...")
    recon = reconstruct_per_slice(sinogram, theta)

    if output_dir is not None and output_name is not None:
        recon_dir = os.path.join(output_dir, "reconstructions")
        os.makedirs(recon_dir, exist_ok=True)
        affine = np.eye(4)
        affine[0, 0], affine[1, 1], affine[2, 2] = voxel_spacing[:3]
        out_path = os.path.join(recon_dir, f"recon_{output_name}.nii.gz")
        nib.save(nib.Nifti1Image(recon.astype(np.float32), affine), out_path)
        if verbose:
            print(f"  saved {out_path}")
    return recon

def _poisson(sino_col, dose_scale=5e-1):
    return np.random.poisson(dose_scale * sino_col).astype(np.float32) / dose_scale

# ======================================================================================
# Driver: build the full motion profile set for one scan
# ======================================================================================


def run_all_motions_for_scan(volume_path: str, mask_path: str, output_dir: str,
                             angles: int = 180, random_angles: bool = False,
                             seed: Optional[int] = 42,
                             config_path: str = DEFAULT_CONFIG,
                             only: Optional[str] = None,
                             add_poisson_noise: bool = False, verbose: bool = True) -> None:
    """Generate all motions for one scan from the unified MotionProfile registry.

    Output files: <output_dir>/reconstructions/recon_<name>_<index>.nii.gz
    `only` restricts to a single profile by name or <name>_<index>
    (e.g. "fast_oscillation", "lambda_pos_balanced", "lambda_pos_balanced_16").
    `random_angles` draws each motion's view count from ANGLE_CHOICES (otherwise the
    fixed `angles` is used for all).
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    volume, voxel_spacing = load_nifti(volume_path)

    for profile in build_motion_profiles(config_path):
        name_idx = f"{profile.name}_{profile.index}"
        if only is not None and only not in (profile.name, name_idx):
            continue
        # introduce some randomness in the number of projection angles
        n_angles = choose_num_angles(angles, random_angles)
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
        # one simulator per motion, reused for track generation and projection.
        # mask is aligned to the volume grid via affines (handles flipped-axis masks)
        calcium_mask = load_mask_on_volume_grid(mask_path, volume_path)
        simulator = CalciumMotionSimulator(volume, voxel_spacing, calcium_mask)
        tracks = profile.make_tracks(simulator, n_angles, seed)
        if verbose:
            print(f"\n[{profile.index:2d}] {name_idx}  ({profile.family}, {n_angles} views)")
        simulate_one_motion(volume, voxel_spacing, mask_path, tracks, theta,
                            output_dir=output_dir, output_name=name_idx,
                            add_poisson_noise=add_poisson_noise, verbose=verbose,
                            simulator=simulator)
        
# ======================================================================================
# Misc: some utility functions for motion simulation
# ======================================================================================

def get_random_displacement_ratios() -> Tuple[float, float, float]:
    """Random (x, y, z) motion ratios, emphasising in-plane motion. 
    Z is always the smallest component because the pixel spacing in z 
    is greatest and therefore have least amount of motion.
    """
    dominant = random.choice(["horizontal", "vertical", "both", "z"])
    if dominant == "horizontal":
        x = random.uniform(2.0, 4.0); y = random.uniform(0.5, 1.5)
    elif dominant == "vertical":
        x = random.uniform(0.5, 1.5); y = random.uniform(2.0, 4.0)
    elif dominant == "both":
        x = random.uniform(2.0, 3.0); y = random.uniform(2.0, 3.0)
    else:  # z-dominant
        x = random.uniform(0.5, 1.5); y = random.uniform(0.5, 1.5)
    z = random.uniform(0.3, 1.0)
    return (x, y, z)

def load_motion_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def load_mask_on_volume_grid(mask_path: str, volume_path: str) -> np.ndarray:
    """Load the calcium mask resampled onto the volume's voxel grid via the NIfTI
    affines (nearest-neighbor, so per-artery labels are preserved).

    The simulator indexes the mask and the volume on the same (i, j, k) grid, but a
    mask may be stored with a different orientation/affine than its volume (e.g. a
    flipped y-axis), in which case using the raw array places calcium at the wrong
    anatomical location. Resampling through the affines spatially aligns the mask;
    if the affines already match, this is a no-op.
    """
    from nibabel.processing import resample_from_to
    vol_img = nib.load(volume_path)
    mask_img = nib.load(mask_path)
    aligned = resample_from_to(mask_img, (vol_img.shape[:3], vol_img.affine), order=0)
    return np.asarray(aligned.get_fdata())


def sample_parameter(param_config: Any, rng: np.random.Generator) -> Any:
    """Resolve one config field: a plain value, or {type: fixed|range|choice}."""
    if not isinstance(param_config, dict):
        return param_config
    t = param_config["type"]
    if t == "fixed":
        return param_config["value"]
    if t == "range":
        return rng.uniform(param_config["min"], param_config["max"])
    if t == "choice":
        return rng.choice(param_config["values"])
    raise ValueError(f"Unknown parameter type: {t}")


# Allowed projection-view counts when --random-angles is enabled (one drawn per motion)
ANGLE_CHOICES: Tuple[int, ...] = (180, 360, 540, 720, 1080)


def choose_num_angles(angles: int, random_angles: bool) -> int:
    return random.choice(ANGLE_CHOICES) if random_angles else angles

# Default relative weights for a "random" motion, default to equally likely across the 4 types.
RANDOM_PROFILE_WEIGHTS: Dict[str, float] = {
    "oscillation": 1.0, "jitter": 1.0, "translation": 1.0, "piecewise": 1.0,
}


def make_random_profile(index: int, name: str, config: Dict[str, Any],
                        weights: Optional[Dict[str, float]] = None) -> MotionProfile:
    """Build a random-type profile, choosing among the 4 motion types by weight.

    `weights` maps family to relative probability with the default being equal probabilities. 
    Wide default parameter ranges are used; a random piecewise picks a random config preset.
    """
    w = weights or RANDOM_PROFILE_WEIGHTS
    fam = random.choices(list(w.keys()), weights=list(w.values()), k=1)[0]
    if fam == "oscillation":
        return OscillationProfile(index, name, (0.2, 1.5), (6, 15))
    if fam == "jitter":
        return JitterProfile(index, name, (0.8, 1.2), (6, 15))
    if fam == "translation":
        return TranslationProfile(index, name, (6, 15))
    if fam == "piecewise":
        presets = _piecewise_presets(config)
        preset_name = random.choice(sorted(presets.keys()))
        return PiecewiseProfile.from_config(index, name, presets[preset_name])
    raise ValueError(f"Unknown motion family: {fam}")


def build_motion_profiles(config_path: str = DEFAULT_CONFIG,
                          random_weights: Optional[Dict[str, float]] = None) -> List[MotionProfile]:
    """Return the ordered list of motion profiles as MotionProfile objects.
    """
    cfg = load_motion_config(config_path)
    profiles: List[MotionProfile] = []
    for key in sorted(cfg.keys(), key=int):
        e = cfg[key]
        i, name, mtype = int(key), e["name"], e["type"]
        if mtype == "random":
            profiles.append(make_random_profile(i, name, cfg, random_weights))
        elif mtype == "oscillation":
            profiles.append(OscillationProfile(i, name, tuple(e["cycles_range"]), tuple(e["disp_range"])))
        elif mtype == "jitter":
            profiles.append(JitterProfile(i, name, tuple(e["cycles_range"]), tuple(e["disp_range"])))
        elif mtype == "translation":
            profiles.append(TranslationProfile(i, name, tuple(e["disp_range"])))
        elif mtype == "piecewise":
            profiles.append(PiecewiseProfile.from_config(i, name, e))
        else:
            raise ValueError(f"Unknown motion type for index {key}: {mtype!r}")
    return profiles


def _piecewise_presets(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """All piecewise entries in the config, keyed by their `name` (for random picks)."""
    return {e["name"]: e for e in config.values() if e.get("type") == "piecewise"}



def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Calcium-only coronary motion simulator.")
    ap.add_argument("--volume", required=True, help="Path to the gated motion-free CT volume (.nii/.nii.gz)")
    ap.add_argument("--mask", required=True, help="Path to the per-artery calcium mask (.nii.gz)")
    ap.add_argument("--output", required=True, help="Output directory (recons saved under <output>/reconstructions)")
    ap.add_argument("--angles", type=int, default=180, help="Number of projection views (used unless --random-angles)")
    ap.add_argument("--random-angles", action="store_true",
                    help=f"Draw each motion's view count randomly from {list(ANGLE_CHOICES)}")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42, matching the dataset batch config)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="Path to motion profile configuration JSON")
    ap.add_argument("--only", default=None, help="Generate a single profile by name (e.g. fast_oscillation, lambda_pos_balanced)")
    ap.add_argument("--poisson", action="store_true", help="Add Poisson noise in the projection domain")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_all_motions_for_scan(
        volume_path=args.volume,
        mask_path=args.mask,
        output_dir=args.output,
        angles=args.angles,
        random_angles=args.random_angles,
        seed=args.seed,
        config_path=args.config,
        only=args.only,
        add_poisson_noise=args.poisson,
        verbose=not args.quiet,
    )
