#!/usr/bin/env python3
import os
import glob
import time
import json
import argparse
from typing import List, Optional

import numpy as np

from calcium_simulator import CalciumMotionSimulator
from utils import load_nifti
import simulate_motion_calcium_only as sim


def discover_patients(volume_dir: str, mask_dir: str) -> List[int]:
    """Return patient ids (int) that have BOTH <id>.nii and mask_<id>.nii.gz."""
    vols = {os.path.basename(f).split(".")[0] for f in glob.glob(os.path.join(volume_dir, "*.nii"))}
    vols = {int(v) for v in vols if v.isdigit()}
    masks = set()
    for f in glob.glob(os.path.join(mask_dir, "mask_*.nii.gz")):
        pid = os.path.basename(f)[5:-7]  # strip "mask_" / ".nii.gz"
        if pid.isdigit():
            masks.add(int(pid))
    return sorted(vols & masks)


def get_recon_names(config_path: str) -> List[str]:
    """The output basenames in index order."""
    return [f"{p.name}_{p.index}" for p in sim.build_motion_profiles(config_path)]


def missing_recons(output_dir: str, patient_id: int, config_path: str) -> List[str]:
    """Finds out which of the specified recons are not yet present for this patient."""
    recon_dir = os.path.join(output_dir, str(patient_id), "reconstructions")
    missing = []
    for name in get_recon_names(config_path):
        if not os.path.exists(os.path.join(recon_dir, f"recon_{name}.nii.gz")):
            missing.append(name)
    return missing


def process_patient(patient_id: int, volume_dir: str, mask_dir: str, output_dir: str,
                    angles: int, random_angles: bool, seed: Optional[int], config_path: str,
                    overwrite: bool, add_poisson_noise: bool, verbose: bool) -> dict:
    """Generate the missing motions for one patient. Returns a result record."""
    volume_path = os.path.join(volume_dir, f"{patient_id}.nii")
    mask_path = os.path.join(mask_dir, f"mask_{patient_id}.nii.gz")
    out_patient = os.path.join(output_dir, str(patient_id))
    t0 = time.time()

    if not overwrite:
        missing = missing_recons(output_dir, patient_id, config_path)
        if not missing:
            return {"patient_id": patient_id, "status": "skipped_complete",
                    "generated": 0, "seconds": 0.0}

    volume, voxel_spacing = load_nifti(volume_path)
    recon_dir = os.path.join(out_patient, "reconstructions")

    # seed per patient so patients differ
    pseed = None if seed is None else seed + patient_id
    import random as _random
    if pseed is not None:
        _random.seed(pseed); np.random.seed(pseed)

    n_gen = 0
    # single unified loop over all motion profiles
    for profile in sim.build_motion_profiles(config_path):
        name_idx = f"{profile.name}_{profile.index}"
        out_file = os.path.join(recon_dir, f"recon_{name_idx}.nii.gz")
        if not overwrite and os.path.exists(out_file):
            continue
        n_angles = sim.choose_num_angles(angles, random_angles)
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
        # align mask to the volume grid via affines (handles flipped-axis masks)
        calcium_mask = sim.load_mask_on_volume_grid(mask_path, volume_path)
        simulator = CalciumMotionSimulator(volume, voxel_spacing, calcium_mask)
        tracks = profile.make_tracks(simulator, n_angles, pseed)
        sim.simulate_one_motion(volume, voxel_spacing, mask_path, tracks, theta,
                                output_dir=out_patient, output_name=name_idx,
                                add_poisson_noise=add_poisson_noise, verbose=verbose,
                                simulator=simulator)
        n_gen += 1

    return {"patient_id": patient_id, "status": "ok", "generated": n_gen,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description="Batch motion simulation over all patients.")
    ap.add_argument("--volume_dir", required=True, help="Dir of gated volumes <id>.nii")
    ap.add_argument("--mask_dir", required=True, help="Dir of calcium masks mask_<id>.nii.gz")
    ap.add_argument("--output_dir", required=True, help="Output base dir (<output>/<id>/reconstructions)")
    ap.add_argument("--angles", type=int, default=180)
    ap.add_argument("--random-angles", action="store_true",
                    help=f"Draw each motion's view count randomly from {list(sim.ANGLE_CHOICES)}")
    ap.add_argument("--seed", type=int, default=42, help="Base seed; per-patient seed = seed + id")
    ap.add_argument("--config", default=sim.DEFAULT_CONFIG)
    ap.add_argument("--patients", type=int, nargs="*", default=None,
                    help="Explicit patient ids (default: auto-discover all)")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate even if recons exist")
    ap.add_argument("--poisson", action="store_true")
    ap.add_argument("--results_json", default=None, help="Optional path to write a run summary JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    patients = args.patients if args.patients else discover_patients(args.volume_dir, args.mask_dir)
    print(f"Patients to process: {len(patients)}")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    t_start = time.time()
    for i, pid in enumerate(patients):
        print(f"\n{'='*60}\n[{i+1}/{len(patients)}] patient {pid}\n{'='*60}")
        try:
            r = process_patient(pid, args.volume_dir, args.mask_dir, args.output_dir,
                                args.angles, args.random_angles, args.seed, args.config,
                                args.overwrite, args.poisson, not args.quiet)
        except Exception as e:
            r = {"patient_id": pid, "status": "failed", "error": str(e), "generated": 0}
            print(f"  FAILED: {e}")
        results.append(r)
        print(f"  -> {r['status']} ({r.get('generated', 0)} generated, {r.get('seconds', 0)}s)")

    total_gen = sum(r.get("generated", 0) for r in results)
    n_ok = sum(1 for r in results if r["status"] in ("ok", "skipped_complete"))
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'='*60}\nDONE: {n_ok}/{len(patients)} patients ok, {n_fail} failed, "
          f"{total_gen} recons generated in {(time.time()-t_start)/3600:.2f} h\n{'='*60}")

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump({"total_patients": len(patients), "ok": n_ok, "failed": n_fail,
                       "recons_generated": total_gen, "results": results}, f, indent=1)
        print(f"summary -> {args.results_json}")


if __name__ == "__main__":
    main()
