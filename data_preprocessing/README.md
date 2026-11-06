# Data preprocessing — calcium patch datasets

Turns the paired motion-corrupted / motion-free CT volumes (from `motion_simulation`)
into the HU-normalized patch datasets consumed by `src`.

## Pipeline

| Stage | Script | Notes |
|---|---|---|
| 1 | `extract_hu_patches.py` | takes recon + gated + mask volumes and output HU-normalized patch `.h5` |
| 2 | `create_optimized_validation_set.py` | takes val patches and produce non-redundant shortlist of patches, include random non-calcium patches and unique calcium patches separated by a specified distance |

Output of stage 1 creates the dataset for training directly, the val set is
first passed through stage 2 to produce the shortlist used for evaluation.


## Usage
```bash
python extract_hu_patches.py \
    --motion_corrupted_dir /path/motion_sim_results \   # <id>/reconstructions/recon_*.nii.gz
    --motion_free_dir      /path/gated_nifti_hu \        # <id>.nii
    --mask_dir             /path/mask_per_artery \       # mask_<id>.nii.gz
    --output_dir           /path/gated_patches/val \
    --ignore_list_file     ignore.txt \
    --patch_size 64 64 16 --max_calcium_patches 45 --max_random_patches 5
# legacy raw-intensity inputs:  add  --input_is_raw
```
Patch values are stored in **[0, 1]** (HU-window normalized) — exactly what
`src` (training and `evaluate.py`, whose `denorm_to_hu` is
`x*1000 - 200`) expects.

## `create_optimized_validation_set.py`

Reduces each validation scan's patches to a non-redundant shortlist, producing a few random
patches plus spatially-distant calcium patches to keep evaluation fast and
representative. 

```bash
python create_optimized_validation_set.py \
    --input_dir  /path/gated_patches/val \
    --output_dir /path/gated_patches/optimized_val
```
