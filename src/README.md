# Training and Evaluation Scripts

Trains and evaluates the model used to suppress coronary-calcium motion artifacts in gated cardiac CT, and scores the results by Agatston calcium score. Operates on the calcium patch datasets produced upstream (see the `motion_simulation` scripts for how the motion-corrupted data is made).

## Model code

The Brownian Bridge Diffusion Model (BBDM) package is **not included** in this repo. Before running `train.py` or `evaluate.py`, obtain the BBDM model code from the upstream [BBDM repository](https://github.com/xuekt98/BBDM) and place it under `src/model/` so that `from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel` resolves.

## Pipeline

The five scripts form one ordered pipeline:

| Stage | Script | Notes |
|---|---|---|
| 1 | `generate_voxel_size_map.py` | go through per-patient NIfTI and create `voxel_size_map.json` for Agatston score calculations |
| 2 | `train.py` | training script outputting model checkpoints `epoch_*.pt` |
| 3 | `evaluate.py` | generates predictions in `*_eval_results.h5` using trained checkpoint|
| 4 | `compute_agatston_scores.py` | takes in `*_eval_results.h5`predictions to compute agatston scores and output `agatston_scores.csv` and `evaluation_summary.txt` |


## Usage

# 1. build the per-patient voxel-size map
python generate_voxel_size_map.py \
    --nifti_dir  /path/gated_nifti --h5_dir dataset/val \
    --output_file voxel_size_map.json

# 2. train the model
python train.py \
    --train_dir dataset/train --val_dir dataset/val \
    --epochs <num_epochs> --batch_size <bs> --lr <lr> --ckpt_dir runs/lambda_10 --gpu 0

# 3. sample predictions from a checkpoint  (config defaults to configs/bbdm_25d_config.yaml)
python evaluate.py \
    --checkpoint_path runs/lambda_10/epoch_020.pt \
    --val_dir dataset/val --output_dir eval_out \
    --batch_size 20 --sampling_steps 20 --gpu 0

# 4. Agatston scoring of the predictions
python compute_agatston_scores.py \
    --input_dir eval_out --output_dir eval_out/scores \
    --voxel_size_map voxel_size_map.json
```