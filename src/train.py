import argparse
import time
import math
import sys
import os
import random
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
DEFAULT_CONFIG = os.path.join(_HERE, "configs", "bbdm_25d_config.yaml")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import logging
import numpy as np
from typing import List, Tuple, Dict
from enum import Enum
from omegaconf import OmegaConf
import wandb

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from data.patched_triplets_dataset import PatchedTripletsDataset_25D
from losses.calcium_consistency_loss import calcium_consistency_loss, window_to_hu
from utils.voxel_spacing_utils import load_voxel_spacing_map, get_spacing_for_patient
from utils.visualize_training_progress import save_patch_prediction_25d
from utils.misc import denormalize_to_hu, compute_agatston_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model, opt, device, train_loader, val_loader, epochs,
                grad_clip=1.0, val_steps=200, add_noise=False,
                ckpt_dir='runs/ckpts', validate_every=1, max_val_batches=None,
                selected_validation_patches_indices=None, save_predictions=True, use_wandb=False,
                no_sampling_progress=False,
                use_calcium_loss=False, calcium_loss_weight=0.1,
                voxel_spacing_map=None,
                start_epoch=1,
                hu_min: float = -200, hu_max: float = 800):

    os.makedirs(ckpt_dir, exist_ok=True)

    total_batches_per_epoch = len(train_loader)
    logger.info(f"Processing all {total_batches_per_epoch} batches per epoch")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        loss_meter = AvgMeter()
        batches_processed = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", 
                          position=0, leave=False, dynamic_ncols=True)

        for batch_idx, batch_data in enumerate(progress_bar):
            opt.zero_grad(set_to_none=True)
            
            x0, y, metadata = batch_data
            x0 = x0.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    

            main_loss, log = model(x0, y, context=None)
            
            total_loss = main_loss
            # Initialize additional losses
            weighted_calcium_loss = torch.tensor(0.0, device=device)

            # Add calcium consistency loss if enabled
            if use_calcium_loss:
                if isinstance(metadata, list) and len(metadata) > 0:
                    patient_id = get_patient_id_from_metadata(metadata[0])
                else:
                    patient_id = 'unknown'
                x0_for_calcium = x0
                
                spacing_xyz = get_spacing_for_patient(voxel_spacing_map, patient_id) if voxel_spacing_map else (0.4, 0.4, 3.0)
                
                # use the predicted reconstruction from the model to calculate the calcium loss
                if 'x0_recon' in log and 't' in log:
                    xhat0 = log['x0_recon']
                    t = log['t']
                else:
                    print("WARNING: No x0_recon or t in log, using model prediction")
                    breakpoint()
                    print(log)    
                    t = torch.full((x0.shape[0],), 0, device=device, dtype=torch.long)

                # convert prediction and target to HU values for computing calcium consistency loss
                hu_pred = window_to_hu(xhat0)  
                hu_target = window_to_hu(x0_for_calcium)  
                
                calcium_loss_per_sample = calcium_consistency_loss(
                    hu_pred=hu_pred,
                    hu_target=hu_target,
                    spacing_xyz=spacing_xyz,
                    roi_mask=None, 
                    kappa=60.0,
                    hu_thresh=130.0,
                    use_log1p=True,
                    delta=50.0,
                    reduction='none'  
                )
                weight = get_auxiliary_loss_weight_schedule(
                    lambda_start=0.0, 
                    lambda_end=calcium_loss_weight, 
                    timestep=t, 
                    total_timesteps=model.num_timesteps,
                    schedule_type=ScheduleType.COSINE)
        
                weighted_calcium_loss = (weight.unsqueeze(1) * calcium_loss_per_sample).mean()
                total_loss += weighted_calcium_loss

            # debugging
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}")
                if use_calcium_loss:
                    print(f"  Average Calcium Loss: {weighted_calcium_loss.item():.6f}")
                print(f"  Total Loss: {total_loss.item():.6f}")
            
        
            total_loss.backward()
        
            
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            # Update loss meter with appropriate batch size
            batch_size = x0.size(0)
            loss_meter.update(total_loss.item(), k=batch_size)
            batches_processed += 1

        dt = time.time() - t0
        train_loss = loss_meter.avg

        progress_bar.close()
        
        print(f"[Epoch {epoch:03d}] Train loss: {train_loss:.4f}  ({dt:.1f}s)")

        if use_wandb:
            log_dict = {
                "train/loss": train_loss,
                "train/epoch_time": dt,
                "train/learning_rate": opt.param_groups[0]['lr'],
                "epoch": epoch
            }

            # Add calcium loss metrics if enabled
            if use_calcium_loss:
                log_dict.update({
                    "train/calcium_loss_weight": calcium_loss_weight,
                    "train/spacing_xyz": spacing_xyz
                })

            wandb.log(log_dict)

        checkpoint_data = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'train_loss': train_loss
        }

        if use_wandb and wandb.run is not None:
            checkpoint_data['wandb_run_id'] = wandb.run.id
        

        latest_path = os.path.join(ckpt_dir, "latest.pt")
        torch.save(checkpoint_data, latest_path)
        
    
        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint_data, epoch_path)
        
        print(f"[Epoch {epoch:03d}] Saved checkpoint to {epoch_path}")

        if save_predictions and selected_validation_patches_indices:
            print(f"[Epoch {epoch:03d}] Saving patch predictions...")
            os.makedirs(os.path.join(ckpt_dir, f'epoch_{epoch}'), exist_ok=True)
            
            model.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save_patch_prediction_25d(
                model=model,
                dataset=val_loader.dataset,
                epoch=epoch-1, 
                save_dir=os.path.join(ckpt_dir, f'epoch_{epoch}'),
                device=device,
                use_wandb=use_wandb,
                selected_patches_indices=selected_validation_patches_indices,
                decoder=None,
                show_progress=not no_sampling_progress,
            )
            
        if (validate_every is not None) and (epoch % validate_every == 0):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            val_metrics = validate_epoch(
                model, val_loader, device, epoch,
                max_batches=max_val_batches,
                voxel_spacing_map=voxel_spacing_map, hu_min=hu_min, hu_max=hu_max
            )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if use_wandb:
                wandb.log({
                    "val/agatston_mae": val_metrics['agatston_mae'],
                    "val/agatston_mae_std": val_metrics['agatston_mae_std'],
                    "epoch": epoch
                })
                
            checkpoint_data['val_agatston_mae'] = val_metrics['agatston_mae']
            checkpoint_data['val_agatston_mae_std'] = val_metrics['agatston_mae_std']
            
            if use_wandb and wandb.run is not None:
                checkpoint_data['wandb_run_id'] = wandb.run.id
                
            torch.save(checkpoint_data, latest_path)

# ------------- in-epoch validation -------------
def validate_epoch(model, val_loader, device, epoch, max_batches=None,
                   voxel_spacing_map=None, hu_min: float = -200, hu_max: float = 800)->Dict:
    model.eval()
    val_metrics = {}
    val_metrics['agatston_mae'] = 0.0
    val_metrics['agatston_mae_count'] = 0

    # Use fewer batches for faster validation
    num_batches = min(max_batches or 50, len(val_loader))
    progress_bar = tqdm(val_loader, desc=f"Validation epoch {epoch}", total=num_batches)
    
    processed_batches = 0
    agatston_errors = []
    
    with torch.no_grad():
        for b_idx, (x0, y, meta) in enumerate(progress_bar):
            if (max_batches is not None) and (b_idx >= max_batches):
                break
            x0 = x0.to(device=device, dtype=torch.float32, non_blocking=True)
            y  = y.to(device=device, dtype=torch.float32, non_blocking=True)

            # Sample clean estimate from corrupted condition
            xhat0 = model.sample(y)

            # Extract center slice (slice 1) from the triplet
            motion_free_center = x0[:, 1, :, :].cpu().numpy()  # (batch_size, H, W)
            predicted_center = xhat0[:, 1, :, :].cpu().numpy()  # (batch_size, H, W)

            # Process each sample in the batch
            for sample_idx in range(motion_free_center.shape[0]):
                # Get the center slice for this sample
                motion_free_slice = motion_free_center[sample_idx]  # (H, W)
                predicted_slice = predicted_center[sample_idx]  # (H, W)

                # Get patient-specific voxel size from metadata
                if isinstance(meta, list) and len(meta) > 0:
                    patient_id = get_patient_id_from_metadata(meta[sample_idx])
                else:
                    patient_id = 'unknown'

                # Get patient-specific voxel size
                voxel_size = get_spacing_for_patient(voxel_spacing_map, patient_id) if voxel_spacing_map else (0.625, 0.625, 2.5)

                # Convert back to HU values
                motion_free_hu = denormalize_to_hu(motion_free_slice, hu_min, hu_max)
                predicted_hu = denormalize_to_hu(predicted_slice, hu_min, hu_max)

                # Add a dummy slice dimension for Agatston score computation
                # The function expects (slices, H, W) but we only have (H, W)
                motion_free_3d = motion_free_hu[np.newaxis, :, :]  # (1, H, W)
                predicted_3d = predicted_hu[np.newaxis, :, :]  # (1, H, W)

                # Compute Agatston scores with patient-specific voxel size
                motion_free_score = compute_agatston_score(motion_free_3d, voxel_size)
                predicted_score = compute_agatston_score(predicted_3d, voxel_size)

                # Calculate absolute error
                agatston_error = abs(predicted_score - motion_free_score)
                agatston_errors.append(agatston_error)

                val_metrics['agatston_mae'] += agatston_error
                val_metrics['agatston_mae_count'] += 1

            processed_batches += 1

    # Calculate average Agatston MAE
    if val_metrics['agatston_mae_count'] > 0:
        val_metrics['agatston_mae'] /= val_metrics['agatston_mae_count']
        val_metrics['agatston_mae_std'] = np.std(agatston_errors) if agatston_errors else 0.0
    else:
        val_metrics['agatston_mae'] = float('inf')
        val_metrics['agatston_mae_std'] = 0.0

    progress_bar.close()

    return val_metrics

#-------Misc----------
class ScheduleType(Enum):
    LINEAR = 'linear'
    COSINE = 'cosine'

def get_auxiliary_loss_weight_schedule(lambda_start: float, lambda_end: float, 
timestep: int, total_timesteps: int, schedule_type: ScheduleType = ScheduleType.LINEAR) -> float:
    """
    Get the auxiliary loss (could be calcium consistency loss, calcium dice loss, etc.) 
    weight schedule based on the timestep sampled in the diffusion process.
    Args:
        lambda_start: float, the starting weight of the auxiliary loss
        lambda_end: float, the ending weight of the auxiliary loss
        timestep: int, the timestep sampled in the diffusion process
        total_timesteps: int, the total number of timesteps in the diffusion process
        schedule_type: str, the schedule type of the auxiliary loss weight
    Returns:
        float, the auxiliary loss weight
    """
    if schedule_type == ScheduleType.LINEAR:
        return lambda_start + (lambda_end - lambda_start) * timestep / total_timesteps
    elif schedule_type == ScheduleType.COSINE:
        return lambda_start + (lambda_end - lambda_start) * \
        (1 - torch.cos(torch.pi * timestep / total_timesteps)) / 2
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")


def get_patient_id_from_metadata(metadata) -> str:
    """Extract patient_id from metadata (handles both dict and tensor)"""
    if isinstance(metadata, dict):
        return metadata.get('patient_id', 'unknown')
    elif isinstance(metadata, torch.Tensor):
        return str(int(metadata[1].item()))
    else:
        return 'unknown'

def get_sampler_type_from_metadata(metadata) -> str:
    """Extract sampler_type from metadata (handles both dict and tensor)"""
    if isinstance(metadata, dict):
        return metadata.get('sampler_type', 'random')
    elif isinstance(metadata, torch.Tensor):
        return 'calcium' if metadata[4].item() > 0.5 else 'random'
    else:
        return 'random'

def sample_validation_patches(
    dataset: PatchedTripletsDataset_25D,
    num_patches: int = 10,
    val_indices: List[int] = []
) -> List[Tuple[int, Dict]]:
    """Sample random validation patches from the dataset"""
    if len(val_indices) == 0:
        logger.warning("No validation indices provided, using all dataset indices")
        val_indices = list(range(len(dataset)))
    
    # Find all calcium-sampled patches in validation data
    calcium_patches = []
    for idx in val_indices:
        metadata = dataset.metadata_list[idx]
        if metadata['sampler_type'] == 'calcium':
            calcium_patches.append((idx, metadata))
    
    if not calcium_patches:
        logger.warning("No calcium-sampled patches found in the validation dataset")
        return []
    
    # Log the number of calcium patches found
    logger.info(f"Found {len(calcium_patches)} calcium patches in the validation dataset")
    
    # Randomly select patches
    selected_patches = random.sample(calcium_patches, min(num_patches, len(calcium_patches)))
    
    # Log the selected patches
    for i, (idx, metadata) in enumerate(selected_patches):
        logger.info(f"\nSelected Validation Patch {i}:")
        logger.info(f"Patient ID: {metadata['patient_id']}")
        logger.info(f"Motion ID: {metadata['motion_id']}")
        logger.info(f"Patch Index: {metadata['patch_idx']}")
        logger.info(f"Patch Center: {metadata['patch_center']}")
        logger.info(f"Sampler Type: {metadata['sampler_type']}")
    
    return selected_patches



class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0
    def update(self, val, k=1):
        self.sum += float(val) * k
        self.n += k
    @property
    def avg(self):
        return self.sum / max(1, self.n)

def set_seed(seed=1337):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)      

  
# Training options parsing and driver
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory of training patch .h5 files (clean-HU)')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Directory of validation patch .h5 files (clean-HU)')
    parser.add_argument('--val_steps', type=int, default=50)  
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--ckpt_dir', type=str,required=True)
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--max_val_batches', type=int, default=None,
                        help='limit val batches for speed; None = all')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save patch predictions during training')
    parser.add_argument('--num_monitoring_patches', type=int, default=5, 
                        help='Number of validation patches to monitor')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='prodm-25d',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (optional)')
    parser.add_argument('--num_workers', type=int, default=2,  
                        help='Number of DataLoader workers (reduce to 1 or 0 if running out of RAM)')
    parser.add_argument('--cache_size', type=int, default=200,  
                        help='Dataset cache size per worker (reduce to 100 or 0 if running out of RAM)')
    parser.add_argument('--val_cache_size', type=int, default=None,
                        help='Cache size for validation dataset (defaults to cache_size/4 if not set)')
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Disable pin_memory in DataLoader to reduce RAM usage')
    parser.add_argument('--no_sampling_progress', action='store_true',
                        help='Disable progress bars during sampling/inference')
    parser.add_argument('--disable_preprocessing', action='store_true',
                        help='Disable all preprocessing for datasets that are already normalized')
    parser.add_argument('--use_calcium_loss', action='store_true',
                        help='Enable calcium consistency loss')
    parser.add_argument('--calcium_loss_weight', type=float, default=20.0,
                        help='Weight for calcium consistency loss')
    parser.add_argument('--voxel_size_map', type=str, default='voxel_size_map.json',
                        help='Path to JSON file containing voxel spacing map')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., runs/ckpts/latest.pt or runs/ckpts/epoch_050.pt)')
    parser.add_argument('--resume_wandb_id', type=str, default=None,
                        help='Wandb run ID to resume (will be auto-detected from checkpoint if not provided)')
    parser.add_argument('--hu_min', type=float, default=-200,
                        help='Minimum HU value for denormalization (default: -200)')
    parser.add_argument('--hu_max', type=float, default=800,
                        help='Maximum HU value for denormalization (default: 800)')
    args = parser.parse_args()

    set_seed(args.seed)

    if args.use_wandb:
        # Determine if we should resume a run
        resume_run = False
        wandb_run_id = None
        if args.resume_wandb_id is not None:
            wandb_run_id = args.resume_wandb_id
            resume_run = True
            print(f"Resuming wandb run with ID: {wandb_run_id}")
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            resume="allow" if resume_run else None,
            id=wandb_run_id if resume_run else None,
            config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "use_calcium_loss": args.use_calcium_loss,
            "calcium_loss_weight": args.calcium_loss_weight,
            "voxel_size_map": args.voxel_size_map,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "validate_every": args.validate_every,
                "num_monitoring_patches": args.num_monitoring_patches,
                "seed": args.seed,
                "disable_preprocessing": args.disable_preprocessing,
            }
        )
        print(f"Initialized wandb run: {wandb.run.name if wandb.run else 'unnamed'}")
        if resume_run:
            print(f"Successfully resumed wandb run: {wandb.run.id}")

    # Memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # --- data ---
    if args.disable_preprocessing:
        print("WARNING: Preprocessing is disabled - assuming data is already normalized")
    else:
        print("Preprocessing is enabled - normalizing data")
    print(f"Loading training dataset from {args.train_dir}")
    train_dataset = PatchedTripletsDataset_25D(
        h5_dir=args.train_dir,
        cache_size=args.cache_size
    )
    print(f"Loading validation dataset from {args.val_dir}")
    # Use smaller cache for validation (used less frequently)
    val_cache_size = args.val_cache_size if args.val_cache_size is not None else max(1, args.cache_size // 4)
    val_dataset = PatchedTripletsDataset_25D(
        h5_dir=args.val_dir,
        cache_size=val_cache_size
    )
    print(f"Creating data loaders")
    print(f"Memory settings:")
    print(f"  - num_workers: {args.num_workers} (each worker caches data)")
    print(f"  - train cache_size: {args.cache_size} patches per worker")
    print(f"  - val cache_size: {val_cache_size} patches")
    print(f"  - pin_memory: {not args.no_pin_memory}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers, 
        persistent_workers=False,  
        prefetch_factor=2 if args.num_workers > 0 else None, 
    )
    print(f"Training data loader created")
    val_loader   = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  
        pin_memory=not args.no_pin_memory, 
        persistent_workers=False, 
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"Validation data loader created")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")


    # --- setup validation patches for monitoring ---
    val_indices = list(range(len(val_dataset)))

    # Select validation patches for monitoring
    selected_validation_patches_indices = []
    if args.num_monitoring_patches > 0:
        # Get calcium patches from the lightweight dataset
        calcium_patches = val_dataset.get_calcium_patches()
        if len(calcium_patches) > 0:
            num_patches = min(args.num_monitoring_patches, len(calcium_patches))
            selected_indices = random.sample(range(len(calcium_patches)), num_patches)
            selected_validation_patches_indices = [[idx] for idx in selected_indices]
        else:
            # Fallback to random patches if no calcium patches found
            total_patches = len(val_dataset)
            num_patches = min(args.num_monitoring_patches, total_patches)
            selected_indices = random.sample(range(total_patches), num_patches)
            selected_validation_patches_indices = [[idx] for idx in selected_indices]

    
    # Load model with config
    config = OmegaConf.load(DEFAULT_CONFIG)
    model = BrownianBridgeModel(config.model).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load checkpoint if resuming training
    start_epoch = 1
    wandb_run_id = None
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Validate checkpoint structure
            required_keys = ['epoch', 'model', 'opt']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")
            
            # Load model state
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded model state from epoch {checkpoint['epoch']}")
            
            # Load optimizer state
            opt.load_state_dict(checkpoint['opt'])
            print(f"Loaded optimizer state from epoch {checkpoint['epoch']}")

            # Set starting epoch
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
            
            # Extract wandb run ID if available
            if 'wandb_run_id' in checkpoint:
                wandb_run_id = checkpoint['wandb_run_id']
                print(f"Found wandb run ID in checkpoint: {wandb_run_id}")
            
            # Log previous training loss if available
            if 'train_loss' in checkpoint:
                print(f"Previous training loss: {checkpoint['train_loss']:.4f}")
            if 'val_mse_loss' in checkpoint:
                print(f"Previous validation loss: {checkpoint['val_mse_loss']:.4f}")
            
            print(f"Resuming training configuration:")
            print(f"  - Starting epoch: {start_epoch}")
            print(f"  - Total epochs: {args.epochs}")
            print(f"  - Learning rate: {args.lr}")
            print(f"  - Batch size: {args.batch_size}")
            print(f"  - Checkpoint directory: {args.ckpt_dir}")
        else:
            print(f"Warning: Checkpoint file {args.resume} not found. Starting training from scratch.")
            start_epoch = 1


    voxel_spacing_map = load_voxel_spacing_map(args.voxel_size_map)
    if not voxel_spacing_map:
        print("Warning: Could not load voxel spacing map. Using default spacing for all patients.")
    else:
        print(f"Loaded voxel spacing map with {len(voxel_spacing_map)} patients")
    
    # --- train ---
    train_model(
        model, opt, device, train_loader, val_loader, args.epochs,
        grad_clip=args.grad_clip,
        val_steps=args.val_steps,
        add_noise=False,               
        ckpt_dir=args.ckpt_dir,
        validate_every=args.validate_every,
        max_val_batches=args.max_val_batches,
        selected_validation_patches_indices=selected_validation_patches_indices,
        save_predictions=args.save_predictions,
        use_wandb=args.use_wandb,
        no_sampling_progress=args.no_sampling_progress,
        use_calcium_loss=args.use_calcium_loss,
        calcium_loss_weight=args.calcium_loss_weight,
        voxel_spacing_map=voxel_spacing_map,
        start_epoch=start_epoch,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
    )
