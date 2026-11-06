import torch
import torch.nn.functional as F

def window_to_hu(x_windowed, lo=-200.0, hi=800.0):
    """Convert windowed [0,1] values back to HU units"""
    return x_windowed * (hi - lo) + lo

@torch.no_grad()
def voxel_metrics_from_spacing(spacing_xyz):
    """Calculate voxel volume and area from spacing"""
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    voxel_volume = sx * sy * sz  # mm^3
    voxel_area = sx * sy  # mm^2
    return voxel_volume, voxel_area

def soft_cac_prob_from_hu(hu, kappa=60.0, hu_thresh=130.0):
    """Compute soft calcium probability from HU values"""
    return torch.sigmoid((hu - hu_thresh) / kappa)

def surrogate_volume_score(hu, spacing_xyz, roi_mask=None, kappa=60.0, hu_thresh=130.0):
    """Returns S_vol (B,) in mm^3 (differentiable)"""
    p = soft_cac_prob_from_hu(hu, kappa=kappa, hu_thresh=hu_thresh)  # (B,D,H,W)
    if roi_mask is not None:
        p = p * roi_mask

    voxel_vol, _ = voxel_metrics_from_spacing(spacing_xyz)
    S_vol = (p * voxel_vol).sum(dim=(1, 2, 3))
    return S_vol

def calcium_consistency_loss(hu_pred, hu_target, spacing_xyz, roi_mask=None,
                           kappa=60.0, hu_thresh=130.0, use_log1p=True, delta=50.0, reduction='mean'):
    """
    Calcium score consistency loss between predicted and target HU values.
    
    Args:
        hu_pred: Predicted HU values (B,1,D,H,W)
        hu_target: Target HU values (B,1,D,H,W) 
        spacing_xyz: Voxel spacing (sx, sy, sz) in mm
        roi_mask: Optional ROI mask (B,1,D,H,W)
        kappa: Soft threshold parameter
        hu_thresh: HU threshold for calcium detection
        use_log1p: Whether to use log1p transformation
        delta: SmoothL1 beta parameter
        reduction: Reduction method ('mean', 'sum', 'none')
    
    Returns:
        Loss value (scalar if reduction='mean'/'sum', tensor if reduction='none')
    """
    S_vol_pred = surrogate_volume_score(hu_pred, spacing_xyz, roi_mask, kappa, hu_thresh)
    S_vol_target = surrogate_volume_score(hu_target, spacing_xyz, roi_mask, kappa, hu_thresh)
    
    if use_log1p:
        S_vol_pred = torch.log1p(S_vol_pred)
        S_vol_target = torch.log1p(S_vol_target)
    
    return F.smooth_l1_loss(S_vol_pred, S_vol_target, beta=delta, reduction=reduction) 