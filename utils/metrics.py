import torch
import math
import numpy as np

#####=========Loss and metric utilities.=========#####

def RMSE(output, target):
    valid_mask = target > 0.1
    output_mm = output[valid_mask].float()
    target_mm = target[valid_mask].float()

    abs_diff =  1e3 * (output_mm - target_mm).abs()
    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)

    return rmse


def MAE(output, target):
    valid_mask = target > 0.1
    output_mm = 1e3 * output[valid_mask]
    target_mm = 1e3 * target[valid_mask]

    abs_diff = (output_mm - target_mm).abs()
    mae = float(abs_diff.mean())

    return mae


def iRMSE(output, target):
    valid_mask = target > 0.1
    inv_output_km = (1e-3 * output[valid_mask]) ** (-1)
    inv_target_km = (1e-3 * target[valid_mask]) ** (-1)

    abs_inv_diff = (inv_output_km - inv_target_km).abs()
    irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())

    return irmse


def iMAE(output, target):
    valid_mask = target > 0.1
    inv_output_km = (1e-3 * output[valid_mask]) ** (-1)
    inv_target_km = (1e-3 * target[valid_mask]) ** (-1)

    abs_inv_diff = (inv_output_km - inv_target_km).abs()
    imae = float(abs_inv_diff.mean())

    return imae

def masked_rmse(output, target, mask):
    """Compute masked RMSE."""
    if not mask.any():
        return float('nan')
    output_masked = output[mask]
    target_masked = target[mask]
    diff = (output_masked - target_masked) * 1e3  # meters to millimeters
    return math.sqrt(torch.mean(diff  **  2).item())

def masked_mae(output, target, mask):
    """Compute masked MAE."""
    if not mask.any():
        return float('nan')
    output_masked = output[mask]
    target_masked = target[mask]
    return torch.abs((output_masked - target_masked) * 1e3).mean().item()


def masked_irmse(output, target, mask):
    """Compute masked iRMSE."""
    if not mask.any():
        return float('nan')
    output_masked = output[mask]
    target_masked = target[mask]

    # Convert to inverse kilometers.
    inv_output_km = (1e-3 * output_masked)  **  (-1)
    inv_target_km = (1e-3 * target_masked)  **  (-1)

    # Compute RMSE of inverse-depth differences.
    inv_diff = inv_output_km - inv_target_km
    return math.sqrt(torch.mean(inv_diff  **  2).item())

def masked_imae(output, target, mask):
    """Compute masked iMAE."""
    if not mask.any():
        return float('nan')
    output_masked = output[mask]
    target_masked = target[mask]

    # Convert to inverse kilometers.
    inv_output_km = (1e-3 * output_masked)  **  (-1)
    inv_target_km = (1e-3 * target_masked)  **  (-1)

    # Compute MAE of inverse-depth differences.
    inv_diff = inv_output_km - inv_target_km
    return torch.abs(inv_diff).mean().item()

def evaluate_metrics(pred, target):
    """
    :param pred:   Prediction in meters(B, H, W)
    :param target: Ground truth in meters(B, H, W)
    """
    metrics = {}
    device = target.device

    # Base valid mask filtering depths below 0.1 m.
    valid_mask = target > 0.1

    # Range-specific masks.
    near_mask = valid_mask & (target <= 20)  # 0.1-20 m
    far_mask = valid_mask & (target > 20) & (target <= 100)  # 20-100 m

    # Global metrics.
    metrics['RMSE'] = masked_rmse(pred, target, valid_mask)
    metrics['MAE'] = masked_mae(pred, target, valid_mask)
    metrics['iRMSE'] = iRMSE(pred, target)
    metrics['iMAE'] = iMAE(pred, target)

    # Range-specific metrics.
    metrics['near_RMSE'] = masked_rmse(pred, target, near_mask)
    metrics['near_MAE'] = masked_mae(pred, target, near_mask)
    metrics['near_iRMSE'] = masked_irmse(pred, target, near_mask)  # near-range mask
    metrics['near_iMAE'] = masked_imae(pred, target, near_mask)  # near-range mask


    metrics['far_RMSE'] = masked_rmse(pred, target, far_mask)
    metrics['far_MAE'] = masked_mae(pred, target, far_mask)
    metrics['far_iRMSE'] = masked_irmse(pred, target, far_mask)  # far-range mask
    metrics['far_iMAE'] = masked_imae(pred, target, far_mask)  # far-range mask

    # num_near = near_mask.sum().item()
    # num_far = far_mask.sum().item()
    # num_total = valid_mask.sum().item()
    #
    # near_sum = metrics['near_RMSE'] * num_near
    # far_sum = metrics['far_RMSE'] * num_far
    # total_rmse_estimate = (near_sum + far_sum) / num_total


    # —— 4. Compute relative depth metrics on valid pixels.
    # Flatten predictions and targets and keep valid pixels only.
    pred_valid   = pred[valid_mask]
    target_valid = target[valid_mask]

    # Difference.
    diff = pred_valid - target_valid                # shape: (K,); unit: meters

    # 4.1 abs_rel = mean( |d_pred - d_true| / d_true )
    abs_rel = torch.mean(torch.abs(diff) / target_valid).item()
    metrics['abs_rel'] = abs_rel

    # 4.2 sq_rel = mean( (d_pred - d_true)^2 / d_true )
    sq_rel = torch.mean((diff ** 2) / target_valid).item()
    metrics['sq_rel'] = sq_rel

    # 4.3 RMS = sqrt( mean( (d_pred - d_true)^2 ) )
    rms = math.sqrt(torch.mean(diff ** 2).item())
    metrics['rms'] = rms

    # 4.4 log10 error (log-rms) = sqrt( mean( (log10 d_pred - log10 d_true)^2 ) )
    # Clamp the minimum value for numerical stability before log10.
    eps = 1e-6
    log_diff = torch.log10(pred_valid.clamp(min=eps)) - torch.log10(target_valid.clamp(min=eps))
    log10_err = math.sqrt(torch.mean(log_diff ** 2).item())
    metrics['log10'] = log10_err

    # Delta accuracy: ratio=max(d_pred/d_true, d_true/d_pred), counting pixels with ratio < 1.25^i.
    #    i = 1, 2, 3
    # Compute ratio first = max(d_pred / d_true, d_true / d_pred)
    #    Add a small epsilon to avoid division by zero.
    ratio = torch.max(
        (pred_valid.clamp(min=eps) / target_valid.clamp(min=eps)),
        (target_valid.clamp(min=eps) / pred_valid.clamp(min=eps))
    )

    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()
    metrics['delta1'] = delta1
    metrics['delta2'] = delta2
    metrics['delta3'] = delta3

    return metrics