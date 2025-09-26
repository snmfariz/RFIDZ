import numpy as np
from sklearn.metrics import average_precision_score

def compute_snr(spectrum, burst_start=None, burst_end=None):
    """Compute per-time S/N using a baseline that excludes the burst window."""
    freq_avg = np.nanmean(spectrum, axis=0)
    if burst_start is not None and burst_end is not None:
        baseline = freq_avg.copy()
        baseline[burst_start:burst_end] = np.nan
    else:
        baseline = freq_avg
    return (freq_avg - np.nanmean(baseline)) / np.nanstd(baseline)

def peak_snr(spectrum, start_time, end_time):
    """Return the maximum S/N within a window."""
    snr_ts = compute_snr(spectrum, start_time, end_time)
    if np.all(~np.isfinite(snr_ts)):
        return np.nan
    return float(np.nanmax(snr_ts))

def channel_mask_from_pixel(sim_mask):
    """Expand a per-pixel mask into a per-channel mask."""
    ch_any = (np.nan_to_num(sim_mask) > 0).any(axis=1)
    return np.repeat(ch_any[:, None], sim_mask.shape[1], axis=1).astype(sim_mask.dtype)

def zap_with_mask(x, mask):
    """Replace masked pixels with NaNs."""
    return np.where(mask, np.nan, x)

def pr_auc(y_true_mask, y_pred_mask):
    """Compute average precision for pixel masks."""
    y_true = np.ravel(np.nan_to_num(y_true_mask).astype(np.uint8))
    y_pred = np.ravel(np.nan_to_num(y_pred_mask).astype(np.float32))
    if y_true.sum() == 0:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_pred))
    except Exception:
        return float("nan")

def robust_threshold(pred_logits, z=5.0):
    """Create a binary mask via a median/MAD z-threshold."""
    x = pred_logits.ravel()
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    sigma = 1.4826 * mad
    thr = med + z * sigma
    return (pred_logits >= thr).astype(np.float32), thr
