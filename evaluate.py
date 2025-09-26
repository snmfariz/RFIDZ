import numpy as np
from time import time
from types import SimpleNamespace
from onnx_api import ONNX_API
from simulate import ground_truth, noise
from utils import peak_snr, channel_mask_from_pixel, zap_with_mask, pr_auc, robust_threshold
from plotting import megaplot

def evaluate(version_denoiser, args: SimpleNamespace, n_samples=1000, batch_size=32, seed=None, n_examples_to_plot=0):
    """Run simulations, score methods, and summarize metrics."""
    t_start_all = time()
    if seed is not None:
        np.random.seed(seed)
    api_deno = ONNX_API(version_denoiser)
    rows = {
        "Clean ground truth":            {"snr": [], "ap": []},
        "Noisy spectra":                 {"snr": [], "ap": []},
        "Channel zapping (GT)":          {"snr": [], "ap": []},
        "Mask zapping (GT)":             {"snr": [], "ap": []},
        "Channel zapping (model)":       {"snr": [], "ap": []},
        "Mask zapping (model)":          {"snr": [], "ap": []},
        "Model denoised spectra":        {"snr": [], "ap": []},
    }
    batch_idx = 0
    for s in range(0, n_samples, batch_size):
        e = min(s + batch_size, n_samples)
        B = e - s
        batch_idx += 1
        clean_list, noisy_list, sim_mask_list = [], [], []
        start_times, end_times = [], []
        for _ in range(B):
            clean, start_t, end_t, bg_noise = ground_truth(args)
            noisy, sim_mask = noise(clean, args, bg_noise)
            clean_list.append(clean); noisy_list.append(noisy); sim_mask_list.append(sim_mask.astype(np.float32))
            start_times.append(start_t); end_times.append(end_t)
        noisy_batch = np.stack(noisy_list, axis=0)
        t0 = time()
        denoised_batch = api_deno.denoise(noisy_batch, batch_size=batch_size)
        t1 = time()

        dt = max(t1 - t0, 1e-9)
        ips = B / dt
        spi = dt / B
        print(f"[eval] batch {batch_idx:02d}: {B} samples | {dt:.3f}s | {ips:.2f} items/s | {spi:.4f} s/item")
        
        if denoised_batch.ndim == 2:
            denoised_batch = denoised_batch[None, ...]
        denoised_batch = denoised_batch.astype(np.float32, copy=False)
        for b in range(B):
            clean    = clean_list[b]
            noisy    = noisy_list[b]
            sim_mask = sim_mask_list[b]
            start_t  = start_times[b]
            end_t    = end_times[b]
            denoised = denoised_batch[b].copy()
            denoised -= np.nanmin(denoised)
            pred_logits = noisy - denoised
            pred_mask, thr = robust_threshold(pred_logits)
            channel_pred_mask = pred_mask.copy()
            ch_frac = np.mean(pred_mask, axis=1)
            channel_pred_mask[ch_frac > 0.5, :] = 1.0
            gt_channel_mask = channel_mask_from_pixel(sim_mask)
            residual = sim_mask - pred_mask
            zapped_gt_mask     = zap_with_mask(noisy, sim_mask)
            zapped_gt_channel  = zap_with_mask(noisy, gt_channel_mask)
            zapped_pred_mask   = zap_with_mask(noisy, pred_mask)
            zapped_pred_chwise = zap_with_mask(noisy, channel_pred_mask)
            if b < n_examples_to_plot and s == 0:
                megaplot(
                    obs=noisy,
                    sim_mask=sim_mask,
                    pred_mask=pred_mask,
                    denoised=denoised,
                    residual=residual,
                    zapped_sim=zapped_gt_mask,
                    zapped_pred=zapped_pred_mask,
                    zapped_pred_channel_wise=zapped_pred_chwise,
                    injected_frb=clean,
                    start_time=start_t,
                    end_time=end_t,
                )
            rows["Clean ground truth"]["snr"].append(peak_snr(clean, start_t, end_t))
            rows["Noisy spectra"]["snr"].append(peak_snr(noisy, start_t, end_t))
            rows["Channel zapping (GT)"]["snr"].append(peak_snr(zapped_gt_channel, start_t, end_t))
            rows["Mask zapping (GT)"]["snr"].append(peak_snr(zapped_gt_mask, start_t, end_t))
            rows["Channel zapping (model)"]["snr"].append(peak_snr(zapped_pred_chwise, start_t, end_t))
            rows["Mask zapping (model)"]["snr"].append(peak_snr(zapped_pred_mask, start_t, end_t))
            rows["Model denoised spectra"]["snr"].append(peak_snr(denoised, start_t, end_t))
            ap_ch  = pr_auc(gt_channel_mask, channel_pred_mask)
            ap_pix = pr_auc(sim_mask, pred_mask)
            rows["Channel zapping (model)"]["ap"].append(ap_ch)
            rows["Mask zapping (model)"]["ap"].append(ap_pix)
    summary = []
    for method, vals in rows.items():
        snr_arr = np.array(vals["snr"], dtype=float)
        snr_mean = float(np.nanmean(snr_arr)) if snr_arr.size else None
        snr_med  = float(np.nanmedian(snr_arr)) if snr_arr.size else None
        ap_arr = np.array(vals["ap"], dtype=float) if vals["ap"] else None
        ap_mean = float(np.nanmean(ap_arr)) if ap_arr is not None and ap_arr.size else None
        summary.append({"method": method, "snr_mean": snr_mean, "snr_median": snr_med, "ap_mean": ap_mean})
    duration = time() - t_start_all
    return summary, rows, duration
