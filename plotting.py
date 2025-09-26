import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from utils import compute_snr


def megaplot(
    obs,
    sim_mask,
    pred_mask,
    denoised,
    residual,
    zapped_sim,
    zapped_pred,
    zapped_pred_channel_wise,
    injected_frb,
    start_time,
    end_time,
    ap_pix=None,
    ap_ch=None,
    figsize_inches=(8, 12),
    dpi=200,
    show_snr_insets=True,
):
    """Render a multi-panel diagnostic plot for one example."""

    def _safe_snr(data):
        try:
            s = compute_snr(data, start_time, end_time)
            return np.asarray(s, dtype=float)
        except Exception:
            return None

    def _snr_str(data):
        s = _safe_snr(data)
        if s is None:
            return "N/A"
        finite = np.isfinite(s)
        if not np.any(finite):
            return "N/A"
        return f"{np.nanmax(s[finite]):.1f}"

    def _title_with_snr(base, data):
        return f"{base}\nFRB S/N={_snr_str(data)}"

    def _add_snr_inset(ax, snr, ymin, ymax):
        if not show_snr_insets or snr is None:
            return

        axins = ax.inset_axes([0, 1, 1, 0.3])

        x = np.arange(len(snr))
        axins.plot(x, snr, lw=0.8)

        axins.set_xlim(0, len(snr) - 1)
        axins.set_ylim(ymin, ymax)

        axins.axvspan(start_time, end_time, alpha=0.2)

        axins.set_xticks([])
        axins.tick_params(labelsize=7)
        axins.set_ylabel("S/N", fontsize=7)

    def _imshow_ds(ax, data, title, show_y=False, show_x=True):
        if data is not None and np.size(data) > 0:
            vmin, vmax = np.nanpercentile(data, (1, 99))
        else:
            vmin, vmax = None, None

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=[0, data.shape[1], 0, data.shape[0]],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax.set_title(title, fontsize=8)

        if show_y:
            ax.set_ylabel("Frequency", fontsize=7)
        else:
            ax.set_yticklabels([])

        if show_x:
            ax.set_xlabel("Time", fontsize=7)
        else:
            ax.set_xticklabels([])

        ax.tick_params(labelsize=7)
        return im

    def _blank(ax):
        ax.axis("off")

    def _mask_residual(sim, pred):
        # +1 = missed (FN), -1 = false positive (FP), 0 = correct
        return np.where(
            sim == 1,
            np.where(pred == 1, 0, 1),
            np.where(pred == 1, -1, 0),
        ).astype(np.int8)

    # Channel-wise masks from pixel masks
    sim_ch_mask_1d = (np.nanmax(sim_mask, axis=1) > 0).astype(np.uint8)
    sim_ch_mask = np.repeat(sim_ch_mask_1d[:, None], obs.shape[1], axis=1)
    zapped_sim_channel_wise = np.where(sim_ch_mask == 1, np.nan, obs)

    pred_ch_mask_1d = (np.nanmax(pred_mask, axis=1) > 0).astype(np.uint8)
    pred_ch_mask = np.repeat(pred_ch_mask_1d[:, None], obs.shape[1], axis=1)

    # Residuals (mask comparison)
    residual_pix = _mask_residual(sim_mask.astype(np.uint8), pred_mask.astype(np.uint8))
    residual_ch = _mask_residual(sim_ch_mask.astype(np.uint8), pred_ch_mask.astype(np.uint8))

    # Values for histograms
    zapped_pix_vals = obs[pred_mask == 1]
    zapped_ch_vals = obs[pred_ch_mask == 1]
    missed_pix_vals = obs[(sim_mask == 1) & (pred_mask == 0)]
    missed_ch_vals = obs[(sim_ch_mask == 1) & (pred_ch_mask == 0)]

    # S/N series for insets
    snr_series = {
        "obs": _safe_snr(obs),
        "gt": _safe_snr(injected_frb),
        "sim_pix": _safe_snr(zapped_sim),
        "sim_ch": _safe_snr(zapped_sim_channel_wise),
        "den": _safe_snr(denoised),
        "pred_pix": _safe_snr(zapped_pred),
        "pred_ch": _safe_snr(zapped_pred_channel_wise),
    }

    valid_snrs = [s for s in snr_series.values() if s is not None and np.any(np.isfinite(s))]
    if valid_snrs:
        ymin = float(np.nanmin([np.nanmin(s) for s in valid_snrs]))
        ymax = float(np.nanmax([np.nanmax(s) for s in valid_snrs]))
    else:
        ymin, ymax = 0.0, 1.0

    fig = plt.figure(figsize=figsize_inches, dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(
        6,
        3,
        height_ratios=[1, 1, 1, 1, 0.9, 0.9],
        width_ratios=[1, 1, 1],
    )



    
    # Row 1
    ax_11 = fig.add_subplot(gs[0, 0])
    _blank(ax_11)

    ax_12 = fig.add_subplot(gs[0, 1])
    _imshow_ds(
        ax=ax_12,
        data=obs,
        title=_title_with_snr("Noisy observed (model input)", obs),
        show_y=True,
        show_x=False,
    )
    _add_snr_inset(ax_12, snr_series["obs"], ymin, ymax)

    ax_13 = fig.add_subplot(gs[0, 2])
    _blank(ax_13)

    # Row 2
    ax_21 = fig.add_subplot(gs[1, 0])
    _imshow_ds(
        ax=ax_21,
        data=injected_frb,
        title=_title_with_snr("Clean ground-truth", injected_frb),
        show_y=True,
        show_x=False,
    )
    _add_snr_inset(ax_21, snr_series["gt"], ymin, ymax)

    ax_22 = fig.add_subplot(gs[1, 1])
    _imshow_ds(
        ax=ax_22,
        data=zapped_sim,
        title=_title_with_snr("Pixel-zapped with simulated mask", zapped_sim),
        show_y=False,
        show_x=False,
    )
    _add_snr_inset(ax_22, snr_series["sim_pix"], ymin, ymax)

    ax_23 = fig.add_subplot(gs[1, 2])
    _imshow_ds(
        ax=ax_23,
        data=zapped_sim_channel_wise,
        title=_title_with_snr("Channel-zapped with simulated mask", zapped_sim_channel_wise),
        show_y=False,
        show_x=False,
    )
    _add_snr_inset(ax_23, snr_series["sim_ch"], ymin, ymax)

    # Row 3
    ax_31 = fig.add_subplot(gs[2, 0])
    _imshow_ds(
        ax=ax_31,
        data=denoised,
        title=_title_with_snr("Denoised (model output)", denoised),
        show_y=True,
        show_x=False,
    )
    _add_snr_inset(ax_31, snr_series["den"], ymin, ymax)

    ax_32 = fig.add_subplot(gs[2, 1])
    title_32 = _title_with_snr("Pixel-zapped with model output mask", zapped_pred)
    if ap_pix is not None:
        title_32 += f"\nAP={ap_pix:.3f}"
    _imshow_ds(
        ax=ax_32,
        data=zapped_pred,
        title=title_32,
        show_y=False,
        show_x=False,
    )
    _add_snr_inset(ax_32, snr_series["pred_pix"], ymin, ymax)

    ax_33 = fig.add_subplot(gs[2, 2])
    title_33 = _title_with_snr("Channel-zapped with model output mask", zapped_pred_channel_wise)
    if ap_ch is not None:
        title_33 += f"\nAP={ap_ch:.3f}"
    _imshow_ds(
        ax=ax_33,
        data=zapped_pred_channel_wise,
        title=title_33,
        show_y=False,
        show_x=False,
    )
    _add_snr_inset(ax_33, snr_series["pred_ch"], ymin, ymax)

    # Row 4
    ax_41 = fig.add_subplot(gs[3, 0])
    _imshow_ds(
        ax=ax_41,
        data=denoised - injected_frb,
        title="Residual (denoised − clean ground-truth)",
        show_y=True,
        show_x=False,
    )
    ax_41.set_xlabel("Time")

    res_cmap = "viridis"
    res_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=256)

    ax_42 = fig.add_subplot(gs[3, 1])
    im42 = ax_42.imshow(
        residual_pix,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[0, residual_pix.shape[1], 0, residual_pix.shape[0]],
        cmap=res_cmap,
        norm=res_norm,
    )
    ax_42.set_title("Pixel mask residual\n(−1 FP / 0 Correct / +1 FN)", fontsize=8)
    ax_42.set_xticklabels([])
    ax_42.set_yticklabels([])
    ax_42.set_xlabel("Time")
    plt.colorbar(im42, ax=ax_42, fraction=0.046, pad=0.02, ticks=[-1, 0, 1])

    ax_43 = fig.add_subplot(gs[3, 2])
    im43 = ax_43.imshow(
        residual_ch,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[0, residual_ch.shape[1], 0, residual_ch.shape[0]],
        cmap=res_cmap,
        norm=res_norm,
    )
    ax_43.set_title("Channel mask residual\n(−1 FP / 0 Correct / +1 FN)", fontsize=8)
    ax_43.set_xticklabels([])
    ax_43.set_yticklabels([])
    ax_43.set_xlabel("Time")
    plt.colorbar(im43, ax=ax_43, fraction=0.046, pad=0.02, ticks=[-1, 0, 1])

    # Row 5
    ax_51 = fig.add_subplot(gs[4, 0])
    _blank(ax_51)

    ax_52 = fig.add_subplot(gs[4, 1])
    ax_52.hist(zapped_pix_vals[~np.isnan(zapped_pix_vals)], bins=50, edgecolor="k")
    ax_52.set_title("Histogram of pixel-zapped RFI intensities", fontsize=8)
    ax_52.set_xlabel("Intensity", fontsize=7)
    ax_52.set_ylabel("Count", fontsize=7)
    ax_52.tick_params(labelsize=7)

    ax_53 = fig.add_subplot(gs[4, 2])
    ax_53.hist(zapped_ch_vals[~np.isnan(zapped_ch_vals)], bins=50, edgecolor="k")
    ax_53.set_title("Histogram of channel-zapped RFI intensities", fontsize=8)
    ax_53.set_xlabel("Intensity", fontsize=7)
    ax_53.set_ylabel("Count", fontsize=7)
    ax_53.tick_params(labelsize=7)

    # Row 6
    ax_61 = fig.add_subplot(gs[5, 0])
    _blank(ax_61)

    ax_62 = fig.add_subplot(gs[5, 1])
    ax_62.hist(missed_pix_vals[~np.isnan(missed_pix_vals)], bins=50, edgecolor="k")
    ax_62.set_title("Histogram of missed pixel RFI intensities", fontsize=8)
    ax_62.set_xlabel("Intensity", fontsize=7)
    ax_62.set_ylabel("Count", fontsize=7)
    ax_62.tick_params(labelsize=7)

    ax_63 = fig.add_subplot(gs[5, 2])
    ax_63.hist(missed_ch_vals[~np.isnan(missed_ch_vals)], bins=50, edgecolor="k")
    ax_63.set_title("Histogram of missed channel RFI intensities", fontsize=8)
    ax_63.set_xlabel("Intensity", fontsize=7)
    ax_63.set_ylabel("Count", fontsize=7)
    ax_63.tick_params(labelsize=7)

    return fig
