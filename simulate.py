import numpy as np

def ground_truth(args):
    """Generate a clean synthetic spectrogram and FRB window."""
    freq_bins, time_bins = args.img_size
    bg_level = np.random.uniform(*args.theta_bg_intensity)
    gaussian_intensity = np.random.uniform(*args.theta_gaussian_intensity)
    bg_noise = gaussian_intensity * np.abs(np.random.randn(freq_bins, time_bins))
    img = np.full(args.img_size, bg_level) + bg_noise

    center_time = time_bins // 2 + np.random.randint(-time_bins // 2, 0)
    center_freq = np.random.randint(freq_bins)
    sigma_time = np.random.uniform(2, 6)
    sigma_freq = np.random.uniform(40, 80)
    amplitude = np.random.uniform(*args.theta_frb_intensity)

    x = np.arange(time_bins)
    y = np.arange(freq_bins)
    X, Y = np.meshgrid(x, y)
    gauss = amplitude * np.exp(-(((X - center_time) ** 2) / (2 * sigma_time ** 2) +
                                 ((Y - center_freq) ** 2) / (2 * sigma_freq ** 2)))
    img += gauss
    start_time = max(0, int(center_time - 3 * sigma_time))
    end_time = min(time_bins, int(center_time + 3 * sigma_time))
    return np.clip(img, 0, None), start_time, end_time, bg_noise

def noise(x, args, bg_noise):
    """Inject structured RFI noise and return noisy image and mask."""
    img = x.copy()
    freq_bins, time_bins = args.img_size
    mask_total = np.zeros_like(x)

    if bg_noise is not None:
        img = np.full(args.img_size, np.random.uniform(*args.theta_bg_intensity)) + bg_noise
        img += x - np.nanmin(x)

    for _ in range(2):
        n_channels = np.random.poisson(args.theta_n_channels) + 1
        channel_height = np.random.exponential(args.theta_channel_height, n_channels).astype(int) + 1
        channel_intensity = np.random.uniform(*args.theta_channel_intensity, size=n_channels)
        used_positions = []

        for i in range(n_channels):
            h = channel_height[i]
            for _ in range(20):
                pos_y = np.random.randint(0, max(1, freq_bins - h))
                overlaps = any((pos_y < end) and ((pos_y + h) > start) for start, end in used_positions)
                if not overlaps:
                    used_positions.append((pos_y, pos_y + h))
                    break
            else:
                continue

            pulse_mask = np.zeros((h, time_bins))
            n_pulses = np.random.randint(3, 100)
            for _ in range(n_pulses):
                pulse_center = np.random.randint(0, time_bins)
                pulse_width = np.random.randint(5, 15)
                t = np.arange(time_bins)
                time_envelope = np.exp(-0.5 * ((t - pulse_center) / pulse_width) ** 4)
                pulse = time_envelope * channel_intensity[i]
                pulse_mask += pulse  # broadcast across h rows

            img[pos_y:pos_y + h] += pulse_mask
            mask_total[pos_y:pos_y + h] += (pulse_mask > (0.1 * channel_intensity[i]))

    return img, np.clip(mask_total, 0, 1).astype(np.float32)