"""Physiological Channel Reinterpretation (PCR): spectral-complexity-guided
channel-adaptive masking on the student branch. `sam_importance` estimates per-channel
masking ratios from spectral density (SAM); `ChannelWiseMaskGenerator` applies block
masking online so dominant bands are masked more, enforcing cross-channel reasoning."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sam_importance(tgt_loader, r_min=0.3, r_max=0.7):
    """
    Compute 3 masking ratios for the R/G/B spectral groups based on SAM sensitivity.
    Return: mask_ratios (3,) in **[R, G, B] order**, i.e. aligned to the channel order
    of the plt.imread RGB tensors (channel 0 = R, 1 = G, 2 = B) so that ratio[c] is the
    masking ratio applied to input channel c. This matches the paper's Eq. r=[r_R,r_G,r_B],
    where spectrally dominant bands (red, for skin) receive the highest masking ratio.
    """

    def split_rgb_indices(num_channels=31):
        wl = np.linspace(400, 700, num_channels)
        B = np.where((wl >= 450) & (wl < 495))[0]   # Blue
        G = np.where((wl >= 495) & (wl < 570))[0]   # Green
        R = np.where((wl >= 620) & (wl <= 700))[0]  # Red
        return {"B": B, "G": G, "R": R}

    def sam_angle(a, b, eps=1e-12):
        # a,b: (N,C) -> (N,)
        num = np.sum(a * b, axis=1)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
        return np.arccos(np.clip(num / denom, -1.0, 1.0))

    sam_sum = {k: 0.0 for k in ('B', 'G', 'R')}
    sam_cnt = {k: 0   for k in ('B', 'G', 'R')}

    for _, hsi in tgt_loader:  # hsi: [B,C,H,W]
        hsi = hsi.cpu().numpy()

        # per-image (simple & stable)
        for b in range(hsi.shape[0]):
            h = hsi[b]  # [C,H,W]
            h = (h - h.min()) / (h.max() - h.min() + 1e-8)

            C, H, W = h.shape
            specs_all = h.reshape(C, -1).T  # [N,C]

            for band, idx in split_rgb_indices(C).items():
                specs_filled = specs_all.copy()
                for ch in idx:
                    mean_val = specs_all[:, ch].mean()
                    specs_filled[:, ch] = mean_val

                sam_values = sam_angle(specs_all, specs_filled)
                sam_sum[band] += float(np.sum(sam_values))
                sam_cnt[band] += int(len(sam_values))

    avg_vals = {b: sam_sum[b] / (sam_cnt[b] + 1e-8) for b in ('B', 'G', 'R')}
    # arrange in [R, G, B] order to match the RGB channel layout (channel 0 = R)
    avg_vals_arr = np.array([avg_vals['R'], avg_vals['G'], avg_vals['B']])  # [3,] = [R,G,B]
    norm = (avg_vals_arr - avg_vals_arr.min()) / (avg_vals_arr.max() - avg_vals_arr.min() + 1e-8)
    mask_ratios = r_min + norm * (r_max - r_min)

    print(f"Spectral density (per-pixel SAM): R={avg_vals_arr[0]:.4f}, G={avg_vals_arr[1]:.4f}, B={avg_vals_arr[2]:.4f}")
    print(f"PCR mask ratios                 : R={mask_ratios[0]:.3f}, G={mask_ratios[1]:.3f}, B={mask_ratios[2]:.3f}")

    return mask_ratios  # [R,G,B], aligned to input channel index


class ChannelWiseMaskGenerator(nn.Module):
    """
    Spectral-density block masking on RGB channels (3ch).
    ratio_dict is array-like length 3 in [R, G, B] order (matching input channel index);
    ratio_dict[i] is the masking ratio applied to input channel i.
    """
    def __init__(self, ratio_dict, block=16):
        super().__init__()
        self.ratio_dict = ratio_dict  # array-like length 3
        self.block = block

    @torch.no_grad()
    def _rand_mask(self, shp, ratio, dev):
        B, _, H, W = shp
        mh, mw = (H - 1) // self.block + 1, (W - 1) // self.block + 1
        total_blocks = mh * mw
        num_mask = int(ratio * total_blocks)

        perm = torch.rand(B, total_blocks, device=dev)
        threshold = torch.topk(perm, k=total_blocks - num_mask, dim=1, largest=False).values[:, -1:]
        keep = (perm > threshold).float()
        keep = keep.view(B, 1, mh, mw)
        mask = F.interpolate(keep, size=(H, W), mode="nearest")
        return mask  # [B,1,H,W]

    @torch.no_grad()
    def forward(self, x):
        for i in range(3):
            mask = self._rand_mask(x.shape, float(self.ratio_dict[i]), x.device).squeeze(1)
            x[:, i, :, :] *= mask
        return x
