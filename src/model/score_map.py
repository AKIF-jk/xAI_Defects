import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def compute_patch_scores(query_patch_features, memory_patch_bank):
    dists = torch.cdist(query_patch_features, memory_patch_bank, p=2.0)
    patch_scores = dists.min(dim=1).values
    return patch_scores


def scores_to_heatmap(patch_scores, img_size=256, patch_size=14, sigma=4.0):
    grid_size = int(patch_scores.shape[0] ** 0.5)
    heat = patch_scores.view(1, 1, grid_size, grid_size).float()
    upsampled = F.interpolate(heat, size=(img_size, img_size),
                               mode="bilinear", align_corners=False)
    kernel = _gaussian_kernel(sigma, device=upsampled.device)
    smoothed = F.conv2d(upsampled, kernel, padding=kernel.shape[-1] // 2)

    lo, hi = smoothed.min(), smoothed.max()
    if hi > lo:
        smoothed = (smoothed - lo) / (hi - lo)
    else:
        smoothed = torch.zeros_like(smoothed)

    return smoothed.squeeze(0)


def _gaussian_kernel(sigma, size=None, device="cpu"):
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    ax = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap="jet"):
    if isinstance(original_img, Image.Image):
        original_np = np.array(original_img.convert("RGB")).astype(np.float32) / 255.0
    elif isinstance(original_img, np.ndarray):
        original_np = original_img.astype(np.float32)
        if original_np.max() > 1.0:
            original_np /= 255.0
    else:
        original_np = np.array(original_img).astype(np.float32)
        if original_np.max() > 1.0:
            original_np /= 255.0

    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.squeeze().cpu().numpy()
    else:
        heatmap_np = np.squeeze(heatmap)

    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_np)[:, :, :3]

    H, W = original_np.shape[:2]
    if colored.shape[:2] != (H, W):
        from skimage.transform import resize
        colored = resize(colored, (H, W), preserve_range=True)

    overlay = (1 - alpha) * original_np + alpha * colored
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)
