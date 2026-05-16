import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.mvtec_dataset import MVTecDataset
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from model.score_map import compute_patch_scores, scores_to_heatmap, overlay_heatmap


def test_score_maps(
    data_dir,
    output_dir,
    device,
    category="bottle",
    n_shots=32,
    patch_metric="cosine",
    patch_top_k=3,
    patch_layer="ln_post",
    normal_percentile=99.0,
    baseline_quantile=0.5,
    sigma=12.0,
    use_foreground_mask=True,
):
    print("Loading CLIP backbone...")
    clip_model, _, _, device = load_backbone(device)
    clip_model.eval()

    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MVTecDataset(data_dir, category, split="train", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    test_ds = MVTecDataset(data_dir, category, split="test",
                            transform=tf,
                            mask_transform=transforms.Compose([
                                transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
                                transforms.ToTensor(),
                            ]))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"Building memory bank ({n_shots}-shot)...")
    bank = MemoryBank(feat_dim=768, mode="global", patch_layer=patch_layer)
    bank.build(clip_model, train_loader, n_shots, device)
    patch_bank = bank.get_patch_bank()
    print(f"  Memory bank: {bank.size} images, patch bank: {patch_bank.shape}")

    print("Collecting normal patch scores...")
    normal_scores = []
    for batch in train_loader:
        img_tensor, _ = batch
        ps = _extract_patch_scores(
            clip_model,
            img_tensor,
            patch_bank,
            device,
            patch_metric,
            patch_top_k,
            patch_layer,
        )
        if ps is not None:
            normal_scores.append(ps.cpu().numpy())

    normal_center, normal_scale = _build_patch_calibration(normal_scores)

    print("Building panels and sampling anomalous scores...")
    anom_panel = []
    norm_panel = []
    anomalous_scores = []

    for batch in test_loader:
        img_tensor, mask_tensor, label = batch
        label = label.item()

        if label == 1 and len(anom_panel) < 3:
            ps = _extract_patch_scores(
                clip_model,
                img_tensor,
                patch_bank,
                device,
                patch_metric,
                patch_top_k,
                patch_layer,
            )
            if ps is not None:
                anomalous_scores.append(ps.cpu().numpy())
            anom_panel.append((img_tensor, mask_tensor))
        elif label == 0 and len(norm_panel) < 3:
            norm_panel.append((img_tensor, mask_tensor))

        if len(anom_panel) == 3 and len(norm_panel) == 3:
            break

    normal_z_scores = [
        _calibrate_patch_scores_array(
            s,
            normal_center,
            normal_scale,
            baseline_quantile,
        )
        for s in normal_scores
    ]
    all_normal = np.concatenate(normal_z_scores)
    global_max = max(float(np.percentile(all_normal, normal_percentile)), 1.0)
    print(f"  normal 50th pct: {np.percentile(all_normal, 50):.4f}")
    print(f"  normal 95th pct: {np.percentile(all_normal, 95):.4f}")
    print(
        "  global_max "
        f"(normal calibrated {normal_percentile:g}th pct): {global_max:.4f}"
    )

    if anomalous_scores:
        anom_z_scores = [
            _calibrate_patch_scores_array(
                s,
                normal_center,
                normal_scale,
                baseline_quantile,
            )
            for s in anomalous_scores
        ]
        anom_maxes = [s.max() for s in anom_z_scores]
        print(
            "  mean anomalous max calibrated patch score: "
            f"{np.mean(anom_maxes):.4f}  (should exceed global_max)"
        )

    min_component_area = 50

    anom_results = []
    for img_tensor, mask_tensor in anom_panel:
        r = _process_one(
            clip_model,
            bank,
            patch_bank,
            img_tensor,
            mask_tensor,
            device,
            global_max,
            min_component_area,
            normal_center,
            normal_scale,
            patch_metric,
            patch_top_k,
            patch_layer,
            baseline_quantile,
            sigma,
            use_foreground_mask,
        )
        anom_results.append(r)

    norm_results = []
    for img_tensor, mask_tensor in norm_panel:
        r = _process_one(
            clip_model,
            bank,
            patch_bank,
            img_tensor,
            mask_tensor,
            device,
            global_max,
            min_component_area,
            normal_center,
            normal_scale,
            patch_metric,
            patch_top_k,
            patch_layer,
            baseline_quantile,
            sigma,
            use_foreground_mask,
        )
        norm_results.append(r)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 4, figsize=(16, 24))

    for row, (label, results) in enumerate([("Anomalous", anom_results),
                                              ("Normal", norm_results)]):
        for col, (orig, heatmap, overlay_img, mask_np) in enumerate(results):
            ax_orig = axes[row * 3 + col][0]
            ax_overlay = axes[row * 3 + col][1]
            ax_heat = axes[row * 3 + col][2]
            ax_mask = axes[row * 3 + col][3]

            ax_orig.imshow(orig)
            ax_orig.set_title(f"{label} #{col+1}: Original")
            ax_orig.axis("off")

            ax_overlay.imshow(overlay_img)
            ax_overlay.set_title(f"{label} #{col+1}: Overlay")
            ax_overlay.axis("off")

            ax_heat.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            ax_heat.set_title(f"{label} #{col+1}: Heatmap")
            ax_heat.axis("off")

            if mask_np is not None:
                ax_mask.imshow(mask_np, cmap="gray")
                ax_mask.set_title(f"{label} #{col+1}: GT Mask")
            else:
                ax_mask.imshow(np.zeros((224, 224)), cmap="gray")
                ax_mask.set_title(f"{label} #{col+1}: No GT")
            ax_mask.axis("off")

    plt.suptitle(
        f"{category}: Score Maps ({n_shots}-shot memory bank, calibrated residual)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{category}_score_maps_test.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _capture_hook(model, patch_layer="ln_post"):
    target = _get_patch_hook_target(model, patch_layer)
    if target is None:
        return None, None

    patch_features = [None]

    def hook(module, inp, out):
        patch_features[0] = _standardize_patch_features(out)

    handle = target.register_forward_hook(hook)
    return patch_features, handle


def _get_patch_hook_target(model, patch_layer):
    if patch_layer in (None, "ln_post"):
        return getattr(model.visual, "ln_post", None)

    layer = str(patch_layer)
    if layer.startswith("resblock:"):
        layer = layer.split(":", 1)[1]

    try:
        idx = int(layer)
    except ValueError:
        return getattr(model.visual, "ln_post", None)

    transformer = getattr(model.visual, "transformer", None)
    blocks = getattr(transformer, "resblocks", None)
    if blocks is None:
        return getattr(model.visual, "ln_post", None)
    if idx < -len(blocks) or idx >= len(blocks):
        return getattr(model.visual, "ln_post", None)
    return blocks[idx]


def _standardize_patch_features(out):
    if isinstance(out, (tuple, list)):
        out = out[0]

    out = out.detach()
    if out.dim() == 3 and out.shape[0] > out.shape[1] and out.shape[0] > 32:
        out = out.permute(1, 0, 2)
    return out


def _extract_patch_scores(
    clip_model,
    img_tensor,
    patch_bank,
    device,
    patch_metric="cosine",
    patch_top_k=3,
    patch_layer="ln_post",
):
    img_tensor = img_tensor.to(device)
    pf_hook, handle = _capture_hook(clip_model, patch_layer)
    if handle is None:
        return None
    with torch.no_grad():
        _ = clip_model.encode_image(img_tensor)
    patch_feats_all = pf_hook[0]
    handle.remove()
    if patch_feats_all is None:
        return None

    if patch_feats_all.shape[-1] == 1024:
        proj = getattr(clip_model.visual, "proj", None)
        if proj is not None:
            patch_feats_all = patch_feats_all @ proj.detach().to(patch_feats_all.device)

    query_patches = patch_feats_all[0, 1:, :]
    pb = patch_bank
    if pb.shape[1] != query_patches.shape[1]:
        proj = getattr(clip_model.visual, "proj", None)
        if proj is not None:
            pb = pb @ proj.detach().to(pb.device)
    return compute_patch_scores(
        query_patches,
        pb.to(query_patches.device),
        metric=patch_metric,
        top_k=patch_top_k,
    )


def _build_patch_calibration(normal_scores):
    normal_matrix = np.stack(normal_scores).astype(np.float32)
    center = np.median(normal_matrix, axis=0)
    mad = np.median(np.abs(normal_matrix - center), axis=0)
    scale = 1.4826 * mad

    positive_scale = scale[scale > 0]
    scale_floor = np.percentile(positive_scale, 10) if positive_scale.size else 1e-6
    scale = np.maximum(scale, max(float(scale_floor), 1e-6))
    return center, scale


def _calibrate_patch_scores_array(
    scores,
    normal_center,
    normal_scale,
    baseline_quantile=0.5,
):
    z = np.maximum((scores - normal_center) / normal_scale, 0.0)
    baseline = np.quantile(z, baseline_quantile)
    return np.maximum(z - baseline, 0.0)


def _calibrate_patch_scores_tensor(
    patch_scores,
    normal_center,
    normal_scale,
    baseline_quantile=0.5,
):
    center = torch.as_tensor(normal_center, device=patch_scores.device, dtype=patch_scores.dtype)
    scale = torch.as_tensor(normal_scale, device=patch_scores.device, dtype=patch_scores.dtype)
    z = torch.clamp((patch_scores - center) / scale, min=0.0)
    baseline = torch.quantile(z, baseline_quantile)
    return torch.clamp(z - baseline, min=0.0)


def _apply_foreground_mask(heatmap, image_np):
    mask = _estimate_foreground_mask(image_np, heatmap.device)
    if mask is None:
        return heatmap
    return heatmap * mask.unsqueeze(0)


def _estimate_foreground_mask(image_np, device):
    img = np.asarray(image_np, dtype=np.float32)
    h, w = img.shape[:2]

    border_width = max(4, min(h, w) // 32)
    border = np.concatenate([
        img[:border_width].reshape(-1, 3),
        img[-border_width:].reshape(-1, 3),
        img[:, :border_width].reshape(-1, 3),
        img[:, -border_width:].reshape(-1, 3),
    ], axis=0)

    if float(border.std(axis=0).mean()) > 0.08:
        return None

    background = np.median(border, axis=0)
    color_dist = np.linalg.norm(img - background, axis=2)
    border_dist = np.linalg.norm(border - background, axis=1)
    threshold = max(0.10, float(np.percentile(border_dist, 95)) * 3.0)

    foreground = color_dist > threshold
    coverage = float(foreground.mean())
    if coverage < 0.05 or coverage > 0.95:
        return None

    mask = torch.from_numpy(foreground.astype(np.float32)).to(device)
    mask = mask.view(1, 1, h, w)
    mask = F.max_pool2d(mask, kernel_size=21, stride=1, padding=10)
    mask = F.avg_pool2d(mask, kernel_size=21, stride=1, padding=10)
    return torch.clamp(mask.squeeze(0).squeeze(0), 0.0, 1.0)


def _process_one(
    clip_model,
    bank,
    patch_bank,
    img_tensor,
    mask_tensor,
    device,
    global_max=None,
    min_component_area=0,
    normal_center=None,
    normal_scale=None,
    patch_metric="cosine",
    patch_top_k=3,
    patch_layer="ln_post",
    baseline_quantile=0.5,
    sigma=12.0,
    use_foreground_mask=True,
):
    img_tensor = img_tensor.to(device)
    orig_np = _denormalize(img_tensor[0]).permute(1, 2, 0).cpu().numpy()

    mask_np = None
    if mask_tensor is not None:
        mask_np = mask_tensor.squeeze().cpu().numpy()

    patch_scores = _extract_patch_scores(
        clip_model,
        img_tensor,
        patch_bank,
        device,
        patch_metric,
        patch_top_k,
        patch_layer,
    )
    if patch_scores is None:
        return orig_np, np.zeros((224, 224)), orig_np, mask_np

    if normal_center is not None and normal_scale is not None:
        patch_scores = _calibrate_patch_scores_tensor(
            patch_scores,
            normal_center,
            normal_scale,
            baseline_quantile,
        )

    heatmap = scores_to_heatmap(
        patch_scores,
        img_size=224,
        sigma=sigma,
        global_max=global_max,
        min_component_area=min_component_area,
    )
    if use_foreground_mask:
        heatmap = _apply_foreground_mask(heatmap, orig_np)
    overlay_img = overlay_heatmap((orig_np * 255).astype(np.uint8), heatmap, alpha=0.5)

    hm_np = heatmap.squeeze().cpu().numpy()

    return orig_np, hm_np, overlay_img, mask_np


def _denormalize(tensor):
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Test score map generation on bottle category"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/heatmaps")
    parser.add_argument("--device", default=None)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--n_shots", type=int, default=32)
    parser.add_argument(
        "--patch_metric",
        choices=["cosine", "l2"],
        default="cosine",
    )
    parser.add_argument("--patch_top_k", type=int, default=3)
    parser.add_argument(
        "--patch_layer",
        default="ln_post",
        help="Patch-token hook: 'ln_post' or transformer resblock index like -2.",
    )
    parser.add_argument("--normal_percentile", type=float, default=99.0)
    parser.add_argument("--baseline_quantile", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=12.0)
    parser.add_argument(
        "--disable_foreground_mask",
        action="store_true",
        help="Disable conservative border-based foreground masking.",
    )
    args = parser.parse_args()
    test_score_maps(
        args.data_dir,
        args.output_dir,
        args.device,
        args.category,
        args.n_shots,
        args.patch_metric,
        args.patch_top_k,
        args.patch_layer,
        args.normal_percentile,
        args.baseline_quantile,
        args.sigma,
        not args.disable_foreground_mask,
    )


if __name__ == "__main__":
    main()
