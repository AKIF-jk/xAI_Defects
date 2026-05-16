import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mvtec_dataset import MVTecDataset
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from model.score_map import compute_patch_scores


CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

SHOT_MODES = {
    "0-shot": 0,
    "1-shot": 1,
    "4-shot": 4,
}

NORMAL_TEMPLATES = [
    "a photo of a {category}",
    "a photo of a normal {category}",
    "a photo of an undamaged {category}",
    "a photo of a flawless {category}",
]

ANOMALOUS_TEMPLATES = [
    "a photo of a defective {category}",
    "a photo of a damaged {category}",
    "a photo of a broken {category}",
    "a photo of a {category} with a manufacturing defect",
]

MAX_PRO_THRESHOLDS = 256


def au_pro_score(ground_truth_masks, predicted_maps, fpr_limit=0.3):
    gt = np.asarray(ground_truth_masks) > 0.5
    pred = np.asarray(predicted_maps, dtype=np.float32)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: masks={gt.shape}, maps={pred.shape}")

    labels = gt.reshape(-1).astype(np.uint8)
    scores = pred.reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")

    fprs, _, thresholds = roc_curve(labels, scores)
    threshold_indices = _select_pro_threshold_indices(fprs, fpr_limit)
    if len(threshold_indices) == 0:
        return float("nan")

    regions = _connected_gt_regions(gt)
    if not regions:
        return float("nan")

    pro_points = []
    fpr_points = []
    for idx in threshold_indices:
        threshold = thresholds[idx]
        if np.isinf(threshold):
            pred_binary = np.zeros_like(gt, dtype=bool)
        else:
            pred_binary = pred >= threshold

        overlaps = [
            pred_binary[img_idx][ys, xs].sum() / area
            for img_idx, ys, xs, area in regions
        ]
        pro_points.append(float(np.mean(overlaps)))
        fpr_points.append(float(fprs[idx]))

    fpr_arr, pro_arr = _dedupe_curve(np.asarray(fpr_points), np.asarray(pro_points))
    if fpr_arr.size == 0:
        return float("nan")

    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    pro_arr = pro_arr[order]

    if fpr_arr[0] > 0.0:
        fpr_arr = np.insert(fpr_arr, 0, 0.0)
        pro_arr = np.insert(pro_arr, 0, pro_arr[0])

    if fpr_arr[-1] < fpr_limit:
        fpr_arr = np.append(fpr_arr, fpr_limit)
        pro_arr = np.append(pro_arr, pro_arr[-1])
    elif fpr_arr[-1] > fpr_limit:
        pro_at_limit = np.interp(fpr_limit, fpr_arr, pro_arr)
        keep = fpr_arr < fpr_limit
        fpr_arr = np.append(fpr_arr[keep], fpr_limit)
        pro_arr = np.append(pro_arr[keep], pro_at_limit)

    return float(auc(fpr_arr, pro_arr) / fpr_limit)


def _select_pro_threshold_indices(fprs, fpr_limit):
    in_range = np.flatnonzero(fprs <= fpr_limit)
    after_range = np.flatnonzero(fprs > fpr_limit)

    indices = list(in_range)
    if len(after_range) > 0:
        indices.append(int(after_range[0]))

    if len(indices) <= MAX_PRO_THRESHOLDS:
        return indices

    positions = np.linspace(0, len(indices) - 1, MAX_PRO_THRESHOLDS)
    return sorted({indices[int(round(pos))] for pos in positions})


def _connected_gt_regions(gt):
    try:
        from skimage.measure import label as connected_components
    except ImportError:
        from scipy.ndimage import label as scipy_label

        def connected_components(mask):
            return scipy_label(mask)[0]

    regions = []
    for img_idx, mask in enumerate(gt):
        labeled = connected_components(mask.astype(np.uint8))
        for region_id in range(1, int(labeled.max()) + 1):
            ys, xs = np.where(labeled == region_id)
            if ys.size == 0:
                continue
            regions.append((img_idx, ys, xs, float(ys.size)))
    return regions


def _dedupe_curve(fprs, pros):
    merged = {}
    for fpr, pro in zip(fprs, pros):
        key = float(fpr)
        merged[key] = max(float(pro), merged.get(key, 0.0))
    if not merged:
        return np.asarray([]), np.asarray([])
    xs = np.asarray(list(merged.keys()), dtype=np.float64)
    ys = np.asarray([merged[x] for x in xs], dtype=np.float64)
    return xs, ys


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_mask_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def encode_text_features(clip_model, tokenizer, category, device):
    normal_texts = [t.format(category=category.replace("_", " ")) for t in NORMAL_TEMPLATES]
    anomalous_texts = [t.format(category=category.replace("_", " ")) for t in ANOMALOUS_TEMPLATES]

    normal_tokens = tokenizer(normal_texts).to(device)
    anomalous_tokens = tokenizer(anomalous_texts).to(device)

    with torch.no_grad():
        normal = F.normalize(clip_model.encode_text(normal_tokens), dim=-1).mean(dim=0)
        anomalous = F.normalize(clip_model.encode_text(anomalous_tokens), dim=-1).mean(dim=0)

    normal = F.normalize(normal, dim=0)
    anomalous = F.normalize(anomalous, dim=0)
    direction = F.normalize(anomalous - normal, dim=0)
    return normal, anomalous, direction


def evaluate_category_shot(
    clip_model,
    tokenizer,
    data_dir,
    category,
    shot_mode,
    n_shots,
    device,
    img_size,
    batch_size,
    patch_metric,
    patch_top_k,
    patch_layer,
    normal_percentile,
    baseline_quantile,
    sigma,
    use_foreground_mask,
):
    tf = build_transform(img_size)
    mtf = build_mask_transform(img_size)

    train_ds = MVTecDataset(data_dir, category, split="train", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    test_ds = MVTecDataset(data_dir, category, split="test", transform=tf, mask_transform=mtf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    normal_text, anomalous_text, text_direction = encode_text_features(
        clip_model, tokenizer, category, device
    )

    bank = None
    patch_bank = None
    normal_center = None
    normal_scale = None

    if n_shots > 0:
        bank = MemoryBank(feat_dim=768, mode="global", patch_layer=patch_layer)
        bank.build(clip_model, train_loader, n_shots, device)
        patch_bank = bank.get_patch_bank()
        raw_normal_scores = _collect_normal_patch_scores(
            clip_model,
            train_loader,
            n_shots,
            patch_bank,
            device,
            patch_metric,
            patch_top_k,
            patch_layer,
        )
        if not raw_normal_scores:
            raise RuntimeError(
                f"No normal calibration scores collected for {category} {shot_mode}"
            )
        normal_center, normal_scale = _build_patch_calibration(raw_normal_scores)
        normal_z = [
            _calibrate_patch_scores_array(
                scores, normal_center, normal_scale, baseline_quantile
            )
            for scores in raw_normal_scores
        ]
        all_normal = np.concatenate(normal_z)
        global_max = max(float(np.percentile(all_normal, normal_percentile)), 1.0)
    else:
        global_max = None

    image_scores = []
    image_labels = []
    predicted_maps = []
    gt_masks = []

    for images, masks, labels in test_loader:
        for idx in range(images.size(0)):
            image = images[idx:idx + 1].to(device)
            mask = masks[idx].squeeze(0).cpu().numpy()
            label = int(labels[idx].item())

            global_feat, query_patches = _extract_features(
                clip_model, image, patch_layer
            )
            if query_patches is None:
                continue

            if n_shots == 0:
                score, patch_scores = _zero_shot_scores(
                    global_feat, query_patches, normal_text, anomalous_text, text_direction
                )
            else:
                raw_scores = compute_patch_scores(
                    query_patches,
                    patch_bank.to(query_patches.device),
                    metric=patch_metric,
                    top_k=patch_top_k,
                )
                patch_scores = _calibrate_patch_scores_tensor(
                    raw_scores,
                    normal_center,
                    normal_scale,
                    baseline_quantile,
                )
                score = float(torch.quantile(patch_scores, 0.95).detach().cpu())

            score_map = _patch_scores_to_map(patch_scores, img_size=img_size, sigma=sigma)
            if n_shots > 0 and global_max is not None:
                score_map = np.clip(score_map / global_max, 0.0, 1.0)

            if use_foreground_mask:
                orig_np = _denormalize(image[0]).permute(1, 2, 0).cpu().numpy()
                score_map = _apply_foreground_mask_np(score_map, orig_np)

            image_scores.append(score)
            image_labels.append(label)
            predicted_maps.append(score_map.astype(np.float32))
            gt_masks.append(mask.astype(np.uint8))

    image_scores = np.asarray(image_scores, dtype=np.float32)
    image_labels = np.asarray(image_labels, dtype=np.uint8)
    predicted_maps = np.asarray(predicted_maps, dtype=np.float32)
    gt_masks = np.asarray(gt_masks, dtype=np.uint8)

    img_auroc = _safe_roc_auc(image_labels, image_scores)
    pix_auroc = _safe_roc_auc(gt_masks.reshape(-1), predicted_maps.reshape(-1))
    au_pro = au_pro_score(gt_masks, predicted_maps, fpr_limit=0.3)

    return {
        "category": category,
        "shot_mode": shot_mode,
        "img_auroc": img_auroc,
        "pix_auroc": pix_auroc,
        "au_pro": au_pro,
        "n_test": int(len(test_ds)),
    }


def _collect_normal_patch_scores(
    clip_model,
    train_loader,
    n_shots,
    patch_bank,
    device,
    patch_metric,
    patch_top_k,
    patch_layer,
):
    scores = []
    count = 0
    for images, _ in train_loader:
        if count >= n_shots:
            break
        image = images.to(device)
        _, query_patches = _extract_features(clip_model, image, patch_layer)
        if query_patches is None:
            continue
        patch_scores = compute_patch_scores(
            query_patches,
            patch_bank.to(query_patches.device),
            metric=patch_metric,
            top_k=patch_top_k,
        )
        scores.append(patch_scores.detach().cpu().numpy())
        count += images.size(0)
    return scores


def _zero_shot_scores(global_feat, query_patches, normal_text, anomalous_text, text_direction):
    image_feat = F.normalize(global_feat.squeeze(0).float(), dim=-1)
    image_score = float((image_feat @ anomalous_text - image_feat @ normal_text).detach().cpu())

    patches = F.normalize(query_patches.float(), dim=-1)
    patch_scores = patches @ text_direction
    return image_score, patch_scores


def _extract_features(clip_model, image, patch_layer="ln_post"):
    target = _get_patch_hook_target(clip_model, patch_layer)
    if target is None:
        return None, None

    patch_features = [None]

    def hook(module, inp, out):
        patch_features[0] = _standardize_patch_features(out)

    handle = target.register_forward_hook(hook)
    with torch.no_grad():
        global_feat = clip_model.encode_image(image)
    handle.remove()

    patch_feats = patch_features[0]
    if patch_feats is None:
        return global_feat, None

    if patch_feats.shape[-1] != global_feat.shape[-1]:
        proj = getattr(clip_model.visual, "proj", None)
        if proj is not None and patch_feats.shape[-1] == proj.shape[0]:
            patch_feats = patch_feats @ proj.detach().to(patch_feats.device)

    return global_feat, patch_feats[0, 1:, :]


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


def _build_patch_calibration(normal_scores):
    normal_matrix = np.stack(normal_scores).astype(np.float32)
    center = np.median(normal_matrix, axis=0)
    mad = np.median(np.abs(normal_matrix - center), axis=0)
    scale = 1.4826 * mad
    positive_scale = scale[scale > 0]
    scale_floor = np.percentile(positive_scale, 10) if positive_scale.size else 1e-6
    scale = np.maximum(scale, max(float(scale_floor), 1e-6))
    return center, scale


def _calibrate_patch_scores_array(scores, normal_center, normal_scale, baseline_quantile):
    z = np.maximum((scores - normal_center) / normal_scale, 0.0)
    baseline = np.quantile(z, baseline_quantile)
    return np.maximum(z - baseline, 0.0)


def _calibrate_patch_scores_tensor(patch_scores, normal_center, normal_scale, baseline_quantile):
    center = torch.as_tensor(normal_center, device=patch_scores.device, dtype=patch_scores.dtype)
    scale = torch.as_tensor(normal_scale, device=patch_scores.device, dtype=patch_scores.dtype)
    z = torch.clamp((patch_scores - center) / scale, min=0.0)
    baseline = torch.quantile(z, baseline_quantile)
    return torch.clamp(z - baseline, min=0.0)


def _patch_scores_to_map(patch_scores, img_size=224, sigma=12.0):
    grid_size = int(patch_scores.numel() ** 0.5)
    heat = patch_scores.view(1, 1, grid_size, grid_size).float()
    upsampled = F.interpolate(
        heat,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    kernel = _gaussian_kernel(sigma, device=upsampled.device)
    smoothed = F.conv2d(upsampled, kernel, padding=kernel.shape[-1] // 2)
    return smoothed.squeeze().detach().cpu().numpy()


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


def _apply_foreground_mask_np(score_map, image_np):
    mask = _estimate_foreground_mask_np(image_np)
    if mask is None:
        return score_map
    return score_map * mask


def _estimate_foreground_mask_np(image_np):
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

    mask = torch.from_numpy(foreground.astype(np.float32)).view(1, 1, h, w)
    mask = F.max_pool2d(mask, kernel_size=21, stride=1, padding=10)
    mask = F.avg_pool2d(mask, kernel_size=21, stride=1, padding=10)
    return torch.clamp(mask.squeeze(), 0.0, 1.0).numpy()


def _denormalize(tensor):
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def _safe_roc_auc(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run_full_evaluation(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading OpenCLIP backbone...")
    clip_model, tokenizer, _, device = load_backbone(args.device)
    clip_model.eval()

    rows = []
    for category in CATEGORIES:
        for shot_mode, n_shots in SHOT_MODES.items():
            print(f"{category:<12} {shot_mode:<7} ... ", end="", flush=True)
            row = evaluate_category_shot(
                clip_model=clip_model,
                tokenizer=tokenizer,
                data_dir=args.data_dir,
                category=category,
                shot_mode=shot_mode,
                n_shots=n_shots,
                device=device,
                img_size=args.img_size,
                batch_size=args.batch_size,
                patch_metric=args.patch_metric,
                patch_top_k=args.patch_top_k,
                patch_layer=args.patch_layer,
                normal_percentile=args.normal_percentile,
                baseline_quantile=args.baseline_quantile,
                sigma=args.sigma,
                use_foreground_mask=not args.disable_foreground_mask,
            )
            rows.append(row)
            print(
                f"img={row['img_auroc']:.4f} "
                f"pix={row['pix_auroc']:.4f} "
                f"pro={row['au_pro']:.4f} "
                f"n={row['n_test']}"
            )

    df = pd.DataFrame(rows, columns=[
        "category", "shot_mode", "img_auroc", "pix_auroc", "au_pro", "n_test",
    ])

    csv_path = os.path.join(args.output_dir, "full_evaluation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    _print_summary_tables(df)

    plot_path = os.path.join(args.output_dir, "full_evaluation_bars.png")
    _plot_grouped_bars(df, plot_path)
    print(f"Saved: {plot_path}")
    return df


def _print_summary_tables(df):
    summary = (
        df.groupby("shot_mode")[["img_auroc", "pix_auroc", "au_pro"]]
        .mean()
        .reindex(SHOT_MODES.keys())
    )
    print("\nMean Across Categories")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    au_pro = df.pivot(index="category", columns="shot_mode", values="au_pro")
    au_pro = au_pro.reindex(index=CATEGORIES, columns=SHOT_MODES.keys())

    print("\nPer-Category AU-PRO")
    print(f"{'category':<14} {'0-shot':>8} {'1-shot':>8} {'4-shot':>8}")
    for category, row in au_pro.iterrows():
        values = []
        for shot_mode in SHOT_MODES:
            value = row[shot_mode]
            text = "nan" if pd.isna(value) else f"{value:.4f}"
            if shot_mode == "4-shot" and not pd.isna(value):
                text = _color_au_pro(text, value)
            values.append(f"{text:>8}")
        print(f"{category:<14} {values[0]} {values[1]} {values[2]}")


def _color_au_pro(text, value):
    green = "\033[92m"
    red = "\033[91m"
    reset = "\033[0m"
    if value > 0.80:
        return f"{green}{text}{reset}"
    if value < 0.60:
        return f"{red}{text}{reset}"
    return text


def _plot_grouped_bars(df, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("img_auroc", "Image-Level AUROC"),
        ("pix_auroc", "Pixel-Level AUROC"),
        ("au_pro", "AU-PRO"),
    ]

    x = np.arange(len(CATEGORIES))
    width = 0.24
    offsets = np.linspace(-width, width, len(SHOT_MODES))

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        pivot = df.pivot(index="category", columns="shot_mode", values=metric)
        pivot = pivot.reindex(index=CATEGORIES, columns=SHOT_MODES.keys())
        for offset, shot_mode in zip(offsets, SHOT_MODES):
            ax.bar(
                x + offset,
                pivot[shot_mode].values,
                width=width,
                label=shot_mode,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(CATEGORIES, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Full 0/1/4-shot AdaptCLIP-style evaluation on MVTec AD"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--patch_top_k", type=int, default=3)
    parser.add_argument("--patch_layer", default="ln_post")
    parser.add_argument("--normal_percentile", type=float, default=99.0)
    parser.add_argument("--baseline_quantile", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=12.0)
    parser.add_argument("--disable_foreground_mask", action="store_true")
    args = parser.parse_args()
    run_full_evaluation(args)


if __name__ == "__main__":
    main()
