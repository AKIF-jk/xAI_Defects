"""
generate_xai_gallery.py
=======================
Comprehensive XAI visual gallery for manual inspection across all 15 MVTec AD categories.

For each category:
  - 4 anomalous test images (one per major defect type if available)
  - 1 normal test image (negative control)
  = ~75 anomalous + 15 normal = ~90 panels total

Each panel: 1x4 grid [Original | Score Map | Grad-CAM | SHAP] + explanation text below

Outputs:
  outputs/heatmaps/{category}_xai_gallery.png   (one tall image per category)
  outputs/results/xai_gallery.html              (summary HTML, sortable, color-coded)

Usage:
  python generate_xai_gallery.py --data_dir /path/to/mvtec_anomaly_detection --threshold 0.4

Scoring mechanism
-----------------
The gallery uses one canonical few-shot inference result for each image:
  1. Patch-level cosine distance to memory bank features
  2. Z-score calibration against normal training images (median + MAD)
  3. Baseline subtraction (median of Z-scores)
  4. Patch-level normalization for score-map display
  5. Image-level normal calibration threshold for the displayed anomaly score.
     Score scaling maps the selected normal percentile to 0.5; the gallery
     decision threshold is configurable and defaults to 0.4 for better recall.

The PromptQueryAdapter is NOT used for scoring (its weights are untrained random).
Grad-CAM and SHAP use patch-memory objectives, so their heatmaps explain the
same decision family as the displayed anomaly score.

Other fixes
-----------
1. cm.get_cmap("jet") deprecated since matplotlib 3.7. Replaced with
   matplotlib.colormaps["jet"].
2. _select_test_images: O(n²) list comprehensions on every loop iteration
   replaced with a simple anomalous_count integer counter.
3. `import matplotlib.pyplot as mpl_plt` stays at module scope instead of the
   per-image processing loop.
4. result["pipe"] popped to avoid holding LLM pipeline reference in result dict.
"""

import argparse
import gc
import html
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as mpl_plt      # alias kept for _render_panel callers
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from model.score_map import compute_patch_scores, scores_to_heatmap, overlay_heatmap
from xai.gradcam import CLIPGradCAM
from xai.shap_explainer import PatchSHAPExplainer

logger = logging.getLogger(__name__)

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

DEFAULT_ANOMALY_THRESHOLD = 0.4

IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _denormalize_image(tensor):
    """Convert a [C, H, W] normalized tensor to [H, W, 3] uint8 numpy."""
    tensor = tensor.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = torch.clamp(tensor * std + mean, 0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _normal_case_explanation(
    predicted_label,
    anomaly_score,
    threshold,
    shap_region=None,
    gradcam_region=None,
):
    """Return a bounded explanation string for normal-labeled samples."""
    if predicted_label == 0:
        return (
            f"No labeled defect; score {anomaly_score:.2f} is below "
            f"threshold {threshold:.2f}, so this sample is treated as normal."
        )

    evidence = _evidence_phrase(shap_region, gradcam_region)
    return (
        f"No labeled defect, but score {anomaly_score:.2f} exceeds "
        f"threshold {threshold:.2f}; {evidence}, suggesting normal variation "
        f"or a false alarm."
    )


def _extract_patch_features(clip_model, image_tensor):
    """Extract global feature and patch tokens from CLIP visual encoder.

    Matches _extract_features() in full_evaluation.py exactly.
    Returns (global_feat, patch_tokens) where patch_tokens excludes CLS token.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    target = getattr(clip_model.visual, "ln_post", None)
    if target is None:
        return None, None

    patch_features = [None]

    def hook(module, inp, out):
        patch_features[0] = out.detach()

    handle = target.register_forward_hook(hook)
    with torch.no_grad():
        global_feat = clip_model.encode_image(image_tensor)
    handle.remove()

    patch_feats = patch_features[0]
    if patch_feats is None:
        return global_feat, None

    # Standardize layout: ensure [B, tokens, C]
    if patch_feats.dim() == 3 and patch_feats.shape[0] > patch_feats.shape[1] and patch_feats.shape[0] > 32:
        patch_feats = patch_feats.permute(1, 0, 2)

    # Project to match global feature dimension if needed
    if patch_feats.shape[-1] != global_feat.shape[-1]:
        proj = getattr(clip_model.visual, "proj", None)
        if proj is not None and patch_feats.shape[-1] == proj.shape[0]:
            patch_feats = patch_feats @ proj.detach().to(patch_feats.device)

    # Drop CLS token
    return global_feat, patch_feats[0, 1:, :]


def _build_patch_calibration(
    clip_model,
    calib_loader,
    patch_memory_tensor,
    device,
    map_percentile=99,
    score_percentile=99,
    baseline_quantile=0.5,
):
    """Build Z-score calibration stats from a disjoint set of normal images.

    Unlike the memory bank (built from support set), this uses a separate
    calibration set to avoid overfitting.

    Returns:
      center, scale: per-patch robust Z-score calibration
      map_global_max: patch-level display scale for heatmaps
      score_threshold_raw: image-level normal threshold for score calibration
    """
    scores = []
    for images, _ in calib_loader:
        image = images.to(device)
        _, query_patches = _extract_patch_features(clip_model, image)
        if query_patches is None:
            continue
        patch_scores = compute_patch_scores(
            query_patches,
            patch_memory_tensor.to(query_patches.device),
            metric="cosine",
            top_k=3,
        )
        scores.append(patch_scores.detach().cpu().numpy())

    if not scores:
        raise RuntimeError("No normal calibration scores collected")

    normal_matrix = np.stack(scores).astype(np.float32)
    center = np.median(normal_matrix, axis=0)
    mad = np.median(np.abs(normal_matrix - center), axis=0)
    scale = 1.4826 * mad
    positive_scale = scale[scale > 0]
    scale_floor = np.percentile(positive_scale, 10) if positive_scale.size else 1e-6
    scale = np.maximum(scale, max(float(scale_floor), 1e-6))
    
    all_normal_z = (normal_matrix - center) / scale
    all_normal_z = np.clip(all_normal_z, 0, None)
    baseline = np.quantile(all_normal_z, baseline_quantile, axis=1, keepdims=True)
    calibrated = np.clip(all_normal_z - baseline, 0, None)
    map_global_max = max(float(np.percentile(calibrated, map_percentile)), 1e-6)

    normal_image_scores = np.quantile(calibrated, 0.95, axis=1)
    score_threshold_raw = max(float(np.percentile(normal_image_scores, score_percentile)), 1e-6)
    
    return center, scale, map_global_max, score_threshold_raw


def _calibrate_patch_scores(query_patches, patch_memory_tensor, normal_center, normal_scale, baseline_quantile=0.5):
    """Return raw and calibrated patch-memory anomaly scores.

    The calibrated scores are the canonical gallery evidence used for score maps
    and final image-level scores.
    """
    raw_scores = compute_patch_scores(
        query_patches,
        patch_memory_tensor.to(query_patches.device),
        metric="cosine",
        top_k=3,
    )

    center = torch.as_tensor(normal_center, device=query_patches.device, dtype=query_patches.dtype)
    scale = torch.as_tensor(normal_scale, device=query_patches.device, dtype=query_patches.dtype)
    z = torch.clamp((raw_scores - center) / scale, min=0.0)
    baseline = torch.quantile(z, baseline_quantile)
    calibrated = torch.clamp(z - baseline, min=0.0)
    return raw_scores, calibrated


def _score_from_calibrated_patches(calibrated_scores, score_threshold_raw):
    """Return raw image score and normalized display score in [0, 1].

    The display scale maps the selected normal calibration percentile to 0.5.
    The actual anomalous/normal decision threshold is configurable.
    """
    raw_score = float(torch.quantile(calibrated_scores, 0.95).detach().cpu())
    display_score = min(raw_score / (2.0 * score_threshold_raw), 1.0)
    return raw_score, display_score


def _canonical_gallery_inference(
    image_tensor,
    clip_model,
    patch_memory_tensor,
    normal_center,
    normal_scale,
    map_global_max,
    score_threshold_raw,
    device,
    img_size=IMG_SIZE,
    baseline_quantile=0.5,
    threshold=DEFAULT_ANOMALY_THRESHOLD,
):
    """Compute the canonical gallery decision and visualization evidence.

    Returns one result object used by the score text, predicted label, score-map
    overlay, confusion counts, HTML row, and LLM prompt.

    1. Compute raw cosine distances to memory bank
    2. Z-score calibrate against normal stats
    3. Subtract baseline (median of Z-scores)
    4. Score = 95th percentile of calibrated patch scores normalized by the
       image-level normal calibration threshold
    5. Score map = calibrated patch scores normalized by map_global_max
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    _, query_patches = _extract_patch_features(clip_model, image_tensor)
    if query_patches is None:
        score_map = np.zeros((img_size, img_size), dtype=np.float32)
        original_np = _denormalize_image(image_tensor[0])
        return {
            "raw_patch_scores": None,
            "calibrated_patch_scores": None,
            "raw_image_score": 0.0,
            "score_map": score_map,
            "score_overlay": original_np,
            "anomaly_score": 0.0,
            "predicted_label": 0,
        }

    raw_scores, calibrated = _calibrate_patch_scores(
        query_patches,
        patch_memory_tensor,
        normal_center,
        normal_scale,
        baseline_quantile=baseline_quantile,
    )
    raw_image_score, anomaly_score = _score_from_calibrated_patches(
        calibrated,
        score_threshold_raw,
    )
    predicted_label = 1 if anomaly_score >= threshold else 0

    heatmap = scores_to_heatmap(
        calibrated,
        img_size=img_size,
        patch_size=14,
        sigma=12.0,
        global_max=map_global_max,
    )
    score_map = heatmap.squeeze().detach().cpu().numpy().astype(np.float32)
    original_np = _denormalize_image(image_tensor[0])
    score_overlay = overlay_heatmap(original_np, score_map, alpha=0.5)

    return {
        "raw_patch_scores": raw_scores.detach().cpu(),
        "calibrated_patch_scores": calibrated.detach().cpu(),
        "raw_image_score": raw_image_score,
        "score_map": score_map,
        "score_overlay": score_overlay,
        "anomaly_score": anomaly_score,
        "predicted_label": predicted_label,
    }


def _normalize_map_for_display(map_2d):
    """Normalize a 2D map to [0, 1] for display (handles signed SHAP maps)."""
    m = np.asarray(map_2d, dtype=np.float32)
    positive = np.maximum(m, 0.0)
    hi = positive.max()
    if hi > 1e-8:
        return positive / hi
    return np.zeros_like(positive)


def _status_from_labels(true_label, predicted_label):
    """Return confusion status from binary ground truth and prediction."""
    if true_label == 1 and predicted_label == 1:
        return "TP"
    if true_label == 0 and predicted_label == 0:
        return "TN"
    if true_label == 1 and predicted_label == 0:
        return "FN"
    return "FP"


def _confusion_summary_from_panels(panels):
    summary = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for panel in panels:
        status = _status_from_labels(panel["true_label"], panel["predicted_label"])
        summary[status] += 1
    return summary


def _position_from_grid_index(patch_idx, grid_size):
    G = max(1, int(grid_size))
    row = int(patch_idx) // G
    col = int(patch_idx) % G

    upper_cut = G / 3.0
    lower_cut = 2.0 * G / 3.0
    vertical = "upper" if row < upper_cut else ("lower" if row >= lower_cut else "middle")
    horizontal = "left" if col < upper_cut else ("right" if col >= lower_cut else "center")

    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}-{horizontal}"


def _strongest_map_region(map_2d, grid_size, positive_only=True):
    """Return the strongest coarse grid region in a map, or None if unfocused."""
    m = np.asarray(map_2d, dtype=np.float32)
    if m.ndim != 2 or m.size == 0:
        return None

    G = max(1, int(grid_size))
    h, w = m.shape
    y_edges = np.linspace(0, h, G + 1, dtype=int)
    x_edges = np.linspace(0, w, G + 1, dtype=int)

    best_idx = None
    best_val = -float("inf")
    for gy in range(G):
        for gx in range(G):
            patch = m[y_edges[gy]: y_edges[gy + 1], x_edges[gx]: x_edges[gx + 1]]
            if patch.size == 0:
                continue
            val = float(patch.mean())
            if positive_only and val <= 0.0:
                continue
            if val > best_val:
                best_val = val
                best_idx = gy * G + gx

    if best_idx is None or best_val <= 1e-8:
        return None
    return _position_from_grid_index(best_idx, G)


def _evidence_phrase(shap_region, gradcam_region):
    parts = []
    if shap_region:
        parts.append(f"SHAP highlights the {shap_region}")
    if gradcam_region:
        parts.append(f"Grad-CAM peaks in the {gradcam_region}")
    return " and ".join(parts) if parts else "XAI maps show no focused region"


def _humanize_defect(defect_type):
    return str(defect_type).replace("_", " ").strip() or "defect"


def _severity_label(anomaly_score, threshold):
    if anomaly_score >= 0.75:
        return "high"
    if anomaly_score >= threshold:
        return "moderate"
    return "low"


def _plausible_cause(defect_type):
    """Return a controlled cause phrase keyed by known MVTec defect labels."""
    defect = str(defect_type).lower()
    rules = [
        (("crack", "broken", "hole", "poke"), "impact or material stress"),
        (("scratch", "cut", "rough"), "abrasion or handling wear"),
        (("contamination", "glue", "oil", "liquid", "color", "gray", "print", "imprint"),
         "process residue or surface-treatment variation"),
        (("bent", "misplaced", "flip", "squeeze", "manipulated", "cable_swap", "missing"),
         "assembly alignment or handling variation"),
        (("thread", "fabric", "teeth"), "local material or weave irregularity"),
    ]
    for keywords, cause in rules:
        if any(keyword in defect for keyword in keywords):
            return cause
    return "localized manufacturing variation"


def _build_deterministic_explanation(
    category,
    defect_type,
    anomaly_score,
    threshold,
    gradcam_map,
    shap_map,
    grid_size,
    predicted_label,
):
    """Build a constrained gallery explanation from measured XAI evidence."""
    shap_region = _strongest_map_region(shap_map, grid_size, positive_only=True)
    gradcam_region = _strongest_map_region(gradcam_map, grid_size, positive_only=False)

    if defect_type == "good":
        return _normal_case_explanation(
            predicted_label,
            anomaly_score,
            threshold,
            shap_region=shap_region,
            gradcam_region=gradcam_region,
        )

    defect = _humanize_defect(defect_type)
    evidence = _evidence_phrase(shap_region, gradcam_region)

    if predicted_label == 0:
        return (
            f"Low evidence (score {anomaly_score:.2f}) for labeled {defect} in "
            f"the {category}; score is below threshold {threshold:.2f}, while "
            f"{evidence}."
        )

    severity = _severity_label(anomaly_score, threshold)
    cause = _plausible_cause(defect_type)
    return (
        f"{severity.capitalize()} evidence (score {anomaly_score:.2f}) for {defect} "
        f"in the {category}: {evidence}, consistent with {cause}."
    )


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def _render_panel(
    original_np,
    score_overlay,
    gradcam_map,
    shap_map,
    explanation,
    anomaly_score,
    true_label,
    predicted_label,
    defect_type="",
    threshold=DEFAULT_ANOMALY_THRESHOLD,
    grid_size=None,
):
    """Render a single panel as a matplotlib figure: 1x4 grid + text."""
    fig, axes = mpl_plt.subplots(1, 4, figsize=(16, 4))

    # Original
    axes[0].imshow(original_np)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Score Map Overlay
    if score_overlay is not None:
        axes[1].imshow(score_overlay)
    else:
        axes[1].imshow(original_np)
    axes[1].set_title("Score Map", fontsize=10)
    axes[1].axis("off")

    jet = matplotlib.colormaps["jet"]

    # Grad-CAM Overlay
    gradcam_display = _normalize_map_for_display(gradcam_map)
    gradcam_colored = jet(gradcam_display)[:, :, :3]
    grad_overlay = (0.5 * original_np.astype(np.float32) / 255.0 + 0.5 * gradcam_colored)
    grad_overlay = np.clip(grad_overlay, 0, 1)
    axes[2].imshow(grad_overlay)
    axes[2].set_title("Grad-CAM", fontsize=10)
    axes[2].axis("off")

    shap_display = _normalize_map_for_display(shap_map)
    shap_colored = jet(shap_display)[:, :, :3]
    shap_overlay = (0.5 * original_np.astype(np.float32) / 255.0 + 0.5 * shap_colored)
    shap_overlay = np.clip(shap_overlay, 0, 1)
    axes[3].imshow(shap_overlay)
    grid_label = f"{grid_size}x{grid_size}" if grid_size else "grid"
    axes[3].set_title(f"SHAP ({grid_label})", fontsize=10)
    axes[3].axis("off")

    # Explanation text below
    status = _status_from_labels(true_label, predicted_label)

    text_str = (
        f"Category: {defect_type or 'N/A'}  |  "
        f"Anomaly Score: {anomaly_score:.4f}  |  Threshold: {threshold:.2f}  |  "
        f"True: {'Anomalous' if true_label == 1 else 'Normal'}  |  "
        f"Predicted: {'Anomalous' if predicted_label == 1 else 'Normal'}  |  "
        f"Status: {status}\n"
        f"Explanation: {explanation}"
    )
    fig.text(0.5, 0.01, text_str, ha="center", va="bottom", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    mpl_plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Image selection: pick diverse anomalous images + 1 normal
# ---------------------------------------------------------------------------

def _select_test_images(test_ds, category, max_anomalous=4, max_normal=1):
    """
    Select test images: up to `max_anomalous` anomalous (one per defect type)
    and `max_normal` normal images.
    Returns list of (index, image_tensor, mask_tensor, label, defect_type).
    """
    selected = []
    defect_seen = set()
    # FIX 6: use plain counters instead of O(n²) list comprehensions on every iteration
    anomalous_count = 0
    normal_count = 0

    for idx in range(len(test_ds)):
        # Early exit once both quotas are filled
        if anomalous_count >= max_anomalous and normal_count >= max_normal:
            break

        image_tensor, mask_tensor, label = test_ds[idx]
        label_int = int(label.item())

        if label_int == 0:
            if normal_count < max_normal:
                selected.append((idx, image_tensor, mask_tensor, label, "good"))
                normal_count += 1
            continue

        # Anomalous: extract defect type from image path
        if anomalous_count >= max_anomalous:
            continue

        img_path = test_ds.image_paths[idx]
        defect_type = os.path.basename(os.path.dirname(img_path))

        if defect_type not in defect_seen:
            selected.append((idx, image_tensor, mask_tensor, label, defect_type))
            defect_seen.add(defect_type)
            anomalous_count += 1

    return selected


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _generate_html(all_panels, html_path, heatmap_dir, threshold):
    """
    Generate a summary HTML file with embedded images, color-coded status,
    sortable by category and anomaly score.
    """
    del heatmap_dir

    summary = _confusion_summary_from_panels(all_panels)
    total = sum(summary.values())
    accuracy = (summary["TP"] + summary["TN"]) / total if total else 0.0
    recall = summary["TP"] / (summary["TP"] + summary["FN"]) if (summary["TP"] + summary["FN"]) else 0.0
    precision = summary["TP"] / (summary["TP"] + summary["FP"]) if (summary["TP"] + summary["FP"]) else 0.0

    rows_html = []
    for panel in all_panels:
        cat = panel["category"]
        score = panel["anomaly_score"]
        true_label = panel["true_label"]
        predicted_label = panel["predicted_label"]
        explanation = panel["explanation"]
        defect_type = panel["defect_type"]
        img_path = panel["img_path"]

        status = _status_from_labels(true_label, predicted_label)
        if status == "TP":
            row_color = "#e6ffe6"
        elif status == "TN":
            row_color = "#e6f0ff"
        elif status == "FN":
            row_color = "#ffe6e6"
        else:
            row_color = "#fff3e0"

        true_str = "Anomalous" if true_label == 1 else "Normal"
        pred_str = "Anomalous" if predicted_label == 1 else "Normal"

        rel_img = html.escape(os.path.relpath(img_path, os.path.dirname(html_path)), quote=True)
        cat_html = html.escape(cat, quote=True)
        defect_html = html.escape(defect_type)
        explanation_html = html.escape(explanation)

        rows_html.append(f"""
        <tr style="background-color:{row_color}" data-score="{score:.4f}" data-category="{cat_html}" data-status="{status}">
            <td><img src="{rel_img}" width="320" loading="lazy"></td>
            <td>{cat_html}</td>
            <td>{defect_html}</td>
            <td>{score:.4f}</td>
            <td>{true_str}</td>
            <td>{pred_str}</td>
            <td><strong>{status}</strong></td>
            <td style="max-width:340px;font-size:12px;">{explanation_html}</td>
        </tr>""")

    rows_str = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XAI Gallery - MVTec AD</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #fafafa; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #333; color: #fff; padding: 8px; position: sticky; top: 0; cursor: pointer; }}
  th:hover {{ background: #555; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  .summary {{ background: #fff; border: 1px solid #ddd; padding: 10px 12px; margin-bottom: 15px; }}
  .summary span {{ display: inline-block; margin-right: 18px; }}
  .controls {{ margin-bottom: 15px; }}
  .controls select, .controls button {{ padding: 6px 12px; margin-right: 8px; font-family: monospace; }}
  .legend {{ margin-bottom: 15px; font-size: 13px; }}
  .legend span {{ display: inline-block; padding: 3px 8px; margin-right: 10px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>XAI Defect Inspection Gallery</h1>

<div class="summary">
  <div><strong>Decision threshold:</strong> score &gt;= {threshold:.2f} is anomalous.</div>
  <span>Total: {total}</span>
  <span>TP: {summary['TP']}</span>
  <span>FN: {summary['FN']}</span>
  <span>TN: {summary['TN']}</span>
  <span>FP: {summary['FP']}</span>
  <span>Accuracy: {accuracy:.3f}</span>
  <span>Recall: {recall:.3f}</span>
  <span>Precision: {precision:.3f}</span>
</div>

<div class="legend">
  <span style="background:#e6ffe6;">TP: Correctly detected anomaly</span>
  <span style="background:#ffe6e6;">FN: Missed defect</span>
  <span style="background:#e6f0ff;">TN: Correctly identified normal</span>
  <span style="background:#fff3e0;">FP: False alarm</span>
</div>

<div class="controls">
  <label>Filter by category:
    <select id="catFilter" onchange="filterTable()">
      <option value="all">All</option>
      {" ".join(f'<option value="{html.escape(c, quote=True)}">{html.escape(c)}</option>' for c in CATEGORIES)}
    </select>
  </label>
  <label>Filter by status:
    <select id="statusFilter" onchange="filterTable()">
      <option value="all">All</option>
      <option value="TP">TP</option>
      <option value="FN">FN</option>
      <option value="TN">TN</option>
      <option value="FP">FP</option>
    </select>
  </label>
  <button onclick="sortTable('score')">Sort by Score</button>
  <button onclick="sortTable('category')">Sort by Category</button>
  <button onclick="sortTable('status')">Sort by Status</button>
</div>

<table id="galleryTable">
  <thead>
    <tr>
      <th onclick="sortTable('img')">Panel</th>
      <th onclick="sortTable('category')">Category</th>
      <th>Defect Type</th>
      <th onclick="sortTable('score')">Anomaly Score</th>
      <th>True Label</th>
      <th>Predicted</th>
      <th onclick="sortTable('status')">Status</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
{rows_str}
  </tbody>
</table>

<script>
let sortAsc = true;
function sortTable(key) {{
  const tbody = document.querySelector('#galleryTable tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    let va, vb;
    if (key === 'score') {{
      va = parseFloat(a.dataset.score);
      vb = parseFloat(b.dataset.score);
    }} else if (key === 'category') {{
      va = a.dataset.category;
      vb = b.dataset.category;
    }} else if (key === 'status') {{
      va = a.dataset.status;
      vb = b.dataset.status;
    }} else {{
      return 0;
    }}
    if (typeof va === 'string') {{
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }}
    return sortAsc ? va - vb : vb - va;
  }});
  sortAsc = !sortAsc;
  rows.forEach(r => tbody.appendChild(r));
}}

function filterTable() {{
  const cat = document.getElementById('catFilter').value;
  const status = document.getElementById('statusFilter').value;
  const rows = document.querySelectorAll('#galleryTable tbody tr');
  rows.forEach(r => {{
    const showCat = cat === 'all' || r.dataset.category === cat;
    const showStatus = status === 'all' || r.dataset.status === status;
    r.style.display = (showCat && showStatus) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html_content)
    logger.info("HTML gallery saved to %s", html_path)


# ---------------------------------------------------------------------------
# Main gallery generation
# ---------------------------------------------------------------------------

def generate_gallery(
    data_dir,
    output_heatmap_dir="./outputs/heatmaps",
    output_results_dir="./outputs/results",
    device=None,
    n_shots=32,
    grid_size=13,
    n_evals=200,
    llm_model="Qwen/Qwen2.5-0.5B-Instruct",
    use_4bit=False,
    max_anomalous=5,
    max_normal=1,
    threshold=DEFAULT_ANOMALY_THRESHOLD,
):
    """Generate the full XAI gallery across all 15 MVTec AD categories."""
    threshold = float(threshold)
    logger.info("Gallery decision threshold: %.4f", threshold)

    # Load backbone and model
    clip_model, _, _, device = load_backbone(device)
    adapt_model = AdaptCLIPModel(clip_model, device).to(device)
    adapt_model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])

    os.makedirs(output_heatmap_dir, exist_ok=True)
    os.makedirs(output_results_dir, exist_ok=True)

    all_panels = []
    tp, fp, tn, fn = 0, 0, 0, 0
    for cat_idx, cat in enumerate(CATEGORIES):
        logger.info("=" * 60)
        logger.info("Category %d/%d: %s", cat_idx + 1, len(CATEGORIES), cat)
        logger.info("=" * 60)
        cat_start = time.perf_counter()

        # Build memory bank from support set (first n_shots images)
        train_ds = MVTecDataset(data_dir, cat, split="train", transform=transform)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        memory = MemoryBank(feat_dim=768, mode="hybrid")
        memory.build(clip_model, train_loader, n_shots, device)
        patch_memory_tensor = memory.get_patch_bank().to(device)
        logger.info("Memory bank: %d vectors for %s (support set: first %d images)", memory.size, cat, n_shots)

        # Build calibration stats from disjoint set (skip first n_shots images)
        calib_indices = list(range(n_shots, len(train_ds)))
        calib_subset = torch.utils.data.Subset(train_ds, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=1, shuffle=False)
        normal_center, normal_scale, map_global_max, score_threshold_raw = _build_patch_calibration(
            clip_model, calib_loader, patch_memory_tensor, device
        )
        logger.info(
            "Patch calibration built for %s "
            "(calibration set: %d images, map_global_max=%.4f, score_threshold_raw=%.4f)",
            cat, len(calib_indices), map_global_max, score_threshold_raw
        )

        # Load test set
        test_ds = MVTecDataset(
            data_dir, cat, split="test",
            transform=transform, mask_transform=mask_transform,
        )

        # Select images
        selected = _select_test_images(test_ds, cat, max_anomalous, max_normal)
        logger.info("Selected %d images for %s", len(selected), cat)

        # Initialize explainers
        gradcam_gen = CLIPGradCAM(adapt_model)
        shap_gen = PatchSHAPExplainer(adapt_model, memory, cat, grid_size=grid_size)

        category_panels = []

        for sel_idx, (ds_idx, image_tensor, mask_tensor, label, defect_type) in enumerate(selected):
            image_tensor = image_tensor.to(device)
            true_label = int(label.item())
            label_kind = "anomalous" if true_label == 1 else "normal"

            logger.info(
                "  [%d/%d] %s image (idx=%d, defect=%s)...",
                sel_idx + 1, len(selected), label_kind, ds_idx, defect_type,
            )

            try:
                canonical = _canonical_gallery_inference(
                    image_tensor,
                    clip_model,
                    patch_memory_tensor,
                    normal_center,
                    normal_scale,
                    map_global_max,
                    score_threshold_raw,
                    device,
                    IMG_SIZE,
                    threshold=threshold,
                )
                anomaly_score = canonical["anomaly_score"]
                predicted_label = canonical["predicted_label"]
                score_overlay = canonical["score_overlay"]

                image_batch = image_tensor if image_tensor.dim() == 4 else image_tensor.unsqueeze(0)
                original_np = _denormalize_image(image_batch[0])
                gradcam_map = gradcam_gen.generate(
                    image_batch,
                    memory,
                    cat,
                    img_size=IMG_SIZE,
                    score_mode="patch",
                )
                shap_map = shap_gen.explain(original_np, n_evals=n_evals)
                explanation = _build_deterministic_explanation(
                    cat,
                    defect_type,
                    anomaly_score,
                    threshold,
                    gradcam_map,
                    shap_map,
                    grid_size,
                    predicted_label,
                )
            except Exception as e:
                logger.error("Failed on %s idx %d: %s", cat, ds_idx, e)
                continue

            # Update confusion counts using the same thresholded prediction shown in the panel.
            status = _status_from_labels(true_label, predicted_label)
            if status == "TP":
                tp += 1
            elif status == "TN":
                tn += 1
            elif status == "FN":
                fn += 1
            else:
                fp += 1

            # Render panel figure
            fig = _render_panel(
                original_np, score_overlay, gradcam_map, shap_map,
                explanation, anomaly_score, true_label, predicted_label,
                defect_type=defect_type, threshold=threshold, grid_size=grid_size,
            )

            # Save panel as individual PNG (for HTML embedding)
            panel_filename = f"{cat}_{sel_idx:02d}_{defect_type}.png"
            panel_path = os.path.join(output_heatmap_dir, panel_filename)
            fig.savefig(panel_path, dpi=120, bbox_inches="tight", facecolor="white")
            mpl_plt.close(fig)

            category_panels.append({
                "category": cat,
                "defect_type": defect_type,
                "anomaly_score": anomaly_score,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "explanation": explanation,
                "img_path": panel_path,
            })

            print(
                f"  [{cat}] {defect_type}: score={anomaly_score:.4f} "
                f"true={'anom' if true_label == 1 else 'norm'} "
                f"pred={'anom' if predicted_label == 1 else 'norm'}"
            )

        # Save per-category tall gallery image
        if category_panels:
            _save_category_gallery(category_panels, cat, output_heatmap_dir)

        all_panels.extend(category_panels)

        # Cleanup
        del gradcam_gen, shap_gen, train_ds, test_ds, train_loader, memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cat_elapsed = time.perf_counter() - cat_start
        logger.info("Category %s complete in %.1fs", cat, cat_elapsed)

    # Generate summary HTML
    html_path = os.path.join(output_results_dir, "xai_gallery.html")
    _generate_html(all_panels, html_path, output_heatmap_dir, threshold)

    # Print confusion summary
    total = tp + fp + tn + fn
    print("\n" + "=" * 60)
    print("CONFUSION SUMMARY (threshold = {:.2f})".format(threshold))
    print("=" * 60)
    print(f"  True Positives  (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Total images:         {total}")
    if total > 0:
        print(f"  Accuracy:             {(tp + tn) / total:.4f}")
        if (tp + fn) > 0:
            print(f"  Recall (TPR):         {tp / (tp + fn):.4f}")
        if (tp + fp) > 0:
            print(f"  Precision:            {tp / (tp + fp):.4f}")
    print("=" * 60)

    return all_panels


def _save_category_gallery(panels, category, output_dir):
    """Combine all panels for a category into one tall PNG."""
    n = len(panels)
    fig, axes = mpl_plt.subplots(n, 1, figsize=(16, 5 * n))
    if n == 1:
        axes = [axes]

    for i, panel in enumerate(panels):
        import matplotlib.image as mpimg
        img = mpimg.imread(panel["img_path"])
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"{panel['category']} | {panel['defect_type']} | "
            f"score={panel['anomaly_score']:.4f} | {panel['explanation'][:80]}",
            fontsize=9, fontfamily="monospace",
        )

    mpl_plt.tight_layout()
    gallery_path = os.path.join(output_dir, f"{category}_xai_gallery.png")
    fig.savefig(gallery_path, dpi=120, bbox_inches="tight", facecolor="white")
    mpl_plt.close(fig)
    logger.info("Saved category gallery: %s", gallery_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive XAI visual gallery for MVTec AD"
    )
    parser.add_argument("--data_dir", required=True, help="Path to mvtec_anomaly_detection/")
    parser.add_argument("--output_heatmap_dir", default="./outputs/heatmaps")
    parser.add_argument("--output_results_dir", default="./outputs/results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_ANOMALY_THRESHOLD,
                        help="Anomaly decision threshold; scores >= threshold are anomalous")
    parser.add_argument("--n_shots", type=int, default=32, help="Normal shots for memory bank")
    parser.add_argument("--grid_size", type=int, default=13, help="SHAP patch grid size")
    parser.add_argument("--n_evals", type=int, default=200, help="SHAP evaluations per image")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Reserved for optional LLM rewriting; gallery explanations are deterministic")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Reserved for optional LLM rewriting; deterministic gallery does not load an LLM")
    parser.add_argument("--max_anomalous", type=int, default=5,
                        help="Max anomalous images per category")
    parser.add_argument("--max_normal", type=int, default=1,
                        help="Max normal images per category")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    generate_gallery(
        data_dir=args.data_dir,
        output_heatmap_dir=args.output_heatmap_dir,
        output_results_dir=args.output_results_dir,
        device=args.device,
        n_shots=args.n_shots,
        grid_size=args.grid_size,
        n_evals=args.n_evals,
        llm_model=args.llm_model,
        use_4bit=args.use_4bit,
        max_anomalous=args.max_anomalous,
        max_normal=args.max_normal,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
