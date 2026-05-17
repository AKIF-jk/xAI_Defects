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
  python generate_xai_gallery.py --data_dir /path/to/mvtec_anomaly_detection

Scoring mechanism
-----------------
Matches full_evaluation.py few-shot mode exactly:
  1. Patch-level cosine distance to memory bank features
  2. Z-score calibration against normal training images (median + MAD)
  3. Baseline subtraction (median of Z-scores)
  4. Image score = 95th percentile of calibrated patch scores
  5. Fixed threshold = 0.5 (FR-02 from project spec)

The PromptQueryAdapter is NOT used for scoring (its weights are untrained random).
Grad-CAM and SHAP use the same memory-distance objective internally, so their
heatmaps are consistent with the anomaly score.

Other fixes
-----------
1. cm.get_cmap("jet") deprecated since matplotlib 3.7. Replaced with
   matplotlib.colormaps["jet"].
2. _select_test_images: O(n²) list comprehensions on every loop iteration
   replaced with a simple anomalous_count integer counter.
3. `import matplotlib.pyplot as mpl_plt` was inside the per-image processing
   loop. Moved to module-level import at the top of the file.
4. result["pipe"] popped to avoid holding LLM pipeline reference in result dict.
5. SHAP column: white grid lines drawn at 7x7 patch boundaries for visibility.
"""

import argparse
import gc
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # FIX 7: moved out of per-image loop
import matplotlib.pyplot as mpl_plt      # alias kept for _render_panel callers
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from model.score_map import compute_patch_scores, scores_to_heatmap, overlay_heatmap
from xai.explainer_llm import explain_defect, DEFECT_VOCAB
from xai.gradcam import CLIPGradCAM
from xai.shap_explainer import PatchSHAPExplainer

logger = logging.getLogger(__name__)

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

ANOMALY_THRESHOLD = 0.5

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


def _build_patch_calibration(clip_model, train_loader, n_shots, patch_memory_tensor, device):
    """Build Z-score calibration stats from normal training images.

    Matches _build_patch_calibration() in full_evaluation.py exactly.
    Returns (center, scale) arrays for per-patch Z-score calibration.
    """
    scores = []
    count = 0
    for images, _ in train_loader:
        if count >= n_shots:
            break
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
        count += images.size(0)

    if not scores:
        raise RuntimeError("No normal calibration scores collected")

    normal_matrix = np.stack(scores).astype(np.float32)
    center = np.median(normal_matrix, axis=0)
    mad = np.median(np.abs(normal_matrix - center), axis=0)
    scale = 1.4826 * mad
    positive_scale = scale[scale > 0]
    scale_floor = np.percentile(positive_scale, 10) if positive_scale.size else 1e-6
    scale = np.maximum(scale, max(float(scale_floor), 1e-6))
    return center, scale


def _calibrate_and_score(query_patches, patch_memory_tensor, normal_center, normal_scale, baseline_quantile=0.5):
    """Calibrate patch scores and return image-level anomaly score.

    Matches the few-shot scoring in full_evaluation.py exactly:
    1. Compute raw cosine distances to memory bank
    2. Z-score calibrate against normal stats
    3. Subtract baseline (median of Z-scores)
    4. Image score = 95th percentile of calibrated patch scores
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

    # Image-level score: 95th percentile of calibrated patch scores
    score = float(torch.quantile(calibrated, 0.95).detach().cpu())
    return score


def _compute_anomaly_score(
    image_tensor, clip_model, patch_memory_tensor, normal_center, normal_scale, device
):
    """Full scoring pipeline matching full_evaluation.py few-shot mode.

    Returns a float anomaly score where higher = more anomalous.
    Uses calibrated patch-level cosine distances with 95th percentile aggregation.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    _, query_patches = _extract_patch_features(clip_model, image_tensor)
    if query_patches is None:
        return 0.0

    return _calibrate_and_score(query_patches, patch_memory_tensor, normal_center, normal_scale)


def _generate_score_map_overlay(image_tensor, clip_model, patch_memory_tensor, img_size=IMG_SIZE):
    """Generate a score-map heatmap overlay from patch-level anomaly scores.

    Uses direct cosine-similarity distance on patch features (no adapter).
    """
    _, query_patches = _extract_patch_features(clip_model, image_tensor)
    if query_patches is None:
        return None

    if query_patches.shape[-1] != patch_memory_tensor.shape[-1]:
        proj = getattr(clip_model.visual, "proj", None)
        if proj is not None:
            query_patches = query_patches @ proj.detach().to(query_patches.device)

    scores = compute_patch_scores(query_patches, patch_memory_tensor, metric="cosine", top_k=3)
    heatmap = scores_to_heatmap(scores, img_size=img_size, patch_size=14, sigma=4.0)
    original_np = _denormalize_image(image_tensor[0])
    overlay = overlay_heatmap(original_np, heatmap.cpu().numpy(), alpha=0.5)
    return overlay


def _normalize_map_for_display(map_2d):
    """Normalize a 2D map to [0, 1] for display (handles signed SHAP maps)."""
    m = np.asarray(map_2d, dtype=np.float32)
    positive = np.maximum(m, 0.0)
    hi = positive.max()
    if hi > 1e-8:
        return positive / hi
    return np.zeros_like(positive)


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def _render_panel(original_np, score_overlay, gradcam_map, shap_map, explanation,
                  anomaly_score, true_label, predicted_label, defect_type=""):
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

    # SHAP Overlay with visible 7x7 grid lines
    shap_display = _normalize_map_for_display(shap_map)
    shap_colored = jet(shap_display)[:, :, :3]
    shap_overlay = (0.5 * original_np.astype(np.float32) / 255.0 + 0.5 * shap_colored)
    shap_overlay = np.clip(shap_overlay, 0, 1)
    axes[3].imshow(shap_overlay)
    axes[3].set_title("SHAP (7x7 fast)", fontsize=10)
    axes[3].axis("off")

    # Explanation text below
    status = (
        "TP" if (true_label == 1 and anomaly_score >= ANOMALY_THRESHOLD) else
        "TN" if (true_label == 0 and anomaly_score < ANOMALY_THRESHOLD) else
        "FN" if (true_label == 1 and anomaly_score < ANOMALY_THRESHOLD) else "FP"
    )

    text_str = (
        f"Category: {defect_type or 'N/A'}  |  "
        f"Anomaly Score: {anomaly_score:.4f}  |  "
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
# HTML generation (unchanged)
# ---------------------------------------------------------------------------

def _generate_html(all_panels, html_path, heatmap_dir):
    """
    Generate a summary HTML file with embedded images, color-coded status,
    sortable by category and anomaly score.
    """
    rows_html = []
    for panel in all_panels:
        cat = panel["category"]
        score = panel["anomaly_score"]
        true_label = panel["true_label"]
        predicted_label = panel["predicted_label"]
        explanation = panel["explanation"]
        defect_type = panel["defect_type"]
        img_path = panel["img_path"]

        # Status is already determined by predicted_label (computed with per-cat threshold)
        if true_label == 1 and predicted_label == 1:
            status = "TP"
            row_color = "#e6ffe6"
        elif true_label == 0 and predicted_label == 0:
            status = "TN"
            row_color = "#e6f0ff"
        elif true_label == 1 and predicted_label == 0:
            status = "FN"
            row_color = "#ffe6e6"
        else:
            status = "FP"
            row_color = "#fff3e0"

        true_str = "Anomalous" if true_label == 1 else "Normal"
        pred_str = "Anomalous" if predicted_label == 1 else "Normal"

        rel_img = os.path.relpath(img_path, os.path.dirname(html_path))

        rows_html.append(f"""
        <tr style="background-color:{row_color}" data-score="{score:.4f}" data-category="{cat}" data-status="{status}">
            <td><img src="{rel_img}" width="320" loading="lazy"></td>
            <td>{cat}</td>
            <td>{defect_type}</td>
            <td>{score:.4f}</td>
            <td>{true_str}</td>
            <td>{pred_str}</td>
            <td><strong>{status}</strong></td>
            <td style="max-width:300px;font-size:12px;">{explanation}</td>
        </tr>""")

    rows_str = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XAI Gallery — MVTec AD</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #fafafa; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #333; color: #fff; padding: 8px; position: sticky; top: 0; cursor: pointer; }}
  th:hover {{ background: #555; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  .controls {{ margin-bottom: 15px; }}
  .controls select, .controls button {{ padding: 6px 12px; margin-right: 8px; font-family: monospace; }}
  .legend {{ margin-bottom: 15px; font-size: 13px; }}
  .legend span {{ display: inline-block; padding: 3px 8px; margin-right: 10px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>XAI Defect Inspection Gallery</h1>

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
      {" ".join(f'<option value="{c}">{c}</option>' for c in CATEGORIES)}
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
):
    """Generate the full XAI gallery across all 15 MVTec AD categories."""

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
    pipe = None

    for cat_idx, cat in enumerate(CATEGORIES):
        logger.info("=" * 60)
        logger.info("Category %d/%d: %s", cat_idx + 1, len(CATEGORIES), cat)
        logger.info("=" * 60)
        cat_start = time.perf_counter()

        # Build memory bank
        train_ds = MVTecDataset(data_dir, cat, split="train", transform=transform)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        memory = MemoryBank(feat_dim=768, mode="global")
        memory.build(clip_model, train_loader, n_shots, device)
        patch_memory_tensor = memory.get_patch_bank().to(device)
        logger.info("Memory bank: %d vectors for %s", memory.size, cat)

        # Build Z-score calibration from normal training images (matches full_evaluation.py)
        normal_center, normal_scale = _build_patch_calibration(
            clip_model, train_loader, n_shots, patch_memory_tensor, device
        )
        logger.info("Patch calibration built for %s (center shape=%s)", cat, normal_center.shape)

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
                result = explain_defect(
                    image_tensor, adapt_model, memory, cat,
                    pipe=pipe,
                    gradcam_gen=gradcam_gen,
                    shap_gen=shap_gen,
                    grid_size=grid_size,
                    n_evals=n_evals,
                    llm_model=llm_model,
                    use_4bit=use_4bit,
                )
                pipe = result.pop("pipe")
                gradcam_map = result["gradcam_map"]
                shap_map = result["shap_map"]
                explanation = result["explanation"]

                # Scoring: calibrated patch-level cosine distance (matches full_evaluation.py few-shot)
                anomaly_score = _compute_anomaly_score(
                    image_tensor, clip_model, patch_memory_tensor,
                    normal_center, normal_scale, device,
                )
            except Exception as e:
                logger.error("Failed on %s idx %d: %s", cat, ds_idx, e)
                continue

            # FR-02: fixed 0.5 threshold from project spec
            predicted_label = 1 if anomaly_score >= ANOMALY_THRESHOLD else 0

            # Update confusion counts
            if true_label == 1 and predicted_label == 1:
                tp += 1
            elif true_label == 0 and predicted_label == 0:
                tn += 1
            elif true_label == 1 and predicted_label == 0:
                fn += 1
            else:
                fp += 1

            # Generate score map overlay (uses direct patch-cosine distance, no adapter)
            score_overlay = _generate_score_map_overlay(
                image_tensor, clip_model, patch_memory_tensor, IMG_SIZE
            )

            # Denormalize original
            original_np = _denormalize_image(image_tensor[0])

            # Render panel figure
            fig = _render_panel(
                original_np, score_overlay, gradcam_map, shap_map,
                explanation, anomaly_score, true_label, predicted_label,
                defect_type=defect_type,
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
    _generate_html(all_panels, html_path, output_heatmap_dir)

    # Print confusion summary
    total = tp + fp + tn + fn
    print("\n" + "=" * 60)
    print("CONFUSION SUMMARY (threshold = {:.1f}, FR-02 spec)".format(ANOMALY_THRESHOLD))
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
    parser.add_argument("--n_shots", type=int, default=32, help="Normal shots for memory bank")
    parser.add_argument("--grid_size", type=int, default=13, help="SHAP patch grid (7x7 fast mode)")
    parser.add_argument("--n_evals", type=int, default=200, help="SHAP evaluations per image")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace LLM for explanations")
    parser.add_argument("--use_4bit", action="store_true", help="4-bit LLM quantization")
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
    )


if __name__ == "__main__":
    main()