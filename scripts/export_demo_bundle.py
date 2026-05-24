"""Export a small Hugging Face Spaces demo bundle from the full pipeline.

Run this in Colab after installing the heavyweight AdaptCLIP dependencies and
mounting Google Drive. The output directory can be copied into the Space repo:

    python scripts/export_demo_bundle.py \
      --data_dir /content/drive/MyDrive/defect_inspector/data/mvtec/mvtec_anomaly_detection \
      --output_dir /content/xAI_Defects \
      --categories bottle,cable,tile \
      --max_anomalous 2 \
      --max_normal 1

The generated public app reads ``demo_manifest.json`` and ``demo_assets/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from generate_xai_gallery import (  # noqa: E402
    CATEGORIES,
    DEFAULT_ANOMALY_THRESHOLD,
    IMG_SIZE,
    _build_deterministic_explanation,
    _build_patch_calibration,
    _canonical_gallery_inference,
    _denormalize_image,
    _normalize_map_for_display,
    _select_test_images,
)
from data.mvtec_dataset import MVTecDataset  # noqa: E402
from model.adaptclip import AdaptCLIPModel  # noqa: E402
from model.backbone import load_backbone  # noqa: E402
from model.memory_bank import MemoryBank  # noqa: E402
from xai.gradcam import CLIPGradCAM  # noqa: E402
from xai.shap_explainer import PatchSHAPExplainer  # noqa: E402


MODEL_VERSION = "adaptclip-vit-l14"
LOGGER = logging.getLogger("export-demo-bundle")


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower() or "sample"


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _overlay_positive_map(original: np.ndarray, map_2d: np.ndarray) -> np.ndarray:
    jet = matplotlib.colormaps["jet"]
    display = _normalize_map_for_display(map_2d)
    colored = jet(display)[:, :, :3]
    overlay = 0.5 * original.astype(np.float32) / 255.0 + 0.5 * colored
    return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


def _parse_categories(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(CATEGORIES)
    requested = [_slug(item).replace("-", "_") for item in raw.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(unknown)}")
    return requested


def export_demo_bundle(
    data_dir: str,
    output_dir: str,
    categories: list[str],
    device: str | None,
    n_shots: int,
    grid_size: int,
    n_evals: int,
    max_anomalous: int,
    max_normal: int,
    threshold: float,
) -> list[dict[str, object]]:
    output_root = Path(output_dir)
    asset_root = output_root / "demo_assets"
    manifest_path = output_root / "demo_manifest.json"
    threshold = float(threshold)

    clip_model, _, _, device = load_backbone(device)
    adapt_model = AdaptCLIPModel(clip_model, device).to(device)
    adapt_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )

    samples: list[dict[str, object]] = []
    for cat in categories:
        LOGGER.info("Exporting category: %s", cat)
        train_ds = MVTecDataset(data_dir, cat, split="train", transform=transform)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

        memory = MemoryBank(feat_dim=768, mode="hybrid")
        memory.build(clip_model, train_loader, n_shots, device)
        patch_memory_tensor = memory.get_patch_bank().to(device)

        calib_indices = list(range(n_shots, len(train_ds)))
        calib_loader = DataLoader(torch.utils.data.Subset(train_ds, calib_indices), batch_size=1)
        normal_center, normal_scale, map_global_max, score_threshold_raw = _build_patch_calibration(
            clip_model,
            calib_loader,
            patch_memory_tensor,
            device,
        )

        test_ds = MVTecDataset(
            data_dir,
            cat,
            split="test",
            transform=transform,
            mask_transform=mask_transform,
        )
        selected = _select_test_images(
            test_ds,
            cat,
            max_anomalous=max_anomalous,
            max_normal=max_normal,
        )

        gradcam_gen = CLIPGradCAM(adapt_model)
        shap_gen = PatchSHAPExplainer(adapt_model, memory, cat, grid_size=grid_size)

        for sample_num, (ds_idx, image_tensor, _mask_tensor, label, defect_type) in enumerate(selected):
            image_tensor = image_tensor.to(device)
            true_label_int = int(label.item())
            image_batch = image_tensor if image_tensor.dim() == 4 else image_tensor.unsqueeze(0)
            original_np = _denormalize_image(image_batch[0])

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
            anomaly_score = float(canonical["anomaly_score"])
            predicted_label = int(canonical["predicted_label"])
            heatmap_overlay = canonical["score_overlay"]

            gradcam_map = gradcam_gen.generate(
                image_batch,
                memory,
                cat,
                img_size=IMG_SIZE,
                score_mode="patch",
            )
            shap_map = shap_gen.explain(original_np, n_evals=n_evals)
            shap_overlay = _overlay_positive_map(original_np, shap_map)

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

            image_id = f"{cat}-{sample_num:02d}-{_slug(str(defect_type))}"
            sample_dir = asset_root / image_id
            original_path = sample_dir / "original.png"
            heatmap_path = sample_dir / "heatmap.png"
            shap_path = sample_dir / "shap.png"

            _save_rgb(original_path, original_np)
            _save_rgb(heatmap_path, heatmap_overlay)
            _save_rgb(shap_path, shap_overlay)

            samples.append(
                {
                    "category": cat,
                    "image_id": image_id,
                    "title": f"{cat} {defect_type}",
                    "defect_type": str(defect_type),
                    "true_label": "anomalous" if true_label_int == 1 else "normal",
                    "score": anomaly_score,
                    "threshold": threshold,
                    "defect_detected": bool(predicted_label == 1),
                    "explanation": explanation,
                    "original_path": str(original_path.relative_to(output_root)),
                    "heatmap_path": str(heatmap_path.relative_to(output_root)),
                    "shap_path": str(shap_path.relative_to(output_root)),
                    "model_version": MODEL_VERSION,
                    "shot_mode": f"{n_shots}-shot",
                    "source_note": "Generated from local MVTec AD data in Colab.",
                    "dataset_index": int(ds_idx),
                }
            )

    manifest = {
        "schema_version": 1,
        "generated_by": "scripts/export_demo_bundle.py",
        "samples": samples,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Export precomputed artifacts for demo_app.py")
    parser.add_argument("--data_dir", required=True, help="Path to mvtec_anomaly_detection/")
    parser.add_argument("--output_dir", default=".", help="Repo or Space root")
    parser.add_argument("--categories", default="bottle,cable,tile", help="Comma list or 'all'")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_ANOMALY_THRESHOLD)
    parser.add_argument("--n_shots", type=int, default=32)
    parser.add_argument("--grid_size", type=int, default=13)
    parser.add_argument("--n_evals", type=int, default=200)
    parser.add_argument("--max_anomalous", type=int, default=2)
    parser.add_argument("--max_normal", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = export_demo_bundle(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        categories=_parse_categories(args.categories),
        device=args.device,
        n_shots=args.n_shots,
        grid_size=args.grid_size,
        n_evals=args.n_evals,
        max_anomalous=args.max_anomalous,
        max_normal=args.max_normal,
        threshold=args.threshold,
    )
    print(f"Exported {len(samples)} samples to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
