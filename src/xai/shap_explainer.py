import argparse
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank


class PatchSHAPExplainer:
    def __init__(self, model, memory_bank, class_name, grid_size=7):
        self.model = model
        self.clip_model = model.clip_model if hasattr(model, "clip_model") else model
        self.memory_bank = self._as_memory_tensor(memory_bank, self._device)
        self.class_name = class_name
        self.grid_size = int(grid_size)
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")

        self.model.eval()

    @property
    def _device(self):
        if hasattr(self.model, "device"):
            return torch.device(self.model.device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_segment_map(self, height, width):
        segment_map = np.zeros((height, width), dtype=np.int32)
        y_edges = np.linspace(0, height, self.grid_size + 1, dtype=int)
        x_edges = np.linspace(0, width, self.grid_size + 1, dtype=int)
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y0, y1 = y_edges[gy], y_edges[gy + 1]
                x0, x1 = x_edges[gx], x_edges[gx + 1]
                segment_map[y0:y1, x0:x1] = gy * self.grid_size + gx
        return segment_map

    def _predict(self, masked_images):
        images = self._numpy_images_to_tensor(masked_images).to(self.memory_bank.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            scores = self._memory_distance_score(image_features, self.memory_bank)
        return scores.detach().cpu().numpy().reshape(-1)

    def explain(self, image_numpy, n_evals=200):
        import shap

        logger.info("Starting SHAP explanation (n_evals=%d, grid_size=%d)", n_evals, self.grid_size)
        image_numpy = self._validate_image_numpy(image_numpy)
        h, w = image_numpy.shape[:2]
        masker = shap.maskers.Image("inpaint_telea", image_numpy.shape)
        explainer = shap.Explainer(self._predict, masker)
        logger.debug("Computing SHAP values with %d evaluations over %d features...", n_evals, self.grid_size ** 2)
        shap_values = explainer(image_numpy[np.newaxis], max_evals=n_evals)
        logger.debug("SHAP values computed, shape=%s", shap_values.values.shape)
        shap_map = self._values_to_map(shap_values.values, h, w)
        logger.info("SHAP explanation complete")
        return shap_map.astype(np.float32)

    def _values_to_map(self, values, height, width):
        values = np.asarray(values)
        if values.ndim == 2:
            values = values[0]
        if values.ndim != 1:
            raise RuntimeError(f"Unsupported SHAP value shape: {values.shape}")

        patch_map = np.zeros((height, width), dtype=np.float32)
        y_edges = np.linspace(0, height, self.grid_size + 1, dtype=int)
        x_edges = np.linspace(0, width, self.grid_size + 1, dtype=int)

        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y0, y1 = y_edges[gy], y_edges[gy + 1]
                x0, x1 = x_edges[gx], x_edges[gx + 1]
                patch_idx = gy * self.grid_size + gx
                patch_map[y0:y1, x0:x1] = values[patch_idx]

        return patch_map

    @staticmethod
    def _validate_image_numpy(image_numpy):
        image_numpy = np.asarray(image_numpy)
        if image_numpy.ndim != 3 or image_numpy.shape[-1] != 3:
            raise ValueError("image_numpy must have shape [H, W, 3]")
        image_numpy = image_numpy.astype(np.float32)
        return np.clip(image_numpy, 0.0, 255.0)

    @staticmethod
    def _numpy_images_to_tensor(images):
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[np.newaxis]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("masked_images must have shape [N, H, W, 3]")

        images = images.astype(np.float32)
        if images.max() <= 1.0:
            images = images * 255.0
        images = np.clip(images, 0.0, 255.0) / 255.0

        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def _memory_distance_score(query_feat, memory_bank, top_k=3):
        if memory_bank.dim() == 1:
            memory_bank = memory_bank.unsqueeze(0)
        k = min(max(int(top_k), 1), memory_bank.shape[0])
        diff = query_feat.float().unsqueeze(1) - memory_bank.float().unsqueeze(0)
        distances = diff.square().mean(dim=-1)
        nearest = distances.topk(k, dim=1, largest=False).values
        return nearest.mean(dim=1)

    @staticmethod
    def _as_memory_tensor(memory_bank, device):
        if isinstance(memory_bank, torch.Tensor):
            memory = memory_bank
        elif hasattr(memory_bank, "index"):
            vectors = memory_bank.index.reconstruct_n(0, memory_bank.index.ntotal)
            memory = torch.from_numpy(vectors)
        else:
            memory = torch.as_tensor(memory_bank)

        if memory.numel() == 0:
            raise ValueError("memory_bank is empty; SHAP needs at least one normal reference")
        return memory.float().to(device)


def visualize_shap(image, shap_map, top_k=5, grid_size=None):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3]")

    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    shap_map = np.asarray(shap_map, dtype=np.float32)
    if shap_map.shape != image_u8.shape[:2]:
        raise ValueError("shap_map must have shape [H, W] matching image")

    grid_size = int(grid_size or 7)
    patch_values, boxes = _grid_patch_values(shap_map, grid_size)
    positive_order = np.argsort(patch_values)[::-1]

    annotated = Image.fromarray(image_u8).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    drawn = 0
    for patch_idx in positive_order:
        if patch_values[patch_idx] <= 0:
            break
        x0, y0, x1, y1 = boxes[patch_idx]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=3)
        drawn += 1
        if drawn >= top_k:
            break

    return np.asarray(annotated)


def run_leather_shap_test(
    data_dir,
    output_dir="./outputs/shap",
    device=None,
    n_shots=8,
    grid_size=7,
    n_evals=200,
):
    clip_model, _, _, device = load_backbone(device)
    adapt_model = AdaptCLIPModel(clip_model, device).to(device)
    adapt_model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = MVTecDataset(data_dir, "leather", split="train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    logger.info("Building memory bank with n_shots=%d...", n_shots)
    memory = MemoryBank(feat_dim=768, mode="global")
    memory.build(clip_model, train_loader, n_shots, device)
    logger.info("Memory bank built with %d vectors", memory.size)

    test_ds = MVTecDataset(
        data_dir,
        "leather",
        split="test",
        transform=transform,
        mask_transform=mask_transform,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    explainer = PatchSHAPExplainer(adapt_model, memory, "leather", grid_size=grid_size)

    rows = []
    anomalous_count = 0
    normal_count = 0
    for idx, (image_tensor, mask_tensor, label) in enumerate(test_loader):
        label_int = int(label.item())
        kind = "anomalous" if label_int == 1 else "normal"
        if label_int == 1:
            if anomalous_count >= 3:
                continue
            anomalous_count += 1
        else:
            if normal_count >= 2:
                continue
            normal_count += 1

        logger.info("Processing %s image %d (anomalous=%d, normal=%d)...", kind, idx, anomalous_count, normal_count)
        image_tensor = image_tensor.to(device)
        image_np = _tensor_to_image_numpy(image_tensor[0])
        mask_np = mask_tensor.squeeze().cpu().numpy() > 0.5

        start = time.perf_counter()
        shap_map = explainer.explain(image_np, n_evals=n_evals)
        runtime = time.perf_counter() - start

        annotated = visualize_shap(image_np, shap_map, top_k=5, grid_size=grid_size)
        heatmap = _normalize_for_display(shap_map)
        best_overlap_value = _best_gt_overlap_patch_value(shap_map, mask_np, grid_size)

        logger.info("Finished %s image %d in %.2fs", kind, idx, runtime)
        kind = "anomalous" if label_int == 1 else "normal"
        if np.isnan(best_overlap_value):
            print(
                f"{kind} image {len(rows) + 1}: no GT defect mask, "
                f"runtime={runtime:.2f}s"
            )
        else:
            print(
                f"{kind} image {len(rows) + 1}: best-GT-overlap patch SHAP="
                f"{best_overlap_value:.6f}, runtime={runtime:.2f}s"
            )

        rows.append((image_np.astype(np.uint8), heatmap, annotated, mask_np.astype(np.float32)))
        if anomalous_count >= 3 and normal_count >= 2:
            break

    if not rows:
        raise RuntimeError("No leather test images found for SHAP demo")

    logger.info("Saving SHAP panel with %d images to %s...", len(rows), output_dir)
    _save_shap_panel(rows, output_dir)


def _grid_patch_values(shap_map, grid_size):
    h, w = shap_map.shape
    y_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_size + 1, dtype=int)
    values = []
    boxes = []

    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = y_edges[gy], y_edges[gy + 1]
            x0, x1 = x_edges[gx], x_edges[gx + 1]
            values.append(float(shap_map[y0:y1, x0:x1].mean()))
            boxes.append((x0, y0, x1, y1))

    return np.asarray(values, dtype=np.float32), boxes


def _best_gt_overlap_patch_value(shap_map, mask, grid_size):
    if mask is None or not np.any(mask):
        return float("nan")

    values, boxes = _grid_patch_values(shap_map, grid_size)
    best_idx = None
    best_overlap = 0
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        overlap = int(mask[y0:y1, x0:x1].sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is None or best_overlap == 0:
        return float("nan")
    return float(values[best_idx])


def _normalize_for_display(shap_map):
    shap_map = np.asarray(shap_map, dtype=np.float32)
    positive = np.maximum(shap_map, 0.0)
    hi = positive.max()
    if hi <= 1e-8:
        return np.zeros_like(positive)
    return positive / hi


def _tensor_to_image_numpy(tensor):
    tensor = tensor.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = torch.clamp(tensor * std + mean, 0, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _save_shap_panel(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["Original", "SHAP Heatmap", "Top-5 Patches", "GT Mask"]
    for row_idx, (original, heatmap, annotated, mask) in enumerate(rows):
        images = [original, heatmap, annotated, mask]
        for col_idx, (title, image) in enumerate(zip(titles, images)):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="jet" if col_idx == 1 else ("gray" if col_idx == 3 else None))
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "leather_shap_test.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    logger.info("SHAP demo complete")


def main():
    parser = argparse.ArgumentParser(description="SHAP demo for AdaptCLIP on MVTec leather")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/shap")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_shots", type=int, default=8)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--n_evals", type=int, default=200)
    args = parser.parse_args()

    run_leather_shap_test(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_shots=args.n_shots,
        grid_size=args.grid_size,
        n_evals=args.n_evals,
    )


if __name__ == "__main__":
    main()
