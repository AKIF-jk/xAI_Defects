"""
Patch-grid SHAP-style explainer for MVTec gallery outputs.

The configured grid is the actual feature space: each cell is masked with a
blurred baseline and added back in random permutation order. Patch contributions
are the average marginal score deltas across complete grid permutations.
"""

import argparse
import gc
import logging
import os
import pickle
import sys
import time

logger = logging.getLogger(__name__)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from model.score_map import compute_patch_scores


class PatchSHAPExplainer:
    def __init__(self, model, memory_bank, class_name, grid_size=5):
        self.model = model
        self.clip_model = model.clip_model if hasattr(model, "clip_model") else model
        self.memory_bank = self._as_memory_tensor(memory_bank, self._device)
        self.patch_memory_bank = self._as_patch_memory_tensor(memory_bank, self._device)
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

    def _predict(self, masked_images):
        images = self._numpy_images_to_tensor(masked_images).to(self.memory_bank.device)
        with torch.no_grad():
            if self.patch_memory_bank is not None:
                patch_tokens = self._encode_patch_tokens(images)
                scores = self._patch_memory_scores(patch_tokens, self.patch_memory_bank)
            else:
                image_features = self.clip_model.encode_image(images)
                scores = self._memory_distance_score(image_features, self.memory_bank)
        return scores.detach().cpu().numpy().reshape(-1)

    def explain(self, image_numpy, n_evals=50):
        """
        Compute a per-pixel SHAP attribution map with true grid masking.

        Parameters
        ----------
        image_numpy : np.ndarray  shape [H, W, 3], uint8 or float32
        n_evals     : int  target number of model evaluations. At least one
                      complete grid permutation is evaluated, so total work is
                      at least grid_size ** 2 + 1 evaluations.

        Returns
        -------
        shap_map : np.ndarray  shape [H, W], float32
        """
        logger.info(
            "Starting SHAP explanation (n_evals=%d, grid_size=%d)", n_evals, self.grid_size
        )
        image_numpy = self._validate_image_numpy(image_numpy)
        shap_map = self._permutation_grid_shap(image_numpy, n_evals=n_evals)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SHAP explanation complete")
        return shap_map.astype(np.float32)

    def _permutation_grid_shap(self, image_numpy, n_evals=50):
        height, width = image_numpy.shape[:2]
        boxes = self._grid_boxes(height, width)
        n_features = len(boxes)
        evals_per_permutation = n_features + 1
        n_permutations = max(1, int(max(n_evals, evals_per_permutation) // evals_per_permutation))
        rng = np.random.default_rng(0)

        baseline = self._blur_baseline(image_numpy)
        patch_values = np.zeros(n_features, dtype=np.float32)

        for _ in range(n_permutations):
            order = rng.permutation(n_features)
            sequence = [baseline.copy()]
            current = baseline.copy()
            for patch_idx in order:
                x0, y0, x1, y1 = boxes[patch_idx]
                current = current.copy()
                current[y0:y1, x0:x1, :] = image_numpy[y0:y1, x0:x1, :]
                sequence.append(current)

            scores = self._predict_in_batches(np.stack(sequence, axis=0), batch_size=8)
            deltas = scores[1:] - scores[:-1]
            for patch_idx, delta in zip(order, deltas):
                patch_values[patch_idx] += float(delta)

        patch_values /= float(n_permutations)
        return self._patch_values_to_map(patch_values, height, width)

    def _predict_in_batches(self, images, batch_size=8):
        scores = []
        for start in range(0, len(images), batch_size):
            scores.append(self._predict(images[start:start + batch_size]))
        return np.concatenate(scores, axis=0)

    @staticmethod
    def _blur_baseline(image_numpy):
        image_u8 = np.clip(image_numpy, 0, 255).astype(np.uint8)
        blurred = Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(radius=5))
        return np.asarray(blurred).astype(np.float32)

    def _patch_values_to_map(self, patch_values, height, width):
        patch_map = np.zeros((height, width), dtype=np.float32)
        boxes = self._grid_boxes(height, width)
        for patch_idx, (x0, y0, x1, y1) in enumerate(boxes):
            patch_map[y0:y1, x0:x1] = patch_values[patch_idx]
        return patch_map

    def _grid_boxes(self, height, width):
        boxes = []
        y_edges = np.linspace(0, height, self.grid_size + 1, dtype=int)
        x_edges = np.linspace(0, width, self.grid_size + 1, dtype=int)
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y0, y1 = y_edges[gy], y_edges[gy + 1]
                x0, x1 = x_edges[gx], x_edges[gx + 1]
                boxes.append((x0, y0, x1, y1))
        return boxes

    @staticmethod
    def _validate_image_numpy(image_numpy):
        image_numpy = np.asarray(image_numpy)
        if image_numpy.ndim != 3 or image_numpy.shape[-1] != 3:
            raise ValueError("image_numpy must have shape [H, W, 3]")
        return np.clip(image_numpy.astype(np.float32), 0.0, 255.0)

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

    def _encode_patch_tokens(self, images):
        patch_features = [None]
        target = getattr(self.clip_model.visual, "ln_post", None)
        if target is None:
            raise RuntimeError("CLIP visual ln_post layer is required for patch SHAP scoring")

        def hook(module, inp, out):
            del module, inp
            patch_features[0] = self._standardize_patch_features(out)

        handle = target.register_forward_hook(hook)
        try:
            global_features = self.clip_model.encode_image(images)
        finally:
            handle.remove()

        tokens = patch_features[0]
        if tokens is None:
            raise RuntimeError("Patch SHAP scoring could not capture CLIP patch tokens")

        if tokens.shape[-1] != global_features.shape[-1]:
            proj = getattr(self.clip_model.visual, "proj", None)
            if proj is not None and tokens.shape[-1] == proj.shape[0]:
                tokens = tokens @ proj.detach().to(tokens.device)

        return self._drop_cls_token(tokens)

    @staticmethod
    def _patch_memory_scores(patch_tokens, patch_memory_bank):
        image_scores = []
        for image_patches in patch_tokens:
            patch_scores = compute_patch_scores(
                image_patches,
                patch_memory_bank.to(image_patches.device),
                metric="cosine",
                top_k=3,
            )
            image_scores.append(torch.quantile(patch_scores, 0.95))
        return torch.stack(image_scores)

    @staticmethod
    def _standardize_patch_features(out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.detach()
        if out.dim() == 3 and out.shape[0] > out.shape[1] and out.shape[0] > 32:
            out = out.permute(1, 0, 2)
        return out

    @staticmethod
    def _drop_cls_token(tokens):
        n_tokens = tokens.shape[1]
        grid_size = int((n_tokens - 1) ** 0.5)
        if grid_size * grid_size == n_tokens - 1:
            return tokens[:, 1:, :]

        grid_size = int(n_tokens ** 0.5)
        if grid_size * grid_size == n_tokens:
            return tokens

        raise RuntimeError(f"Token count {n_tokens} is not compatible with a square ViT grid")

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

    @staticmethod
    def _as_patch_memory_tensor(memory_bank, device):
        if not hasattr(memory_bank, "get_patch_bank"):
            return None

        memory = memory_bank.get_patch_bank()
        if memory.numel() == 0:
            return None
        return memory.float().to(device)


# ---------------------------------------------------------------------------
# Visualisation helpers (unchanged)
# ---------------------------------------------------------------------------

def visualize_shap(image, shap_map, top_k=5, grid_size=None):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3]")

    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    shap_map = np.asarray(shap_map, dtype=np.float32)
    if shap_map.shape != image_u8.shape[:2]:
        raise ValueError("shap_map must have shape [H, W] matching image")

    grid_size = int(grid_size or 5)
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


# ---------------------------------------------------------------------------
# Checkpoint helpers (unchanged)
# ---------------------------------------------------------------------------

def _checkpoint_path(output_dir):
    return os.path.join(output_dir, "checkpoint.pkl")


def _save_checkpoint(output_dir, rows, processed_indices, anomalous_count, normal_count):
    path = _checkpoint_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(
            rows=rows,
            processed_indices=processed_indices,
            anomalous_count=anomalous_count,
            normal_count=normal_count,
        ), f)
    logger.info("Checkpoint saved to %s", path)


def _load_checkpoint(output_dir):
    path = _checkpoint_path(output_dir)
    if not os.path.exists(path):
        logger.info("No checkpoint found at %s", path)
        return None, set(), 0, 0
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(
        "Loaded checkpoint from %s (%d images processed)",
        path, len(data["rows"]),
    )
    return (
        data["rows"],
        set(data["processed_indices"]),
        data["anomalous_count"],
        data["normal_count"],
    )


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def run_leather_shap_test(
    data_dir,
    output_dir="./outputs/shap",
    device=None,
    n_shots=4,
    grid_size=5,       # ↓ was 7
    n_evals=50,        # ↓ was 200
    resume=False,
):
    clip_model, _, _, device = load_backbone(device)
    adapt_model = AdaptCLIPModel(clip_model, device).to(device)
    adapt_model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        data_dir, "leather", split="test",
        transform=transform, mask_transform=mask_transform,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    explainer = PatchSHAPExplainer(adapt_model, memory, "leather", grid_size=grid_size)

    if resume:
        rows, processed_indices, anomalous_count, normal_count = _load_checkpoint(output_dir)
        rows = rows or []
    else:
        rows, processed_indices, anomalous_count, normal_count = [], set(), 0, 0

    for idx, (image_tensor, mask_tensor, label) in enumerate(test_loader):
        label_int = int(label.item())
        kind = "anomalous" if label_int == 1 else "normal"

        if idx in processed_indices:
            logger.info("Skipping already-processed image %d (%s)", idx, kind)
            continue

        if label_int == 1:
            if anomalous_count >= 3:
                continue
            anomalous_count += 1
        else:
            if normal_count >= 2:
                continue
            normal_count += 1

        logger.info(
            "Processing %s image %d (anomalous=%d, normal=%d)...",
            kind, idx, anomalous_count, normal_count,
        )
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
        if np.isnan(best_overlap_value):
            print(f"{kind} image {len(rows) + 1}: no GT defect mask, runtime={runtime:.2f}s")
        else:
            print(
                f"{kind} image {len(rows) + 1}: "
                f"best-GT-overlap patch SHAP={best_overlap_value:.6f}, "
                f"runtime={runtime:.2f}s"
            )

        rows.append((image_np.astype(np.uint8), heatmap, annotated, mask_np.astype(np.float32)))
        processed_indices.add(idx)
        _save_checkpoint(output_dir, rows, processed_indices, anomalous_count, normal_count)

        if anomalous_count >= 3 and normal_count >= 2:
            break

    if not rows:
        raise RuntimeError("No leather test images found for SHAP demo")

    logger.info("Saving SHAP panel with %d images to %s...", len(rows), output_dir)
    _save_shap_panel(rows, output_dir)


# ---------------------------------------------------------------------------
# Grid / metric helpers (unchanged)
# ---------------------------------------------------------------------------

def _grid_patch_values(shap_map, grid_size):
    h, w = shap_map.shape
    y_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_size + 1, dtype=int)
    values, boxes = [], []
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
    best_idx, best_overlap = None, 0
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        overlap = int(mask[y0:y1, x0:x1].sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    if best_idx is None or best_overlap == 0:
        return float("nan")
    return float(values[best_idx])


def _normalize_for_display(shap_map):
    positive = np.maximum(np.asarray(shap_map, dtype=np.float32), 0.0)
    hi = positive.max()
    return positive / hi if hi > 1e-8 else np.zeros_like(positive)


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
        for col_idx, (title, img) in enumerate(zip(titles, [original, heatmap, annotated, mask])):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap="jet" if col_idx == 1 else ("gray" if col_idx == 3 else None))
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "leather_shap_test.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    logger.info("SHAP demo complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SHAP demo for AdaptCLIP on MVTec leather (Colab free-tier optimised)"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/shap")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_shots", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=5,
                        help="Patch grid size (default 5, was 7)")
    parser.add_argument("--n_evals", type=int, default=50,
                        help="SHAP evaluations per image (default 50, was 200)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_leather_shap_test(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_shots=args.n_shots,
        grid_size=args.grid_size,
        n_evals=args.n_evals,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
