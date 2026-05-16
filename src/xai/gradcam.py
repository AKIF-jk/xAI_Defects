import argparse
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from model.score_map import compute_patch_scores, overlay_heatmap


class CLIPGradCAM:
    def __init__(self, model):
        try:
            from captum.attr import LayerGradCam
        except ImportError as exc:
            raise ImportError(
                "captum is required for CLIPGradCAM. Install it with `pip install captum`."
            ) from exc

        self.model = model
        self.clip_model = model.clip_model if hasattr(model, "clip_model") else model
        self.target_layer = self._find_target_layer()
        self.gradcam = LayerGradCam(self._forward_score, self.target_layer)

        self._memory_bank = None
        self._class_name = None
        self._last_activation = None
        self._activation_handle = None

    def generate(self, image_tensor, memory_bank, class_name, img_size=256, score_mode="global"):
        self.model.eval()
        score_mode = score_mode.lower()
        if score_mode == "patch":
            self._memory_bank = self._as_patch_memory_tensor(memory_bank, image_tensor.device)
        elif score_mode == "global":
            self._memory_bank = self._as_memory_tensor(memory_bank, image_tensor.device)
        else:
            raise ValueError(f"Unsupported Grad-CAM score_mode: {score_mode}")

        self._class_name = class_name
        self._score_mode = score_mode

        image_tensor = image_tensor.to(self._memory_bank.device)
        image_tensor = image_tensor.requires_grad_(True)

        with self._temporarily_unfreeze(self.target_layer):
            attribution = self.gradcam.attribute(
                image_tensor,
                attr_dim_summation=False,
            )

        gradcam_map = self._attribution_to_map(attribution, img_size)
        return gradcam_map

    def generate_scorecam(self, image_tensor, memory_bank, class_name, img_size=256):
        # Score-CAM is gradient-free but very slow on ViT-L/14 because it needs
        # one forward pass per activation channel, often ~50x slower than Grad-CAM.
        self.model.eval()
        memory_tensor = self._as_memory_tensor(memory_bank, image_tensor.device)
        image_tensor = image_tensor.to(memory_tensor.device)

        with torch.no_grad():
            base_score, activation = self._forward_with_activation(
                image_tensor,
                memory_tensor,
                class_name,
            )
            base_score = base_score.mean()

        tokens = self._standardize_layer_output(activation)
        tokens = self._drop_cls_token(tokens)
        grid_size = int(tokens.shape[1] ** 0.5)
        if grid_size * grid_size != tokens.shape[1]:
            raise RuntimeError(
                f"Cannot reshape target activation with {tokens.shape[1]} tokens into a square map"
            )

        acts = tokens[0].T.reshape(tokens.shape[-1], grid_size, grid_size)
        acts = F.relu(acts)
        weighted = torch.zeros((img_size, img_size), device=image_tensor.device)

        base_h, base_w = image_tensor.shape[-2:]
        for channel_idx in range(acts.shape[0]):
            act = acts[channel_idx:channel_idx + 1].unsqueeze(0)
            mask = F.interpolate(
                act,
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            )
            mask = self._normalize_tensor(mask)
            if torch.count_nonzero(mask) == 0:
                continue

            masked_image = image_tensor * mask
            with torch.no_grad():
                channel_score = self._score_tensor(
                    masked_image,
                    memory_tensor,
                    class_name,
                ).mean()
                channel_weight = F.relu(channel_score - base_score)

            upsampled = F.interpolate(
                act,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            weighted = weighted + channel_weight * upsampled

        weighted = F.relu(weighted)
        weighted = self._normalize_tensor(weighted)
        return weighted.detach().cpu().numpy()

    def _forward_score(self, image_tensor):
        if getattr(self, "_score_mode", "global") == "global":
            return self._score_tensor(image_tensor, self._memory_bank, self._class_name)
        return self._patch_anomaly_score_tensor(image_tensor, self._memory_bank)

    def _score_tensor(self, image_tensor, memory_bank, class_name):
        del class_name
        global_feat = self.clip_model.encode_image(image_tensor)
        score = self._memory_distance_score(global_feat, memory_bank)
        # Use a direct memory-bank anomaly objective for Grad-CAM. The
        # PromptQueryAdapter head is randomly initialized in this repo and wraps
        # its output in sigmoid, which can saturate and produce near-zero
        # gradients. Nearest-normal squared distance stays differentiable and
        # matches the memory bank features built from clip_model.encode_image().
        return score.view(image_tensor.shape[0], -1)

    @staticmethod
    def _memory_distance_score(query_feat, memory_bank, top_k=3):
        if memory_bank.dim() == 1:
            memory_bank = memory_bank.unsqueeze(0)

        if query_feat.shape[-1] != memory_bank.shape[-1]:
            raise RuntimeError(
                "Grad-CAM memory bank feature dim does not match image feature dim: "
                f"{memory_bank.shape[-1]} vs {query_feat.shape[-1]}"
            )

        k = min(max(int(top_k), 1), memory_bank.shape[0])
        diff = query_feat.float().unsqueeze(1) - memory_bank.float().unsqueeze(0)
        distances = diff.square().mean(dim=-1)
        nearest = distances.topk(k, dim=1, largest=False).values
        return nearest.mean(dim=1, keepdim=True)

    def _patch_anomaly_score_tensor(self, image_tensor, memory_bank):
        # The memory bank passed here must contain patch-level normal features.
        # We optimize the most anomalous 10% of patches instead of summing all
        # patches, which would encourage a broad image-level attribution map.
        global_feat, patch_tokens = self._encode_image_and_patch_tokens(image_tensor)
        del global_feat
        if patch_tokens is None:
            return self._score_tensor(image_tensor, memory_bank, None)

        patch_scores = compute_patch_scores(
            patch_tokens[0],
            memory_bank.to(patch_tokens.device),
            metric="cosine",
            top_k=3,
        )
        focus_k = max(1, patch_scores.numel() // 10)
        score = patch_scores.topk(focus_k, largest=True).values.mean()
        return score.view(1, 1)

    def _encode_image_and_patch_tokens(self, image_tensor):
        patch_tokens = [None]

        def hook(module, inp, out):
            del module, inp
            patch_tokens[0] = self._standardize_layer_output(out)

        target = getattr(self.clip_model.visual, "ln_post", None)
        if target is None:
            target = self.target_layer

        handle = target.register_forward_hook(hook)
        try:
            global_feat = self.clip_model.encode_image(image_tensor)
        finally:
            handle.remove()

        tokens = patch_tokens[0]
        if tokens is None:
            return global_feat, None

        if tokens.shape[-1] != global_feat.shape[-1]:
            proj = getattr(self.clip_model.visual, "proj", None)
            if proj is not None and tokens.shape[-1] == proj.shape[0]:
                tokens = tokens @ proj.detach().to(tokens.device)

        tokens = self._drop_cls_token(tokens)
        return global_feat, tokens

    def _forward_with_activation(self, image_tensor, memory_bank, class_name):
        self._last_activation = None

        def hook(module, inp, out):
            del module, inp
            self._last_activation = out.detach()

        handle = self.target_layer.register_forward_hook(hook)
        try:
            score = self._score_tensor(image_tensor, memory_bank, class_name)
        finally:
            handle.remove()

        if self._last_activation is None:
            raise RuntimeError("Target layer activation was not captured")
        return score, self._last_activation

    def _find_target_layer(self):
        visual = getattr(self.clip_model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is not None and len(blocks) > 0:
            # CLIP ViT uses CLS-token pooling. At the output of the final block,
            # patch tokens no longer affect the pooled score, so their gradients
            # are zero. Hook the pre-attention norm instead; patch tokens at this
            # point can still influence the CLS token through the last attention.
            target = getattr(blocks[-1], "ln_1", None)
            if target is not None:
                return target
            return blocks[-2] if len(blocks) > 1 else blocks[-1]

        # Fallback for non-standard CLIP visual backbones.
        target = getattr(visual, "ln_post", None)
        if target is not None:
            return target

        raise RuntimeError("Could not find CLIP visual target layer for Grad-CAM")

    def _attribution_to_map(self, attribution, img_size):
        token_scores = self._attribution_to_token_scores(attribution)
        grid_size = int(token_scores.shape[1] ** 0.5)
        if grid_size * grid_size != token_scores.shape[1]:
            raise RuntimeError(
                f"Cannot reshape attribution with {token_scores.shape[1]} tokens into a square map"
            )

        cam = token_scores.view(token_scores.shape[0], 1, grid_size, grid_size)
        cam = F.interpolate(
            cam,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        # FIX #3: _normalize_signed_tensor replaced with _normalize_tensor (ReLU + scale).
        # The old helper fell back to negating the map when positive values were weak,
        # which happened to invert the heatmap (highlighting *normal* regions instead
        # of anomalies). We simply clamp negatives to zero and normalise positives.
        cam = self._normalize_tensor(F.relu(cam[0, 0]))
        return cam.detach().cpu().numpy()

    def _attribution_to_token_scores(self, attribution):
        if isinstance(attribution, (tuple, list)):
            attribution = attribution[0]

        # Do NOT apply F.relu here. Captum has already multiplied gradients and
        # activations. We keep channel attributions until the tensor is in
        # [B, tokens, C] layout, then reduce only the channel dimension.
        attr = attribution

        if attr.dim() == 3:
            standardized = self._standardize_layer_output(attr)

            if standardized.shape[-1] == 1 and self._is_token_count(standardized.shape[1]):
                return self._drop_cls_token_scores(standardized[:, :, 0])

            if standardized.shape[1] == 1 and self._is_token_count(standardized.shape[2]):
                return self._drop_cls_token_scores(standardized[:, 0, :])

            standardized = self._drop_cls_token(standardized)
            return standardized.sum(dim=-1)

        if attr.dim() == 2 and self._is_token_count(attr.shape[1]):
            return self._drop_cls_token_scores(attr)

        raise RuntimeError(f"Unsupported Grad-CAM attribution shape: {tuple(attr.shape)}")

    @staticmethod
    def _is_token_count(n_tokens):
        patch_tokens = n_tokens - 1
        patch_grid = int(patch_tokens ** 0.5)
        full_grid = int(n_tokens ** 0.5)
        return patch_grid * patch_grid == patch_tokens or full_grid * full_grid == n_tokens

    @staticmethod
    def _drop_cls_token_scores(token_scores):
        n_tokens = token_scores.shape[1]
        grid_size = int((n_tokens - 1) ** 0.5)
        if grid_size * grid_size == n_tokens - 1:
            return token_scores[:, 1:]

        grid_size = int(n_tokens ** 0.5)
        if grid_size * grid_size == n_tokens:
            return token_scores

        raise RuntimeError(f"Token count {n_tokens} is not compatible with a square ViT grid")

    @staticmethod
    def _standardize_layer_output(tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]

        if tensor.dim() == 4 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.dim() != 3:
            raise RuntimeError(f"Expected 3D layer output, got shape {tuple(tensor.shape)}")

        if tensor.shape[0] > tensor.shape[1] and tensor.shape[0] > 32:
            tensor = tensor.permute(1, 0, 2)
        return tensor

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
    def _normalize_tensor(tensor):
        tensor = tensor - tensor.min()
        denom = tensor.max()
        if denom <= 1e-8:
            return torch.zeros_like(tensor)
        return tensor / denom

    # FIX #3: _normalize_signed_tensor removed. It silently negated the heatmap
    # when positive attributions were weak (exactly the broken state), producing
    # an inverted map that highlights normal regions. _normalize_tensor (above)
    # is now used directly after an explicit F.relu() call in _attribution_to_map,
    # which is clearer and correct.

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
            raise ValueError("memory_bank is empty; Grad-CAM scoring needs at least one reference")
        return memory.float().to(device)

    @staticmethod
    def _as_patch_memory_tensor(memory_bank, device):
        if hasattr(memory_bank, "get_patch_bank"):
            memory = memory_bank.get_patch_bank()
        else:
            memory = CLIPGradCAM._as_memory_tensor(memory_bank, device)

        if memory.numel() == 0:
            raise ValueError("patch memory_bank is empty; patch Grad-CAM needs normal patch features")
        return memory.float().to(device)

    @contextmanager
    def _temporarily_unfreeze(self, layer):
        params = list(layer.parameters())
        old_flags = [p.requires_grad for p in params]
        for param in params:
            param.requires_grad_(True)
        try:
            yield
        finally:
            for param, old_flag in zip(params, old_flags):
                param.requires_grad_(old_flag)


def run_cable_demo(
    data_dir,
    output_dir,
    device=None,
    n_shots=4,
    img_size=224,
    threshold_percentile=95.0,
):
    clip_model, _, _, device = load_backbone(device)
    adapt_model = AdaptCLIPModel(clip_model, device).to(device)
    adapt_model.eval()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    train_ds = MVTecDataset(data_dir, "cable", split="train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    # FIX #4 (revised): Revert to mode="global". The patch mode in this MemoryBank
    # implementation still collapses patch tokens to a single mean vector per image
    # (patch_mean = patch_feats[:, 1:, :].mean(dim=1)), making it functionally
    # identical to global mode. Using "global" is therefore correct and unambiguous.
    memory = MemoryBank(feat_dim=768, mode="global")
    memory.build(clip_model, train_loader, n_shots, device)
    memory_tensor = torch.from_numpy(memory.index.reconstruct_n(0, memory.index.ntotal)).to(device)

    test_ds = MVTecDataset(
        data_dir,
        "cable",
        split="test",
        transform=transform,
        mask_transform=mask_transform,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    xai = CLIPGradCAM(adapt_model)

    rows = []
    ious = []
    for image_tensor, mask_tensor, label in test_loader:
        if int(label.item()) != 1:
            continue
        if len(rows) >= 5:
            break

        image_tensor = image_tensor.to(device)
        gradcam = xai.generate(
            image_tensor,
            memory,
            "cable",
            img_size=img_size,
            score_mode="patch",
        )
        scorecam = xai.generate_scorecam(image_tensor, memory_tensor, "cable", img_size=img_size)

        original = _denormalize(image_tensor[0]).permute(1, 2, 0).detach().cpu().numpy()
        original_u8 = (np.clip(original, 0, 1) * 255).astype(np.uint8)
        grad_overlay = overlay_heatmap(original_u8, gradcam, alpha=0.5)
        score_overlay = overlay_heatmap(original_u8, scorecam, alpha=0.5)
        mask = mask_tensor.squeeze().cpu().numpy() > 0.5

        threshold = float(np.percentile(gradcam, threshold_percentile))
        pred = gradcam > threshold
        union = np.logical_or(pred, mask).sum()
        intersection = np.logical_and(pred, mask).sum()
        iou = float(intersection / union) if union > 0 else 0.0
        ious.append(iou)

        mask_area = float(mask.mean())
        pred_area = float(pred.mean())
        max_iou_at_area = _max_iou_for_areas(mask_area, pred_area)

        print(
            f"sample {len(rows) + 1}: "
            f"gradcam min={gradcam.min():.4f} max={gradcam.max():.4f} "
            f"mean={gradcam.mean():.4f} threshold={threshold:.4f} "
            f"mask_area={mask_area:.4f} pred_area={pred_area:.4f} "
            f"iou={iou:.4f} max_iou@area={max_iou_at_area:.4f}"
        )

        rows.append((original_u8, grad_overlay, score_overlay, mask.astype(np.float32)))

    if not rows:
        raise RuntimeError("No anomalous cable images found for Grad-CAM demo")

    _save_comparison_panel(rows, ious, output_dir)
    print(
        "Average Grad-CAM IoU @ "
        f"{threshold_percentile:g}th-percentile threshold: {np.mean(ious):.4f}"
    )


def _max_iou_for_areas(mask_area, pred_area):
    if mask_area <= 0.0 or pred_area <= 0.0:
        return 0.0
    return min(mask_area, pred_area) / max(mask_area, pred_area)


def _save_comparison_panel(rows, ious, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["Original", "Grad-CAM Overlay", "Score-CAM Overlay", "GT Mask"]
    for row_idx, (original, grad_overlay, score_overlay, mask) in enumerate(rows):
        for col_idx, (title, image) in enumerate(zip(titles, [original, grad_overlay, score_overlay, mask])):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="gray" if col_idx == 3 else None)
            ax.set_title(f"{title}" if col_idx != 1 else f"{title} | IoU={ious[row_idx]:.3f}")
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "gradcam_cable_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _denormalize(tensor):
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM / Score-CAM demo for CLIP AdaptCLIP")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/heatmaps")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_shots", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    args = parser.parse_args()

    run_cable_demo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_shots=args.n_shots,
        img_size=args.img_size,
        threshold_percentile=args.threshold_percentile,
    )


if __name__ == "__main__":
    main()
