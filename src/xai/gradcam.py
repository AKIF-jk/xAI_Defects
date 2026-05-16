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
        # FIX #2: Default score_mode changed from "patch" to "global".
        # Using patch-sum as a scalar gives uniform gradients everywhere (no spatial signal).
        # Global scoring produces a single meaningful logit that Captum can differentiate
        # spatially through the visual transformer stack.
        self.model.eval()
        self._memory_bank = self._as_memory_tensor(memory_bank, image_tensor.device)
        self._class_name = class_name
        self._score_mode = score_mode

        image_tensor = image_tensor.to(self._memory_bank.device)
        image_tensor = image_tensor.requires_grad_(True)

        with self._temporarily_unfreeze(self.target_layer):
            attribution = self.gradcam.attribute(image_tensor)

        gradcam_map = self._attribution_to_map(attribution, img_size)
        return gradcam_map

    def generate_scorecam(self, image_tensor, memory_bank, class_name, img_size=256):
        # Score-CAM is gradient-free but very slow on ViT-L/14 because it needs
        # one forward pass per activation channel, often ~50x slower than Grad-CAM.
        self.model.eval()
        memory_tensor = self._as_memory_tensor(memory_bank, image_tensor.device)
        image_tensor = image_tensor.to(memory_tensor.device)

        with torch.no_grad():
            score, activation = self._forward_with_activation(
                image_tensor,
                memory_tensor,
                class_name,
            )
            del score

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

            upsampled = F.interpolate(
                act,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            weighted = weighted + F.relu(channel_score) * upsampled

        weighted = F.relu(weighted)
        weighted = self._normalize_tensor(weighted)
        return weighted.detach().cpu().numpy()

    def _forward_score(self, image_tensor):
        # FIX #2 (continued): "patch" mode is removed from this hot path.
        # _patch_anomaly_score_tensor summed all patch scores into a single scalar,
        # giving Captum no spatial gradient signal. Always use global scoring here.
        if getattr(self, "_score_mode", "global") == "global":
            return self._score_tensor(image_tensor, self._memory_bank, self._class_name)
        return self._patch_anomaly_score_tensor(image_tensor, self._memory_bank)

    def _score_tensor(self, image_tensor, memory_bank, class_name):
        del class_name
        global_feat = self.clip_model.encode_image(image_tensor)
        adapted = self.model.visual_adapter(global_feat)
        score = self.model.prompt_query_adapter(adapted, memory_bank)
        return score.view(image_tensor.shape[0], -1)

    def _patch_anomaly_score_tensor(self, image_tensor, memory_bank):
        # NOTE: Only used when score_mode="patch" is explicitly requested.
        # The memory_bank passed here should be a patch-level bank (mode="patch"),
        # not a global one — mixing global memory with patch tokens yields
        # meaningless cosine similarities (FIX #4).
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
        score = patch_scores.sum()
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
        # FIX #1: Hook the full last residual block instead of its ln_2 sub-layer.
        #
        # Original code: target = getattr(last_block, "ln_2", None)
        # ln_2 is a LayerNorm applied after the MLP; it normalises activations to
        # roughly unit variance, so gradients flowing back through it collapse to
        # near-zero and carry no spatial discrimination. Attaching Grad-CAM here
        # gives a flat, near-zero attribution map (mean ~0.02, IoU ~0).
        #
        # The correct hook point for ViT Grad-CAM is the residual block itself
        # (resblocks[-1]). Its output is the sum of the attention and MLP branches
        # *before* any normalisation, so gradients are strong and spatially varied.
        visual = getattr(self.clip_model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is not None and len(blocks) > 0:
            # Return the full last residual block — not a sub-layer of it.
            return blocks[-1]

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

        attr = F.relu(attribution)

        if attr.dim() == 3:
            # Captum LayerGradCam commonly returns [B, 1, tokens] for ViT
            # layers because it sums over the hidden dimension. Treat that as
            # an already channel-reduced token score map.
            if attr.shape[1] == 1 and self._is_token_count(attr.shape[2]):
                return self._drop_cls_token_scores(attr[:, 0, :])

            if attr.shape[2] == 1 and self._is_token_count(attr.shape[1]):
                return self._drop_cls_token_scores(attr[:, :, 0])

            standardized = self._standardize_layer_output(attr)
            standardized = self._drop_cls_token(standardized)
            return standardized.mean(dim=-1)

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


def run_cable_demo(data_dir, output_dir, device=None, n_shots=4, img_size=224):
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

    # FIX #4: MemoryBank mode changed from "global" to "patch".
    # The original code built a global-average memory bank (one vector per image)
    # and then compared individual patch tokens against it inside
    # _patch_anomaly_score_tensor. Global vectors and patch tokens live in
    # different statistical spaces, so cosine similarity is meaningless.
    # A patch-level bank stores per-patch features, making patch-vs-memory
    # comparisons geometrically valid.
    # NOTE: feat_dim may need adjustment to match your patch token dimensionality
    # (commonly 1024 for ViT-L/14 patch tokens before projection, or 768 for ViT-B).
    memory = MemoryBank(feat_dim=768, mode="patch")
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
        # score_mode="global" is now the default in generate(), so this is explicit
        # for clarity but not strictly required.
        gradcam = xai.generate(image_tensor, memory_tensor, "cable", img_size=img_size, score_mode="global")
        scorecam = xai.generate_scorecam(image_tensor, memory_tensor, "cable", img_size=img_size)

        original = _denormalize(image_tensor[0]).permute(1, 2, 0).detach().cpu().numpy()
        original_u8 = (np.clip(original, 0, 1) * 255).astype(np.uint8)
        grad_overlay = overlay_heatmap(original_u8, gradcam, alpha=0.5)
        score_overlay = overlay_heatmap(original_u8, scorecam, alpha=0.5)
        mask = mask_tensor.squeeze().cpu().numpy() > 0.5

        # FIX #5: Threshold replaced with a simple top-20% percentile cut.
        # The original "mean + 2*std" formula on a near-zero map (mean ~0.02)
        # produced a threshold so high that almost no pixel cleared it, making
        # pred all-False and IoU = 0 by construction.
        # np.percentile(gradcam, 80) selects the top 20% of pixels, which is a
        # robust, distribution-agnostic way to identify high-activation regions.
        threshold = float(np.percentile(gradcam, 80))
        pred = gradcam >= threshold
        union = np.logical_or(pred, mask).sum()
        intersection = np.logical_and(pred, mask).sum()
        iou = float(intersection / union) if union > 0 else 0.0
        ious.append(iou)

        print(
            f"sample {len(rows) + 1}: "
            f"gradcam min={gradcam.min():.4f} max={gradcam.max():.4f} "
            f"mean={gradcam.mean():.4f} threshold={threshold:.4f} iou={iou:.4f}"
        )

        rows.append((original_u8, grad_overlay, score_overlay, mask.astype(np.float32)))

    if not rows:
        raise RuntimeError("No anomalous cable images found for Grad-CAM demo")

    _save_comparison_panel(rows, ious, output_dir)
    print(f"Average Grad-CAM IoU @ 80th-percentile threshold: {np.mean(ious):.4f}")


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
    args = parser.parse_args()

    run_cable_demo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_shots=args.n_shots,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()