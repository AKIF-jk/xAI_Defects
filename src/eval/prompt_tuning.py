import argparse
import json
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

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

NORMAL_TEMPLATES = [
    "a photo of a {c}",
    "a flawless {c}",
    "a perfect {c} with no defects",
    "a {c} in good condition",
    "an undamaged {c}",
    "a {c} without any damage",
    "a normal {c}",
    "a well-manufactured {c}",
]

ANOMALOUS_TEMPLATES = [
    "a photo of a defective {c}",
    "a damaged {c}",
    "a {c} with visible defects",
    "a broken {c}",
    "a {c} with surface anomaly",
    "a contaminated {c}",
    "a flawed {c}",
    "a {c} with manufacturing defect",
]


def encode_texts(clip_model, tokenizer, templates, class_name, device):
    texts = [t.format(c=class_name) for t in templates]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
    return F.normalize(feats, dim=-1)


def compute_image_centroids(data_dir, category, clip_model, device):
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MVTecDataset(data_dir, category, split="train", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

    normal_feats = []
    for images, _ in train_loader:
        images = images.to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(images)
        normal_feats.append(feats.cpu())
    normal_mean = torch.cat(normal_feats).mean(dim=0)

    test_ds = MVTecDataset(data_dir, category, split="test", transform=tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    anom_feats = []
    for batch in test_loader:
        images, _, labels = batch
        images = images.to(device)
        labels = labels.cpu()
        with torch.no_grad():
            feats = clip_model.encode_image(images).cpu()
        anom_mask = labels == 1
        if anom_mask.any():
            anom_feats.append(feats[anom_mask])

    if anom_feats:
        anom_mean = torch.cat(anom_feats).mean(dim=0)
    else:
        anom_mean = normal_mean.clone()

    return F.normalize(normal_mean.unsqueeze(0), dim=-1), \
           F.normalize(anom_mean.unsqueeze(0), dim=-1)


def run_prompt_search(data_dir, output_dir, device):
    print("Loading CLIP backbone...")
    clip_model, tokenizer, _, device = load_backbone(device)
    clip_model.eval()

    results = {}
    alignment_matrix = np.zeros((len(NORMAL_TEMPLATES), len(CATEGORIES)))

    for ci, cat in enumerate(CATEGORIES):
        print(f"\n  {cat} ... ", end="", flush=True)

        norm_feat, anom_feat = compute_image_centroids(
            data_dir, cat, clip_model, device
        )
        img_dir = F.normalize(anom_feat - norm_feat, dim=-1)

        normal_texts = encode_texts(
            clip_model, tokenizer, NORMAL_TEMPLATES, cat, device
        )
        anom_texts = encode_texts(
            clip_model, tokenizer, ANOMALOUS_TEMPLATES, cat, device
        )
        text_dir = F.normalize(anom_texts - normal_texts, dim=-1)

        alignments = (text_dir * img_dir).sum(dim=1).cpu().numpy()
        alignment_matrix[:, ci] = alignments

        best_idx = int(alignments.argmax())
        results[cat] = {
            "best_pair": best_idx,
            "best_alignment": float(alignments[best_idx]),
            "normal_template": NORMAL_TEMPLATES[best_idx],
            "anomalous_template": ANOMALOUS_TEMPLATES[best_idx],
            "all_alignments": {str(i): float(a)
                               for i, a in enumerate(alignments)},
        }
        print(f"best={best_idx}  align={alignments[best_idx]:.4f}")

    pair_names = [f"T{i}" for i in range(len(NORMAL_TEMPLATES))]
    print(f"\n{'=' * 70}")
    print(f"{'Template':<10}", end="")
    for cat in CATEGORIES:
        print(f"{cat[:6]:>7}", end="")
    print()
    for pi in range(len(NORMAL_TEMPLATES)):
        print(f"{pair_names[pi]:<10}", end="")
        for ci in range(len(CATEGORIES)):
            print(f"{alignment_matrix[pi, ci]:>7.3f}", end="")
        print()

    best_by_template = {}
    for pi in range(len(NORMAL_TEMPLATES)):
        matches = [(CATEGORIES[ci], alignment_matrix[pi, ci])
                    for ci in range(len(CATEGORIES))]
        best_by_template[pair_names[pi]] = {
            "normal": NORMAL_TEMPLATES[pi],
            "anomalous": ANOMALOUS_TEMPLATES[pi],
            "per_category": {c: float(alignment_matrix[pi, ci])
                             for ci, c in enumerate(CATEGORIES)},
            "mean_alignment": float(alignment_matrix[pi].mean()),
        }

    os.makedirs(output_dir, exist_ok=True)

    out = {
        "per_category": results,
        "per_template": best_by_template,
    }
    path = os.path.join(output_dir, "best_prompts.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cat_labels = [c[:8] for c in CATEGORIES]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        alignment_matrix,
        xticklabels=cat_labels,
        yticklabels=pair_names,
        annot=True, fmt=".3f", cmap="RdYlGn",
        center=0.0, vmin=-0.5, vmax=0.5,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("Template Pair")
    ax.set_title("Prompt Alignment: Text Direction vs Image Direction")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "prompt_alignment_heatmap.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal text prompts for AdaptCLIP"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to mvtec_anomaly_detection/",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/results",
        help="Output directory (default: ./outputs/results)",
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    args = parser.parse_args()
    run_prompt_search(args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
