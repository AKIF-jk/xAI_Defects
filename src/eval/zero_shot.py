import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mvtec_dataset import MVTecDataset
from model.backbone import load_backbone
from model.adaptclip import AdaptCLIPModel

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

IMG_SIZE = 224
BATCH_SIZE = 32
N_SHOT = 8


def build_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_memory_bank(clip_model, data_dir, category, device):
    tf = build_transform()
    ds = MVTecDataset(data_dir, category, split="train", transform=tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    feats = []
    for i, (img, _) in enumerate(loader):
        if i >= N_SHOT:
            break
        img = img.to(device)
        with torch.no_grad():
            f = clip_model.encode_image(img)
        feats.append(f)
    return torch.cat(feats, dim=0) if feats else torch.zeros(0, 768, device=device)


def build_test_loader(data_dir, category):
    tf = build_transform()
    mtf = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    ds = MVTecDataset(data_dir, category, split="test", transform=tf, mask_transform=mtf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False), len(ds)


def evaluate_category(clip_model, adapt_model, data_dir, category, device):
    memory_bank = build_memory_bank(clip_model, data_dir, category, device)

    test_loader, num_test = build_test_loader(data_dir, category)

    all_labels = []
    all_scores = []

    for batch_idx, batch in enumerate(test_loader):
        if len(batch) == 3:
            images, _, labels = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            scores, _ = adapt_model(images, memory_bank, category)
        scores = scores.detach().cpu()
        labels = labels.detach().cpu()

        if scores.dim() == 2 and scores.shape[1] == 1:
            scores = scores.squeeze(1)
        elif scores.dim() == 2 and scores.shape[1] > 1:
            scores = scores.mean(dim=1)

        all_scores.append(scores)
        all_labels.append(labels)

    all_scores = torch.cat([s.view(-1) for s in all_scores]).numpy()
    all_labels = torch.cat([l.view(-1) for l in all_labels]).numpy()

    print(f"    shapes: labels={all_labels.shape} scores={all_scores.shape}", end="")

    if len(np.unique(all_labels)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(all_labels, all_scores)

    return auroc, num_test


def run_evaluation(data_dir, output_dir, device):
    print("Loading CLIP backbone...")
    clip_model, tokenizer, preprocess, device = load_backbone(device)

    print("Building AdaptCLIP model (random adapters)...")
    adapt_model = AdaptCLIPModel(clip_model, device).to(device)
    adapt_model.eval()

    results = []
    for cat in CATEGORIES:
        print(f"  {cat} ... ", end="", flush=True)
        auroc, n = evaluate_category(clip_model, adapt_model, data_dir, cat, device)
        results.append((cat, n, auroc))
        print(f"n={n}  AUROC={auroc:.4f}")

    results.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 50)
    print(f"{'Category':<15} {'Test':>5} {'AUROC':>8}")
    print("=" * 50)
    for cat, n, auroc in results:
        marker = " *" if auroc >= 0.85 else ""
        print(f"{cat:<15} {n:>5} {auroc:>8.4f}{marker}")
    print("=" * 50)
    mean_auroc = np.nanmean([r[2] for r in results])
    print(f"{'MEAN':<15} {'':>5} {mean_auroc:>8.4f}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "zero_shot_auroc.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "NumTest", "AUROC"])
        for cat, n, auroc in results:
            w.writerow([cat, n, round(auroc, 4)])
        w.writerow(["MEAN", "", round(mean_auroc, 4)])
    print(f"\nSaved: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = [r[0] for r in results]
    vals = [r[2] for r in results]
    colors = ["green" if v >= 0.85 else "crimson" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(cats)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel("Image-Level AUROC")
    ax.set_title("Zero-Shot AdaptCLIP on MVTec AD")
    ax.set_xlim(0, 1.0)
    ax.axvline(0.85, color="gray", linestyle="--", linewidth=1.5, label="Target (0.85)")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "zero_shot_auroc.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    return results, mean_auroc


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot AdaptCLIP evaluation on MVTec AD"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to mvtec_anomaly_detection/",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/results",
        help="Directory for CSV and plots (default: ./outputs/results)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. 'cuda:0' or 'cpu')",
    )
    args = parser.parse_args()

    run_evaluation(args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
