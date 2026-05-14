import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.mvtec_dataset import MVTecDataset
from model.backbone import load_backbone
from model.memory_bank import MemoryBank


def test_memory_bank(data_dir, output_dir, device):
    print("Loading CLIP backbone...")
    clip_model, _, _, device = load_backbone(device)
    clip_model.eval()

    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    mtf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_ds = MVTecDataset(data_dir, "bottle", split="train", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    test_ds = MVTecDataset(data_dir, "bottle", split="test",
                            transform=tf, mask_transform=mtf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    normal_images = []
    anom_images = []
    for batch in test_loader:
        img, _, label = batch
        if label.item() == 0 and len(normal_images) < 5:
            normal_images.append(img)
        elif label.item() == 1 and len(anom_images) < 5:
            anom_images.append(img)
        if len(normal_images) == 5 and len(anom_images) == 5:
            break

    modes = ["global", "patch", "hybrid"]
    n_shot_values = [1, 2, 4]
    all_results = {}

    for mode in modes:
        for n_shots in n_shot_values:
            print(f"\n{'=' * 60}")
            print(f"Mode={mode:>8}  n_shots={n_shots}")
            print(f"{'=' * 60}")

            bank = MemoryBank(feat_dim=768, mode=mode)
            n_stored = bank.build(clip_model, train_loader, n_shots, device)
            print(f"  Memory bank: {n_stored} features stored")

            normal_dists = []
            for img in normal_images:
                img = img.to(device)
                qf = bank.encode(clip_model, img, device)
                d, _ = bank.query(qf, k=1)
                normal_dists.append(d[0, 0])

            anom_dists = []
            for img in anom_images:
                img = img.to(device)
                qf = bank.encode(clip_model, img, device)
                d, _ = bank.query(qf, k=1)
                anom_dists.append(d[0, 0])

            normal_mean = np.mean(normal_dists)
            anom_mean = np.mean(anom_dists)

            print(f"  Normal NN dist:    mean={normal_mean:.4f}  {[f'{x:.3f}' for x in normal_dists]}")
            print(f"  Anomalous NN dist: mean={anom_mean:.4f}  {[f'{x:.3f}' for x in anom_dists]}")
            print(f"  Separation:        {anom_mean - normal_mean:.4f}")

            all_results[(mode, n_shots)] = {
                "normal_dists": normal_dists,
                "anom_dists": anom_dists,
                "normal_mean": normal_mean,
                "anom_mean": anom_mean,
            }

    print(f"\n\n{'=' * 70}")
    print(f"{'Mode':>8} {'Shots':>5} {'NormDist':>10} {'AnomDist':>10} {'Separation':>12}")
    print(f"{'=' * 70}")
    for mode in modes:
        for n_shots in n_shot_values:
            r = all_results[(mode, n_shots)]
            s = r["anom_mean"] - r["normal_mean"]
            print(f"{mode:>8} {n_shots:>5} {r['normal_mean']:>10.4f} {r['anom_mean']:>10.4f} {s:>12.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(modes), len(n_shot_values),
                                 figsize=(15, 10), sharex=True, sharey="row")
        for mi, mode in enumerate(modes):
            for ni, n_shots in enumerate(n_shot_values):
                ax = axes[mi, ni]
                r = all_results[(mode, n_shots)]
                ax.hist(r["normal_dists"], bins=8, alpha=0.6, color="steelblue",
                        label="Normal", edgecolor="black")
                ax.hist(r["anom_dists"], bins=8, alpha=0.6, color="crimson",
                        label="Anomalous", edgecolor="black")
                ax.set_title(f"{mode}, n={n_shots}")
                ax.legend(fontsize=7)
                if mi == len(modes) - 1:
                    ax.set_xlabel("NN Distance")
                if ni == 0:
                    ax.set_ylabel("Count")
                ax.grid(alpha=0.3)

        plt.suptitle("Memory Bank: NN Distance Distributions (bottle)", fontsize=14)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "memory_bank_distances.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Test MemoryBank on bottle category"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/results")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    test_memory_bank(args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
