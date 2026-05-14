import argparse
import json
import os
import glob
import csv
from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def collect_stats(data_dir):
    stats = {}
    total = {"train": 0, "test_normal": 0, "test_anom": 0, "categories": 0}

    for cat in CATEGORIES:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        total["categories"] += 1

        train_dir = os.path.join(cat_dir, "train", "good")
        train_paths = sorted(glob.glob(os.path.join(train_dir, "*.png")))
        train_count = len(train_paths)

        test_dir = os.path.join(cat_dir, "test")
        test_dirs = sorted(
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        )
        defect_types = [d for d in test_dirs if d != "good"]
        has_normal = "good" in test_dirs

        test_good_count = 0
        if has_normal:
            test_good_count = len(
                glob.glob(os.path.join(test_dir, "good", "*.png"))
            )

        defect_info = {}
        for dt in defect_types:
            paths = sorted(glob.glob(os.path.join(test_dir, dt, "*.png")))
            defect_info[dt] = len(paths)
        test_anom_count = sum(defect_info.values())

        sample_img = Image.open(train_paths[0]).convert("RGB")
        img_width, img_height = sample_img.size

        info = {
            "category": cat,
            "train": train_count,
            "test_normal": test_good_count,
            "test_anomalous": test_anom_count,
            "test_total": test_good_count + test_anom_count,
            "defect_types": defect_info,
            "defect_type_names": defect_types,
            "image_width": img_width,
            "image_height": img_height,
        }
        stats[cat] = info
        total["train"] += train_count
        total["test_normal"] += test_good_count
        total["test_anom"] += test_anom_count

    return stats, total


def save_stats_csv(stats, total, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dataset_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Train", "Test_Normal", "Test_Anomalous", "Test_Total",
                     "Width", "Height", "Defect_Types"])
        for info in stats.values():
            w.writerow([
                info["category"], info["train"], info["test_normal"],
                info["test_anomalous"], info["test_total"],
                info["image_width"], info["image_height"],
                ", ".join(info["defect_type_names"]),
            ])
        w.writerow([])
        w.writerow(["TOTAL", total["train"], total["test_normal"],
                     total["test_anom"], total["test_normal"] + total["test_anom"],
                     "", "", f"{total['categories']} categories"])
    print(f"  CSV -> {path}")


def save_stats_json(stats, total, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dataset_summary.json")
    payload = {"categories": stats, "total": total}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON -> {path}")


def plot_anomaly_bar(stats, output_dir):
    cats = sorted(stats.keys())
    anom = [stats[c]["test_anomalous"] for c in cats]
    normal = [stats[c]["test_normal"] for c in cats]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, normal, w, label="Normal", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, anom, w, label="Anomalous", color="crimson", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Images")
    ax.set_title("MVTec AD: Test Images per Category")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, a, n in zip(x, anom, normal):
        ax.text(xi + w / 2, a + 1, str(a), ha="center", va="bottom", fontsize=8)
        ax.text(xi - w / 2, n + 1, str(n), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "test_images_per_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {path}")


def plot_defect_distribution(stats, output_dir):
    cat_defect_counts = {}
    for cat, info in stats.items():
        total_d = sum(info["defect_types"].values())
        cat_defect_counts[cat] = total_d

    cats = sorted(cat_defect_counts.keys())
    counts = [cat_defect_counts[c] for c in cats]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(cats)), counts, color="darkorange", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Anomalous Images")
    ax.set_title("MVTec AD: Total Anomalous Images per Category")
    ax.grid(axis="y", alpha=0.3)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(c), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "anomalous_per_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {path}")


def plot_defect_type_breakdown(stats, output_dir):
    for cat in sorted(stats.keys()):
        info = stats[cat]
        defect_counts = info["defect_types"]
        if not defect_counts:
            continue
        names = list(defect_counts.keys())
        counts = list(defect_counts.values())
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(range(len(names)), counts, color="teal", edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Images")
        ax.set_title(f"{cat}: Defect Type Breakdown")
        ax.invert_yaxis()
        for bar, c in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(c), va="center", fontsize=9)
        plt.tight_layout()
        p = os.path.join(output_dir, f"defect_breakdown_{cat}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Defect breakdown plots -> {output_dir}/")


def plot_image_size_scatter(stats, output_dir):
    sizes = {}
    for info in stats.values():
        key = (info["image_width"], info["image_height"])
        sizes.setdefault(key, []).append(info["category"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for (w, h), cats_list in sizes.items():
        ax.scatter(w, h, s=100, c="purple", edgecolors="black", zorder=3)
        offset = 5
        for c in cats_list:
            ax.annotate(c, (w + offset, h + offset), fontsize=7)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("MVTec AD: Image Resolutions per Category")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "image_resolutions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {path}")


def analyze_defect_areas(data_dir, output_dir):
    rows = []
    for cat in CATEGORIES:
        gt_dir = os.path.join(data_dir, cat, "ground_truth")
        if not os.path.isdir(gt_dir):
            continue
        for defect_type in sorted(os.listdir(gt_dir)):
            dp = os.path.join(gt_dir, defect_type)
            if not os.path.isdir(dp):
                continue
            for mask_path in sorted(glob.glob(os.path.join(dp, "*.png"))):
                mask = Image.open(mask_path).convert("L")
                mask_arr = np.array(mask)
                total_px = mask_arr.size
                defect_px = int(np.sum(mask_arr > 0))
                pct = (defect_px / total_px) * 100 if total_px > 0 else 0.0
                fname = os.path.basename(mask_path)
                rows.append({
                    "category": cat,
                    "defect_type": defect_type,
                    "mask_file": fname,
                    "total_pixels": total_px,
                    "defect_pixels": defect_px,
                    "defect_pct": round(pct, 4),
                })

    if not rows:
        return

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "defect_areas.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV -> {path}")

    pcts = [r["defect_pct"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(pcts, bins=50, color="coral", edgecolor="black")
    axes[0].set_xlabel("Defect Area (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Defect Area Percentage")

    axes[1].boxplot(pcts, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[1].set_ylabel("Defect Area (%)")
    axes[1].set_title("Defect Area Boxplot")
    axes[1].set_xticks([])
    plt.tight_layout()
    path2 = os.path.join(output_dir, "defect_area_distribution.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {path2}")


def save_sample_grid(data_dir, output_dir, max_categories=15):
    os.makedirs(output_dir, exist_ok=True)
    for cat in CATEGORIES[:max_categories]:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        train_good = sorted(glob.glob(os.path.join(cat_dir, "train", "good", "*.png")))
        test_dir = os.path.join(cat_dir, "test")
        defect_types = sorted(
            d for d in os.listdir(test_dir)
            if d != "good" and os.path.isdir(os.path.join(test_dir, d))
        )

        n_samples = min(3, len(train_good))
        n_defects = min(3, len(defect_types))
        total_cols = n_samples + n_defects
        if total_cols == 0:
            continue

        fig, axes = plt.subplots(1, total_cols, figsize=(3 * total_cols, 3.5))
        if total_cols == 1:
            axes = [axes]

        for i in range(n_samples):
            img = Image.open(train_good[i]).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(f"Normal #{i+1}", fontsize=9)
            axes[i].axis("off")

        for j, dt in enumerate(defect_types[:n_defects]):
            paths = sorted(glob.glob(os.path.join(test_dir, dt, "*.png")))
            if not paths:
                continue
            img = Image.open(paths[0]).convert("RGB")
            axes[n_samples + j].imshow(img)
            axes[n_samples + j].set_title(f"{dt}", fontsize=8)
            axes[n_samples + j].axis("off")

        plt.suptitle(f"{cat}", fontsize=12, y=1.02)
        plt.tight_layout()
        p = os.path.join(output_dir, f"samples_{cat}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Sample grids -> {output_dir}/")


def run_eda(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Collecting statistics...")
    stats, total = collect_stats(data_dir)
    print(f"  Found {total['categories']} categories, "
          f"{total['train']} train, "
          f"{total['test_normal'] + total['test_anom']} test "
          f"({total['test_anom']} anomalous)")

    print("Saving summaries...")
    save_stats_csv(stats, total, output_dir)
    save_stats_json(stats, total, output_dir)

    print("Generating plots...")
    plot_anomaly_bar(stats, output_dir)
    plot_defect_distribution(stats, output_dir)
    plot_defect_type_breakdown(stats, output_dir)
    plot_image_size_scatter(stats, output_dir)

    print("Analyzing defect areas...")
    analyze_defect_areas(data_dir, os.path.join(output_dir, "defect_areas"))

    print("Saving sample grids...")
    save_sample_grid(data_dir, os.path.join(output_dir, "sample_grids"))

    print("EDA complete.")


def main():
    parser = argparse.ArgumentParser(
        description="MVTec AD Exploratory Data Analysis"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to mvtec_anomaly_detection/ directory",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/results/eda",
        help="Directory to save EDA outputs (default: ./outputs/results/eda)",
    )
    args = parser.parse_args()

    run_eda(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
