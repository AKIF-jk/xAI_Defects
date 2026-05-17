import argparse
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from xai.gradcam import CLIPGradCAM
from xai.shap_explainer import PatchSHAPExplainer


def patch_position_to_text(patch_idx, grid_size, img_size=None):
    G = int(grid_size)
    row = patch_idx // G
    col = patch_idx % G
    if G <= 1:
        return "center"
    vert = "top" if row == 0 else ("bottom" if row == G - 1 else "")
    horz = "left" if col == 0 else ("right" if col == G - 1 else "")
    if vert and horz:
        return f"{vert}-{horz}"
    if vert:
        return vert
    if horz:
        return horz
    return "center"


DEFECT_VOCAB = {
    "bottle": ["broken large", "broken small", "contamination"],
    "cable": ["bent wire", "cable swap", "cut inner", "cut outer",
              "missing cable", "missing wire", "poke insulation"],
    "capsule": ["crack", "faulty imprint", "poke", "scratch", "squeeze"],
    "carpet": ["color stain", "cut", "hole", "metal contamination", "thread"],
    "grid": ["bent", "broken", "glue contamination", "metal contamination", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print defect"],
    "leather": ["color stain", "cut", "fold", "glue contamination", "poke hole"],
    "metal_nut": ["bent", "color defect", "flip", "scratch"],
    "pill": ["color defect", "contamination", "crack", "faulty imprint", "scratch"],
    "screw": ["manipulated front", "scratch head", "scratch neck",
              "thread side defect", "thread top defect"],
    "tile": ["crack", "glue strip", "gray stroke", "oil", "rough surface"],
    "toothbrush": ["defective bristles", "deformed body"],
    "transistor": ["bent lead", "cut lead", "damaged case", "misplaced"],
    "wood": ["color defect", "hole", "liquid stain", "scratch"],
    "zipper": ["broken teeth", "fabric border defect", "fabric interior defect",
               "rough surface", "split teeth", "squeezed teeth"],
}


def build_explanation_prompt(category, anomaly_score, top_shap_patches, gradcam_region):
    vocab = ", ".join(DEFECT_VOCAB.get(category, ["surface defect"]))
    severity = "high" if anomaly_score > 0.75 else ("medium" if anomaly_score > 0.5 else "low")
    if not top_shap_patches or top_shap_patches[0]["position"] == "none":
        shap_lines = "  - No strong patch-level evidence detected"
    else:
        shap_lines = "\n".join(
            f"  - Patch at {p['position']} (contribution: {p['value']:.3f})"
            for p in top_shap_patches
        )
    prompt = (
        f"You are inspecting a {category} on a manufacturing line.\n"
        f"Known defect types for this product: {vocab}\n"
        f"Anomaly severity: {severity} (score={anomaly_score:.2f}, 0=normal, 1=defective)\n"
        f"Primary defect region (GradCAM): {gradcam_region} of the image\n"
        f"Supporting evidence (SHAP patches):\n{shap_lines}\n\n"
        "In one sentence (max 35 words): name the defect type, its location, "
        "and a likely cause. Be specific and actionable."
    )
    return prompt


def get_explanation(
    prompt_text,
    pipe=None,
    model="mistralai/Mistral-7B-Instruct-v0.3",
    device="cpu",
    max_new_tokens=100,
    temperature=0.3,
):
    if pipe is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

        tokenizer = AutoTokenizer.from_pretrained(model)
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            device_map="auto" if device != "cpu" else None,
        )
        pipe = hf_pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            device=0 if device != "cpu" else -1,
        )

    system = (
        "You are a quality control expert. Given anomaly detection results, "
        "write exactly ONE sentence (max 35 words) explaining what defect was found, "
        "where it is, and a likely cause. Be specific and actionable."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False,
    )
    if isinstance(outputs[0]["generated_text"], list):
        return outputs[0]["generated_text"][-1]["content"].strip(), pipe
    return outputs[0]["generated_text"].strip(), pipe


def _gradcam_hotspot_location(gradcam_map, grid_size):
    G = int(grid_size)
    h, w = gradcam_map.shape
    y_edges = np.linspace(0, h, G + 1, dtype=int)
    x_edges = np.linspace(0, w, G + 1, dtype=int)
    best_val = -1.0
    best_idx = 0
    for gy in range(G):
        for gx in range(G):
            patch_mean = float(
                gradcam_map[
                    y_edges[gy] : y_edges[gy + 1],
                    x_edges[gx] : x_edges[gx + 1],
                ].mean()
            )
            if patch_mean > best_val:
                best_val = patch_mean
                best_idx = gy * G + gx
    return patch_position_to_text(best_idx, G)


def _get_top_shap_patches(shap_map, grid_size, top_k=5):
    G = int(grid_size)
    h, w = shap_map.shape
    y_edges = np.linspace(0, h, G + 1, dtype=int)
    x_edges = np.linspace(0, w, G + 1, dtype=int)
    patches = []
    for gy in range(G):
        for gx in range(G):
            val = float(
                shap_map[
                    y_edges[gy] : y_edges[gy + 1],
                    x_edges[gx] : x_edges[gx + 1],
                ].mean()
            )
            patches.append({"index": gy * G + gx, "value": val})
    patches.sort(key=lambda p: p["value"], reverse=True)
    positive = [p for p in patches if p["value"] > 0.0]
    if not positive:
        return [{"position": "none", "value": 0.0}]
    return [
        {"position": patch_position_to_text(p["index"], G), "value": p["value"]}
        for p in positive[:top_k]
    ]


def _denormalize_image(tensor):
    tensor = tensor.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = torch.clamp(tensor * std + mean, 0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _as_memory_tensor(memory_bank, device):
    if isinstance(memory_bank, torch.Tensor):
        memory = memory_bank
    elif hasattr(memory_bank, "index"):
        vectors = memory_bank.index.reconstruct_n(0, memory_bank.index.ntotal)
        memory = torch.from_numpy(vectors)
    else:
        memory = torch.as_tensor(memory_bank)
    if memory.numel() == 0:
        raise ValueError("memory_bank is empty; need at least one reference")
    return memory.float().to(device)


def explain_defect(
    image,
    model,
    memory_bank,
    class_name,
    pipe=None,
    gradcam_gen=None,
    shap_gen=None,
    grid_size=7,
    n_evals=200,
):
    start = time.perf_counter()
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    device = image.device
    memory_tensor = _as_memory_tensor(memory_bank, device)

    # 1. Score via AdaptCLIP forward
    logger.info("Computing anomaly score...")
    with torch.no_grad():
        score_tensor, _ = model(image, memory_tensor, class_name)
    anomaly_score = float(score_tensor[0].item())
    logger.info("Anomaly score: %.4f", anomaly_score)

    # 2. GradCAM heatmap
    logger.info("Generating GradCAM heatmap...")
    t0 = time.perf_counter()
    if gradcam_gen is None:
        gradcam_gen = CLIPGradCAM(model)
    gradcam_map = gradcam_gen.generate(
        image, memory_bank, class_name, img_size=image.shape[-1]
    )
    logger.info("GradCAM done in %.2fs", time.perf_counter() - t0)

    # 3. SHAP attribution map
    logger.info("Running SHAP explanation (this may take a while)...")
    image_np = _denormalize_image(image[0])
    if shap_gen is None:
        shap_gen = PatchSHAPExplainer(model, memory_bank, class_name, grid_size=grid_size)
    t0 = time.perf_counter()
    shap_map = shap_gen.explain(image_np, n_evals=n_evals)
    logger.info("SHAP done in %.2fs", time.perf_counter() - t0)

    # 4. Extract interpretable signals from maps
    logger.info("Extracting interpretable signals...")
    gradcam_region = _gradcam_hotspot_location(gradcam_map, grid_size)
    top_shap = _get_top_shap_patches(shap_map, grid_size, top_k=5)

    # 5. LLM explanation
    logger.info("Generating LLM explanation...")
    prompt = build_explanation_prompt(
        class_name, anomaly_score, top_shap, gradcam_region
    )
    t0 = time.perf_counter()
    explanation, pipe = get_explanation(prompt, pipe=pipe)
    logger.info("LLM generation done in %.2fs", time.perf_counter() - t0)

    latency_ms = round((time.perf_counter() - start) * 1000)

    return {
        "score": anomaly_score,
        "gradcam_map": gradcam_map,
        "shap_map": shap_map,
        "explanation": explanation,
        "latency_ms": latency_ms,
        "pipe": pipe,
    }


def run_explanation_test(
    data_dir,
    output_dir="./outputs/explanation",
    device=None,
    n_shots=4,
    grid_size=7,
    n_evals=200,
    categories=("metal_nut", "leather", "cable"),
    max_anomalous=5,
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

    all_explanations = []
    total_latency = 0.0
    total_images = 0

    pipe = None
    for cat_idx, cat in enumerate(categories):
        logger.info("=" * 50)
        logger.info("Category %d/%d: %s", cat_idx + 1, len(categories), cat)
        logger.info("=" * 50)
        cat_start = time.perf_counter()

        train_ds = MVTecDataset(data_dir, cat, split="train", transform=transform)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        memory = MemoryBank(feat_dim=768, mode="global")
        memory.build(clip_model, train_loader, n_shots, device)
        logger.info("Memory bank built with %d vectors for %s", memory.size, cat)

        test_ds = MVTecDataset(
            data_dir,
            cat,
            split="test",
            transform=transform,
            mask_transform=mask_transform,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        gradcam_gen = CLIPGradCAM(adapt_model)
        shap_gen = PatchSHAPExplainer(adapt_model, memory, cat, grid_size=grid_size)
        logger.info("Initialized GradCAM and SHAP explainers for %s", cat)

        images_done = 0
        for img_idx, (image_tensor, mask_tensor, label) in enumerate(test_loader):
            if int(label.item()) != 1:
                continue
            if images_done >= max_anomalous:
                logger.info("Reached max %d anomalous images for %s", max_anomalous, cat)
                break

            logger.info(
                "Processing anomalous image %d/%d for %s (dataset idx %d)...",
                images_done + 1, max_anomalous, cat, img_idx,
            )
            image_tensor = image_tensor.to(device)
            img_start = time.perf_counter()
            result = explain_defect(
                image_tensor, adapt_model, memory, cat,
                pipe=pipe, gradcam_gen=gradcam_gen, shap_gen=shap_gen,
                grid_size=grid_size, n_evals=n_evals,
            )
            img_elapsed = time.perf_counter() - img_start
            pipe = result.pop("pipe")

            logger.info(
                "Finished %s image %d/%d in %.1fs (score=%.4f)",
                cat, images_done + 1, max_anomalous, img_elapsed, result['score'],
            )
            print(
                f"[{cat} #{images_done + 1}] "
                f"score={result['score']:.4f} "
                f"latency={result['latency_ms']}ms"
            )
            print(f"  Explanation: {result['explanation']}")

            all_explanations.append(result)
            total_latency += result["latency_ms"]
            total_images += 1
            images_done += 1

        cat_elapsed = time.perf_counter() - cat_start
        logger.info("Category %s complete in %.1fs", cat, cat_elapsed)

    avg_latency = total_latency / total_images if total_images > 0 else 0
    logger.info("=" * 50)
    logger.info("SUMMARY: %d images across %d categories", total_images, len(categories))
    logger.info("Average latency: %.0fms  |  Total: %.0fms", avg_latency, total_latency)
    logger.info("=" * 50)
    print(f"\nTotal images: {total_images}")
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"Total latency: {total_latency:.0f}ms")
    return all_explanations


def main():
    parser = argparse.ArgumentParser(
        description="LLM explanation demo for AdaptCLIP on MVTec AD"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/explanation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_shots", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--n_evals", type=int, default=200)
    parser.add_argument(
        "--categories", nargs="+",
        default=["metal_nut", "leather", "cable"],
    )
    parser.add_argument("--max_anomalous", type=int, default=5)
    args = parser.parse_args()

    run_explanation_test(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_shots=args.n_shots,
        grid_size=args.grid_size,
        n_evals=args.n_evals,
        categories=tuple(args.categories),
        max_anomalous=args.max_anomalous,
    )


if __name__ == "__main__":
    main()
