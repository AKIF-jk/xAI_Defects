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


def build_explanation_prompt(category, anomaly_score, top_shap_patches, gradcam_region):
    shap_lines = "\n".join(
        f"  - Patch at {p['position']} (contribution: {p['value']:.3f})"
        for p in top_shap_patches
    )
    prompt = (
        f"Product category: {category}\n"
        f"Anomaly score: {anomaly_score:.2f} (0 = normal, 1 = defective)\n"
        f"GradCAM hotspot: {gradcam_region}\n"
        f"Top SHAP-contributing patches:\n{shap_lines}\n\n"
        "Based on the above, what defect is present, where is it located, "
        "and what is a likely cause?"
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
        return outputs[0]["generated_text"][-1]["content"].strip()
    return outputs[0]["generated_text"].strip()


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
    return [
        {"position": patch_position_to_text(p["index"], G), "value": p["value"]}
        for p in patches[:top_k]
    ]


def _denormalize_image(tensor):
    tensor = tensor.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = torch.clamp(tensor * std + mean, 0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def explain_defect(
    image,
    model,
    memory_bank,
    class_name,
    pipe=None,
    grid_size=7,
    n_evals=200,
):
    start = time.perf_counter()
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    # 1. Score via AdaptCLIP forward
    with torch.no_grad():
        score_tensor, _ = model(image, memory_bank, class_name)
    anomaly_score = float(score_tensor[0].sigmoid().item())

    # 2. GradCAM heatmap
    gradcam_gen = CLIPGradCAM(model)
    gradcam_map = gradcam_gen.generate(
        image, memory_bank, class_name, img_size=image.shape[-1]
    )

    # 3. SHAP attribution map
    image_np = _denormalize_image(image[0])
    shap_gen = PatchSHAPExplainer(model, memory_bank, class_name, grid_size=grid_size)
    shap_map = shap_gen.explain(image_np, n_evals=n_evals)

    # 4. Extract interpretable signals from maps
    gradcam_region = _gradcam_hotspot_location(gradcam_map, grid_size)
    top_shap = _get_top_shap_patches(shap_map, grid_size, top_k=5)

    # 5. LLM explanation
    prompt = build_explanation_prompt(
        class_name, anomaly_score, top_shap, gradcam_region
    )
    explanation = get_explanation(prompt, pipe=pipe)

    latency_ms = round((time.perf_counter() - start) * 1000)

    return {
        "score": anomaly_score,
        "gradcam_map": gradcam_map,
        "shap_map": shap_map,
        "explanation": explanation,
        "latency_ms": latency_ms,
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

    for cat in categories:
        logger.info("Processing category: %s", cat)
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

        pipe = None
        images_done = 0
        for image_tensor, mask_tensor, label in test_loader:
            if int(label.item()) != 1:
                continue
            if images_done >= max_anomalous:
                break

            image_tensor = image_tensor.to(device)
            result = explain_defect(
                image_tensor, adapt_model, memory, cat,
                pipe=pipe, grid_size=grid_size, n_evals=n_evals,
            )
            pipe = None

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

    avg_latency = total_latency / total_images if total_images > 0 else 0
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
