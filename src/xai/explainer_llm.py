"""
explainer_llm.py  —  Colab free-tier optimised version
=======================================================
Key changes vs original
-----------------------
1. LLM: Mistral-7B (≈14 GB fp16) → google/flan-t5-base (≈1 GB)
   - Uses text2text-generation pipeline (no chat-message format needed)
   - Optional: pass --llm_model with any causal-LM + --use_4bit for quantised Mistral
2. SHAP masker: inpaint_telea (slow) → blur(11,11) (fast, low RAM)
3. Colab-safe defaults: n_evals=50, grid_size=5, max_anomalous=2, n_shots=2
4. LLM is unloaded from GPU (moved to CPU) while CLIP/SHAP run, then swapped back
5. gc.collect() + torch.cuda.empty_cache() after every category
6. --resume flag works as before (checkpoint survives runtime restarts)
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
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from xai.gradcam import CLIPGradCAM
from xai.shap_explainer import PatchSHAPExplainer


# ---------------------------------------------------------------------------
# Checkpoint helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _ckpt_path(output_dir):
    return os.path.join(output_dir, "explainer_checkpoint.pkl")


def _save_ckpt(output_dir, explanations, total_latency, total_images, processed_pairs):
    path = _ckpt_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    data = dict(
        explanations=[
            {k: v for k, v in e.items() if k not in ("gradcam_map", "shap_map")}
            for e in explanations
        ],
        explanations_full=explanations,
        total_latency=total_latency,
        total_images=total_images,
        processed_pairs=list(processed_pairs),
    )
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Checkpoint saved to %s (%d images)", path, total_images)


def _load_ckpt(output_dir):
    path = _ckpt_path(output_dir)
    if not os.path.exists(path):
        logger.info("No checkpoint found at %s", path)
        return [], 0.0, 0, set()
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(
        "Loaded checkpoint from %s (%d images done, %d pairs processed)",
        path, data["total_images"], len(data["processed_pairs"]),
    )
    return (
        data.get("explanations_full", data["explanations"]),
        data["total_latency"],
        data["total_images"],
        set(data["processed_pairs"]),
    )


# ---------------------------------------------------------------------------
# Spatial helpers (unchanged)
# ---------------------------------------------------------------------------

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
        shap_lines = "no strong patch-level evidence"
    else:
        shap_lines = "; ".join(
            f"{p['position']} (contribution {p['value']:.3f})"
            for p in top_shap_patches
        )
    # Flat prompt works for both seq2seq (flan-t5) and causal LMs
    prompt = (
        f"Inspect a {category} on a manufacturing line. "
        f"Known defect types: {vocab}. "
        f"Anomaly severity: {severity} (score={anomaly_score:.2f}). "
        f"GradCAM hotspot: {gradcam_region}. "
        f"SHAP evidence: {shap_lines}. "
        "In one sentence (max 35 words) name the defect type, its location, and the likely cause."
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def load_llm_pipeline(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cpu",
    use_4bit: bool = False,
):
    """
    Load a lightweight LLM pipeline (text-generation only).

    Default: Qwen/Qwen2.5-0.5B-Instruct  — 0.5B params, ~1 GB fp16.
    Fits alongside CLIP on a T4 with room to spare.

    Alternatives:
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"       ~2 GB fp16
        "mistralai/Mistral-7B-Instruct-v0.3"        needs --use_4bit

    NOTE: "text2text-generation" (flan-t5 / T5) was removed from transformers
    ≥ 4.52. All models now use "text-generation" via the unified pipeline.
    """
    from transformers import pipeline as hf_pipeline

    dtype = torch.float16 if device != "cpu" else torch.float32

    kwargs = dict(
        model=model_name,
        torch_dtype=dtype,
    )

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs.pop("torch_dtype", None)
            kwargs["device_map"] = "auto"
            logger.info("4-bit quantisation enabled for %s", model_name)
        except ImportError:
            logger.warning("bitsandbytes not installed — falling back to fp16")
            kwargs["device_map"] = "auto" if device != "cpu" else None
    else:
        kwargs["device_map"] = "auto" if device != "cpu" else None

    pipe = hf_pipeline("text-generation", **kwargs)
    logger.info("LLM pipeline loaded: %s", model_name)
    return pipe


def _move_pipe_to_cpu(pipe):
    """Offload the LLM to CPU RAM so CLIP/SHAP can use the GPU."""
    try:
        if hasattr(pipe, "model"):
            pipe.model.to("cpu")
            logger.debug("LLM moved to CPU")
    except Exception as e:
        logger.debug("Could not move LLM to CPU: %s", e)
    torch.cuda.empty_cache()


def _move_pipe_to_gpu(pipe, device):
    """Bring the LLM back to GPU for inference."""
    if device == "cpu":
        return
    try:
        if hasattr(pipe, "model"):
            pipe.model.to(device)
            logger.debug("LLM moved back to %s", device)
    except Exception as e:
        logger.debug("Could not move LLM to GPU: %s", e)


def get_explanation(
    prompt_text: str,
    pipe=None,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cpu",
    max_new_tokens: int = 80,
    temperature: float = 0.3,
    use_4bit: bool = False,
) -> tuple:
    """
    Generate a one-sentence defect explanation via text-generation pipeline.
    Returns (explanation_text, pipe) so the caller can reuse the pipe.
    """
    if pipe is None:
        pipe = load_llm_pipeline(model, device, use_4bit)

    do_sample = temperature > 0
    system = (
        "You are a quality control expert. "
        "Write exactly ONE sentence (max 35 words) explaining the defect, location, and cause."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        return_full_text=False,
    )
    raw = outputs[0]["generated_text"]
    text = (raw[-1]["content"] if isinstance(raw, list) else raw).strip()
    return text, pipe


# ---------------------------------------------------------------------------
# Map utilities (unchanged)
# ---------------------------------------------------------------------------

def _gradcam_hotspot_location(gradcam_map, grid_size):
    G = int(grid_size)
    h, w = gradcam_map.shape
    y_edges = np.linspace(0, h, G + 1, dtype=int)
    x_edges = np.linspace(0, w, G + 1, dtype=int)
    best_val, best_idx = -1.0, 0
    for gy in range(G):
        for gx in range(G):
            patch_mean = float(
                gradcam_map[y_edges[gy]: y_edges[gy + 1], x_edges[gx]: x_edges[gx + 1]].mean()
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
            val = float(shap_map[y_edges[gy]: y_edges[gy + 1], x_edges[gx]: x_edges[gx + 1]].mean())
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


# ---------------------------------------------------------------------------
# Core explain_defect — with GPU-swap memory management
# ---------------------------------------------------------------------------

def explain_defect(
    image,
    model,
    memory_bank,
    class_name,
    pipe=None,
    gradcam_gen=None,
    shap_gen=None,
    grid_size=5,
    n_evals=50,
    llm_model="google/flan-t5-base",
    use_4bit=False,
):
    """
    Run GradCAM + SHAP + LLM for one image.

    GPU-swap strategy
    -----------------
    1. CLIP forward (anomaly score) — GPU
    2. GradCAM                      — GPU
    3. Offload LLM to CPU (if loaded), run SHAP on GPU, reload LLM
    4. LLM inference                — GPU (or CPU for flan-t5-base)
    """
    start = time.perf_counter()
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    device = image.device
    memory_tensor = _as_memory_tensor(memory_bank, device)

    # 1. Anomaly score
    logger.info("Computing anomaly score...")
    with torch.no_grad():
        score_tensor, _ = model(image, memory_tensor, class_name)
    anomaly_score = float(score_tensor[0].item())
    logger.info("Anomaly score: %.4f", anomaly_score)

    # 2. GradCAM
    logger.info("Generating GradCAM heatmap...")
    t0 = time.perf_counter()
    if gradcam_gen is None:
        gradcam_gen = CLIPGradCAM(model)
    gradcam_map = gradcam_gen.generate(image, memory_bank, class_name, img_size=image.shape[-1])
    logger.info("GradCAM done in %.2fs", time.perf_counter() - t0)

    # 3. Offload LLM → run SHAP → restore LLM
    if pipe is not None:
        _move_pipe_to_cpu(pipe)

    logger.info("Running SHAP (n_evals=%d, grid_size=%d)...", n_evals, grid_size)
    image_np = _denormalize_image(image[0])
    if shap_gen is None:
        shap_gen = PatchSHAPExplainer(model, memory_bank, class_name, grid_size=grid_size)
    t0 = time.perf_counter()
    shap_map = shap_gen.explain(image_np, n_evals=n_evals)
    logger.info("SHAP done in %.2fs", time.perf_counter() - t0)

    if pipe is not None:
        _move_pipe_to_gpu(pipe, str(device))

    # 4. Extract signals
    gradcam_region = _gradcam_hotspot_location(gradcam_map, grid_size)
    top_shap = _get_top_shap_patches(shap_map, grid_size, top_k=5)

    # 5. LLM explanation
    logger.info("Generating LLM explanation...")
    prompt = build_explanation_prompt(class_name, anomaly_score, top_shap, gradcam_region)
    t0 = time.perf_counter()
    explanation, pipe = get_explanation(
        prompt, pipe=pipe, model=llm_model,
        device=str(device), use_4bit=use_4bit,
    )
    logger.info("LLM done in %.2fs", time.perf_counter() - t0)

    latency_ms = round((time.perf_counter() - start) * 1000)
    return {
        "score": anomaly_score,
        "gradcam_map": gradcam_map,
        "shap_map": shap_map,
        "explanation": explanation,
        "latency_ms": latency_ms,
        "pipe": pipe,
    }


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def run_explanation_test(
    data_dir,
    output_dir="./outputs/explanation",
    device=None,
    n_shots=2,          # ↓ was 4
    grid_size=5,        # ↓ was 7  (25 patches vs 49)
    n_evals=50,         # ↓ was 200
    categories=("metal_nut", "leather", "cable"),
    max_anomalous=2,    # ↓ was 5
    resume=False,
    llm_model="Qwen/Qwen2.5-0.5B-Instruct",
    use_4bit=False,
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

    if resume:
        all_explanations, total_latency, total_images, processed_pairs = _load_ckpt(output_dir)
    else:
        all_explanations, total_latency, total_images, processed_pairs = [], 0.0, 0, set()

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
            data_dir, cat, split="test",
            transform=transform, mask_transform=mask_transform,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        gradcam_gen = CLIPGradCAM(adapt_model)
        shap_gen = PatchSHAPExplainer(adapt_model, memory, cat, grid_size=grid_size)
        logger.info("Explainers initialised for %s", cat)

        images_done = 0
        for img_idx, (image_tensor, mask_tensor, label) in enumerate(test_loader):
            if int(label.item()) != 1:
                continue
            if images_done >= max_anomalous:
                logger.info("Reached max %d anomalous images for %s", max_anomalous, cat)
                break

            pair = (cat, img_idx)
            if pair in processed_pairs:
                logger.info("Skipping already-processed %s image idx %d", cat, img_idx)
                images_done += 1
                continue

            logger.info(
                "Processing anomalous image %d/%d for %s (dataset idx %d)...",
                images_done + 1, max_anomalous, cat, img_idx,
            )
            image_tensor = image_tensor.to(device)
            img_start = time.perf_counter()
            result = explain_defect(
                image_tensor, adapt_model, memory, cat,
                pipe=pipe,
                gradcam_gen=gradcam_gen,
                shap_gen=shap_gen,
                grid_size=grid_size,
                n_evals=n_evals,
                llm_model=llm_model,
                use_4bit=use_4bit,
            )
            img_elapsed = time.perf_counter() - img_start
            pipe = result.pop("pipe")

            logger.info(
                "Finished %s image %d/%d in %.1fs (score=%.4f)",
                cat, images_done + 1, max_anomalous, img_elapsed, result["score"],
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
            processed_pairs.add(pair)
            _save_ckpt(output_dir, all_explanations, total_latency, total_images, processed_pairs)

        # ---- Per-category cleanup -------------------------------------------
        cat_elapsed = time.perf_counter() - cat_start
        logger.info("Category %s complete in %.1fs", cat, cat_elapsed)

        del gradcam_gen, shap_gen, train_ds, test_ds, train_loader, test_loader, memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                "GPU memory after cleanup: %.1f MB allocated",
                torch.cuda.memory_allocated() / 1e6,
            )
        # ---------------------------------------------------------------------

    avg_latency = total_latency / total_images if total_images > 0 else 0
    logger.info("=" * 50)
    logger.info("SUMMARY: %d images across %d categories", total_images, len(categories))
    logger.info("Average latency: %.0fms  |  Total: %.0fms", avg_latency, total_latency)
    logger.info("=" * 50)
    print(f"\nTotal images: {total_images}")
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"Total latency: {total_latency:.0f}ms")
    return all_explanations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM explanation demo for AdaptCLIP on MVTec AD (Colab free-tier optimised)"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/explanation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_shots", type=int, default=2,
                        help="Normal shots for memory bank (default 2, was 4)")
    parser.add_argument("--grid_size", type=int, default=5,
                        help="Patch grid size (default 5, was 7)")
    parser.add_argument("--n_evals", type=int, default=50,
                        help="SHAP evaluations per image (default 50, was 200)")
    parser.add_argument("--categories", nargs="+",
                        default=["metal_nut", "leather", "cable"])
    parser.add_argument("--max_anomalous", type=int, default=2,
                        help="Max anomalous images per category (default 2, was 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument(
        "--llm_model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "HuggingFace model for text explanation. "
            "Default: Qwen/Qwen2.5-0.5B-Instruct (~1 GB fp16, fits on T4). "
            "Alternatives: TinyLlama/TinyLlama-1.1B-Chat-v1.0, "
            "mistralai/Mistral-7B-Instruct-v0.3 (needs --use_4bit)"
        ),
    )
    parser.add_argument("--use_4bit", action="store_true",
                        help="Load LLM in 4-bit (requires bitsandbytes). "
                             "Needed for Mistral-7B on Colab free tier.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_explanation_test(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_shots=args.n_shots,
        grid_size=args.grid_size,
        n_evals=args.n_evals,
        categories=tuple(args.categories),
        max_anomalous=args.max_anomalous,
        resume=args.resume,
        llm_model=args.llm_model,
        use_4bit=args.use_4bit,
    )


if __name__ == "__main__":
    main()