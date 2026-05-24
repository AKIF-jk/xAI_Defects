import asyncio
import logging
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.mvtec_dataset import MVTecDataset
from model.adaptclip import AdaptCLIPModel
from model.backbone import load_backbone
from model.memory_bank import MemoryBank
from model.score_map import compute_patch_scores, overlay_heatmap, scores_to_heatmap
from xai.explainer_llm import (
    _get_top_shap_patches,
    _gradcam_hotspot_location,
    build_explanation_prompt,
    get_explanation,
)
from xai.gradcam import CLIPGradCAM
from xai.shap_explainer import PatchSHAPExplainer


MODEL_VERSION = "adaptclip-vit-l14"
DEFAULT_SHOT_MODE = "4-shot"
DEFAULT_N_SHOTS = 4
DEFAULT_IMG_SIZE = 224
DEFAULT_GRID_SIZE = 5
DEFAULT_SHAP_EVALS = 50
DEFAULT_THRESHOLD = float(os.getenv("ADAPTCLIP_DEFECT_THRESHOLD", "0.5"))
DEFAULT_DATA_DIR = os.getenv(
    "MVTEC_DATA_DIR",
    os.getenv("DATA_DIR", "./mvtec_anomaly_detection"),
)
ARTIFACT_DIR = Path(tempfile.mkdtemp(prefix="adaptclip_api_"))

VALID_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("adaptclip-api")


def _load_memory_banks(
    clip_model: torch.nn.Module,
    preprocess: Any,
    data_dir: str,
    device: str,
    n_shots: int,
) -> Dict[str, MemoryBank]:
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(
            f"MVTec data directory not found: {data_root}. Set MVTEC_DATA_DIR."
        )

    banks: Dict[str, MemoryBank] = {}
    for category in VALID_CATEGORIES:
        train_dir = data_root / category / "train" / "good"
        if not train_dir.exists():
            raise FileNotFoundError(f"Missing MVTec support directory: {train_dir}")

        dataset = MVTecDataset(str(data_root), category, split="train", transform=preprocess)
        if len(dataset) < n_shots:
            raise ValueError(
                f"Category {category} has {len(dataset)} normal images; need {n_shots}"
            )

        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        bank = MemoryBank(feat_dim=768, mode="hybrid")
        bank.build(clip_model, loader, n_shots, device)
        if bank.size != n_shots or bank.get_patch_bank().numel() == 0:
            raise RuntimeError(f"Failed to build complete memory bank for {category}")
        banks[category] = bank
        logger.info("Loaded %s memory bank with %d shots", category, bank.size)

    return banks


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting %s on %s", MODEL_VERSION, device)
    clip_model, _, preprocess, device = load_backbone(device)
    adapt_model = AdaptCLIPModel(clip_model, device).to(device)
    adapt_model.eval()
    memory_banks = _load_memory_banks(
        clip_model=clip_model,
        preprocess=preprocess,
        data_dir=DEFAULT_DATA_DIR,
        device=device,
        n_shots=DEFAULT_N_SHOTS,
    )

    app.state.device = device
    app.state.clip_model = clip_model
    app.state.preprocess = preprocess
    app.state.model = adapt_model
    app.state.memory_banks = memory_banks
    app.state.gradcam = CLIPGradCAM(adapt_model)
    app.state.shap_generators = {}
    app.state.llm_pipe = None
    app.state.artifact_dir = ARTIFACT_DIR
    app.state.inference_lock = asyncio.Lock()
    yield

    if hasattr(adapt_model, "remove_hook"):
        adapt_model.remove_hook()


app = FastAPI(title="AdaptCLIP Inference API", version=MODEL_VERSION, lifespan=lifespan)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACT_DIR)), name="artifacts")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error at %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def _memory_tensor(memory_bank: MemoryBank, device: str) -> torch.Tensor:
    vectors = memory_bank.index.reconstruct_n(0, memory_bank.index.ntotal)
    if vectors.size == 0:
        raise ValueError("memory bank is empty")
    return torch.from_numpy(vectors).float().to(device)


def _read_image(upload: UploadFile) -> Image.Image:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="file must be an image MIME type")
    try:
        return Image.open(upload.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="file is not a valid image") from exc


def _display_image(image: Image.Image, size: int) -> np.ndarray:
    prepared = image.resize((size, size), Image.BICUBIC)
    return np.asarray(prepared.convert("RGB"), dtype=np.uint8)


def _patch_tokens_from_forward(
    patch_features: torch.Tensor,
    clip_model: torch.nn.Module,
    global_dim: int,
) -> torch.Tensor:
    if patch_features is None:
        raise RuntimeError("CLIP patch features were not captured")
    tokens = patch_features
    if tokens.dim() == 3 and tokens.shape[0] > tokens.shape[1] and tokens.shape[0] > 32:
        tokens = tokens.permute(1, 0, 2)
    if tokens.shape[-1] != global_dim:
        proj = getattr(clip_model.visual, "proj", None)
        if proj is not None and tokens.shape[-1] == proj.shape[0]:
            tokens = tokens @ proj.detach().to(tokens.device)
    n_tokens = tokens.shape[1]
    grid_without_cls = int((n_tokens - 1) ** 0.5)
    if grid_without_cls * grid_without_cls == n_tokens - 1:
        return tokens[:, 1:, :]
    grid = int(n_tokens ** 0.5)
    if grid * grid == n_tokens:
        return tokens
    raise RuntimeError(f"Patch token count {n_tokens} is not square-grid compatible")


def _score_and_heatmap(
    model: AdaptCLIPModel,
    clip_model: torch.nn.Module,
    image_tensor: torch.Tensor,
    memory_bank: MemoryBank,
    category: str,
    device: str,
) -> tuple[float, np.ndarray]:
    memory = _memory_tensor(memory_bank, device)
    with torch.no_grad():
        score_tensor, patch_features = model(image_tensor, memory, category)

    score = float(score_tensor.squeeze().item())
    patch_tokens = _patch_tokens_from_forward(
        patch_features,
        clip_model=clip_model,
        global_dim=memory.shape[-1],
    )
    patch_scores = compute_patch_scores(
        patch_tokens[0],
        memory_bank.get_patch_bank().to(device),
        metric="cosine",
        top_k=3,
    )
    heatmap = scores_to_heatmap(patch_scores, img_size=DEFAULT_IMG_SIZE)
    return score, heatmap.squeeze().detach().cpu().numpy().astype(np.float32)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(arr.min())
    hi = float(arr.max())
    if hi <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / hi


def _save_rgb(array: np.ndarray, artifact_dir: Path, prefix: str) -> str:
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    path = artifact_dir / filename
    Image.fromarray(array.astype(np.uint8)).save(path)
    return filename


def _shap_overlay(original: np.ndarray, shap_map: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm

    shap_norm = _normalize_map(shap_map)
    colored = cm.get_cmap("jet")(shap_norm)[:, :, :3]
    base = original.astype(np.float32) / 255.0
    overlay = np.clip(0.5 * base + 0.5 * colored, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def _get_shap_generator(app_state: Any, category: str) -> PatchSHAPExplainer:
    generator = app_state.shap_generators.get(category)
    if generator is None:
        generator = PatchSHAPExplainer(
            app_state.model,
            app_state.memory_banks[category],
            category,
            grid_size=DEFAULT_GRID_SIZE,
        )
        app_state.shap_generators[category] = generator
    return generator


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_VERSION,
        "categories_loaded": len(app.state.memory_banks),
        "device": app.state.device,
    }


@app.get("/categories")
async def categories():
    return VALID_CATEGORIES


@app.post("/inspect")
async def inspect(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form(...),
    shot_mode: str = Form(DEFAULT_SHOT_MODE),
    explain: bool = Form(True),
):
    start = time.perf_counter()
    if category not in VALID_CATEGORIES:
        logger.error("Invalid category requested: %s", category)
        raise HTTPException(status_code=422, detail=f"invalid category: {category}")

    image = _read_image(file)
    state = request.app.state

    try:
        async with state.inference_lock:
            image_tensor = state.preprocess(image).unsqueeze(0).to(state.device)
            original = _display_image(image, DEFAULT_IMG_SIZE)
            memory_bank = state.memory_banks[category]

            score, score_map = _score_and_heatmap(
                model=state.model,
                clip_model=state.clip_model,
                image_tensor=image_tensor,
                memory_bank=memory_bank,
                category=category,
                device=state.device,
            )

            gradcam_map = state.gradcam.generate(
                image_tensor,
                memory_bank,
                category,
                img_size=DEFAULT_IMG_SIZE,
                score_mode="patch",
            )
            combined_heatmap = np.maximum(score_map, _normalize_map(gradcam_map))
            heatmap_file = _save_rgb(
                overlay_heatmap(original, combined_heatmap, alpha=0.5),
                state.artifact_dir,
                "heatmap",
            )

            shap_url: Optional[str] = None
            explanation = ""
            if explain:
                shap_gen = _get_shap_generator(state, category)
                shap_map = shap_gen.explain(original, n_evals=DEFAULT_SHAP_EVALS)
                shap_file = _save_rgb(
                    _shap_overlay(original, shap_map),
                    state.artifact_dir,
                    "shap",
                )
                shap_url = str(request.url_for("artifacts", path=shap_file))

                gradcam_region = _gradcam_hotspot_location(
                    gradcam_map,
                    DEFAULT_GRID_SIZE,
                )
                top_shap = _get_top_shap_patches(
                    shap_map,
                    DEFAULT_GRID_SIZE,
                    top_k=5,
                )
                prompt = build_explanation_prompt(category, score, top_shap, gradcam_region)
                explanation, state.llm_pipe = get_explanation(
                    prompt,
                    pipe=state.llm_pipe,
                    device=state.device,
                )

            latency_ms = round((time.perf_counter() - start) * 1000)
            return {
                "score": score,
                "defect_detected": bool(score >= DEFAULT_THRESHOLD),
                "explanation": explanation,
                "heatmap_url": str(request.url_for("artifacts", path=heatmap_file)),
                "shap_url": shap_url,
                "latency_ms": latency_ms,
                "model_version": MODEL_VERSION,
                "shot_mode": shot_mode,
            }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failure for category=%s: %s", category, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# %% Test cell
def run_api_test_cell(
    host: str = "127.0.0.1",
    port: int = 8000,
    data_dir: str = DEFAULT_DATA_DIR,
):
    import glob
    import json
    import threading

    import requests
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    for _ in range(120):
        try:
            if requests.get(f"{base_url}/health", timeout=2).ok:
                break
        except requests.RequestException:
            time.sleep(1)
    else:
        raise RuntimeError("server did not become healthy")

    candidates = sorted(
        glob.glob(os.path.join(data_dir, "bottle", "test", "*", "*.png"))
    )
    if not candidates:
        raise FileNotFoundError(f"no bottle test images found under {data_dir}")

    with open(candidates[0], "rb") as image_file:
        response = requests.post(
            f"{base_url}/inspect",
            files={"file": ("bottle.png", image_file, "image/png")},
            data={"category": "bottle", "shot_mode": "4-shot", "explain": "true"},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()

    required = {
        "score",
        "defect_detected",
        "explanation",
        "heatmap_url",
        "shap_url",
        "latency_ms",
        "model_version",
        "shot_mode",
    }
    assert required.issubset(payload), f"missing fields: {required - set(payload)}"
    print(json.dumps(payload, indent=2))
    server.should_exit = True
    thread.join(timeout=10)
    return payload


if __name__ == "__main__" and os.getenv("RUN_API_TEST_CELL") == "1":
    run_api_test_cell()
