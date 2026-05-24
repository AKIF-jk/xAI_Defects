"""CPU-friendly Gradio demo for the AdaptCLIP portfolio Space.

This app intentionally renders precomputed inspection artifacts from
``demo_manifest.json`` and ``demo_assets/``. It does not import torch,
open_clip, SHAP, Grad-CAM, or any LLM dependency, so Hugging Face Spaces can
start quickly on the free CPU tier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr


APP_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = APP_DIR / "demo_manifest.json"
ASSET_ROOT = APP_DIR / "demo_assets"


@dataclass(frozen=True)
class DemoSample:
    category: str
    image_id: str
    title: str
    score: float | None
    defect_detected: bool | None
    explanation: str
    original_path: str
    heatmap_path: str
    shap_path: str
    model_version: str
    shot_mode: str
    defect_type: str = ""
    true_label: str = ""
    source_note: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DemoSample":
        category = str(record.get("category", "unknown")).strip() or "unknown"
        image_id = str(record.get("image_id", record.get("id", ""))).strip()
        if not image_id:
            image_id = f"{category}-sample"

        score = record.get("score", record.get("anomaly_score"))
        try:
            score_value = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_value = None

        defect = record.get("defect_detected")
        if defect is None and score_value is not None:
            defect = score_value >= float(record.get("threshold", 0.4))

        title = str(record.get("title", image_id)).strip() or image_id
        return cls(
            category=category,
            image_id=image_id,
            title=title,
            score=score_value,
            defect_detected=bool(defect) if defect is not None else None,
            explanation=str(record.get("explanation", "")).strip(),
            original_path=str(record.get("original_path", "")).strip(),
            heatmap_path=str(record.get("heatmap_path", "")).strip(),
            shap_path=str(record.get("shap_path", "")).strip(),
            model_version=str(record.get("model_version", "unknown")).strip(),
            shot_mode=str(record.get("shot_mode", "unknown")).strip(),
            defect_type=str(record.get("defect_type", "")).strip(),
            true_label=str(record.get("true_label", "")).strip(),
            source_note=str(record.get("source_note", "")).strip(),
        )


def _load_manifest() -> list[DemoSample]:
    if not MANIFEST_PATH.exists():
        return []

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        records = payload.get("samples", [])
    else:
        records = payload

    if not isinstance(records, list):
        raise ValueError("demo_manifest.json must contain a list or a {'samples': [...]} object")

    samples = []
    for record in records:
        if isinstance(record, dict):
            samples.append(DemoSample.from_record(record))
    return samples


SAMPLES = _load_manifest()


def _sample_key(sample: DemoSample) -> str:
    return f"{sample.category}::{sample.image_id}"


def _sample_label(sample: DemoSample) -> str:
    parts = [sample.title]
    if sample.defect_type:
        parts.append(sample.defect_type.replace("_", " "))
    if sample.score is not None:
        parts.append(f"score {sample.score:.2f}")
    return " | ".join(parts)


SAMPLE_BY_KEY = {_sample_key(sample): sample for sample in SAMPLES}
CATEGORIES = sorted({sample.category for sample in SAMPLES})


def _choices_for_category(category: str) -> list[tuple[str, str]]:
    return [
        (_sample_label(sample), _sample_key(sample))
        for sample in SAMPLES
        if sample.category == category
    ]


def _safe_asset_path(relative_path: str) -> tuple[str | None, str | None]:
    if not relative_path:
        return None, "Asset path is empty."

    candidate = (APP_DIR / relative_path).resolve()
    try:
        candidate.relative_to(APP_DIR)
    except ValueError:
        return None, f"Asset path escapes the app directory: {relative_path}"

    if not candidate.exists():
        return None, f"Missing asset: {relative_path}"
    return str(candidate), None


def _initial_selection() -> tuple[str | None, str | None]:
    if not CATEGORIES:
        return None, None
    category = CATEGORIES[0]
    choices = _choices_for_category(category)
    sample_key = choices[0][1] if choices else None
    return category, sample_key


def category_changed(category: str):
    choices = _choices_for_category(category)
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value), *render_sample(value)


def render_sample(sample_key: str | None):
    if not sample_key:
        return (
            None,
            None,
            None,
            "No sample selected.",
            "",
            {},
            "Add artifacts with `python scripts/export_demo_bundle.py ...`.",
        )

    sample = SAMPLE_BY_KEY.get(sample_key)
    if sample is None:
        return None, None, None, "Unknown sample.", "", {}, "Select another sample."

    original, original_error = _safe_asset_path(sample.original_path)
    heatmap, heatmap_error = _safe_asset_path(sample.heatmap_path)
    shap, shap_error = _safe_asset_path(sample.shap_path)
    errors = [err for err in (original_error, heatmap_error, shap_error) if err]

    status = "Defect detected" if sample.defect_detected else "No defect detected"
    if sample.defect_detected is None:
        status = "Detection unavailable"
    score = "n/a" if sample.score is None else f"{sample.score:.3f}"

    summary = (
        f"**{status}**  \n"
        f"Category: `{sample.category}`  \n"
        f"Score: `{score}`  \n"
        f"Model: `{sample.model_version}`  \n"
        f"Shot mode: `{sample.shot_mode}`"
    )
    if sample.true_label:
        summary += f"  \nGround truth: `{sample.true_label}`"

    metadata = {
        "category": sample.category,
        "image_id": sample.image_id,
        "defect_type": sample.defect_type,
        "score": sample.score,
        "defect_detected": sample.defect_detected,
        "model_version": sample.model_version,
        "shot_mode": sample.shot_mode,
        "source_note": sample.source_note,
    }

    message = "\n".join(errors) if errors else "Ready."
    return original, heatmap, shap, summary, sample.explanation, metadata, message


initial_category, initial_sample = _initial_selection()

with gr.Blocks(title="AdaptCLIP Defect Inspector") as demo:
    gr.Markdown(
        "# AdaptCLIP Explainable Defect Inspector\n"
        "Public portfolio demo using precomputed sample inspections for fast, "
        "zero-cost CPU hosting. Full live inference is reproducible in Colab."
    )

    if not SAMPLES:
        gr.Warning("No demo samples found. Add demo_manifest.json and demo_assets/.")

    with gr.Row():
        category_dropdown = gr.Dropdown(
            label="Category",
            choices=CATEGORIES,
            value=initial_category,
            interactive=bool(CATEGORIES),
        )
        sample_dropdown = gr.Dropdown(
            label="Sample",
            choices=_choices_for_category(initial_category) if initial_category else [],
            value=initial_sample,
            interactive=bool(SAMPLES),
        )

    with gr.Row():
        original_image = gr.Image(label="Original", type="filepath", height=320)
        heatmap_image = gr.Image(label="Score / Grad-CAM overlay", type="filepath", height=320)
        shap_image = gr.Image(label="SHAP overlay", type="filepath", height=320)

    with gr.Row():
        with gr.Column(scale=1):
            score_markdown = gr.Markdown()
            asset_message = gr.Textbox(label="Asset status", interactive=False)
        with gr.Column(scale=2):
            explanation_box = gr.Textbox(
                label="Explanation",
                lines=5,
                interactive=False,
            )

    metadata_json = gr.JSON(label="Sample metadata")

    with gr.Accordion("Upload mode", open=False):
        gr.Markdown(
            "The public Space does not run ViT-L/14, Grad-CAM, SHAP, or an LLM "
            "on visitor uploads. Use the Colab live-mode instructions in the "
            "README to run full inference against Google Drive data."
        )
        gr.File(label="Upload disabled on free hosted demo", interactive=False)

    category_dropdown.change(
        category_changed,
        inputs=category_dropdown,
        outputs=[
            sample_dropdown,
            original_image,
            heatmap_image,
            shap_image,
            score_markdown,
            explanation_box,
            metadata_json,
            asset_message,
        ],
    )
    sample_dropdown.change(
        render_sample,
        inputs=sample_dropdown,
        outputs=[
            original_image,
            heatmap_image,
            shap_image,
            score_markdown,
            explanation_box,
            metadata_json,
            asset_message,
        ],
    )

    demo.load(
        render_sample,
        inputs=sample_dropdown,
        outputs=[
            original_image,
            heatmap_image,
            shap_image,
            score_markdown,
            explanation_box,
            metadata_json,
            asset_message,
        ],
    )


if __name__ == "__main__":
    demo.launch()
