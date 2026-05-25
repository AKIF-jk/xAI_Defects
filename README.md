---
title: AdaptCLIP Defect Inspector
sdk: gradio
app_file: demo_app.py
pinned: false
---

# AdaptCLIP Explainable Defect Inspector

Portfolio demo for an explainable industrial anomaly detector built around
AdaptCLIP-style few-shot scoring, score maps, Grad-CAM, SHAP, and deterministic
inspection explanations.

## Public demo

The Hugging Face Space runs `demo_app.py`, a CPU-friendly Gradio app that loads
precomputed examples from `demo_manifest.json` and `demo_assets/`. It is meant
to open quickly on free hosting and does not download model weights or load
`torch`, `open_clip`, `captum`, `shap`, or transformer LLMs at startup.

Add the public Space URL here after deployment:

`https://huggingface.co/spaces/<user>/<space-name>`

## Why the public Space is sample-based

The full pipeline uses ViT-L/14 features, few-shot memory banks, Grad-CAM, SHAP,
and optional explanation generation. Hugging Face Spaces is designed for public
ML demos and includes a free CPU tier, but free hardware is not a reliable
always-on GPU service and Spaces may sleep when inactive. For zero-cost
portfolio hosting, this repo publishes a curated artifact bundle instead of
running full inference for every visitor request.

Reference: https://huggingface.co/docs/hub/main/spaces-overview

## Architecture

```text
Colab / GPU live mode
  Google Drive MVTec AD dataset
    -> AdaptCLIP few-shot memory bank
    -> score map + Grad-CAM + SHAP
    -> deterministic explanation
    -> scripts/export_demo_bundle.py

Hugging Face Space / free CPU
  demo_manifest.json + demo_assets/
    -> demo_app.py
    -> Gradio sample browser
```

## Local demo

Install only the public demo dependencies:

```bash
pip install -r requirements.txt
python demo_app.py
```

The checked-in manifest uses synthetic placeholder previews so the app can be
tested immediately. Before publishing the final portfolio Space, replace them
with Colab-exported artifacts from real inspection runs.

## Export real demo artifacts from Colab

After mounting Google Drive and installing the full dependencies:

```bash
pip install -r requirements-colab.txt
python scripts/export_demo_bundle.py \
  --data_dir /content/drive/MyDrive/defect_inspector/data/mvtec/mvtec_anomaly_detection \
  --output_dir /content/xAI_Defects \
  --categories bottle,cable,tile \
  --max_anomalous 2 \
  --max_normal 1
```

The exporter writes:

- `demo_manifest.json`
- `demo_assets/<sample-id>/original.png`
- `demo_assets/<sample-id>/heatmap.png`
- `demo_assets/<sample-id>/shap.png`

Use 10-20 strong examples across categories for a compact portfolio demo. Do
not upload the full MVTec dataset to the Space repo.

## Colab live mode

For interviews or walkthroughs, run the full API in Colab where the dataset and
GPU session are available:

```bash
export MVTEC_DATA_DIR=/content/drive/MyDrive/defect_inspector/data/mvtec/mvtec_anomaly_detection
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then test:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/categories
```

During a live presentation, expose the Colab server with a temporary tunnel if
needed. Treat that as a session-based live demo, not the public always-on
deployment.

## Deployment checklist

- Replace placeholder assets with real exported artifacts.
- Keep `requirements.txt` lightweight for the Space.
- Keep heavyweight ML dependencies in `requirements-colab.txt`.
- Push `README.md`, `demo_app.py`, `demo_manifest.json`, and `demo_assets/` to
  the Space repo.
- Confirm cold start completes without model downloads.
- Confirm each category and sample renders all three images and explanation.

## Dataset note

The public repo should contain only a small demo artifact bundle. If
redistribution of original MVTec images is uncertain for the final Space, use
derived demo panels or screenshots and document the MVTec AD source rather than
publishing the dataset itself.
