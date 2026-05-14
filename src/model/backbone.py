import argparse
import os
import json
import torch
import open_clip


def load_backbone(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer, preprocess, device


def print_architecture(model, device):
    total = sum(p.numel() for p in model.parameters())
    print(f"Device:               {device}")
    print(f"Total parameters:     {total:,}  (all frozen)")

    if hasattr(model.visual, "output_dim"):
        print(f"Visual encoder dim:   {model.visual.output_dim}")
    if hasattr(model.visual, "transformer"):
        if hasattr(model.visual.transformer, "resblocks"):
            n_layers = len(model.visual.transformer.resblocks)
            print(f"Transformer layers:   {n_layers}")
    if hasattr(model.visual, "patch_size"):
        print(f"Patch size:           {model.visual.patch_size}")
    if hasattr(model.visual, "image_size"):
        print(f"Input resolution:     {model.visual.image_size}")
    if hasattr(model.visual, "grid_size"):
        print(f"Grid size:            {model.visual.grid_size}")


def sanity_check(model, tokenizer, preprocess, device, image_path):
    from PIL import Image
    import torch.nn.functional as F

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)

    texts = ["a photo of a bottle", "a photo of a defective bottle"]
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    similarities = (image_features @ text_features.T).squeeze(0).cpu().tolist()

    print(f"\nSanity check image: {image_path}")
    for text, sim in zip(texts, similarities):
        print(f"  {sim:.4f}  —  \"{text}\"")
    print(f"\n  -> Normal prompt scored {'higher' if similarities[0] > similarities[1] else 'lower'}, "
          f"difference: {abs(similarities[0] - similarities[1]):.4f}")


def verify_backbone(data_dir=None):
    print("Loading OpenCLIP ViT-L/14 backbone...")
    model, tokenizer, preprocess, device = load_backbone()
    print_architecture(model, device)

    if data_dir is not None:
        bottle_dir = os.path.join(data_dir, "bottle", "train", "good")
        if os.path.isdir(bottle_dir):
            images = sorted(
                p for p in os.listdir(bottle_dir) if p.endswith(".png")
            )
            if images:
                sanity_check(
                    model, tokenizer, preprocess, device,
                    os.path.join(bottle_dir, images[0]),
                )
            else:
                print("No PNG images found in bottle/train/good/")
        else:
            print(f"bottle/train/good/ not found under {data_dir}")
    else:
        print("Skip sanity check (no --data_dir provided)")

    config = {
        "model_name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "device": device,
        "embed_dim": model.visual.output_dim
        if hasattr(model.visual, "output_dim")
        else None,
        "patch_size": model.visual.patch_size
        if hasattr(model.visual, "patch_size")
        else None,
        "input_resolution": model.visual.image_size
        if hasattr(model.visual, "image_size")
        else None,
    }

    print("\nBackbone config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    return model, tokenizer, preprocess, device, config


def main():
    parser = argparse.ArgumentParser(
        description="Verify OpenCLIP ViT-L/14 backbone"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to mvtec_anomaly_detection/ for sanity check image",
    )
    args = parser.parse_args()
    verify_backbone(args.data_dir)


if __name__ == "__main__":
    main()
