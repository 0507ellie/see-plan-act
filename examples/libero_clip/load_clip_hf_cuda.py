from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL_PATH = Path(r"D:\huggingface_models\openai_clip-vit-base-patch32")


def get_model_path() -> Path:
    return Path(os.environ.get("CLIP_MODEL_PATH", DEFAULT_MODEL_PATH))


def load_clip(
    model_path: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool = True,
) -> tuple[CLIPModel, CLIPProcessor, str]:
    """Load HuggingFace CLIP and move it to CUDA when available."""
    resolved_model_path = Path(model_path) if model_path is not None else get_model_path()
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = CLIPProcessor.from_pretrained(
        str(resolved_model_path),
        local_files_only=local_files_only,
    )
    model = CLIPModel.from_pretrained(
        str(resolved_model_path),
        local_files_only=local_files_only,
    ).to(resolved_device)
    model.eval()

    return model, processor, resolved_device


def main() -> None:
    model, processor, device = load_clip()
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    texts = [
        "a robot arm picking up alphabet soup",
        "a basket on the floor",
    ]

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print("image_embeds:", outputs.image_embeds.shape, outputs.image_embeds.device)
    print("text_embeds:", outputs.text_embeds.shape, outputs.text_embeds.device)
    print("logits_per_image:", outputs.logits_per_image.detach().cpu().numpy())


if __name__ == "__main__":
    main()
