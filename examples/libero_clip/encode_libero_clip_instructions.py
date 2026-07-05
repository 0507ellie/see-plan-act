from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from load_clip_hf_cuda import load_clip


INSTRUCTIONS = [
    "pick up the ketchup",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the alphabet soup and place it in the basket",
]


def default_output_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "outputs" / "libero_clip_instruction_embeddings.npy"


def main() -> None:
    model, processor, device = load_clip()
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    inputs = processor(text=INSTRUCTIONS, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    features = text_features.detach().cpu().numpy()
    print("embedding shape:", features.shape)
    print()

    for instruction, vector in zip(INSTRUCTIONS, features):
        preview = np.array2string(vector[:10], precision=4, suppress_small=False)
        print(f"instruction: {instruction}")
        print(f"first 10 dims: {preview}")
        print()

    similarity = features @ features.T
    print("cosine similarity matrix:")
    header = "      " + " ".join([f"{i:>7}" for i in range(len(INSTRUCTIONS))])
    print(header)
    for i, row in enumerate(similarity):
        values = " ".join([f"{value:7.3f}" for value in row])
        print(f"{i:>3}:  {values}")

    output_path = Path(os.environ.get("LIBERO_CLIP_EMBEDDINGS_OUT", default_output_path()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    print()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
