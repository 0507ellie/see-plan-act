from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from transformers import CLIPModel, CLIPProcessor

from load_clip_hf_cuda import get_model_path


class FrozenCLIPTextEncoder:
    """Frozen CLIP text encoder: string(s) -> 512-dim normalized vector(s)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        normalize: bool = True,
        local_files_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else get_model_path()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        self.processor = CLIPProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=local_files_only,
        )
        self.model = CLIPModel.from_pretrained(
            str(self.model_path),
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.projection_dim)

    def encode(
        self,
        texts: str | Iterable[str],
        return_tensor: bool = True,
    ) -> torch.Tensor:
        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)

        inputs = self.processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if self.normalize:
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if single_input:
            features = features[0]

        return features if return_tensor else features.detach().cpu().numpy()


def main() -> None:
    encoder = FrozenCLIPTextEncoder()
    examples = [
        "pick up the ketchup",
        "pick up the ketchup and place it in the basket",
        "pick up the milk and place it in the basket",
    ]

    vectors = encoder.encode(examples)
    print("device:", encoder.device)
    if encoder.device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    print("embedding_dim:", encoder.embedding_dim)
    print("shape:", tuple(vectors.shape))
    print("requires_grad:", vectors.requires_grad)
    print("first vector first 10 dims:", vectors[0, :10].detach().cpu().numpy())


if __name__ == "__main__":
    main()
