"""timm-based frozen embedding backend for Stage 1 baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

from .base import EncoderSpec, TileEmbeddingEncoder


class TimmTileEmbeddingEncoder(TileEmbeddingEncoder):
    """Frozen timm encoder adapter with Stage 1 fallback semantics."""

    def __init__(self, spec: EncoderSpec, require_pretrained: bool) -> None:
        self.spec = spec
        self.device = torch.device(spec.device)
        self.weight_source = "pretrained"
        try:
            self.model = timm.create_model(spec.model_name, pretrained=spec.pretrained, num_classes=0)
        except Exception:
            if require_pretrained:
                raise
            self.model = timm.create_model(spec.model_name, pretrained=False, num_classes=0)
            self.weight_source = "random_init_fallback"
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.to(self.device)
        self.data_cfg = resolve_model_data_config(self.model)
        self.transform = create_transform(**self.data_cfg, is_training=False)

    @property
    def batch_size(self) -> int:
        return int(self.spec.batch_size)

    def metadata(self) -> dict[str, Any]:
        return {
            "encoder_backend": "timm",
            "encoder_model_name": self.spec.model_name,
            "encoder_weight_source": self.weight_source,
        }

    def encode_tiles(self, tiles: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.transform(tile) for tile in tiles], dim=0).to(self.device)
        with torch.inference_mode():
            emb = self.model(batch)
        return emb.detach().cpu().numpy().astype(np.float32)

