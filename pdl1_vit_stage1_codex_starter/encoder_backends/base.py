"""Base abstractions for embedding encoder backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EncoderSpec:
    """Normalized encoder configuration used by backend adapters."""

    encoder_id: str
    display_name: str
    backend: str
    model_name: str
    pretrained: bool = True
    frozen: bool = True
    device: str = "cpu"
    batch_size: int = 32
    input_size: int | None = None
    embedding_format: str = "npy"
    trust_remote_code: bool = False
    requires_hf_auth: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized_config(self) -> dict[str, Any]:
        """Return stable dictionary for cache signatures/metadata."""
        return asdict(self)


class TileEmbeddingEncoder(Protocol):
    """Protocol implemented by embedding backends."""

    @property
    def batch_size(self) -> int:
        """Preferred embedding batch size."""

    def metadata(self) -> dict[str, Any]:
        """Return backend metadata suitable for cache meta json."""

    def encode_tiles(self, tiles: list[Image.Image]) -> np.ndarray:
        """Encode RGB PIL tiles into float32 numpy embeddings [n_tiles, embedding_dim]."""

