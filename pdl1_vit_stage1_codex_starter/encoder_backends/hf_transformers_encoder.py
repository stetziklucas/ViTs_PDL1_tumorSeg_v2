"""Hugging Face Transformers frozen embedding backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import EncoderSpec, TileEmbeddingEncoder


class HfTransformersTileEmbeddingEncoder(TileEmbeddingEncoder):
    def __init__(self, spec: EncoderSpec) -> None:
        self.spec = spec
        self.device = torch.device(spec.device)
        self.pooling = str(spec.extra.get("pooling", "auto")).lower()
        try:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(spec.model_name, trust_remote_code=spec.trust_remote_code)
            self.model = AutoModel.from_pretrained(spec.model_name, trust_remote_code=spec.trust_remote_code)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face encoder '{spec.model_name}'. Hibou-B is gated and may require accepted Hugging Face access. "
                "Use huggingface-cli login or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN. trust_remote_code=True is required by the model card. "
                f"Original exception: {exc}"
            ) from exc
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)
        self.embedding_dim: int | None = None

    @property
    def batch_size(self) -> int:
        return int(self.spec.batch_size)

    def _extract(self, outputs: Any) -> torch.Tensor:
        pooling = self.pooling
        if pooling == 'pooler':
            t = getattr(outputs, 'pooler_output', None)
            if isinstance(t, torch.Tensor):
                return t
            raise ValueError('pooling=pooler requested but pooler_output was unavailable.')
        if pooling == 'cls':
            t = getattr(outputs, 'last_hidden_state', None)
            if isinstance(t, torch.Tensor):
                return t[:,0]
            raise ValueError('pooling=cls requested but last_hidden_state was unavailable.')
        if pooling == 'mean':
            t = getattr(outputs, 'last_hidden_state', None)
            if isinstance(t, torch.Tensor):
                return t.mean(dim=1)
            raise ValueError('pooling=mean requested but last_hidden_state was unavailable.')
        if pooling != 'auto':
            raise ValueError(f'Unsupported pooling mode: {pooling}')
        t = getattr(outputs, 'pooler_output', None)
        if isinstance(t, torch.Tensor):
            return t
        t = getattr(outputs, 'last_hidden_state', None)
        if isinstance(t, torch.Tensor):
            return t[:,0]
        if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], torch.Tensor):
            first = outputs[0]
            if first.ndim == 2:
                return first
            if first.ndim == 3:
                return first[:,0]
        raise ValueError('Could not extract embeddings from transformer outputs for pooling=auto.')

    def metadata(self) -> dict[str, Any]:
        md = {
            'encoder_id': self.spec.encoder_id,
            'encoder_backend': 'hf_transformers',
            'encoder_model_name': self.spec.model_name,
            'encoder_display_name': self.spec.display_name,
            'encoder_pretrained': self.spec.pretrained,
            'encoder_frozen': self.spec.frozen,
            'encoder_device': self.spec.device,
            'encoder_batch_size': self.spec.batch_size,
            'encoder_input_size': self.spec.input_size,
            'encoder_trust_remote_code': self.spec.trust_remote_code,
            'encoder_requires_hf_auth': self.spec.requires_hf_auth,
            'encoder_pooling': self.pooling,
            'encoder_weight_source': 'pretrained',
        }
        if self.embedding_dim is not None:
            md['embedding_dim'] = int(self.embedding_dim)
        return md

    def encode_tiles(self, tiles: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=tiles, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            emb = self._extract(outputs)
        out = emb.detach().to('cpu').numpy().astype(np.float32)
        if out.ndim != 2:
            raise ValueError(f'Expected [batch, dim] embeddings; got shape {out.shape}')
        self.embedding_dim = int(out.shape[1])
        return out
