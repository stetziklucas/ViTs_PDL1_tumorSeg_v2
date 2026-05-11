"""Encoder registry and legacy config fallback helpers."""

from __future__ import annotations

from typing import Any

from .base import EncoderSpec


def _from_dict(encoder_id: str, cfg: dict[str, Any]) -> EncoderSpec:
    spec = EncoderSpec(
        encoder_id=encoder_id,
        display_name=str(cfg.get("display_name", encoder_id)),
        backend=str(cfg.get("backend", "timm")),
        model_name=str(cfg.get("model_name", "vit_base_patch16_224")),
        pretrained=bool(cfg.get("pretrained", True)),
        frozen=bool(cfg.get("frozen", True)),
        device=str(cfg.get("device", "cpu")),
        batch_size=int(cfg.get("batch_size", 32)),
        input_size=cfg.get("input_size"),
        embedding_format=str(cfg.get("embedding_format", "npy")),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        requires_hf_auth=bool(cfg.get("requires_hf_auth", False)),
        extra={**dict(cfg.get("extra", {})), "pooling": cfg.get("pooling", "auto")},
    )
    if not spec.frozen:
        raise ValueError(f"Stage 1 requires frozen encoders; got frozen=false for encoder_id='{encoder_id}'.")
    return spec


def resolve_encoder_spec(config: dict[str, Any], cli_encoder_id: str | None = None) -> EncoderSpec:
    embedding_cfg = config.get("embedding_encoder")
    if isinstance(embedding_cfg, dict) and isinstance(embedding_cfg.get("registry"), dict):
        registry = {k: _from_dict(k, v) for k, v in embedding_cfg["registry"].items()}
        selected = cli_encoder_id or str(embedding_cfg.get("selected", "current_timm"))
    else:
        vit_cfg = dict(config.get("vit", {}))
        vit_cfg.setdefault("display_name", "Current ViT baseline")
        vit_cfg.setdefault("backend", "timm")
        vit_cfg.setdefault("model_name", "vit_base_patch16_224")
        vit_cfg.setdefault("pretrained", True)
        registry = {"current_timm": _from_dict("current_timm", vit_cfg)}
        selected = cli_encoder_id or "current_timm"

    if selected not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown embedding encoder_id '{selected}'. Available encoder IDs: {available}")
    spec = registry[selected]
    if spec.backend not in {"timm", "hf_transformers"}:
        raise ValueError(f"Unknown embedding backend '{spec.backend}' for encoder_id '{spec.encoder_id}'.")
    return spec

