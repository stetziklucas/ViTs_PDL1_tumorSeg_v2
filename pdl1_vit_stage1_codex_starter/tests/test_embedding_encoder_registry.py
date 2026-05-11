from __future__ import annotations

import pytest

from encoder_backends.registry import resolve_encoder_spec


def test_registry_loads_current_timm() -> None:
    cfg = {"embedding_encoder": {"selected": "current_timm", "registry": {"current_timm": {"display_name": "Current ViT baseline", "backend": "timm", "model_name": "vit_base_patch16_224", "frozen": True}}}}
    spec = resolve_encoder_spec(cfg)
    assert spec.encoder_id == "current_timm"
    assert spec.backend == "timm"


def test_legacy_vit_fallback_synthesizes_current_timm() -> None:
    spec = resolve_encoder_spec({"vit": {"backend": "timm", "model_name": "vit_base_patch16_224", "frozen": True}})
    assert spec.encoder_id == "current_timm"


def test_cli_override_selects_encoder() -> None:
    cfg = {"embedding_encoder": {"selected": "current_timm", "registry": {"current_timm": {"backend": "timm", "model_name": "vit_base_patch16_224", "frozen": True}, "alt": {"backend": "timm", "model_name": "vit_small_patch16_224", "frozen": True}}}}
    spec = resolve_encoder_spec(cfg, "alt")
    assert spec.encoder_id == "alt"


def test_unknown_encoder_lists_available_ids() -> None:
    with pytest.raises(ValueError, match="Available encoder IDs"):
        resolve_encoder_spec({"embedding_encoder": {"registry": {"current_timm": {"backend": "timm", "frozen": True}}}}, "missing")


def test_frozen_false_raises() -> None:
    with pytest.raises(ValueError, match="frozen"):
        resolve_encoder_spec({"embedding_encoder": {"registry": {"current_timm": {"backend": "timm", "model_name": "vit_base_patch16_224", "frozen": False}}}})


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        resolve_encoder_spec({"embedding_encoder": {"registry": {"current_timm": {"backend": "unknown", "model_name": "vit_base_patch16_224", "frozen": True}}}})


def test_current_timm_metadata_has_backend_and_model_name() -> None:
    spec = resolve_encoder_spec({"embedding_encoder": {"registry": {"current_timm": {"backend": "timm", "model_name": "vit_base_patch16_224", "frozen": True}}}})
    assert spec.backend == "timm"
    assert spec.model_name == "vit_base_patch16_224"
