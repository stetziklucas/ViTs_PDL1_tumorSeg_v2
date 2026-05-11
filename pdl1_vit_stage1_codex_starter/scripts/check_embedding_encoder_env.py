"""Preflight checks for embedding encoder dependencies and optional model loading."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from encoder_backends import resolve_encoder_spec
from scripts.embed_vit import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check embedding encoder environment readiness.")
    parser.add_argument("--embedding-encoder", "--encoder", dest="embedding_encoder", default=None)
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"))
    parser.add_argument("--try-load", action="store_true", help="Attempt HF processor/model loading (may require network/auth).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    spec = resolve_encoder_spec(cfg, args.embedding_encoder)
    print(f"encoder_id={spec.encoder_id}")
    print(f"backend={spec.backend}")
    print(f"model_name={spec.model_name}")

    if spec.backend == "timm":
        import timm

        print(f"timm_version={getattr(timm, '__version__', 'unknown')}")
        print("status=ok")
        return

    import transformers

    print(f"transformers_version={transformers.__version__}")
    has_onnx = importlib.util.find_spec("transformers.onnx") is not None
    print(f"transformers_onnx_available={has_onnx}")

    import huggingface_hub

    print(f"huggingface_hub_version={getattr(huggingface_hub, '__version__', 'unknown')}")
    whoami_fn = getattr(huggingface_hub, "whoami", None)
    if callable(whoami_fn):
        try:
            who = whoami_fn()
            if isinstance(who, dict):
                print(f"huggingface_user={who.get('name', 'unknown')}")
            else:
                print("huggingface_user=available")
        except Exception:
            print("huggingface_user=unavailable")

    if not has_onnx:
        raise SystemExit("Hibou-B remote code currently requires transformers.onnx. Install transformers>=4.53.3,<5.")

    if args.try_load:
        from transformers import AutoImageProcessor, AutoModel

        print("try_load=starting")
        AutoImageProcessor.from_pretrained(spec.model_name, trust_remote_code=True)
        AutoModel.from_pretrained(spec.model_name, trust_remote_code=True)
        print("try_load=ok")

    print("status=ok")


if __name__ == "__main__":
    main()
