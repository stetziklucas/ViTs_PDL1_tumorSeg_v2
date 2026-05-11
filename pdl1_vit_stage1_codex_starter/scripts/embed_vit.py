"""Stage 1 frozen ViT embedding generation with cache/skip behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from encoder_backends import HfTransformersTileEmbeddingEncoder, TimmTileEmbeddingEncoder, resolve_encoder_spec

import numpy as np
import pandas as pd
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for frozen ViT embedding generation."""
    parser = argparse.ArgumentParser(description="Stage 1 frozen ViT embedding generation.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=True, help="Image identifier to process.")
    parser.add_argument("--input", type=Path, default=Path("outputs/tiles"), help="Tile manifest/input directory.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw image directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/embeddings"), help="Embedding output directory.")
    parser.add_argument("--embedding-encoder", type=str, default=None, help="Optional encoder_id override.")
    parser.add_argument("--encoder", type=str, default=None, help="Alias for --embedding-encoder.")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config did not parse into a dictionary.")
    return config


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash for file content."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_cache_signature(
    image_hash: str,
    tile_manifest_hash: str,
    encoder_cfg: dict[str, Any],
    tile_size_px: int,
) -> str:
    """Create deterministic signature for embedding cache validity."""
    payload = {
        "image_sha256": image_hash,
        "tile_manifest_sha256": tile_manifest_hash,
        "encoder_id": encoder_cfg.get("encoder_id", "current_timm"),
        "encoder_backend": encoder_cfg.get("backend", "timm"),
        "encoder_model_name": encoder_cfg.get("model_name", "vit_base_patch16_224"),
        "encoder_pretrained": bool(encoder_cfg.get("pretrained", True)),
        "encoder_frozen": bool(encoder_cfg.get("frozen", True)),
        "encoder_input_size": encoder_cfg.get("input_size"),
        "encoder_pooling": encoder_cfg.get("extra", {}).get("pooling"),
        "encoder_trust_remote_code": bool(encoder_cfg.get("trust_remote_code", False)),
        "encoder_requires_hf_auth": bool(encoder_cfg.get("requires_hf_auth", False)),
        "tile_size_px": int(tile_size_px),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_image_path_for_manifest(manifest_df: pd.DataFrame, raw_dir: Path, image_id: str) -> Path:
    """Resolve source image path for a tile manifest."""
    if manifest_df.empty:
        raise ValueError("Tile manifest is empty; cannot generate embeddings.")
    if "source_image" in manifest_df.columns and manifest_df["source_image"].notna().all():
        source_path = Path(str(manifest_df["source_image"].iloc[0]))
        if source_path.exists():
            return source_path
    candidates = sorted(raw_dir.glob(f"{image_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"Could not resolve source image for image_id={image_id}")
    return candidates[0]


def select_svs_level(
    slide: Any,
    max_dimension: int,
    expected_dimensions_wh: tuple[int, int] | None = None,
) -> tuple[int, tuple[int, int], float]:
    """Select deterministic SVS read level bounded by max_dimension."""
    if expected_dimensions_wh is not None:
        for level_idx, dims in enumerate(slide.level_dimensions):
            if (int(dims[0]), int(dims[1])) == expected_dimensions_wh:
                level_dims = slide.level_dimensions[level_idx]
                downsample = float(slide.level_downsamples[level_idx])
                return level_idx, (int(level_dims[1]), int(level_dims[0])), downsample

    best_level = 0
    for level_idx, dims in enumerate(slide.level_dimensions):
        if max(dims) <= max_dimension:
            best_level = level_idx
            break
    level_dims = slide.level_dimensions[best_level]
    downsample = float(slide.level_downsamples[best_level])
    return best_level, (int(level_dims[1]), int(level_dims[0])), downsample


def load_source_image(
    path: Path,
    svs_max_dimension: int,
    expected_svs_dimensions_wh: tuple[int, int] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Load source image as PIL RGB image with SVS-aware level policy."""
    if path.suffix.lower() != ".svs":
        image = Image.open(path).convert("RGB")
        return image, {"backend": "pil", "svs_level": 0, "svs_level_downsample": 1.0}

    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "SVS input requires openslide-python + OpenSlide runtime. "
            "Install openslide-python and system OpenSlide libraries."
        ) from exc

    with openslide.OpenSlide(str(path)) as slide:
        level, (height, width), downsample = select_svs_level(
            slide,
            max_dimension=svs_max_dimension,
            expected_dimensions_wh=expected_svs_dimensions_wh,
        )
        region = slide.read_region((0, 0), level, (width, height)).convert("RGB")
        return region, {"backend": "openslide", "svs_level": int(level), "svs_level_downsample": float(downsample)}


def main() -> None:
    """Run Stage 1 frozen ViT embedding generation with cache/skip behavior."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    if args.embedding_encoder and args.encoder and args.embedding_encoder != args.encoder:
        raise ValueError("--embedding-encoder and --encoder conflict; provide only one encoder_id value.")
    cli_encoder = args.embedding_encoder or args.encoder
    spec = resolve_encoder_spec(cfg, cli_encoder)
    tiling_cfg = cfg.get("tiling", {})
    svs_max_dimension = int(tiling_cfg.get("svs_max_dimension", 4096))

    tile_manifest_path = args.input / "tile_manifest.csv"
    if not tile_manifest_path.exists():
        raise FileNotFoundError(f"Tile manifest not found: {tile_manifest_path}")

    manifest_df = pd.read_csv(tile_manifest_path)
    if manifest_df.empty:
        raise ValueError(f"Tile manifest has zero accepted tiles: {tile_manifest_path}")

    image_path = load_image_path_for_manifest(manifest_df, args.raw_dir, args.image_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / "embeddings.npy"
    index_manifest_path = args.output_dir / "tile_manifest_with_embeddings_index.csv"
    cache_meta_path = args.output_dir / "embeddings_cache_meta.json"

    image_hash = sha256_file(image_path)
    tile_manifest_hash = sha256_file(tile_manifest_path)
    tile_size_px = int(tiling_cfg.get("tile_size_px", 224))
    cache_signature = make_cache_signature(image_hash, tile_manifest_hash, spec.normalized_config(), tile_size_px)

    if cache_meta_path.exists() and embeddings_path.exists() and index_manifest_path.exists():
        with cache_meta_path.open("r", encoding="utf-8") as handle:
            cache_meta = json.load(handle)
        if cache_meta.get("cache_signature") == cache_signature:
            logging.info("Cache hit: embedding recomputation skipped.")
            logging.info("Embeddings: %s", embeddings_path)
            return

    require_pretrained = image_path.suffix.lower() == ".svs"
    if spec.backend == "timm":
        try:
            encoder = TimmTileEmbeddingEncoder(spec, require_pretrained=require_pretrained)
        except Exception as exc:
            if require_pretrained:
                raise RuntimeError(
                    "Pretrained timm weights are required for real .svs embedding runs. "
                    f"Original error: {exc}"
                ) from exc
            raise
    else:
        encoder = HfTransformersTileEmbeddingEncoder(spec)
    expected_dims: tuple[int, int] | None = None
    if image_path.suffix.lower() == ".svs":
        width = int((manifest_df["tile_x"] + manifest_df["tile_w"]).max())
        height = int((manifest_df["tile_y"] + manifest_df["tile_h"]).max())
        expected_dims = (width, height)
    image, image_meta = load_source_image(
        image_path,
        svs_max_dimension=svs_max_dimension,
        expected_svs_dimensions_wh=expected_dims,
    )

    outputs: list[np.ndarray] = []
    tile_batch: list[Image.Image] = []
    for row in manifest_df.itertuples(index=False):
        x = int(row.tile_x)
        y = int(row.tile_y)
        w = int(row.tile_w)
        h = int(row.tile_h)
        tile_batch.append(image.crop((x, y, x + w, y + h)))

        if len(tile_batch) == encoder.batch_size:
            outputs.append(encoder.encode_tiles(tile_batch))
            tile_batch.clear()

    if tile_batch:
        outputs.append(encoder.encode_tiles(tile_batch))

    embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
    if embeddings.shape[0] != len(manifest_df):
        raise RuntimeError(
            f"Embedding row count mismatch: expected={len(manifest_df)} got={embeddings.shape[0]}"
        )

    manifest_with_idx = manifest_df.copy()
    manifest_with_idx["embedding_index"] = np.arange(len(manifest_df), dtype=np.int64)

    np.save(embeddings_path, embeddings)
    manifest_with_idx.to_csv(index_manifest_path, index=False)

    cache_meta = {
        "image_id": args.image_id,
        "image_path": str(image_path.as_posix()),
        "tile_manifest_path": str(tile_manifest_path.as_posix()),
        "cache_signature": cache_signature,
        "image_sha256": image_hash,
        "tile_manifest_sha256": tile_manifest_hash,
        "encoder_id": spec.encoder_id,
        "encoder_display_name": spec.display_name,
        "encoder_backend": spec.backend,
        "encoder_model_name": spec.model_name,
        "encoder_pretrained": spec.pretrained,
        "encoder_frozen": spec.frozen,
        "encoder_input_size": spec.input_size,
        "encoder_batch_size": spec.batch_size,
        "encoder_config_normalized": spec.normalized_config(),
        "image_loader_backend": image_meta["backend"],
        "svs_level": image_meta["svs_level"],
        "svs_level_downsample": image_meta["svs_level_downsample"],
        "svs_max_dimension": svs_max_dimension,
        "tile_size_px": tile_size_px,
        "embedding_shape": [int(v) for v in embeddings.shape],
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "encoder_weight_source": encoder.metadata().get("encoder_weight_source", "pretrained"),
    }
    cache_meta.update(encoder.metadata())
    with cache_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(cache_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logging.info("Wrote embeddings: %s", embeddings_path)
    logging.info("Embedding shape: %s", embeddings.shape)
    logging.info(
        "Image loader=%s svs_level=%s downsample=%.3f",
        image_meta["backend"],
        image_meta["svs_level"],
        float(image_meta["svs_level_downsample"]),
    )
    logging.info("Wrote embedding index manifest: %s", index_manifest_path)


if __name__ == "__main__":
    main()
