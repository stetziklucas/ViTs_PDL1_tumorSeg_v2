"""Generate deterministic Stage 1 tile labels from annotation masks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

POSITIVE_CLASS = 1
NEGATIVE_TUMOR_CLASS = 2
NONTUMOR_CLASS = 3


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for tile label generation."""
    parser = argparse.ArgumentParser(description="Generate Stage 1 tile labels from scribble class masks.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=True, help="Image identifier to process.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Annotation artifact directory containing roi_masks/ and scribbles/.",
    )
    parser.add_argument("--tiles-dir", type=Path, default=Path("outputs/tiles"), help="Tile manifest directory.")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("outputs/embeddings"),
        help="Embedding output directory containing tile_manifest_with_embeddings_index.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiles"), help="Tile-label output directory.")
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


def load_mask(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a 2D uint8 mask with optional shape validation."""
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {path}, got shape={arr.shape}")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f"Mask shape mismatch for {path}: expected={expected_shape}, got={arr.shape}")
    return arr.astype(np.uint8)


def load_trusted_large_png_mask(path: Path, trusted_root: Path) -> np.ndarray:
    """Load a potentially large repo-owned annotation PNG mask with strict trust checks."""
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    resolved_path = path.resolve()
    resolved_root = trusted_root.resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing large-mask trust bypass outside annotations dir: {resolved_path}")
    if resolved_path.suffix.lower() != ".png":
        raise ValueError(f"Trusted large-mask loader only supports PNG: {resolved_path}")

    prior_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        arr = np.asarray(Image.open(resolved_path))
    finally:
        Image.MAX_IMAGE_PIXELS = prior_limit

    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {resolved_path}, got shape={arr.shape}")
    return arr.astype(np.uint8)


def load_repo_annotation_mask(
    path: Path,
    *,
    annotations_dir: Path,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a repo-owned annotation PNG mask with large-image safety bypass and optional shape check."""
    mask = load_trusted_large_png_mask(path, annotations_dir)
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"Mask shape mismatch for {path}: expected={expected_shape}, got={mask.shape}")
    return mask


def load_json_if_exists(path: Path) -> dict[str, Any]:
    """Load a JSON object if it exists, else return empty dict."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def compute_target_shape_from_tiles(tile_df: pd.DataFrame) -> tuple[int, int]:
    """Infer (height, width) coordinate space from tile extents."""
    width = int((tile_df["tile_x"] + tile_df["tile_w"]).max())
    height = int((tile_df["tile_y"] + tile_df["tile_h"]).max())
    return (height, width)


def _coerce_hw_pair(value: Any) -> tuple[int, int] | None:
    """Parse an [h, w] or [w, h] style metadata sequence into (h, w)."""
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def determine_authoritative_target_shape(
    *,
    tile_meta: dict[str, Any],
    embedding_meta: dict[str, Any],
    tile_df: pd.DataFrame,
    image_id: str,
) -> tuple[tuple[int, int], str]:
    """Resolve authoritative target image geometry for mask reconciliation."""
    tile_image_hw = _coerce_hw_pair(tile_meta.get("image_hw"))
    extents_hw = compute_target_shape_from_tiles(tile_df)
    if tile_image_hw is not None:
        if tile_image_hw[0] < extents_hw[0] or tile_image_hw[1] < extents_hw[1]:
            raise ValueError(
                "Tile-manifest metadata image_hw is smaller than accepted-tile extents for "
                f"image_id={image_id}: image_hw={tile_image_hw} extents_hw={extents_hw}. "
                "Re-run extract_tiles.py for this image."
            )
        return tile_image_hw, "tile_manifest_meta.image_hw"

    logging.warning(
        "tile_manifest_meta.json missing image_hw for image_id=%s; falling back to accepted-tile extents=%s",
        image_id,
        extents_hw,
    )

    emb_level_dims_wh = _coerce_hw_pair(embedding_meta.get("svs_level_dimensions_wh"))
    if emb_level_dims_wh is not None:
        emb_level_hw = (emb_level_dims_wh[1], emb_level_dims_wh[0])
        if emb_level_hw != extents_hw:
            logging.warning(
                "Embedding SVS level dims differ from accepted-tile extents for image_id=%s: "
                "embedding_level_hw=%s extents_hw=%s",
                image_id,
                emb_level_hw,
                extents_hw,
            )

    return extents_hw, "accepted_tile_extents_fallback"


def reconcile_annotation_mask_to_tile_space(
    annotation_mask: np.ndarray,
    target_shape_hw: tuple[int, int],
    *,
    tile_meta: dict[str, Any],
    embedding_meta: dict[str, Any],
    image_id: str,
) -> np.ndarray:
    """Reconcile annotation-mask geometry with tile coordinate space using nearest-neighbor."""
    source_shape_hw = tuple(int(v) for v in annotation_mask.shape)
    is_svs_run = tile_meta.get("coordinate_space") == "svs_level_pixels_xywh" or embedding_meta.get("svs_level", 0) != 0
    logging.info("Annotation-mask geometry | image_id=%s source_hw=%s target_hw=%s", image_id, source_shape_hw, target_shape_hw)

    meta_downsample = tile_meta.get("svs_level_downsample", embedding_meta.get("svs_level_downsample"))
    meta_level_dims = tile_meta.get("svs_level_dimensions_wh")
    meta_level0_dims = tile_meta.get("svs_level0_dimensions_wh")
    logging.info(
        "Mask-space metadata | image_id=%s tile_space=%s svs_level_dims_wh=%s svs_level0_dims_wh=%s svs_level_downsample=%s",
        image_id,
        tile_meta.get("coordinate_space", "unknown"),
        meta_level_dims,
        meta_level0_dims,
        meta_downsample,
    )

    if source_shape_hw == target_shape_hw:
        logging.info("Mask transform | image_id=%s transform=identity", image_id)
        return annotation_mask

    source_h, source_w = source_shape_hw
    target_h, target_w = target_shape_hw
    if source_h <= 0 or source_w <= 0 or target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid source/target geometry for image_id={image_id}: source={source_shape_hw} target={target_shape_hw}")

    sy = float(target_h) / float(source_h)
    sx = float(target_w) / float(source_w)
    anisotropy_tolerance = 0.01
    if abs(sx - sy) > anisotropy_tolerance:
        raise ValueError(
            "Unable to reconcile annotation mask to tile coordinate space safely due to anisotropic scale "
            f"for image_id={image_id}: source_hw={source_shape_hw} target_hw={target_shape_hw} "
            f"scale_y={sy:.6f} scale_x={sx:.6f}. "
            "Expected isotropic SVS downsample/resize. Re-check tile_manifest_meta.json image_hw and annotation exports."
        )
    if is_svs_run:
        level_downsample = meta_downsample
        if level_downsample is not None:
            expected_scale = 1.0 / float(level_downsample)
            if abs(sx - expected_scale) > anisotropy_tolerance or abs(sy - expected_scale) > anisotropy_tolerance:
                raise ValueError(
                    "SVS mask reconciliation scale mismatch for image_id="
                    f"{image_id}: expected~{expected_scale:.6f} from svs_level_downsample={float(level_downsample):.6f}, "
                    f"observed scale_y={sy:.6f} scale_x={sx:.6f}. "
                    "Confirm annotator exports are level-0 and tiling metadata matches extraction level."
                )

    transform_reason = "uniform_scale_from_mask_to_tile_space"
    if isinstance(meta_level0_dims, list) and len(meta_level0_dims) == 2:
        level0_wh = (int(meta_level0_dims[0]), int(meta_level0_dims[1]))
        if source_shape_hw == (level0_wh[1], level0_wh[0]):
            transform_reason = "level0_to_tile_space_resize"
    elif isinstance(meta_level_dims, list) and len(meta_level_dims) == 2:
        level_wh = (int(meta_level_dims[0]), int(meta_level_dims[1]))
        if target_shape_hw == (level_wh[1], level_wh[0]):
            transform_reason = "svs_level_to_tile_space_resize"

    logging.info(
        "Mask transform | image_id=%s transform=%s source_hw=%s target_hw=%s scale_y=%.6f scale_x=%.6f interpolation=nearest",
        image_id,
        transform_reason,
        source_shape_hw,
        target_shape_hw,
        sy,
        sx,
    )
    resized = Image.fromarray(annotation_mask, mode="L").resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    resolved = np.asarray(resized).astype(np.uint8)
    if resolved.shape != target_shape_hw:
        raise ValueError(
            f"Annotation-mask reconciliation failed for image_id={image_id}: expected={target_shape_hw} got={resolved.shape}"
        )
    return resolved


def classify_tile(
    tile_mask: np.ndarray,
    *,
    min_labeled_fraction: float,
    positive_min_fraction: float,
    negative_min_fraction: float,
    mixed_max_fraction: float,
    mixed_dominant_purity_min: float = 0.75,
    sparse_min_sample_weight: float = 0.10,
    mixed_dominant_weight_scale: float = 0.75,
) -> tuple[str, dict[str, float], str]:
    """Classify one tile into Positive_Context, Negative_Context, or Ignore."""
    tile_area = float(tile_mask.size)
    pos_fraction = float(np.mean(tile_mask == POSITIVE_CLASS))
    neg_tumor_fraction = float(np.mean(tile_mask == NEGATIVE_TUMOR_CLASS))
    non_tumor_fraction = float(np.mean(tile_mask == NONTUMOR_CLASS))
    ignore_fraction = float(np.mean(tile_mask == 0))

    negative_fraction = neg_tumor_fraction + non_tumor_fraction
    supervised_fraction = pos_fraction + negative_fraction
    purity = (max(pos_fraction, negative_fraction) / supervised_fraction) if supervised_fraction > 0.0 else 0.0
    positive_px = int(np.count_nonzero(tile_mask == POSITIVE_CLASS))
    negative_tumor_px = int(np.count_nonzero(tile_mask == NEGATIVE_TUMOR_CLASS))
    non_tumor_px = int(np.count_nonzero(tile_mask == NONTUMOR_CLASS))
    ignore_px = int(np.count_nonzero(tile_mask == 0))
    negative_px = negative_tumor_px + non_tumor_px
    supervised_px = positive_px + negative_px

    fractions = {
        "positive_fraction": pos_fraction,
        "negative_tumor_fraction": neg_tumor_fraction,
        "non_tumor_fraction": non_tumor_fraction,
        "nontumor_fraction": non_tumor_fraction,  # Backward-compatible alias.
        "ignore_fraction": ignore_fraction,
        "negative_fraction": negative_fraction,
        "supervised_fraction": supervised_fraction,
        "purity": purity,
        "positive_px": positive_px,
        "negative_tumor_px": negative_tumor_px,
        "non_tumor_px": non_tumor_px,
        "nontumor_px": non_tumor_px,  # Backward-compatible alias.
        "ignore_px": ignore_px,
        "negative_px": negative_px,
        "supervised_px": supervised_px,
        "tile_area_px": tile_area,
    }

    pos_ok = pos_fraction >= positive_min_fraction and negative_fraction <= mixed_max_fraction
    neg_ok = negative_fraction >= negative_min_fraction and pos_fraction <= mixed_max_fraction

    if pos_ok and not neg_ok:
        label, reason = "Positive_Context", "dominant_positive"
    elif neg_ok and not pos_ok:
        label, reason = "Negative_Context", "dominant_negative"
    elif supervised_fraction == 0.0:
        label, reason = "Ignore", "no_supervision"
    elif pos_fraction > 0.0 and negative_fraction == 0.0:
        label, reason = "Positive_Context", "sparse_positive_seed"
    elif negative_fraction > 0.0 and pos_fraction == 0.0:
        label, reason = "Negative_Context", "sparse_negative_seed"
    elif purity >= mixed_dominant_purity_min:
        if pos_fraction >= negative_fraction:
            label, reason = "Positive_Context", "mixed_dominant_positive"
        else:
            label, reason = "Negative_Context", "mixed_dominant_negative"
    else:
        label, reason = "Ignore", "mixed_or_ambiguous"

    coverage_weight = min(1.0, supervised_fraction / max(min_labeled_fraction, 1e-8))
    if label == "Ignore":
        sample_weight = 0.0
    else:
        sample_weight = max(sparse_min_sample_weight, coverage_weight)
        if reason.startswith("mixed_dominant_"):
            sample_weight *= mixed_dominant_weight_scale
    fractions["sample_weight"] = float(min(1.0, sample_weight))
    return label, fractions, reason


def main() -> None:
    """Run Stage 1 tile label generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    label_cfg = cfg.get("tile_labels", {})
    min_labeled_fraction = float(label_cfg.get("min_labeled_fraction", 0.05))
    positive_min_fraction = float(label_cfg.get("positive_min_fraction", 0.20))
    negative_min_fraction = float(label_cfg.get("negative_min_fraction", 0.20))
    mixed_max_fraction = float(label_cfg.get("mixed_max_fraction", 0.10))
    mixed_dominant_purity_min = float(label_cfg.get("mixed_dominant_purity_min", 0.75))
    sparse_min_sample_weight = float(label_cfg.get("sparse_min_sample_weight", 0.10))
    mixed_dominant_weight_scale = float(label_cfg.get("mixed_dominant_weight_scale", 0.75))

    tile_manifest_path = args.tiles_dir / "tile_manifest.csv"
    embedding_manifest_path = args.embeddings_dir / "tile_manifest_with_embeddings_index.csv"
    scribble_path = args.annotations_dir / "scribbles" / f"{args.image_id}_scribble_labels.png"
    roi_path = args.annotations_dir / "roi_masks" / f"{args.image_id}_roi_mask.png"

    if not tile_manifest_path.exists():
        raise FileNotFoundError(f"Tile manifest not found: {tile_manifest_path}")
    if not embedding_manifest_path.exists():
        raise FileNotFoundError(f"Embedding-linked tile manifest not found: {embedding_manifest_path}")

    tile_df = pd.read_csv(tile_manifest_path)
    emb_df = pd.read_csv(embedding_manifest_path)
    if tile_df.empty:
        raise ValueError("Tile manifest is empty; cannot derive labels.")
    if emb_df.empty:
        raise ValueError("Embedding index manifest is empty; cannot link tile labels.")

    tile_df = tile_df.loc[tile_df["image_id"] == args.image_id].copy()
    emb_df = emb_df.loc[emb_df["image_id"] == args.image_id].copy()
    if tile_df.empty:
        raise ValueError(f"No tile rows for image_id={args.image_id} in {tile_manifest_path}")
    if emb_df.empty:
        raise ValueError(f"No embedding rows for image_id={args.image_id} in {embedding_manifest_path}")

    merged = tile_df.merge(
        emb_df[["tile_id", "embedding_index"]],
        on="tile_id",
        how="left",
        validate="1:1",
    )
    if merged["embedding_index"].isna().any():
        missing = merged.loc[merged["embedding_index"].isna(), "tile_id"].head(5).tolist()
        raise ValueError(f"Missing embedding_index linkage for tile_ids={missing}")

    tile_meta = load_json_if_exists(args.tiles_dir / "tile_manifest_meta.json")
    embedding_meta = load_json_if_exists(args.embeddings_dir / "embeddings_cache_meta.json")
    target_shape_hw, target_shape_source = determine_authoritative_target_shape(
        tile_meta=tile_meta,
        embedding_meta=embedding_meta,
        tile_df=merged,
        image_id=args.image_id,
    )
    logging.info(
        "Authoritative target mask geometry | image_id=%s source=%s target_hw=%s",
        args.image_id,
        target_shape_source,
        target_shape_hw,
    )

    scribble_raw = load_repo_annotation_mask(scribble_path, annotations_dir=args.annotations_dir)
    scribble = reconcile_annotation_mask_to_tile_space(
        scribble_raw,
        target_shape_hw,
        tile_meta=tile_meta,
        embedding_meta=embedding_meta,
        image_id=args.image_id,
    )
    roi_raw = load_repo_annotation_mask(roi_path, annotations_dir=args.annotations_dir)
    roi = reconcile_annotation_mask_to_tile_space(
        roi_raw,
        target_shape_hw,
        tile_meta=tile_meta,
        embedding_meta=embedding_meta,
        image_id=args.image_id,
    )
    if roi.shape != scribble.shape:
        raise ValueError(
            f"ROI and scribble shapes must match after reconciliation for image_id={args.image_id}: "
            f"roi={roi.shape} scribble={scribble.shape}"
        )

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        x = int(row.tile_x)
        y = int(row.tile_y)
        w = int(row.tile_w)
        h = int(row.tile_h)
        tile_mask = scribble[y : y + h, x : x + w]
        if tile_mask.shape != (h, w):
            raise ValueError(
                f"Tile crop out of bounds for tile_id={row.tile_id}: expected={(h, w)} got={tile_mask.shape}"
            )

        label, fractions, reason = classify_tile(
            tile_mask,
            min_labeled_fraction=min_labeled_fraction,
            positive_min_fraction=positive_min_fraction,
            negative_min_fraction=negative_min_fraction,
            mixed_max_fraction=mixed_max_fraction,
            mixed_dominant_purity_min=mixed_dominant_purity_min,
            sparse_min_sample_weight=sparse_min_sample_weight,
            mixed_dominant_weight_scale=mixed_dominant_weight_scale,
        )

        rows.append(
            {
                "image_id": str(row.image_id),
                "tile_id": str(row.tile_id),
                "tile_index": int(row.tile_index),
                "embedding_index": int(row.embedding_index),
                "tile_row": int(row.tile_row),
                "tile_col": int(row.tile_col),
                "tile_x": x,
                "tile_y": y,
                "tile_w": w,
                "tile_h": h,
                "label": label,
                "label_reason": reason,
                **fractions,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["tile_row", "tile_col", "tile_id"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "tile_labels.csv"
    out_df.to_csv(out_path, index=False)

    positive_count = int((out_df["label"] == "Positive_Context").sum())
    negative_count = int((out_df["label"] == "Negative_Context").sum())
    ignored_count = int((out_df["label"] == "Ignore").sum())

    logging.info("Inspected artifacts:")
    logging.info("- ROI mask: %s", roi_path)
    logging.info("- Scribble mask: %s", scribble_path)
    logging.info("- Tile manifest: %s", tile_manifest_path)
    logging.info("- Embedding manifest: %s", embedding_manifest_path)
    logging.info("Wrote tile labels: %s", out_path)
    logging.info("Tile label counts | positive=%d negative=%d ignored=%d", positive_count, negative_count, ignored_count)


if __name__ == "__main__":
    main()
