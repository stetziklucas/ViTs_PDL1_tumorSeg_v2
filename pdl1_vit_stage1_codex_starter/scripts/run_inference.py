"""Run Stage 1 fused inference with tile-prior + pixel classifier fusion."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.color import rgb2hed

DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svs")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for fused inference."""
    parser = argparse.ArgumentParser(description="Run Stage 1 fused inference.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=True, help="Image identifier to process.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing source images.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Directory containing ROI masks.",
    )
    parser.add_argument("--tiles-dir", type=Path, default=Path("outputs/tiles"), help="Directory with tile manifest artifacts.")
    parser.add_argument(
        "--tile-probabilities",
        type=Path,
        default=Path("outputs/maps/tile_probabilities.csv"),
        help="CSV with tile-level probabilities from train_tile_head.py.",
    )
    parser.add_argument(
        "--pixel-model",
        type=Path,
        default=Path("models/pixel_classifier/pixel_model.pkl"),
        help="Path to trained pixel model pickle.",
    )
    parser.add_argument(
        "--pixel-feature-spec",
        type=Path,
        default=Path("models/pixel_classifier/pixel_feature_spec.json"),
        help="Path to pixel feature specification JSON.",
    )
    parser.add_argument("--maps-dir", type=Path, default=Path("outputs/maps"), help="Output directory for probability maps.")
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/masks"), help="Output directory for fused masks.")
    parser.add_argument("--overlays-dir", type=Path, default=Path("outputs/overlays"), help="Output directory for overlays.")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"), help="Output directory for metrics JSON.")
    parser.add_argument(
        "--tile-threshold",
        type=float,
        default=None,
        help="Optional override for tile prior threshold; defaults to tile_head.positive_threshold in config.",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=None,
        help="Optional override for pixel probability threshold; defaults to pixel_model.positive_threshold in config.",
    )
    parser.add_argument("--chunk-rows", type=int, default=256, help="Rows per inference chunk.")
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


def resolve_image_path(image_id: str, input_dir: Path) -> Path:
    """Resolve image path by image_id in input directory."""
    candidates = sorted(p for p in input_dir.glob(f"{image_id}.*") if p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS)
    if not candidates:
        raise FileNotFoundError(f"No image found for image_id='{image_id}' in {input_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple images found for image_id='{image_id}': {candidates}")
    return candidates[0]


def select_svs_level(
    slide: Any,
    max_dimension: int,
    expected_dimensions_wh: tuple[int, int] | None = None,
) -> tuple[int, tuple[int, int], float]:
    """Select deterministic SVS level bounded by max_dimension."""
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


def load_source_image_rgb(
    path: Path,
    svs_max_dimension: int,
    expected_svs_dimensions_wh: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load source image into RGB array with SVS-aware level policy."""
    if path.suffix.lower() != ".svs":
        arr = np.asarray(Image.open(path).convert("RGB"))
        return arr, {
            "backend": "pil",
            "svs_level": 0,
            "svs_level_downsample": 1.0,
            "svs_level_dimensions_wh": [int(arr.shape[1]), int(arr.shape[0])],
            "svs_level0_dimensions_wh": [int(arr.shape[1]), int(arr.shape[0])],
            "coordinate_space": "loaded_image_level_pixels_xywh",
        }

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
        arr = np.asarray(region)
        return arr, {
            "backend": "openslide",
            "svs_level": int(level),
            "svs_level_downsample": float(downsample),
            "svs_level_dimensions_wh": [int(width), int(height)],
            "svs_level0_dimensions_wh": [int(slide.level_dimensions[0][0]), int(slide.level_dimensions[0][1])],
            "coordinate_space": "svs_level_pixels_xywh",
        }


def load_trusted_large_png_mask(path: Path, trusted_root: Path) -> np.ndarray:
    """Load a potentially large trusted repo-owned PNG mask with scoped PIL pixel-limit bypass."""
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    resolved_path = path.resolve()
    resolved_root = trusted_root.resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing large-mask trust bypass outside trusted root: {resolved_path}")
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


def compute_tissue_mask(rgb_image: np.ndarray, sat_threshold: float, value_threshold: float) -> np.ndarray:
    """Compute tissue mask via HSV-like saturation/value heuristics."""
    rgb = rgb_image[..., :3].astype(np.float32) / 255.0
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-8, None), 0.0)
    value = maxc
    tissue = (sat >= sat_threshold) & (value <= value_threshold)
    return tissue.astype(np.uint8)


def reconcile_mask_to_target(mask: np.ndarray, target_shape_hw: tuple[int, int], image_id: str) -> tuple[np.ndarray, str]:
    """Resize mask to target shape in a controlled way if needed."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask for image_id={image_id}, got shape={mask.shape}")
    if mask.shape == target_shape_hw:
        return mask.astype(np.uint8), "identity"

    src_h, src_w = mask.shape
    tgt_h, tgt_w = target_shape_hw
    scale_y = float(tgt_h) / float(src_h)
    scale_x = float(tgt_w) / float(src_w)
    if abs(scale_y - scale_x) > 0.01:
        raise ValueError(
            f"Refusing anisotropic mask resize for image_id={image_id}: source={mask.shape} target={target_shape_hw}"
        )

    resized = Image.fromarray(mask.astype(np.uint8), mode="L").resize((tgt_w, tgt_h), resample=Image.Resampling.NEAREST)
    out = np.asarray(resized).astype(np.uint8)
    return out, "nearest_uniform_scale"


def build_tile_prior_map(tile_df: pd.DataFrame, shape_hw: tuple[int, int]) -> np.ndarray:
    """Rasterize tile probabilities to pixel space using max-overlap fill."""
    h, w = shape_hw
    prior = np.zeros((h, w), dtype=np.float32)
    for row in tile_df.itertuples(index=False):
        x = int(row.tile_x)
        y = int(row.tile_y)
        tw = int(row.tile_w)
        th = int(row.tile_h)
        p = float(row.prob_positive)
        prior[y : y + th, x : x + tw] = np.maximum(prior[y : y + th, x : x + tw], p)
    return prior


def compute_feature_maps(image_rgb: np.ndarray, tile_prior: np.ndarray) -> dict[str, np.ndarray]:
    """Compute full-image feature maps used by the pixel classifier."""
    rgb = image_rgb.astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    hed = rgb2hed(np.clip(rgb, 1e-6, 1.0)).astype(np.float32)

    g_sigma1 = gaussian_filter(gray, sigma=1.0).astype(np.float32)
    g_sigma3 = gaussian_filter(gray, sigma=3.0).astype(np.float32)

    mean = uniform_filter(gray, size=9)
    mean_sq = uniform_filter(gray * gray, size=9)
    local_var = np.clip(mean_sq - mean * mean, a_min=0.0, a_max=None).astype(np.float32)

    return {
        "rgb_r": rgb[..., 0].astype(np.float32),
        "rgb_g": rgb[..., 1].astype(np.float32),
        "rgb_b": rgb[..., 2].astype(np.float32),
        "hed_h": hed[..., 0].astype(np.float32),
        "hed_e": hed[..., 1].astype(np.float32),
        "hed_d": hed[..., 2].astype(np.float32),
        "gaussian_sigma1": g_sigma1,
        "gaussian_sigma3": g_sigma3,
        "local_variance_9": local_var,
        "tile_prior": tile_prior.astype(np.float32),
    }


def render_overlay(image_rgb: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    """Render a simple red overlay for positive mask pixels."""
    base = image_rgb.astype(np.float32)
    overlay = base.copy()
    alpha = 0.45
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    m = positive_mask.astype(bool)
    overlay[m] = (1.0 - alpha) * overlay[m] + alpha * red
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    """Run Stage 1 fused inference."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    for required in (args.pixel_model, args.pixel_feature_spec, args.tile_probabilities):
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")

    tile_manifest_meta_path = args.tiles_dir / "tile_manifest_meta.json"
    tile_manifest_path = args.tiles_dir / "tile_manifest.csv"
    if not tile_manifest_meta_path.exists() or not tile_manifest_path.exists():
        raise FileNotFoundError(f"Missing tile artifacts in {args.tiles_dir}")

    with tile_manifest_meta_path.open("r", encoding="utf-8") as handle:
        tile_meta = json.load(handle)
    with args.pixel_feature_spec.open("r", encoding="utf-8") as handle:
        feature_spec = json.load(handle)

    image_path = resolve_image_path(args.image_id, args.raw_dir)

    svs_max_dimension = int(cfg.get("tiling", {}).get("svs_max_dimension", 4096))
    expected_dims = None
    dims = tile_meta.get("svs_level_dimensions_wh")
    if isinstance(dims, list) and len(dims) == 2:
        expected_dims = (int(dims[0]), int(dims[1]))

    image_rgb, image_meta = load_source_image_rgb(
        image_path,
        svs_max_dimension=svs_max_dimension,
        expected_svs_dimensions_wh=expected_dims,
    )
    shape_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))

    limitations: list[str] = []

    tissue_path = args.tiles_dir / f"{args.image_id}_tissue_mask.png"
    tissue_transform = "computed_from_image"
    if tissue_path.exists():
        tissue_raw = load_trusted_large_png_mask(tissue_path, args.tiles_dir)
        tissue_mask, tissue_transform = reconcile_mask_to_target((tissue_raw > 0).astype(np.uint8), shape_hw, args.image_id)
    else:
        sat_threshold = float(cfg.get("tiling", {}).get("tissue_sat_threshold", 0.08))
        value_threshold = float(cfg.get("tiling", {}).get("tissue_value_threshold", 0.95))
        tissue_mask = compute_tissue_mask(image_rgb, sat_threshold=sat_threshold, value_threshold=value_threshold)
        limitations.append("tissue_mask_fallback=computed_from_loaded_image")

    roi_path = args.annotations_dir / "roi_masks" / f"{args.image_id}_roi_mask.png"
    if roi_path.exists():
        roi_raw = load_trusted_large_png_mask(roi_path, args.annotations_dir)
        roi_mask, roi_transform = reconcile_mask_to_target((roi_raw > 0).astype(np.uint8), shape_hw, args.image_id)
    else:
        roi_mask = np.ones(shape_hw, dtype=np.uint8)
        roi_transform = "full_image_fallback"
        limitations.append("roi_mask_missing_fallback=full_image")

    tile_probs_df = pd.read_csv(args.tile_probabilities)
    tile_probs_df = tile_probs_df.loc[tile_probs_df["image_id"].astype(str) == args.image_id].copy()
    if tile_probs_df.empty:
        raise ValueError(f"No tile probabilities for image_id={args.image_id} in {args.tile_probabilities}")
    tile_prior = build_tile_prior_map(tile_probs_df, shape_hw)

    feature_maps = compute_feature_maps(image_rgb, tile_prior)
    feature_names = feature_spec.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("pixel_feature_spec.json missing feature_names list")
    missing = [name for name in feature_names if name not in feature_maps]
    if missing:
        raise ValueError(f"Model expects unsupported features: {missing}")

    with args.pixel_model.open("rb") as handle:
        model = pickle.load(handle)

    h, w = shape_hw
    pixel_prob = np.zeros((h, w), dtype=np.float32)
    chunk_rows = max(16, int(args.chunk_rows))

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        chunk_arrays = [feature_maps[name][y0:y1, :].reshape(-1) for name in feature_names]
        x_chunk = np.stack(chunk_arrays, axis=1).astype(np.float32)
        prob_chunk = model.predict_proba(x_chunk)[:, 1].astype(np.float32)
        pixel_prob[y0:y1, :] = prob_chunk.reshape(y1 - y0, w)

    tile_threshold = (
        float(args.tile_threshold)
        if args.tile_threshold is not None
        else float(cfg.get("tile_head", {}).get("positive_threshold", 0.60))
    )
    pixel_threshold = (
        float(args.pixel_threshold)
        if args.pixel_threshold is not None
        else float(cfg.get("pixel_model", {}).get("positive_threshold", 0.55))
    )

    fused = (
        (tissue_mask > 0)
        & (roi_mask > 0)
        & (tile_prior > tile_threshold)
        & (pixel_prob > pixel_threshold)
    )
    positive_mask = fused.astype(np.uint8)

    args.maps_dir.mkdir(parents=True, exist_ok=True)
    args.masks_dir.mkdir(parents=True, exist_ok=True)
    args.overlays_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    pixel_prob_map_path = args.maps_dir / "pixel_prob_map.png"
    positive_mask_path = args.masks_dir / str(cfg.get("fusion", {}).get("output_mask_filename", "positive_mask.png"))
    overlay_path = args.overlays_dir / "overlay.png"
    metrics_path = args.reports_dir / "metrics.json"

    Image.fromarray(np.round(pixel_prob * 255.0).astype(np.uint8), mode="L").save(pixel_prob_map_path)
    Image.fromarray((positive_mask * 255).astype(np.uint8), mode="L").save(positive_mask_path)
    Image.fromarray(render_overlay(image_rgb, positive_mask), mode="RGB").save(overlay_path)

    positive_area = int(positive_mask.sum())
    roi_area = int((roi_mask > 0).sum())
    metrics = {
        "image_id": args.image_id,
        "source_image_path": str(image_path.as_posix()),
        "working_image_shape_hw": [int(shape_hw[0]), int(shape_hw[1])],
        "original_image_shape_hw": (
            [int(tile_meta["svs_level0_dimensions_wh"][1]), int(tile_meta["svs_level0_dimensions_wh"][0])]
            if isinstance(tile_meta.get("svs_level0_dimensions_wh"), list) and len(tile_meta.get("svs_level0_dimensions_wh")) == 2
            else None
        ),
        "image_loader_backend": image_meta["backend"],
        "svs_level": int(image_meta["svs_level"]),
        "svs_level_downsample": float(image_meta["svs_level_downsample"]),
        "thresholds": {
            "tile_prior_gt": float(tile_threshold),
            "pixel_prob_gt": float(pixel_threshold),
        },
        "positive_area_px": positive_area,
        "roi_area_px": roi_area,
        "positive_fraction_of_roi": (float(positive_area) / float(roi_area)) if roi_area > 0 else 0.0,
        "tile_prior_source": str(args.tile_probabilities.as_posix()),
        "pixel_model_path": str(args.pixel_model.as_posix()),
        "pixel_feature_spec_path": str(args.pixel_feature_spec.as_posix()),
        "tissue_mask_source": str(tissue_path.as_posix()) if tissue_path.exists() else "computed",
        "tissue_mask_transform": tissue_transform,
        "roi_mask_path": str(roi_path.as_posix()) if roi_path.exists() else None,
        "roi_mask_transform": roi_transform,
        "working_coordinate_space": image_meta.get("coordinate_space", "unknown"),
        "output_space_note": "All masks/maps/overlay are exported in working image space (selected SVS level when applicable).",
        "limitations_or_fallbacks": limitations,
        "artifacts": {
            "positive_mask": str(positive_mask_path.as_posix()),
            "pixel_prob_map": str(pixel_prob_map_path.as_posix()),
            "overlay": str(overlay_path.as_posix()),
            "metrics": str(metrics_path.as_posix()),
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logging.info("Wrote pixel probability map: %s", pixel_prob_map_path)
    logging.info("Wrote fused positive mask: %s", positive_mask_path)
    logging.info("Wrote overlay: %s", overlay_path)
    logging.info("Wrote metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
