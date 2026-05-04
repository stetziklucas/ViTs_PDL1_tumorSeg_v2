"""Stage 1 tissue masking and deterministic tile extraction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svs")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Stage 1 tile extraction."""
    parser = argparse.ArgumentParser(description="Stage 1 tissue masking + deterministic tile extraction.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=True, help="Image identifier (stem) to process.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Input image directory.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Base annotation directory containing roi_masks/.",
    )
    parser.add_argument("--roi-mask", type=Path, default=None, help="Optional explicit ROI mask path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiles"), help="Tile output directory.")
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


def compute_tissue_mask(rgb_image: np.ndarray, sat_threshold: float, value_threshold: float) -> np.ndarray:
    """Compute a simple tissue mask via HSV saturation/value heuristics."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] < 3:
        raise ValueError(f"Expected RGB-like image, got shape={rgb_image.shape}")

    rgb = rgb_image[..., :3].astype(np.float32) / 255.0
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-8, None), 0.0)
    value = maxc

    tissue = (sat >= sat_threshold) & (value <= value_threshold)
    return tissue.astype(np.uint8)


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


def load_roi_mask(roi_mask_path: Path | None, annotations_dir: Path, image_id: str, shape_hw: tuple[int, int]) -> np.ndarray:
    """Load ROI mask if present, else default to full-image ROI."""
    if roi_mask_path is None:
        candidate = annotations_dir / "roi_masks" / f"{image_id}_roi_mask.png"
        roi_mask_path = candidate if candidate.exists() else None

    if roi_mask_path is None:
        logging.warning("ROI mask not found; using full image as ROI for image_id=%s", image_id)
        return np.ones(shape_hw, dtype=np.uint8)

    if not roi_mask_path.exists():
        raise FileNotFoundError(f"ROI mask path does not exist: {roi_mask_path}")

    roi_raw = load_trusted_large_png_mask(roi_mask_path, annotations_dir)
    roi, roi_transform = reconcile_mask_to_target((roi_raw > 0).astype(np.uint8), shape_hw, image_id=image_id)
    if roi_transform != "identity":
        logging.warning(
            "ROI mask geometry %s does not match loaded image shape %s. Resizing ROI mask with nearest-neighbor.",
            roi_raw.shape,
            shape_hw,
        )
    return roi


def iter_tile_origins(height: int, width: int, tile_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    """Generate deterministic row-major tile geometry in original image pixel space."""
    if height < tile_size or width < tile_size:
        raise ValueError(f"Image size {(height, width)} is smaller than tile_size={tile_size}")

    y_positions = list(range(0, height - tile_size + 1, stride))
    x_positions = list(range(0, width - tile_size + 1, stride))

    if y_positions[-1] != height - tile_size:
        y_positions.append(height - tile_size)
    if x_positions[-1] != width - tile_size:
        x_positions.append(width - tile_size)

    rows: list[tuple[int, int, int, int]] = []
    for tile_row, y in enumerate(y_positions):
        for tile_col, x in enumerate(x_positions):
            rows.append((tile_row, tile_col, x, y))
    return rows


def evaluate_tile_acceptance(
    *,
    tissue_fraction: float,
    roi_fraction: float,
    min_tissue_fraction: float,
    min_roi_fraction: float,
    allow_sparse_roi_seed_tiles: bool,
) -> tuple[bool, str]:
    """Evaluate deterministic tile acceptance with optional sparse ROI fallback."""
    if tissue_fraction < min_tissue_fraction:
        return False, "rejected_low_tissue"
    if roi_fraction >= min_roi_fraction:
        return True, "accepted_roi_threshold"
    if allow_sparse_roi_seed_tiles and roi_fraction > 0.0:
        return True, "accepted_sparse_roi_fallback"
    return False, "rejected_low_roi"


def main() -> None:
    """Run Stage 1 tissue mask + tile extraction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    tiling = cfg.get("tiling", {})
    tile_size = int(tiling.get("tile_size_px", 224))
    stride = int(tiling.get("tile_stride_px", 112))
    min_tissue_fraction = float(tiling.get("min_tissue_fraction", 0.30))
    min_roi_fraction = float(tiling.get("min_roi_fraction", 0.001))
    allow_sparse_roi_seed_tiles = bool(tiling.get("allow_sparse_roi_seed_tiles", False))
    sat_threshold = float(tiling.get("tissue_sat_threshold", 0.08))
    value_threshold = float(tiling.get("tissue_value_threshold", 0.95))
    svs_max_dimension = int(tiling.get("svs_max_dimension", 4096))

    image_path = resolve_image_path(args.image_id, args.input)
    roi_candidate = args.annotations_dir / "roi_masks" / f"{args.image_id}_roi_mask.png"
    expected_dims: tuple[int, int] | None = None
    if image_path.suffix.lower() == ".svs" and roi_candidate.exists():
        roi_probe = load_trusted_large_png_mask(roi_candidate, args.annotations_dir)
        expected_dims = (int(roi_probe.shape[1]), int(roi_probe.shape[0]))
        if expected_dims is not None and max(expected_dims) > svs_max_dimension:
            expected_dims = None

    image, image_meta = load_source_image_rgb(
        image_path,
        svs_max_dimension=svs_max_dimension,
        expected_svs_dimensions_wh=expected_dims,
    )
    height, width = image.shape[:2]

    tissue_mask = compute_tissue_mask(image, sat_threshold=sat_threshold, value_threshold=value_threshold)
    roi_mask = load_roi_mask(args.roi_mask, args.annotations_dir, args.image_id, (height, width))

    rows: list[dict[str, Any]] = []
    coords: list[tuple[int, int, int, int]] = []
    accepted_by_roi_threshold = 0
    accepted_by_sparse_fallback = 0
    tile_origins = iter_tile_origins(height=height, width=width, tile_size=tile_size, stride=stride)

    for tile_idx, (tile_row, tile_col, x, y) in enumerate(tile_origins):
        y2 = y + tile_size
        x2 = x + tile_size

        tissue_fraction = float(tissue_mask[y:y2, x:x2].mean())
        roi_fraction = float(roi_mask[y:y2, x:x2].mean())
        accept, accept_reason = evaluate_tile_acceptance(
            tissue_fraction=tissue_fraction,
            roi_fraction=roi_fraction,
            min_tissue_fraction=min_tissue_fraction,
            min_roi_fraction=min_roi_fraction,
            allow_sparse_roi_seed_tiles=allow_sparse_roi_seed_tiles,
        )
        if not accept:
            continue
        if accept_reason == "accepted_roi_threshold":
            accepted_by_roi_threshold += 1
        elif accept_reason == "accepted_sparse_roi_fallback":
            accepted_by_sparse_fallback += 1

        rows.append(
            {
                "image_id": args.image_id,
                "tile_id": f"{args.image_id}_r{tile_row:04d}_c{tile_col:04d}",
                "tile_index": len(rows),
                "tile_row": tile_row,
                "tile_col": tile_col,
                "tile_x": x,
                "tile_y": y,
                "tile_w": tile_size,
                "tile_h": tile_size,
                "tissue_fraction": tissue_fraction,
                "roi_fraction": roi_fraction,
                "source_image": str(image_path.as_posix()),
            }
        )
        coords.append((x, y, tile_size, tile_size))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile_manifest_path = args.output_dir / "tile_manifest.csv"
    coords_path = args.output_dir / "tile_coords.npy"
    tissue_mask_path = args.output_dir / f"{args.image_id}_tissue_mask.png"
    metadata_path = args.output_dir / "tile_manifest_meta.json"

    manifest_df = pd.DataFrame(rows)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(["tile_row", "tile_col"]).reset_index(drop=True)
        manifest_df["tile_index"] = np.arange(len(manifest_df), dtype=np.int64)

    manifest_df.to_csv(tile_manifest_path, index=False)
    np.save(coords_path, np.array(coords, dtype=np.int32))
    Image.fromarray((tissue_mask * 255).astype(np.uint8), mode="L").save(tissue_mask_path)

    meta = {
        "image_id": args.image_id,
        "image_path": str(image_path.as_posix()),
        "image_hw": [int(height), int(width)],
        "tile_size_px": tile_size,
        "tile_stride_px": stride,
        "min_tissue_fraction": min_tissue_fraction,
        "min_roi_fraction": min_roi_fraction,
        "allow_sparse_roi_seed_tiles": allow_sparse_roi_seed_tiles,
        "tissue_sat_threshold": sat_threshold,
        "tissue_value_threshold": value_threshold,
        "svs_max_dimension": svs_max_dimension,
        "image_loader_backend": image_meta["backend"],
        "svs_level": image_meta["svs_level"],
        "svs_level_downsample": image_meta["svs_level_downsample"],
        "svs_level_dimensions_wh": image_meta["svs_level_dimensions_wh"],
        "svs_level0_dimensions_wh": image_meta.get("svs_level0_dimensions_wh"),
        "tile_count": int(len(manifest_df)),
        "accepted_by_roi_threshold": int(accepted_by_roi_threshold),
        "accepted_by_sparse_roi_fallback": int(accepted_by_sparse_fallback),
        "coordinate_space": image_meta["coordinate_space"],
        "ordering": "row_major_by_tile_row_then_tile_col",
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logging.info("Wrote tile manifest: %s", tile_manifest_path)
    logging.info("Wrote tile coords: %s", coords_path)
    logging.info("Wrote tissue mask: %s", tissue_mask_path)
    logging.info(
        "Image loader=%s svs_level=%s downsample=%.3f shape_hw=%s",
        image_meta["backend"],
        image_meta["svs_level"],
        float(image_meta["svs_level_downsample"]),
        (height, width),
    )
    logging.info("Accepted tiles: %d", len(manifest_df))
    logging.info(
        "Acceptance breakdown | roi_threshold=%d sparse_roi_fallback=%d",
        accepted_by_roi_threshold,
        accepted_by_sparse_fallback,
    )


if __name__ == "__main__":
    main()
