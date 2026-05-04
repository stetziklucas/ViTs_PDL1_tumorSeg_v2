"""Train the Stage 1 pixel classifier from scribble-derived pixel samples."""

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
from sklearn.ensemble import RandomForestClassifier
from skimage.color import rgb2hed

DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svs")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for pixel-classifier training."""
    parser = argparse.ArgumentParser(description="Train Stage 1 random-forest pixel classifier.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=False, help="Image identifier to process.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing source images.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Directory containing roi_masks/ and scribbles/.",
    )
    parser.add_argument("--tiles-dir", type=Path, default=Path("outputs/tiles"), help="Directory containing tile_manifest artifacts.")
    parser.add_argument(
        "--tile-probabilities",
        type=Path,
        default=Path("outputs/maps/tile_probabilities.csv"),
        help="CSV with tile-level probabilities from train_tile_head.py.",
    )
    parser.add_argument(
        "--cohort-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with shared-training rows: alias,image_id,tiles_dir,tile_probabilities. "
            "When provided, one shared model is trained across all listed images."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/pixel_classifier"),
        help="Output directory for pixel model + feature spec.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=75000,
        help="Maximum sampled pixels per class (positive and pooled negatives) for single-image mode.",
    )
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed.")
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


def compute_feature_maps(image_rgb: np.ndarray, tile_prior: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    """Compute pixel feature maps required for Stage 1."""
    rgb = image_rgb.astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    hed = rgb2hed(np.clip(rgb, 1e-6, 1.0)).astype(np.float32)

    g_sigma1 = gaussian_filter(gray, sigma=1.0).astype(np.float32)
    g_sigma3 = gaussian_filter(gray, sigma=3.0).astype(np.float32)

    mean = uniform_filter(gray, size=9)
    mean_sq = uniform_filter(gray * gray, size=9)
    local_var = np.clip(mean_sq - mean * mean, a_min=0.0, a_max=None).astype(np.float32)

    feature_names = [
        "rgb_r",
        "rgb_g",
        "rgb_b",
        "hed_h",
        "hed_e",
        "hed_d",
        "gaussian_sigma1",
        "gaussian_sigma3",
        "local_variance_9",
        "tile_prior",
    ]
    feature_maps: dict[str, np.ndarray] = {
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
    return feature_names, feature_maps


def sample_indices(indices: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample without replacement up to max_count rows."""
    if len(indices) <= max_count:
        return indices
    keep = rng.choice(len(indices), size=max_count, replace=False)
    return indices[keep]


def _resolve_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cohort_file is None:
        if not args.image_id:
            raise ValueError("Single-image mode requires --image-id.")
        return [
            {
                "alias": args.image_id,
                "image_id": args.image_id,
                "tiles_dir": args.tiles_dir,
                "tile_probabilities": args.tile_probabilities,
            }
        ]

    cohort_df = pd.read_csv(args.cohort_file)
    required = {"alias", "image_id", "tiles_dir", "tile_probabilities"}
    missing = sorted(required - set(cohort_df.columns))
    if missing:
        raise ValueError(f"Cohort file missing required columns: {missing}")

    return [
        {
            "alias": str(row["alias"]),
            "image_id": str(row["image_id"]),
            "tiles_dir": Path(str(row["tiles_dir"])),
            "tile_probabilities": Path(str(row["tile_probabilities"])),
        }
        for row in cohort_df.to_dict(orient="records")
    ]


def _build_case_samples(
    *,
    case: dict[str, Any],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    max_samples_per_image_per_class: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    image_id = str(case["image_id"])
    tiles_dir = Path(case["tiles_dir"])
    tile_probabilities = Path(case["tile_probabilities"])

    tile_manifest_meta_path = tiles_dir / "tile_manifest_meta.json"
    scribble_path = args.annotations_dir / "scribbles" / f"{image_id}_scribble_labels.png"

    for required in (tile_manifest_meta_path, scribble_path, tile_probabilities):
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")

    with tile_manifest_meta_path.open("r", encoding="utf-8") as handle:
        tile_meta = json.load(handle)

    svs_max_dimension = int(cfg.get("tiling", {}).get("svs_max_dimension", 4096))
    image_path = resolve_image_path(image_id, args.raw_dir)

    expected_dims = None
    dims = tile_meta.get("svs_level_dimensions_wh")
    if isinstance(dims, list) and len(dims) == 2:
        expected_dims = (int(dims[0]), int(dims[1]))

    image_rgb, image_meta = load_source_image_rgb(
        image_path,
        svs_max_dimension=svs_max_dimension,
        expected_svs_dimensions_wh=expected_dims,
    )
    target_shape_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))

    scribble_raw = load_trusted_large_png_mask(scribble_path, args.annotations_dir)
    scribble, scribble_transform = reconcile_mask_to_target(scribble_raw, target_shape_hw, image_id=image_id)

    tile_probs_df = pd.read_csv(tile_probabilities)
    tile_probs_df = tile_probs_df.loc[tile_probs_df["image_id"].astype(str) == image_id].copy()
    if tile_probs_df.empty:
        raise ValueError(f"No tile probabilities for image_id={image_id} in {tile_probabilities}")

    tile_prior = build_tile_prior_map(tile_probs_df, target_shape_hw)
    feature_names, feature_maps = compute_feature_maps(image_rgb, tile_prior)

    labels_cfg = cfg.get("classes", {}).get("label_encoding", {})
    positive_code = int(labels_cfg.get("Positive_Tumor", 1))
    negative_tumor_code = int(labels_cfg.get("Negative_Tumor", 2))
    nontumor_code = int(labels_cfg.get("NonTumor", 3))

    pos_yx = np.argwhere(scribble == positive_code)
    neg_yx = np.argwhere((scribble == negative_tumor_code) | (scribble == nontumor_code))

    if len(pos_yx) == 0 or len(neg_yx) == 0:
        raise RuntimeError(
            "Need both positive and negative scribble pixels to train pixel classifier. "
            f"image_id={image_id} found positive={len(pos_yx)} negative={len(neg_yx)}"
        )

    pos_yx = sample_indices(pos_yx, max_samples_per_image_per_class, rng)
    neg_yx = sample_indices(neg_yx, max_samples_per_image_per_class, rng)

    yx = np.concatenate([pos_yx, neg_yx], axis=0)
    y = np.concatenate([np.ones(len(pos_yx), dtype=np.int64), np.zeros(len(neg_yx), dtype=np.int64)], axis=0)
    x_cols = [feature_maps[name][yx[:, 0], yx[:, 1]] for name in feature_names]
    x = np.stack(x_cols, axis=1).astype(np.float32)

    return {
        "alias": str(case["alias"]),
        "image_id": image_id,
        "x": x,
        "y": y,
        "feature_names": feature_names,
        "n_positive_samples": int(len(pos_yx)),
        "n_negative_samples": int(len(neg_yx)),
        "image_path": str(image_path.as_posix()),
        "image_loader_backend": image_meta["backend"],
        "svs_level": int(image_meta["svs_level"]),
        "svs_level_downsample": float(image_meta["svs_level_downsample"]),
        "svs_level_dimensions_wh": image_meta["svs_level_dimensions_wh"],
        "svs_level0_dimensions_wh": image_meta.get("svs_level0_dimensions_wh"),
        "working_image_shape_hw": [int(target_shape_hw[0]), int(target_shape_hw[1])],
        "annotation_mask_transform": scribble_transform,
        "tile_prior_source": str(tile_probabilities.as_posix()),
    }


def main() -> None:
    """Run Stage 1 pixel-classifier training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    pcfg = cfg.get("pixel_classifier", {})
    equalize_image_class_sampling = bool(pcfg.get("equalize_image_class_sampling", True))
    max_samples_per_image_per_class = int(pcfg.get("max_samples_per_image_per_class", args.max_samples_per_class))

    rng = np.random.default_rng(args.random_seed)
    cases = _resolve_cases(args)

    if not equalize_image_class_sampling and len(cases) == 1:
        max_samples_per_image_per_class = int(args.max_samples_per_class)

    sample_sets = [
        _build_case_samples(
            case=case,
            cfg=cfg,
            args=args,
            max_samples_per_image_per_class=max_samples_per_image_per_class,
            rng=rng,
        )
        for case in cases
    ]

    feature_name_sets = {tuple(sample["feature_names"]) for sample in sample_sets}
    if len(feature_name_sets) != 1:
        raise ValueError("Feature-name mismatch across cohort images.")
    feature_names = list(next(iter(feature_name_sets)))

    x = np.concatenate([sample["x"] for sample in sample_sets], axis=0)
    y = np.concatenate([sample["y"] for sample in sample_sets], axis=0)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=args.random_seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(x, y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "pixel_model.pkl"
    feature_spec_path = args.output_dir / "pixel_feature_spec.json"

    with model_path.open("wb") as handle:
        pickle.dump(rf, handle)

    labels_cfg = cfg.get("classes", {}).get("label_encoding", {})
    per_image_counts = [
        {
            "alias": sample["alias"],
            "image_id": sample["image_id"],
            "n_positive_samples": sample["n_positive_samples"],
            "n_negative_samples": sample["n_negative_samples"],
            "n_total_samples": sample["n_positive_samples"] + sample["n_negative_samples"],
            "tile_prior_source": sample["tile_prior_source"],
        }
        for sample in sample_sets
    ]

    spec = {
        "image_id": args.image_id if args.cohort_file is None else None,
        "model_scope": "single_image_model" if args.cohort_file is None else "shared_project_model",
        "feature_names": feature_names,
        "class_map": {"negative": 0, "positive": 1},
        "scribble_codes": {
            "positive": int(labels_cfg.get("Positive_Tumor", 1)),
            "negative_tumor": int(labels_cfg.get("Negative_Tumor", 2)),
            "nontumor": int(labels_cfg.get("NonTumor", 3)),
            "ignored": [int(labels_cfg.get("Unlabeled", 0)), int(labels_cfg.get("Ignore", 4))],
        },
        "n_images": int(len(sample_sets)),
        "n_samples": int(len(y)),
        "n_positive_samples": int((y == 1).sum()),
        "n_negative_samples": int((y == 0).sum()),
        "max_samples_per_image_per_class": int(max_samples_per_image_per_class),
        "equalize_image_class_sampling": bool(equalize_image_class_sampling),
        "per_image_sample_counts": per_image_counts,
        "image_loader_backend": sample_sets[0]["image_loader_backend"],
        "svs_level": sample_sets[0]["svs_level"],
        "svs_level_downsample": sample_sets[0]["svs_level_downsample"],
        "svs_level_dimensions_wh": sample_sets[0]["svs_level_dimensions_wh"],
        "svs_level0_dimensions_wh": sample_sets[0]["svs_level0_dimensions_wh"],
        "working_image_shape_hw": sample_sets[0]["working_image_shape_hw"],
        "annotation_mask_transform": sample_sets[0]["annotation_mask_transform"],
        "notes": [
            "Training occurs in selected working image space (SVS level when applicable).",
            "Ignore and Unlabeled classes are excluded from supervised pixel training.",
        ],
    }
    with feature_spec_path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logging.info("Wrote pixel model: %s", model_path)
    logging.info("Wrote feature spec: %s", feature_spec_path)
    logging.info("Training samples | positive=%d negative=%d total=%d", int((y == 1).sum()), int((y == 0).sum()), len(y))


if __name__ == "__main__":
    main()
