"""Shared supervision-audit helpers for Stage 1 development reporting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


_LOW_SUPPORT_TILE_THRESHOLD = 5
_HIGH_IGNORE_SHARE_THRESHOLD = 0.40


def _load_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload)!r}")
    return payload


def _load_mask(path: Path) -> np.ndarray:
    prior_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        arr = np.asarray(Image.open(path))
    finally:
        Image.MAX_IMAGE_PIXELS = prior_limit
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got shape={arr.shape}")
    return arr.astype(np.uint8)


def _count_polygons_by_class(annotation_meta: dict[str, Any]) -> dict[str, int]:
    polygons = annotation_meta.get("polygons", [])
    if not isinstance(polygons, list):
        return {}
    counts: Counter[str] = Counter()
    for polygon in polygons:
        if not isinstance(polygon, dict):
            continue
        class_name = str(polygon.get("class_name", "")).strip()
        if class_name:
            counts[class_name] += 1
    return dict(sorted(counts.items()))


def _count_pixels_by_class(scribble_labels: np.ndarray, label_encoding: dict[str, int]) -> dict[str, int]:
    return {
        class_name: int(np.count_nonzero(scribble_labels == int(class_code)))
        for class_name, class_code in label_encoding.items()
    }


def _counter_from_series(values: pd.Series) -> dict[str, int]:
    if values.empty:
        return {}
    return {str(k): int(v) for k, v in values.value_counts(dropna=False).to_dict().items()}


def _selection_source_counts(tile_labels: pd.DataFrame, tile_manifest_path: Path | None) -> dict[str, int]:
    if "selection_source" in tile_labels.columns:
        return _counter_from_series(tile_labels["selection_source"])
    if tile_manifest_path is None or not tile_manifest_path.exists():
        return {}
    manifest = pd.read_csv(tile_manifest_path)
    if manifest.empty or "selection_source" not in manifest.columns:
        return {}
    if "tile_id" in tile_labels.columns and "tile_id" in manifest.columns:
        subset = manifest.loc[manifest["tile_id"].isin(tile_labels["tile_id"])].copy()
        return _counter_from_series(subset["selection_source"])
    return _counter_from_series(manifest["selection_source"])


def audit_supervision(
    *,
    image_id: str,
    annotation_meta_path: Path,
    scribble_labels_path: Path,
    tile_labels_path: Path,
    label_encoding: dict[str, int],
    tile_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Compute deterministic class-aware supervision diagnostics for one image."""
    annotation_meta = _load_json_dict(annotation_meta_path)
    scribble_labels = _load_mask(scribble_labels_path)
    tile_labels = pd.read_csv(tile_labels_path)

    if tile_labels.empty:
        raise ValueError(f"Tile labels are empty: {tile_labels_path}")

    polygon_counts = _count_polygons_by_class(annotation_meta)
    annotated_pixel_counts = _count_pixels_by_class(scribble_labels, label_encoding)

    label_counts = _counter_from_series(tile_labels["label"]) if "label" in tile_labels.columns else {}
    label_reason_counts = _counter_from_series(tile_labels["label_reason"]) if "label_reason" in tile_labels.columns else {}
    selection_counts = _selection_source_counts(tile_labels, tile_manifest_path)

    accepted_tile_count = int(len(tile_labels.index))
    ignored_tile_count = int(label_counts.get("Ignore", 0))
    usable_positive_tiles = int(label_counts.get("Positive_Context", 0))
    usable_negative_tiles = int(label_counts.get("Negative_Context", 0))
    usable_tile_count = int(usable_positive_tiles + usable_negative_tiles)
    ignored_share = float(ignored_tile_count / accepted_tile_count) if accepted_tile_count else 0.0

    ignored_reason_counts = {
        reason: count
        for reason, count in label_reason_counts.items()
        if reason in {"no_supervision", "mixed_or_ambiguous"}
    }
    if not ignored_reason_counts:
        ignored_reason_counts = {
            reason: int(count)
            for reason, count in label_reason_counts.items()
            if reason.startswith("mixed_") or reason == "no_supervision"
        }

    warnings: list[str] = []
    if usable_positive_tiles < _LOW_SUPPORT_TILE_THRESHOLD:
        warnings.append("very low usable positive tile support")
    if usable_negative_tiles < _LOW_SUPPORT_TILE_THRESHOLD:
        warnings.append("very low usable negative tile support")
    if ignored_share >= _HIGH_IGNORE_SHARE_THRESHOLD:
        warnings.append("high ignored-tile share")
    if int(annotated_pixel_counts.get("Negative_Tumor", 0)) + int(annotated_pixel_counts.get("NonTumor", 0)) == 0:
        warnings.append("no explicit negative supervision")
    if int(annotated_pixel_counts.get("Positive_Tumor", 0)) == 0:
        warnings.append("no explicit positive supervision")

    return {
        "image_id": image_id,
        "polygon_counts": polygon_counts,
        "annotated_pixel_counts": annotated_pixel_counts,
        "accepted_tile_count": accepted_tile_count,
        "usable_tile_count": usable_tile_count,
        "ignored_tile_count": ignored_tile_count,
        "ignored_tile_share": float(ignored_share),
        "tile_label_counts": label_counts,
        "tile_label_reason_counts": label_reason_counts,
        "ignored_tile_reasons": ignored_reason_counts,
        "selection_source_counts": selection_counts,
        "warnings": warnings,
    }
