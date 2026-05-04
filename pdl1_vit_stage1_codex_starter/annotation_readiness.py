"""Shared annotation readiness checks for Stage 1 polygon artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


POSITIVE_CLASS_NAMES = {"Positive_Tumor"}
NEGATIVE_CLASS_NAMES = {"Negative_Tumor", "NonTumor"}
NONTUMOR_ONLY_CLASS_NAMES = {"NonTumor", "Ignore"}


@dataclass(frozen=True)
class AnnotationReadinessResult:
    """Serializable readiness summary for one annotated image."""

    image_id: str
    artifact_exists: dict[str, bool]
    polygon_counts: dict[str, int]
    pixel_counts: dict[str, int]
    roi_positive_pixels: int
    status_code: str
    status_label: str
    summary_message: str
    next_action: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly dictionary payload."""
        return asdict(self)


class ReadinessArtifactsError(RuntimeError):
    """Raised when readiness artifacts are missing or malformed."""


def annotation_artifact_paths(annotations_dir: Path, image_id: str) -> dict[str, Path]:
    """Return canonical annotation artifact paths for one image id."""
    return {
        "annotation_meta": annotations_dir / f"{image_id}_annotation_meta.json",
        "roi_mask": annotations_dir / "roi_masks" / f"{image_id}_roi_mask.png",
        "scribble_labels": annotations_dir / "scribbles" / f"{image_id}_scribble_labels.png",
    }


def _status_text(status_code: str) -> tuple[str, str, str]:
    if status_code == "READY":
        return (
            "Ready",
            "Annotations are sufficient to continue with Stage 1 training/inference steps.",
            "You can continue to downstream Stage 1 CLI steps.",
        )
    if status_code == "NEEDS_POSITIVE":
        return (
            "Needs positive tumor supervision",
            "Negative supervision exists, but positive tumor supervision is missing.",
            "Add at least one Positive_Tumor annotation and save again.",
        )
    if status_code == "NEEDS_NEGATIVE":
        return (
            "Needs negative supervision",
            "Positive tumor supervision exists, but negative supervision is missing.",
            "Add Negative_Tumor or NonTumor annotation and save again.",
        )
    if status_code == "NO_TUMOR_ROI":
        return (
            "No tumor ROI",
            "Tumor ROI is empty or only NonTumor/Ignore supervision is present.",
            "Add Positive_Tumor and/or Negative_Tumor polygons to define tumor ROI, then save again.",
        )
    if status_code == "NO_USABLE_SUPERVISION":
        return (
            "No usable supervision",
            "No usable tumor supervision was found in the saved artifacts.",
            "Add tumor-related polygon annotations, then press 's' to save again.",
        )
    if status_code == "ERROR":
        return (
            "Artifact error",
            "Annotation artifacts are missing or malformed.",
            "Fix artifact paths/content, then save annotations again.",
        )
    raise ValueError(f"Unsupported status code: {status_code}")


def _count_polygons(metadata: dict[str, Any]) -> dict[str, int]:
    polygons = metadata.get("polygons", [])
    if not isinstance(polygons, list):
        raise ReadinessArtifactsError("Metadata polygons payload must be a list.")

    counts: dict[str, int] = {
        "Positive_Tumor": 0,
        "Negative_Tumor": 0,
        "NonTumor": 0,
        "Ignore": 0,
    }
    for polygon in polygons:
        class_name = str((polygon or {}).get("class_name", ""))
        if class_name in counts:
            counts[class_name] += 1
    return counts


def _count_pixels(scribble_mask: np.ndarray, label_encoding: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name, label in label_encoding.items():
        if class_name == "Unlabeled":
            continue
        counts[class_name] = int(np.count_nonzero(scribble_mask == int(label)))
    return counts


def compute_annotation_readiness(
    *,
    config: dict[str, Any],
    annotations_dir: Path,
    image_id: str,
) -> AnnotationReadinessResult:
    """Load artifacts and summarize annotation readiness for one image."""
    paths = annotation_artifact_paths(annotations_dir, image_id)
    exists_flags = {key: path.exists() for key, path in paths.items()}

    notes: list[str] = [
        "This is an annotation-readiness check only; it is not a model-quality or validation check.",
    ]

    allow_sparse = bool((config.get("tiling") or {}).get("allow_sparse_roi_seed_tiles", False))
    if allow_sparse:
        notes.append("Sparse annotations are accepted with current config (allow_sparse_roi_seed_tiles=true).")

    missing = [name for name, exists in exists_flags.items() if not exists]
    if missing:
        status_label, summary_message, next_action = _status_text("ERROR")
        notes.append(f"Missing artifacts: {', '.join(sorted(missing))}.")
        return AnnotationReadinessResult(
            image_id=image_id,
            artifact_exists=exists_flags,
            polygon_counts={"Positive_Tumor": 0, "Negative_Tumor": 0, "NonTumor": 0, "Ignore": 0},
            pixel_counts={"Positive_Tumor": 0, "Negative_Tumor": 0, "NonTumor": 0, "Ignore": 0},
            roi_positive_pixels=0,
            status_code="ERROR",
            status_label=status_label,
            summary_message=summary_message,
            next_action=next_action,
            notes=notes,
        )

    try:
        with paths["annotation_meta"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ReadinessArtifactsError("Metadata root must be a JSON object.")

        roi_mask = np.asarray(Image.open(paths["roi_mask"]))
        scribble_mask = np.asarray(Image.open(paths["scribble_labels"]))
        if roi_mask.ndim != 2 or scribble_mask.ndim != 2:
            raise ReadinessArtifactsError("ROI and scribble masks must be 2D grayscale images.")
        if roi_mask.shape != scribble_mask.shape:
            raise ReadinessArtifactsError("ROI and scribble mask shapes do not match.")

        label_encoding = (config.get("classes") or {}).get("label_encoding") or {}
        if not isinstance(label_encoding, dict) or "Positive_Tumor" not in label_encoding:
            raise ReadinessArtifactsError("Config classes.label_encoding is missing required labels.")

        polygon_counts = _count_polygons(metadata)
        pixel_counts = _count_pixels(scribble_mask, label_encoding)
        roi_positive_pixels = int(np.count_nonzero(roi_mask > 0))

        positive_present = pixel_counts.get("Positive_Tumor", 0) > 0
        negative_present = (pixel_counts.get("Negative_Tumor", 0) > 0) or (pixel_counts.get("NonTumor", 0) > 0)
        any_usable = positive_present or negative_present

        non_tumor_or_ignore_only = any(
            polygon_counts.get(name, 0) > 0 for name in NONTUMOR_ONLY_CLASS_NAMES
        ) and polygon_counts.get("Positive_Tumor", 0) == 0 and polygon_counts.get("Negative_Tumor", 0) == 0

        if not any_usable:
            status_code = "NO_USABLE_SUPERVISION"
        elif roi_positive_pixels == 0 or non_tumor_or_ignore_only:
            status_code = "NO_TUMOR_ROI"
        elif positive_present and negative_present:
            status_code = "READY"
        elif negative_present and not positive_present:
            status_code = "NEEDS_POSITIVE"
        elif positive_present and not negative_present:
            status_code = "NEEDS_NEGATIVE"
        else:
            status_code = "NO_USABLE_SUPERVISION"

        status_label, summary_message, next_action = _status_text(status_code)
    except Exception as exc:
        status_label, summary_message, next_action = _status_text("ERROR")
        notes.append(f"Artifact read/parse error: {exc}")
        return AnnotationReadinessResult(
            image_id=image_id,
            artifact_exists=exists_flags,
            polygon_counts={"Positive_Tumor": 0, "Negative_Tumor": 0, "NonTumor": 0, "Ignore": 0},
            pixel_counts={"Positive_Tumor": 0, "Negative_Tumor": 0, "NonTumor": 0, "Ignore": 0},
            roi_positive_pixels=0,
            status_code="ERROR",
            status_label=status_label,
            summary_message=summary_message,
            next_action=next_action,
            notes=notes,
        )

    return AnnotationReadinessResult(
        image_id=image_id,
        artifact_exists=exists_flags,
        polygon_counts=polygon_counts,
        pixel_counts=pixel_counts,
        roi_positive_pixels=roi_positive_pixels,
        status_code=status_code,
        status_label=status_label,
        summary_message=summary_message,
        next_action=next_action,
        notes=notes,
    )
