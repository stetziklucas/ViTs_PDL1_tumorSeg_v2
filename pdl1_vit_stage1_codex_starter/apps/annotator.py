"""Stage 1 Napari annotator for classed polygon/lasso annotation."""

from __future__ import annotations

import argparse
import json
import logging
import inspect
import sys
import time
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image
from skimage.draw import polygon2mask
from skimage.morphology import binary_closing, disk, remove_small_holes

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation_readiness import compute_annotation_readiness
from project_case_discovery import discover_project_cases
from report_history import newest_history_entry, oldest_history_entry, slugify_image_id
from project_report_history import (
    auto_select_latest_indices,
    discover_current_image_shared_reports,
    discover_project_summaries,
    format_current_image_report_label,
    format_project_summary_label,
)
from display_format import format_display_float


def qt_orientation(Qt: Any, name: str) -> Any:
    orientation = getattr(Qt, "Orientation", None)
    if orientation is not None and hasattr(orientation, name):
        return getattr(orientation, name)
    if hasattr(Qt, name):
        return getattr(Qt, name)
    raise AttributeError(f"Qt orientation {name} not available")



def qt_align_center(Qt: Any) -> Any:
    align = getattr(Qt, "AlignmentFlag", None)
    if align is not None and hasattr(align, "AlignCenter"):
        return align.AlignCenter
    return getattr(Qt, "AlignCenter")


def qt_keep_aspect_ratio(Qt: Any) -> Any:
    mode = getattr(Qt, "AspectRatioMode", None)
    if mode is not None and hasattr(mode, "KeepAspectRatio"):
        return mode.KeepAspectRatio
    return getattr(Qt, "KeepAspectRatio")


def qt_smooth_transformation(Qt: Any) -> Any:
    mode = getattr(Qt, "TransformationMode", None)
    if mode is not None and hasattr(mode, "SmoothTransformation"):
        return mode.SmoothTransformation
    return getattr(Qt, "SmoothTransformation")
from apps.verification_results_viewer import (
    load_verification_regions,
    load_verification_regions_payload,
    build_label_layer_transform_from_entry_or_payload,
    filter_verification_regions,
    sort_verification_regions,
    verification_region_label,
    viewer_bbox_from_region,
    compute_jump_zoom,
    canvas_size_wh,
    set_camera_center_yx,
    resolve_verification_regions_path,
    rectangle_vertices_from_bbox_yxhw,
    get_display_image_shape_hw,
    label_layer_transform_from_working_crop,
    resolve_region_image_path,
    verification_regions_message,
)


DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svs")
ROI_CLASSES = ("Positive_Tumor", "Negative_Tumor")


@dataclass(frozen=True)
class AnnotationPaths:
    """Container of deterministic output paths for one image annotation."""

    roi_mask_path: Path
    scribble_path: Path
    metadata_path: Path


class_mapping_default = {
    "Unlabeled": 0,
    "Positive_Tumor": 1,
    "Negative_Tumor": 2,
    "NonTumor": 3,
    "Ignore": 4,
}

CLASS_COLORS_HEX = {
    "Positive_Tumor": "#ef4444",
    "Negative_Tumor": "#f59e0b",
    "NonTumor": "#10b981",
    "Ignore": "#64748b",
}


def _hex_to_rgba(color_hex: str, alpha: float) -> np.ndarray:
    """Convert #RRGGBB to an RGBA float array in [0, 1]."""
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        raise ValueError(f"Expected 6-char hex color, received: {color_hex}")
    rgb = np.array([int(color_hex[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
    return np.concatenate([rgb, np.array([alpha], dtype=np.float32)])


def _class_names_from_layer(layer: Any) -> list[str]:
    """Return class_name property values as a list[str]."""
    values = layer.properties.get("class_name", np.array([], dtype=object))
    return [str(v) for v in values.tolist()]


def _set_layer_classes(layer: Any, class_names: list[str]) -> None:
    """Set layer class_name properties with dtype object."""
    layer.properties = {"class_name": np.asarray(class_names, dtype=object)}


def _sync_layer_colors(layer: Any, class_names: list[str]) -> None:
    """Apply deterministic per-class edge/face colors for polygon visibility."""
    if not class_names:
        return
    edge_colors = np.stack([_hex_to_rgba(CLASS_COLORS_HEX[name], alpha=1.0) for name in class_names], axis=0)
    face_colors = np.stack([_hex_to_rgba(CLASS_COLORS_HEX[name], alpha=0.28) for name in class_names], axis=0)
    layer.edge_color = edge_colors
    layer.face_color = face_colors


def _set_active_class_defaults(layer: Any, class_name: str) -> None:
    """Set active class defaults for newly drawn shapes across napari versions."""
    default_payload = {"class_name": np.array([class_name], dtype=object)}
    if hasattr(layer, "feature_defaults"):
        try:
            layer.feature_defaults = default_payload
        except Exception:  # pragma: no cover - compatibility fallback
            pass
    layer.current_properties = default_payload


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the Stage 1 Napari annotator."""
    parser = argparse.ArgumentParser(description="Stage 1 Napari annotator for classed polygon/lasso annotations.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image-id", help="Image identifier (stem) to load from --input.")
    source_group.add_argument("--image-path", type=Path, help="Direct image path to annotate.")

    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Input image directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Base output directory for annotation artifacts.",
    )
    parser.add_argument("--annotator", default="unknown", help="Annotator name for metadata.")
    parser.add_argument("--brush-size", type=int, default=16, help="Stored as metadata for compatibility.")
    parser.add_argument(
        "--roi-closing-radius",
        type=int,
        default=2,
        help="Disk radius for optional ROI morphological closing (0 disables).",
    )
    parser.add_argument(
        "--roi-fill-holes-area-threshold",
        type=int,
        default=64,
        help="Area threshold for optional ROI fill-holes operation (0 disables).",
    )
    parser.add_argument("--notes", default="", help="Free-text annotation notes.")
    parser.add_argument(
        "--uncertainty-comment",
        default="",
        help="Free-text uncertainty comment for ambiguous regions/classes.",
    )
    parser.add_argument(
        "--headless-smoke-test",
        action="store_true",
        help="Run helper export smoke test without launching GUI.",
    )
    parser.add_argument(
        "--synthetic-save-load-test",
        action="store_true",
        help="Run synthetic polygon save/load round-trip test without launching GUI.",
    )
    parser.add_argument(
        "--svs-load-test",
        action="store_true",
        help="Run non-GUI .svs load-path fallback/error-handling test.",
    )
    parser.add_argument(
        "--class-ui-init-check",
        action="store_true",
        help="Run a dependency diagnostic for Polygon Class Controls initialization.",
    )
    return parser


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config did not parse into a dictionary.")
    return config


def _read_svs_image(image_path: Path) -> np.ndarray:
    """Read an SVS whole-slide image at level 0 into RGB array."""
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "SVS support requires openslide-python and OpenSlide. "
            "Install with `pip install openslide-python` and system OpenSlide libraries."
        ) from exc

    try:
        with openslide.OpenSlide(str(image_path)) as slide:
            width, height = slide.dimensions
            region = slide.read_region((0, 0), 0, (width, height)).convert("RGB")
            return np.asarray(region)
    except Exception as exc:  # pragma: no cover - error path varies by OpenSlide build
        raise RuntimeError(f"Failed to read SVS slide '{image_path}': {exc}") from exc


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from disk into a NumPy array."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() == ".svs":
        return _read_svs_image(image_path)

    return np.asarray(Image.open(image_path))


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save a 2D integer mask as an 8-bit PNG file."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D. Received shape={mask.shape}.")
    if np.any(mask < 0):
        raise ValueError("Mask values must be non-negative.")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def load_mask_png(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load a 2D mask PNG and validate geometry."""
    if not path.exists():
        raise FileNotFoundError(path)
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(f"Loaded mask is not 2D: {path}")
    if mask.shape != expected_shape:
        raise ValueError(f"Geometry mismatch for {path}: expected={expected_shape} got={mask.shape}")
    return mask.astype(np.uint8)


def resolve_image_path(image_id: str | None, image_path: Path | None, input_dir: Path) -> Path:
    """Resolve image path from either explicit path or image_id lookup in input directory."""
    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return image_path

    if image_id is None:
        raise ValueError("Either image_id or image_path must be provided.")

    candidates = sorted(input_dir.glob(f"{image_id}.*"))
    candidates = [p for p in candidates if p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS]
    if not candidates:
        raise FileNotFoundError(f"No image found for image_id='{image_id}' in {input_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple images found for image_id='{image_id}': {candidates}")
    return candidates[0]


def derive_image_id(image_path: Path, cli_image_id: str | None) -> str:
    """Derive deterministic image_id from CLI value or image filename stem."""
    return cli_image_id or image_path.stem


def build_annotation_paths(output_dir: Path, image_id: str) -> AnnotationPaths:
    """Build deterministic annotation artifact paths."""
    return AnnotationPaths(
        roi_mask_path=output_dir / "roi_masks" / f"{image_id}_roi_mask.png",
        scribble_path=output_dir / "scribbles" / f"{image_id}_scribble_labels.png",
        metadata_path=output_dir / f"{image_id}_annotation_meta.json",
    )


def classes_used_from_scribbles(scribbles: np.ndarray, label_to_name: dict[int, str]) -> list[str]:
    """Get sorted class names present in scribble labels, excluding unlabeled=0."""
    used_values = sorted(int(v) for v in np.unique(scribbles) if int(v) != 0)
    return [label_to_name[v] for v in used_values if v in label_to_name]


def write_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Write annotation metadata as JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sanitize_class_name(class_name: str, class_mapping: dict[str, int]) -> str:
    if class_name not in class_mapping or class_name == "Unlabeled":
        raise ValueError(f"Unsupported class name in polygon annotation: {class_name}")
    return class_name


def _normalize_polygons(polygons: list[dict[str, Any]], class_mapping: dict[str, int]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for polygon in polygons:
        class_name = _sanitize_class_name(str(polygon["class_name"]), class_mapping)
        vertices = np.asarray(polygon["vertices"], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
            raise ValueError("Polygon vertices must be an Nx2 array with N>=3.")
        normalized.append({"class_name": class_name, "vertices": vertices.tolist()})
    return normalized


def rasterize_polygons_to_class_mask(
    polygons: list[dict[str, Any]],
    image_shape: tuple[int, int],
    class_mapping: dict[str, int],
) -> np.ndarray:
    """Rasterize classed polygons into a dense class mask."""
    class_mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        class_name = _sanitize_class_name(str(polygon["class_name"]), class_mapping)
        vertices = np.asarray(polygon["vertices"], dtype=float)
        poly_mask = polygon2mask(image_shape, vertices)
        class_mask[poly_mask] = class_mapping[class_name]
    return class_mask


def derive_roi_mask_from_polygons(
    polygons: list[dict[str, Any]],
    image_shape: tuple[int, int],
    *,
    closing_radius: int,
    fill_holes_area_threshold: int,
) -> np.ndarray:
    """Build ROI mask from Positive/Negative tumor polygon union."""
    roi = np.zeros(image_shape, dtype=bool)
    for polygon in polygons:
        class_name = str(polygon["class_name"])
        if class_name not in ROI_CLASSES:
            continue
        vertices = np.asarray(polygon["vertices"], dtype=float)
        roi |= polygon2mask(image_shape, vertices)

    if closing_radius > 0 and np.any(roi):
        roi = binary_closing(roi, disk(closing_radius))
    if fill_holes_area_threshold > 0 and np.any(roi):
        roi = remove_small_holes(roi, area_threshold=fill_holes_area_threshold)

    return roi.astype(np.uint8)


def export_annotation_artifacts(
    polygons: list[dict[str, Any]],
    paths: AnnotationPaths,
    *,
    image_shape: tuple[int, int],
    image_id: str,
    annotator: str,
    brush_size: int,
    notes: str,
    uncertainty_comment: str,
    class_mapping: dict[str, int],
    roi_closing_radius: int,
    roi_fill_holes_area_threshold: int,
) -> dict[str, Any]:
    """Save derived ROI mask, class mask, and metadata for one image."""
    normalized_polygons = _normalize_polygons(polygons, class_mapping)
    scribbles = rasterize_polygons_to_class_mask(normalized_polygons, image_shape, class_mapping)
    roi_binary = derive_roi_mask_from_polygons(
        normalized_polygons,
        image_shape,
        closing_radius=roi_closing_radius,
        fill_holes_area_threshold=roi_fill_holes_area_threshold,
    )

    save_mask_png(roi_binary, paths.roi_mask_path)
    save_mask_png(scribbles, paths.scribble_path)

    label_to_name = {v: k for k, v in class_mapping.items()}
    metadata = {
        "image_id": image_id,
        "annotator": annotator,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brush_size": int(brush_size),
        "classes": classes_used_from_scribbles(scribbles, label_to_name),
        "notes": notes,
        "uncertainty_comment": uncertainty_comment,
        "annotation_mode": "polygon_lasso",
        "roi_derivation": {
            "source_classes": list(ROI_CLASSES),
            "closing_radius": int(roi_closing_radius),
            "fill_holes_area_threshold": int(roi_fill_holes_area_threshold),
        },
        "polygons": normalized_polygons,
    }
    write_metadata(metadata, paths.metadata_path)
    return metadata


def load_previous_polygons(paths: AnnotationPaths, class_mapping: dict[str, int]) -> list[dict[str, Any]]:
    """Load previous polygons from metadata if available."""
    if not paths.metadata_path.exists():
        return []

    with paths.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    polygons = metadata.get("polygons", [])
    if not isinstance(polygons, list):
        raise ValueError(f"Invalid polygons payload in metadata: {paths.metadata_path}")
    return _normalize_polygons(polygons, class_mapping)


def build_shapes_layer_kwargs(
    prior_vertices: list[np.ndarray],
    prior_classes: list[str],
    class_choices: np.ndarray,
) -> dict[str, Any]:
    """Build napari Shapes layer kwargs for classed polygon workflow."""
    return {
        "data": prior_vertices,
        "shape_type": "polygon",
        "name": "class_polygons",
        "edge_width": 2,
        "edge_color": "white",
        "face_color": "transparent",
        "properties": {"class_name": np.array(prior_classes or [], dtype=object)},
        "property_choices": {"class_name": class_choices},
    }


def load_class_control_widgets() -> tuple[Any, Any, Any, Any]:
    """Load required magicgui widget classes for polygon class controls."""
    try:
        from magicgui.widgets import ComboBox, Container, Label, PushButton
    except ImportError as exc:
        raise RuntimeError(
            "Polygon Class Controls require 'magicgui'. "
            "Install dependencies from requirements.txt and relaunch the annotator."
        ) from exc
    return ComboBox, Container, Label, PushButton


def default_gui_run_tag(image_id: str, now: datetime | None = None) -> str:
    """Build default unique run tag for annotator-launched Stage 1 runs."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    alias = slugify_image_id(image_id).split("_")[0] or "image"
    return f"{alias[:16]}_gui_{stamp}"


def build_stage1_run_command(
    *,
    config_path: Path,
    image_id: str,
    run_tag: str,
    raw_dir: Path,
    annotations_dir: Path,
    outputs_root: Path,
    models_root: Path,
) -> list[str]:
    """Build subprocess command used by GUI Run Stage 1 action."""
    return [
        sys.executable,
        "scripts/run_stage1_image.py",
        "--config",
        str(config_path),
        "--image-id",
        image_id,
        "--run-tag",
        run_tag,
        "--raw-dir",
        str(raw_dir),
        "--annotations-dir",
        str(annotations_dir),
        "--outputs-root",
        str(outputs_root),
        "--models-root",
        str(models_root),
    ]




def default_project_tag(embedding_encoder_id: str = "current_timm", now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"training_{stamp}__{embedding_encoder_id}"


def build_stage1_project_command(*, config_path: Path, project_tag: str, raw_dir: Path, annotations_dir: Path, outputs_root: Path, models_root: Path, embedding_encoder_id: str = "current_timm") -> list[str]:
    return [sys.executable, "scripts/run_stage1_project.py", "--config", str(config_path), "--project-tag", project_tag, "--discover-ready-cases", "--allow-skip-not-ready", "--embedding-encoder", embedding_encoder_id, "--raw-dir", str(raw_dir), "--annotations-dir", str(annotations_dir), "--outputs-root", str(outputs_root), "--models-root", str(models_root)]


def resolve_current_image_shared_report(*, outputs_root: Path, project_tag: str, image_id: str) -> Path | None:
    cases_json = outputs_root / f"reports_training_{project_tag}" / "stage1_project_cases.json"
    if not cases_json.exists():
        return None
    payload = json.loads(cases_json.read_text(encoding="utf-8"))
    for row in payload.get("included_ready_cases", []):
        if row.get("image_id") == image_id:
            alias = row.get("alias")
            if alias:
                p = outputs_root / f"reports_{project_tag}__{alias}" / "report_summary.md"
                return p
    return None


def _resolve_existing_path(entry: dict[str, Any] | None, key: str) -> Path | None:
    if not isinstance(entry, dict):
        return None
    raw = entry.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    return path if path.exists() else None


def resolve_verification_overlay_path(entry: dict[str, Any] | None) -> Path | None:
    """Resolve an existing verification overlay path from a shared-report history entry."""
    return _resolve_existing_path(entry, "verification_overlay_path")


def resolve_verification_annotation_labels_path(entry: dict[str, Any] | None) -> Path | None:
    return _resolve_existing_path(entry, "verification_annotation_labels_path")
def resolve_verification_prediction_labels_path(entry: dict[str, Any] | None) -> Path | None:
    return _resolve_existing_path(entry, "verification_prediction_labels_path")


def verification_overlay_translate(entry: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(entry, dict):
        return (0, 0)
    try:
        return (int(entry.get("crop_y0") or 0), int(entry.get("crop_x0") or 0))
    except (TypeError, ValueError):
        return (0, 0)


def build_verification_label_layer_kwargs(entry: dict[str, Any] | None, display_shape_hw: tuple[int, int] | None, regions_payload: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    pred_kwargs = {"name": "verification_prediction_labels", "opacity": 0.88, "blending": "translucent_no_depth"}
    ann_kwargs = {"name": "verification_annotation_labels", "opacity": 0.24}
    meta = build_label_layer_transform_from_entry_or_payload(entry, regions_payload, display_shape_hw)
    if meta.get("warning"):
        return None, None, f"Cannot place verification labels: {meta['warning']}"
    tfm = {"scale": meta["scale"], "translate": meta["translate"]}
    pred_kwargs.update(tfm)
    ann_kwargs.update(tfm)
    return pred_kwargs, ann_kwargs, None


def verification_mask_layer_kwargs(mask: np.ndarray, entry: dict[str, Any] | None) -> dict[str, Any]:
    """Build deterministic napari kwargs for class-aware verification labels rendering."""
    return {
        "name": "verification_prediction_labels",
        "opacity": 0.88,
        "blending": "translucent_no_depth",
        "translate": verification_overlay_translate(entry),
    }


def build_polygon_review_face_colors(class_names: list[str], alpha: float = 0.12) -> np.ndarray:
    """Build class-preserving polygon face colors for verification review mode."""
    if not class_names:
        return np.empty((0, 4), dtype=np.float32)
    return np.stack([_hex_to_rgba(CLASS_COLORS_HEX[name], alpha=alpha) for name in class_names], axis=0)
def latest_project_summary_path(outputs_root: Path, project_tag: str) -> Path:
    return outputs_root / f"reports_training_{project_tag}" / "training_summary.md"
def load_markdown_preview(preferred_paths: list[Path]) -> tuple[str, Path | None]:
    """Load first available markdown content from preferred path candidates."""
    for path in preferred_paths:
        if path.exists():
            return path.read_text(encoding="utf-8"), path
    return "No markdown summary available yet.", None


def history_preview_candidates(entry: dict[str, Any]) -> list[Path]:
    """Build preferred markdown preview candidates for a history row."""
    candidates: list[Path] = []
    report_md = str(entry.get("report_summary_md") or "").strip()
    if report_md:
        candidates.append(Path(report_md))
    stage1_md = str(entry.get("stage1_run_summary_md") or "").strip()
    if stage1_md:
        candidates.append(Path(stage1_md))
    report_json = str(entry.get("report_summary_json") or "").strip()
    if report_json:
        candidates.append(Path(report_json).with_suffix(".md"))
    return candidates


def select_history_index_by_jump(entries: list[dict[str, Any]], jump: str) -> int | None:
    """Return history-row index for newest/oldest jump actions."""
    if not entries:
        return None
    target = newest_history_entry(entries) if jump == "newest" else oldest_history_entry(entries)
    if not target:
        return None
    target_tag = str(target.get("run_tag", ""))
    for idx, row in enumerate(entries):
        if str(row.get("run_tag", "")) == target_tag:
            return idx
    return 0


def format_selected_run_tag_text(entry: dict[str, Any] | None) -> str:
    """Return concise selected-run-tag text for clear panel visibility."""
    if not entry:
        return "Selected history run tag: none"
    return f"Selected history run tag: {entry.get('run_tag') or 'n/a'}"


def runner_log_visibility_status(show_log: bool) -> str:
    """Status line for log visibility toggle."""
    return "Status: runner log shown." if show_log else "Status: runner log hidden; report preview expanded."


def compact_path_label(path_text: str, max_chars: int = 72) -> str:
    """Compact long path/status text for narrow UI panels."""
    clean = str(path_text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return f"...{clean[-(max_chars - 3):]}"


def run_class_ui_init_check() -> None:
    """Diagnostic check that class-control UI dependencies are importable."""
    ComboBox, Container, Label, PushButton = load_class_control_widgets()
    assert ComboBox is not None and Container is not None and Label is not None and PushButton is not None
    logging.info("Class-control UI dependency check passed (magicgui widgets importable).")


def run_class_event_compatibility_check() -> None:
    """Regression check: class UI wiring must not bind to unsupported selected_data event."""
    source = inspect.getsource(launch_napari_app)
    assert ".events.selected_data.connect" not in source
    logging.info("Class-control event compatibility check passed (no selected_data event binding).")


def launch_napari_app(
    image: np.ndarray,
    image_id: str,
    paths: AnnotationPaths,
    *,
    config: dict[str, Any],
    annotator: str,
    brush_size: int,
    notes: str,
    uncertainty_comment: str,
    class_mapping: dict[str, int],
    roi_closing_radius: int,
    roi_fill_holes_area_threshold: int,
) -> None:
    """Launch Napari viewer and register save shortcut for annotation export."""
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for GUI mode. Install with `pip install napari[all]` or similar."
        ) from exc

    if image.ndim < 2:
        raise ValueError(f"Expected image with >=2 dimensions, received shape={image.shape}")

    image_shape = image.shape[:2]
    prior_polygons = load_previous_polygons(paths, class_mapping)
    prior_vertices = [np.asarray(poly["vertices"], dtype=float) for poly in prior_polygons]
    prior_classes = [poly["class_name"] for poly in prior_polygons]

    viewer = napari.Viewer(title=f"Stage 1 Annotator - {image_id}")
    viewer.add_image(image, name="image")

    class_choices = np.array([k for k in class_mapping if k != "Unlabeled"], dtype=object)
    polygon_layer = viewer.add_shapes(**build_shapes_layer_kwargs(prior_vertices, prior_classes, class_choices))
    active_class: dict[str, str] = {"value": str(class_choices[0])}
    _set_active_class_defaults(polygon_layer, active_class["value"])
    polygon_layer.mode = "add_polygon"
    _sync_layer_colors(polygon_layer, _class_names_from_layer(polygon_layer))

    def _sync_class_array_to_data() -> None:
        class_names = _class_names_from_layer(polygon_layer)
        n_shapes = len(polygon_layer.data)
        if len(class_names) < n_shapes:
            class_names.extend([active_class["value"]] * (n_shapes - len(class_names)))
            _set_layer_classes(polygon_layer, class_names)
        elif len(class_names) > n_shapes:
            _set_layer_classes(polygon_layer, class_names[:n_shapes])
            class_names = class_names[:n_shapes]
        _sync_layer_colors(polygon_layer, class_names)

    ComboBox, Container, Label, PushButton = load_class_control_widgets()

    active_class_widget = ComboBox(
        label="Active polygon class",
        choices=[str(choice) for choice in class_choices.tolist()],
        value=active_class["value"],
    )
    selected_class_label = Label(value="Selected polygon class: none")
    apply_selected_class_btn = PushButton(text="Apply active class to selected polygon(s)")
    refresh_selected_class_btn = PushButton(text="Refresh selected polygon class")

    def _refresh_selected_class_label() -> None:
        selected = sorted(int(i) for i in polygon_layer.selected_data)
        if not selected:
            selected_class_label.value = "Selected polygon class: none"
            return
        class_names = _class_names_from_layer(polygon_layer)
        selected_classes = [class_names[i] for i in selected if i < len(class_names)]
        if not selected_classes:
            selected_class_label.value = "Selected polygon class: none"
            return
        unique_classes = sorted(set(selected_classes))
        if len(unique_classes) == 1:
            selected_class_label.value = f"Selected polygon class: {unique_classes[0]}"
        else:
            selected_class_label.value = f"Selected polygon class: mixed ({', '.join(unique_classes)})"

    @active_class_widget.changed.connect
    def _on_active_class_changed(value: str) -> None:
        active_class["value"] = str(value)
        _set_active_class_defaults(polygon_layer, active_class["value"])

    @apply_selected_class_btn.clicked.connect
    def _apply_active_class_to_selection() -> None:
        selected = sorted(int(i) for i in polygon_layer.selected_data)
        if not selected:
            return
        class_names = _class_names_from_layer(polygon_layer)
        for idx in selected:
            if 0 <= idx < len(class_names):
                class_names[idx] = active_class["value"]
        _set_layer_classes(polygon_layer, class_names)
        _sync_layer_colors(polygon_layer, class_names)
        _refresh_selected_class_label()

    @refresh_selected_class_btn.clicked.connect
    def _on_refresh_selected_class_clicked() -> None:
        _refresh_selected_class_label()

    @polygon_layer.events.data.connect
    def _on_shapes_data_changed(_: Any = None) -> None:
        _sync_class_array_to_data()
        _refresh_selected_class_label()

    @polygon_layer.events.set_data.connect
    def _on_shapes_set_data(_: Any = None) -> None:
        _sync_class_array_to_data()
        _refresh_selected_class_label()

    if hasattr(polygon_layer.events, "features"):
        @polygon_layer.events.features.connect
        def _on_features_changed(_: Any = None) -> None:
            _refresh_selected_class_label()

    class_panel = Container(
        widgets=[active_class_widget, apply_selected_class_btn, refresh_selected_class_btn, selected_class_label],
        labels=False,
    )
    viewer.window.add_dock_widget(class_panel, area='left', name='Polygon Class Controls')

    save_status_label = Label(value="Save status: Not saved yet")
    readiness_label = Label(value="Readiness: Not checked yet")
    next_action_label = Label(value="Next action: Press 's' to save annotations and run readiness check")
    status_panel = Container(
        widgets=[save_status_label, readiness_label, next_action_label],
        labels=False,
    )
    viewer.window.add_dock_widget(status_panel, area='left', name='Annotation Save Status')

    from qtpy.QtCore import QProcess, Qt
    from qtpy.QtGui import QPixmap
    from qtpy.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QLineEdit, QPushButton, QPlainTextEdit, QTextEdit, QVBoxLayout, QWidget, QSizePolicy, QTableWidget, QTableWidgetItem

    gui_outputs_root = Path("outputs")
    gui_models_root = Path("models")
    gui_raw_dir = Path("data/raw")
    gui_annotations_dir = paths.metadata_path.parent

    workflow_panel = QWidget()
    workflow_layout = QVBoxLayout(workflow_panel)
    from qtpy.QtWidgets import QSplitter
    workflow_status = QLabel("Project status: idle")
    latest_project_tag = QLabel("Latest generated/shared project tag: none")
    selected_project_tag = QLabel("Selected project summary tag: none")
    selected_image_tag = QLabel("Selected current-image shared-report tag: none")
    project_counts = QLabel("READY case count: 0 | skipped case count: 0")
    project_details = QLabel("Current image included = false")
    preview_path_label = QLabel("Currently loaded preview path: none")
    encoder_cfg = config.get("embedding_encoder", {}) if isinstance(config.get("embedding_encoder", {}), dict) else {}
    encoder_registry = encoder_cfg.get("registry", {}) if isinstance(encoder_cfg.get("registry", {}), dict) else {}
    if not encoder_registry:
        encoder_registry = {"current_timm": {"display_name": "Current ViT baseline", "backend": "timm", "model_name": "vit_base_patch16_224"}}
    default_encoder_id = str(encoder_cfg.get("selected") or "current_timm")
    if default_encoder_id not in encoder_registry:
        default_encoder_id = next(iter(encoder_registry.keys()))
    encoder_combo = QComboBox()
    for enc_id, meta in encoder_registry.items():
        display = str(meta.get("display_name") or enc_id)
        encoder_combo.addItem(f"{display} ({enc_id})", enc_id)
    encoder_hint = QLabel("Embedding encoder: n/a")
    encoder_note = QLabel("")
    encoder_note.setWordWrap(True)
    project_combo = QComboBox(); image_combo = QComboBox()
    run_btn = QPushButton("Train model and run verification")
    show_log_toggle = QCheckBox("Show runner log"); show_log_toggle.setChecked(True)
    show_verification_toggle = QCheckBox("Debug: show raw verification labels"); show_verification_toggle.setChecked(False)
    open_verification_viewer_btn = QPushButton("Open verification results viewer")
    verification_status = QLabel("Verification review: positive=red, negative=gold, nontumor=green, ignore=gray")
    preview_box = QTextEdit(); preview_box.setReadOnly(True)
    project_log = QPlainTextEdit(); project_log.setReadOnly(True)
    workflow_panel.setMinimumWidth(280)
    workflow_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    for label in (workflow_status, latest_project_tag, selected_project_tag, selected_image_tag, project_counts, project_details, verification_status, preview_path_label):
        label.setWordWrap(True); label.setTextInteractionFlags(Qt.TextSelectableByMouse); label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum); label.setMinimumWidth(0)
    for combo in (project_combo, image_combo):
        combo.setMinimumWidth(0); combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); combo.setMinimumContentsLength(16); combo.view().setTextElideMode(Qt.ElideRight); combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    preview_box.setLineWrapMode(QTextEdit.WidgetWidth); preview_box.setMinimumWidth(0); preview_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    project_log.setLineWrapMode(QPlainTextEdit.WidgetWidth); project_log.setMinimumWidth(0); project_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    splitter = QSplitter(); splitter.setOrientation(qt_orientation(Qt, "Vertical")); splitter.addWidget(preview_box); splitter.addWidget(project_log); splitter.setStretchFactor(0,5); splitter.setStretchFactor(1,1); splitter.setChildrenCollapsible(True); splitter.setCollapsible(0, True); splitter.setCollapsible(1, True); splitter.setMinimumWidth(0); splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    for w in (QLabel('Embedding encoder:'), encoder_combo, encoder_hint, encoder_note, run_btn, open_verification_viewer_btn, show_log_toggle, show_verification_toggle, QLabel('Project summaries'), project_combo, QLabel('Current image shared reports'), image_combo, workflow_status, latest_project_tag, selected_project_tag, selected_image_tag, project_counts, project_details, verification_status, preview_path_label, splitter): workflow_layout.addWidget(w)
    viewer.window.add_dock_widget(workflow_panel, area='right', name='Stage 1 Workflow')

    project_runner = QProcess(viewer.window.qt_viewer); project_runner.setProcessChannelMode(QProcess.MergedChannels)
    wf_state={'project_entries':[],'image_entries':[]}
    verification_layer_name = "verification_prediction_labels"
    verification_labels_layer_name = "verification_annotation_labels"
    polygon_review_state: dict[str, Any] = {"saved": False, "face_color": None, "edge_color": None}
    verification_viewer_state: dict[str, Any] = {"dock": None, "table": None, "regions": [], "status": None, "class_filter": None, "issue_filter": None, "sort_filter": None, "preview": None, "report": None}

    def _load_preview(path: Path|None)->None:
        if path and path.exists():
            preview_box.setMarkdown(path.read_text(encoding='utf-8')); preview_path_label.setText(f"Currently loaded preview path: {compact_path_label(path.as_posix())}"); preview_path_label.setToolTip(path.as_posix())
        else:
            preview_box.setMarkdown('No report history found yet.'); preview_path_label.setText('Currently loaded preview path: none'); preview_path_label.setToolTip("")
    def _apply_polygon_review_style(enable: bool) -> None:
        class_names = _class_names_from_layer(polygon_layer)
        if enable:
            if not polygon_review_state["saved"]:
                polygon_review_state["face_color"] = np.asarray(polygon_layer.face_color).copy()
                polygon_review_state["edge_color"] = np.asarray(polygon_layer.edge_color).copy()
                polygon_review_state["saved"] = True
            edge = np.stack([_hex_to_rgba(CLASS_COLORS_HEX[name], alpha=1.0) for name in class_names], axis=0) if class_names else np.empty((0, 4))
            face = build_polygon_review_face_colors(class_names, alpha=0.12)
            polygon_layer.edge_color = edge
            polygon_layer.face_color = face
            return
        if polygon_review_state["saved"] and polygon_review_state["face_color"] is not None and polygon_review_state["edge_color"] is not None:
            polygon_layer.face_color = polygon_review_state["face_color"]
            polygon_layer.edge_color = polygon_review_state["edge_color"]
            polygon_review_state["saved"] = False
            polygon_review_state["face_color"] = None
            polygon_review_state["edge_color"] = None
        _sync_layer_colors(polygon_layer, class_names)

    def _sync_verification_overlay()->None:
        if not show_verification_toggle.isChecked():
            if verification_layer_name in viewer.layers:
                viewer.layers.remove(viewer.layers[verification_layer_name])
            if verification_labels_layer_name in viewer.layers:
                viewer.layers.remove(viewer.layers[verification_labels_layer_name])
            _apply_polygon_review_style(False)
            verification_status.setText("Verification review: off")
            return
        idx=image_combo.currentIndex()
        if idx<0 or idx>=len(wf_state['image_entries']):
            selected_image_tag.setText("Selected current-image shared-report tag: none")
            verification_status.setText("Verification mask: unavailable for selected report"); return
        entry = wf_state['image_entries'][idx]
        selected_image_tag.setText(f"Selected current-image shared-report tag: {entry.get('project_tag')} | shared F1 {format_display_float(entry.get('f1'))}")
        path = resolve_verification_prediction_labels_path(entry)
        labels_path = resolve_verification_annotation_labels_path(entry)
        if path is None or labels_path is None:
            verification_status.setText("Verification review: unavailable for selected report")
            show_verification_toggle.setChecked(False)
            return
        prediction_labels = np.asarray(Image.open(path))
        if prediction_labels.ndim == 3:
            prediction_labels = prediction_labels[..., 0]
        if np.count_nonzero(prediction_labels) == 0:
            verification_status.setText("Verification review: unavailable for selected report")
            show_verification_toggle.setChecked(False)
            return
        ann = np.asarray(Image.open(labels_path)).astype(np.uint8)
        display_shape = get_display_image_shape_hw(viewer)
        regions_payload = None
        report_path = Path(entry.get("report_summary_md")) if entry.get("report_summary_md") else None
        regions_path, _ = resolve_verification_regions_path(verification_regions_path=entry.get("verification_regions_path"), report_path=report_path, repo_root=REPO_ROOT)
        if regions_path is not None:
            regions_payload = load_verification_regions_payload(regions_path)
        layer_kwargs, ann_kwargs, placement_warning = build_verification_label_layer_kwargs(entry, display_shape, regions_payload)
        if placement_warning:
            verification_status.setText(placement_warning)
            return
        ann_color = {1: "#fbcfe8", 2: "#fde68a", 3: "#bbf7d0", 4: "#d1d5db", 0: [0,0,0,0]}
        pred_color = {1: "#be123c", 2: "#d97706", 3: "#15803d", 4: "#6b7280", 5: "#c026d3", 0: [0,0,0,0]}
        if verification_layer_name in viewer.layers:
            layer = viewer.layers[verification_layer_name]
            layer.data = prediction_labels
            layer.translate = layer_kwargs["translate"]
            layer.scale = layer_kwargs["scale"]
            layer.opacity = layer_kwargs["opacity"]
            layer.blending = layer_kwargs["blending"]
            layer.color = pred_color
            layer.visible = True
        else:
            viewer.add_labels(prediction_labels, **layer_kwargs); viewer.layers[verification_layer_name].color = pred_color
        if verification_labels_layer_name in viewer.layers:
            ll = viewer.layers[verification_labels_layer_name]; ll.data = ann; ll.translate = ann_kwargs["translate"]; ll.scale = ann_kwargs["scale"]; ll.opacity = ann_kwargs["opacity"]; ll.color = ann_color; ll.visible = True
        else:
            viewer.add_labels(ann, **ann_kwargs); viewer.layers[verification_labels_layer_name].color = ann_color
        viewer.layers.move(viewer.layers.index(verification_labels_layer_name), len(viewer.layers) - 2)
        viewer.layers.move(viewer.layers.index(verification_layer_name), len(viewer.layers) - 1)
        _apply_polygon_review_style(True)
        verification_status.setText(f"Verification review loaded for {entry.get('project_tag')}: positive=red, negative=gold, nontumor=green, ignore=gray")


    def _selected_encoder_id() -> str:
        return str(encoder_combo.currentData() or default_encoder_id)

    def _selected_encoder_meta() -> dict[str, Any]:
        return encoder_registry.get(_selected_encoder_id(), {}) if isinstance(encoder_registry, dict) else {}

    def _refresh_encoder_hint() -> None:
        enc_id = _selected_encoder_id()
        meta = _selected_encoder_meta()
        display = str(meta.get("display_name") or enc_id)
        backend = str(meta.get("backend") or "n/a")
        model = str(meta.get("model_name") or "n/a")
        trust = bool(meta.get("trust_remote_code"))
        if enc_id == "hibou_b":
            encoder_hint.setText(f"{display}: {model}; requires Hugging Face access; trust_remote_code={trust}")
        else:
            encoder_hint.setText(f"{display}: {backend} / {model}")
        if bool(meta.get("requires_hf_auth")):
            encoder_note.setText("Requires accepted Hugging Face model access and VM login. Run scripts/check_embedding_encoder_env.py if loading fails.")
        else:
            encoder_note.setText("")

    def _refresh_workflow(select_tag: str|None=None)->None:
        summary = discover_project_cases(config=config, annotations_dir=gui_annotations_dir, raw_dir=gui_raw_dir)
        included = summary.get('included_ready_cases',[]); skipped = summary.get('skipped_cases',[])
        project_counts.setText(f"READY case count: {len(included)} | skipped case count: {len(skipped)}")
        included_short=', '.join([f"{r['alias']}={r['image_id']}" for r in included[:4]]) or 'none'
        skipped_short=', '.join([f"{r.get('alias')}:{r.get('reason')}" for r in skipped[:4]]) or 'none'
        current_included=any(r.get('image_id')==image_id for r in included)
        project_details.setText(f"Current image included={str(current_included).lower()} | included: {included_short} | skipped: {skipped_short}")
        projects = discover_project_summaries(outputs_root=gui_outputs_root, current_image_id=image_id)
        images = discover_current_image_shared_reports(image_id=image_id, outputs_root=gui_outputs_root)
        wf_state['project_entries']=projects; wf_state['image_entries']=images
        project_combo.blockSignals(True); project_combo.clear(); [project_combo.addItem(format_project_summary_label(e)) for e in projects]; project_combo.blockSignals(False)
        image_combo.blockSignals(True); image_combo.clear(); [image_combo.addItem(format_current_image_report_label(e)) for e in images]; image_combo.blockSignals(False)
        pidx,iidx=auto_select_latest_indices(projects,images)
        if select_tag:
            for i,e in enumerate(projects):
                if e.get('project_tag')==select_tag: pidx=i; break
        if pidx is not None:
            project_combo.setCurrentIndex(pidx); e=projects[pidx]; selected_project_tag.setText(f"Selected project summary tag: {e.get('project_tag')} | agg F1 {format_display_float(e.get('aggregate_f1'))}"); _load_preview(Path(e.get('training_summary_md','')))
        if iidx is not None:
            image_combo.setCurrentIndex(iidx); e=images[iidx]; selected_image_tag.setText(f"Selected current-image shared-report tag: {e.get('project_tag')} | shared F1 {format_display_float(e.get('f1'))}"); _load_preview(Path(e.get('report_summary_md','')))
        else:
            selected_image_tag.setText("Selected current-image shared-report tag: none")
            if pidx is None:
                _load_preview(None)
        _sync_verification_overlay()


    def _open_verification_results_viewer()->None:
        idx=image_combo.currentIndex()
        if idx<0 or idx>=len(wf_state['image_entries']):
            verification_status.setText('Verification results viewer: select a current-image shared report first.')
            return
        entry=wf_state['image_entries'][idx]
        report_path = Path(entry.get("report_summary_md")) if entry.get("report_summary_md") else None
        resolved, candidates = resolve_verification_regions_path(verification_regions_path=entry.get('verification_regions_path'), report_path=report_path, repo_root=REPO_ROOT)
        if resolved is None:
            verification_status.setText(f"Verification results viewer: verification_regions.json missing. candidates={[c.as_posix() for c in candidates]}")
            return
        payload = load_verification_regions_payload(resolved)
        regions = payload.get("regions", [])
        resolved_regions_path = resolved
        msg = verification_regions_message(resolved, regions)
        if msg:
            verification_status.setText(msg); return
        if verification_viewer_state["dock"] is None:
            box = QWidget(); layout = QVBoxLayout(box)
            report_lbl, status_lbl = QLabel("Selected report: none"), QLabel("Status: idle")
            cf, inf, sf = QComboBox(), QComboBox(), QComboBox()
            cf.addItems(["All"] + sorted({str(r.get("class_name","Unknown")) for r in regions}))
            inf.addItems(["All"] + sorted({str(r.get("issue","unknown")) for r in regions}))
            sf.addItems(["Highest error first","Lowest score first","Highest score first","Class then error","Annotation order / region id"])
            table = QTableWidget(); table.setColumnCount(8); table.setHorizontalHeaderLabels(["class_name","source_type","score_name","score","error_px","annotated_px","issue","thumbnail"])
            preview = QLabel("Preview: select a row")
            preview.setMinimumHeight(280)
            preview.setScaledContents(False)
            preview.setAlignment(qt_align_center(Qt))
            preview_status = QLabel("Preview path: none")
            jump_btn = QPushButton("Jump to selected region")
            for w in (report_lbl, status_lbl, cf, inf, sf, table, preview, preview_status, jump_btn): layout.addWidget(w)
            viewer.window.add_dock_widget(box, area="right", name="Verification Results Viewer")
            verification_viewer_state.update({"dock": box, "table": table, "status": status_lbl, "class_filter": cf, "issue_filter": inf, "sort_filter": sf, "preview": preview, "report": report_lbl})
            jump_summary = {"text": "none"}
            def _refresh_table():
                rs = sort_verification_regions(filter_verification_regions(verification_viewer_state["regions"], cf.currentText(), inf.currentText()), sf.currentText())
                verification_viewer_state["filtered"] = rs; table.setRowCount(len(rs))
                for ri, rr in enumerate(rs):
                    vals=[rr.get("class_name",""), rr.get("source_type",""), rr.get("score_name",""), f"{rr.get('score')}", str(rr.get("error_px",0)), str(rr.get("annotated_px",0)), rr.get("issue",""), Path(rr.get("thumbnail_path","")).name]
                    for ci, v in enumerate(vals): table.setItem(ri, ci, QTableWidgetItem(str(v)))
                sel = table.currentRow()
                sel_region = rs[sel].get("region_id") if 0 <= sel < len(rs) else "none"
                status_lbl.setText(f"Status: total={len(verification_viewer_state['regions'])} filtered={len(rs)} source_counts={payload.get('region_source_counts',{})} selected_region={sel_region} jump={jump_summary['text']}")
            def _on_select():
                idxr=table.currentRow()
                if idxr < 0 or idxr >= len(verification_viewer_state.get("filtered",[])):
                    return
                rr = verification_viewer_state["filtered"][idxr]
                resolved=None; candidates=[]
                for key in ("preview_path","thumbnail_path"):
                    resolved, candidates = resolve_region_image_path(image_path=rr.get(key), regions_json_path=resolved_regions_path, repo_root=REPO_ROOT)
                    if resolved is not None:
                        pm=QPixmap(resolved.as_posix())
                        preview.setPixmap(pm.scaled(preview.size(), qt_keep_aspect_ratio(Qt), qt_smooth_transformation(Qt)))
                        preview_status.setText(f"Preview path resolved: {resolved.as_posix()}")
                        verification_status.setText(f"Verification results viewer selected: region_id={rr.get('region_id')} preview={resolved.name}")
                        _refresh_table()
                        return
                preview.setText("Preview unavailable")
                preview_status.setText(f"Preview path failed candidates: {[c.as_posix() for c in candidates]}")

            def _jump():
                idxr=table.currentRow()
                if idxr < 0 or idxr >= len(verification_viewer_state.get("filtered",[])): return
                rr = verification_viewer_state["filtered"][idxr]
                ds = get_display_image_shape_hw(viewer)
                bbox = viewer_bbox_from_region(rr, display_shape_hw=ds); cy, cx = bbox["center_yx"]
                center_used = set_camera_center_yx(viewer, [cy, cx])
                old_zoom = float(viewer.camera.zoom)
                canvas = getattr(getattr(viewer.window, "qt_viewer", None), "canvas", None)
                canvas_shape = canvas_size_wh(canvas)
                new_zoom = compute_jump_zoom([bbox["y"], bbox["x"], bbox["h"], bbox["w"]], canvas_shape_wh=canvas_shape, current_zoom=old_zoom)
                if new_zoom is not None:
                    viewer.camera.zoom = float(new_zoom)
                rect=np.asarray(bbox["vertices"], dtype=float); lname='verification_region_bbox'
                lyr = viewer.layers[lname] if lname in viewer.layers else None
                if lyr is None:
                    lyr = viewer.add_shapes([rect], shape_type='polygon', name=lname, edge_color='cyan', face_color=[0,0,0,0], edge_width=6, opacity=1.0)
                else:
                    lyr.data=[rect]
                lyr.visible = True
                lyr.opacity = 1.0
                lyr.edge_color = 'cyan'
                lyr.face_color = [0,0,0,0]
                lyr.edge_width = 6
                try:
                    viewer.layers.move(viewer.layers.index(lyr), len(viewer.layers)-1)
                except Exception:
                    pass
                jump_summary["text"] = f"region_id={rr.get('region_id')} old_zoom={old_zoom:.3f} new_zoom={viewer.camera.zoom:.3f}"
                verification_status.setText(f"Jump region_id={rr.get('region_id')} source_type={rr.get('source_type')} bbox_working_yxhw={rr.get('bbox_working_yxhw')} working_shape_hw={rr.get('working_shape_hw')} display_shape_hw={ds} bbox_display_yxhw={[bbox['y'],bbox['x'],bbox['h'],bbox['w']]} center_display_yx={list(center_used[-2:])} canvas_shape_wh={canvas_shape} old_zoom={old_zoom:.3f} new_zoom={float(viewer.camera.zoom):.3f}")
                _refresh_table()
            table.itemSelectionChanged.connect(_on_select)
            table.cellDoubleClicked.connect(lambda *_: _jump())
            jump_btn.clicked.connect(_jump); cf.currentTextChanged.connect(lambda *_: _refresh_table()); inf.currentTextChanged.connect(lambda *_: _refresh_table()); sf.currentTextChanged.connect(lambda *_: _refresh_table())
            verification_viewer_state["refresh"] = _refresh_table
        verification_viewer_state["regions"] = regions
        verification_viewer_state["report"].setText(f"Selected report: {entry.get('project_tag')} | {resolved.as_posix()}")
        verification_viewer_state["refresh"]()
        verification_status.setText(f"Verification results viewer loaded: {len(regions)} regions")

    def _append_output()->None:
        text=bytes(project_runner.readAllStandardOutput()).decode('utf-8',errors='replace')
        if text: project_log.appendPlainText(text.rstrip('\n'))

    def _run_project()->None:
        summary=discover_project_cases(config=config, annotations_dir=gui_annotations_dir, raw_dir=gui_raw_dir)
        if len(summary.get('included_ready_cases',[])) < 2:
            workflow_status.setText('Project status: blocked; please save annotations so at least 2 cases are READY.')
            return
        enc_id=_selected_encoder_id(); enc_meta=_selected_encoder_meta(); enc_display=str(enc_meta.get("display_name") or enc_id)
        tag=default_project_tag(enc_id); latest_project_tag.setText(f'Latest generated/shared project tag: {tag}')
        cmd=build_stage1_project_command(config_path=Path(config.get('__config_path__','config/base.yaml')),project_tag=tag,raw_dir=gui_raw_dir,annotations_dir=gui_annotations_dir,outputs_root=gui_outputs_root,models_root=gui_models_root,embedding_encoder_id=enc_id)
        workflow_status.setText(f"Project status: queued. Embedding encoder: {enc_display} ({enc_id})")
        project_log.appendPlainText(f"Embedding encoder: {enc_display} ({enc_id})")
        project_log.appendPlainText(f"$ {shlex.join(cmd)}")
        project_runner.setProgram(cmd[0]); project_runner.setArguments(cmd[1:]); project_runner.setWorkingDirectory(str(REPO_ROOT)); project_runner.start()

    def _started()->None:
        run_btn.setEnabled(False); show_log_toggle.setChecked(True); workflow_status.setText('Project status: running shared Stage 1...')
    def _finished(exit_code:int, exit_status:QProcess.ExitStatus)->None:
        _append_output(); run_btn.setEnabled(True)
        if exit_status == QProcess.NormalExit and int(exit_code)==0:
            workflow_status.setText('Project status: success'); tag=latest_project_tag.text().split(':',1)[-1].strip(); _refresh_workflow(select_tag=tag)
        else: workflow_status.setText(f'Project status: failed (exit_code={exit_code})')

    def _on_project_changed(index:int)->None:
        if index<0: return
        e=wf_state['project_entries'][index]; selected_project_tag.setText(f"Selected project summary tag: {e.get('project_tag')} | agg F1 {format_display_float(e.get('aggregate_f1'))}"); _load_preview(Path(e.get('training_summary_md','')))
    def _on_image_changed(index:int)->None:
        if index<0: return
        e=wf_state['image_entries'][index]; selected_image_tag.setText(f"Selected current-image shared-report tag: {e.get('project_tag')} | shared F1 {format_display_float(e.get('f1'))}"); _load_preview(Path(e.get('report_summary_md',''))); _sync_verification_overlay()
    def _on_toggle(checked:bool)->None: project_log.setVisible(bool(checked))
    def _on_verification_toggle(_:bool)->None: _sync_verification_overlay()

    project_runner.readyReadStandardOutput.connect(_append_output); project_runner.started.connect(_started); project_runner.finished.connect(_finished)
    encoder_combo.currentIndexChanged.connect(lambda *_: _refresh_encoder_hint())
    run_btn.clicked.connect(_run_project); open_verification_viewer_btn.clicked.connect(_open_verification_results_viewer); project_combo.currentIndexChanged.connect(_on_project_changed); image_combo.currentIndexChanged.connect(_on_image_changed); show_log_toggle.toggled.connect(_on_toggle); show_verification_toggle.toggled.connect(_on_verification_toggle)
    encoder_combo.setCurrentIndex(max(0, encoder_combo.findData(default_encoder_id)))
    _refresh_encoder_hint()
    _refresh_workflow()
    save_state = {"in_progress": False}

    logging.info(
        "Class encoding for exported scribble_labels mask: %s",
        {k: v for k, v in class_mapping.items() if k != "Unlabeled"},
    )

    @viewer.bind_key("s")
    def _save(_: Any) -> None:
        if save_state["in_progress"]:
            save_status_label.value = "Save status: Save already in progress... please wait"
            viewer.status = "Save already in progress... please wait"
            logging.info("Ignoring repeated save request while save is in progress for image_id=%s", image_id)
            return
        save_state["in_progress"] = True
        started_at = time.monotonic()
        save_status_label.value = "Save status: Saving... please wait"
        viewer.window._qt_window.repaint()
        QApplication.processEvents()
        readiness_label.value = "Readiness: Checking after save..."
        next_action_label.value = "Next action: Wait for save and readiness result"
        viewer.status = "Saving annotation artifacts... please wait"

        classes = polygon_layer.properties.get("class_name", np.array([], dtype=object)).tolist()
        polygons: list[dict[str, Any]] = []
        for vertices, class_name in zip(polygon_layer.data, classes, strict=False):
            polygons.append({"class_name": str(class_name), "vertices": np.asarray(vertices, dtype=float).tolist()})

        try:
            export_annotation_artifacts(
                polygons,
                paths,
                image_shape=image_shape,
                image_id=image_id,
                annotator=annotator,
                brush_size=brush_size,
                notes=notes,
                uncertainty_comment=uncertainty_comment,
                class_mapping=class_mapping,
                roi_closing_radius=roi_closing_radius,
                roi_fill_holes_area_threshold=roi_fill_holes_area_threshold,
            )
            elapsed = time.monotonic() - started_at
            readiness = compute_annotation_readiness(
                config=config,
                annotations_dir=paths.metadata_path.parent,
                image_id=image_id,
            )
            save_status_label.value = f"Save status: Saved successfully in {elapsed:.2f}s"
            readiness_label.value = f"Readiness: {readiness.status_code} ({readiness.status_label})"
            next_action_label.value = f"Next action: {readiness.next_action}"
            viewer.status = (
                f"Saved in {elapsed:.2f}s. "
                f"Readiness={readiness.status_code}. {readiness.summary_message} {readiness.next_action}"
            )
            logging.info(
                "Saved annotation artifacts for image_id=%s to %s. readiness=%s summary=%s next_action=%s notes=%s",
                image_id,
                paths,
                readiness.status_code,
                readiness.summary_message,
                readiness.next_action,
                readiness.notes,
            )
            QApplication.processEvents()
        except Exception as exc:
            save_status_label.value = f"Save status: ERROR ({exc})"
            readiness_label.value = "Readiness: ERROR (save failed)"
            next_action_label.value = "Next action: Resolve error and press 's' again"
            viewer.status = f"Save failed: {exc}"
            logging.exception("Failed to save annotation artifacts for image_id=%s: %s", image_id, exc)
        finally:
            save_state["in_progress"] = False

    logging.info(
        "Use 'Polygon Class Controls' dock to choose active class before drawing. Select polygon(s) and click apply to edit class, then press 's' to save."
    )
    napari.run()


def run_headless_smoke_test() -> None:
    """Run minimal helper smoke test for deterministic artifact export."""
    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        image_id = "synthetic_001"
        image_shape = (12, 10)
        paths = build_annotation_paths(output_dir, image_id)

        polygons = [
            {"class_name": "Positive_Tumor", "vertices": [[2, 2], [2, 6], [6, 6], [6, 2]]},
            {"class_name": "Negative_Tumor", "vertices": [[7, 2], [7, 5], [10, 5], [10, 2]]},
            {"class_name": "NonTumor", "vertices": [[2, 7], [2, 9], [5, 9], [5, 7]]},
            {"class_name": "Ignore", "vertices": [[6, 7], [6, 9], [9, 9], [9, 7]]},
        ]

        metadata = export_annotation_artifacts(
            polygons,
            paths,
            image_shape=image_shape,
            image_id=image_id,
            annotator="smoke_tester",
            brush_size=9,
            notes="smoke",
            uncertainty_comment="none",
            class_mapping=class_mapping_default,
            roi_closing_radius=0,
            roi_fill_holes_area_threshold=0,
        )

        roi_reloaded = load_mask_png(paths.roi_mask_path, image_shape)
        scribble_reloaded = load_mask_png(paths.scribble_path, image_shape)

        assert np.any(roi_reloaded == 1), "ROI should be generated from tumor polygons"
        assert np.any(scribble_reloaded == class_mapping_default["Positive_Tumor"])
        assert np.any(scribble_reloaded == class_mapping_default["NonTumor"])
        assert metadata["classes"] == ["Positive_Tumor", "Negative_Tumor", "NonTumor", "Ignore"]

        logging.info("Headless smoke test passed. Artifacts written under temp dir: %s", output_dir)


def run_synthetic_save_load_test() -> None:
    """Verify polygon class metadata round-trip and derived mask artifacts."""
    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        image_id = "synthetic_roundtrip"
        image_shape = (32, 24)
        paths = build_annotation_paths(output_dir, image_id)
        class_choices = np.array([k for k in class_mapping_default if k != "Unlabeled"], dtype=object)
        shapes_kwargs = build_shapes_layer_kwargs([], [], class_choices)
        assert "property_choices" in shapes_kwargs, "Shapes kwargs must set property_choices at layer creation time"
        assert "class_name" in shapes_kwargs["property_choices"], "Missing class_name property choices"

        polygons = [
            {"class_name": "Positive_Tumor", "vertices": [[3, 3], [3, 12], [12, 12], [12, 3]]},
            {"class_name": "Ignore", "vertices": [[18, 14], [18, 20], [26, 20], [26, 14]]},
        ]

        export_annotation_artifacts(
            polygons,
            paths,
            image_shape=image_shape,
            image_id=image_id,
            annotator="synthetic_tester",
            brush_size=8,
            notes="roundtrip",
            uncertainty_comment="none",
            class_mapping=class_mapping_default,
            roi_closing_radius=1,
            roi_fill_holes_area_threshold=16,
        )

        loaded_polygons = load_previous_polygons(paths, class_mapping_default)
        assert len(loaded_polygons) == len(polygons), "Polygon count mismatch after metadata reload"
        assert [p["class_name"] for p in loaded_polygons] == [p["class_name"] for p in polygons]
        assert paths.roi_mask_path.exists(), "Derived ROI mask missing"
        assert paths.scribble_path.exists(), "Derived class mask missing"

        roi = load_mask_png(paths.roi_mask_path, image_shape)
        class_mask = load_mask_png(paths.scribble_path, image_shape)
        assert np.any(roi == 1), "Derived ROI mask is empty"
        assert np.any(class_mask == class_mapping_default["Positive_Tumor"]), "Class mask missing Positive_Tumor label"
        assert np.any(class_mask == class_mapping_default["Ignore"]), "Class mask missing Ignore label"
        observed_ids = set(int(v) for v in np.unique(class_mask))
        expected_ids = {0, class_mapping_default["Positive_Tumor"], class_mapping_default["Ignore"]}
        assert observed_ids == expected_ids, f"Unexpected scribble label ids. observed={observed_ids}, expected={expected_ids}"

        logging.info("Synthetic save/load test passed. Artifacts written under temp dir: %s", output_dir)


def run_svs_load_path_test() -> None:
    """Verify .svs loader gives clear fallback error or loads when supported."""
    with TemporaryDirectory() as tmp:
        svs_path = Path(tmp) / "synthetic.svs"
        svs_path.write_bytes(b"not-a-real-svs")
        try:
            _ = read_image(svs_path)
        except RuntimeError as exc:
            message = str(exc)
            assert "SVS support requires openslide-python" in message or "Failed to read SVS slide" in message
            logging.info("SVS load-path test passed with expected RuntimeError: %s", message)
            return
        raise AssertionError("SVS load-path test expected a RuntimeError for synthetic .svs file")


def main() -> None:
    """Run Stage 1 Napari annotator CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()

    config = load_config(args.config)
    config["__config_path__"] = str(args.config)

    if args.headless_smoke_test:
        run_headless_smoke_test()
        return
    if args.synthetic_save_load_test:
        run_synthetic_save_load_test()
        return
    if args.svs_load_test:
        run_svs_load_path_test()
        return
    if args.class_ui_init_check:
        run_class_ui_init_check()
        run_class_event_compatibility_check()
        return

    class_mapping = class_mapping_default.copy()
    image_path = resolve_image_path(args.image_id, args.image_path, args.input)
    image_id = derive_image_id(image_path, args.image_id)
    image = read_image(image_path)
    paths = build_annotation_paths(args.output_dir, image_id)

    logging.info("Launching annotator for image_id=%s (%s)", image_id, image_path)
    launch_napari_app(
        image,
        image_id,
        paths,
        config=config,
        annotator=args.annotator,
        brush_size=args.brush_size,
        notes=args.notes,
        uncertainty_comment=args.uncertainty_comment,
        class_mapping=class_mapping,
        roi_closing_radius=args.roi_closing_radius,
        roi_fill_holes_area_threshold=args.roi_fill_holes_area_threshold,
    )


if __name__ == "__main__":
    main()
