"""Stage 1 manifest builder + lightweight QC pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_MANIFEST_COLUMNS = [
    "image_id",
    "source_url",
    "marker",
    "stain",
    "cancer_type",
    "download_date",
    "license_checked",
    "roi_status",
    "annotation_status",
    "qc_status",
]

MANIFEST_EXTRA_COLUMNS = [
    "image_path",
    "file_size_bytes",
    "last_modified_utc",
    "width_px",
    "height_px",
    "qc_notes",
    "qc_error",
    "qc_last_run_utc",
]

QC_REPORT_COLUMNS = [
    "image_id",
    "image_path",
    "readable",
    "width_px",
    "height_px",
    "tissue_fraction",
    "blank_fraction",
    "blur_score",
    "pen_mark_fraction",
    "fold_tear_fraction",
    "qc_status",
    "qc_notes",
    "qc_error",
]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".pgm", ".svs"}


@dataclass
class ImageData:
    """In-memory image payload used for simple QC checks."""

    width: int
    height: int
    channels: int
    data: list[int]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for manifest and QC workflow."""
    parser = argparse.ArgumentParser(description="Build/update image manifest and run lightweight QC.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw image files when not fully specified by manifest.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Optional existing manifest to preserve curated metadata while refreshing QC-related fields.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/qc"), help="QC output directory.")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/image_manifest.csv"),
        help="Manifest CSV output path.",
    )
    parser.add_argument("--force", action="store_true", help="Allow deterministic overwrite of existing outputs.")
    parser.add_argument("--thumb-size", type=int, default=256, help="Longest edge size for generated thumbnails.")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config if available; fail clearly when missing."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ModuleNotFoundError:
        logging.warning("pyyaml not installed; proceeding with config path check only for bootstrap smoke runs.")
        return {}

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config did not parse into a dictionary.")
    return config


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write deterministic CSV with fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def discover_image_files(raw_dir: Path) -> list[Path]:
    """Discover candidate image files in deterministic order."""
    if not raw_dir.exists():
        return []
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files, key=lambda p: (p.name.lower(), p.name))


def build_discovered_rows(raw_files: list[Path]) -> list[dict[str, str]]:
    """Construct default manifest rows for discovered images."""
    rows: list[dict[str, str]] = []
    for raw_path in raw_files:
        image_id = raw_path.stem
        rows.append(
            {
                "image_id": image_id,
                "source_url": "",
                "marker": "PD-L1",
                "stain": "IHC",
                "cancer_type": "",
                "download_date": "",
                "license_checked": "",
                "roi_status": "not_started",
                "annotation_status": "not_started",
                "qc_status": "pending",
                "image_path": str(raw_path.as_posix()),
            }
        )
    return rows


def resolve_manifest_image_path(existing_path: str, raw_dir: Path) -> Path | None:
    """
    Resolve a manifest image path without duplicating raw_dir prefixes.

    Priority:
    1. keep absolute paths as-is,
    2. keep already-valid repo/project-relative paths as-is,
    3. try raw_dir / candidate for bare filenames or raw-relative paths,
    4. otherwise preserve the original candidate path.
    """
    if not existing_path:
        return None

    candidate = Path(existing_path)

    if candidate.is_absolute() or candidate.exists():
        return candidate

    prefixed = raw_dir / candidate
    if prefixed.exists():
        return prefixed

    return candidate


def normalize_manifest_rows(
    input_manifest_rows: list[dict[str, str]],
    raw_files: list[Path],
    raw_dir: Path,
) -> list[dict[str, str]]:
    """Create deterministic manifest rows preserving curated metadata when provided."""
    raw_by_stem = {p.stem: p for p in raw_files}
    raw_by_name = {p.name: p for p in raw_files}

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in input_manifest_rows:
        item = {col: row.get(col, "") for col in BASE_MANIFEST_COLUMNS}
        image_id = item.get("image_id", "").strip()

        existing_path = row.get("image_path", "").strip()
        resolved_path = resolve_manifest_image_path(existing_path, raw_dir)

        if resolved_path is None and image_id in raw_by_stem:
            resolved_path = raw_by_stem[image_id]

        if resolved_path is None and row.get("filename"):
            maybe = raw_by_name.get(row["filename"])
            if maybe is not None:
                resolved_path = maybe

        if not image_id and resolved_path is not None:
            image_id = resolved_path.stem
            item["image_id"] = image_id

        item["image_path"] = str(resolved_path.as_posix()) if resolved_path is not None else existing_path

        key = (item.get("image_id", ""), item.get("image_path", ""))
        if key not in seen:
            rows.append(item)
            seen.add(key)

    existing_ids = {r.get("image_id", "") for r in rows}
    for discovered in build_discovered_rows(raw_files):
        if discovered["image_id"] not in existing_ids:
            rows.append(discovered)

    rows.sort(key=lambda r: (r.get("image_id", ""), r.get("image_path", "")))
    return rows


def read_ppm(path: Path) -> ImageData:
    """Read PPM/PGM (P3/P6/P2/P5) images without third-party dependencies."""
    data = path.read_bytes()
    idx = 0

    def read_token() -> str:
        nonlocal idx
        while idx < len(data) and chr(data[idx]).isspace():
            idx += 1
        if idx < len(data) and data[idx] == ord("#"):
            while idx < len(data) and data[idx] not in (10, 13):
                idx += 1
            return read_token()
        start = idx
        while idx < len(data) and not chr(data[idx]).isspace():
            idx += 1
        return data[start:idx].decode("ascii")

    magic = read_token()
    if magic not in {"P3", "P6", "P2", "P5"}:
        raise ValueError(f"Unsupported PPM/PGM magic: {magic}")

    width = int(read_token())
    height = int(read_token())
    max_val = int(read_token())
    if max_val <= 0:
        raise ValueError("Invalid max value in PPM/PGM header")

    channels = 3 if magic in {"P3", "P6"} else 1
    expected = width * height * channels

    if magic in {"P3", "P2"}:
        values: list[int] = []
        while len(values) < expected:
            values.append(int(read_token()))
    else:
        while idx < len(data) and chr(data[idx]).isspace():
            idx += 1
        payload = data[idx:]
        if len(payload) < expected:
            raise ValueError("PPM/PGM payload is shorter than expected")
        values = list(payload[:expected])

    if max_val != 255:
        values = [int(round(v * 255.0 / max_val)) for v in values]

    return ImageData(width=width, height=height, channels=channels, data=values)


def load_image(path: Path) -> ImageData:
    """Load image data using Pillow (if available) or fallback PPM reader."""
    suffix = path.suffix.lower()

    if suffix in {".ppm", ".pgm"}:
        return read_ppm(path)

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for non-PPM images in this environment. Use .ppm for smoke tests or install deps."
        ) from exc

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        values = list(rgb.tobytes())
    return ImageData(width=width, height=height, channels=3, data=values)


def grayscale_values(image: ImageData) -> list[float]:
    """Convert image channels to grayscale values in [0, 255]."""
    if image.channels == 1:
        return [float(v) for v in image.data]

    gray: list[float] = []
    for i in range(0, len(image.data), 3):
        r, g, b = image.data[i], image.data[i + 1], image.data[i + 2]
        gray.append(0.299 * r + 0.587 * g + 0.114 * b)
    return gray


def estimate_tissue_and_blank(gray: list[float]) -> tuple[float, float]:
    """Estimate coarse tissue and blank fractions from grayscale intensity."""
    if not gray:
        return 0.0, 0.0
    tissue_count = sum(1 for g in gray if g < 220.0)
    blank_count = sum(1 for g in gray if g > 245.0)
    total = float(len(gray))
    return tissue_count / total, blank_count / total


def estimate_blur(gray: list[float], width: int, height: int) -> float:
    """Estimate blur with a simple horizontal gradient magnitude mean."""
    if width < 2 or height < 1:
        return 0.0
    grad_sum = 0.0
    count = 0
    for y in range(height):
        row_start = y * width
        for x in range(width - 1):
            g1 = gray[row_start + x]
            g2 = gray[row_start + x + 1]
            grad_sum += abs(g2 - g1)
            count += 1
    return grad_sum / count if count else 0.0


def estimate_pen_marks(image: ImageData) -> float:
    """Estimate pen-mark fraction via coarse blue/green ink heuristic."""
    if image.channels != 3:
        return 0.0
    pen_like = 0
    total = image.width * image.height
    for i in range(0, len(image.data), 3):
        r, g, b = image.data[i], image.data[i + 1], image.data[i + 2]
        is_blue = b > 150 and b > (r + 30) and b > (g + 20)
        is_green = g > 150 and g > (r + 25) and g > (b + 10)
        if is_blue or is_green:
            pen_like += 1
    return pen_like / total if total else 0.0


def estimate_fold_tear(gray: list[float]) -> float:
    """Estimate fold/tear-like bright linear fraction (very coarse heuristic)."""
    if not gray:
        return 0.0
    bright = sum(1 for g in gray if g > 250.0)
    return bright / float(len(gray))


def resize_nearest(image: ImageData, max_edge: int) -> ImageData:
    """Resize image to thumbnail using nearest-neighbor sampling."""
    scale = min(1.0, max_edge / max(image.width, image.height))
    out_w = max(1, int(round(image.width * scale)))
    out_h = max(1, int(round(image.height * scale)))

    out_data: list[int] = []
    for oy in range(out_h):
        sy = min(image.height - 1, int(oy / scale)) if scale > 0 else oy
        for ox in range(out_w):
            sx = min(image.width - 1, int(ox / scale)) if scale > 0 else ox
            idx = (sy * image.width + sx) * image.channels
            for c in range(image.channels):
                out_data.append(image.data[idx + c])

    return ImageData(width=out_w, height=out_h, channels=image.channels, data=out_data)


def make_tissue_preview(gray: list[float], width: int, height: int) -> ImageData:
    """Build a binary tissue preview image as grayscale PGM-style payload."""
    payload = [255 if g < 220.0 else 20 for g in gray]
    return ImageData(width=width, height=height, channels=1, data=payload)


def write_ppm(path: Path, image: ImageData) -> None:
    """Write image as PPM/PGM depending on channel count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.channels == 1:
        header = f"P5\n{image.width} {image.height}\n255\n".encode("ascii")
        payload = bytes(max(0, min(255, int(v))) for v in image.data)
    else:
        header = f"P6\n{image.width} {image.height}\n255\n".encode("ascii")
        payload = bytes(max(0, min(255, int(v))) for v in image.data)
    path.write_bytes(header + payload)


def analyze_image(path: Path, thumb_size: int) -> tuple[dict[str, Any], ImageData | None, ImageData | None]:
    """Run lightweight QC analysis for a single image path."""
    if path.suffix.lower() == ".svs":
        return analyze_svs_metadata(path)

    qc = {
        "readable": "false",
        "width_px": "",
        "height_px": "",
        "tissue_fraction": "",
        "blank_fraction": "",
        "blur_score": "",
        "pen_mark_fraction": "",
        "fold_tear_fraction": "",
        "qc_status": "failed_load",
        "qc_notes": "",
        "qc_error": "",
    }

    try:
        image = load_image(path)
        gray = grayscale_values(image)
        tissue_fraction, blank_fraction = estimate_tissue_and_blank(gray)
        blur_score = estimate_blur(gray, image.width, image.height)
        pen_marks = estimate_pen_marks(image)
        fold_tear = estimate_fold_tear(gray)

        notes = []
        if blank_fraction > 0.6:
            notes.append("high_blank_background")
        if blur_score < 3.0:
            notes.append("potential_blur")
        if pen_marks > 0.01:
            notes.append("possible_pen_marks")
        if fold_tear > 0.03:
            notes.append("possible_fold_or_tear")

        qc.update(
            {
                "readable": "true",
                "width_px": str(image.width),
                "height_px": str(image.height),
                "tissue_fraction": f"{tissue_fraction:.4f}",
                "blank_fraction": f"{blank_fraction:.4f}",
                "blur_score": f"{blur_score:.4f}",
                "pen_mark_fraction": f"{pen_marks:.4f}",
                "fold_tear_fraction": f"{fold_tear:.4f}",
                "qc_status": "needs_review" if notes else "pass",
                "qc_notes": ";".join(notes),
            }
        )

        thumb = resize_nearest(image, max_edge=thumb_size)
        tissue_preview = make_tissue_preview(gray, image.width, image.height)
        tissue_preview = resize_nearest(tissue_preview, max_edge=thumb_size)
        return qc, thumb, tissue_preview
    except Exception as exc:  # noqa: BLE001
        qc["qc_error"] = str(exc)
        qc["qc_notes"] = "unreadable_or_unsupported"
        return qc, None, None


def try_read_svs_dimensions(path: Path) -> tuple[int | None, int | None, str]:
    """Return SVS level-0 dimensions when openslide runtime is available."""
    if importlib.util.find_spec("openslide") is None:
        return None, None, "openslide_python_not_installed"

    try:
        import openslide  # type: ignore

        with openslide.OpenSlide(str(path)) as slide:
            width, height = slide.level_dimensions[0]
        return int(width), int(height), ""
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def analyze_svs_metadata(path: Path) -> tuple[dict[str, Any], ImageData | None, ImageData | None]:
    """Run metadata-only QC path for SVS files without raster-level QC heuristics."""
    qc = {
        "readable": "false",
        "width_px": "",
        "height_px": "",
        "tissue_fraction": "",
        "blank_fraction": "",
        "blur_score": "",
        "pen_mark_fraction": "",
        "fold_tear_fraction": "",
        "qc_status": "svs_metadata_only_no_openslide",
        "qc_notes": "svs_metadata_qc_only;openslide_unavailable",
        "qc_error": "",
    }

    if not path.exists():
        qc["qc_status"] = "failed_load"
        qc["qc_notes"] = "missing_file"
        qc["qc_error"] = "SVS file does not exist"
        return qc, None, None

    width, height, err = try_read_svs_dimensions(path)
    if width is not None and height is not None:
        qc["readable"] = "true"
        qc["width_px"] = str(width)
        qc["height_px"] = str(height)
        qc["qc_status"] = "svs_metadata_only"
        qc["qc_notes"] = "svs_metadata_qc_only;no_full_raster_qc"
        return qc, None, None

    if err and err != "openslide_python_not_installed":
        qc["qc_notes"] = "svs_metadata_qc_only;openslide_runtime_unavailable_or_slide_read_failed"
        qc["qc_error"] = err

    return qc, None, None


def ensure_writable_file(path: Path, force: bool) -> None:
    """Ensure output file obeys force/overwrite contract."""
    if path.exists() and not force:
        raise FileExistsError(f"Output exists: {path}. Re-run with --force to overwrite deterministically.")


def prepare_thumbnail_dir(path: Path, force: bool) -> None:
    """Prepare thumbnail directory with deterministic overwrite behavior."""
    if path.exists():
        has_contents = any(path.iterdir())
        if has_contents and not force:
            raise FileExistsError(
                f"Thumbnail directory already contains files: {path}. Re-run with --force to overwrite deterministically."
            )
        if force:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run Stage 1 manifest builder + lightweight QC workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()

    _ = load_config(args.config)

    output_dir = args.output_dir
    qc_report_path = output_dir / "qc_report.csv"
    thumb_dir = output_dir / "qc_thumbnails"
    manifest_path = args.manifest_out

    ensure_writable_file(qc_report_path, args.force)
    ensure_writable_file(manifest_path, args.force)
    prepare_thumbnail_dir(thumb_dir, args.force)

    raw_files = discover_image_files(args.input)

    input_manifest_rows: list[dict[str, str]] = []
    if args.input_manifest:
        if not args.input_manifest.exists():
            raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")
        input_manifest_rows = read_csv_rows(args.input_manifest)

    manifest_rows = normalize_manifest_rows(
        input_manifest_rows=input_manifest_rows,
        raw_files=raw_files,
        raw_dir=args.input,
    )

    if not manifest_rows:
        logging.warning("No manifest rows resolved from input directory or input manifest.")

    qc_rows: list[dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for row in manifest_rows:
        image_id = row.get("image_id", "")
        image_path = row.get("image_path", "")
        path_obj = Path(image_path) if image_path else args.input / f"{image_id}.png"

        qc_metrics, thumb, tissue_thumb = analyze_image(path_obj, thumb_size=args.thumb_size)

        qc_row = {"image_id": image_id, "image_path": str(path_obj.as_posix())}
        qc_row.update(qc_metrics)
        qc_rows.append(qc_row)

        row["image_path"] = str(path_obj.as_posix())
        row["file_size_bytes"] = str(path_obj.stat().st_size) if path_obj.exists() else ""
        row["last_modified_utc"] = (
            datetime.fromtimestamp(path_obj.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
            if path_obj.exists()
            else ""
        )
        row["width_px"] = qc_metrics.get("width_px", "")
        row["height_px"] = qc_metrics.get("height_px", "")
        row["qc_status"] = qc_metrics.get("qc_status", row.get("qc_status", "pending"))
        row["qc_notes"] = qc_metrics.get("qc_notes", "")
        row["qc_error"] = qc_metrics.get("qc_error", "")
        row["qc_last_run_utc"] = now_utc

        if thumb is not None and tissue_thumb is not None:
            write_ppm(thumb_dir / f"{image_id}_original_thumb.ppm", thumb)
            write_ppm(thumb_dir / f"{image_id}_tissue_thumb.pgm", tissue_thumb)

            sidecar = {
                "image_id": image_id,
                "image_path": str(path_obj.as_posix()),
                "qc_status": row.get("qc_status", ""),
                "qc_notes": row.get("qc_notes", ""),
            }
            (thumb_dir / f"{image_id}_qc_sidecar.json").write_text(
                json.dumps(sidecar, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    manifest_columns = BASE_MANIFEST_COLUMNS + MANIFEST_EXTRA_COLUMNS
    write_csv(manifest_path, manifest_columns, manifest_rows)
    write_csv(qc_report_path, QC_REPORT_COLUMNS, qc_rows)

    logging.info("Wrote manifest: %s", manifest_path)
    logging.info("Wrote QC report: %s", qc_report_path)
    logging.info("Wrote thumbnails (readable images only) to: %s", thumb_dir)


if __name__ == "__main__":
    main()
