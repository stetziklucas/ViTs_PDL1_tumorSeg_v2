"""Generate report artifacts (markdown/json/pdf) for a single Stage 1 image run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.report_metrics import compute_metrics_from_paths, load_json
from scripts.supervision_audit import audit_supervision
from display_format import format_display_float
from verification_overlay import generate_verification_overlay

REQUIRED_ARTIFACT_BASENAMES = {
    "metrics": "metrics.json",
    "overlay": "overlay.png",
    "positive_mask": "positive_mask.png",
    "pixel_prob_map": "pixel_prob_map.png",
    "tile_prob_map": "tile_prob_map.png",
}

OPTIONAL_ARTIFACT_BASENAMES = {
    "tile_cv_metrics": "tile_cv_metrics.json",
}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Stage 1 report generation."""
    parser = argparse.ArgumentParser(description="Generate one-page Stage 1 report artifacts from existing outputs.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=True, help="Image identifier to summarize.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Annotation directory containing scribbles/.",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("outputs/maps"),
        help="Default directory for probability maps when per-map dirs are not provided.",
    )
    parser.add_argument(
        "--tile-maps-dir",
        type=Path,
        default=None,
        help="Optional directory containing tile_prob_map.png (defaults to --maps-dir).",
    )
    parser.add_argument(
        "--pixel-maps-dir",
        type=Path,
        default=None,
        help="Optional directory containing pixel_prob_map.png (defaults to --maps-dir).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("outputs/masks"),
        help="Directory containing positive_mask.png.",
    )
    parser.add_argument(
        "--overlays-dir",
        type=Path,
        default=Path("outputs/overlays"),
        help="Directory containing overlay.png.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory containing metrics.json and report outputs.",
    )
    parser.add_argument(
        "--tile-model-dir",
        type=Path,
        default=Path("models/tile_head"),
        help="Directory that may contain optional tile_cv_metrics.json.",
    )
    parser.add_argument(
        "--model-scope",
        default="single_image_model",
        choices=["single_image_model", "shared_project_model"],
        help="Model scope metadata for report summaries.",
    )
    parser.add_argument("--shared-model-tag", default=None, help="Optional shared model tag metadata.")
    parser.add_argument("--training-image-count", type=int, default=None, help="Optional shared-training image count.")
    parser.add_argument(
        "--included-training-aliases",
        default=None,
        help="Optional comma-separated aliases used in shared training.",
    )
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


def require_artifacts(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve and validate required artifact paths."""
    tile_maps_dir = args.tile_maps_dir or args.maps_dir
    pixel_maps_dir = args.pixel_maps_dir or args.maps_dir

    artifact_paths = {
        "metrics": args.reports_dir / REQUIRED_ARTIFACT_BASENAMES["metrics"],
        "overlay": args.overlays_dir / REQUIRED_ARTIFACT_BASENAMES["overlay"],
        "positive_mask": args.masks_dir / REQUIRED_ARTIFACT_BASENAMES["positive_mask"],
        "pixel_prob_map": pixel_maps_dir / REQUIRED_ARTIFACT_BASENAMES["pixel_prob_map"],
        "tile_prob_map": tile_maps_dir / REQUIRED_ARTIFACT_BASENAMES["tile_prob_map"],
    }
    missing = [f"{name}={path}" for name, path in artifact_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required report input artifacts:\n" + "\n".join(missing))
    return artifact_paths


def optional_artifacts(args: argparse.Namespace) -> dict[str, Path | None]:
    """Resolve optional artifact paths."""
    tile_cv = args.tile_model_dir / OPTIONAL_ARTIFACT_BASENAMES["tile_cv_metrics"]
    return {"tile_cv_metrics": tile_cv if tile_cv.exists() else None}


def _as_image(path: Path) -> np.ndarray:
    import PIL.Image

    return np.asarray(PIL.Image.open(path))


def _fmt_shape(shape_hw: Any) -> str:
    if isinstance(shape_hw, list) and len(shape_hw) == 2:
        return f"{int(shape_hw[0])} x {int(shape_hw[1])} (H x W)"
    return "unavailable"


def _format_metrics_block(metrics: dict[str, Any]) -> list[str]:
    return [
        f"false_positive_px: {int(metrics['false_positive_px'])}",
        f"false_negative_px: {int(metrics['false_negative_px'])}",
        f"precision: {format_display_float(metrics['precision'])}",
        f"sensitivity: {format_display_float(metrics['sensitivity'])}",
        f"f1: {format_display_float(metrics['f1'])}",
        f"training_log_loss_total: {format_display_float(metrics['training_log_loss_total'])}",
    ]


def derive_operator_summary(development_metrics: dict[str, Any]) -> dict[str, str]:
    """Derive concise operator-facing result language from fixed metric counts."""
    fp = int(development_metrics["false_positive_px"])
    fn = int(development_metrics["false_negative_px"])
    if fp == 0 and fn > 0:
        return {
            "result_description": "Tumor-enriched PD-L1-positive mask in working image space",
            "evaluation_scope": "Annotated-region development metrics only; not whole-slide validation",
            "error_pattern": "Conservative / false-negative dominant",
            "next_review_focus": "Review undercalled annotated positive tumor regions and add representative positive supervision.",
        }
    if fn == 0 and fp > 0:
        return {
            "result_description": "Tumor-enriched PD-L1-positive mask in working image space",
            "evaluation_scope": "Annotated-region development metrics only; not whole-slide validation",
            "error_pattern": "False-positive dominant",
            "next_review_focus": "Review overcalled annotated negative/non-tumor regions and strengthen negative supervision.",
        }
    if fn > 0 and fp > 0:
        return {
            "result_description": "Tumor-enriched PD-L1-positive mask in working image space",
            "evaluation_scope": "Annotated-region development metrics only; not whole-slide validation",
            "error_pattern": "Mixed false-positive and false-negative errors",
            "next_review_focus": "Review both undercalled and overcalled annotated regions before broader reruns.",
        }
    return {
        "result_description": "Tumor-enriched PD-L1-positive mask in working image space",
        "evaluation_scope": "Annotated-region development metrics only; not whole-slide validation",
        "error_pattern": "No annotated-region errors in the evaluated region",
        "next_review_focus": "Confirm consistency on additional annotated regions before wider use.",
    }


def make_summary_lines(
    image_id: str,
    metrics_json: dict[str, Any],
    report_metrics: dict[str, Any],
    tile_cv_metrics: dict[str, Any] | None,
) -> list[str]:
    """Build text section for PDF report with metric-forward content."""
    limitations = list(metrics_json.get("limitations_or_fallbacks", []))
    if tile_cv_metrics is not None:
        lim = tile_cv_metrics.get("single_image_limitation")
        if isinstance(lim, str) and lim:
            limitations.append(f"tile_head: {lim}")

    lines = [
        f"image_id: {image_id}",
        f"source image basename: {Path(str(metrics_json.get('source_image_path', '') or '')).name or 'unavailable'}",
        f"working image shape: {_fmt_shape(metrics_json.get('working_image_shape_hw'))}",
        "",
        "annotated-region development metrics:",
        *_format_metrics_block(report_metrics),
        "",
        f"annotated_total_px: {int(report_metrics['annotated_total_px'])}",
        f"annotated_positive_px: {int(report_metrics['annotated_positive_px'])}",
        f"annotated_negative_px: {int(report_metrics['annotated_negative_px'])}",
        "",
        "space interpretation:",
        "- Metrics are computed on annotated training regions only (scribble labels).",
        "- Ignore/Unlabeled pixels are excluded from evaluation.",
        "- Outputs (maps/masks/overlay) are in selected working image space.",
        f"- metrics output note: {metrics_json.get('output_space_note', 'not provided')}",
        "",
        "key caveats / limitations:",
    ]
    if limitations:
        lines.extend([f"- {str(item)}" for item in limitations])
    else:
        lines.append("- none recorded by upstream metrics")

    return lines


def write_report_summary_markdown(output_path: Path, payload: dict[str, Any]) -> None:
    """Write CRD-friendly markdown summary for one image."""
    m = payload["development_metrics"]
    audit = payload["supervision_audit"]
    class_metrics = payload["class_metrics"]

    def _dict_lines(values: dict[str, Any]) -> list[str]:
        if not values:
            return ["- none"]
        return [f"- {k}: {v}" for k, v in values.items()]

    warning_lines = [f"- {warning}" for warning in payload.get("warnings", [])]
    if not warning_lines:
        warning_lines = ["- none"]

    lines = [
        "# Stage 1 Single-Image Development Report",
        "",
        f"image_id: {payload['image_id']}",
        f"run_tag: {payload.get('run_tag', 'unavailable')}",
        f"model_scope: {payload.get('model_scope', 'single_image_model')}",
        f"shared_model_tag: {payload.get('shared_model_tag') or 'n/a'}",
        f"training_image_count: {payload.get('training_image_count') if payload.get('training_image_count') is not None else 'n/a'}",
        f"included_training_aliases: {', '.join(payload.get('included_training_aliases', [])) if payload.get('included_training_aliases') else 'n/a'}",
        "",
        "## Operator-facing summary",
        "",
        f"- result_description: {payload['result_description']}",
        f"- evaluation_scope: {payload['evaluation_scope']}",
        f"- error_pattern: {payload['error_pattern']}",
        f"- next_review_focus: {payload['next_review_focus']}",
        "",
        "Note: Metrics below are computed on annotated training regions only, not whole-slide validation.",
        "Ignore/Unlabeled pixels are excluded.",
        "",
        "## Development metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| false_positive_px | {int(m['false_positive_px'])} |",
        f"| false_negative_px | {int(m['false_negative_px'])} |",
        f"| precision | {format_display_float(m['precision'])} |",
        f"| sensitivity | {format_display_float(m['sensitivity'])} |",
        f"| f1 | {format_display_float(m['f1'])} |",
        f"| training_log_loss_total | {format_display_float(m['training_log_loss_total'])} |",
        "",
        "## Supervision summary",
        "",
        "### Polygon counts by class",
        *_dict_lines(audit.get("polygon_counts", {})),
        "",
        "### Annotated pixel counts by class",
        *_dict_lines(audit.get("annotated_pixel_counts", {})),
        "",
        "## Tile supervision summary",
        "",
        f"- accepted_tile_count: {int(audit.get('accepted_tile_count', 0))}",
        f"- usable_tile_count: {int(audit.get('usable_tile_count', 0))}",
        f"- ignored_tile_count: {int(audit.get('ignored_tile_count', 0))}",
        f"- ignored_tile_share: {format_display_float(audit.get('ignored_tile_share', 0.0))}",
        "",
        "### Tile label counts",
        *_dict_lines(audit.get("tile_label_counts", {})),
        "",
        "### Tile label_reason counts",
        *_dict_lines(audit.get("tile_label_reason_counts", {})),
        "",
        "### Ignored tile reasons",
        *_dict_lines(audit.get("ignored_tile_reasons", {})),
        "",
        "### Selection-source counts",
        *_dict_lines(audit.get("selection_source_counts", {})),
        "",
        "## Class-specific annotated-region metrics",
        "",
        "| class | annotated_px | primary_correct_px | primary_error_px | score_name | score |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
        f"| Positive_Tumor | {int(class_metrics['Positive_Tumor']['annotated_px'])} | {int(class_metrics['Positive_Tumor']['tp_px'])} | {int(class_metrics['Positive_Tumor']['fn_px'])} | sensitivity | {format_display_float(class_metrics['Positive_Tumor']['sensitivity'])} |",
        f"| Negative_Tumor | {int(class_metrics['Negative_Tumor']['annotated_px'])} | {int(class_metrics['Negative_Tumor']['tn_px'])} | {int(class_metrics['Negative_Tumor']['fp_px'])} | specificity | {format_display_float(class_metrics['Negative_Tumor']['specificity'])} |",
        f"| NonTumor | {int(class_metrics['NonTumor']['annotated_px'])} | {int(class_metrics['NonTumor']['tn_px'])} | {int(class_metrics['NonTumor']['fp_px'])} | specificity | {format_display_float(class_metrics['NonTumor']['specificity'])} |",
        "",
        "## Additional machine-readable counts",
        "",
        f"- tp_px: {int(m['tp_px'])}",
        f"- annotated_positive_px: {int(m['annotated_positive_px'])}",
        f"- annotated_negative_px: {int(m['annotated_negative_px'])}",
        f"- annotated_total_px: {int(m['annotated_total_px'])}",
        f"- training_log_loss_mean: {format_display_float(m['training_log_loss_mean'])}",
        "",
        "## Warnings / review focus",
        "",
        *warning_lines,
        "",
        "## Working-space caveat",
        "",
        f"- {payload['working_space_note']}",
        "",
        "## Verification overlay",
        "",
        f"- verification review mask available: {'yes' if payload.get('verification_overlay_available') else 'no'}",
        f"- verification annotation labels available: {'yes' if payload.get('verification_annotation_labels_available') else 'no'}",
        "- verification review layers available: annotation labels + class-aware prediction labels",
        f"- verification mode: {payload.get('verification_overlay_mode') or 'unavailable'}",
        "- verification purpose: annotated-region development review (ROI-cropped), not whole-slide validation",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_pdf(
    output_pdf: Path,
    summary_lines: list[str],
    overlay_path: Path,
    tile_prob_map_path: Path,
    pixel_prob_map_path: Path,
    positive_mask_path: Path,
) -> None:
    """Render a deterministic one-page PDF summary."""
    overlay = _as_image(overlay_path)
    tile_prob = _as_image(tile_prob_map_path)
    pixel_prob = _as_image(pixel_prob_map_path)
    positive = _as_image(positive_mask_path)

    fig = plt.figure(figsize=(11.69, 8.27), dpi=140)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.15, hspace=0.15)

    ax_text = fig.add_subplot(grid[:, 0])
    ax_overlay = fig.add_subplot(grid[0, 1])
    ax_tile = fig.add_subplot(grid[0, 2])
    ax_pixel = fig.add_subplot(grid[1, 1])
    ax_mask = fig.add_subplot(grid[1, 2])

    ax_text.axis("off")
    ax_text.text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", fontsize=9, family="monospace")

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("overlay.png", fontsize=10)
    ax_overlay.axis("off")

    ax_tile.imshow(tile_prob, cmap="viridis")
    ax_tile.set_title("tile_prob_map.png", fontsize=10)
    ax_tile.axis("off")

    ax_pixel.imshow(pixel_prob, cmap="magma")
    ax_pixel.set_title("pixel_prob_map.png", fontsize=10)
    ax_pixel.axis("off")

    ax_mask.imshow(positive, cmap="gray")
    ax_mask.set_title("positive_mask.png", fontsize=10)
    ax_mask.axis("off")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _derive_run_tag(reports_dir: Path) -> str | None:
    stem = reports_dir.name
    prefix = "reports_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix) :]
    return None


def main() -> None:
    """Run one-page report generation from existing Stage 1 artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    required_inputs = require_artifacts(args)
    metrics_json = load_json(required_inputs["metrics"])

    metrics_image_id = str(metrics_json.get("image_id", ""))
    if metrics_image_id and metrics_image_id != args.image_id:
        raise ValueError(
            f"metrics.json image_id mismatch: expected '{args.image_id}' but found '{metrics_image_id}' in {required_inputs['metrics']}"
        )

    label_encoding = cfg.get("classes", {}).get("label_encoding", {})
    for key in ("Positive_Tumor", "Negative_Tumor", "NonTumor"):
        if key not in label_encoding:
            raise ValueError(f"Missing classes.label_encoding.{key} in config")

    scribble_path = args.annotations_dir / "scribbles" / f"{args.image_id}_scribble_labels.png"
    if not scribble_path.exists():
        raise FileNotFoundError(f"Missing scribble labels for report metrics: {scribble_path}")

    development_metrics = compute_metrics_from_paths(
        image_id=args.image_id,
        scribble_labels_path=scribble_path,
        positive_mask_path=required_inputs["positive_mask"],
        pixel_prob_map_path=required_inputs["pixel_prob_map"],
        label_encoding={k: int(v) for k, v in label_encoding.items()},
    )

    optional_inputs = optional_artifacts(args)
    tile_cv_metrics = load_json(optional_inputs["tile_cv_metrics"]) if optional_inputs["tile_cv_metrics"] else None

    annotation_meta_path = args.annotations_dir / f"{args.image_id}_annotation_meta.json"
    if not annotation_meta_path.exists():
        raise FileNotFoundError(f"Missing annotation metadata for supervision audit: {annotation_meta_path}")

    run_tag = _derive_run_tag(args.reports_dir)
    candidate_tile_label_paths = [
        args.reports_dir.parent / f"tiles_{run_tag}" / "tile_labels.csv" if run_tag else None,
        Path("outputs") / f"tiles_{run_tag}" / "tile_labels.csv" if run_tag else None,
        Path("outputs") / "tiles" / "tile_labels.csv",
    ]
    tile_labels_path = next((p for p in candidate_tile_label_paths if p is not None and p.exists()), None)
    if tile_labels_path is None:
        raise FileNotFoundError("Missing tile labels for supervision audit in expected locations.")
    tile_manifest_candidate = (
        args.reports_dir.parent / f"tiles_{run_tag}" / "tile_manifest.csv"
        if run_tag
        else Path("outputs/tiles/tile_manifest.csv")
    )
    supervision_audit = audit_supervision(
        image_id=args.image_id,
        annotation_meta_path=annotation_meta_path,
        scribble_labels_path=scribble_path,
        tile_labels_path=tile_labels_path,
        tile_manifest_path=tile_manifest_candidate if tile_manifest_candidate.exists() else None,
        label_encoding={k: int(v) for k, v in label_encoding.items()},
    )
    class_metrics = development_metrics.get("class_metrics", {})
    operator_summary = derive_operator_summary(development_metrics)
    warnings = list(supervision_audit.get("warnings", []))
    positive_fn = int(class_metrics.get("Positive_Tumor", {}).get("fn_px", 0))
    positive_tp = int(class_metrics.get("Positive_Tumor", {}).get("tp_px", 0))
    if positive_fn > positive_tp:
        warnings.append("false-negative dominant on Positive_Tumor")
    for negative_class in ("Negative_Tumor", "NonTumor"):
        class_fp = int(class_metrics.get(negative_class, {}).get("fp_px", 0))
        class_tn = int(class_metrics.get(negative_class, {}).get("tn_px", 0))
        if class_fp > class_tn:
            warnings.append(f"false-positive burden concentrated in {negative_class}")

    warnings = sorted(set(warnings))
    included_aliases = [a.strip() for a in str(args.included_training_aliases or "").split(",") if a.strip()]

    payload = {
        "image_id": args.image_id,
        "run_tag": run_tag,
        "model_scope": args.model_scope,
        "shared_model_tag": args.shared_model_tag,
        "training_image_count": args.training_image_count,
        "included_training_aliases": included_aliases,
        "development_metrics": development_metrics,
        "class_metrics": class_metrics,
        "supervision_audit": supervision_audit,
        "warnings": warnings,
        **operator_summary,
        "working_space_note": metrics_json.get(
            "output_space_note", "All masks/maps/overlay are exported in working image space."
        ),
        "artifacts": {
            "metrics_json": required_inputs["metrics"].as_posix(),
            "positive_mask": required_inputs["positive_mask"].as_posix(),
            "pixel_prob_map": required_inputs["pixel_prob_map"].as_posix(),
            "overlay": required_inputs["overlay"].as_posix(),
            "tile_prob_map": required_inputs["tile_prob_map"].as_posix(),
            "scribble_labels": scribble_path.as_posix(),
        },
    }
    verification_meta = {"verification_overlay_available": False}
    try:
        verification_meta = generate_verification_overlay(
            image_id=args.image_id,
            run_tag=run_tag,
            scribble_labels_path=scribble_path,
            positive_mask_path=required_inputs["positive_mask"],
            output_dir=args.overlays_dir,
            label_encoding={k: int(v) for k, v in label_encoding.items()},
        )
    except Exception as exc:
        logging.warning("Verification overlay not generated: %s", exc)
    payload.update(
        {
            k: verification_meta.get(k)
            for k in (
                "verification_overlay_available",
                "verification_overlay_path",
                "verification_overlay_summary_path",
                "verification_overlay_mode",
                "verification_annotation_labels_available",
                "verification_annotation_labels_path",
                "verification_prediction_labels_available",
                "verification_prediction_labels_path",
                "crop_y0",
                "crop_x0",
                "crop_h",
                "crop_w",
            )
        }
    )

    report_json_path = args.reports_dir / "report_summary.json"
    report_md_path = args.reports_dir / "report_summary.md"
    output_pdf = args.reports_dir / "one_page_report.pdf"
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_report_summary_markdown(report_md_path, payload)

    lines = make_summary_lines(
        image_id=args.image_id,
        metrics_json=metrics_json,
        report_metrics=development_metrics,
        tile_cv_metrics=tile_cv_metrics,
    )
    render_pdf(
        output_pdf=output_pdf,
        summary_lines=lines,
        overlay_path=required_inputs["overlay"],
        tile_prob_map_path=required_inputs["tile_prob_map"],
        pixel_prob_map_path=required_inputs["pixel_prob_map"],
        positive_mask_path=required_inputs["positive_mask"],
    )

    logging.info("Wrote report markdown: %s", report_md_path)
    logging.info("Wrote report json: %s", report_json_path)
    logging.info("Wrote one-page report: %s", output_pdf)


if __name__ == "__main__":
    main()
