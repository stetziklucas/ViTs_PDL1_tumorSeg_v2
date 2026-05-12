"""Create project-level training report rollup across run tags."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.report_metrics import aggregate_class_metrics, aggregate_micro_average, compute_metrics_from_paths, load_json
from scripts.supervision_audit import audit_supervision
from display_format import format_display_float


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for project report rollup."""
    parser = argparse.ArgumentParser(description="Build project-level Stage 1 development training summary.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Annotation root with scribbles/ masks.",
    )
    parser.add_argument(
        "--run-tag",
        action="append",
        dest="run_tags",
        required=True,
        help="Run tag to include (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports_training"),
        help="Output directory for training_summary.{md,json}.",
    )
    return parser


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config did not parse into a dictionary.")
    return config


def _required_paths_for_tag(run_tag: str) -> dict[str, Path]:
    return {
        "maps_dir": Path("outputs") / f"maps_{run_tag}",
        "maps_fused_dir": Path("outputs") / f"maps_{run_tag}_fused",
        "masks_dir": Path("outputs") / f"masks_{run_tag}",
        "overlays_dir": Path("outputs") / f"overlays_{run_tag}",
        "reports_dir": Path("outputs") / f"reports_{run_tag}",
        "tile_model_dir": Path("models") / f"tile_head_{run_tag}",
    }


def _build_warnings_from_report_payload(payload: dict[str, Any]) -> list[str]:
    warnings = [str(w) for w in payload.get("warnings", []) if str(w).strip()]
    class_metrics = payload.get("class_metrics", {})
    positive = class_metrics.get("Positive_Tumor", {})
    if int(positive.get("fn_px", 0)) > int(positive.get("tp_px", 0)):
        warnings.append("false-negative dominant on Positive_Tumor")
    for class_name in ("Negative_Tumor", "NonTumor"):
        cm = class_metrics.get(class_name, {})
        if int(cm.get("fp_px", 0)) > int(cm.get("tn_px", 0)):
            warnings.append(f"false-positive burden concentrated in {class_name}")
    return sorted(set(warnings))


def _coerce_report_payload_from_summary(
    run_tag: str,
    report_summary: dict[str, Any],
    metrics_path: Path,
    tile_prob_path: Path,
    prob_path: Path,
    mask_path: Path,
    overlay_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    image_id = str(report_summary.get("image_id", "")).strip()
    if not image_id:
        return None, "report_summary.json missing image_id"

    development = report_summary.get("development_metrics")
    if not isinstance(development, dict):
        return None, "report_summary.json missing development_metrics"
    class_metrics = report_summary.get("class_metrics")
    if not isinstance(class_metrics, dict):
        return None, "report_summary.json missing class_metrics"
    supervision = report_summary.get("supervision_audit")
    if not isinstance(supervision, dict):
        return None, "report_summary.json missing supervision_audit"

    return {
        "run_tag": run_tag,
        "image_id": image_id,
        "model_scope": report_summary.get("model_scope", "single_image_model"),
        "shared_model_tag": report_summary.get("shared_model_tag"),
        "training_image_count": report_summary.get("training_image_count"),
        "included_training_aliases": report_summary.get("included_training_aliases", []),
        "development_metrics": development,
        "class_metrics": class_metrics,
        "supervision_audit": supervision,
        "warnings": _build_warnings_from_report_payload(report_summary),
        "working_space_note": report_summary.get("working_space_note", ""),
        "encoder_provenance": report_summary.get("encoder_provenance"),
        "artifacts": {
            "metrics_json": metrics_path.as_posix(),
            "positive_mask": mask_path.as_posix(),
            "pixel_prob_map": prob_path.as_posix(),
            "overlay": overlay_path.as_posix(),
            "tile_prob_map": tile_prob_path.as_posix(),
            "report_summary_json": (metrics_path.parent / "report_summary.json").as_posix(),
        },
    }, None


def evaluate_run_tag(run_tag: str, annotations_dir: Path, label_encoding: dict[str, int]) -> tuple[dict[str, Any] | None, str | None]:
    """Compute report metrics for one run tag; return skip reason if incomplete."""
    dirs = _required_paths_for_tag(run_tag)
    required_dirs = {k: v for k, v in dirs.items() if k != "tile_model_dir"}
    missing_dirs = [name for name, path in required_dirs.items() if not path.exists()]
    if missing_dirs:
        return None, f"missing required directories: {', '.join(missing_dirs)}"

    metrics_path = dirs["reports_dir"] / "metrics.json"
    report_summary_path = dirs["reports_dir"] / "report_summary.json"
    tile_labels_path = Path("outputs") / f"tiles_{run_tag}" / "tile_labels.csv"
    tile_manifest_path = Path("outputs") / f"tiles_{run_tag}" / "tile_manifest.csv"
    mask_path = dirs["masks_dir"] / "positive_mask.png"
    prob_path = dirs["maps_fused_dir"] / "pixel_prob_map.png"
    overlay_path = dirs["overlays_dir"] / "overlay.png"
    tile_prob_path = dirs["maps_dir"] / "tile_prob_map.png"

    missing_files = [
        name
        for name, path in {
            "metrics.json": metrics_path,
            "positive_mask.png": mask_path,
            "pixel_prob_map.png": prob_path,
            "overlay.png": overlay_path,
            "tile_prob_map.png": tile_prob_path,
            "tile_labels.csv": tile_labels_path,
        }.items()
        if not path.exists()
    ]
    if missing_files:
        return None, f"missing required artifacts: {', '.join(missing_files)}"

    if report_summary_path.exists():
        report_summary = load_json(report_summary_path)
        payload, reason = _coerce_report_payload_from_summary(
            run_tag=run_tag,
            report_summary=report_summary,
            metrics_path=metrics_path,
            tile_prob_path=tile_prob_path,
            prob_path=prob_path,
            mask_path=mask_path,
            overlay_path=overlay_path,
        )
        if payload is not None:
            return payload, None
        logging.warning("run_tag=%s falling back to recompute path because: %s", run_tag, reason)

    metrics_json = load_json(metrics_path)
    image_id = str(metrics_json.get("image_id", "")).strip()
    if not image_id:
        return None, "metrics.json missing image_id"

    scribble_path = annotations_dir / "scribbles" / f"{image_id}_scribble_labels.png"
    annotation_meta_path = annotations_dir / f"{image_id}_annotation_meta.json"
    if not scribble_path.exists():
        return None, f"missing scribble labels: {scribble_path.as_posix()}"
    if not annotation_meta_path.exists():
        return None, f"missing annotation meta: {annotation_meta_path.as_posix()}"

    image_metrics = compute_metrics_from_paths(
        image_id=image_id,
        scribble_labels_path=scribble_path,
        positive_mask_path=mask_path,
        pixel_prob_map_path=prob_path,
        label_encoding=label_encoding,
    )
    supervision = audit_supervision(
        image_id=image_id,
        annotation_meta_path=annotation_meta_path,
        scribble_labels_path=scribble_path,
        tile_labels_path=tile_labels_path,
        tile_manifest_path=tile_manifest_path if tile_manifest_path.exists() else None,
        label_encoding=label_encoding,
    )
    class_metrics = image_metrics.get("class_metrics", {})
    warnings = list(supervision.get("warnings", []))
    warnings.extend(_build_warnings_from_report_payload({"class_metrics": class_metrics}))

    return {
        "run_tag": run_tag,
        "image_id": image_id,
        "model_scope": "single_image_model",
        "shared_model_tag": None,
        "training_image_count": None,
        "included_training_aliases": [],
        "development_metrics": image_metrics,
        "class_metrics": class_metrics,
        "supervision_audit": supervision,
        "warnings": sorted(set(warnings)),
        "working_space_note": metrics_json.get("output_space_note", ""),
        "artifacts": {
            "metrics_json": metrics_path.as_posix(),
            "positive_mask": mask_path.as_posix(),
            "pixel_prob_map": prob_path.as_posix(),
            "overlay": overlay_path.as_posix(),
            "tile_prob_map": tile_prob_path.as_posix(),
            "scribble_labels": scribble_path.as_posix(),
        },
    }, None




def _encoder_consensus(included: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    per_run = {}
    encs = []
    missing = 0
    for row in included:
        ep = row.get("encoder_provenance") if isinstance(row.get("encoder_provenance"), dict) else None
        if ep:
            per_run[row.get("run_tag", "unknown")] = ep
            encs.append(ep.get("encoder_id"))
        else:
            missing += 1
    unique = sorted({e for e in encs if e})
    cons = {"consistent": len(unique) <= 1, "encoder_ids": unique, "warning": None}
    if missing:
        cons["warning"] = f"{missing} run(s) missing encoder provenance"
    if len(unique) > 1:
        cons["warning"] = "mixed encoder provenance detected across included runs"
    top = None
    if cons["consistent"] and per_run:
        top = next(iter(per_run.values()))
    return top, per_run, cons


def _encoder_field(provenance: dict[str, Any] | None, key: str, default: str = "n/a") -> str:
    """Return normalized encoder provenance field."""
    if not isinstance(provenance, dict):
        return default
    value = provenance.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _encoder_display_name(provenance: dict[str, Any] | None) -> str:
    """Return friendly encoder display name."""
    if not isinstance(provenance, dict):
        return "not recorded"
    return (
        str(provenance.get("encoder_display_name") or "")
        or str(provenance.get("encoder_id") or "")
        or "not recorded"
    )

def build_training_summary_markdown(payload: dict[str, Any]) -> str:
    """Render plain markdown summary for CRD/editor consumption."""
    a = payload["aggregate_metrics"]
    class_agg = payload["aggregate_class_metrics"]
    provenance = payload.get("encoder_provenance") if isinstance(payload.get("encoder_provenance"), dict) else None
    encoder_name = _encoder_display_name(provenance)
    encoder_id = _encoder_field(provenance, "encoder_id", "n/a")
    encoder_backend = _encoder_field(provenance, "encoder_backend", "n/a")
    encoder_model = _encoder_field(provenance, "encoder_model_name", "n/a")
    encoder_pooling = _encoder_field(provenance, "encoder_pooling", "n/a")
    encoder_dim = _encoder_field(provenance, "embedding_dim", "n/a")
    lines = [
        "# Stage 1 Training Development Summary",
        "",
        "Note: Metrics are computed on annotated training regions only (scribble labels),",
        "not on whole-slide unlabeled validation regions. Ignore/Unlabeled pixels are excluded.",
        "",
        f"Embedding encoder: {encoder_name} ({encoder_id})",
        f"Embedding backend: {encoder_backend}",
        f"Embedding model: {encoder_model}",
        f"Embedding pooling: {encoder_pooling}",
        f"Embedding dimension: {encoder_dim}",
        "",
        "## Aggregate project summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| false_positive_px | {int(a['false_positive_px'])} |",
        f"| false_negative_px | {int(a['false_negative_px'])} |",
        f"| precision | {format_display_float(a['precision'])} |",
        f"| sensitivity | {format_display_float(a['sensitivity'])} |",
        f"| f1 | {format_display_float(a['f1'])} |",
        f"| training_log_loss_total | {format_display_float(a['training_log_loss_total'])} |",
        "",
        "## Aggregate class-specific annotated-region metrics",
        "",
        "| class | annotated_px | primary_correct_px | primary_error_px | score_name | score |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
        f"| Positive_Tumor | {int(class_agg['Positive_Tumor']['annotated_px'])} | {int(class_agg['Positive_Tumor']['tp_px'])} | {int(class_agg['Positive_Tumor']['fn_px'])} | sensitivity | {format_display_float(class_agg['Positive_Tumor']['sensitivity'])} |",
        f"| Negative_Tumor | {int(class_agg['Negative_Tumor']['annotated_px'])} | {int(class_agg['Negative_Tumor']['tn_px'])} | {int(class_agg['Negative_Tumor']['fp_px'])} | specificity | {format_display_float(class_agg['Negative_Tumor']['specificity'])} |",
        f"| NonTumor | {int(class_agg['NonTumor']['annotated_px'])} | {int(class_agg['NonTumor']['tn_px'])} | {int(class_agg['NonTumor']['fp_px'])} | specificity | {format_display_float(class_agg['NonTumor']['specificity'])} |",
        "",
        "## Per-image breakdown",
        "",
        "| run_tag | image_id | encoder | polygons(+/-tumor/non) | pixels(+/-tumor/non) | tile labels(P/N/I) | ignored tiles | false_positive_px | false_negative_px | precision | sensitivity | f1 |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["included_runs"]:
        m = row["development_metrics"]
        audit = row["supervision_audit"]
        polygon_counts = audit.get("polygon_counts", {})
        pixel_counts = audit.get("annotated_pixel_counts", {})
        tile_counts = audit.get("tile_label_counts", {})
        lines.append(
            f"| {row['run_tag']} | {row['image_id']} | {(row.get('encoder_provenance') or {}).get('encoder_id', 'n/a')} / {(row.get('encoder_provenance') or {}).get('encoder_backend', 'n/a')} / {(row.get('encoder_provenance') or {}).get('embedding_dim', 'n/a')} | "
            f"{int(polygon_counts.get('Positive_Tumor', 0))}/{int(polygon_counts.get('Negative_Tumor', 0))}/{int(polygon_counts.get('NonTumor', 0))} | "
            f"{int(pixel_counts.get('Positive_Tumor', 0))}/{int(pixel_counts.get('Negative_Tumor', 0))}/{int(pixel_counts.get('NonTumor', 0))} | "
            f"{int(tile_counts.get('Positive_Context', 0))}/{int(tile_counts.get('Negative_Context', 0))}/{int(tile_counts.get('Ignore', 0))} | "
            f"{int(audit.get('ignored_tile_count', 0))} | "
            f"{int(m['false_positive_px'])} | {int(m['false_negative_px'])} | "
            f"{format_display_float(m['precision'])} | {format_display_float(m['sensitivity'])} | {format_display_float(m['f1'])} |"
        )

    lines.extend(["", "## Images needing attention", ""])
    if payload["images_needing_attention"]:
        for row in payload["images_needing_attention"]:
            lines.append(f"- {row['run_tag']} | {row['image_id']}: {', '.join(row['warnings'])}")
    else:
        lines.append("- none")

    lines.extend(["", "## Skipped runs", ""])
    if payload["skipped_runs"]:
        for row in payload["skipped_runs"]:
            lines.append(f"- run_tag={row['run_tag']}: {row['reason']}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entrypoint for training project report rollup."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    label_encoding_raw = cfg.get("classes", {}).get("label_encoding", {})
    label_encoding = {k: int(v) for k, v in label_encoding_raw.items()}
    for key in ("Positive_Tumor", "Negative_Tumor", "NonTumor"):
        if key not in label_encoding:
            raise ValueError(f"Missing classes.label_encoding.{key} in config")

    included: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for run_tag in args.run_tags:
        result, reason = evaluate_run_tag(run_tag, args.annotations_dir, label_encoding)
        if reason is not None:
            skipped.append({"run_tag": run_tag, "reason": reason})
            logging.warning("Skipping run_tag=%s: %s", run_tag, reason)
            continue
        included.append(result)

    aggregate = aggregate_micro_average([row["development_metrics"] for row in included])
    class_aggregate = aggregate_class_metrics([row["development_metrics"] for row in included])
    attention = [
        {"run_tag": row["run_tag"], "image_id": row["image_id"], "warnings": row["warnings"]}
        for row in included
        if row.get("warnings")
    ]
    encoder_prov, per_run_encoder_provenance, encoder_consistency = _encoder_consensus(included)
    payload = {
        "run_tags_requested": args.run_tags,
        "included_runs": included,
        "skipped_runs": skipped,
        "aggregate_metrics": aggregate,
        "aggregate_class_metrics": class_aggregate,
        "images_needing_attention": attention,
        "encoder_provenance": encoder_prov,
        "per_run_encoder_provenance": per_run_encoder_provenance,
        "encoder_consistency": encoder_consistency,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = args.output_dir / "training_summary.json"
    summary_md_path = args.output_dir / "training_summary.md"

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary_md_path.write_text(build_training_summary_markdown(payload), encoding="utf-8")

    logging.info("Wrote project training report JSON: %s", summary_json_path)
    logging.info("Wrote project training report markdown: %s", summary_md_path)


if __name__ == "__main__":
    main()
