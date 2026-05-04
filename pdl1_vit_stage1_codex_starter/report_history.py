"""Helpers for per-image Stage 1 report history indexing and comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from display_format import format_display_float


def slugify_image_id(image_id: str) -> str:
    """Convert image id to filesystem-safe slug for history directories."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in image_id).strip("_") or "unknown_image"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _path_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def _best_timestamp(report_summary: dict[str, Any], runner_summary: dict[str, Any] | None, report_path: Path) -> str:
    for key in ("ended_at_utc", "generated_at_utc", "timestamp", "created_at_utc"):
        value = report_summary.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if runner_summary:
        for key in ("ended_at_utc", "started_at_utc"):
            value = runner_summary.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return _path_timestamp(report_path)


def _extract_run_tag(report_summary: dict[str, Any], report_dir: Path) -> str:
    value = report_summary.get("run_tag")
    if isinstance(value, str) and value.strip():
        return value.strip()
    stem = report_dir.name
    return stem[len("reports_") :] if stem.startswith("reports_") else stem


def _extract_entry(report_dir: Path, report_summary: dict[str, Any]) -> dict[str, Any]:
    runner_path = report_dir / "stage1_run_summary.json"
    runner_md_path = report_dir / "stage1_run_summary.md"
    runner_summary = _read_json(runner_path)
    development = report_summary.get("development_metrics", {}) if isinstance(report_summary, dict) else {}
    supervision = report_summary.get("supervision_audit", {}) if isinstance(report_summary, dict) else {}
    warnings = report_summary.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    tile_counts = supervision if isinstance(supervision, dict) else {}

    return {
        "image_id": report_summary.get("image_id", ""),
        "run_tag": _extract_run_tag(report_summary, report_dir),
        "report_dir": report_dir.as_posix(),
        "report_summary_json": (report_dir / "report_summary.json").as_posix(),
        "report_summary_md": (report_dir / "report_summary.md").as_posix(),
        "runner_summary_json": runner_path.as_posix() if runner_path.exists() else None,
        "stage1_run_summary_json": runner_path.as_posix() if runner_path.exists() else None,
        "stage1_run_summary_md": runner_md_path.as_posix() if runner_md_path.exists() else None,
        "model_scope": report_summary.get("model_scope", "single_image_model"),
        "shared_model_tag": report_summary.get("shared_model_tag"),
        "training_image_count": report_summary.get("training_image_count"),
        "precision": development.get("precision"),
        "sensitivity": development.get("sensitivity"),
        "f1": development.get("f1"),
        "false_positive_px": development.get("false_positive_px"),
        "false_negative_px": development.get("false_negative_px"),
        "training_log_loss_total": development.get("training_log_loss_total"),
        "supervision_warnings": [str(w) for w in warnings],
        "usable_tile_count": tile_counts.get("usable_tile_count"),
        "ignored_tile_count": tile_counts.get("ignored_tile_count"),
        "accepted_tile_count": tile_counts.get("accepted_tile_count"),
        "timestamp_utc": _best_timestamp(report_summary, runner_summary, report_dir / "report_summary.json"),
        "report_summary_mtime_utc": _path_timestamp(report_dir / "report_summary.json"),
    }


def _parse_timestamp(timestamp_text: Any) -> datetime:
    if not isinstance(timestamp_text, str) or not timestamp_text.strip():
        return datetime.min.replace(tzinfo=timezone.utc)
    candidate = timestamp_text.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def sort_history_entries(entries: list[dict[str, Any]], descending: bool = True) -> list[dict[str, Any]]:
    """Sort history entries by best-known timestamp, then run tag."""
    return sorted(
        entries,
        key=lambda row: (_parse_timestamp(row.get("timestamp_utc")), str(row.get("run_tag", ""))),
        reverse=descending,
    )


def _normalize_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized.setdefault("stage1_run_summary_json", normalized.get("runner_summary_json"))
    stage1_json = normalized.get("stage1_run_summary_json")
    if stage1_json and not normalized.get("stage1_run_summary_md"):
        normalized["stage1_run_summary_md"] = Path(str(stage1_json)).with_suffix(".md").as_posix()
    normalized.setdefault("report_summary_md", "")
    normalized.setdefault("report_summary_json", "")
    normalized.setdefault("timestamp_utc", "")
    normalized.setdefault("model_scope", "single_image_model")
    return normalized


def discover_history_entries_for_image(image_id: str, outputs_root: Path = Path("outputs")) -> list[dict[str, Any]]:
    """Discover per-run report entries for one image directly from outputs/reports_* directories."""
    records: list[dict[str, Any]] = []
    for report_dir in discover_reports_for_image(image_id=image_id, outputs_root=outputs_root):
        report_summary = _read_json(report_dir / "report_summary.json")
        if not report_summary:
            continue
        records.append(_extract_entry(report_dir, report_summary))
    return sort_history_entries(records, descending=True)


def load_history_entries_for_image(image_id: str, outputs_root: Path = Path("outputs")) -> list[dict[str, Any]]:
    """Load history entries from index when available, with discovery fallback/backfill."""
    paths = _history_paths(image_id, outputs_root)
    indexed: list[dict[str, Any]] = []
    index_payload = _read_json(paths["history_index_json"])
    if index_payload and str(index_payload.get("image_id", "")).strip() == image_id:
        runs = index_payload.get("runs", [])
        if isinstance(runs, list):
            indexed = [_normalize_history_entry(row) for row in runs if isinstance(row, dict)]

    discovered = discover_history_entries_for_image(image_id=image_id, outputs_root=outputs_root)
    discovered_by_tag = {str(row.get("run_tag", "")): row for row in discovered}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in indexed:
        run_tag = str(row.get("run_tag", ""))
        if run_tag in discovered_by_tag:
            merged_row = dict(row)
            merged_row.update(discovered_by_tag[run_tag])
            merged.append(_normalize_history_entry(merged_row))
            seen.add(run_tag)
        else:
            merged.append(_normalize_history_entry(row))
            seen.add(run_tag)

    for row in discovered:
        run_tag = str(row.get("run_tag", ""))
        if run_tag not in seen:
            merged.append(_normalize_history_entry(row))
    return sort_history_entries(merged, descending=True)


def format_history_entry_label(entry: dict[str, Any]) -> str:
    """Format concise dropdown label for one history entry."""
    ts = _parse_timestamp(entry.get("timestamp_utc"))
    timestamp = ts.strftime("%Y-%m-%d %H:%M") if ts.year > 1900 else "unknown time"
    run_tag = str(entry.get("run_tag") or "unknown_run")
    scope_raw = str(entry.get("model_scope") or "single_image_model")
    scope = "shared" if "shared" in scope_raw else "single"
    f1_text = f"F1 {format_display_float(_to_float(entry.get('f1')))}"
    return f"{timestamp} | {run_tag} | {scope} | {f1_text}"


def newest_history_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return newest history entry by timestamp."""
    if not entries:
        return None
    return sort_history_entries(entries, descending=True)[0]


def oldest_history_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return oldest history entry by timestamp."""
    if not entries:
        return None
    return sort_history_entries(entries, descending=False)[0]


def discover_reports_for_image(image_id: str, outputs_root: Path = Path("outputs")) -> list[Path]:
    """Discover report directories under outputs/reports_* that belong to image_id."""
    report_dirs: list[Path] = []
    for report_dir in sorted(outputs_root.glob("reports_*")):
        report_summary_path = report_dir / "report_summary.json"
        report_summary = _read_json(report_summary_path)
        if not report_summary:
            continue
        if str(report_summary.get("image_id", "")).strip() != image_id:
            continue
        report_dirs.append(report_dir)
    return report_dirs


def _history_paths(image_id: str, outputs_root: Path) -> dict[str, Path]:
    history_root = outputs_root / "report_history" / slugify_image_id(image_id)
    return {
        "history_root": history_root,
        "history_index_json": history_root / "history_index.json",
        "latest_vs_previous_json": history_root / "latest_vs_previous.json",
        "latest_vs_previous_md": history_root / "latest_vs_previous.md",
    }


def build_history_index(image_id: str, outputs_root: Path = Path("outputs")) -> dict[str, Any]:
    """Build complete per-image history index from discovered report summaries."""
    records = sort_history_entries(discover_history_entries_for_image(image_id=image_id, outputs_root=outputs_root), descending=False)

    paths = _history_paths(image_id, outputs_root)
    payload = {
        "image_id": image_id,
        "image_slug": slugify_image_id(image_id),
        "updated_at_utc": _iso_utc_now(),
        "run_count": len(records),
        "runs": records,
    }
    paths["history_root"].mkdir(parents=True, exist_ok=True)
    paths["history_index_json"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(latest: Any, previous: Any) -> float | None:
    latest_f = _to_float(latest)
    previous_f = _to_float(previous)
    if latest_f is None or previous_f is None:
        return None
    return latest_f - previous_f


def build_latest_vs_previous(index_payload: dict[str, Any], outputs_root: Path = Path("outputs")) -> dict[str, Any]:
    """Build latest-vs-previous comparison payload and write JSON+markdown artifacts."""
    image_id = str(index_payload.get("image_id", ""))
    runs = list(index_payload.get("runs", []))
    latest = runs[-1] if runs else None
    previous = runs[-2] if len(runs) >= 2 else None

    result: dict[str, Any] = {
        "image_id": image_id,
        "image_slug": slugify_image_id(image_id),
        "generated_at_utc": _iso_utc_now(),
        "comparison_available": bool(latest and previous),
        "latest_run_tag": latest.get("run_tag") if latest else None,
        "previous_run_tag": previous.get("run_tag") if previous else None,
    }

    if latest and previous:
        result["metric_deltas"] = {
            "precision": _delta(latest.get("precision"), previous.get("precision")),
            "sensitivity": _delta(latest.get("sensitivity"), previous.get("sensitivity")),
            "f1": _delta(latest.get("f1"), previous.get("f1")),
            "false_positive_px": _delta(latest.get("false_positive_px"), previous.get("false_positive_px")),
            "false_negative_px": _delta(latest.get("false_negative_px"), previous.get("false_negative_px")),
            "training_log_loss_total": _delta(
                latest.get("training_log_loss_total"),
                previous.get("training_log_loss_total"),
            ),
            "usable_tile_count": _delta(latest.get("usable_tile_count"), previous.get("usable_tile_count")),
            "ignored_tile_count": _delta(latest.get("ignored_tile_count"), previous.get("ignored_tile_count")),
            "accepted_tile_count": _delta(latest.get("accepted_tile_count"), previous.get("accepted_tile_count")),
        }
        latest_warnings = set(str(w) for w in latest.get("supervision_warnings", []))
        previous_warnings = set(str(w) for w in previous.get("supervision_warnings", []))
        result["warning_changes"] = {
            "added": sorted(latest_warnings - previous_warnings),
            "removed": sorted(previous_warnings - latest_warnings),
        }
        result["provenance_changes"] = {
            "model_scope_changed": latest.get("model_scope") != previous.get("model_scope"),
            "latest_model_scope": latest.get("model_scope"),
            "previous_model_scope": previous.get("model_scope"),
            "shared_model_tag_changed": latest.get("shared_model_tag") != previous.get("shared_model_tag"),
            "latest_shared_model_tag": latest.get("shared_model_tag"),
            "previous_shared_model_tag": previous.get("shared_model_tag"),
            "training_image_count_changed": latest.get("training_image_count") != previous.get("training_image_count"),
            "latest_training_image_count": latest.get("training_image_count"),
            "previous_training_image_count": previous.get("training_image_count"),
        }

    paths = _history_paths(image_id, outputs_root)
    paths["history_root"].mkdir(parents=True, exist_ok=True)
    paths["latest_vs_previous_json"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["latest_vs_previous_md"].write_text(render_latest_vs_previous_markdown(result), encoding="utf-8")
    return result


def render_latest_vs_previous_markdown(compare_payload: dict[str, Any]) -> str:
    """Render markdown preview for latest-vs-previous payload."""
    lines = [
        "# Latest vs Previous Stage 1 Report",
        "",
        f"image_id: {compare_payload.get('image_id', 'unknown')}",
        f"latest_run_tag: {compare_payload.get('latest_run_tag') or 'n/a'}",
        f"previous_run_tag: {compare_payload.get('previous_run_tag') or 'n/a'}",
        "",
    ]
    if not compare_payload.get("comparison_available"):
        lines.extend(["Comparison not available yet (need at least two runs in history).", ""])
        return "\n".join(lines)

    lines.extend(["## Metric deltas (latest - previous)", ""])
    deltas = compare_payload.get("metric_deltas", {})
    for key in (
        "precision",
        "sensitivity",
        "f1",
        "false_positive_px",
        "false_negative_px",
        "training_log_loss_total",
        "usable_tile_count",
        "ignored_tile_count",
        "accepted_tile_count",
    ):
        lines.append(f"- {key}: {deltas.get(key)}")
    warning_changes = compare_payload.get("warning_changes", {})
    lines.extend(["", "## Warning changes", ""])
    lines.append(f"- added: {', '.join(warning_changes.get('added', [])) or 'none'}")
    lines.append(f"- removed: {', '.join(warning_changes.get('removed', [])) or 'none'}")
    provenance = compare_payload.get("provenance_changes", {})
    lines.extend(["", "## Provenance changes", ""])
    lines.append(
        f"- model_scope: {provenance.get('previous_model_scope')} -> {provenance.get('latest_model_scope')}"
    )
    lines.append(
        f"- shared_model_tag: {provenance.get('previous_shared_model_tag')} -> {provenance.get('latest_shared_model_tag')}"
    )
    lines.append(
        f"- training_image_count: {provenance.get('previous_training_image_count')} -> {provenance.get('latest_training_image_count')}"
    )
    lines.append("")
    return "\n".join(lines)


def refresh_history_for_image(image_id: str, outputs_root: Path = Path("outputs")) -> dict[str, Any]:
    """Rebuild history index and latest-vs-previous artifacts for image."""
    index_payload = build_history_index(image_id=image_id, outputs_root=outputs_root)
    compare_payload = build_latest_vs_previous(index_payload=index_payload, outputs_root=outputs_root)
    return {"history_index": index_payload, "latest_vs_previous": compare_payload}
