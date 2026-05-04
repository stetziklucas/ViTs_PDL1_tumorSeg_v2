"""Discover shared-project Stage 1 cases from saved annotation artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from annotation_readiness import compute_annotation_readiness

SAFE_ALIAS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def make_case_alias(image_id: str, used_aliases: set[str]) -> str:
    """Build deterministic filesystem-safe unique alias from image_id."""
    base = SAFE_ALIAS_RE.sub("_", image_id.strip().lower()).strip("._-") or "case"
    alias = base
    counter = 2
    while alias in used_aliases:
        alias = f"{base}_{counter}"
        counter += 1
    used_aliases.add(alias)
    return alias


def _find_raw_image(raw_dir: Path, image_id: str) -> Path | None:
    candidates = sorted([p for p in raw_dir.glob(f"{image_id}.*") if p.is_file()])
    return candidates[0] if candidates else None


def discover_project_cases(*, config: dict[str, Any], annotations_dir: Path, raw_dir: Path) -> dict[str, Any]:
    """Discover READY and skipped project cases from saved annotation artifacts."""
    candidates = sorted(annotations_dir.glob("*_annotation_meta.json"))
    used_aliases: set[str] = set()
    rows: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    alias_map: dict[str, str] = {}

    for meta_path in candidates:
        image_id = meta_path.name[: -len("_annotation_meta.json")]
        alias = make_case_alias(image_id, used_aliases)
        raw_image = _find_raw_image(raw_dir, image_id)
        if raw_image is None:
            row = {"alias": alias, "image_id": image_id, "status": "SKIPPED", "reason": "MISSING_RAW_IMAGE"}
            rows.append(row)
            skipped.append(row)
            continue

        readiness = compute_annotation_readiness(config=config, annotations_dir=annotations_dir, image_id=image_id)
        if readiness.status_code == "READY":
            row = {
                "alias": alias,
                "image_id": image_id,
                "status": "READY",
                "reason": "READY",
                "raw_image_path": raw_image.as_posix(),
            }
            rows.append(row)
            included.append(row)
            alias_map[alias] = image_id
            continue

        reason = "ARTIFACT_ERROR" if readiness.status_code == "ERROR" else "NOT_READY"
        row = {
            "alias": alias,
            "image_id": image_id,
            "status": "SKIPPED",
            "reason": reason,
            "readiness_status_code": readiness.status_code,
            "readiness_summary": readiness.summary_message,
            "raw_image_path": raw_image.as_posix(),
        }
        rows.append(row)
        skipped.append(row)

    counts_by_reason: dict[str, int] = {}
    for row in skipped:
        counts_by_reason[row["reason"]] = counts_by_reason.get(row["reason"], 0) + 1

    return {
        "project_image_candidates": [{"image_id": r["image_id"], "alias": r["alias"]} for r in rows],
        "included_ready_cases": [{"alias": r["alias"], "image_id": r["image_id"]} for r in included],
        "skipped_cases": skipped,
        "alias_to_image_id": alias_map,
        "counts": {
            "candidate_count": len(rows),
            "included_ready_count": len(included),
            "skipped_count": len(skipped),
            "skipped_by_reason": counts_by_reason,
        },
    }
