from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ZERO_REGION_MESSAGE = "No verification review regions were generated for this report. Regenerate after annotation/report fix or inspect annotation artifacts."
COORDINATE_SCHEMA_VERSION = 2

def load_verification_regions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('regions', 'verification_regions', 'items'):
            regions = payload.get(key)
            if isinstance(regions, list):
                return regions
    return []

def resolve_verification_regions_path(*, verification_regions_path: str | None, report_path: Path | None, repo_root: Path | None) -> tuple[Path | None, list[Path]]:
    if not verification_regions_path:
        return None, []
    raw = Path(verification_regions_path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        if repo_root is not None:
            candidates.append(repo_root / raw)
        if report_path is not None:
            candidates.append(report_path.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates

def verification_regions_message(path: Path | None, regions: list[dict] | None = None) -> str | None:
    if path is None or not path.exists():
        return 'Verification results viewer: verification_regions.json missing; regenerate report/project run.'
    if regions is None:
        regions = load_verification_regions(path)
    if not regions:
        return ZERO_REGION_MESSAGE
    return None

def filter_verification_regions(regions, class_filter='All', issue_filter='All'):
    out=[]
    for r in regions:
        if class_filter not in ('All', None) and r.get('class_name') != class_filter:
            continue
        if issue_filter not in ('All', None) and r.get('issue') != issue_filter:
            continue
        out.append(r)
    return out

def sort_verification_regions(regions, sort_key='review_priority'):
    if sort_key == 'class_name':
        return sorted(regions, key=lambda r: (str(r.get('class_name')), -int(r.get('review_priority') or 0)))
    return sorted(regions, key=lambda r: -int(r.get('review_priority') or 0))

def verification_region_label(region) -> str:
    cls = region.get('class_name', 'Unknown')
    issue = region.get('issue', 'unknown')
    score_name = region.get('score_name', 'score')
    score = region.get('score')
    sc = 'n/a' if score is None else f"{float(score):.3f}"
    return f"{cls} | src={region.get('source_type','unknown')} | {issue} | {score_name}={sc} | err={int(region.get('error_px') or 0)}"

def compute_working_to_display_transform(working_shape_hw, display_shape_hw, crop_origin_working_yx=None):
    wy, wx = [max(1.0, float(v)) for v in working_shape_hw]
    dy, dx = [max(1.0, float(v)) for v in display_shape_hw]
    sy, sx = dy / wy, dx / wx
    oy, ox = (0.0, 0.0) if crop_origin_working_yx is None else (float(crop_origin_working_yx[0]), float(crop_origin_working_yx[1]))
    return {"scale_yx": (sy, sx), "translate_yx": (oy * sy, ox * sx)}

def rectangle_vertices_from_bbox_yxhw(bbox_yxhw):
    y, x, h, w = [float(v) for v in bbox_yxhw]
    return [[y, x], [y, x + w], [y + h, x + w], [y + h, x]]

def napari_bbox_from_region(region, display_shape_hw=None):
    bbox = region.get("bbox_annotation_yxhw")
    if bbox is None:
        wb = region.get("bbox_working_yxhw") or [0, 0, 1, 1]
        if display_shape_hw is not None and region.get("working_shape_hw"):
            t = compute_working_to_display_transform(region["working_shape_hw"], display_shape_hw)
            y, x, h, w = [float(v) for v in wb]
            sy, sx = t["scale_yx"]
            bbox = [y * sy, x * sx, max(1.0, h * sy), max(1.0, w * sx)]
        else:
            bbox = wb
    y, x, h, w = [int(round(float(v))) for v in bbox]
    return {"y": y, "x": x, "h": max(1, h), "w": max(1, w), "vertices": rectangle_vertices_from_bbox_yxhw([y, x, max(1, h), max(1, w)])}

def viewer_bbox_from_region(region) -> dict[str, Any]:
    nb = napari_bbox_from_region(region, display_shape_hw=region.get("annotation_shape_hw") or region.get("display_shape_hw"))
    y, x, h, w = nb["y"], nb["x"], nb["h"], nb["w"]
    cyx = region.get('center_annotation_yx')
    if cyx is None and region.get("center_working_yx") is not None and region.get("annotation_shape_hw") and region.get("working_shape_hw"):
        t = compute_working_to_display_transform(region["working_shape_hw"], region["annotation_shape_hw"])
        cyx = [int(round(float(region["center_working_yx"][0]) * t["scale_yx"][0])), int(round(float(region["center_working_yx"][1]) * t["scale_yx"][1]))]
    return {'y': y, 'x': x, 'h': h, 'w': w, 'center_yx': cyx or [y + h//2, x + w//2]}
