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
        candidates.extend([Path.cwd() / raw, Path.cwd() / 'verification_regions' / raw.name])
        if repo_root is not None:
            candidates.extend([repo_root / raw, repo_root / 'verification_regions' / raw.name])
        if report_path is not None:
            candidates.extend([report_path.parent / raw, report_path.parent / 'verification_regions' / raw.name])
    uniq = []
    seen = set()
    for c in candidates:
        key = c.as_posix()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    for candidate in uniq:
        if candidate.exists():
            return candidate, uniq
    return None, uniq


def resolve_region_image_path(*, image_path: str | None, regions_json_path: Path, repo_root: Path | None) -> tuple[Path | None, list[Path]]:
    if not image_path:
        return None, []
    raw = Path(str(image_path))
    if raw.is_absolute():
        return (raw if raw.exists() else None), [raw]
    vr_dir = regions_json_path.parent / 'verification_regions'
    candidates = [Path.cwd() / raw, regions_json_path.parent / raw, vr_dir / raw, vr_dir / raw.name]
    if repo_root is not None:
        candidates.extend([repo_root / raw, repo_root / 'verification_regions' / raw.name])
    for c in candidates:
        if c.exists():
            return c, candidates
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
    out = []
    for r in regions:
        if class_filter not in ('All', None) and r.get('class_name') != class_filter:
            continue
        if issue_filter not in ('All', None) and r.get('issue') != issue_filter:
            continue
        out.append(r)
    return out


def sort_verification_regions(regions, sort_key='Highest error first'):
    if sort_key == 'Lowest score first':
        return sorted(regions, key=lambda r: (float(r.get('score') if r.get('score') is not None else 999), -int(r.get('error_px') or 0), str(r.get('region_id', ''))))
    if sort_key == 'Highest score first':
        return sorted(regions, key=lambda r: (-float(r.get('score') if r.get('score') is not None else -1), -int(r.get('error_px') or 0), str(r.get('region_id', ''))))
    if sort_key == 'Class then error':
        return sorted(regions, key=lambda r: (str(r.get('class_name')), -int(r.get('error_px') or 0), str(r.get('region_id', ''))))
    if sort_key == 'Annotation order / region id':
        return sorted(regions, key=lambda r: (int(r.get('annotation_index') or 10**9), str(r.get('region_id', ''))))
    return sorted(regions, key=lambda r: (-int(r.get('error_px') or 0), str(r.get('region_id', ''))))


def get_display_image_shape_hw(viewer) -> tuple[int, int] | None:
    if viewer is None:
        return None
    lyr = viewer.layers.get('image') if 'image' in viewer.layers else (viewer.layers[0] if len(viewer.layers) else None)
    if lyr is None:
        return None
    shape = tuple(int(v) for v in lyr.data.shape)
    return shape[:2]


def working_to_display_scale(working_shape_hw, display_shape_hw) -> tuple[float, float]:
    wy, wx = [max(1.0, float(v)) for v in working_shape_hw]
    dy, dx = [max(1.0, float(v)) for v in display_shape_hw]
    return dy / wy, dx / wx


def working_yx_to_display_yx(yx, working_shape_hw, display_shape_hw):
    sy, sx = working_to_display_scale(working_shape_hw, display_shape_hw)
    return [float(yx[0]) * sy, float(yx[1]) * sx]


def working_bbox_yxhw_to_display_bbox_yxhw(bbox_yxhw, working_shape_hw, display_shape_hw):
    y, x = working_yx_to_display_yx(bbox_yxhw[:2], working_shape_hw, display_shape_hw)
    sy, sx = working_to_display_scale(working_shape_hw, display_shape_hw)
    return [y, x, max(1.0, float(bbox_yxhw[2]) * sy), max(1.0, float(bbox_yxhw[3]) * sx)]


def label_layer_transform_from_working_crop(working_shape_hw, display_shape_hw, crop_origin_working_yx):
    sy, sx = working_to_display_scale(working_shape_hw, display_shape_hw)
    oy, ox = [float(v) for v in crop_origin_working_yx]
    return {"scale": (sy, sx), "translate": (oy * sy, ox * sx)}


def verification_region_label(region) -> str:
    cls = region.get('class_name', 'Unknown')
    issue = region.get('issue', 'unknown')
    score_name = region.get('score_name', 'score')
    score = region.get('score')
    sc = 'n/a' if score is None else f"{float(score):.3f}"
    return f"{cls} | src={region.get('source_type','unknown')} | {issue} | {score_name}={sc} | err={int(region.get('error_px') or 0)}"


def rectangle_vertices_from_bbox_yxhw(bbox_yxhw):
    y, x, h, w = [float(v) for v in bbox_yxhw]
    return [[y, x], [y, x + w], [y + h, x + w], [y + h, x]]


def viewer_bbox_from_region(region, display_shape_hw=None) -> dict[str, Any]:
    wb = region.get("bbox_working_yxhw") or [0, 0, 1, 1]
    wshape = region.get("working_shape_hw")
    bbox = working_bbox_yxhw_to_display_bbox_yxhw(wb, wshape, display_shape_hw) if (wshape and display_shape_hw) else wb
    y, x, h, w = [int(round(float(v))) for v in bbox]
    return {'y': y, 'x': x, 'h': max(1, h), 'w': max(1, w), 'center_yx': [y + max(1, h)//2, x + max(1, w)//2], 'vertices': rectangle_vertices_from_bbox_yxhw([y, x, max(1, h), max(1, w)])}
