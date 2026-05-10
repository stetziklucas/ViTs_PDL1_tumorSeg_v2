from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ZERO_REGION_MESSAGE = "No verification review regions were generated for this report. Regenerate after annotation/report fix or inspect annotation artifacts."
COORDINATE_SCHEMA_VERSION = 2

def load_verification_regions_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"regions": []}
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        return {"regions": payload}
    if not isinstance(payload, dict):
        return {"regions": []}
    out = dict(payload)
    if not isinstance(out.get("regions"), list):
        for key in ("verification_regions", "items"):
            if isinstance(out.get(key), list):
                out["regions"] = out.get(key)
                break
    if not isinstance(out.get("regions"), list):
        out["regions"] = []
    return out


def load_verification_regions(path: Path) -> list[dict]:
    return load_verification_regions_payload(path).get("regions", [])


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


def get_layer_by_name(layers, name: str):
    if layers is None:
        return None
    if hasattr(layers, 'get'):
        try:
            lyr = layers.get(name)
            if lyr is not None:
                return lyr
        except Exception:
            pass
    try:
        lyr = layers[name]
        if lyr is not None:
            return lyr
    except Exception:
        pass
    for lyr in list(layers):
        if getattr(lyr, 'name', None) == name:
            return lyr
    return None


def get_display_image_shape_hw(viewer) -> tuple[int, int] | None:
    if viewer is None or getattr(viewer, 'layers', None) is None:
        return None
    layers = viewer.layers
    lyr = get_layer_by_name(layers, 'image')
    if lyr is None:
        for cand in list(layers):
            data = getattr(cand, 'data', None)
            shape = getattr(data, 'shape', None)
            if shape is not None and len(shape) >= 2:
                lyr = cand
                break
    if lyr is None:
        return None
    shape = tuple(int(v) for v in getattr(lyr.data, 'shape', ()))
    if len(shape) < 2:
        return None
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


def build_label_layer_transform_from_entry_or_payload(entry: dict[str, Any] | None, payload: dict[str, Any] | None, display_shape_hw: tuple[int, int] | None) -> dict[str, Any]:
    if not display_shape_hw:
        return {"warning": "missing display_shape_hw"}
    srcs = [entry if isinstance(entry, dict) else {}, payload if isinstance(payload, dict) else {}]
    working_shape_hw = None
    crop_origin = None
    for src in srcs:
        shp = src.get("working_shape_hw")
        if isinstance(shp, (list, tuple)) and len(shp) >= 2:
            working_shape_hw = [int(shp[0]), int(shp[1])]
            break
    for src in srcs:
        cor = src.get("crop_origin_working_yx")
        if isinstance(cor, (list, tuple)) and len(cor) >= 2:
            crop_origin = [int(cor[0]), int(cor[1])]
            break
        if src.get("crop_y0") is not None and src.get("crop_x0") is not None:
            crop_origin = [int(src["crop_y0"]), int(src["crop_x0"])]
            break
    missing = []
    if not working_shape_hw:
        missing.append("working_shape_hw")
    if not crop_origin:
        missing.append("crop_origin_working_yx (or crop_y0/crop_x0)")
    if missing:
        return {"warning": f"missing {', '.join(missing)}"}
    tfm = label_layer_transform_from_working_crop(working_shape_hw, display_shape_hw, crop_origin)
    return {"working_shape_hw": working_shape_hw, "crop_origin_working_yx": crop_origin, **tfm, "warning": None}


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


def _coerce_positive_int(value: Any) -> int | None:
    try:
        iv = int(round(float(value)))
    except Exception:
        return None
    return iv if iv > 0 else None


def _extract_wh_from_size_like(size_obj: Any) -> tuple[int, int] | None:
    if size_obj is None:
        return None
    if isinstance(size_obj, (list, tuple)) and len(size_obj) >= 2:
        w = _coerce_positive_int(size_obj[0]); h = _coerce_positive_int(size_obj[1])
        return (w, h) if (w and h) else None
    width = height = None
    if hasattr(size_obj, "width") and hasattr(size_obj, "height"):
        try:
            width = size_obj.width() if callable(size_obj.width) else size_obj.width
            height = size_obj.height() if callable(size_obj.height) else size_obj.height
        except Exception:
            width = height = None
    w = _coerce_positive_int(width); h = _coerce_positive_int(height)
    return (w, h) if (w and h) else None


def canvas_size_wh(canvas) -> tuple[int, int] | None:
    """Return canvas (width, height) if available across qt/vispy runtime variants."""
    if canvas is None:
        return None
    for cand in (
        getattr(canvas, "size", None),
        getattr(canvas, "physical_size", None),
        getattr(getattr(canvas, "native", None), "size", None),
    ):
        wh = _extract_wh_from_size_like(cand() if callable(cand) else cand)
        if wh is not None:
            return wh
    return None


def compute_jump_zoom(
    bbox_display_yxhw,
    canvas_shape_wh,
    current_zoom=None,
    min_context_px=1600,
    bbox_margin_factor=5.0,
    min_zoom=0.04,
    max_zoom=0.90,
):
    try:
        h = max(1.0, float(bbox_display_yxhw[2]))
        w = max(1.0, float(bbox_display_yxhw[3]))
    except Exception:
        h = w = 1.0
    target_h = max(h * float(bbox_margin_factor), float(min_context_px))
    target_w = max(w * float(bbox_margin_factor), float(min_context_px))
    zoom = None
    if canvas_shape_wh and len(canvas_shape_wh) >= 2:
        cw = _coerce_positive_int(canvas_shape_wh[0]); ch = _coerce_positive_int(canvas_shape_wh[1])
        if cw and ch:
            zoom = min(float(ch) / target_h, float(cw) / target_w)
    if zoom is None:
        return float(current_zoom) if current_zoom is not None else None
    return max(float(min_zoom), min(float(max_zoom), float(zoom)))


def set_camera_center_yx(viewer, center_yx):
    y, x = float(center_yx[0]), float(center_yx[1])
    fallback = (y, x)
    try:
        cur = tuple(getattr(getattr(viewer, "camera", None), "center", fallback))
        if len(cur) >= 2:
            out = tuple(cur[:-2]) + (y, x)
        else:
            out = fallback
        viewer.camera.center = out
        return out
    except Exception:
        viewer.camera.center = fallback
        return fallback
