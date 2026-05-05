from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def load_verification_regions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        regions = payload.get('regions', [])
        return regions if isinstance(regions, list) else []
    return []

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
    return f"{cls} | {issue} | {score_name}={sc} | err={int(region.get('error_px') or 0)}"

def viewer_bbox_from_region(region) -> dict[str, Any]:
    bbox = region.get('bbox_annotation_yxhw') or region.get('bbox_working_yxhw') or [0,0,1,1]
    y,x,h,w = [int(v) for v in bbox]
    return {'y': y, 'x': x, 'h': h, 'w': w, 'center_yx': region.get('center_annotation_yx') or [y + h//2, x + w//2]}
