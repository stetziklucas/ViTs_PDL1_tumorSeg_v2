"""History helpers for shared-project Stage 1 workflow browsing."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from display_format import format_display_float

def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else None

def _parse_ts(val: Any) -> datetime:
    if not isinstance(val,str) or not val.strip():
        return datetime.min.replace(tzinfo=timezone.utc)
    c=val.strip().replace('Z','+00:00')
    try:
        d=datetime.fromisoformat(c)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)

def _best_ts(summary: dict[str,Any], run: dict[str,Any]|None, p: Path)->str:
    for src in (summary or {}, run or {}):
        for k in ('ended_at_utc','generated_at_utc','started_at_utc','timestamp_utc','timestamp'):
            v=src.get(k)
            if isinstance(v,str) and v.strip(): return v
    return datetime.fromtimestamp(p.stat().st_mtime,tz=timezone.utc).replace(microsecond=0).isoformat()

def discover_project_summaries(outputs_root: Path=Path('outputs'), current_image_id: str|None=None)->list[dict[str,Any]]:
    rows=[]
    for d in sorted(outputs_root.glob('reports_training_*')):
        if not (d / 'training_summary.json').exists():
            continue
        js=_read_json(d/'training_summary.json') or {}
        cases=_read_json(d/'stage1_project_cases.json') or {}
        run=_read_json(d/'stage1_project_run_summary.json') or {}
        tag=str(js.get('project_tag') or d.name.replace('reports_training_',''))
        included=cases.get('included_ready_cases',[]) if isinstance(cases.get('included_ready_cases'),list) else []
        skipped=cases.get('skipped_cases',[]) if isinstance(cases.get('skipped_cases'),list) else []
        current=None
        cur_inc=False
        if current_image_id:
            for r in included:
                if r.get('image_id')==current_image_id:
                    cur_inc=True
                    alias=r.get('alias')
                    if alias:
                        current=(outputs_root/f'reports_{tag}__{alias}'/'report_summary.md').as_posix()
        agg=js.get('aggregate_metrics',{}) if isinstance(js.get('aggregate_metrics'),dict) else {}
        rows.append({'project_tag':tag,'timestamp_utc':_best_ts(js,run,d/'training_summary.json'),'included_ready_case_count':len(included),'skipped_case_count':len(skipped),'aggregate_precision':agg.get('precision'),'aggregate_sensitivity':agg.get('sensitivity'),'aggregate_f1':agg.get('f1'),'training_summary_md':(d/'training_summary.md').as_posix(),'training_summary_json':(d/'training_summary.json').as_posix(),'current_image_included':cur_inc,'current_image_child_report_md':current,'encoder_provenance':js.get('encoder_provenance')})
    return sorted(rows,key=lambda r:(_parse_ts(r.get('timestamp_utc')),str(r.get('project_tag'))), reverse=True)

def discover_current_image_shared_reports(image_id: str, outputs_root: Path=Path('outputs'))->list[dict[str,Any]]:
    rows=[]
    for p in outputs_root.glob('reports_*/report_summary.json'):
        js=_read_json(p) or {}
        if js.get('image_id')!=image_id or js.get('model_scope')!='shared_project_model':
            continue
        parent=p.parent.name
        tag=js.get('shared_model_tag') or (parent.split('__')[0].replace('reports_',''))
        dev=js.get('development_metrics',{}) if isinstance(js.get('development_metrics'),dict) else {}
        overlay_path = js.get("verification_overlay_path")
        if not isinstance(overlay_path, str) or not overlay_path.strip():
            candidate = p.parent.parent / "overlays" / "verification_overlay.png"
            if not candidate.exists():
                run_suffix = p.parent.name.replace("reports_", "", 1)
                candidate = outputs_root / f"overlays_{run_suffix}" / "verification_overlay.png"
            overlay_path = candidate.as_posix() if candidate.exists() else None
        overlay_available = bool(js.get("verification_overlay_available"))
        if not overlay_available and isinstance(overlay_path, str):
            overlay_available = Path(overlay_path).exists()
        ann_path = js.get("verification_annotation_labels_path")
        ann_available = bool(js.get("verification_annotation_labels_available"))
        if not ann_available and isinstance(ann_path, str):
            ann_available = Path(ann_path).exists()
        pred_labels_path = js.get("verification_prediction_labels_path")
        pred_labels_available = bool(js.get("verification_prediction_labels_available"))
        if not pred_labels_available and isinstance(pred_labels_path, str):
            pred_labels_available = Path(pred_labels_path).exists()
        rows.append({
            'project_tag':str(tag),
            'timestamp_utc':_best_ts(js,None,p),
            'report_summary_md':(p.parent/'report_summary.md').as_posix(),
            'report_summary_json': p.as_posix(),
            'f1':dev.get('f1'),
            'verification_overlay_path':overlay_path,
            'verification_overlay_available':overlay_available,
            'verification_overlay_summary_path': js.get("verification_overlay_summary_path"),
            'verification_overlay_mode': js.get("verification_overlay_mode"),
            'verification_annotation_labels_path': ann_path,
            'verification_annotation_labels_available': ann_available,
            'verification_prediction_labels_path': pred_labels_path,
            'verification_prediction_labels_available': pred_labels_available,
            'crop_y0': js.get("crop_y0"),
            'crop_x0': js.get("crop_x0"),
            'crop_h': js.get("crop_h"),
            'crop_w': js.get("crop_w"),
            'verification_regions_available': bool(js.get('verification_regions_available')) ,
            'verification_regions_path': js.get('verification_regions_path'),
            'verification_region_count': int(js.get('verification_region_count') or 0),
            'verification_regions_warning': js.get('verification_regions_warning'),'encoder_provenance':js.get('encoder_provenance'),
        })
    return sorted(rows,key=lambda r:(_parse_ts(r.get('timestamp_utc')),str(r.get('project_tag'))), reverse=True)

def format_project_summary_label(entry: dict[str,Any])->str:
    ts=_parse_ts(entry.get('timestamp_utc')).strftime('%Y-%m-%d %H:%M')
    f1t=f"agg F1 {format_display_float(entry.get('aggregate_f1'))}"
    enc=(entry.get('encoder_provenance') or {})
    enc_label=enc.get('encoder_display_name') or enc.get('encoder_id') or 'encoder n/a'
    return f"{entry.get('project_tag','unknown')} | {enc_label} | {f1t}"

def format_current_image_report_label(entry: dict[str,Any])->str:
    ts=_parse_ts(entry.get('timestamp_utc')).strftime('%Y-%m-%d %H:%M')
    f1t=f"shared F1 {format_display_float(entry.get('f1'))}"
    enc=(entry.get('encoder_provenance') or {})
    enc_label=enc.get('encoder_display_name') or enc.get('encoder_id') or 'encoder n/a'
    return f"{entry.get('project_tag','unknown')} | {enc_label} | {f1t}"

def auto_select_latest_indices(project_entries:list[dict[str,Any]], image_entries:list[dict[str,Any]])->tuple[int|None,int|None]:
    return (0 if project_entries else None, 0 if image_entries else None)
