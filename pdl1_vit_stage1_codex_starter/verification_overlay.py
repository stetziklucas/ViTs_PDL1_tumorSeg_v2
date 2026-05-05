"""Helpers for generating annotated-region cropped verification masks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ANNOTATION_LABEL_MAPPING = {
    "background": 0,
    "Positive_Tumor": 1,
    "Negative_Tumor": 2,
    "NonTumor": 3,
    "Ignore": 4,
}
PREDICTION_LABEL_MAPPING = {
    "background": 0,
    "pred_on_positive_tumor": 1,
    "pred_on_negative_tumor": 2,
    "pred_on_nontumor": 3,
    "pred_on_ignore": 4,
    "pred_outside_annotated_roi": 5,
}

import numpy as np
from PIL import Image, ImageOps
from scripts.report_metrics import load_working_supervision_and_prediction_masks

def _crop_bounds(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    return y0, x0, y1 - y0, x1 - x0




def _load_annotation_regions(annotation_metadata_path: Path | None) -> list[dict[str, Any]]:
    if annotation_metadata_path is None or not annotation_metadata_path.exists():
        return []
    payload = json.loads(annotation_metadata_path.read_text(encoding="utf-8"))
    polygons = payload.get("polygons", []) if isinstance(payload, dict) else []
    regions: list[dict[str, Any]] = []
    for idx, poly in enumerate(polygons):
        if not isinstance(poly, dict):
            continue
        vertices = poly.get("vertices") or poly.get("points") or []
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        regions.append({"annotation_index": idx, "class_name": str(poly.get("class_name") or "Unknown"), "vertices": vertices, "source_type": "annotation_polygon"})
    return regions


def _polygon_mask(shape: tuple[int, int], vertices_yx: np.ndarray) -> np.ndarray:
    from PIL import ImageDraw
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    pts = [(float(x), float(y)) for y, x in vertices_yx]
    draw.polygon(pts, fill=1, outline=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def _write_region_preview(base_rgb: np.ndarray, ann_mask: np.ndarray, pred_mask: np.ndarray, out_path: Path, thumb_path: Path) -> None:
    img = base_rgb.copy()
    img = np.clip(img * 0.9 + ann_mask[..., None] * np.array([60, 90, 255])[None, None, :], 0, 255).astype(np.uint8)
    fn = ann_mask & (~pred_mask)
    tp = ann_mask & pred_mask
    fp = (~ann_mask) & pred_mask
    img[tp] = np.array([0, 220, 0], dtype=np.uint8)
    img[fn] = np.array([255, 60, 60], dtype=np.uint8)
    img[fp] = np.array([255, 200, 40], dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(out_path)
    thumb = Image.fromarray(img, mode="RGB").resize((128, 128), Image.Resampling.NEAREST)
    thumb.save(thumb_path)

def generate_verification_overlay(
    *,
    image_id: str,
    run_tag: str | None,
    scribble_labels_path: Path,
    positive_mask_path: Path,
    output_dir: Path,
    label_encoding: dict[str, int],
    crop_padding_px: int = 64,
    annotation_metadata_path: Path | None = None,
    overlay_base_path: Path | None = None,
) -> dict[str, Any]:
    """Generate cropped positive-mask verification artifact for GUI review."""
    aligned = load_working_supervision_and_prediction_masks(
        image_id=image_id,
        scribble_labels_path=scribble_labels_path,
        positive_mask_path=positive_mask_path,
    )
    scribble = aligned["scribble_working"]
    pred = aligned["positive_mask"] > 0
    if scribble.shape != pred.shape:
        raise ValueError(f"Shape mismatch: scribble={scribble.shape} positive_mask={pred.shape}")

    annotated_vals = [
        int(label_encoding[k])
        for k in ("Positive_Tumor", "Negative_Tumor", "NonTumor", "Ignore")
        if k in label_encoding
    ]
    annotated_mask = np.isin(scribble, np.asarray(annotated_vals, dtype=np.uint8))
    bounds = _crop_bounds(annotated_mask, crop_padding_px)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "verification_overlay.png"
    summary_path = output_dir / "verification_overlay_summary.json"
    regions_json_path = output_dir / "verification_regions.json"

    summary: dict[str, Any] = {
        "verification_regions_available": False,
        "verification_regions_path": None,
        "verification_region_count": 0,
        "image_id": image_id,
        "run_tag": run_tag,
        "verification_overlay_available": False,
        "verification_overlay_mode": "positive_mask_working_crop",
        "verification_annotation_labels_available": False,
        "verification_annotation_labels_path": None,
        "verification_prediction_labels_available": False,
        "verification_prediction_labels_path": None,
        "annotation_label_mapping": ANNOTATION_LABEL_MAPPING,
        "prediction_label_mapping": PREDICTION_LABEL_MAPPING,
        "full_image_h": int(scribble.shape[0]),
        "full_image_w": int(scribble.shape[1]),
        "source_positive_mask_path": positive_mask_path.as_posix(),
        "source_scribble_labels_path": scribble_labels_path.as_posix(),
    }
    if bounds is None:
        summary["verification_annotation_labels_available"] = False
        summary["verification_annotation_labels_path"] = None
        summary["annotation_label_mapping"] = ANNOTATION_LABEL_MAPPING
        summary["note"] = "No annotated pixels found in scribble_labels; overlay unavailable."
    else:
        y0, x0, ch, cw = bounds
        pred_crop = pred[y0 : y0 + ch, x0 : x0 + cw]
        overlay = (pred_crop.astype(np.uint8) * 255).astype(np.uint8)
        Image.fromarray(overlay, mode="L").save(overlay_path)

        annotation_labels_path = output_dir / "verification_annotation_labels.png"
        annotation_labels = np.zeros_like(scribble, dtype=np.uint8)
        for class_name, label_val in ANNOTATION_LABEL_MAPPING.items():
            if class_name == "background":
                continue
            src = label_encoding.get(class_name)
            if src is not None:
                annotation_labels[scribble == int(src)] = int(label_val)
        annotation_labels_crop = annotation_labels[y0 : y0 + ch, x0 : x0 + cw]
        Image.fromarray(annotation_labels_crop, mode="L").save(annotation_labels_path)
        prediction_labels_path = output_dir / "verification_prediction_labels.png"
        prediction_labels = np.zeros_like(annotation_labels_crop, dtype=np.uint8)
        pred_pos = pred_crop > 0
        prediction_labels[pred_pos & (annotation_labels_crop == ANNOTATION_LABEL_MAPPING["Positive_Tumor"])] = PREDICTION_LABEL_MAPPING["pred_on_positive_tumor"]
        prediction_labels[pred_pos & (annotation_labels_crop == ANNOTATION_LABEL_MAPPING["Negative_Tumor"])] = PREDICTION_LABEL_MAPPING["pred_on_negative_tumor"]
        prediction_labels[pred_pos & (annotation_labels_crop == ANNOTATION_LABEL_MAPPING["NonTumor"])] = PREDICTION_LABEL_MAPPING["pred_on_nontumor"]
        prediction_labels[pred_pos & (annotation_labels_crop == ANNOTATION_LABEL_MAPPING["Ignore"])] = PREDICTION_LABEL_MAPPING["pred_on_ignore"]
        prediction_labels[pred_pos & (annotation_labels_crop == ANNOTATION_LABEL_MAPPING["background"])] = PREDICTION_LABEL_MAPPING["pred_outside_annotated_roi"]
        Image.fromarray(prediction_labels, mode="L").save(prediction_labels_path)
        summary.update(
            {
                "verification_overlay_available": True,
                "crop_y0": int(y0),
                "crop_x0": int(x0),
                "crop_h": int(ch),
                "crop_w": int(cw),
                "annotated_crop_pixel_count": int(annotated_mask[y0 : y0 + ch, x0 : x0 + cw].sum()),
                "verification_overlay_path": overlay_path.as_posix(),
                "verification_annotation_labels_available": True,
                "verification_annotation_labels_path": annotation_labels_path.as_posix(),
                "verification_prediction_labels_available": True,
                "verification_prediction_labels_path": prediction_labels_path.as_posix(),
                "annotation_label_mapping": ANNOTATION_LABEL_MAPPING,
                "prediction_label_mapping": PREDICTION_LABEL_MAPPING,
                "note": "Cropped working-space positive-mask overlay generated for annotated-region review.",
            }
        )

    if bounds is not None:
        y0, x0, ch, cw = bounds
        pred_crop = pred[y0:y0+ch, x0:x0+cw]
        ann_crop = annotation_labels_crop if 'annotation_labels_crop' in locals() else np.zeros((ch,cw), dtype=np.uint8)
        regions_dir = output_dir / "verification_regions"
        regions_dir.mkdir(parents=True, exist_ok=True)
        if overlay_base_path is not None and overlay_base_path.exists():
            base = np.asarray(Image.open(overlay_base_path).convert("RGB"))
            if base.shape[:2] != pred.shape:
                base = np.full((pred.shape[0], pred.shape[1], 3), 96, dtype=np.uint8)
        else:
            base = np.full((pred.shape[0], pred.shape[1], 3), 96, dtype=np.uint8)
        base_crop = base[y0:y0+ch, x0:x0+cw]
        regions = _load_annotation_regions(annotation_metadata_path)
        if not regions:
            ids = [v for v in np.unique(ann_crop).tolist() if v > 0]
            for ridx, lbl in enumerate(ids):
                ys, xs = np.where(ann_crop == lbl)
                if ys.size == 0: continue
                verts = [[int(ys.min()), int(xs.min())], [int(ys.min()), int(xs.max())], [int(ys.max()), int(xs.max())], [int(ys.max()), int(xs.min())]]
                cname = [k for k,v in ANNOTATION_LABEL_MAPPING.items() if v==int(lbl)][0]
                regions.append({"annotation_index": None, "class_name": cname, "vertices": verts, "source_type":"annotation_component"})
        rows=[]
        for ridx, region in enumerate(regions):
            verts=np.asarray(region["vertices"], dtype=float)
            if verts.shape[1]!=2: continue
            sy = pred.shape[0]/max(1, float(np.max(verts[:,0])+1))
            sx = pred.shape[1]/max(1, float(np.max(verts[:,1])+1))
            vyx=np.column_stack([verts[:,0]*sy - y0, verts[:,1]*sx - x0])
            mask = _polygon_mask((ch,cw), vyx)
            if not np.any(mask):
                continue
            pred_pos = pred_crop > 0
            annotated_px=int(mask.sum()); pred_positive_px=int((pred_pos & mask).sum())
            cname = region.get("class_name","Unknown")
            if cname=="Positive_Tumor":
                correct_px=pred_positive_px; error_px=annotated_px-correct_px; score_name="sensitivity"; score=(correct_px/annotated_px) if annotated_px else None; issue="positive_annotation_missed_by_prediction" if error_px>0 else "positive_annotation_detected"
            elif cname in {"Negative_Tumor","NonTumor"}:
                correct_px=int((~pred_pos & mask).sum()); error_px=pred_positive_px; score_name="specificity"; score=(correct_px/annotated_px) if annotated_px else None; issue="false_positive_in_negative_context" if error_px>0 else "negative_context_clean"
            else:
                correct_px=0; error_px=pred_positive_px; score_name="ignored"; score=None; issue="ignored_annotation_contains_prediction" if pred_positive_px>0 else "ignored_annotation_clean"
            ys,xs=np.where(mask)
            bbox=[int(ys.min()+y0),int(xs.min()+x0),int(ys.max()-ys.min()+1),int(xs.max()-xs.min()+1)]
            preview=regions_dir/f"region_{ridx:04d}_preview.png"; thumb=regions_dir/f"region_{ridx:04d}_thumb.png"
            _write_region_preview(base_crop, mask, pred_pos, preview, thumb)
            rows.append({"annotation_index":region.get("annotation_index"),"source_type":region.get("source_type","annotation_polygon"),"class_name":cname,"annotated_px":annotated_px,"pred_positive_px":pred_positive_px,"correct_px":int(correct_px),"error_px":int(error_px),"score_name":score_name,"score":score,"review_priority":int(error_px),"issue":issue,"bbox_working_yxhw":[int(ys.min()),int(xs.min()),int(ys.max()-ys.min()+1),int(xs.max()-xs.min()+1)],"bbox_annotation_yxhw":bbox,"center_annotation_yx":[int((bbox[0]+bbox[0]+bbox[2]-1)/2),int((bbox[1]+bbox[1]+bbox[3]-1)/2)],"preview_path":preview.as_posix(),"thumbnail_path":thumb.as_posix()})
        regions_json_path.write_text(json.dumps({"image_id": image_id, "run_tag": run_tag, "regions": rows}, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        summary["verification_regions_available"]=True
        summary["verification_regions_path"]=regions_json_path.as_posix()
        summary["verification_region_count"]=len(rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "verification_overlay_available": bool(summary.get("verification_overlay_available")),
        "verification_overlay_path": overlay_path.as_posix() if overlay_path.exists() else None,
        "verification_overlay_summary_path": summary_path.as_posix(),
        "verification_overlay_mode": "positive_mask_working_crop",
        "verification_annotation_labels_available": bool(summary.get("verification_annotation_labels_available")),
        "verification_annotation_labels_path": summary.get("verification_annotation_labels_path"),
        "verification_prediction_labels_available": bool(summary.get("verification_prediction_labels_available")),
        "verification_prediction_labels_path": summary.get("verification_prediction_labels_path"),
        "annotation_label_mapping": summary.get("annotation_label_mapping"),
        "prediction_label_mapping": summary.get("prediction_label_mapping"),
        "verification_overlay_summary": summary,
        "crop_y0": summary.get("crop_y0"),
        "crop_x0": summary.get("crop_x0"),
        "crop_h": summary.get("crop_h"),
        "crop_w": summary.get("crop_w"),
        "verification_regions_available": bool(summary.get("verification_regions_available")),
        "verification_regions_path": summary.get("verification_regions_path"),
        "verification_region_count": int(summary.get("verification_region_count") or 0),
    }
