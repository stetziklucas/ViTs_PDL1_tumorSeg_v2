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
from PIL import Image
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


def generate_verification_overlay(
    *,
    image_id: str,
    run_tag: str | None,
    scribble_labels_path: Path,
    positive_mask_path: Path,
    output_dir: Path,
    label_encoding: dict[str, int],
    crop_padding_px: int = 64,
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

    summary: dict[str, Any] = {
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
    }
