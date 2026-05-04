"""Helpers for annotated-region development metrics used by Stage 1 reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload)!r}")
    return payload


def load_trusted_large_png_mask(path: Path, trusted_root: Path) -> np.ndarray:
    """Load a repo-owned PNG while safely bypassing PIL large-image limits."""
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    resolved_path = path.resolve()
    resolved_root = trusted_root.resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing large-mask trust bypass outside trusted root: {resolved_path}")
    if resolved_path.suffix.lower() != ".png":
        raise ValueError(f"Trusted large-mask loader only supports PNG: {resolved_path}")

    prior_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        arr = np.asarray(Image.open(resolved_path))
    finally:
        Image.MAX_IMAGE_PIXELS = prior_limit

    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {resolved_path}, got shape={arr.shape}")
    return arr.astype(np.uint8)


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a uint8 2D mask and convert to binary (>0)."""
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {path}, got shape={arr.shape}")
    return (arr > 0).astype(np.uint8)


def load_probability_map(path: Path) -> np.ndarray:
    """Load a uint8 probability map and decode to [0, 1] float32."""
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Probability map must be 2D: {path}, got shape={arr.shape}")
    return arr.astype(np.float32) / 255.0


def _ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def reconcile_mask_to_target(mask: np.ndarray, target_shape_hw: tuple[int, int], image_id: str) -> tuple[np.ndarray, str]:
    """Resize mask to target shape in a controlled way if needed."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask for image_id={image_id}, got shape={mask.shape}")
    if mask.shape == target_shape_hw:
        return mask.astype(np.uint8), "identity"

    src_h, src_w = mask.shape
    tgt_h, tgt_w = target_shape_hw
    scale_y = float(tgt_h) / float(src_h)
    scale_x = float(tgt_w) / float(src_w)
    if abs(scale_y - scale_x) > 0.01:
        raise ValueError(
            f"Refusing anisotropic mask resize for image_id={image_id}: source={mask.shape} target={target_shape_hw}"
        )

    resized = Image.fromarray(mask.astype(np.uint8), mode="L").resize((tgt_w, tgt_h), resample=Image.Resampling.NEAREST)
    out = np.asarray(resized).astype(np.uint8)
    return out, "nearest_uniform_scale"


def compute_development_metrics(
    *,
    scribble_labels: np.ndarray,
    positive_mask: np.ndarray,
    pixel_prob_map: np.ndarray,
    label_encoding: dict[str, int],
) -> dict[str, float | int]:
    """Compute annotated-region development metrics for one image."""
    positive_code = int(label_encoding["Positive_Tumor"])
    negative_tumor_code = int(label_encoding["Negative_Tumor"])
    nontumor_code = int(label_encoding["NonTumor"])

    # Stage 1 remains a binary positive-mask system:
    #   positive class => Positive_Tumor
    #   negative class => Negative_Tumor + NonTumor
    # The class split below is for annotated-region diagnostics only.
    gt_positive = scribble_labels == positive_code
    gt_negative_tumor = scribble_labels == negative_tumor_code
    gt_nontumor = scribble_labels == nontumor_code
    gt_negative = gt_negative_tumor | gt_nontumor
    annotated = gt_positive | gt_negative

    pred_positive = positive_mask > 0
    pred_negative = ~pred_positive

    tp = int(np.logical_and(pred_positive, gt_positive).sum())
    fp = int(np.logical_and(pred_positive, gt_negative).sum())
    fn = int(np.logical_and(pred_negative, gt_positive).sum())
    tn = int(np.logical_and(pred_negative, gt_negative).sum())

    annotated_total = int(annotated.sum())
    annotated_positive = int(gt_positive.sum())
    annotated_negative = int(gt_negative.sum())
    annotated_negative_tumor = int(gt_negative_tumor.sum())
    annotated_nontumor = int(gt_nontumor.sum())

    precision = _ratio(tp, tp + fp)
    sensitivity = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, (2 * tp) + fp + fn)

    p = np.clip(pixel_prob_map[annotated], 1e-6, 1.0 - 1e-6)
    y = gt_positive[annotated].astype(np.float32)
    log_loss_total = float((-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).sum())
    log_loss_mean = _ratio(log_loss_total, annotated_total)

    positive_tp = int(np.logical_and(pred_positive, gt_positive).sum())
    positive_fn = int(np.logical_and(pred_negative, gt_positive).sum())
    negative_tumor_tn = int(np.logical_and(pred_negative, gt_negative_tumor).sum())
    negative_tumor_fp = int(np.logical_and(pred_positive, gt_negative_tumor).sum())
    nontumor_tn = int(np.logical_and(pred_negative, gt_nontumor).sum())
    nontumor_fp = int(np.logical_and(pred_positive, gt_nontumor).sum())

    return {
        "tp_px": tp,
        "tn_px": tn,
        "false_positive_px": fp,
        "false_negative_px": fn,
        "annotated_positive_px": annotated_positive,
        "annotated_negative_px": annotated_negative,
        "annotated_total_px": annotated_total,
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "f1": float(f1),
        "training_log_loss_total": float(log_loss_total),
        "training_log_loss_mean": float(log_loss_mean),
        "class_metrics": {
            "Positive_Tumor": {
                "annotated_px": annotated_positive,
                "tp_px": positive_tp,
                "fn_px": positive_fn,
                "sensitivity": float(_ratio(positive_tp, positive_tp + positive_fn)),
            },
            "Negative_Tumor": {
                "annotated_px": annotated_negative_tumor,
                "tn_px": negative_tumor_tn,
                "fp_px": negative_tumor_fp,
                "specificity": float(_ratio(negative_tumor_tn, negative_tumor_tn + negative_tumor_fp)),
            },
            "NonTumor": {
                "annotated_px": annotated_nontumor,
                "tn_px": nontumor_tn,
                "fp_px": nontumor_fp,
                "specificity": float(_ratio(nontumor_tn, nontumor_tn + nontumor_fp)),
            },
        },
    }


def compute_metrics_from_paths(
    *,
    image_id: str,
    scribble_labels_path: Path,
    positive_mask_path: Path,
    pixel_prob_map_path: Path,
    label_encoding: dict[str, int],
) -> dict[str, Any]:
    """Load and reconcile artifacts, then compute per-image report metrics."""
    working = load_working_supervision_and_prediction_masks(
        image_id=image_id,
        scribble_labels_path=scribble_labels_path,
        positive_mask_path=positive_mask_path,
    )
    scribble_raw = working["scribble_raw"]
    positive_mask = working["positive_mask"]
    reconciled_scribble = working["scribble_working"]
    scribble_transform = working["scribble_transform"]
    pixel_prob_map = load_probability_map(pixel_prob_map_path)

    target_shape = positive_mask.shape
    if pixel_prob_map.shape != target_shape:
        raise ValueError(
            f"pixel_prob_map shape mismatch for image_id={image_id}: "
            f"pixel_prob_map={pixel_prob_map.shape} positive_mask={target_shape}"
        )

    metrics = compute_development_metrics(
        scribble_labels=reconciled_scribble,
        positive_mask=positive_mask,
        pixel_prob_map=pixel_prob_map,
        label_encoding=label_encoding,
    )
    return {
        "image_id": image_id,
        "scribble_transform": scribble_transform,
        "working_shape_hw": [int(target_shape[0]), int(target_shape[1])],
        **metrics,
    }


def load_working_supervision_and_prediction_masks(
    *,
    image_id: str,
    scribble_labels_path: Path,
    positive_mask_path: Path,
) -> dict[str, Any]:
    """Load source scribbles and aligned working-space masks used by reports."""
    annotations_root = scribble_labels_path.parent.parent
    scribble_raw = load_trusted_large_png_mask(scribble_labels_path, annotations_root)
    positive_mask = load_binary_mask(positive_mask_path)
    target_shape = positive_mask.shape
    reconciled_scribble, scribble_transform = reconcile_mask_to_target(scribble_raw, target_shape, image_id=image_id)
    return {
        "scribble_raw": scribble_raw,
        "positive_mask": positive_mask,
        "scribble_working": reconciled_scribble,
        "scribble_transform": scribble_transform,
        "working_shape_hw": [int(target_shape[0]), int(target_shape[1])],
    }


def aggregate_micro_average(metrics_by_image: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate per-image metrics with summed counts and micro-averaged derived metrics."""
    tp = int(sum(int(row.get("tp_px", 0)) for row in metrics_by_image))
    tn = int(sum(int(row.get("tn_px", 0)) for row in metrics_by_image))
    fp = int(sum(int(row.get("false_positive_px", 0)) for row in metrics_by_image))
    fn = int(sum(int(row.get("false_negative_px", 0)) for row in metrics_by_image))
    annotated_positive = int(sum(int(row.get("annotated_positive_px", 0)) for row in metrics_by_image))
    annotated_negative = int(sum(int(row.get("annotated_negative_px", 0)) for row in metrics_by_image))
    annotated_total = int(sum(int(row.get("annotated_total_px", 0)) for row in metrics_by_image))
    log_loss_total = float(sum(float(row.get("training_log_loss_total", 0.0)) for row in metrics_by_image))

    precision = _ratio(tp, tp + fp)
    sensitivity = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, (2 * tp) + fp + fn)
    log_loss_mean = _ratio(log_loss_total, annotated_total)

    return {
        "tp_px": tp,
        "tn_px": tn,
        "false_positive_px": fp,
        "false_negative_px": fn,
        "annotated_positive_px": annotated_positive,
        "annotated_negative_px": annotated_negative,
        "annotated_total_px": annotated_total,
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "f1": float(f1),
        "training_log_loss_total": float(log_loss_total),
        "training_log_loss_mean": float(log_loss_mean),
    }


def aggregate_class_metrics(metrics_by_image: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate class-specific annotated-region diagnostics across images."""
    totals: dict[str, dict[str, float | int]] = {
        "Positive_Tumor": {"annotated_px": 0, "tp_px": 0, "fn_px": 0},
        "Negative_Tumor": {"annotated_px": 0, "tn_px": 0, "fp_px": 0},
        "NonTumor": {"annotated_px": 0, "tn_px": 0, "fp_px": 0},
    }
    for row in metrics_by_image:
        class_metrics = row.get("class_metrics", {})
        for class_name, class_totals in totals.items():
            src = class_metrics.get(class_name, {})
            for key in class_totals:
                class_totals[key] = int(class_totals[key]) + int(src.get(key, 0))

    positive = totals["Positive_Tumor"]
    negative_tumor = totals["Negative_Tumor"]
    nontumor = totals["NonTumor"]

    positive["sensitivity"] = float(_ratio(int(positive["tp_px"]), int(positive["tp_px"]) + int(positive["fn_px"])))
    negative_tumor["specificity"] = float(
        _ratio(int(negative_tumor["tn_px"]), int(negative_tumor["tn_px"]) + int(negative_tumor["fp_px"]))
    )
    nontumor["specificity"] = float(_ratio(int(nontumor["tn_px"]), int(nontumor["tn_px"]) + int(nontumor["fp_px"])))
    return totals
