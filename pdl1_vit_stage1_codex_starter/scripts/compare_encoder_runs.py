"""Compare encoder run outputs with aggregate and per-image diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

METRICS = [
    "false_positive_px",
    "false_negative_px",
    "precision",
    "sensitivity",
    "f1",
    "training_log_loss_total",
    "annotated_positive_px",
    "annotated_negative_px",
    "annotated_total_px",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _field(provenance: dict[str, Any] | None, key: str, default: str = "n/a") -> str:
    if not isinstance(provenance, dict):
        return default
    value = provenance.get(key)
    return default if value in (None, "") else str(value)


def _display_name(provenance: dict[str, Any] | None) -> str:
    if not isinstance(provenance, dict):
        return "not recorded"
    return str(provenance.get("encoder_display_name") or provenance.get("encoder_id") or "not recorded")


def _resolve_encoder_provenance(summary: dict[str, Any]) -> dict[str, Any] | None:
    direct = summary.get("encoder_provenance")
    if isinstance(direct, dict):
        return direct
    per_run = summary.get("per_run_encoder_provenance")
    if isinstance(per_run, dict):
        first = next(iter(per_run.values()), None)
        if isinstance(first, dict):
            return first
    return None


def _metric_side_by_side(baseline_agg: dict[str, Any], candidate_agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        b, c = baseline_agg.get(metric), candidate_agg.get(metric)
        out[metric] = {
            "baseline": b,
            "candidate": c,
            "candidate_minus_baseline": c - b if isinstance(b, (int, float)) and isinstance(c, (int, float)) else None,
        }
    return out


def _read_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _find_artifacts(outputs_root: Path, tag: str) -> dict[str, list[Path]]:
    return {
        "reports": sorted(outputs_root.glob(f"reports_{tag}__*/report_summary.json")),
        "emb_meta": sorted(outputs_root.glob(f"embeddings_{tag}__*/embeddings_cache_meta.json")),
        "emb_npy": sorted(outputs_root.glob(f"embeddings_{tag}__*/embeddings.npy")),
        "tile_probs": sorted(outputs_root.glob(f"maps_{tag}__*/tile_probabilities.csv")),
        "pixel_prob_map": sorted(outputs_root.glob(f"maps_{tag}__*_fused/pixel_prob_map.png")),
        "positive_mask": sorted(outputs_root.glob(f"masks_{tag}__*/positive_mask.png")),
        "verification_regions": sorted(outputs_root.glob(f"overlays_{tag}__*/verification_regions.json")),
    }


def _child_key(path: Path) -> str:
    d = path.parent.name
    key = d.split("__", 1)[1] if "__" in d else d
    return key[:-6] if key.endswith("_fused") else key


def _load_tile_probs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tile_id = row.get("tile_id")
            if not tile_id:
                continue
            rows[tile_id] = row
    return rows


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _pair_images(base: dict[str, list[Path]], cand: dict[str, list[Path]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    keys = set()
    for k in ("emb_meta", "tile_probs", "pixel_prob_map", "positive_mask", "verification_regions"):
        keys.update(_child_key(p) for p in base[k])
        keys.update(_child_key(p) for p in cand[k])
    pairs: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        pairs[key] = {"pairing_method": "alias_suffix", "baseline": {}, "candidate": {}, "warnings": []}
        for kind in base:
            bp = next((p for p in base[kind] if _child_key(p) == key), None)
            cp = next((p for p in cand[kind] if _child_key(p) == key), None)
            if bp:
                pairs[key]["baseline"][kind] = str(bp)
            if cp:
                pairs[key]["candidate"][kind] = str(cp)
        if not pairs[key]["baseline"]:
            msg = f"No baseline artifacts matched for image key: {key}"
            pairs[key]["warnings"].append(msg)
            warnings.append(msg)
        if not pairs[key]["candidate"]:
            msg = f"No candidate artifacts matched for image key: {key}"
            pairs[key]["warnings"].append(msg)
            warnings.append(msg)
    return pairs, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tag", required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load_json(args.outputs_root / f"reports_training_{args.baseline_tag}" / "training_summary.json")
    candidate = _load_json(args.outputs_root / f"reports_training_{args.candidate_tag}" / "training_summary.json")
    baseline_agg = baseline.get("aggregate_metrics", {})
    candidate_agg = candidate.get("aggregate_metrics", {})
    side_by_side = _metric_side_by_side(baseline_agg, candidate_agg)

    baseline_encoder_provenance = _resolve_encoder_provenance(baseline)
    candidate_encoder_provenance = _resolve_encoder_provenance(candidate)

    base_artifacts = _find_artifacts(args.outputs_root, args.baseline_tag)
    cand_artifacts = _find_artifacts(args.outputs_root, args.candidate_tag)
    pairs, global_warnings = _pair_images(base_artifacts, cand_artifacts)

    threshold = _to_float(baseline.get("tile_head", {}).get("positive_threshold", 0.60), 0.60)
    per_image: dict[str, Any] = {}

    tile_dir = args.output_dir / "tile_probability_deltas"
    img_dir = args.output_dir / "images"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    for key, pair in pairs.items():
        image_warnings = list(pair["warnings"])
        image_data: dict[str, Any] = {"pairing_method": pair["pairing_method"], "warnings": image_warnings}

        b_meta = pair["baseline"].get("emb_meta")
        c_meta = pair["candidate"].get("emb_meta")
        if b_meta and c_meta:
            bm, cm = _load_json(Path(b_meta)), _load_json(Path(c_meta))
            bshape, cshape = bm.get("embedding_shape"), cm.get("embedding_shape")
            image_data["encoder_swap_verification"] = {
                "baseline_embedding_shape": bshape,
                "candidate_embedding_shape": cshape,
                "embedding_dim_match": (bshape[-1] == cshape[-1]) if isinstance(bshape, list) and isinstance(cshape, list) and bshape and cshape else None,
                "baseline_cache_signature": bm.get("cache_signature"),
                "candidate_cache_signature": cm.get("cache_signature"),
                "cache_signature_match": bm.get("cache_signature") == cm.get("cache_signature"),
                "baseline_tile_manifest_sha256": bm.get("tile_manifest_sha256"),
                "candidate_tile_manifest_sha256": cm.get("tile_manifest_sha256"),
                "tile_manifest_sha256_match": bm.get("tile_manifest_sha256") == cm.get("tile_manifest_sha256"),
                "tile_count_match": bm.get("tile_count") == cm.get("tile_count"),
                "warnings": [],
            }

        if pair["baseline"].get("tile_probs") and pair["candidate"].get("tile_probs"):
            b_rows = _load_tile_probs(Path(pair["baseline"]["tile_probs"]))
            c_rows = _load_tile_probs(Path(pair["candidate"]["tile_probs"]))
            bids, cids = set(b_rows), set(c_rows)
            matched = sorted(bids & cids)
            deltas = []
            flips = 0
            bpos = 0
            cpos = 0
            for tid in matched:
                br, cr = b_rows[tid], c_rows[tid]
                bp = _to_float(br.get("prob_positive"))
                cp = _to_float(cr.get("prob_positive"))
                ba, ca = bp >= threshold, cp >= threshold
                flips += int(ba != ca)
                bpos += int(ba)
                cpos += int(ca)
                deltas.append((tid, bp, cp, abs(cp - bp), ba, ca, br, cr))
            abs_vals = [d[3] for d in deltas]
            corr = None
            if len(deltas) >= 2:
                bvals = np.array([d[1] for d in deltas], dtype=float)
                cvals = np.array([d[2] for d in deltas], dtype=float)
                if np.var(bvals) > 0 and np.var(cvals) > 0:
                    corr = float(np.corrcoef(bvals, cvals)[0, 1])
            top = sorted(deltas, key=lambda x: x[3], reverse=True)[:5]
            image_data["tile_probability_deltas"] = {
                "baseline_tile_count": len(bids),
                "candidate_tile_count": len(cids),
                "matched_tile_count": len(matched),
                "unmatched_baseline_tile_count": len(bids - cids),
                "unmatched_candidate_tile_count": len(cids - bids),
                "mean_abs_prob_delta": float(np.mean(abs_vals)) if abs_vals else 0.0,
                "median_abs_prob_delta": float(np.median(abs_vals)) if abs_vals else 0.0,
                "max_abs_prob_delta": float(np.max(abs_vals)) if abs_vals else 0.0,
                "pearson_correlation": corr,
                "threshold_flip_count": flips,
                "baseline_positive_tile_count": bpos,
                "candidate_positive_tile_count": cpos,
                "candidate_minus_baseline_positive_tile_count": cpos - bpos,
                "top_changed_tiles": [{"tile_id": t[0], "abs_delta": t[3]} for t in top],
            }
            out_csv = tile_dir / f"{_safe_key(key)}_tile_probability_delta.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["tile_id","image_id","tile_x","tile_y","tile_w","tile_h","baseline_prob_positive","candidate_prob_positive","candidate_minus_baseline","abs_delta","baseline_above_threshold","candidate_above_threshold","threshold_flipped"])
                for tid, bp, cp, ad, ba, ca, br, _ in deltas:
                    w.writerow([tid, br.get("image_id", key), br.get("tile_x"), br.get("tile_y"), br.get("tile_w"), br.get("tile_h"), bp, cp, cp-bp, ad, ba, ca, ba!=ca])

        if pair["baseline"].get("pixel_prob_map") and pair["candidate"].get("pixel_prob_map"):
            bimg = _read_png(Path(pair["baseline"]["pixel_prob_map"]))
            cimg = _read_png(Path(pair["candidate"]["pixel_prob_map"]))
            if bimg.shape == cimg.shape:
                delta = np.abs(cimg.astype(float) - bimg.astype(float))
                image_data["pixel_probability_map_deltas"] = {
                    "shape": list(bimg.shape),
                    "mean_abs_pixel_prob_delta": float(np.mean(delta)),
                    "median_abs_pixel_prob_delta": float(np.median(delta)),
                    "max_abs_pixel_prob_delta": float(np.max(delta)),
                    "differing_pixel_count": int(np.count_nonzero(delta)),
                    "differing_pixel_fraction": float(np.count_nonzero(delta) / delta.size),
                    "artifact_note": "Computed from decoded artifact-encoded pixel_prob_map.png values.",
                }
                _write_png(img_dir / f"{_safe_key(key)}_pixel_prob_abs_delta.png", delta.astype(np.uint8))
            else:
                image_warnings.append(f"Pixel probability map shape mismatch for {key}: {bimg.shape} vs {cimg.shape}")

        if pair["baseline"].get("positive_mask") and pair["candidate"].get("positive_mask"):
            bm = _read_png(Path(pair["baseline"]["positive_mask"]))
            cm = _read_png(Path(pair["candidate"]["positive_mask"]))
            if bm.shape == cm.shape:
                bb = bm > 0
                cb = cm > 0
                inter = int(np.logical_and(bb, cb).sum())
                union = int(np.logical_or(bb, cb).sum())
                xor = np.logical_xor(bb, cb)
                xor_px = int(xor.sum())
                bpx, cpx = int(bb.sum()), int(cb.sum())
                dice = (2 * inter / (bpx + cpx)) if (bpx + cpx) else 1.0
                jacc = (inter / union) if union else 1.0
                image_data["positive_mask_deltas"] = {
                    "baseline_positive_px": bpx,
                    "candidate_positive_px": cpx,
                    "intersection_positive_px": inter,
                    "union_positive_px": union,
                    "xor_px": xor_px,
                    "xor_fraction": float(xor_px / xor.size),
                    "dice_similarity": float(dice),
                    "jaccard_similarity": float(jacc),
                    "masks_identical": xor_px == 0,
                }
                _write_png(img_dir / f"{_safe_key(key)}_positive_mask_xor.png", (xor.astype(np.uint8) * 255))
            else:
                image_warnings.append(f"Positive mask shape mismatch for {key}: {bm.shape} vs {cm.shape}")

        if pair["baseline"].get("verification_regions") and pair["candidate"].get("verification_regions"):
            br = _load_json(Path(pair["baseline"]["verification_regions"]))
            cr = _load_json(Path(pair["candidate"]["verification_regions"]))
            bregs = br.get("regions", br if isinstance(br, list) else [])
            cregs = cr.get("regions", cr if isinstance(cr, list) else [])
            def _summ(regs: list[dict[str, Any]]) -> dict[str, Any]:
                sc = Counter(str(r.get("source", "unknown")) for r in regs if isinstance(r, dict))
                issues = Counter(str(r.get("issue", "unknown")) for r in regs if isinstance(r, dict))
                classes = Counter(str(r.get("class_name", "unknown")) for r in regs if isinstance(r, dict))
                return {"count": len(regs), "source_counts": dict(sc), "top_issues": dict(issues.most_common(5)), "class_name_counts": dict(classes)}
            image_data["verification_region_summary"] = {"baseline": _summ(bregs), "candidate": _summ(cregs)}

        per_image[key] = image_data

    enc_verif = {
        "encoder_ids_differ": _field(baseline_encoder_provenance, "encoder_id") != _field(candidate_encoder_provenance, "encoder_id"),
        "model_names_differ": _field(baseline_encoder_provenance, "encoder_model_name") != _field(candidate_encoder_provenance, "encoder_model_name"),
        "per_image": {k: v.get("encoder_swap_verification", {"warnings": v.get("warnings", [])}) for k, v in per_image.items()},
    }

    hard_mask_changed = any(v.get("positive_mask_deltas", {}).get("masks_identical") is False for v in per_image.values() if "positive_mask_deltas" in v)
    tile_changed = any(v.get("tile_probability_deltas", {}).get("max_abs_prob_delta", 0) > 0 for v in per_image.values())
    tile_flip = any(v.get("tile_probability_deltas", {}).get("threshold_flip_count", 0) > 0 for v in per_image.values())
    pixel_changed = any(v.get("pixel_probability_map_deltas", {}).get("differing_pixel_count", 0) > 0 for v in per_image.values())
    metrics_changed = any((vals.get("candidate_minus_baseline") or 0) != 0 for vals in side_by_side.values())
    loss_delta = side_by_side.get("training_log_loss_total", {}).get("candidate_minus_baseline")

    interpretation = [
        f"Encoder swap confirmed: {_field(baseline_encoder_provenance, 'encoder_id')} -> {_field(candidate_encoder_provenance, 'encoder_id')}.",
        f"Same tile manifests were used for both runs: {'yes' if all(v.get('encoder_swap_verification',{}).get('tile_manifest_sha256_match') for v in per_image.values() if v.get('encoder_swap_verification')) else 'no/partial'}.",
        f"Tile probabilities changed: {'yes' if tile_changed else 'no'}.",
        f"Tile threshold decisions changed: {'yes' if tile_flip else 'no'}.",
        f"Pixel probability maps changed: {'yes' if pixel_changed else 'no'}.",
        f"Final positive masks changed: {'yes' if hard_mask_changed else 'no'}.",
        f"Annotated-region hard-mask metrics changed: {'yes' if metrics_changed else 'no'}.",
        f"Continuous training loss changed by {loss_delta}.",
    ]
    if (not hard_mask_changed) and (tile_changed or pixel_changed):
        interpretation.append("The encoder change altered continuous model outputs, but no final hard-mask decisions crossed the current tile/pixel/fusion thresholds in the annotated evaluation regions.")

    caveat = "Annotated-region development metrics only; not whole-slide validation and not clinical performance."
    summary = {
        "benchmark_schema_version": 1,
        "baseline_tag": args.baseline_tag,
        "candidate_tag": args.candidate_tag,
        "baseline_encoder_provenance": baseline_encoder_provenance,
        "candidate_encoder_provenance": candidate_encoder_provenance,
        "aggregate_metrics": side_by_side,
        "encoder_swap_verification": enc_verif,
        "per_image_deltas": per_image,
        "warnings": global_warnings,
        "comparisons": [{"baseline_tag": args.baseline_tag, "candidate_tag": args.candidate_tag, "aggregate_metric_deltas": side_by_side, "per_image_deltas": per_image}],
        "runs": [
            {"role": "baseline", "tag": args.baseline_tag, "encoder_provenance": baseline_encoder_provenance},
            {"role": "candidate", "tag": args.candidate_tag, "encoder_provenance": candidate_encoder_provenance},
        ],
        "interpretation": interpretation,
        "caveat": caveat,
    }

    (args.output_dir / "encoder_comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "artifact_delta_manifest.json").write_text(json.dumps({"pairs": pairs, "warnings": global_warnings}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (args.output_dir / "per_image_delta_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_key", "pairing_method", "threshold_flip_count", "masks_identical", "xor_px", "mean_abs_pixel_prob_delta", "warnings"])
        for k, v in per_image.items():
            w.writerow([
                k,
                v.get("pairing_method"),
                v.get("tile_probability_deltas", {}).get("threshold_flip_count"),
                v.get("positive_mask_deltas", {}).get("masks_identical"),
                v.get("positive_mask_deltas", {}).get("xor_px"),
                v.get("pixel_probability_map_deltas", {}).get("mean_abs_pixel_prob_delta"),
                " | ".join(v.get("warnings", [])),
            ])

    md = ["# Encoder Comparison Summary", "", f"baseline_tag: {args.baseline_tag}", f"candidate_tag: {args.candidate_tag}", "", "## Interpretation", ""]
    md.extend([f"- {x}" for x in interpretation])
    md.extend(["", caveat, "", "## Baseline encoder", "", f"- display_name: {_display_name(baseline_encoder_provenance)}", f"- encoder_id: {_field(baseline_encoder_provenance,'encoder_id')}", f"- backend: {_field(baseline_encoder_provenance,'encoder_backend')}", f"- model_name: {_field(baseline_encoder_provenance,'encoder_model_name')}", f"- embedding_dim: {_field(baseline_encoder_provenance,'embedding_dim')}", "", "## Candidate encoder", "", f"- display_name: {_display_name(candidate_encoder_provenance)}", f"- encoder_id: {_field(candidate_encoder_provenance,'encoder_id')}", f"- backend: {_field(candidate_encoder_provenance,'encoder_backend')}", f"- model_name: {_field(candidate_encoder_provenance,'encoder_model_name')}", f"- embedding_dim: {_field(candidate_encoder_provenance,'embedding_dim')}", "", "## Encoder swap verification", "", f"- encoder_ids_differ: {enc_verif['encoder_ids_differ']}", f"- model_names_differ: {enc_verif['model_names_differ']}", "", "| metric | baseline | candidate | delta |", "| --- | ---: | ---: | ---: |"])
    for metric, values in side_by_side.items():
        md.append(f"| {metric} | {values['baseline']} | {values['candidate']} | {values['candidate_minus_baseline']} |")
    for k, v in per_image.items():
        if v.get("positive_mask_deltas", {}).get("masks_identical"):
            md.extend(["", f"- Final positive masks are identical for this image under current thresholds. ({k})"])
    (args.output_dir / "encoder_comparison_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
