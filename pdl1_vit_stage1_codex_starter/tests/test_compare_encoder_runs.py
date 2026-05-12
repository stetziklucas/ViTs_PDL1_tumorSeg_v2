import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import compare_encoder_runs


def _wjson(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _wpng(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array(arr, dtype=np.uint8)).save(path)


def _wcsv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _run(monkeypatch, out, target):
    monkeypatch.setattr("sys.argv", ["x", "--baseline-tag", "a", "--candidate-tag", "b", "--outputs-root", str(out), "--output-dir", str(target)])
    compare_encoder_runs.main()


def test_compare_encoder_diagnostics(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    base = {
        "aggregate_metrics": {"precision": 0.1, "false_positive_px": 1, "false_negative_px": 2, "sensitivity": 0.2, "f1": 0.3, "training_log_loss_total": 54.164, "annotated_positive_px": 10, "annotated_negative_px": 20, "annotated_total_px": 30},
        "encoder_provenance": {"encoder_id": "current_timm", "encoder_model_name": "vit_base_patch16_224", "encoder_backend": "timm", "embedding_dim": 768},
        "tile_head": {"positive_threshold": 0.60},
    }
    cand = {
        "aggregate_metrics": {"precision": 0.1, "false_positive_px": 1, "false_negative_px": 2, "sensitivity": 0.2, "f1": 0.3, "training_log_loss_total": 54.032, "annotated_positive_px": 10, "annotated_negative_px": 20, "annotated_total_px": 30},
        "encoder_provenance": {"encoder_id": "hibou_b", "encoder_model_name": "histai/hibou-b", "encoder_backend": "hf_transformers", "embedding_dim": 768},
        "tile_head": {"positive_threshold": 0.60},
    }
    _wjson(out / "reports_training_a/training_summary.json", base)
    _wjson(out / "reports_training_b/training_summary.json", cand)

    _wjson(out / "embeddings_a__img1/embeddings_cache_meta.json", {"embedding_shape": [2, 768], "cache_signature": "A", "tile_manifest_sha256": "SAME", "tile_count": 2})
    _wjson(out / "embeddings_b__img1/embeddings_cache_meta.json", {"embedding_shape": [2, 768], "cache_signature": "B", "tile_manifest_sha256": "SAME", "tile_count": 2})

    rows_a = [
        {"tile_id": "t1", "image_id": "img1", "tile_x": 0, "tile_y": 0, "tile_w": 16, "tile_h": 16, "prob_positive": 0.55},
        {"tile_id": "t2", "image_id": "img1", "tile_x": 16, "tile_y": 0, "tile_w": 16, "tile_h": 16, "prob_positive": 0.58},
    ]
    rows_b = [
        {"tile_id": "t1", "image_id": "img1", "tile_x": 0, "tile_y": 0, "tile_w": 16, "tile_h": 16, "prob_positive": 0.57},
        {"tile_id": "t2", "image_id": "img1", "tile_x": 16, "tile_y": 0, "tile_w": 16, "tile_h": 16, "prob_positive": 0.62},
    ]
    _wcsv(out / "maps_a__img1/tile_probabilities.csv", rows_a)
    _wcsv(out / "maps_b__img1/tile_probabilities.csv", rows_b)

    _wpng(out / "maps_a__img1_fused/pixel_prob_map.png", [[0, 10, 20], [0, 0, 0], [30, 40, 50]])
    _wpng(out / "maps_b__img1_fused/pixel_prob_map.png", [[0, 11, 20], [0, 0, 1], [30, 42, 50]])
    _wpng(out / "masks_a__img1/positive_mask.png", [[0, 255, 0], [0, 0, 0], [255, 0, 0]])
    _wpng(out / "masks_b__img1/positive_mask.png", [[0, 255, 0], [0, 0, 0], [255, 0, 0]])

    _wjson(out / "overlays_a__img1/verification_regions.json", {"regions": [{"source": "auto", "issue": "fp", "class_name": "tumor", "score": 0.2}]})
    _wjson(out / "overlays_b__img1/verification_regions.json", {"regions": [{"source": "auto", "issue": "fp", "class_name": "tumor", "score": 0.3}]})

    target = out / "cmp"
    _run(monkeypatch, out, target)

    assert (target / "encoder_comparison_summary.json").exists()
    assert (target / "encoder_comparison_summary.md").exists()
    assert (target / "per_image_delta_summary.csv").exists()
    assert (target / "artifact_delta_manifest.json").exists()
    assert (target / "tile_probability_deltas/img1_tile_probability_delta.csv").exists()
    assert (target / "images/img1_pixel_prob_abs_delta.png").exists()
    assert (target / "images/img1_positive_mask_xor.png").exists()

    payload = json.loads((target / "encoder_comparison_summary.json").read_text(encoding="utf-8"))
    assert payload["benchmark_schema_version"] == 1
    assert payload["encoder_swap_verification"]["encoder_ids_differ"] is True
    assert payload["encoder_swap_verification"]["per_image"]["img1"]["tile_manifest_sha256_match"] is True
    assert payload["per_image_deltas"]["img1"]["tile_probability_deltas"]["threshold_flip_count"] == 1
    assert payload["per_image_deltas"]["img1"]["pixel_probability_map_deltas"]["differing_pixel_count"] == 3
    assert payload["per_image_deltas"]["img1"]["positive_mask_deltas"]["masks_identical"] is True
    assert payload["runs"][0]["role"] == "baseline"
    assert payload["comparisons"][0]["candidate_tag"] == "b"

    md = (target / "encoder_comparison_summary.md").read_text(encoding="utf-8")
    assert "Annotated-region development metrics only" in md
    assert "## Encoder swap verification" in md
    assert "continuous model outputs" in md
    assert "no final hard-mask decisions crossed" in md


def test_compare_missing_artifacts_warns(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    _wjson(out / "reports_training_a/training_summary.json", {"aggregate_metrics": {}, "encoder_provenance": {"encoder_id": "a"}})
    _wjson(out / "reports_training_b/training_summary.json", {"aggregate_metrics": {}, "encoder_provenance": {"encoder_id": "b"}})
    target = out / "cmp2"
    _run(monkeypatch, out, target)
    payload = json.loads((target / "encoder_comparison_summary.json").read_text(encoding="utf-8"))
    assert "warnings" in payload
    assert (target / "encoder_comparison_summary.md").exists()
