"""Unit tests for project report rollup diagnostics."""

from __future__ import annotations

import os
import tempfile
import unittest
import json
from pathlib import Path

from scripts.make_project_report import build_training_summary_markdown, evaluate_run_tag
from scripts.report_metrics import aggregate_class_metrics


class MakeProjectReportTests(unittest.TestCase):
    def test_aggregate_class_metrics(self) -> None:
        rows = [
            {
                "class_metrics": {
                    "Positive_Tumor": {"annotated_px": 10, "tp_px": 6, "fn_px": 4},
                    "Negative_Tumor": {"annotated_px": 8, "tn_px": 7, "fp_px": 1},
                    "NonTumor": {"annotated_px": 12, "tn_px": 9, "fp_px": 3},
                }
            },
            {
                "class_metrics": {
                    "Positive_Tumor": {"annotated_px": 6, "tp_px": 3, "fn_px": 3},
                    "Negative_Tumor": {"annotated_px": 4, "tn_px": 2, "fp_px": 2},
                    "NonTumor": {"annotated_px": 5, "tn_px": 4, "fp_px": 1},
                }
            },
        ]
        aggregated = aggregate_class_metrics(rows)
        self.assertEqual(aggregated["Positive_Tumor"]["tp_px"], 9)
        self.assertEqual(aggregated["Positive_Tumor"]["fn_px"], 7)
        self.assertAlmostEqual(float(aggregated["Positive_Tumor"]["sensitivity"]), 9 / 16)
        self.assertEqual(aggregated["Negative_Tumor"]["fp_px"], 3)
        self.assertEqual(aggregated["NonTumor"]["tn_px"], 13)

    def test_training_summary_markdown_contains_class_and_attention_sections(self) -> None:
        payload = {
            "aggregate_metrics": {
                "false_positive_px": 1,
                "false_negative_px": 1,
                "precision": 0.5,
                "sensitivity": 0.5,
                "f1": 0.5,
                "training_log_loss_total": 5.0,
            },
            "aggregate_class_metrics": {
                "Positive_Tumor": {"annotated_px": 10, "tp_px": 6, "fn_px": 4, "sensitivity": 0.6},
                "Negative_Tumor": {"annotated_px": 8, "tn_px": 7, "fp_px": 1, "specificity": 0.875},
                "NonTumor": {"annotated_px": 6, "tn_px": 5, "fp_px": 1, "specificity": 0.8333},
            },
            "included_runs": [
                {
                    "run_tag": "pf0229",
                    "image_id": "PF0229",
                    "development_metrics": {
                        "false_positive_px": 1,
                        "false_negative_px": 1,
                        "precision": 0.5,
                        "sensitivity": 0.5,
                        "f1": 0.5,
                    },
                    "supervision_audit": {
                        "polygon_counts": {"Positive_Tumor": 2, "Negative_Tumor": 1, "NonTumor": 0},
                        "annotated_pixel_counts": {"Positive_Tumor": 100, "Negative_Tumor": 30, "NonTumor": 20},
                        "tile_label_counts": {"Positive_Context": 6, "Negative_Context": 1, "Ignore": 4},
                        "ignored_tile_count": 4,
                    },
                }
            ],
            "images_needing_attention": [
                {"run_tag": "pf0229", "image_id": "PF0229", "warnings": ["very low usable negative tile support"]}
            ],
            "skipped_runs": [{"run_tag": "pf083", "reason": "missing required directories: masks_dir"}],
        }
        text = build_training_summary_markdown(payload)
        for required in (
            "Aggregate class-specific annotated-region metrics",
            "Per-image breakdown",
            "Images needing attention",
            "Skipped runs",
            "very low usable negative tile support",
        ):
            self.assertIn(required, text)
        self.assertIn("| precision | 0.500 |", text)
        self.assertIn("| NonTumor | 6 | 5 | 1 | specificity | 0.833 |", text)

    def test_project_rollup_skips_incomplete_run_tag_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotations_dir = root / "annotations"
            annotations_dir.mkdir(parents=True)

            cwd = Path.cwd()
            try:
                os.chdir(root)
                result, reason = evaluate_run_tag(
                    run_tag="pf083",
                    annotations_dir=annotations_dir,
                    label_encoding={"Positive_Tumor": 1, "Negative_Tumor": 2, "NonTumor": 3},
                )
            finally:
                os.chdir(cwd)

            self.assertIsNone(result)
            self.assertIsNotNone(reason)
            assert reason is not None
            self.assertIn("missing required directories", reason)

    def test_shared_project_run_uses_report_summary_without_tile_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotations_dir = root / "annotations"
            annotations_dir.mkdir(parents=True)

            run_tag = "proj__pf0229"
            reports_dir = root / "outputs" / f"reports_{run_tag}"
            maps_dir = root / "outputs" / f"maps_{run_tag}"
            maps_fused_dir = root / "outputs" / f"maps_{run_tag}_fused"
            masks_dir = root / "outputs" / f"masks_{run_tag}"
            overlays_dir = root / "outputs" / f"overlays_{run_tag}"
            tiles_dir = root / "outputs" / f"tiles_{run_tag}"
            for path in (reports_dir, maps_dir, maps_fused_dir, masks_dir, overlays_dir, tiles_dir):
                path.mkdir(parents=True, exist_ok=True)

            for file_path in (
                reports_dir / "metrics.json",
                maps_dir / "tile_prob_map.png",
                maps_fused_dir / "pixel_prob_map.png",
                masks_dir / "positive_mask.png",
                overlays_dir / "overlay.png",
                tiles_dir / "tile_labels.csv",
            ):
                file_path.write_bytes(b"ok")

            report_summary = {
                "image_id": "IMG_A",
                "run_tag": run_tag,
                "model_scope": "shared_project_model",
                "shared_model_tag": "proj",
                "training_image_count": 2,
                "development_metrics": {
                    "false_positive_px": 4,
                    "false_negative_px": 3,
                    "precision": 0.7,
                    "sensitivity": 0.8,
                    "f1": 0.75,
                    "training_log_loss_total": 12.3,
                },
                "class_metrics": {
                    "Positive_Tumor": {"annotated_px": 10, "tp_px": 8, "fn_px": 2, "sensitivity": 0.8},
                    "Negative_Tumor": {"annotated_px": 10, "tn_px": 9, "fp_px": 1, "specificity": 0.9},
                    "NonTumor": {"annotated_px": 10, "tn_px": 8, "fp_px": 2, "specificity": 0.8},
                },
                "supervision_audit": {"warnings": [], "usable_tile_count": 5, "ignored_tile_count": 1},
                "warnings": ["review sample"],
            }
            (reports_dir / "report_summary.json").write_text(json.dumps(report_summary), encoding="utf-8")

            cwd = Path.cwd()
            try:
                os.chdir(root)
                result, reason = evaluate_run_tag(
                    run_tag=run_tag,
                    annotations_dir=annotations_dir,
                    label_encoding={"Positive_Tumor": 1, "Negative_Tumor": 2, "NonTumor": 3},
                )
            finally:
                os.chdir(cwd)

            self.assertIsNone(reason)
            assert result is not None
            self.assertEqual(result["model_scope"], "shared_project_model")
            self.assertEqual(result["shared_model_tag"], "proj")
            self.assertEqual(result["development_metrics"]["false_positive_px"], 4)


    def test_training_summary_markdown_includes_encoder_columns(self) -> None:
        payload = {"aggregate_metrics":{"false_positive_px":0,"false_negative_px":0,"precision":1.0,"sensitivity":1.0,"f1":1.0,"training_log_loss_total":1.0},"aggregate_class_metrics":{"Positive_Tumor":{"annotated_px":1,"tp_px":1,"fn_px":0,"sensitivity":1.0},"Negative_Tumor":{"annotated_px":1,"tn_px":1,"fp_px":0,"specificity":1.0},"NonTumor":{"annotated_px":1,"tn_px":1,"fp_px":0,"specificity":1.0}},"included_runs":[{"run_tag":"r","image_id":"i","encoder_provenance":{"encoder_id":"hibou_b","encoder_backend":"hf_transformers","embedding_dim":768},"development_metrics":{"false_positive_px":0,"false_negative_px":0,"precision":1.0,"sensitivity":1.0,"f1":1.0},"supervision_audit":{"polygon_counts":{},"annotated_pixel_counts":{},"tile_label_counts":{},"ignored_tile_count":0}}],"images_needing_attention":[],"skipped_runs":[]}
        text = build_training_summary_markdown(payload)
        self.assertIn("hibou_b / hf_transformers / 768", text)

if __name__ == "__main__":
    unittest.main()
