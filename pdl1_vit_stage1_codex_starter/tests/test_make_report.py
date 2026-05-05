"""Unit tests for single-image report outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.make_report import derive_operator_summary, write_report_summary_markdown


class MakeReportTests(unittest.TestCase):
    def test_operator_summary_derivation_is_deterministic(self) -> None:
        conservative = derive_operator_summary({"false_positive_px": 0, "false_negative_px": 7})
        self.assertEqual(conservative["error_pattern"], "Conservative / false-negative dominant")

        fp_dominant = derive_operator_summary({"false_positive_px": 4, "false_negative_px": 0})
        self.assertEqual(fp_dominant["error_pattern"], "False-positive dominant")

        mixed = derive_operator_summary({"false_positive_px": 2, "false_negative_px": 3})
        self.assertEqual(mixed["error_pattern"], "Mixed false-positive and false-negative errors")

        clean = derive_operator_summary({"false_positive_px": 0, "false_negative_px": 0})
        self.assertEqual(clean["error_pattern"], "No annotated-region errors in the evaluated region")

    def test_single_image_markdown_contains_required_metric_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report_summary.md"
            payload = {
                "image_id": "PF0229",
                "run_tag": "pf0229",
                "model_scope": "shared_project_model",
                "shared_model_tag": "proj",
                "training_image_count": 2,
                "included_training_aliases": ["a", "b"],
                "result_description": "Tumor-enriched PD-L1-positive mask in working image space",
                "evaluation_scope": "Annotated-region development metrics only; not whole-slide validation",
                "error_pattern": "Mixed false-positive and false-negative errors",
                "next_review_focus": "Review both undercalled and overcalled annotated regions before broader reruns.",
                "working_space_note": "working-space note",
                "development_metrics": {
                    "false_positive_px": 1,
                    "false_negative_px": 2,
                    "precision": 0.5,
                    "sensitivity": 0.25,
                    "f1": 0.33,
                    "training_log_loss_total": 12.34,
                    "tp_px": 3,
                    "tn_px": 4,
                    "annotated_positive_px": 5,
                    "annotated_negative_px": 6,
                    "annotated_total_px": 11,
                    "training_log_loss_mean": 1.122,
                    "class_metrics": {
                        "Positive_Tumor": {"annotated_px": 5, "tp_px": 3, "fn_px": 2, "sensitivity": 0.6},
                        "Negative_Tumor": {"annotated_px": 4, "tn_px": 3, "fp_px": 1, "specificity": 0.75},
                        "NonTumor": {"annotated_px": 2, "tn_px": 1, "fp_px": 1, "specificity": 0.5},
                    },
                },
                "class_metrics": {
                    "Positive_Tumor": {"annotated_px": 5, "tp_px": 3, "fn_px": 2, "sensitivity": 0.6},
                    "Negative_Tumor": {"annotated_px": 4, "tn_px": 3, "fp_px": 1, "specificity": 0.75},
                    "NonTumor": {"annotated_px": 2, "tn_px": 1, "fp_px": 1, "specificity": 0.5},
                },
                "supervision_audit": {
                    "polygon_counts": {"Positive_Tumor": 2, "Negative_Tumor": 1, "NonTumor": 1},
                    "annotated_pixel_counts": {"Positive_Tumor": 5, "Negative_Tumor": 4, "NonTumor": 2},
                    "accepted_tile_count": 11,
                    "usable_tile_count": 7,
                    "ignored_tile_count": 4,
                    "ignored_tile_share": 0.36,
                    "tile_label_counts": {"Positive_Context": 6, "Negative_Context": 1, "Ignore": 4},
                    "tile_label_reason_counts": {"dominant_positive": 6, "dominant_negative": 1, "mixed_or_ambiguous": 4},
                    "ignored_tile_reasons": {"mixed_or_ambiguous": 4},
                    "selection_source_counts": {},
                },
                "warnings": ["very low usable negative tile support"],
                "verification_overlay_available": True,
                "verification_overlay_mode": "positive_mask_working_crop",
                "verification_annotation_labels_available": True,
                "verification_prediction_labels_available": True,
                "verification_prediction_labels_path": "/tmp/verification_prediction_labels.png",
                "verification_regions_available": True,
                "verification_region_count": 4,
            }

            write_report_summary_markdown(path, payload)
            text = path.read_text(encoding="utf-8")

            for required in (
                "Operator-facing summary",
                "result_description",
                "evaluation_scope",
                "error_pattern",
                "next_review_focus",
                "model_scope",
                "shared_model_tag",
                "training_image_count",
                "included_training_aliases",
                "Supervision summary",
                "Tile supervision summary",
                "Class-specific annotated-region metrics",
                "Warnings / review focus",
                "false_positive_px",
                "false_negative_px",
                "precision",
                "sensitivity",
                "f1",
                "training_log_loss_total",
            ):
                self.assertIn(required, text)
            self.assertIn("| precision | 0.500 |", text)
            self.assertIn("| training_log_loss_total | 12.340 |", text)
            self.assertIn("verification review mask available: yes", text)
            self.assertIn("verification mode: positive_mask_working_crop", text)
            self.assertIn("verification annotation labels available: yes", text)
            self.assertIn("verification review layers available: annotation labels + class-aware prediction labels", text)
            self.assertIn("verification regions available: yes", text)
            self.assertIn("verification region count: 4", text)


if __name__ == "__main__":
    unittest.main()
