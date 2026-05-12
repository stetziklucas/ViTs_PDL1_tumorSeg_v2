"""Unit tests for single-image report outputs."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import make_report
from scripts.make_report import derive_operator_summary, write_report_summary_markdown


class MakeReportTests(unittest.TestCase):
    def _run_main_smoke(self, with_encoder_provenance: bool) -> tuple[dict, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotations_dir = root / "annotations"
            scribbles_dir = annotations_dir / "scribbles"
            reports_dir = root / "reports"
            overlays_dir = root / "overlays"
            maps_dir = root / "maps"
            masks_dir = root / "masks"
            tile_model_dir = root / "tile_model"
            for d in (scribbles_dir, reports_dir, overlays_dir, maps_dir, masks_dir, tile_model_dir):
                d.mkdir(parents=True, exist_ok=True)

            (scribbles_dir / "IMG_scribble_labels.png").write_bytes(b"x")
            (annotations_dir / "IMG_annotation_meta.json").write_text("{}", encoding="utf-8")
            run_tag = "smoke"
            tiles_dir = root / f"tiles_{run_tag}"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            (tiles_dir / "tile_labels.csv").write_text("image_id,tile_id,label\n", encoding="utf-8")

            args = SimpleNamespace(
                config=root / "base.yaml",
                image_id="IMG",
                annotations_dir=annotations_dir,
                maps_dir=maps_dir,
                tile_maps_dir=None,
                pixel_maps_dir=None,
                masks_dir=masks_dir,
                overlays_dir=overlays_dir,
                reports_dir=reports_dir,
                tile_model_dir=tile_model_dir,
                model_scope="single_image_model",
                shared_model_tag=None,
                training_image_count=None,
                included_training_aliases=None,
            )

            metrics_payload = {"image_id": "IMG", "output_space_note": "note"}
            tile_cv_payload = (
                {
                    "encoder_provenance": {
                        "encoder_id": "current_timm",
                        "encoder_display_name": "Current ViT baseline",
                        "encoder_backend": "timm",
                        "encoder_model_name": "vit_base_patch16_224",
                        "embedding_dim": 768,
                        "embedding_dtype": "float32",
                        "encoder_weight_source": "pretrained",
                    }
                }
                if with_encoder_provenance
                else {}
            )

            with patch.object(make_report, "build_parser") as parser_mock, patch.object(
                make_report, "load_config", return_value={"classes": {"label_encoding": {"Positive_Tumor": 1, "Negative_Tumor": 2, "NonTumor": 3}}}
            ), patch.object(
                make_report, "require_artifacts",
                return_value={
                    "metrics": reports_dir / "metrics.json",
                    "overlay": overlays_dir / "overlay.png",
                    "positive_mask": masks_dir / "positive_mask.png",
                    "pixel_prob_map": maps_dir / "pixel_prob_map.png",
                    "tile_prob_map": maps_dir / "tile_prob_map.png",
                },
            ), patch.object(
                make_report, "optional_artifacts", return_value={"tile_cv_metrics": tile_model_dir / "tile_cv_metrics.json"}
            ), patch.object(
                make_report, "_derive_run_tag", return_value=run_tag
            ), patch.object(
                make_report, "compute_metrics_from_paths",
                return_value={"false_positive_px": 0, "false_negative_px": 0, "precision": 1.0, "sensitivity": 1.0, "f1": 1.0, "training_log_loss_total": 0.0, "training_log_loss_mean": 0.0, "tp_px": 1, "tn_px": 1, "annotated_positive_px": 1, "annotated_negative_px": 1, "annotated_total_px": 2, "class_metrics": {"Positive_Tumor": {"annotated_px": 1, "tp_px": 1, "fn_px": 0, "sensitivity": 1.0}, "Negative_Tumor": {"annotated_px": 1, "tn_px": 1, "fp_px": 0, "specificity": 1.0}, "NonTumor": {"annotated_px": 0, "tn_px": 0, "fp_px": 0, "specificity": 1.0}}},
            ), patch.object(
                make_report, "audit_supervision",
                return_value={"warnings": [], "polygon_counts": {}, "annotated_pixel_counts": {}, "accepted_tile_count": 0, "usable_tile_count": 0, "ignored_tile_count": 0, "ignored_tile_share": 0.0, "tile_label_counts": {}, "tile_label_reason_counts": {}, "ignored_tile_reasons": {}, "selection_source_counts": {}},
            ), patch.object(
                make_report, "generate_verification_overlay", return_value={"verification_overlay_available": False}
            ), patch.object(make_report, "render_pdf", return_value=None), patch.object(make_report, "load_json") as load_json_mock:
                parser_mock.return_value.parse_args.return_value = args
                load_json_mock.side_effect = [metrics_payload, tile_cv_payload]
                make_report.main()

            summary_json = json.loads((reports_dir / "report_summary.json").read_text(encoding="utf-8"))
            summary_md = (reports_dir / "report_summary.md").read_text(encoding="utf-8")
            return summary_json, summary_md

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
                "verification_regions_warning": "none",
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
            self.assertIn("verification warning: none", text)


    def test_markdown_includes_encoder_line_when_provenance_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report_summary.md"
            payload = {"image_id":"I","run_tag":"r","model_scope":"single_image_model","shared_model_tag":None,"training_image_count":None,"included_training_aliases":[],"result_description":"x","evaluation_scope":"x","error_pattern":"x","next_review_focus":"x","working_space_note":"x","development_metrics":{"false_positive_px":0,"false_negative_px":0,"precision":1.0,"sensitivity":1.0,"f1":1.0,"training_log_loss_total":0.1,"training_log_loss_mean":0.1,"tp_px":1,"tn_px":1,"annotated_positive_px":1,"annotated_negative_px":1,"annotated_total_px":2},"class_metrics":{"Positive_Tumor":{"annotated_px":1,"tp_px":1,"fn_px":0,"sensitivity":1.0},"Negative_Tumor":{"annotated_px":1,"tn_px":1,"fp_px":0,"specificity":1.0},"NonTumor":{"annotated_px":0,"tn_px":0,"fp_px":0,"specificity":1.0}},"supervision_audit":{"polygon_counts":{},"annotated_pixel_counts":{},"accepted_tile_count":0,"usable_tile_count":0,"ignored_tile_count":0,"ignored_tile_share":0,"tile_label_counts":{},"tile_label_reason_counts":{},"ignored_tile_reasons":{},"selection_source_counts":{}},"warnings":[],"encoder_provenance":{"encoder_id":"hibou_b","encoder_backend":"hf_transformers","embedding_dim":768}}
            write_report_summary_markdown(path, payload)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Embedding encoder: hibou_b (hibou_b)", text)

    def test_make_report_main_with_encoder_provenance_writes_summary(self) -> None:
        payload, markdown = self._run_main_smoke(with_encoder_provenance=True)
        self.assertEqual(payload["encoder_provenance"]["encoder_id"], "current_timm")
        self.assertIn("Embedding encoder: Current ViT baseline (current_timm)", markdown)
        self.assertIn("Embedding backend: timm", markdown)
        self.assertIn("Embedding dimension: 768", markdown)

    def test_make_report_main_without_encoder_provenance_still_succeeds(self) -> None:
        payload, markdown = self._run_main_smoke(with_encoder_provenance=False)
        self.assertNotIn("encoder_provenance", payload)
        self.assertIn("Embedding encoder: not recorded (n/a)", markdown)

if __name__ == "__main__":
    unittest.main()
