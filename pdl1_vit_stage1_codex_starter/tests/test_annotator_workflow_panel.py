import unittest
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import numpy as np
from apps.annotator import (
    default_project_tag,
    build_stage1_project_command,
    compact_path_label,
    resolve_verification_overlay_path,
    resolve_verification_annotation_labels_path,
    resolve_verification_prediction_labels_path,
    verification_overlay_translate,
    verification_mask_layer_kwargs,
    build_polygon_review_face_colors,
)

class AnnotatorWorkflowPanelTests(unittest.TestCase):
    def test_auto_generated_project_tag(self):
        self.assertEqual(default_project_tag(datetime(2026,1,2,3,4,5,tzinfo=timezone.utc)), 'training_20260102_030405')

    def test_project_runner_command(self):
        cmd=build_stage1_project_command(config_path=Path('config/base.yaml'), project_tag='training_x', raw_dir=Path('data/raw'), annotations_dir=Path('data/annotations'), outputs_root=Path('outputs'), models_root=Path('models'))
        text=' '.join(cmd)
        self.assertIn('scripts/run_stage1_project.py', text)
        self.assertIn('--discover-ready-cases', text)

    def test_compact_path_label(self):
        self.assertEqual(compact_path_label("short/path.md", max_chars=20), "short/path.md")
        self.assertTrue(compact_path_label("/very/long/path/to/file/report_summary.md", max_chars=20).startswith("..."))

    def test_resolve_verification_overlay_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "verification_overlay.png"
            p.write_bytes(b"x")
            self.assertEqual(resolve_verification_overlay_path({"verification_overlay_path": p.as_posix()}), p)
            self.assertIsNone(resolve_verification_overlay_path({"verification_overlay_path": (Path(d) / "missing.png").as_posix()}))
    def test_verification_overlay_translate_defaults(self):
        self.assertEqual(verification_overlay_translate({"crop_y0": 10, "crop_x0": 20}), (10, 20))
        self.assertEqual(verification_overlay_translate({"crop_y0": None}), (0, 0))
    def test_verification_mask_layer_kwargs_use_selected_report_crop(self):
        kwargs = verification_mask_layer_kwargs(np.ones((2, 3), dtype=np.uint8), {"crop_y0": 11, "crop_x0": 13})
        self.assertEqual(kwargs["name"], "verification_prediction_labels")
        self.assertEqual(kwargs["translate"], (11, 13))
        self.assertEqual(kwargs["opacity"], 0.88)
    def test_polygon_review_face_colors_lighten_not_remove(self):
        face = build_polygon_review_face_colors(["Positive_Tumor", "Negative_Tumor"], alpha=0.12)
        self.assertEqual(face.shape, (2, 4))
        self.assertAlmostEqual(float(face[0, 3]), 0.12, places=4)
        self.assertAlmostEqual(float(face[1, 3]), 0.12, places=4)

    def test_resolve_verification_annotation_labels_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "verification_annotation_labels.png"
            p.write_bytes(b"x")
            self.assertEqual(resolve_verification_annotation_labels_path({"verification_annotation_labels_path": p.as_posix()}), p)
            self.assertIsNone(resolve_verification_annotation_labels_path({"verification_annotation_labels_path": (Path(d) / "missing.png").as_posix()}))
    def test_resolve_verification_prediction_labels_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "verification_prediction_labels.png"
            p.write_bytes(b"x")
            self.assertEqual(resolve_verification_prediction_labels_path({"verification_prediction_labels_path": p.as_posix()}), p)
            self.assertIsNone(resolve_verification_prediction_labels_path({"verification_prediction_labels_path": (Path(d) / "missing.png").as_posix()}))

if __name__=="__main__":
    unittest.main()
