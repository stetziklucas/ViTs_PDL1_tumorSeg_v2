from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile
import unittest

from apps.annotator import build_stage1_project_command, default_project_tag, latest_project_summary_path, resolve_current_image_shared_report


class AnnotatorProjectPanelTests(unittest.TestCase):
    def test_project_tag_shape(self) -> None:
        self.assertTrue(default_project_tag(datetime(2026,1,1,tzinfo=timezone.utc)).startswith("training_20260101_000000"))

    def test_build_project_command(self) -> None:
        cmd = build_stage1_project_command(config_path=Path("config/base.yaml"), project_tag="t1", raw_dir=Path("data/raw"), annotations_dir=Path("data/annotations"), outputs_root=Path("outputs"), models_root=Path("models"))
        self.assertIn("--discover-ready-cases", cmd)

    def test_resolve_current_shared_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            d = out / "reports_training_t1"; d.mkdir()
            (d / "stage1_project_cases.json").write_text(json.dumps({"included_ready_cases":[{"alias":"a","image_id":"IMG"}]}), encoding="utf-8")
            p = resolve_current_image_shared_report(outputs_root=out, project_tag="t1", image_id="IMG")
            self.assertEqual(p, out / "reports_t1__a" / "report_summary.md")
            self.assertEqual(latest_project_summary_path(out, "t1"), out / "reports_training_t1" / "training_summary.md")

if __name__ == '__main__':
    unittest.main()
