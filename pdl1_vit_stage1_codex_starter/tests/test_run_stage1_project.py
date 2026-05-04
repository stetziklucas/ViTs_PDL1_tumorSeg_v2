"""Unit tests for scripts/run_stage1_project.py orchestration behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation_readiness import AnnotationReadinessResult
from scripts import run_stage1_project


def _result(image_id: str, status_code: str) -> AnnotationReadinessResult:
    return AnnotationReadinessResult(
        image_id=image_id,
        artifact_exists={"annotation_meta": True, "roi_mask": True, "scribble_labels": True},
        polygon_counts={"Positive_Tumor": 1, "Negative_Tumor": 1, "NonTumor": 0, "Ignore": 0},
        pixel_counts={"Positive_Tumor": 10, "Negative_Tumor": 8, "NonTumor": 0, "Ignore": 0},
        roi_positive_pixels=100,
        status_code=status_code,
        status_label=status_code,
        summary_message=f"{status_code} summary",
        next_action="continue",
        notes=[],
    )


class RunStage1ProjectTests(unittest.TestCase):
    def test_collect_cases_supports_case_and_cases_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases_file = root / "cases.txt"
            cases_file.write_text("\n# comment\nb=img_b\n\n", encoding="utf-8")
            out = run_stage1_project.collect_cases(["a=img_a"], cases_file)
            self.assertEqual(out, [{"alias": "a", "image_id": "img_a"}, {"alias": "b", "image_id": "img_b"}])

    def test_collect_cases_duplicate_alias_fails(self) -> None:
        with self.assertRaises(ValueError):
            run_stage1_project.collect_cases(["dup=img1", "dup=img2"], None)

    def _run_main(self, argv: list[str], readiness_map: dict[str, str], rc_map: dict[str, int] | None = None) -> int:
        rc_map = rc_map or {}

        def fake_readiness(*, config: dict[str, object], annotations_dir: Path, image_id: str) -> AnnotationReadinessResult:
            _ = (config, annotations_dir)
            return _result(image_id, readiness_map[image_id])

        def fake_step(command: list[str], log_handle: object) -> int:
            _ = log_handle
            key = Path(command[1]).name
            return int(rc_map.get(key, 0))

        with patch("scripts.run_stage1_project.compute_annotation_readiness", side_effect=fake_readiness), patch(
            "scripts.run_stage1_project._run_subprocess_streaming", side_effect=fake_step
        ), patch("sys.argv", argv):
            return run_stage1_project.main()

    def test_readiness_failure_exits_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = root / "config.yaml"
            cfg.write_text("classes:\n  label_encoding:\n    Positive_Tumor: 1\n", encoding="utf-8")
            rc = self._run_main(
                [
                    "run_stage1_project.py",
                    "--config",
                    str(cfg),
                    "--project-tag",
                    "proj",
                    "--case",
                    "a=img_a",
                    "--case",
                    "b=img_b",
                    "--outputs-root",
                    str(root / "outputs"),
                    "--models-root",
                    str(root / "models"),
                ],
                readiness_map={"img_a": "READY", "img_b": "NEEDS_NEGATIVE"},
            )
            self.assertEqual(rc, 1)

    def test_allow_skip_not_ready_requires_two_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = root / "config.yaml"
            cfg.write_text("classes:\n  label_encoding:\n    Positive_Tumor: 1\n", encoding="utf-8")
            rc = self._run_main(
                [
                    "run_stage1_project.py",
                    "--config",
                    str(cfg),
                    "--project-tag",
                    "proj",
                    "--case",
                    "a=img_a",
                    "--case",
                    "b=img_b",
                    "--allow-skip-not-ready",
                    "--outputs-root",
                    str(root / "outputs"),
                    "--models-root",
                    str(root / "models"),
                ],
                readiness_map={"img_a": "READY", "img_b": "NEEDS_NEGATIVE"},
            )
            self.assertEqual(rc, 1)

    def test_success_writes_project_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = root / "config.yaml"
            cfg.write_text("classes:\n  label_encoding:\n    Positive_Tumor: 1\n", encoding="utf-8")
            rc = self._run_main(
                [
                    "run_stage1_project.py",
                    "--config",
                    str(cfg),
                    "--project-tag",
                    "proj",
                    "--case",
                    "a=img_a",
                    "--case",
                    "b=img_b",
                    "--outputs-root",
                    str(root / "outputs"),
                    "--models-root",
                    str(root / "models"),
                ],
                readiness_map={"img_a": "READY", "img_b": "READY"},
            )
            self.assertEqual(rc, 0)
            summary = root / "outputs" / "reports_training_proj" / "stage1_project_run_summary.json"
            self.assertTrue(summary.exists())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["final_status"], "SUCCESS")
            self.assertEqual(len(payload["included_cases"]), 2)

    def test_discover_ready_cases_mode_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = root / "config.yaml"
            cfg.write_text("classes:\n  label_encoding:\n    Positive_Tumor: 1\n", encoding="utf-8")
            discovered = {
                "included_ready_cases": [{"alias": "a", "image_id": "img_a"}, {"alias": "b", "image_id": "img_b"}],
                "skipped_cases": [{"alias": "c", "image_id": "img_c", "reason": "NOT_READY"}],
                "counts": {"included_ready_count": 2},
            }
            with patch("scripts.run_stage1_project.discover_project_cases", return_value=discovered):
                rc = self._run_main(
                    [
                        "run_stage1_project.py","--config",str(cfg),"--project-tag","proj","--discover-ready-cases",
                        "--outputs-root",str(root / "outputs"),"--models-root",str(root / "models"),
                    ],
                    readiness_map={"img_a": "READY", "img_b": "READY"},
                )
            self.assertEqual(rc, 0)
            manifest = root / "outputs" / "reports_training_proj" / "stage1_project_cases.json"
            self.assertTrue(manifest.exists())


if __name__ == "__main__":
    unittest.main()
