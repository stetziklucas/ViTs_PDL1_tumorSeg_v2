"""Unit tests for scripts/run_stage1_image.py orchestration behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from annotation_readiness import AnnotationReadinessResult
from scripts import run_stage1_image


def _ready_result() -> AnnotationReadinessResult:
    return AnnotationReadinessResult(
        image_id="PF0229",
        artifact_exists={"annotation_meta": True, "roi_mask": True, "scribble_labels": True},
        polygon_counts={"Positive_Tumor": 1, "Negative_Tumor": 1, "NonTumor": 0, "Ignore": 0},
        pixel_counts={"Positive_Tumor": 10, "Negative_Tumor": 8, "NonTumor": 0, "Ignore": 0},
        roi_positive_pixels=100,
        status_code="READY",
        status_label="Ready",
        summary_message="ready",
        next_action="continue",
        notes=[],
    )


class RunStage1ImageTests(unittest.TestCase):
    def _args(self, root: Path) -> Namespace:
        config = root / "config.yaml"
        config.write_text("classes:\n  label_encoding:\n    Positive_Tumor: 1\n", encoding="utf-8")
        return Namespace(
            config=config,
            image_id="PF0229",
            run_tag="pf0229_test",
            raw_dir=root / "raw",
            annotations_dir=root / "annotations",
            outputs_root=root / "outputs",
            models_root=root / "models",
        )

    def test_ready_path_runs_steps_in_order_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._args(root)
            calls: list[list[str]] = []

            def fake_runner(command: list[str], _log_handle: object) -> int:
                calls.append(command)
                return 0

            exit_code = run_stage1_image.run_pipeline(
                args,
                readiness_fn=lambda **_: _ready_result(),
                step_runner=fake_runner,
            )
            self.assertEqual(exit_code, 0)

            step_scripts = [Path(cmd[1]).name for cmd in calls]
            self.assertEqual(
                step_scripts,
                [
                    "extract_tiles.py",
                    "embed_vit.py",
                    "make_tile_labels.py",
                    "train_tile_head.py",
                    "train_pixel_classifier.py",
                    "run_inference.py",
                    "make_report.py",
                ],
            )

            summary_json = args.outputs_root / f"reports_{args.run_tag}" / "stage1_run_summary.json"
            summary_md = args.outputs_root / f"reports_{args.run_tag}" / "stage1_run_summary.md"
            self.assertTrue(summary_json.exists())
            self.assertTrue(summary_md.exists())

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["final_status"], "SUCCESS")
            self.assertEqual(len(payload["steps"]), 7)
            self.assertIn("next_review_files", payload)

    def test_not_ready_exits_early_without_running_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._args(root)
            calls: list[list[str]] = []

            result = _ready_result()
            not_ready = AnnotationReadinessResult(**{**result.to_dict(), "status_code": "NEEDS_NEGATIVE", "status_label": "Needs negative supervision"})

            exit_code = run_stage1_image.run_pipeline(
                args,
                readiness_fn=lambda **_: not_ready,
                step_runner=lambda cmd, log: calls.append(cmd) or 0,
            )
            self.assertEqual(exit_code, 1)
            self.assertEqual(calls, [])

    def test_readiness_error_exits_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._args(root)

            def broken_readiness(**_: object) -> AnnotationReadinessResult:
                raise RuntimeError("broken readiness")

            exit_code = run_stage1_image.run_pipeline(args, readiness_fn=broken_readiness)
            self.assertEqual(exit_code, 2)

    def test_downstream_failure_exits_three_and_records_partial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._args(root)
            calls: list[list[str]] = []

            def fake_runner(command: list[str], _log_handle: object) -> int:
                calls.append(command)
                if Path(command[1]).name == "train_pixel_classifier.py":
                    return 9
                return 0

            exit_code = run_stage1_image.run_pipeline(
                args,
                readiness_fn=lambda **_: _ready_result(),
                step_runner=fake_runner,
            )
            self.assertEqual(exit_code, 3)

            summary_json = args.outputs_root / f"reports_{args.run_tag}" / "stage1_run_summary.json"
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["final_status"], "FAILED")
            self.assertEqual(len(payload["steps"]), 5)
            self.assertEqual(payload["steps"][-1]["step"], "train_pixel_classifier")
            self.assertNotEqual(payload["steps"][-1]["exit_code"], 0)

    def test_log_path_handling_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._args(root)

            exit_code = run_stage1_image.run_pipeline(
                args,
                readiness_fn=lambda **_: _ready_result(),
                step_runner=lambda _cmd, _log: 0,
            )
            self.assertEqual(exit_code, 0)

            reports_dir = args.outputs_root / f"reports_{args.run_tag}"
            log_path = reports_dir / "stage1_runner.log"
            self.assertTrue(log_path.exists())
            summary_payload = json.loads((reports_dir / "stage1_run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["key_artifacts"]["runner_log"], log_path.as_posix())

    def test_success_refreshes_report_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = self._args(root)
            with patch("scripts.run_stage1_image.refresh_history_for_image") as history_mock:
                history_mock.return_value = {"history_index": {"run_count": 1}, "latest_vs_previous": {}}
                exit_code = run_stage1_image.run_pipeline(
                    args,
                    readiness_fn=lambda **_: _ready_result(),
                    step_runner=lambda _cmd, _log: 0,
                )
            self.assertEqual(exit_code, 0)
            history_mock.assert_called_once_with("PF0229", outputs_root=args.outputs_root)


if __name__ == "__main__":
    unittest.main()
