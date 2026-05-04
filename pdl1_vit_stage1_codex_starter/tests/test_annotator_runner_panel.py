"""Lightweight tests for annotator Stage 1 runner-panel helper logic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from apps.annotator import (
    build_stage1_run_command,
    default_gui_run_tag,
    format_selected_run_tag_text,
    history_preview_candidates,
    load_markdown_preview,
    runner_log_visibility_status,
    select_history_index_by_jump,
)


class AnnotatorRunnerPanelTests(unittest.TestCase):
    def test_default_gui_run_tag_has_expected_shape(self) -> None:
        tag = default_gui_run_tag("PF0229 sample", now=datetime(2026, 4, 18, 12, 30, 0, tzinfo=timezone.utc))
        self.assertTrue(tag.startswith("pf0229_gui_20260418_123000"))

    def test_build_stage1_run_command_contains_required_args(self) -> None:
        command = build_stage1_run_command(
            config_path=Path("config/base.yaml"),
            image_id="IMG_A",
            run_tag="img_gui_1",
            raw_dir=Path("data/raw"),
            annotations_dir=Path("data/annotations"),
            outputs_root=Path("outputs"),
            models_root=Path("models"),
        )
        text = " ".join(command)
        self.assertIn("scripts/run_stage1_image.py", text)
        self.assertIn("--image-id IMG_A", text)
        self.assertIn("--run-tag img_gui_1", text)

    def test_load_markdown_preview_uses_first_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = root / "missing.md"
            existing = root / "summary.md"
            existing.write_text("# hello\n", encoding="utf-8")
            text, selected = load_markdown_preview([missing, existing])
            self.assertEqual(text.strip(), "# hello")
            self.assertEqual(selected, existing)

    def test_history_preview_candidates_prefer_report_then_stage1_summary(self) -> None:
        entry = {
            "report_summary_md": "outputs/reports_x/report_summary.md",
            "stage1_run_summary_md": "outputs/reports_x/stage1_run_summary.md",
            "report_summary_json": "outputs/reports_x/report_summary.json",
        }
        candidates = history_preview_candidates(entry)
        self.assertEqual(candidates[0], Path("outputs/reports_x/report_summary.md"))
        self.assertEqual(candidates[1], Path("outputs/reports_x/stage1_run_summary.md"))
        self.assertEqual(candidates[2], Path("outputs/reports_x/report_summary.md"))

    def test_history_preview_loads_stage1_fallback_when_report_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage1 = root / "stage1_run_summary.md"
            stage1.write_text("# stage1 fallback\n", encoding="utf-8")
            entry = {
                "report_summary_md": str(root / "missing_report.md"),
                "stage1_run_summary_md": str(stage1),
                "report_summary_json": str(root / "report_summary.json"),
            }
            preview_text, preview_path = load_markdown_preview(history_preview_candidates(entry))
            self.assertIn("stage1 fallback", preview_text)
            self.assertEqual(preview_path, stage1)

    def test_jump_helpers_choose_newest_and_oldest(self) -> None:
        entries = [
            {"run_tag": "new", "timestamp_utc": "2026-01-03T00:00:00+00:00"},
            {"run_tag": "mid", "timestamp_utc": "2026-01-02T00:00:00+00:00"},
            {"run_tag": "old", "timestamp_utc": "2026-01-01T00:00:00+00:00"},
        ]
        self.assertEqual(select_history_index_by_jump(entries, jump="newest"), 0)
        self.assertEqual(select_history_index_by_jump(entries, jump="oldest"), 2)

    def test_selected_run_tag_and_log_visibility_status_text(self) -> None:
        self.assertEqual(format_selected_run_tag_text(None), "Selected history run tag: none")
        self.assertEqual(
            format_selected_run_tag_text({"run_tag": "pf0229_gui_a"}),
            "Selected history run tag: pf0229_gui_a",
        )
        self.assertIn("shown", runner_log_visibility_status(True))
        self.assertIn("hidden", runner_log_visibility_status(False))


if __name__ == "__main__":
    unittest.main()
