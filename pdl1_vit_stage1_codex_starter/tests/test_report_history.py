"""Unit tests for report_history helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from report_history import (
    build_history_index,
    build_latest_vs_previous,
    discover_reports_for_image,
    format_history_entry_label,
    load_history_entries_for_image,
    newest_history_entry,
    oldest_history_entry,
)


def _write_report(report_dir: Path, payload: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report_summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (report_dir / "report_summary.md").write_text("# summary\n", encoding="utf-8")


class ReportHistoryTests(unittest.TestCase):
    def test_discover_reports_filters_by_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir) / "outputs"
            _write_report(
                outputs / "reports_run_a",
                {"image_id": "IMG_1", "run_tag": "run_a", "development_metrics": {}, "class_metrics": {}, "supervision_audit": {}},
            )
            _write_report(
                outputs / "reports_run_b",
                {"image_id": "IMG_2", "run_tag": "run_b", "development_metrics": {}, "class_metrics": {}, "supervision_audit": {}},
            )
            found = discover_reports_for_image("IMG_1", outputs_root=outputs)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].name, "reports_run_a")

    def test_build_index_and_latest_vs_previous(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir) / "outputs"
            _write_report(
                outputs / "reports_old",
                {
                    "image_id": "IMG_1",
                    "run_tag": "old",
                    "model_scope": "single_image_model",
                    "development_metrics": {
                        "precision": 0.5,
                        "sensitivity": 0.4,
                        "f1": 0.45,
                        "false_positive_px": 10,
                        "false_negative_px": 20,
                        "training_log_loss_total": 5.0,
                    },
                    "class_metrics": {},
                    "supervision_audit": {"usable_tile_count": 8, "ignored_tile_count": 2},
                    "warnings": ["w_old"],
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            )
            _write_report(
                outputs / "reports_new",
                {
                    "image_id": "IMG_1",
                    "run_tag": "new",
                    "model_scope": "shared_project_model",
                    "shared_model_tag": "proj",
                    "training_image_count": 2,
                    "development_metrics": {
                        "precision": 0.8,
                        "sensitivity": 0.7,
                        "f1": 0.75,
                        "false_positive_px": 6,
                        "false_negative_px": 12,
                        "training_log_loss_total": 4.0,
                    },
                    "class_metrics": {},
                    "supervision_audit": {"usable_tile_count": 12, "ignored_tile_count": 1},
                    "warnings": ["w_new"],
                    "timestamp": "2026-01-02T00:00:00+00:00",
                },
            )

            index_payload = build_history_index("IMG_1", outputs_root=outputs)
            self.assertEqual(index_payload["run_count"], 2)
            self.assertEqual(index_payload["runs"][-1]["run_tag"], "new")
            compare = build_latest_vs_previous(index_payload, outputs_root=outputs)
            self.assertTrue(compare["comparison_available"])
            self.assertEqual(compare["latest_run_tag"], "new")
            self.assertEqual(compare["previous_run_tag"], "old")
            self.assertAlmostEqual(compare["metric_deltas"]["f1"], 0.30)
            self.assertEqual(compare["warning_changes"]["added"], ["w_new"])
            self.assertEqual(compare["warning_changes"]["removed"], ["w_old"])
            self.assertTrue((outputs / "report_history" / "img_1" / "latest_vs_previous.md").exists())

    def test_load_history_entries_sorted_descending_and_discovery_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir) / "outputs"
            _write_report(
                outputs / "reports_run_old",
                {"image_id": "IMG_1", "run_tag": "run_old", "development_metrics": {"f1": 0.1}, "class_metrics": {}, "supervision_audit": {}, "timestamp": "2026-01-01T00:00:00+00:00"},
            )
            _write_report(
                outputs / "reports_run_new",
                {"image_id": "IMG_1", "run_tag": "run_new", "development_metrics": {"f1": 0.9}, "class_metrics": {}, "supervision_audit": {}, "timestamp": "2026-01-02T00:00:00+00:00"},
            )
            entries = load_history_entries_for_image("IMG_1", outputs_root=outputs)
            self.assertEqual([row["run_tag"] for row in entries], ["run_new", "run_old"])

            # Simulate stale index and ensure discovery still backfills missing rows.
            history_index = outputs / "report_history" / "img_1" / "history_index.json"
            history_index.parent.mkdir(parents=True, exist_ok=True)
            history_index.write_text(
                json.dumps({"image_id": "IMG_1", "runs": [entries[-1]]}, indent=2),
                encoding="utf-8",
            )
            merged = load_history_entries_for_image("IMG_1", outputs_root=outputs)
            self.assertEqual([row["run_tag"] for row in merged], ["run_new", "run_old"])

    def test_history_label_and_newest_oldest_helpers(self) -> None:
        rows = [
            {"run_tag": "old", "timestamp_utc": "2026-01-01T00:00:00+00:00", "model_scope": "single_image_model", "f1": 0.21},
            {"run_tag": "new", "timestamp_utc": "2026-01-05T01:30:00+00:00", "model_scope": "shared_project_model", "f1": 0.91},
        ]
        self.assertEqual(newest_history_entry(rows)["run_tag"], "new")
        self.assertEqual(oldest_history_entry(rows)["run_tag"], "old")
        label = format_history_entry_label(rows[1])
        self.assertIn("2026-01-05 01:30", label)
        self.assertIn("new", label)
        self.assertIn("shared", label)
        self.assertIn("F1 0.910", label)


if __name__ == "__main__":
    unittest.main()
