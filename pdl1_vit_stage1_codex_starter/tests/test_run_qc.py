"""Unit tests for scripts/run_qc.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_qc import analyze_image, discover_image_files


class RunQcTests(unittest.TestCase):
    def test_discover_includes_svs_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample.svs").write_bytes(b"svs")
            (root / "sample.png").write_bytes(b"not_a_real_png")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            files = discover_image_files(root)
            self.assertEqual([p.name for p in files], ["sample.png", "sample.svs"])

    def test_analyze_svs_without_openslide_has_explicit_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            svs_path = Path(tmpdir) / "case.svs"
            svs_path.write_bytes(b"not_real_svs")

            with patch("scripts.run_qc.try_read_svs_dimensions", return_value=(None, None, "openslide_python_not_installed")):
                qc, thumb, tissue = analyze_image(svs_path, thumb_size=128)

            self.assertIsNone(thumb)
            self.assertIsNone(tissue)
            self.assertEqual(qc["qc_status"], "svs_metadata_only_no_openslide")
            self.assertIn("svs_metadata_qc_only", qc["qc_notes"])
            self.assertEqual(qc["width_px"], "")
            self.assertEqual(qc["height_px"], "")

    def test_analyze_svs_records_dimensions_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            svs_path = Path(tmpdir) / "case.svs"
            svs_path.write_bytes(b"not_real_svs")

            with patch("scripts.run_qc.try_read_svs_dimensions", return_value=(4000, 3000, "")):
                qc, thumb, tissue = analyze_image(svs_path, thumb_size=128)

            self.assertIsNone(thumb)
            self.assertIsNone(tissue)
            self.assertEqual(qc["qc_status"], "svs_metadata_only")
            self.assertEqual(qc["width_px"], "4000")
            self.assertEqual(qc["height_px"], "3000")


if __name__ == "__main__":
    unittest.main()
