from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from project_case_discovery import discover_project_cases, make_case_alias


class ProjectCaseDiscoveryTests(unittest.TestCase):
    def test_alias_generation_unique_and_stable(self) -> None:
        used: set[str] = set()
        self.assertEqual(make_case_alias("IMG A", used), "img_a")
        self.assertEqual(make_case_alias("IMG A", used), "img_a_2")

    def test_discovery_includes_ready_and_skips_missing_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ann = root / "ann"; ann.mkdir()
            (ann / "roi_masks").mkdir(); (ann / "scribbles").mkdir()
            raw = root / "raw"; raw.mkdir()
            cfg = {"classes": {"label_encoding": {"Positive_Tumor": 1, "Negative_Tumor": 2, "NonTumor": 3, "Ignore": 4}}}
            (ann / "A_annotation_meta.json").write_text('{"polygons":[{"class_name":"Positive_Tumor"},{"class_name":"Negative_Tumor"}]}')
            Image.fromarray(np.ones((4,4), dtype=np.uint8)).save(ann / "roi_masks" / "A_roi_mask.png")
            s=np.zeros((4,4),dtype=np.uint8); s[0,0]=1; s[1,1]=2
            Image.fromarray(s).save(ann / "scribbles" / "A_scribble_labels.png")
            (raw / "A.png").write_bytes(b"x")
            (ann / "B_annotation_meta.json").write_text('{"polygons":[]}')
            summary = discover_project_cases(config=cfg, annotations_dir=ann, raw_dir=raw)
            self.assertEqual(summary["counts"]["included_ready_count"], 1)
            self.assertGreaterEqual(summary["counts"]["skipped_count"], 1)
