"""Unit tests for annotation readiness summary logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from annotation_readiness import compute_annotation_readiness


class AnnotationReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.annotations_dir = Path(self.tmpdir.name)
        (self.annotations_dir / "roi_masks").mkdir(parents=True, exist_ok=True)
        (self.annotations_dir / "scribbles").mkdir(parents=True, exist_ok=True)
        self.image_id = "case123"
        self.config = {
            "tiling": {"allow_sparse_roi_seed_tiles": True},
            "classes": {
                "label_encoding": {
                    "Unlabeled": 0,
                    "Positive_Tumor": 1,
                    "Negative_Tumor": 2,
                    "NonTumor": 3,
                    "Ignore": 4,
                }
            },
        }

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_artifacts(self, roi: np.ndarray, scribble: np.ndarray, polygons: list[dict[str, object]]) -> None:
        Image.fromarray(roi.astype(np.uint8), mode="L").save(
            self.annotations_dir / "roi_masks" / f"{self.image_id}_roi_mask.png"
        )
        Image.fromarray(scribble.astype(np.uint8), mode="L").save(
            self.annotations_dir / "scribbles" / f"{self.image_id}_scribble_labels.png"
        )
        meta = {"image_id": self.image_id, "polygons": polygons}
        with (self.annotations_dir / f"{self.image_id}_annotation_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle)

    def test_ready(self) -> None:
        roi = np.zeros((10, 10), dtype=np.uint8)
        roi[1:4, 1:4] = 1
        scribble = np.zeros((10, 10), dtype=np.uint8)
        scribble[1:3, 1:3] = 1
        scribble[5:7, 5:7] = 2
        self._write_artifacts(
            roi,
            scribble,
            [{"class_name": "Positive_Tumor"}, {"class_name": "Negative_Tumor"}],
        )

        result = compute_annotation_readiness(
            config=self.config,
            annotations_dir=self.annotations_dir,
            image_id=self.image_id,
        )
        self.assertEqual(result.status_code, "READY")
        self.assertGreater(result.roi_positive_pixels, 0)

    def test_needs_positive(self) -> None:
        roi = np.ones((8, 8), dtype=np.uint8)
        scribble = np.zeros((8, 8), dtype=np.uint8)
        scribble[1:4, 1:4] = 2
        self._write_artifacts(roi, scribble, [{"class_name": "Negative_Tumor"}])

        result = compute_annotation_readiness(config=self.config, annotations_dir=self.annotations_dir, image_id=self.image_id)
        self.assertEqual(result.status_code, "NEEDS_POSITIVE")

    def test_needs_negative(self) -> None:
        roi = np.ones((8, 8), dtype=np.uint8)
        scribble = np.zeros((8, 8), dtype=np.uint8)
        scribble[1:4, 1:4] = 1
        self._write_artifacts(roi, scribble, [{"class_name": "Positive_Tumor"}])

        result = compute_annotation_readiness(config=self.config, annotations_dir=self.annotations_dir, image_id=self.image_id)
        self.assertEqual(result.status_code, "NEEDS_NEGATIVE")

    def test_no_tumor_roi(self) -> None:
        roi = np.zeros((8, 8), dtype=np.uint8)
        scribble = np.zeros((8, 8), dtype=np.uint8)
        scribble[1:4, 1:4] = 3
        self._write_artifacts(roi, scribble, [{"class_name": "NonTumor"}, {"class_name": "Ignore"}])

        result = compute_annotation_readiness(config=self.config, annotations_dir=self.annotations_dir, image_id=self.image_id)
        self.assertEqual(result.status_code, "NO_TUMOR_ROI")

    def test_no_usable_supervision(self) -> None:
        roi = np.zeros((8, 8), dtype=np.uint8)
        scribble = np.zeros((8, 8), dtype=np.uint8)
        self._write_artifacts(roi, scribble, [])

        result = compute_annotation_readiness(config=self.config, annotations_dir=self.annotations_dir, image_id=self.image_id)
        self.assertEqual(result.status_code, "NO_USABLE_SUPERVISION")

    def test_error_missing_or_broken_artifacts(self) -> None:
        # Missing all artifacts
        missing_result = compute_annotation_readiness(
            config=self.config,
            annotations_dir=self.annotations_dir,
            image_id=self.image_id,
        )
        self.assertEqual(missing_result.status_code, "ERROR")

        # Broken metadata
        roi = np.ones((4, 4), dtype=np.uint8)
        scribble = np.ones((4, 4), dtype=np.uint8)
        Image.fromarray(roi.astype(np.uint8), mode="L").save(
            self.annotations_dir / "roi_masks" / f"{self.image_id}_roi_mask.png"
        )
        Image.fromarray(scribble.astype(np.uint8), mode="L").save(
            self.annotations_dir / "scribbles" / f"{self.image_id}_scribble_labels.png"
        )
        with (self.annotations_dir / f"{self.image_id}_annotation_meta.json").open("w", encoding="utf-8") as handle:
            handle.write("not-json")

        broken_result = compute_annotation_readiness(
            config=self.config,
            annotations_dir=self.annotations_dir,
            image_id=self.image_id,
        )
        self.assertEqual(broken_result.status_code, "ERROR")


if __name__ == "__main__":
    unittest.main()
