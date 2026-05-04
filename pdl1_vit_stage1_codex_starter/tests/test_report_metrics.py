"""Unit tests for scripts/report_metrics.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.report_metrics import compute_development_metrics, compute_metrics_from_paths


LABEL_ENCODING = {
    "Positive_Tumor": 1,
    "Negative_Tumor": 2,
    "NonTumor": 3,
}


class ReportMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        scribble = np.array([[1, 2], [3, 0]], dtype=np.uint8)
        positive_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        pixel_prob = np.array([[0.99, 0.01], [0.01, 0.50]], dtype=np.float32)

        m = compute_development_metrics(
            scribble_labels=scribble,
            positive_mask=positive_mask,
            pixel_prob_map=pixel_prob,
            label_encoding=LABEL_ENCODING,
        )

        self.assertEqual(m["false_positive_px"], 0)
        self.assertEqual(m["false_negative_px"], 0)
        self.assertEqual(m["tp_px"], 1)
        self.assertEqual(m["tn_px"], 2)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["sensitivity"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)
        self.assertEqual(m["class_metrics"]["Positive_Tumor"]["tp_px"], 1)
        self.assertEqual(m["class_metrics"]["Negative_Tumor"]["tn_px"], 1)
        self.assertEqual(m["class_metrics"]["NonTumor"]["tn_px"], 1)

    def test_fp_only_case(self) -> None:
        scribble = np.array([[2, 2], [2, 0]], dtype=np.uint8)
        positive_mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        pixel_prob = np.array([[0.8, 0.2], [0.9, 0.0]], dtype=np.float32)

        m = compute_development_metrics(
            scribble_labels=scribble,
            positive_mask=positive_mask,
            pixel_prob_map=pixel_prob,
            label_encoding=LABEL_ENCODING,
        )

        self.assertEqual(m["tp_px"], 0)
        self.assertEqual(m["false_positive_px"], 2)
        self.assertEqual(m["false_negative_px"], 0)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["sensitivity"], 0.0)
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(m["class_metrics"]["Negative_Tumor"]["fp_px"], 2)
        self.assertEqual(m["class_metrics"]["NonTumor"]["annotated_px"], 0)

    def test_fn_only_case(self) -> None:
        scribble = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        positive_mask = np.zeros((2, 2), dtype=np.uint8)
        pixel_prob = np.array([[0.1, 0.2], [0.5, 0.5]], dtype=np.float32)

        m = compute_development_metrics(
            scribble_labels=scribble,
            positive_mask=positive_mask,
            pixel_prob_map=pixel_prob,
            label_encoding=LABEL_ENCODING,
        )

        self.assertEqual(m["tp_px"], 0)
        self.assertEqual(m["false_positive_px"], 0)
        self.assertEqual(m["false_negative_px"], 2)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["sensitivity"], 0.0)
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(m["class_metrics"]["Positive_Tumor"]["fn_px"], 2)

    def test_unlabeled_pixels_are_excluded(self) -> None:
        scribble = np.array([[1, 0], [2, 0]], dtype=np.uint8)
        positive_mask = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        pixel_prob = np.array([[0.9, 0.9], [0.1, 0.9]], dtype=np.float32)

        m = compute_development_metrics(
            scribble_labels=scribble,
            positive_mask=positive_mask,
            pixel_prob_map=pixel_prob,
            label_encoding=LABEL_ENCODING,
        )

        self.assertEqual(m["annotated_total_px"], 2)
        self.assertEqual(m["false_positive_px"], 0)
        self.assertEqual(m["false_negative_px"], 0)
        self.assertEqual(m["class_metrics"]["Positive_Tumor"]["annotated_px"], 1)
        self.assertEqual(m["class_metrics"]["Negative_Tumor"]["annotated_px"], 1)

    def test_resize_reconcile_alignment_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scribble_path = root / "scribble.png"
            mask_path = root / "positive_mask.png"
            prob_path = root / "pixel_prob_map.png"

            scribble_small = np.array([[1, 2], [3, 0]], dtype=np.uint8)
            Image.fromarray(scribble_small, mode="L").save(scribble_path)

            positive = np.array(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
            Image.fromarray((positive * 255).astype(np.uint8), mode="L").save(mask_path)
            probs = np.where(positive > 0, 220, 35).astype(np.uint8)
            Image.fromarray(probs, mode="L").save(prob_path)

            m = compute_metrics_from_paths(
                image_id="PF0229",
                scribble_labels_path=scribble_path,
                positive_mask_path=mask_path,
                pixel_prob_map_path=prob_path,
                label_encoding=LABEL_ENCODING,
            )

            self.assertEqual(m["scribble_transform"], "nearest_uniform_scale")
            self.assertEqual(m["annotated_total_px"], 12)
            self.assertGreater(m["precision"], 0.99)


if __name__ == "__main__":
    unittest.main()
