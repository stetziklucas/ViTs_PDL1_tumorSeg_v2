"""Unit tests for trusted large-mask loading and reconciliation guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError

from scripts.extract_tiles import load_trusted_large_png_mask as load_extract_mask
from scripts.extract_tiles import reconcile_mask_to_target as reconcile_extract_mask
from scripts.run_inference import load_trusted_large_png_mask as load_inference_mask
from scripts.run_inference import reconcile_mask_to_target as reconcile_inference_mask
from scripts.train_pixel_classifier import load_trusted_large_png_mask as load_train_mask


class LargeMaskLoadingTests(unittest.TestCase):
    def test_train_loader_rejects_paths_outside_trusted_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trusted_root = Path(tmpdir) / "annotations"
            trusted_root.mkdir(parents=True)
            outside = Path(tmpdir) / "outside.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(outside)
            with self.assertRaises(ValueError):
                load_train_mask(outside, trusted_root)

    def test_inference_loader_accepts_png_inside_trusted_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trusted_root = Path(tmpdir) / "tiles"
            trusted_root.mkdir(parents=True)
            mask_path = trusted_root / "mask.png"
            Image.fromarray(np.ones((4, 4), dtype=np.uint8), mode="L").save(mask_path)

            loaded = load_inference_mask(mask_path, trusted_root)

            self.assertEqual(loaded.shape, (4, 4))
            self.assertEqual(loaded.dtype, np.uint8)

    def test_reconcile_mask_to_target_rejects_anisotropic_resize(self) -> None:
        source = np.zeros((4, 8), dtype=np.uint8)
        with self.assertRaises(ValueError):
            reconcile_inference_mask(source, (8, 12), image_id="PF0229")

    def test_extract_reconcile_mask_to_target_rejects_anisotropic_resize(self) -> None:
        source = np.zeros((6, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            reconcile_extract_mask(source, (12, 14), image_id="PF0229")

    def test_extract_reconcile_mask_to_target_resizes_uniformly_with_nearest(self) -> None:
        source = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        reconciled, transform = reconcile_extract_mask(source, (4, 4), image_id="PF083")
        self.assertEqual(transform, "nearest_uniform_scale")
        self.assertEqual(reconciled.shape, (4, 4))
        self.assertEqual(reconciled.dtype, np.uint8)

    def test_trusted_loader_bypasses_global_decompression_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trusted_root = Path(tmpdir) / "annotations"
            trusted_root.mkdir(parents=True)
            mask_path = trusted_root / "large.png"
            Image.fromarray(np.ones((128, 128), dtype=np.uint8), mode="L").save(mask_path)

            prior_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = 10
            try:
                with self.assertRaises(DecompressionBombError):
                    np.asarray(Image.open(mask_path))

                loaded = load_train_mask(mask_path, trusted_root)
                self.assertEqual(loaded.shape, (128, 128))
            finally:
                Image.MAX_IMAGE_PIXELS = prior_limit

    def test_extract_loader_bypasses_global_decompression_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trusted_root = Path(tmpdir) / "annotations"
            trusted_root.mkdir(parents=True)
            mask_path = trusted_root / "large.png"
            Image.fromarray(np.ones((128, 128), dtype=np.uint8), mode="L").save(mask_path)

            prior_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = 10
            try:
                with self.assertRaises(DecompressionBombError):
                    np.asarray(Image.open(mask_path))

                loaded = load_extract_mask(mask_path, trusted_root)
                self.assertEqual(loaded.shape, (128, 128))
            finally:
                Image.MAX_IMAGE_PIXELS = prior_limit


if __name__ == "__main__":
    unittest.main()
