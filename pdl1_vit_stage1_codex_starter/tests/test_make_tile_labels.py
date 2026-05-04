"""Unit tests for scripts/make_tile_labels.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.make_tile_labels import (
    classify_tile,
    compute_target_shape_from_tiles,
    determine_authoritative_target_shape,
    load_repo_annotation_mask,
    load_trusted_large_png_mask,
    reconcile_annotation_mask_to_tile_space,
)


class MakeTileLabelsTests(unittest.TestCase):
    def test_sparse_pure_positive_becomes_positive_context(self) -> None:
        tile_mask = np.zeros((10, 10), dtype=np.uint8)
        tile_mask[0, 0] = 1
        label, fractions, reason = classify_tile(
            tile_mask,
            min_labeled_fraction=0.05,
            positive_min_fraction=0.20,
            negative_min_fraction=0.20,
            mixed_max_fraction=0.10,
        )
        self.assertEqual(label, "Positive_Context")
        self.assertEqual(reason, "sparse_positive_seed")
        self.assertAlmostEqual(float(fractions["sample_weight"]), 0.2)

    def test_sparse_pure_negative_becomes_negative_context(self) -> None:
        tile_mask = np.zeros((10, 10), dtype=np.uint8)
        tile_mask[0, 0] = 2
        label, fractions, reason = classify_tile(
            tile_mask,
            min_labeled_fraction=0.05,
            positive_min_fraction=0.20,
            negative_min_fraction=0.20,
            mixed_max_fraction=0.10,
        )
        self.assertEqual(label, "Negative_Context")
        self.assertEqual(reason, "sparse_negative_seed")
        self.assertAlmostEqual(float(fractions["sample_weight"]), 0.2)

    def test_mixed_ambiguous_tile_is_ignored(self) -> None:
        tile_mask = np.array([[1, 2], [0, 0]], dtype=np.uint8)
        label, fractions, reason = classify_tile(
            tile_mask,
            min_labeled_fraction=0.05,
            positive_min_fraction=0.20,
            negative_min_fraction=0.20,
            mixed_max_fraction=0.10,
        )
        self.assertEqual(label, "Ignore")
        self.assertEqual(reason, "mixed_or_ambiguous")
        self.assertEqual(float(fractions["sample_weight"]), 0.0)

    def test_mixed_dominant_tile_can_be_accepted(self) -> None:
        tile_mask = np.array(
            [
                [1, 1, 1, 2],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        label, fractions, reason = classify_tile(
            tile_mask,
            min_labeled_fraction=0.10,
            positive_min_fraction=0.80,
            negative_min_fraction=0.80,
            mixed_max_fraction=0.05,
            mixed_dominant_purity_min=0.75,
            mixed_dominant_weight_scale=0.50,
        )
        self.assertEqual(label, "Positive_Context")
        self.assertEqual(reason, "mixed_dominant_positive")
        self.assertAlmostEqual(float(fractions["sample_weight"]), 0.5)

    def test_dominant_rules_still_take_priority(self) -> None:
        tile_mask = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        label, fractions, reason = classify_tile(
            tile_mask,
            min_labeled_fraction=0.05,
            positive_min_fraction=0.20,
            negative_min_fraction=0.20,
            mixed_max_fraction=0.10,
        )
        self.assertEqual(label, "Positive_Context")
        self.assertEqual(reason, "dominant_positive")
        self.assertAlmostEqual(float(fractions["sample_weight"]), 1.0)

    def test_compute_target_shape_from_tiles(self) -> None:
        import pandas as pd

        tile_df = pd.DataFrame(
            [
                {"tile_x": 0, "tile_y": 0, "tile_w": 224, "tile_h": 224},
                {"tile_x": 112, "tile_y": 224, "tile_w": 224, "tile_h": 224},
            ]
        )
        self.assertEqual(compute_target_shape_from_tiles(tile_df), (448, 336))

    def test_determine_authoritative_target_shape_prefers_tile_meta_image_hw(self) -> None:
        import pandas as pd

        tile_df = pd.DataFrame(
            [
                {"tile_x": 0, "tile_y": 0, "tile_w": 224, "tile_h": 224},
                {"tile_x": 112, "tile_y": 224, "tile_w": 224, "tile_h": 224},
            ]
        )
        target_hw, source = determine_authoritative_target_shape(
            tile_meta={"image_hw": [512, 512]},
            embedding_meta={"svs_level_dimensions_wh": [512, 512]},
            tile_df=tile_df,
            image_id="PFZ083",
        )
        self.assertEqual(target_hw, (512, 512))
        self.assertEqual(source, "tile_manifest_meta.image_hw")

    def test_determine_authoritative_target_shape_rejects_too_small_meta(self) -> None:
        import pandas as pd

        tile_df = pd.DataFrame([{"tile_x": 300, "tile_y": 300, "tile_w": 224, "tile_h": 224}])
        with self.assertRaises(ValueError):
            determine_authoritative_target_shape(
                tile_meta={"image_hw": [400, 400]},
                embedding_meta={},
                tile_df=tile_df,
                image_id="PFZ083",
            )

    def test_determine_authoritative_target_shape_falls_back_for_toy(self) -> None:
        import pandas as pd

        tile_df = pd.DataFrame(
            [
                {"tile_x": 0, "tile_y": 0, "tile_w": 224, "tile_h": 224},
                {"tile_x": 112, "tile_y": 224, "tile_w": 224, "tile_h": 224},
            ]
        )
        target_hw, source = determine_authoritative_target_shape(
            tile_meta={},
            embedding_meta={},
            tile_df=tile_df,
            image_id="toy_001",
        )
        self.assertEqual(target_hw, (448, 336))
        self.assertEqual(source, "accepted_tile_extents_fallback")

    def test_reconcile_annotation_mask_resizes_uniform_scale(self) -> None:
        source = np.arange(16, dtype=np.uint8).reshape(4, 4)
        resolved = reconcile_annotation_mask_to_tile_space(
            source,
            (8, 8),
            tile_meta={"svs_level_downsample": 2.0},
            embedding_meta={},
            image_id="PFZ083",
        )
        self.assertEqual(resolved.shape, (8, 8))
        self.assertEqual(int(resolved[0, 0]), int(source[0, 0]))

    def test_reconcile_annotation_mask_rejects_anisotropic_scale(self) -> None:
        source = np.zeros((4, 8), dtype=np.uint8)
        with self.assertRaises(ValueError):
            reconcile_annotation_mask_to_tile_space(
                source,
                (8, 12),
                tile_meta={},
                embedding_meta={},
                image_id="PFZ083",
            )

    def test_reconcile_annotation_mask_rejects_svs_downsample_mismatch(self) -> None:
        source = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            reconcile_annotation_mask_to_tile_space(
                source,
                (40, 40),
                tile_meta={"coordinate_space": "svs_level_pixels_xywh", "svs_level_downsample": 2.0},
                embedding_meta={},
                image_id="PFZ083",
            )

    def test_trusted_large_loader_refuses_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "annotations"
            root.mkdir(parents=True)
            mask_path = Path(tmpdir) / "outside.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(mask_path)
            with self.assertRaises(ValueError):
                load_trusted_large_png_mask(mask_path, root)

    def test_repo_annotation_mask_applies_shape_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "annotations"
            root.mkdir(parents=True)
            mask_path = root / "roi_mask.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(mask_path)

            loaded = load_repo_annotation_mask(mask_path, annotations_dir=root, expected_shape=(4, 4))
            self.assertEqual(loaded.shape, (4, 4))

            with self.assertRaises(ValueError):
                load_repo_annotation_mask(mask_path, annotations_dir=root, expected_shape=(8, 8))


if __name__ == "__main__":
    unittest.main()
