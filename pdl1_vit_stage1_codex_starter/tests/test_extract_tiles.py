"""Unit tests for scripts/extract_tiles.py."""

from __future__ import annotations

import unittest

from scripts.extract_tiles import evaluate_tile_acceptance


class ExtractTilesTests(unittest.TestCase):
    def test_acceptance_uses_standard_roi_threshold_when_met(self) -> None:
        accept, reason = evaluate_tile_acceptance(
            tissue_fraction=0.9,
            roi_fraction=0.02,
            min_tissue_fraction=0.3,
            min_roi_fraction=0.01,
            allow_sparse_roi_seed_tiles=False,
        )
        self.assertTrue(accept)
        self.assertEqual(reason, "accepted_roi_threshold")

    def test_acceptance_uses_sparse_fallback_when_enabled(self) -> None:
        accept, reason = evaluate_tile_acceptance(
            tissue_fraction=0.8,
            roi_fraction=0.0001,
            min_tissue_fraction=0.3,
            min_roi_fraction=0.01,
            allow_sparse_roi_seed_tiles=True,
        )
        self.assertTrue(accept)
        self.assertEqual(reason, "accepted_sparse_roi_fallback")

    def test_acceptance_rejects_sparse_roi_when_fallback_disabled(self) -> None:
        accept, reason = evaluate_tile_acceptance(
            tissue_fraction=0.8,
            roi_fraction=0.0001,
            min_tissue_fraction=0.3,
            min_roi_fraction=0.01,
            allow_sparse_roi_seed_tiles=False,
        )
        self.assertFalse(accept)
        self.assertEqual(reason, "rejected_low_roi")


if __name__ == "__main__":
    unittest.main()
