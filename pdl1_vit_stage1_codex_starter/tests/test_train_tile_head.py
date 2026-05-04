"""Focused tests for scripts/train_tile_head.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.train_tile_head import main as train_tile_head_main


class TrainTileHeadTests(unittest.TestCase):
    def _write_case(self, root: Path, alias: str, image_id: str, base_value: float) -> dict[str, Path]:
        embeddings_dir = root / f"embeddings_{alias}"
        maps_dir = root / f"maps_{alias}"
        labels_path = root / f"tile_labels_{alias}.csv"
        embeddings_dir.mkdir(parents=True)
        maps_dir.mkdir(parents=True)

        labels_df = pd.DataFrame(
            [
                {
                    "image_id": image_id,
                    "tile_id": f"{image_id}_0",
                    "embedding_index": 0,
                    "tile_row": 0,
                    "tile_col": 0,
                    "tile_x": 0,
                    "tile_y": 0,
                    "tile_w": 8,
                    "tile_h": 8,
                    "label": "Positive_Context",
                    "sample_weight": 1.0,
                },
                {
                    "image_id": image_id,
                    "tile_id": f"{image_id}_1",
                    "embedding_index": 1,
                    "tile_row": 0,
                    "tile_col": 1,
                    "tile_x": 8,
                    "tile_y": 0,
                    "tile_w": 8,
                    "tile_h": 8,
                    "label": "Negative_Context",
                    "sample_weight": 1.0,
                },
                {
                    "image_id": image_id,
                    "tile_id": f"{image_id}_2",
                    "embedding_index": 2,
                    "tile_row": 1,
                    "tile_col": 0,
                    "tile_x": 0,
                    "tile_y": 8,
                    "tile_w": 8,
                    "tile_h": 8,
                    "label": "Ignore",
                    "sample_weight": 0.0,
                },
            ]
        )
        labels_df.to_csv(labels_path, index=False)
        labels_df[["tile_id", "embedding_index"]].to_csv(embeddings_dir / "tile_manifest_with_embeddings_index.csv", index=False)
        np.save(
            embeddings_dir / "embeddings.npy",
            np.array([[base_value, base_value + 1], [-base_value, -base_value - 1], [0.0, 0.0]], dtype=np.float32),
        )
        (embeddings_dir / "embeddings_cache_meta.json").write_text(
            json.dumps({"encoder_weight_source": "pretrained"}), encoding="utf-8"
        )
        return {"labels": labels_path, "embeddings": embeddings_dir, "maps": maps_dir}

    def test_scores_all_tiles_but_trains_on_usable_labels_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            cfg_path = root / "config.yaml"
            cfg_path.write_text("tile_head:\n  positive_threshold: 0.60\n", encoding="utf-8")
            case = self._write_case(root, alias="a", image_id="toy_001", base_value=5.0)
            probs_path = case["maps"] / "tile_probabilities.csv"

            argv = [
                "train_tile_head.py",
                "--config",
                str(cfg_path),
                "--labels",
                str(case["labels"]),
                "--embeddings-dir",
                str(case["embeddings"]),
                "--output-dir",
                str(models_dir),
                "--maps-dir",
                str(case["maps"]),
                "--probs-manifest",
                str(probs_path),
            ]
            with patch("sys.argv", argv):
                train_tile_head_main()

            out_probs = pd.read_csv(probs_path)
            self.assertEqual(len(out_probs), 3)
            self.assertIn("prob_positive", out_probs.columns)

            with (models_dir / "tile_cv_metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertEqual(metrics["n_usable_tiles"], 2)
            self.assertEqual(metrics["n_scored_tiles"], 3)
            self.assertEqual(metrics["class_counts"]["Ignore"], 1)

    def test_shared_cohort_writes_per_case_outputs_and_group_cv_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg_path = root / "config.yaml"
            cfg_path.write_text("tile_head:\n  positive_threshold: 0.60\n  equalize_image_weight: true\n", encoding="utf-8")
            models_dir = root / "models_shared"

            case_a = self._write_case(root, alias="a", image_id="img_a", base_value=4.0)
            case_b = self._write_case(root, alias="b", image_id="img_b", base_value=6.0)
            cohort = pd.DataFrame(
                [
                    {
                        "alias": "a",
                        "image_id": "img_a",
                        "labels_path": case_a["labels"].as_posix(),
                        "embeddings_dir": case_a["embeddings"].as_posix(),
                        "maps_dir": case_a["maps"].as_posix(),
                    },
                    {
                        "alias": "b",
                        "image_id": "img_b",
                        "labels_path": case_b["labels"].as_posix(),
                        "embeddings_dir": case_b["embeddings"].as_posix(),
                        "maps_dir": case_b["maps"].as_posix(),
                    },
                ]
            )
            cohort_path = root / "cohort.csv"
            cohort.to_csv(cohort_path, index=False)

            with patch("sys.argv", ["train_tile_head.py", "--config", str(cfg_path), "--cohort-file", str(cohort_path), "--output-dir", str(models_dir)]):
                train_tile_head_main()

            metrics = json.loads((models_dir / "tile_cv_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["n_images"], 2)
            self.assertEqual(metrics["cv_mode"], "group_kfold_by_image_id")
            self.assertTrue(metrics["equalize_image_weight"])

            self.assertTrue((case_a["maps"] / "tile_probabilities.csv").exists())
            self.assertTrue((case_b["maps"] / "tile_probabilities.csv").exists())
            self.assertTrue((case_a["maps"] / "tile_prob_map.png").exists())
            self.assertTrue((case_b["maps"] / "tile_prob_map.png").exists())


if __name__ == "__main__":
    unittest.main()
