"""Unit tests for scripts/train_pixel_classifier.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from scripts.train_pixel_classifier import main as train_pixel_classifier_main


class TrainPixelClassifierTests(unittest.TestCase):
    def _write_case(self, root: Path, alias: str, image_id: str) -> dict[str, Path]:
        raw_dir = root / "raw"
        annotations_dir = root / "annotations"
        tiles_dir = root / f"tiles_{alias}"
        maps_dir = root / f"maps_{alias}"
        raw_dir.mkdir(exist_ok=True)
        (annotations_dir / "scribbles").mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)
        maps_dir.mkdir(parents=True, exist_ok=True)

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[..., 0] = 128
        Image.fromarray(img).save(raw_dir / f"{image_id}.png")

        scribble = np.zeros((16, 16), dtype=np.uint8)
        scribble[0:4, 0:4] = 1
        scribble[8:12, 8:12] = 2
        Image.fromarray(scribble).save(annotations_dir / "scribbles" / f"{image_id}_scribble_labels.png")

        (tiles_dir / "tile_manifest_meta.json").write_text(json.dumps({"svs_level_dimensions_wh": [16, 16]}), encoding="utf-8")

        tile_probs = pd.DataFrame(
            [
                {"image_id": image_id, "tile_x": 0, "tile_y": 0, "tile_w": 8, "tile_h": 8, "prob_positive": 0.8},
                {"image_id": image_id, "tile_x": 8, "tile_y": 8, "tile_w": 8, "tile_h": 8, "prob_positive": 0.2},
            ]
        )
        tile_probs_path = maps_dir / "tile_probabilities.csv"
        tile_probs.to_csv(tile_probs_path, index=False)
        return {"tiles": tiles_dir, "tile_probs": tile_probs_path, "raw": raw_dir, "annotations": annotations_dir}

    def test_single_image_cli_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = root / "config.yaml"
            cfg.write_text(
                "classes:\n  label_encoding:\n    Unlabeled: 0\n    Positive_Tumor: 1\n    Negative_Tumor: 2\n    NonTumor: 3\n    Ignore: 4\n",
                encoding="utf-8",
            )
            case = self._write_case(root, "a", "img_a")
            out_dir = root / "models_single"

            argv = [
                "train_pixel_classifier.py",
                "--config",
                str(cfg),
                "--image-id",
                "img_a",
                "--raw-dir",
                str(case["raw"]),
                "--annotations-dir",
                str(case["annotations"]),
                "--tiles-dir",
                str(case["tiles"]),
                "--tile-probabilities",
                str(case["tile_probs"]),
                "--output-dir",
                str(out_dir),
            ]
            with patch("sys.argv", argv):
                train_pixel_classifier_main()

            spec = json.loads((out_dir / "pixel_feature_spec.json").read_text(encoding="utf-8"))
            self.assertEqual(spec["model_scope"], "single_image_model")
            self.assertEqual(spec["n_images"], 1)

    def test_shared_cohort_records_balanced_sampling_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = root / "config.yaml"
            cfg.write_text(
                "classes:\n  label_encoding:\n    Unlabeled: 0\n    Positive_Tumor: 1\n    Negative_Tumor: 2\n    NonTumor: 3\n    Ignore: 4\npixel_classifier:\n  max_samples_per_image_per_class: 5\n  equalize_image_class_sampling: true\n",
                encoding="utf-8",
            )
            case_a = self._write_case(root, "a", "img_a")
            case_b = self._write_case(root, "b", "img_b")
            out_dir = root / "models_shared"

            cohort = pd.DataFrame(
                [
                    {
                        "alias": "a",
                        "image_id": "img_a",
                        "tiles_dir": case_a["tiles"].as_posix(),
                        "tile_probabilities": case_a["tile_probs"].as_posix(),
                    },
                    {
                        "alias": "b",
                        "image_id": "img_b",
                        "tiles_dir": case_b["tiles"].as_posix(),
                        "tile_probabilities": case_b["tile_probs"].as_posix(),
                    },
                ]
            )
            cohort_path = root / "cohort.csv"
            cohort.to_csv(cohort_path, index=False)

            argv = [
                "train_pixel_classifier.py",
                "--config",
                str(cfg),
                "--cohort-file",
                str(cohort_path),
                "--raw-dir",
                str(case_a["raw"]),
                "--annotations-dir",
                str(case_a["annotations"]),
                "--output-dir",
                str(out_dir),
            ]
            with patch("sys.argv", argv):
                train_pixel_classifier_main()

            spec = json.loads((out_dir / "pixel_feature_spec.json").read_text(encoding="utf-8"))
            self.assertEqual(spec["model_scope"], "shared_project_model")
            self.assertEqual(spec["n_images"], 2)
            self.assertTrue(spec["equalize_image_class_sampling"])
            self.assertEqual(len(spec["per_image_sample_counts"]), 2)


if __name__ == "__main__":
    unittest.main()
