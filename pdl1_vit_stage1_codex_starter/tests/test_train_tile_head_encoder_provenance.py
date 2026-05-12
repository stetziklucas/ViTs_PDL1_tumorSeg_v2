from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.train_tile_head import _collect_encoder_provenance, _resolve_cases, main as train_tile_head_main


def _write_case(root: Path, alias: str, image_id: str, encoder_id: str = "current_timm") -> dict[str, Path]:
    embeddings_dir = root / f"embeddings_{alias}"
    maps_dir = root / f"maps_{alias}"
    labels_path = root / f"tile_labels_{alias}.csv"
    embeddings_dir.mkdir(parents=True)
    maps_dir.mkdir(parents=True)

    labels = pd.DataFrame(
        [
            {"image_id": image_id, "tile_id": f"{image_id}_0", "embedding_index": 0, "tile_row": 0, "tile_col": 0, "tile_x": 0, "tile_y": 0, "tile_w": 8, "tile_h": 8, "label": "Positive_Context"},
            {"image_id": image_id, "tile_id": f"{image_id}_1", "embedding_index": 1, "tile_row": 0, "tile_col": 1, "tile_x": 8, "tile_y": 0, "tile_w": 8, "tile_h": 8, "label": "Negative_Context"},
        ]
    )
    labels.to_csv(labels_path, index=False)
    labels[["tile_id", "embedding_index"]].to_csv(embeddings_dir / "tile_manifest_with_embeddings_index.csv", index=False)
    np.save(embeddings_dir / "embeddings.npy", np.array([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=np.float32))
    meta = {
        "encoder_id": encoder_id,
        "encoder_display_name": encoder_id,
        "encoder_backend": "hf_transformers" if encoder_id == "hibou_b" else "timm",
        "encoder_model_name": "histai/hibou-b" if encoder_id == "hibou_b" else "vit_base_patch16_224",
        "encoder_pooling": "cls",
        "embedding_dim": 4,
        "embedding_dtype": "float32",
        "encoder_weight_source": "pretrained",
        "encoder_trust_remote_code": encoder_id == "hibou_b",
        "encoder_requires_hf_auth": encoder_id == "hibou_b",
    }
    (embeddings_dir / "embeddings_cache_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"labels": labels_path, "embeddings": embeddings_dir, "maps": maps_dir}


def test_resolve_cases_single_and_cohort_no_nameerror() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        c = _write_case(root, "a", "img_a")
        args = Namespace(cohort_file=None, labels=c["labels"], embeddings_dir=c["embeddings"], maps_dir=c["maps"], probs_manifest=c["maps"] / "tile_probabilities.csv")
        single = _resolve_cases(args)
        assert len(single) == 1

        cohort_path = root / "cohort.csv"
        pd.DataFrame([
            {"alias": "a", "image_id": "img_a", "labels_path": c["labels"], "embeddings_dir": c["embeddings"], "maps_dir": c["maps"]}
        ]).to_csv(cohort_path, index=False)
        args.cohort_file = cohort_path
        cohort_cases = _resolve_cases(args)
        assert len(cohort_cases) == 1


def test_single_image_writes_encoder_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cfg = root / "config.yaml"
        cfg.write_text("tile_head:\n  positive_threshold: 0.6\n", encoding="utf-8")
        c = _write_case(root, "a", "img_a", encoder_id="current_timm")
        out = root / "models"
        with patch("sys.argv", ["train_tile_head.py", "--config", str(cfg), "--labels", str(c["labels"]), "--embeddings-dir", str(c["embeddings"]), "--maps-dir", str(c["maps"]), "--probs-manifest", str(c["maps"] / "tile_probabilities.csv"), "--output-dir", str(out)]):
            train_tile_head_main()
        metrics = json.loads((out / "tile_cv_metrics.json").read_text(encoding="utf-8"))
        assert metrics["encoder_provenance"]["encoder_id"] == "current_timm"


def test_shared_project_provenance_and_mixed_guard() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        c1 = _write_case(root, "a", "img_a", encoder_id="current_timm")
        c2 = _write_case(root, "b", "img_b", encoder_id="current_timm")
        encoder, per_case = _collect_encoder_provenance([
            {"alias": "a", "image_id": "img_a", "embeddings_dir": c1["embeddings"]},
            {"alias": "b", "image_id": "img_b", "embeddings_dir": c2["embeddings"]},
        ])
        assert encoder is not None and encoder["encoder_id"] == "current_timm"
        assert set(per_case.keys()) == {"a", "b"}

        c3 = _write_case(root, "c", "img_c", encoder_id="hibou_b")
        with pytest.raises(ValueError, match="Mixed encoder provenance|inconsistent"):
            _collect_encoder_provenance([
                {"alias": "a", "image_id": "img_a", "embeddings_dir": c1["embeddings"]},
                {"alias": "c", "image_id": "img_c", "embeddings_dir": c3["embeddings"]},
            ])


def test_shared_training_writes_per_case_provenance_and_probabilities() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cfg = root / "config.yaml"
        cfg.write_text("tile_head:\n  positive_threshold: 0.6\n", encoding="utf-8")
        c1 = _write_case(root, "a", "img_a")
        c2 = _write_case(root, "b", "img_b")
        cohort = root / "cohort.csv"
        pd.DataFrame([
            {"alias": "a", "image_id": "img_a", "labels_path": c1["labels"], "embeddings_dir": c1["embeddings"], "maps_dir": c1["maps"]},
            {"alias": "b", "image_id": "img_b", "labels_path": c2["labels"], "embeddings_dir": c2["embeddings"], "maps_dir": c2["maps"]},
        ]).to_csv(cohort, index=False)
        out = root / "models"
        with patch("sys.argv", ["train_tile_head.py", "--config", str(cfg), "--cohort-file", str(cohort), "--output-dir", str(out)]):
            train_tile_head_main()
        metrics = json.loads((out / "tile_cv_metrics.json").read_text(encoding="utf-8"))
        assert metrics["encoder_provenance"]["encoder_id"] == "current_timm"
        assert "per_case_encoder_provenance" in metrics
        assert (c1["maps"] / "tile_probabilities.csv").exists()
        assert (c2["maps"] / "tile_probabilities.csv").exists()
