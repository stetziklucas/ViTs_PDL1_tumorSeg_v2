from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts import embed_vit


def test_cache_signature_changes_on_encoder_id_and_model_name() -> None:
    base = {"encoder_id": "current_timm", "backend": "timm", "model_name": "vit_base_patch16_224", "pretrained": True, "frozen": True}
    sig1 = embed_vit.make_cache_signature("a", "b", base, 224)
    sig2 = embed_vit.make_cache_signature("a", "b", {**base, "encoder_id": "other"}, 224)
    sig3 = embed_vit.make_cache_signature("a", "b", {**base, "model_name": "vit_small_patch16_224"}, 224)
    assert sig1 != sig2
    assert sig1 != sig3


def test_metadata_and_artifacts(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    out_dir = tmp_path / "emb"
    image_path = raw_dir / "img1.png"
    Image.new("RGB", (224, 224), (128, 0, 0)).save(image_path)

    manifest = pd.DataFrame([{"tile_x": 0, "tile_y": 0, "tile_w": 224, "tile_h": 224, "source_image": str(image_path)}])
    manifest.to_csv(input_dir / "tile_manifest.csv", index=False)

    class Dummy:
        batch_size = 32
        def __init__(self, spec, require_pretrained):
            self.spec = spec
        def encode_tiles(self, tiles):
            return np.ones((len(tiles), 4), dtype=np.float32)
        def metadata(self):
            return {"encoder_weight_source": "pretrained"}

    monkeypatch.setattr(embed_vit, "TimmTileEmbeddingEncoder", Dummy)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""
tiling:
  tile_size_px: 224
embedding_encoder:
  selected: current_timm
  registry:
    current_timm:
      display_name: Current ViT baseline
      backend: timm
      model_name: vit_base_patch16_224
      pretrained: true
      frozen: true
      batch_size: 32
""", encoding="utf-8")

    class _P:
        def parse_args(self):
            from argparse import Namespace
            return Namespace(config=cfg, image_id="img1", input=input_dir, raw_dir=raw_dir, output_dir=out_dir, embedding_encoder=None, encoder=None)

    monkeypatch.setattr(embed_vit, "build_parser", lambda: _P())
    embed_vit.main()

    emb = np.load(out_dir / "embeddings.npy")
    assert emb.dtype == np.float32
    df = pd.read_csv(out_dir / "tile_manifest_with_embeddings_index.csv")
    assert len(df) == len(manifest)
    assert list(df["embedding_index"]) == list(range(len(manifest)))
    meta = json.loads((out_dir / "embeddings_cache_meta.json").read_text(encoding="utf-8"))
    assert meta["encoder_id"] == "current_timm"
    assert meta["encoder_display_name"] == "Current ViT baseline"
    assert meta["embedding_dim"] == 4
    assert meta["embedding_dtype"] == "float32"
