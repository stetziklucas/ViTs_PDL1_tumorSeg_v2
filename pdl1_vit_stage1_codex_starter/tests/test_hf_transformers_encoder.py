import pytest
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

from encoder_backends.base import EncoderSpec
from encoder_backends.hf_transformers_encoder import HfTransformersTileEmbeddingEncoder


def test_hf_encoder_pooling_auto(monkeypatch):
    class Proc:
        @classmethod
        def from_pretrained(cls, *a, **k):
            assert k["trust_remote_code"] is True
            return cls()
        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros((len(images),3,4,4))}
    class Model:
        @classmethod
        def from_pretrained(cls, *a, **k):
            assert k["trust_remote_code"] is True
            return cls()
        def eval(self): return self
        def parameters(self): return []
        def to(self, d): return self
        def __call__(self, **kwargs):
            return SimpleNamespace(pooler_output=torch.ones((kwargs['pixel_values'].shape[0],5)))
    import encoder_backends.hf_transformers_encoder as m
    monkeypatch.setitem(__import__('sys').modules, 'transformers', SimpleNamespace(AutoImageProcessor=Proc, AutoModel=Model))
    spec=EncoderSpec('hibou_b','H','hf_transformers','histai/hibou-b',trust_remote_code=True,extra={'pooling':'auto'})
    enc=HfTransformersTileEmbeddingEncoder(spec)
    out=enc.encode_tiles([Image.new('RGB',(4,4))])
    assert out.shape==(1,5) and out.dtype==np.float32




def test_hf_encoder_transformers_onnx_guidance(monkeypatch):
    class Proc:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()
    class Model:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise ModuleNotFoundError("No module named 'transformers.onnx'")
    monkeypatch.setitem(__import__('sys').modules, 'transformers', SimpleNamespace(AutoImageProcessor=Proc, AutoModel=Model))
    spec=EncoderSpec('hibou_b','H','hf_transformers','histai/hibou-b',trust_remote_code=True,extra={'pooling':'auto'})
    with pytest.raises(RuntimeError, match=r"transformers\.onnx.*transformers>=4\.53\.3,<5.*check_embedding_encoder_env\.py"):
        HfTransformersTileEmbeddingEncoder(spec)
