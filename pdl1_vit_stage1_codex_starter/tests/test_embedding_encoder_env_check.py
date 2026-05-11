from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from scripts import check_embedding_encoder_env as env_check


BASE_CFG = """
embedding_encoder:
  selected: current_timm
  registry:
    current_timm:
      backend: timm
      model_name: vit_base_patch16_224
      frozen: true
    hibou_b:
      backend: hf_transformers
      model_name: histai/hibou-b
      trust_remote_code: true
      requires_hf_auth: true
      frozen: true
"""


def test_current_timm_check_ok(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "cfg.yaml"; cfg.write_text(BASE_CFG, encoding="utf-8")
    fake_timm = ModuleType("timm"); fake_timm.__version__ = "1.2.3"
    monkeypatch.setitem(__import__("sys").modules, "timm", fake_timm)
    monkeypatch.setattr("sys.argv", ["x", "--config", str(cfg), "--embedding-encoder", "current_timm"])
    env_check.main()
    out = capsys.readouterr().out
    assert "timm_version=1.2.3" in out


def test_hf_reports_transformers_version_without_try_load(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "cfg.yaml"; cfg.write_text(BASE_CFG, encoding="utf-8")
    fake_t = ModuleType("transformers"); fake_t.__version__ = "4.53.3"
    fake_hf = ModuleType("huggingface_hub"); fake_hf.__version__ = "1.0.0"; fake_hf.whoami = lambda: {"name": "tester", "token": "hf_secret"}
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_t)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(env_check.importlib.util, "find_spec", lambda n: object() if n == "transformers.onnx" else None)
    monkeypatch.setattr("sys.argv", ["x", "--config", str(cfg), "--embedding-encoder", "hibou_b"])
    env_check.main()
    out = capsys.readouterr().out
    assert "transformers_version=4.53.3" in out
    assert "hf_secret" not in out


def test_missing_transformers_onnx_fails(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"; cfg.write_text(BASE_CFG, encoding="utf-8")
    fake_t = ModuleType("transformers"); fake_t.__version__ = "5.0.0"
    fake_hf = ModuleType("huggingface_hub"); fake_hf.__version__ = "1.0.0"
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_t)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(env_check.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr("sys.argv", ["x", "--config", str(cfg), "--embedding-encoder", "hibou_b"])
    with pytest.raises(SystemExit, match="transformers.onnx.*transformers>=4.53.3,<5"):
        env_check.main()
