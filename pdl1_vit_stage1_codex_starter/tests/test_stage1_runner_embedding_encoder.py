from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from annotation_readiness import AnnotationReadinessResult
from scripts import embed_vit, run_stage1_image, run_stage1_project


def _ready(image_id: str = "IMG") -> AnnotationReadinessResult:
    return AnnotationReadinessResult(image_id=image_id, artifact_exists={"annotation_meta": True, "roi_mask": True, "scribble_labels": True}, polygon_counts={"Positive_Tumor": 1, "Negative_Tumor": 1, "NonTumor": 0, "Ignore": 0}, pixel_counts={"Positive_Tumor": 10, "Negative_Tumor": 10, "NonTumor": 0, "Ignore": 0}, roi_positive_pixels=10, status_code="READY", status_label="READY", summary_message="ready", next_action="continue", notes=[])


def test_parsers_accept_embedding_encoder_aliases() -> None:
    assert run_stage1_image.build_parser().parse_args(["--image-id", "i", "--run-tag", "r", "--embedding-encoder", "hibou_b"]).embedding_encoder == "hibou_b"
    assert run_stage1_image.build_parser().parse_args(["--image-id", "i", "--run-tag", "r", "--encoder", "hibou_b"]).encoder == "hibou_b"
    assert run_stage1_project.build_parser().parse_args(["--config", "c.yaml", "--project-tag", "p", "--case", "a=i", "--embedding-encoder", "hibou_b"]).embedding_encoder == "hibou_b"
    assert run_stage1_project.build_parser().parse_args(["--config", "c.yaml", "--project-tag", "p", "--case", "a=i", "--encoder", "hibou_b"]).encoder == "hibou_b"


def test_embed_vit_conflict_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("tiling:\n  tile_size_px: 224\nembedding_encoder:\n  selected: current_timm\n  registry: {current_timm: {display_name: x, backend: timm, model_name: vit_base_patch16_224, pretrained: true, frozen: true}}\n", encoding="utf-8")
    t = tmp_path / "tiles"; t.mkdir(); (t / "tile_manifest.csv").write_text("tile_x,tile_y,tile_w,tile_h,source_image\n0,0,1,1,x.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflict"):
        monkeypatch.setattr("sys.argv", ["embed_vit.py", "--config", str(cfg), "--image-id", "i", "--input", str(t), "--output-dir", str(tmp_path / "o"), "--embedding-encoder", "a", "--encoder", "b"])
        embed_vit.main()


def test_image_and_project_runner_propagate_encoder_and_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "cfg.yaml"; cfg.write_text("classes:\n  label_encoding:\n    Positive_Tumor: 1\n", encoding="utf-8")
    args = Namespace(config=cfg, image_id="IMG", run_tag="r1", raw_dir=tmp_path/"raw", annotations_dir=tmp_path/"ann", outputs_root=tmp_path/"out", models_root=tmp_path/"models", embedding_encoder="hibou_b", encoder=None)
    calls: list[list[str]] = []
    assert run_stage1_image.run_pipeline(args, readiness_fn=lambda **_: _ready("IMG"), step_runner=lambda c, _l: calls.append(c) or 0) == 0
    assert ["--embedding-encoder", "hibou_b"] == next(c for c in calls if Path(c[1]).name == "embed_vit.py")[-2:]
    assert json.loads((tmp_path/"out"/"reports_r1"/"stage1_run_summary.json").read_text(encoding="utf-8"))["embedding_encoder"] == "hibou_b"
    assert "distinct project tag" in capsys.readouterr().out

    monkeypatch.setattr(run_stage1_project, "compute_annotation_readiness", lambda **kw: _ready(kw["image_id"]))
    proj_calls: list[list[str]] = []
    monkeypatch.setattr(run_stage1_project, "_run_subprocess_streaming", lambda c, _l: proj_calls.append(c) or 0)
    monkeypatch.setattr("sys.argv", ["run_stage1_project.py", "--config", str(cfg), "--project-tag", "proj", "--case", "a=img_a", "--case", "b=img_b", "--outputs-root", str(tmp_path/"o2"), "--models-root", str(tmp_path/"m2"), "--embedding-encoder", "hibou_b"])
    assert run_stage1_project.main() == 0
    embeds = [c for c in proj_calls if Path(c[1]).name == "embed_vit.py"]
    assert len(embeds) == 2 and all(["--embedding-encoder", "hibou_b"] == c[-2:] for c in embeds)
    assert json.loads((tmp_path/"o2"/"reports_training_proj"/"stage1_project_run_summary.json").read_text(encoding="utf-8"))["embedding_encoder"] == "hibou_b"


def test_project_runner_conflicting_flags_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"; cfg.write_text("classes:\n  label_encoding:\n    Positive_Tumor: 1\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_stage1_project.py", "--config", str(cfg), "--project-tag", "proj", "--case", "a=img_a", "--case", "b=img_b", "--embedding-encoder", "a", "--encoder", "b"])
    with pytest.raises(ValueError, match="conflict"):
        run_stage1_project.main()
