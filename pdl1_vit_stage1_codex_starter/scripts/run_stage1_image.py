"""Run the full Stage 1 single-image pipeline after annotation-readiness gating."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from annotation_readiness import AnnotationReadinessResult, compute_annotation_readiness
from report_history import refresh_history_for_image


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for one-command Stage 1 execution."""
    parser = argparse.ArgumentParser(description="Run full Stage 1 pipeline for one image after readiness preflight.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=True, help="Image identifier to process.")
    parser.add_argument("--run-tag", required=True, help="Run tag suffix used for outputs/models directories.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing source images.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Directory containing canonical annotation artifacts.",
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Root directory for output artifacts.")
    parser.add_argument("--models-root", type=Path, default=Path("models"), help="Root directory for model artifacts.")
    parser.add_argument("--embedding-encoder", type=str, default=None, help="Embedding encoder override.")
    parser.add_argument("--encoder", type=str, default=None, help="Alias for --embedding-encoder.")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config did not parse into a dictionary.")
    return config


def derive_paths(*, run_tag: str, outputs_root: Path, models_root: Path) -> dict[str, Path]:
    """Derive canonical Stage 1 path layout from run_tag."""
    return {
        "tiles_dir": outputs_root / f"tiles_{run_tag}",
        "embeddings_dir": outputs_root / f"embeddings_{run_tag}",
        "tile_maps_dir": outputs_root / f"maps_{run_tag}",
        "fused_maps_dir": outputs_root / f"maps_{run_tag}_fused",
        "masks_dir": outputs_root / f"masks_{run_tag}",
        "overlays_dir": outputs_root / f"overlays_{run_tag}",
        "reports_dir": outputs_root / f"reports_{run_tag}",
        "tile_model_dir": models_root / f"tile_head_{run_tag}",
        "pixel_model_dir": models_root / f"pixel_classifier_{run_tag}",
    }


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _readiness_payload(result: AnnotationReadinessResult) -> dict[str, Any]:
    payload = result.to_dict()
    return {
        "status_code": payload["status_code"],
        "status_label": payload["status_label"],
        "summary_message": payload["summary_message"],
        "next_action": payload["next_action"],
        "roi_positive_pixels": payload["roi_positive_pixels"],
        "polygon_counts": payload["polygon_counts"],
        "pixel_counts": payload["pixel_counts"],
        "notes": payload["notes"],
    }


def _print_readiness_summary(image_id: str, readiness: dict[str, Any]) -> None:
    print(f"[readiness] image_id={image_id}")
    print(f"[readiness] status={readiness['status_code']} ({readiness['status_label']})")
    print(f"[readiness] summary={readiness['summary_message']}")
    print(f"[readiness] next_action={readiness['next_action']}")


def _run_subprocess_streaming(command: list[str], log_handle: Any) -> int:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()
    return int(process.wait())


def _build_steps(args: argparse.Namespace, paths: dict[str, Path], selected_encoder: str | None) -> list[tuple[str, list[str]]]:
    python = sys.executable
    tile_probs_csv = paths["tile_maps_dir"] / "tile_probabilities.csv"
    pixel_model = paths["pixel_model_dir"] / "pixel_model.pkl"
    pixel_feature_spec = paths["pixel_model_dir"] / "pixel_feature_spec.json"

    embed_cmd = [
        python,
        "scripts/embed_vit.py",
        "--config",
        str(args.config),
        "--image-id",
        args.image_id,
        "--input",
        str(paths["tiles_dir"]),
        "--raw-dir",
        str(args.raw_dir),
        "--output-dir",
        str(paths["embeddings_dir"]),
    ]
    if selected_encoder:
        embed_cmd.extend(["--embedding-encoder", selected_encoder])

    return [
        (
            "extract_tiles",
            [
                python,
                "scripts/extract_tiles.py",
                "--config",
                str(args.config),
                "--image-id",
                args.image_id,
                "--input",
                str(args.raw_dir),
                "--annotations-dir",
                str(args.annotations_dir),
                "--output-dir",
                str(paths["tiles_dir"]),
            ],
        ),
        (
            "embed_vit",
            embed_cmd,
        ),
        (
            "make_tile_labels",
            [
                python,
                "scripts/make_tile_labels.py",
                "--config",
                str(args.config),
                "--image-id",
                args.image_id,
                "--annotations-dir",
                str(args.annotations_dir),
                "--tiles-dir",
                str(paths["tiles_dir"]),
                "--embeddings-dir",
                str(paths["embeddings_dir"]),
                "--output-dir",
                str(paths["tiles_dir"]),
            ],
        ),
        (
            "train_tile_head",
            [
                python,
                "scripts/train_tile_head.py",
                "--config",
                str(args.config),
                "--labels",
                str(paths["tiles_dir"] / "tile_labels.csv"),
                "--embeddings-dir",
                str(paths["embeddings_dir"]),
                "--output-dir",
                str(paths["tile_model_dir"]),
                "--maps-dir",
                str(paths["tile_maps_dir"]),
                "--probs-manifest",
                str(tile_probs_csv),
                "--smoke-image-id",
                args.image_id,
            ],
        ),
        (
            "train_pixel_classifier",
            [
                python,
                "scripts/train_pixel_classifier.py",
                "--config",
                str(args.config),
                "--image-id",
                args.image_id,
                "--raw-dir",
                str(args.raw_dir),
                "--annotations-dir",
                str(args.annotations_dir),
                "--tiles-dir",
                str(paths["tiles_dir"]),
                "--tile-probabilities",
                str(tile_probs_csv),
                "--output-dir",
                str(paths["pixel_model_dir"]),
            ],
        ),
        (
            "run_inference",
            [
                python,
                "scripts/run_inference.py",
                "--config",
                str(args.config),
                "--image-id",
                args.image_id,
                "--raw-dir",
                str(args.raw_dir),
                "--annotations-dir",
                str(args.annotations_dir),
                "--tiles-dir",
                str(paths["tiles_dir"]),
                "--tile-probabilities",
                str(tile_probs_csv),
                "--pixel-model",
                str(pixel_model),
                "--pixel-feature-spec",
                str(pixel_feature_spec),
                "--maps-dir",
                str(paths["fused_maps_dir"]),
                "--masks-dir",
                str(paths["masks_dir"]),
                "--overlays-dir",
                str(paths["overlays_dir"]),
                "--reports-dir",
                str(paths["reports_dir"]),
            ],
        ),
        (
            "make_report",
            [
                python,
                "scripts/make_report.py",
                "--config",
                str(args.config),
                "--image-id",
                args.image_id,
                "--annotations-dir",
                str(args.annotations_dir),
                "--tile-maps-dir",
                str(paths["tile_maps_dir"]),
                "--pixel-maps-dir",
                str(paths["fused_maps_dir"]),
                "--masks-dir",
                str(paths["masks_dir"]),
                "--overlays-dir",
                str(paths["overlays_dir"]),
                "--reports-dir",
                str(paths["reports_dir"]),
                "--tile-model-dir",
                str(paths["tile_model_dir"]),
            ],
        ),
    ]


def _build_key_artifacts(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "tile_probabilities_csv": (paths["tile_maps_dir"] / "tile_probabilities.csv").as_posix(),
        "tile_prob_map_png": (paths["tile_maps_dir"] / "tile_prob_map.png").as_posix(),
        "pixel_prob_map_png": (paths["fused_maps_dir"] / "pixel_prob_map.png").as_posix(),
        "positive_mask_png": (paths["masks_dir"] / "positive_mask.png").as_posix(),
        "overlay_png": (paths["overlays_dir"] / "overlay.png").as_posix(),
        "metrics_json": (paths["reports_dir"] / "metrics.json").as_posix(),
        "report_summary_md": (paths["reports_dir"] / "report_summary.md").as_posix(),
        "report_summary_json": (paths["reports_dir"] / "report_summary.json").as_posix(),
        "report_pdf": (paths["reports_dir"] / "one_page_report.pdf").as_posix(),
        "runner_log": (paths["reports_dir"] / "stage1_runner.log").as_posix(),
    }


def _write_run_summary(reports_dir: Path, summary_payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = reports_dir / "stage1_run_summary.json"
    md_path = reports_dir / "stage1_run_summary.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# Stage 1 Run Summary",
        "",
        f"image_id: {summary_payload['image_id']}",
        f"run_tag: {summary_payload['run_tag']}",
        f"final_status: {summary_payload['final_status']}",
        f"started_at_utc: {summary_payload['started_at_utc']}",
        f"ended_at_utc: {summary_payload['ended_at_utc']}",
        f"elapsed_seconds: {summary_payload['elapsed_seconds']:.2f}",
        "",
        "## Readiness at launch",
        f"- status: {summary_payload['readiness']['status_code']} ({summary_payload['readiness']['status_label']})",
        f"- summary: {summary_payload['readiness']['summary_message']}",
        f"- next action: {summary_payload['readiness']['next_action']}",
        "",
        "## Step outcomes",
        "",
        "| step | exit_code | elapsed_seconds |",
        "| --- | ---: | ---: |",
    ]
    for item in summary_payload["steps"]:
        lines.append(f"| {item['step']} | {item['exit_code']} | {item['elapsed_seconds']:.2f} |")

    lines.extend(
        [
            "",
            "## Next review files",
            *[f"- `{path}`" for path in summary_payload["next_review_files"]],
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_pipeline(
    args: argparse.Namespace,
    *,
    readiness_fn: Callable[..., AnnotationReadinessResult] = compute_annotation_readiness,
    step_runner: Callable[[list[str], Any], int] = _run_subprocess_streaming,
) -> int:
    """Execute Stage 1 pipeline with readiness gating and persisted run summary."""
    start_ts = _iso_utc_now()
    selected_encoder = args.embedding_encoder or args.encoder
    print(f"Embedding encoder: {selected_encoder or 'config default'}. Use a distinct project tag for side-by-side comparisons.")
    start_monotonic = time.monotonic()
    paths = derive_paths(run_tag=args.run_tag, outputs_root=args.outputs_root, models_root=args.models_root)
    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "stage1_runner.log"

    config: dict[str, Any]
    readiness: dict[str, Any]
    steps: list[dict[str, Any]] = []
    final_status = "UNKNOWN"
    exit_code = 0

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"stage1_runner_start_utc={start_ts}\n")
        try:
            config = load_config(args.config)
            readiness_result = readiness_fn(config=config, annotations_dir=args.annotations_dir, image_id=args.image_id)
            readiness = _readiness_payload(readiness_result)
            _print_readiness_summary(args.image_id, readiness)
            log_handle.write(json.dumps({"readiness": readiness}, sort_keys=True) + "\n")

            if readiness_result.status_code == "ERROR":
                final_status = "READINESS_ERROR"
                exit_code = 2
            elif readiness_result.status_code != "READY":
                final_status = "NOT_READY"
                exit_code = 1
            else:
                for step_name, command in _build_steps(args, paths, selected_encoder):
                    started = time.monotonic()
                    command_text = shlex.join(command)
                    print(f"\n[step] {step_name}: {command_text}")
                    log_handle.write(f"[step] {step_name}: {command_text}\n")
                    rc = step_runner(command, log_handle)
                    elapsed = time.monotonic() - started
                    steps.append(
                        {
                            "step": step_name,
                            "command": command_text,
                            "exit_code": int(rc),
                            "elapsed_seconds": float(elapsed),
                        }
                    )
                    if rc != 0:
                        final_status = "FAILED"
                        exit_code = 3
                        break
                if final_status == "UNKNOWN":
                    final_status = "SUCCESS"
                    exit_code = 0
        except Exception as exc:
            print(f"[runner] readiness/artifact error: {exc}")
            log_handle.write(f"[runner] readiness/artifact error: {exc}\n")
            readiness = {
                "status_code": "ERROR",
                "status_label": "Artifact error",
                "summary_message": str(exc),
                "next_action": "Fix readiness/config artifacts and rerun.",
                "roi_positive_pixels": 0,
                "polygon_counts": {},
                "pixel_counts": {},
                "notes": [],
            }
            final_status = "READINESS_ERROR"
            exit_code = 2

    end_ts = _iso_utc_now()
    elapsed_total = time.monotonic() - start_monotonic
    key_artifacts = _build_key_artifacts(paths)
    next_review_files = [
        key_artifacts["report_summary_md"],
        key_artifacts["report_summary_json"],
        key_artifacts["report_pdf"],
        key_artifacts["runner_log"],
    ]
    summary_payload = {
        "image_id": args.image_id,
        "run_tag": args.run_tag,
        "embedding_encoder": selected_encoder,
        "readiness": readiness,
        "started_at_utc": start_ts,
        "ended_at_utc": end_ts,
        "elapsed_seconds": float(elapsed_total),
        "steps": steps,
        "final_status": final_status,
        "key_artifacts": key_artifacts,
        "next_review_files": next_review_files,
    }
    summary_json_path, summary_md_path = _write_run_summary(reports_dir, summary_payload)
    print(f"[runner] wrote summary json: {summary_json_path}")
    print(f"[runner] wrote summary markdown: {summary_md_path}")
    if final_status == "SUCCESS":
        refresh_payload = refresh_history_for_image(args.image_id, outputs_root=args.outputs_root)
        history_root = args.outputs_root / "report_history"
        print(f"[runner] refreshed report history in: {history_root}")
        _ = refresh_payload

    return exit_code


def main() -> int:
    """CLI entrypoint for one-command Stage 1 execution."""
    args = build_parser().parse_args()
    if args.embedding_encoder and args.encoder and args.embedding_encoder != args.encoder:
        raise ValueError("--embedding-encoder and --encoder conflict")
    selected_encoder = args.embedding_encoder or args.encoder
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
