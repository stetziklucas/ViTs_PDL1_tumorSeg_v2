"""Run Stage 1 shared project training across multiple READY images."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from annotation_readiness import AnnotationReadinessResult, compute_annotation_readiness
from report_history import refresh_history_for_image
from project_case_discovery import discover_project_cases
from scripts.run_stage1_image import derive_paths, load_config

_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run shared Stage 1 project training across READY image cohort.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--project-tag", required=True, help="Project tag used for shared model and child run tags.")
    parser.add_argument("--case", action="append", default=[], help="Case mapping alias=image_id (repeatable).")
    parser.add_argument("--cases-file", type=Path, default=None, help="Optional file containing alias=image_id rows.")
    parser.add_argument("--discover-ready-cases", action="store_true", help="Auto-discover READY cases from annotations/raw dirs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing source images.")
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/annotations"), help="Annotation root directory.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Outputs root directory.")
    parser.add_argument("--models-root", type=Path, default=Path("models"), help="Models root directory.")
    parser.add_argument("--embedding-encoder", type=str, default=None, help="Embedding encoder override.")
    parser.add_argument("--encoder", type=str, default=None, help="Alias for --embedding-encoder.")
    parser.add_argument(
        "--allow-skip-not-ready",
        action="store_true",
        help="Skip non-READY cases and continue with READY subset.",
    )
    return parser


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_case_entry(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Malformed case '{value}': expected alias=image_id")
    alias, image_id = value.split("=", 1)
    alias = alias.strip()
    image_id = image_id.strip()
    if not alias or not image_id:
        raise ValueError(f"Malformed case '{value}': alias and image_id must be non-empty")
    if not _ALIAS_PATTERN.match(alias):
        raise ValueError(f"Malformed alias '{alias}': expected filesystem-safe pattern [A-Za-z0-9._-]+")
    return alias, image_id


def collect_cases(case_args: list[str], cases_file: Path | None) -> list[dict[str, str]]:
    rows: list[str] = list(case_args)
    if cases_file is not None:
        for line in cases_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            rows.append(text)

    parsed: list[dict[str, str]] = []
    seen_aliases: set[str] = set()
    for raw in rows:
        alias, image_id = _parse_case_entry(raw)
        if alias in seen_aliases:
            raise ValueError(f"Duplicate alias found: {alias}")
        seen_aliases.add(alias)
        parsed.append({"alias": alias, "image_id": image_id})

    if not parsed:
        raise ValueError("No cases supplied. Use --case or --cases-file.")
    return parsed



def write_project_cases_manifest(report_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = report_dir / "stage1_project_cases.json"
    md_path = report_dir / "stage1_project_cases.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Stage 1 Project Cases", "", "## Included READY cases"]
    for case in payload.get("included_ready_cases", []):
        lines.append(f"- {case['alias']}={case['image_id']} (child_run_tag={case.get('child_run_tag','n/a')})")
    if not payload.get("included_ready_cases"):
        lines.append("- none")
    lines.extend(["", "## Skipped cases"])
    for case in payload.get("skipped_cases", []):
        lines.append(f"- {case['alias']}={case['image_id']}: {case.get('reason')} {case.get('readiness_status_code','')}")
    if not payload.get("skipped_cases"):
        lines.append("- none")
    lines.extend(["", "## Counts", ""])
    counts = payload.get("counts", {})
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return json_path, md_path

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


def _readiness_to_dict(result: AnnotationReadinessResult) -> dict[str, Any]:
    payload = result.to_dict()
    return {
        "status_code": payload["status_code"],
        "status_label": payload["status_label"],
        "summary_message": payload["summary_message"],
        "next_action": payload["next_action"],
    }


def _run_step(step_name: str, command: list[str], log_handle: Any, steps: list[dict[str, Any]]) -> int:
    started = time.monotonic()
    command_text = shlex.join(command)
    print(f"\n[step] {step_name}: {command_text}")
    log_handle.write(f"[step] {step_name}: {command_text}\n")
    rc = _run_subprocess_streaming(command, log_handle)
    steps.append(
        {
            "step": step_name,
            "command": command_text,
            "exit_code": int(rc),
            "elapsed_seconds": float(time.monotonic() - started),
        }
    )
    return rc


def _write_summary_json_md(report_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = report_dir / "stage1_project_run_summary.json"
    md_path = report_dir / "stage1_project_run_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Stage 1 Project Run Summary",
        "",
        f"project_tag: {payload['project_tag']}",
        f"final_status: {payload['final_status']}",
        f"started_at_utc: {payload['started_at_utc']}",
        f"ended_at_utc: {payload['ended_at_utc']}",
        f"elapsed_seconds: {payload['elapsed_seconds']:.2f}",
        "",
        "## Requested cases",
    ]
    lines.extend([f"- {c['alias']}={c['image_id']}" for c in payload["requested_cases"]] or ["- none"])
    lines.extend(["", "## Included READY cases"])
    lines.extend([f"- {c['alias']}={c['image_id']}" for c in payload["included_cases"]] or ["- none"])
    lines.extend(["", "## Skipped cases"])
    if payload["skipped_cases"]:
        for row in payload["skipped_cases"]:
            lines.append(f"- {row['alias']}={row['image_id']}: {row['reason']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Shared model artifacts"])
    for key, path in payload["shared_model_artifacts"].items():
        lines.append(f"- {key}: `{path}`")

    lines.extend(["", "## Step outcomes", "", "| step | exit_code | elapsed_seconds |", "| --- | ---: | ---: |"])
    for step in payload["steps"]:
        lines.append(f"| {step['step']} | {step['exit_code']} | {step['elapsed_seconds']:.2f} |")

    lines.extend(["", "## Next review files"])
    lines.extend([f"- `{p}`" for p in payload["next_review_files"]])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = build_parser().parse_args()
    if args.embedding_encoder and args.encoder and args.embedding_encoder != args.encoder:
        raise ValueError("--embedding-encoder and --encoder conflict")
    selected_encoder = args.embedding_encoder or args.encoder
    _ = load_config(args.config)
    start_ts = _iso_utc_now()
    print(f"Embedding encoder: {selected_encoder or 'config default'}. Use a distinct project tag for side-by-side comparisons.")
    start_t = time.monotonic()

    requested_cases: list[dict[str, str]] = []
    discovery_summary: dict[str, Any] | None = None
    if args.case or args.cases_file is not None:
        requested_cases = collect_cases(args.case, args.cases_file)
    elif args.discover_ready_cases:
        discovery_summary = discover_project_cases(config=load_config(args.config), annotations_dir=args.annotations_dir, raw_dir=args.raw_dir)
        requested_cases = list(discovery_summary.get("included_ready_cases", []))
    else:
        raise ValueError("No cases supplied. Use --case/--cases-file or --discover-ready-cases.")
    reports_training_dir = args.outputs_root / f"reports_training_{args.project_tag}"
    reports_training_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_training_dir / "stage1_project_runner.log"

    steps: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, str]] = []
    included_cases: list[dict[str, str]] = []
    final_status = "UNKNOWN"

    python = sys.executable

    with log_path.open("w", encoding="utf-8") as log_handle:
        for case in requested_cases:
            result = compute_annotation_readiness(config=load_config(args.config), annotations_dir=args.annotations_dir, image_id=case["image_id"])
            readiness = _readiness_to_dict(result)
            readiness_rows.append({**case, **readiness})

        not_ready = [row for row in readiness_rows if row["status_code"] != "READY"]
        for row in readiness_rows:
            print(f"[readiness] {row['alias']}={row['image_id']} -> {row['status_code']} ({row['status_label']})")
            log_handle.write(json.dumps({"readiness": row}, sort_keys=True) + "\n")

        if not_ready and not args.allow_skip_not_ready:
            final_status = "READINESS_BLOCKED"
        else:
            for row in readiness_rows:
                if row["status_code"] == "READY":
                    included_cases.append({"alias": row["alias"], "image_id": row["image_id"]})
                else:
                    skipped_cases.append(
                        {
                            "alias": row["alias"],
                            "image_id": row["image_id"],
                            "reason": f"{row['status_code']}: {row['summary_message']}",
                        }
                    )

            if len(included_cases) < 2:
                final_status = "INSUFFICIENT_READY_CASES"
            else:
                preprocessing_failed = False
                for case in included_cases:
                    run_tag = f"{args.project_tag}__{case['alias']}"
                    paths = derive_paths(run_tag=run_tag, outputs_root=args.outputs_root, models_root=args.models_root)
                    case["run_tag"] = run_tag
                    case["paths"] = {k: v.as_posix() for k, v in paths.items()}
                    commands = [
                        (
                            f"{case['alias']}:extract_tiles",
                            [
                                python,
                                "scripts/extract_tiles.py",
                                "--config",
                                str(args.config),
                                "--image-id",
                                case["image_id"],
                                "--input",
                                str(args.raw_dir),
                                "--annotations-dir",
                                str(args.annotations_dir),
                                "--output-dir",
                                str(paths["tiles_dir"]),
                            ],
                        ),
                        (
                            f"{case['alias']}:embed_vit",
                            ([
                                python,
                                "scripts/embed_vit.py",
                                "--config",
                                str(args.config),
                                "--image-id",
                                case["image_id"],
                                "--input",
                                str(paths["tiles_dir"]),
                                "--raw-dir",
                                str(args.raw_dir),
                                "--output-dir",
                                str(paths["embeddings_dir"]),
                            ] + (["--embedding-encoder", selected_encoder] if selected_encoder else [])),
                        ),
                        (
                            f"{case['alias']}:make_tile_labels",
                            [
                                python,
                                "scripts/make_tile_labels.py",
                                "--config",
                                str(args.config),
                                "--image-id",
                                case["image_id"],
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
                    ]
                    for step_name, command in commands:
                        rc = _run_step(step_name, command, log_handle, steps)
                        if rc != 0:
                            preprocessing_failed = True
                            break
                    if preprocessing_failed:
                        break

                if preprocessing_failed:
                    final_status = "PREPROCESSING_FAILED"
                else:
                    tile_model_dir = args.models_root / f"tile_head_{args.project_tag}_shared"
                    pixel_model_dir = args.models_root / f"pixel_classifier_{args.project_tag}_shared"

                    cohort_tile_csv = reports_training_dir / "tile_training_cases.csv"
                    cohort_pixel_csv = reports_training_dir / "pixel_training_cases.csv"
                    tile_rows = []
                    pixel_rows = []
                    run_tags = []
                    aliases = []
                    for case in included_cases:
                        run_tag = case["run_tag"]
                        aliases.append(case["alias"])
                        run_tags.append(run_tag)
                        paths = derive_paths(run_tag=run_tag, outputs_root=args.outputs_root, models_root=args.models_root)
                        tile_rows.append(
                            {
                                "alias": case["alias"],
                                "image_id": case["image_id"],
                                "labels_path": (paths["tiles_dir"] / "tile_labels.csv").as_posix(),
                                "embeddings_dir": paths["embeddings_dir"].as_posix(),
                                "maps_dir": paths["tile_maps_dir"].as_posix(),
                                "probs_manifest": (paths["tile_maps_dir"] / "tile_probabilities.csv").as_posix(),
                            }
                        )
                        pixel_rows.append(
                            {
                                "alias": case["alias"],
                                "image_id": case["image_id"],
                                "tiles_dir": paths["tiles_dir"].as_posix(),
                                "tile_probabilities": (paths["tile_maps_dir"] / "tile_probabilities.csv").as_posix(),
                            }
                        )

                    pd.DataFrame(tile_rows).to_csv(cohort_tile_csv, index=False)
                    pd.DataFrame(pixel_rows).to_csv(cohort_pixel_csv, index=False)

                    if _run_step(
                        "shared_train_tile_head",
                        [
                            python,
                            "scripts/train_tile_head.py",
                            "--config",
                            str(args.config),
                            "--cohort-file",
                            str(cohort_tile_csv),
                            "--output-dir",
                            str(tile_model_dir),
                        ],
                        log_handle,
                        steps,
                    ) != 0:
                        final_status = "SHARED_TRAINING_FAILED"
                    elif _run_step(
                        "shared_train_pixel_classifier",
                        [
                            python,
                            "scripts/train_pixel_classifier.py",
                            "--config",
                            str(args.config),
                            "--cohort-file",
                            str(cohort_pixel_csv),
                            "--raw-dir",
                            str(args.raw_dir),
                            "--annotations-dir",
                            str(args.annotations_dir),
                            "--output-dir",
                            str(pixel_model_dir),
                        ],
                        log_handle,
                        steps,
                    ) != 0:
                        final_status = "SHARED_TRAINING_FAILED"
                    else:
                        downstream_failed = False
                        for case in included_cases:
                            paths = derive_paths(run_tag=case["run_tag"], outputs_root=args.outputs_root, models_root=args.models_root)
                            if _run_step(
                                f"{case['alias']}:run_inference",
                                [
                                    python,
                                    "scripts/run_inference.py",
                                    "--config",
                                    str(args.config),
                                    "--image-id",
                                    case["image_id"],
                                    "--raw-dir",
                                    str(args.raw_dir),
                                    "--annotations-dir",
                                    str(args.annotations_dir),
                                    "--tiles-dir",
                                    str(paths["tiles_dir"]),
                                    "--tile-probabilities",
                                    str(paths["tile_maps_dir"] / "tile_probabilities.csv"),
                                    "--pixel-model",
                                    str(pixel_model_dir / "pixel_model.pkl"),
                                    "--pixel-feature-spec",
                                    str(pixel_model_dir / "pixel_feature_spec.json"),
                                    "--maps-dir",
                                    str(paths["fused_maps_dir"]),
                                    "--masks-dir",
                                    str(paths["masks_dir"]),
                                    "--overlays-dir",
                                    str(paths["overlays_dir"]),
                                    "--reports-dir",
                                    str(paths["reports_dir"]),
                                ],
                                log_handle,
                                steps,
                            ) != 0:
                                downstream_failed = True
                                break
                            if _run_step(
                                f"{case['alias']}:make_report",
                                [
                                    python,
                                    "scripts/make_report.py",
                                    "--config",
                                    str(args.config),
                                    "--image-id",
                                    case["image_id"],
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
                                    str(tile_model_dir),
                                    "--model-scope",
                                    "shared_project_model",
                                    "--shared-model-tag",
                                    args.project_tag,
                                    "--training-image-count",
                                    str(len(included_cases)),
                                    "--included-training-aliases",
                                    ",".join(aliases),
                                ],
                                log_handle,
                                steps,
                            ) != 0:
                                downstream_failed = True
                                break

                        if downstream_failed:
                            final_status = "DOWNSTREAM_FAILED"
                        else:
                            project_report_rc = _run_step(
                                "make_project_report",
                                [
                                    python,
                                    "scripts/make_project_report.py",
                                    "--config",
                                    str(args.config),
                                    "--annotations-dir",
                                    str(args.annotations_dir),
                                    "--output-dir",
                                    str(reports_training_dir),
                                    *sum([["--run-tag", tag] for tag in run_tags], []),
                                ],
                                log_handle,
                                steps,
                            )
                            final_status = "SUCCESS" if project_report_rc == 0 else "DOWNSTREAM_FAILED"
                            if final_status == "SUCCESS":
                                for case in included_cases:
                                    refresh_history_for_image(case["image_id"], outputs_root=args.outputs_root)

    if final_status == "UNKNOWN":
        final_status = "READINESS_BLOCKED"

    if final_status in {"READINESS_BLOCKED", "INSUFFICIENT_READY_CASES"}:
        exit_code = 1
    elif final_status in {"PREPROCESSING_FAILED", "SHARED_TRAINING_FAILED"}:
        exit_code = 2
    elif final_status == "DOWNSTREAM_FAILED":
        exit_code = 3
    else:
        exit_code = 0

    shared_model_artifacts = {
        "tile_model": (args.models_root / f"tile_head_{args.project_tag}_shared" / "tile_head.pkl").as_posix(),
        "tile_metrics": (args.models_root / f"tile_head_{args.project_tag}_shared" / "tile_cv_metrics.json").as_posix(),
        "pixel_model": (args.models_root / f"pixel_classifier_{args.project_tag}_shared" / "pixel_model.pkl").as_posix(),
        "pixel_feature_spec": (args.models_root / f"pixel_classifier_{args.project_tag}_shared" / "pixel_feature_spec.json").as_posix(),
    }
    included_with_tags = []
    for case in included_cases:
        row = dict(case)
        row["child_run_tag"] = row.get("run_tag")
        included_with_tags.append(row)

    cases_manifest_payload = discovery_summary or {}
    cases_manifest_payload.update({
        "project_tag": args.project_tag,
        "embedding_encoder": selected_encoder,
        "requested_cases": requested_cases,
        "included_ready_cases": included_with_tags,
        "skipped_cases": skipped_cases if skipped_cases else cases_manifest_payload.get("skipped_cases", []),
        "alias_to_image_id": {c["alias"]: c["image_id"] for c in included_cases},
        "counts": {
            "requested_count": len(requested_cases),
            "included_ready_count": len(included_cases),
            "skipped_count": len(skipped_cases),
            "readiness_counts": {row["status_code"]: sum(1 for r in readiness_rows if r["status_code"] == row["status_code"]) for row in readiness_rows},
        },
    })
    write_project_cases_manifest(reports_training_dir, cases_manifest_payload)

    payload = {
        "project_tag": args.project_tag,
        "embedding_encoder": selected_encoder,
        "requested_cases": requested_cases,
        "readiness": readiness_rows,
        "included_cases": included_cases,
        "skipped_cases": skipped_cases,
        "shared_model_artifacts": shared_model_artifacts,
        "steps": steps,
        "started_at_utc": start_ts,
        "ended_at_utc": _iso_utc_now(),
        "elapsed_seconds": float(time.monotonic() - start_t),
        "final_status": final_status,
        "next_review_files": [
            (reports_training_dir / "training_summary.md").as_posix(),
            (reports_training_dir / "training_summary.json").as_posix(),
            (reports_training_dir / "stage1_project_run_summary.md").as_posix(),
            (reports_training_dir / "stage1_project_run_summary.json").as_posix(),
            log_path.as_posix(),
        ],
    }
    _write_summary_json_md(reports_training_dir, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
