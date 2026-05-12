"""Compare aggregate metrics and encoder provenance between two training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = [
    "false_positive_px",
    "false_negative_px",
    "precision",
    "sensitivity",
    "f1",
    "training_log_loss_total",
    "annotated_positive_px",
    "annotated_negative_px",
    "annotated_total_px",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _field(provenance: dict[str, Any] | None, key: str, default: str = "n/a") -> str:
    if not isinstance(provenance, dict):
        return default
    value = provenance.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _display_name(provenance: dict[str, Any] | None) -> str:
    if not isinstance(provenance, dict):
        return "not recorded"
    return (
        str(provenance.get("encoder_display_name") or "")
        or str(provenance.get("encoder_id") or "")
        or "not recorded"
    )


def _append_encoder_block(lines: list[str], title: str, provenance: dict[str, Any] | None) -> None:
    name = _display_name(provenance)
    encoder_id = _field(provenance, "encoder_id", "n/a")
    backend = _field(provenance, "encoder_backend", "n/a")
    model = _field(provenance, "encoder_model_name", "n/a")
    pooling = _field(provenance, "encoder_pooling", "n/a")
    dim = _field(provenance, "embedding_dim", "n/a")
    lines.extend(
        [
            f"## {title}",
            "",
            f"- display_name: {name}",
            f"- encoder_id: {encoder_id}",
            f"- backend: {backend}",
            f"- model_name: {model}",
            f"- pooling: {pooling}",
            f"- embedding_dim: {dim}",
            "",
        ]
    )


def _resolve_encoder_provenance(summary: dict[str, Any]) -> dict[str, Any] | None:
    direct = summary.get("encoder_provenance")
    if isinstance(direct, dict):
        return direct
    per_run = summary.get("per_run_encoder_provenance")
    if isinstance(per_run, dict):
        first = next(iter(per_run.values()), None)
        if isinstance(first, dict):
            return first
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tag", required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load(args.outputs_root / f"reports_training_{args.baseline_tag}" / "training_summary.json")
    candidate = _load(args.outputs_root / f"reports_training_{args.candidate_tag}" / "training_summary.json")
    baseline_agg = baseline.get("aggregate_metrics", {})
    candidate_agg = candidate.get("aggregate_metrics", {})
    side_by_side = {
        metric: {
            "baseline": baseline_agg.get(metric),
            "candidate": candidate_agg.get(metric),
            "candidate_minus_baseline": (
                candidate_agg.get(metric) - baseline_agg.get(metric)
                if isinstance(baseline_agg.get(metric), (int, float))
                and isinstance(candidate_agg.get(metric), (int, float))
                else None
            ),
        }
        for metric in METRICS
    }

    baseline_encoder_provenance = _resolve_encoder_provenance(baseline)
    candidate_encoder_provenance = _resolve_encoder_provenance(candidate)
    output_payload = {
        "baseline_tag": args.baseline_tag,
        "candidate_tag": args.candidate_tag,
        "baseline_encoder_provenance": baseline_encoder_provenance,
        "candidate_encoder_provenance": candidate_encoder_provenance,
        "aggregate_metrics": side_by_side,
        "caveat": "Annotated-region development metrics only; not whole-slide validation and not clinical performance.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "encoder_comparison_summary.json").write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Encoder Comparison Summary",
        "",
        f"baseline_tag: {args.baseline_tag}",
        f"candidate_tag: {args.candidate_tag}",
        "",
        output_payload["caveat"],
        "",
    ]
    _append_encoder_block(lines, "Baseline encoder", baseline_encoder_provenance)
    _append_encoder_block(lines, "Candidate encoder", candidate_encoder_provenance)
    lines.extend(["| metric | baseline | candidate | delta |", "| --- | ---: | ---: | ---: |"])
    for metric, values in side_by_side.items():
        lines.append(
            f"| {metric} | {values['baseline']} | {values['candidate']} | {values['candidate_minus_baseline']} |"
        )
    (args.output_dir / "encoder_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
