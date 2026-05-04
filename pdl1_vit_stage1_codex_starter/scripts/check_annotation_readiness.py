"""Lightweight CLI preflight for Stage 1 annotation readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation_readiness import compute_annotation_readiness


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for annotation readiness check."""
    parser = argparse.ArgumentParser(description="Check Stage 1 annotation readiness for one image id.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--image-id", required=True, help="Image identifier to check.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Directory containing canonical Stage 1 annotation artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")
    return parser


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file from disk."""
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config did not parse into a dictionary.")
    return config


def _human_readable_output(payload: dict[str, Any]) -> str:
    exists = payload["artifact_exists"]
    lines = [
        f"image_id: {payload['image_id']}",
        f"status: {payload['status_code']} ({payload['status_label']})",
        f"summary: {payload['summary_message']}",
        f"next_action: {payload['next_action']}",
        (
            "artifacts: "
            f"meta={'yes' if exists['annotation_meta'] else 'no'}, "
            f"roi={'yes' if exists['roi_mask'] else 'no'}, "
            f"scribble={'yes' if exists['scribble_labels'] else 'no'}"
        ),
        f"roi_positive_pixels: {payload['roi_positive_pixels']}",
        f"polygon_counts: {payload['polygon_counts']}",
        f"pixel_counts: {payload['pixel_counts']}",
    ]
    for note in payload["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main() -> int:
    """Run annotation readiness check and return status exit code."""
    args = build_parser().parse_args()
    config = load_config(args.config)
    result = compute_annotation_readiness(config=config, annotations_dir=args.annotations_dir, image_id=args.image_id)
    payload = result.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_human_readable_output(payload))

    if result.status_code == "READY":
        return 0
    if result.status_code == "ERROR":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
