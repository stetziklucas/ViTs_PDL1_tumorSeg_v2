"""Train the Stage 1 tile classifier head on frozen ViT embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


LABEL_MAP = {"Negative_Context": 0, "Positive_Context": 1}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for tile-head training."""
    parser = argparse.ArgumentParser(description="Train Stage 1 logistic-regression tile head.")
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--labels", type=Path, default=Path("outputs/tiles/tile_labels.csv"), help="Tile labels CSV path.")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("outputs/embeddings"),
        help="Directory containing embeddings.npy and cache metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/tile_head"), help="Tile-head model output directory.")
    parser.add_argument("--maps-dir", type=Path, default=Path("outputs/maps"), help="Directory for tile probability maps.")
    parser.add_argument(
        "--probs-manifest",
        type=Path,
        default=Path("outputs/maps/tile_probabilities.csv"),
        help="Output CSV path for tile probabilities.",
    )
    parser.add_argument(
        "--smoke-image-id",
        default=None,
        help="Optional image_id for tile_prob_map.png; defaults to first image_id in sorted order.",
    )
    parser.add_argument(
        "--cohort-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with shared-training cohort rows: alias,image_id,labels_path,embeddings_dir,maps_dir. "
            "When provided, pooled shared training and per-case scoring are executed."
        ),
    )
    parser.add_argument(
        "--allow-nonpretrained-embeddings",
        action="store_true",
        help="Allow training when embeddings were generated from random-init fallback encoder.",
    )
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


def metric_bundle(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | None]:
    """Compute a compact metrics bundle."""
    y_pred = (prob >= threshold).astype(np.int64)

    out: dict[str, float | None] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": None,
        "roc_auc": None,
    }

    if len(np.unique(y_true)) >= 2:
        out["average_precision"] = float(average_precision_score(y_true, prob))
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    return out


def render_probability_map(image_df: pd.DataFrame, out_path: Path) -> None:
    """Render a deterministic tile-level probability map PNG."""
    width = int((image_df["tile_x"] + image_df["tile_w"]).max())
    height = int((image_df["tile_y"] + image_df["tile_h"]).max())
    canvas = np.zeros((height, width), dtype=np.float32)

    for row in image_df.itertuples(index=False):
        x = int(row.tile_x)
        y = int(row.tile_y)
        w = int(row.tile_w)
        h = int(row.tile_h)
        p = float(row.prob_positive)
        canvas[y : y + h, x : x + w] = np.maximum(canvas[y : y + h, x : x + w], p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(canvas, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Tile positive probability map")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _validate_cache_meta(cache_meta_path: Path, *, allow_nonpretrained_embeddings: bool) -> None:
    with cache_meta_path.open("r", encoding="utf-8") as handle:
        cache_meta = json.load(handle)
    source = str(cache_meta.get("encoder_weight_source", "unknown"))
    if source != "pretrained" and not allow_nonpretrained_embeddings:
        raise RuntimeError(
            "Embeddings were not produced with pretrained frozen ViT weights. "
            "Action: rerun scripts/embed_vit.py in an environment with pretrained timm weights "
            "or pass --allow-nonpretrained-embeddings only for toy/smoke runs."
        )


def _resolve_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cohort_file is None:
        return [
            {
                "alias": None,
                "image_id": None,
                "labels_path": args.labels,
                "embeddings_dir": args.embeddings_dir,
                "maps_dir": args.maps_dir,
                "probs_manifest": args.probs_manifest,
            }
        ]

    if not args.cohort_file.exists():
        raise FileNotFoundError(f"Cohort file not found: {args.cohort_file}")
    cohort_df = pd.read_csv(args.cohort_file)
    required = {"alias", "image_id", "labels_path", "embeddings_dir", "maps_dir"}
    missing = sorted(required - set(cohort_df.columns))
    if missing:
        raise ValueError(f"Cohort file missing required columns: {missing}")

    cases: list[dict[str, Any]] = []
    for row in cohort_df.to_dict(orient="records"):
        probs_manifest = row.get("probs_manifest")
        if probs_manifest is None or str(probs_manifest).strip() == "":
            probs_manifest = str(Path(row["maps_dir"]) / "tile_probabilities.csv")
        cases.append(
            {
                "alias": str(row["alias"]),
                "image_id": str(row["image_id"]),
                "labels_path": Path(str(row["labels_path"])),
                "embeddings_dir": Path(str(row["embeddings_dir"])),
                "maps_dir": Path(str(row["maps_dir"])),
                "probs_manifest": Path(str(probs_manifest)),
            }
        )
    return cases


def _load_case_rows(case: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    labels_path = case["labels_path"]
    embeddings_dir = case["embeddings_dir"]

    embeddings_path = embeddings_dir / "embeddings.npy"
    embedding_manifest_path = embeddings_dir / "tile_manifest_with_embeddings_index.csv"
    cache_meta_path = embeddings_dir / "embeddings_cache_meta.json"
    for required_path in (labels_path, embeddings_path, embedding_manifest_path, cache_meta_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required artifact missing: {required_path}")

    _validate_cache_meta(cache_meta_path, allow_nonpretrained_embeddings=args.allow_nonpretrained_embeddings)

    labels_df = pd.read_csv(labels_path)
    emb_manifest_df = pd.read_csv(embedding_manifest_path)
    embeddings = np.load(embeddings_path)
    if labels_df.empty:
        raise ValueError(f"Tile labels CSV is empty: {labels_path}")

    merged = labels_df.merge(
        emb_manifest_df[["tile_id", "embedding_index"]],
        on="tile_id",
        how="left",
        suffixes=("", "_manifest"),
    )
    if merged["embedding_index_manifest"].notna().any():
        merged["embedding_index"] = merged["embedding_index_manifest"].fillna(merged["embedding_index"])
        merged = merged.drop(columns=["embedding_index_manifest"])
    if merged["embedding_index"].isna().any():
        bad = merged.loc[merged["embedding_index"].isna(), "tile_id"].head(5).tolist()
        raise ValueError(f"Missing embedding_index for tiles: {bad}")

    merged = merged.sort_values(["image_id", "tile_row", "tile_col", "tile_id"]).reset_index(drop=True)
    merged["target"] = merged["label"].map(LABEL_MAP)
    idx = merged["embedding_index"].astype(int).to_numpy()
    if np.any(idx < 0) or np.any(idx >= len(embeddings)):
        raise IndexError("Embedding index out of bounds for embeddings.npy")

    merged["__embedding_vector__"] = [embeddings[i] for i in idx]
    merged["alias"] = case["alias"] if case["alias"] is not None else merged["image_id"].astype(str)
    if case.get("image_id"):
        merged["image_id"] = str(case["image_id"])
    merged["__maps_dir__"] = str(case["maps_dir"])
    merged["__probs_manifest__"] = str(case["probs_manifest"])
    return merged


def _calculate_sample_weights(usable: pd.DataFrame, equalize_image_weight: bool) -> np.ndarray | None:
    base_weight: np.ndarray | None = None
    if "sample_weight" in usable.columns:
        base_weight = usable["sample_weight"].astype(float).to_numpy()

    if not equalize_image_weight:
        return base_weight

    image_counts = usable["alias"].value_counts().to_dict()
    if not image_counts:
        return base_weight
    mean_count = float(np.mean(list(image_counts.values())))
    scale = usable["alias"].map(lambda x: mean_count / float(image_counts[str(x)])).astype(float).to_numpy()
    if base_weight is None:
        return scale
    return base_weight * scale


def _cv_details(x: np.ndarray, y: np.ndarray, groups: np.ndarray, sample_weight: np.ndarray | None, threshold: float) -> tuple[str, list[dict[str, Any]], str | None]:
    unique_images = sorted(np.unique(groups).tolist())
    cv_folds: list[dict[str, Any]] = []
    limitation_note: str | None = None
    cv_mode = "single_image_in_sample"

    if len(unique_images) >= 2:
        cv_mode = "group_kfold_by_image_id"
        n_splits = min(5, len(unique_images))
        splitter = GroupKFold(n_splits=n_splits)
        for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
            y_tr = y[tr_idx]
            y_te = y[te_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                cv_folds.append(
                    {
                        "fold": fold_idx,
                        "skipped": True,
                        "reason": "single_class_in_train_or_test",
                        "train_images": sorted(np.unique(groups[tr_idx]).tolist()),
                        "test_images": sorted(np.unique(groups[te_idx]).tolist()),
                    }
                )
                continue
            fold_model = LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
                random_state=0,
            )
            if sample_weight is not None:
                fold_model.fit(x[tr_idx], y_tr, sample_weight=sample_weight[tr_idx])
            else:
                fold_model.fit(x[tr_idx], y_tr)
            prob = fold_model.predict_proba(x[te_idx])[:, 1]
            fold_metrics = metric_bundle(y_te, prob, threshold=threshold)
            cv_folds.append(
                {
                    "fold": fold_idx,
                    "skipped": False,
                    "train_images": sorted(np.unique(groups[tr_idx]).tolist()),
                    "test_images": sorted(np.unique(groups[te_idx]).tolist()),
                    **fold_metrics,
                }
            )
    else:
        limitation_note = (
            "Only one image_id available; cross-image CV is not possible. "
            "Reported metrics are tile-level in-sample smoke metrics only."
        )
    return cv_mode, cv_folds, limitation_note


def main() -> None:
    """Run Stage 1 tile-head training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    threshold = float(cfg.get("tile_head", {}).get("positive_threshold", 0.60))
    equalize_image_weight = bool(cfg.get("tile_head", {}).get("equalize_image_weight", True))

    cases = _resolve_cases(args)
    merged_all = pd.concat([_load_case_rows(case, args) for case in cases], ignore_index=True)

    usable = merged_all.dropna(subset=["target"]).copy()
    usable_classes = sorted(int(v) for v in usable["target"].unique())
    if len(usable_classes) < 2:
        class_names = sorted(usable["label"].dropna().unique().tolist())
        raise RuntimeError(
            "Fewer than 2 usable classes remain after tile-label generation; refusing to train. "
            f"usable_labels={class_names}. Action: add scribbles for both Positive_Tumor and "
            "Negative_Tumor/NonTumor, then rerun scripts/make_tile_labels.py."
        )

    x = np.stack(usable["__embedding_vector__"].to_numpy(), axis=0).astype(np.float32)
    y = usable["target"].astype(int).to_numpy()
    groups = usable["image_id"].astype(str).to_numpy()
    sample_weight = _calculate_sample_weights(usable, equalize_image_weight=equalize_image_weight)

    cv_mode, cv_folds, limitation_note = _cv_details(x, y, groups, sample_weight, threshold)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=0,
    )
    if sample_weight is not None:
        model.fit(x, y, sample_weight=sample_weight)
    else:
        model.fit(x, y)

    merged_all["prob_positive"] = model.predict_proba(np.stack(merged_all["__embedding_vector__"].to_numpy(), axis=0))[:, 1]
    usable["prob_positive"] = model.predict_proba(x)[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "tile_head.pkl"
    metrics_path = args.output_dir / "tile_cv_metrics.json"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    per_image_usable = (
        usable.assign(target_name=usable["target"].map({0: "Negative_Context", 1: "Positive_Context"}))
        .groupby(["alias", "image_id", "target_name"], as_index=False)
        .size()
    )
    per_image_usable_payload: dict[str, dict[str, int]] = {}
    for row in per_image_usable.itertuples(index=False):
        key = f"{row.alias}|{row.image_id}"
        per_image_usable_payload.setdefault(key, {"Negative_Context": 0, "Positive_Context": 0})
        per_image_usable_payload[key][str(row.target_name)] = int(row.size)

    included_aliases = sorted(str(v) for v in merged_all["alias"].astype(str).unique().tolist())
    included_image_ids = sorted(str(v) for v in merged_all["image_id"].astype(str).unique().tolist())

    for key, image_df in merged_all.groupby(["alias", "image_id"], sort=True):
        alias, image_id = str(key[0]), str(key[1])
        case_rows = image_df.sort_values(["tile_row", "tile_col", "tile_id"]).reset_index(drop=True).copy()
        case_maps_dir = Path(str(case_rows["__maps_dir__"].iloc[0]))
        case_probs_manifest = Path(str(case_rows["__probs_manifest__"].iloc[0]))
        case_maps_dir.mkdir(parents=True, exist_ok=True)
        case_probs_manifest.parent.mkdir(parents=True, exist_ok=True)

        export_df = case_rows.drop(columns=["__embedding_vector__", "__maps_dir__", "__probs_manifest__"])
        export_df.to_csv(case_probs_manifest, index=False)
        render_probability_map(export_df, case_maps_dir / "tile_prob_map.png")
        logging.info("Scored alias=%s image_id=%s -> %s", alias, image_id, case_probs_manifest)

    training_metrics = metric_bundle(y, usable["prob_positive"].to_numpy(dtype=float), threshold=threshold)

    summary = {
        "model_type": "logistic_regression",
        "threshold": threshold,
        "n_total_tiles": int(len(merged_all)),
        "n_usable_tiles": int(len(usable)),
        "n_scored_tiles": int(len(merged_all)),
        "n_images": int(len(included_image_ids)),
        "image_ids": included_image_ids,
        "aliases": included_aliases,
        "alias_to_image_id": {
            str(alias): sorted(set(df["image_id"].astype(str).tolist()))
            for alias, df in merged_all.groupby("alias", sort=True)
        },
        "equalize_image_weight": bool(equalize_image_weight),
        "cv_mode": cv_mode,
        "single_image_limitation": limitation_note,
        "cross_image_cv": cv_folds,
        "class_counts": {
            "Negative_Context": int((usable["target"] == 0).sum()),
            "Positive_Context": int((usable["target"] == 1).sum()),
            "Ignore": int((merged_all["label"] == "Ignore").sum()),
        },
        "per_image_usable_class_counts": per_image_usable_payload,
        "train_all_metrics": training_metrics,
        "artifacts": {
            "model_path": str(model_path.as_posix()),
            "metrics_path": str(metrics_path.as_posix()),
        },
    }

    if args.cohort_file is None:
        summary["artifacts"]["probability_manifest_path"] = str(args.probs_manifest.as_posix())
        summary["artifacts"]["probability_map_path"] = str((args.maps_dir / "tile_prob_map.png").as_posix())

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logging.info("Wrote model artifact: %s", model_path)
    logging.info("Wrote metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
