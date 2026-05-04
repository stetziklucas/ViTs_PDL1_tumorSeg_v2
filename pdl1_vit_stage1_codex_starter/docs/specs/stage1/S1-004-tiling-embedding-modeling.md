# S1-004 Tiling, embeddings, tile labels, and modeling

## Goal

Define the implemented Stage 1 single-image modeling path from ROI/tissue to fused outputs:

- tile extraction,
- frozen ViT embeddings,
- tile labels from annotation rasters,
- tile classifier head,
- pixel classifier,
- fused inference.

## Working-space geometry (SVS-aware)

For `.svs`, Stage 1 scripts run in a selected working image level (bounded by `tiling.svs_max_dimension`) rather than always in level-0.

Implications:

- tile manifests, embeddings, labels, and inference maps are produced in working coordinate space,
- exported `tile_prob_map`, `pixel_prob_map`, fused mask, and overlay may be working-space-sized,
- `metrics.json` records geometry metadata (working shape, SVS level/downsample, level-0 dimensions when available, coordinate-space notes).

## Subsystems and artifacts

### A. Tissue mask + tiles

Behavior:

- simple tissue mask heuristic,
- default tile geometry: size 224, stride 112,
- tile filtering by tissue fraction and ROI intersection,
- deterministic manifest + coords + metadata.

Artifacts (examples):

- `outputs/tiles/tile_manifest.csv`
- `outputs/tiles/tile_coords.npy`
- `outputs/tiles/<image_id>_tissue_mask.png`
- `outputs/tiles/tile_manifest_meta.json`

### B. Frozen ViT embeddings

Behavior:

- load frozen encoder,
- embed accepted tiles,
- write cache artifacts including metadata.

Artifacts (examples):

- `outputs/embeddings/embeddings.npy`
- `outputs/embeddings/tile_manifest_with_embeddings_index.csv`
- `outputs/embeddings/embeddings_cache_meta.json`

### C. Tile labels from annotation masks

Behavior:

- consume derived `*_scribble_labels.png` + ROI mask,
- reconcile annotation-mask geometry into tile/working space,
- emit deterministic tile-label CSV with ignore handling.

Artifact:

- `outputs/tiles/tile_labels.csv`

### D. Tile classifier head

Baseline model:

- logistic regression on frozen embeddings.

Artifacts:

- `models/tile_head/tile_head.pkl`
- `models/tile_head/tile_cv_metrics.json`
- `outputs/maps/tile_probabilities.csv`
- `outputs/maps/tile_prob_map.png`

### E. Pixel classifier

Baseline model:

- random forest over RGB/HED/texture + tile prior.

Artifacts:

- `models/pixel_classifier/pixel_model.pkl`
- `models/pixel_classifier/pixel_feature_spec.json`

### F. Fused inference

Fusion rule:

`positive = tissue AND ROI AND (tile_prob >= tile_threshold) AND (pixel_prob >= pixel_threshold)`

Artifacts:

- `outputs/maps/pixel_prob_map.png`
- `outputs/masks/positive_mask.png`
- `outputs/overlays/overlay.png`
- `outputs/reports/metrics.json`
- `outputs/reports/report_summary.md`
- `outputs/reports/report_summary.json`
- `outputs/reports/one_page_report.pdf`

### G. One-command orchestration (readiness-gated)

`scripts/run_stage1_image.py` orchestrates the Stage 1 sequence for one image and run tag:

1. readiness gate (`compute_annotation_readiness`) before any model step,
2. extraction → embedding → tile labels → tile head → pixel classifier → inference → report,
3. live console streaming plus combined log file,
4. concise run summary artifacts.

Runner output artifacts:

- `outputs/reports_<run_tag>/stage1_runner.log`
- `outputs/reports_<run_tag>/stage1_run_summary.json`
- `outputs/reports_<run_tag>/stage1_run_summary.md`
- `outputs/report_history/<image_slug>/history_index.json`
- `outputs/report_history/<image_slug>/latest_vs_previous.json`
- `outputs/report_history/<image_slug>/latest_vs_previous.md`

Exit codes:

- `0` success
- `1` not-ready annotation set (no downstream steps run)
- `2` readiness/config/artifact error
- `3` downstream step failure


### H. Shared project orchestration (multi-image shared training)

`scripts/run_stage1_project.py` orchestrates shared Stage 1 training across a cohort:

1. parse requested cases from repeated `--case alias=image_id` and/or `--cases-file`,
2. run readiness for all requested images first using `compute_annotation_readiness`,
3. require at least 2 READY included cases for shared training,
4. run per-image preprocessing (`extract_tiles.py`, `embed_vit.py`, `make_tile_labels.py`),
5. train one shared tile head (`models/tile_head_<project_tag>_shared`),
6. train one shared pixel classifier (`models/pixel_classifier_<project_tag>_shared`),
7. run per-image inference/report using per-case run tags `<project_tag>__<alias>`,
8. run `make_project_report.py` automatically into `outputs/reports_training_<project_tag>/`.

Shared tile-head mode pools usable rows across images, retains sample-weight behavior, adds image-balance weighting (config: `tile_head.equalize_image_weight`, default `true`), and uses grouped CV by image id when multiple images are present.

Shared pixel-classifier mode pools positive vs negative pixel samples across images with per-image per-class caps/balancing (config: `pixel_classifier.max_samples_per_image_per_class`, `pixel_classifier.equalize_image_class_sampling`).

Per-image report summaries include model-scope metadata (`model_scope`, `shared_model_tag`, `training_image_count`, `included_training_aliases`) without changing metric formulas or canonical artifact filenames.
Project rollup consumes `outputs/reports_<run_tag>/report_summary.json` as authoritative when available, which keeps shared child runs compatible even when per-child `models/tile_head_<run_tag>` directories are not present.

### I. Annotator Stage 1 runner panel (thin GUI layer)

`apps/annotator.py` provides a right-dock **Stage 1 Run Controls** panel for the current image only:

- editable run tag (default `<image_alias>_gui_YYYYMMDD_HHMMSS`),
- run-tag regenerate button,
- non-blocking Stage 1 launch (`QProcess`) that calls `scripts/run_stage1_image.py`,
- run-status and live merged stdout/stderr log,
- in-dock markdown preview for latest report summary and latest-vs-previous comparison.

Run launch remains readiness-gated via shared annotation readiness logic and does not auto-save annotations.

## Directory suffix conventions in real runs

Default config points to unsuffixed paths, but real runs often use suffixed output/model directories per image/run tag, e.g.:

- `outputs/tiles_pf083`, `outputs/embeddings_pf083`
- `outputs/tiles_pf0229`, `outputs/embeddings_pf0229`
- `outputs/maps_pf0229`, `outputs/maps_pf0229_fused`
- `models/tile_head_pf0229`, `models/pixel_classifier_pf0229`
- `outputs/masks_pf0229`, `outputs/overlays_pf0229`, `outputs/reports_pf0229`

This is supported via CLI path flags and should be used consistently so cross-step artifacts stay aligned.

## Non-goals

- end-to-end deep segmentation,
- multi-image benchmark claims,
- hyperparameter-heavy optimization.

## Acceptance criteria

- one annotated image runs end-to-end,
- tile manifest/embeddings/labels are produced reproducibly,
- tile + pixel models train and write model artifacts,
- tile/pixel/fused outputs are exported,
- `metrics.json` includes geometry and output-space metadata.

## Verification

Minimum single-image smoke test:

- run extraction → embedding → labels → tile head → pixel model → inference,
- inspect exported maps/overlay/mask,
- confirm interpretation uses `metrics.json` geometry metadata.

## Risks

- tiny label sets and class imbalance,
- artifact-driven false positives,
- coordinate-space confusion if working-space outputs are misread as level-0.

## Reporting for model development

Single-image report (`scripts/make_report.py`) computes annotated-region development metrics from scribble labels reconciled to working-space outputs with nearest-neighbor geometry-safe reconciliation.

`report_summary.md` and `report_summary.json` also include operator-facing deterministic summary fields derived from FP/FN counts:

- `result_description`
- `evaluation_scope`
- `error_pattern`
- `next_review_focus`

Reports additionally include deterministic class-aware supervision diagnostics:

- `supervision_audit` object in JSON with polygon counts by class, annotated pixel counts by class, accepted/usable/ignored tile counts, tile label/label_reason counts, ignored-tile reasons, selection-source counts (when present), and warnings.
- markdown sections: `Supervision summary`, `Tile supervision summary`, `Class-specific annotated-region metrics`, and `Warnings / review focus`.
- class-specific metrics are split as:
  - `Positive_Tumor`: `annotated_px`, `tp_px`, `fn_px`, `sensitivity`
  - `Negative_Tumor`: `annotated_px`, `tn_px`, `fp_px`, `specificity`
  - `NonTumor`: `annotated_px`, `tn_px`, `fp_px`, `specificity`

Important: Stage 1 remains a binary positive-mask system (`Positive_Tumor` vs combined negatives). Splitting `Negative_Tumor` and `NonTumor` in reports is for supervision diagnostics/review only, not a multiclass model claim.

These fields clarify review direction without changing metric formulas or canonical artifact filenames.

Required human-facing metrics:

- `false_positive_px`
- `false_negative_px`
- `precision`
- `sensitivity`
- `f1`
- `training_log_loss_total`

Machine-readable report JSON may also include:

- `tp_px`, `tn_px`
- `annotated_positive_px`, `annotated_negative_px`, `annotated_total_px`
- `training_log_loss_mean`

Project rollup (`scripts/make_project_report.py`) accepts repeated run tags, skips incomplete tags with explicit reasons, and computes aggregate metrics using micro-average style count summation (sum TP/FP/FN/TN and recompute precision/sensitivity/F1).

Project rollup now also includes:

- aggregate class-specific annotated-region metrics across included run tags,
- per-image rows with class polygon/pixel support and tile supervision counts,
- deterministic `images_needing_attention` entries derived from supervision warnings and class-metric patterns.


Verification review UX: GUI review now renders two aligned layers for annotated-region development review (not whole-slide validation): class-aware annotation labels plus a darker/more-opaque positive prediction mask.
