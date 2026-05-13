# PD-L1 ViT Stage 1 (SVS-aware single-image pipeline)

This directory is the active Stage 1 project root for the PD-L1 IHC proof-of-concept pipeline.

> Status: Stage 1 is implemented for real single-image runs (including `.svs` input in the modeling/inference stack), with explicit limitations documented below.

## Stage 1 purpose

Provide a reproducible research pipeline from:

1. manifest + QC,
2. polygon annotation,
3. tile extraction,
4. frozen ViT embeddings,
5. tile labels,
6. tile head training,
7. pixel classifier training,
8. fused inference outputs.

This repo is for research/screening enrichment prototyping only, not diagnostic use.

## Current architecture and what is implemented

Current main supports:

- classed polygon/lasso annotation in `apps/annotator.py`, with polygons as source of truth,
- derived exports on save: `roi_mask.png` + `scribble_labels.png` + metadata JSON,
- persistent non-ephemeral save/readiness feedback in annotator dock after pressing `s`,
- SVS-aware image loading for Stage 1 modeling/inference scripts using a selected working level,
- real tile manifests, embeddings, tile labels, tile head outputs,
- pixel classifier training and fused inference outputs (maps, masks, overlays, metrics).

In short: Stage 1 is no longer placeholder-only; most core scripts now run real artifact-producing workflows.

## Embedding encoder registry (PR1)

Stage 1 embedding generation now resolves a configurable encoder registry via `embedding_encoder` in `config/base.yaml`.

- PR1 supports only `current_timm` (the existing timm baseline behavior).
- Future backends (for example Hibou) should be added as new registry entries/adapters.
- Embedding artifact filenames and layout are unchanged: `embeddings.npy`, `tile_manifest_with_embeddings_index.csv`, `embeddings_cache_meta.json`.
- Encoder provenance is recorded in `embeddings_cache_meta.json` (for example `encoder_id`, backend/model, dtype/dim, and normalized encoder config).

## Working image space for `.svs`

For `.svs` in tile extraction / embedding / tile-label generation / pixel training / inference:

- scripts select a working SVS level using `tiling.svs_max_dimension`,
- geometry is tracked in metadata (`svs_level`, `svs_level_downsample`, level dimensions, and coordinate-space fields),
- exported maps/masks/overlays are in the selected working image space (not automatically full level-0),
- `outputs/.../metrics.json` records the geometry/space metadata needed to interpret outputs.

## Artifact directory conventions

Defaults in `config/base.yaml` point to unsuffixed paths (for example `outputs/tiles`, `outputs/embeddings`, `models/tile_head`).

In practice, per-image/per-run separation is commonly done by passing suffixed output dirs, for example:

- `outputs/tiles_pf083`, `outputs/embeddings_pf083`,
- `outputs/tiles_pf0229`, `outputs/embeddings_pf0229`,
- `outputs/maps_pf0229`, `outputs/maps_pf0229_fused`,
- `models/tile_head_pf0229`, `models/pixel_classifier_pf0229`,
- `outputs/masks_pf0229`, `outputs/overlays_pf0229`, `outputs/reports_pf0229`.

Use CLI `--output-dir`, `--maps-dir`, `--model-dir`, etc. flags consistently so linked artifacts stay aligned.

## Annotation semantics (current contract)

Stage 1 annotation contract is:

- classed polygons saved in `data/annotations/{image_id}_annotation_meta.json` are canonical,
- `data/annotations/roi_masks/{image_id}_roi_mask.png` is derived from tumor-class polygon union,
- `data/annotations/scribbles/{image_id}_scribble_labels.png` is a derived rasterized class-label map,
- downstream scripts consume derived mask artifacts and reconcile geometry into tile/working space when needed.

Annotator save UX now includes persistent status labels in the right dock:

- **Save status**
- **Readiness**
- **Next action**

Recommended pathologist workflow:

1. Draw/update polygons.
2. Press `s`.
3. Wait for explicit saved + readiness message in the dock/status bar.
4. Continue only after readiness guidance is shown.

`s` does **not** auto-close the annotator.

Annotator now also includes a **Stage 1 Workflow** dock for the current image:

- editable `Run tag` with `Generate new run tag`,
- non-blocking `Run Stage 1` launcher,
- history browser controls:
  - `History` dropdown,
  - `Refresh history`,
  - `Load selected report`,
  - `Jump to newest`,
  - `Jump to oldest`,
- explicit selected-run details (`run tag`, `timestamp`, `model scope`, shared-model provenance when present),
- `Show runner log` toggle (hide log to give report preview more space),
- live log output (when shown) and markdown preview in-dock.

Recommended GUI loop is: **save (`s`) -> run -> choose report history entry -> preview selected markdown summary**.
Latest-vs-previous artifacts are still generated on disk for compatibility, but they are no longer the primary in-panel review path.

## Annotation readiness CLI (SSH-friendly preflight)

Use this lightweight checker to confirm annotation readiness without running model steps:

```bash
python scripts/check_annotation_readiness.py --config config/base.yaml --image-id <image_id> --annotations-dir data/annotations
```

Optional JSON output:

```bash
python scripts/check_annotation_readiness.py --config config/base.yaml --image-id <image_id> --annotations-dir data/annotations --json
```

Exit codes:

- `0`: `READY`
- `1`: needs more annotation (`NEEDS_POSITIVE`, `NEEDS_NEGATIVE`, `NO_TUMOR_ROI`, or `NO_USABLE_SUPERVISION`)
- `2`: `ERROR` (missing or broken artifacts)

## Known limitations (explicit)

- `scripts/run_qc.py` now discovers `.svs` and records manifest/QC metadata rows, but SVS QC remains metadata-only (file presence/path/size/mtime plus dimensions when OpenSlide is available), not full blur/pen/fold parity with raster-image QC.
- Stage 1 workflow is effectively single-image oriented for modeling/training/inference runs.
- Output geometry is working-space-first for `.svs`; if level-0 deliverables are required, an explicit projection/export step is still needed.
- `scripts/make_report.py` now generates a single-image report set (`report_summary.md`, `report_summary.json`, `one_page_report.pdf`) with annotated-region development metrics plus class-aware supervision diagnostics.
- `scripts/make_project_report.py` generates project rollups (`training_summary.md`, `training_summary.json`) across run tags, includes class-aware supervision/audit summaries, and skips incomplete tags with recorded reasons.
- Project rollup now prefers each run's `report_summary.json` when present, which keeps shared-project child runs (`model_scope=shared_project_model`) compatible even when per-child `models/tile_head_<run_tag>` is absent.

## Important path note

All example commands below assume you are inside this directory:

```bash
cd pdl1_vit_stage1_codex_starter
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical Stage 1 run sequence (single image)

```bash
python scripts/run_qc.py --config config/base.yaml --input data/raw --force
python apps/annotator.py --config config/base.yaml --image-id <image_id>
python scripts/extract_tiles.py --config config/base.yaml --image-id <image_id> --output-dir outputs/tiles_<tag>
python scripts/embed_vit.py --config config/base.yaml --image-id <image_id> --input outputs/tiles_<tag> --output-dir outputs/embeddings_<tag>
python scripts/make_tile_labels.py --config config/base.yaml --image-id <image_id> --tiles-dir outputs/tiles_<tag> --embeddings-dir outputs/embeddings_<tag> --output-dir outputs/tiles_<tag>
python scripts/train_tile_head.py --config config/base.yaml --image-id <image_id> --labels outputs/tiles_<tag>/tile_labels.csv --embeddings-dir outputs/embeddings_<tag> --output-dir models/tile_head_<tag> --maps-dir outputs/maps_<tag>
python scripts/train_pixel_classifier.py --config config/base.yaml --image-id <image_id> --tiles-dir outputs/tiles_<tag> --tile-probs outputs/maps_<tag>/tile_probabilities.csv --output-dir models/pixel_classifier_<tag>
python scripts/run_inference.py --config config/base.yaml --image-id <image_id> --tiles-dir outputs/tiles_<tag> --tile-probs outputs/maps_<tag>/tile_probabilities.csv --pixel-model models/pixel_classifier_<tag>/pixel_model.pkl --pixel-feature-spec models/pixel_classifier_<tag>/pixel_feature_spec.json --maps-dir outputs/maps_<tag>_fused --masks-dir outputs/masks_<tag> --overlays-dir outputs/overlays_<tag> --reports-dir outputs/reports_<tag>
```

Generate report artifacts for model development metrics:

```bash
python scripts/make_report.py --config config/base.yaml --image-id <image_id> --annotations-dir data/annotations --tile-maps-dir outputs/maps_<tag> --pixel-maps-dir outputs/maps_<tag>_fused --masks-dir outputs/masks_<tag> --overlays-dir outputs/overlays_<tag> --reports-dir outputs/reports_<tag> --tile-model-dir models/tile_head_<tag>
python scripts/make_project_report.py --config config/base.yaml --annotations-dir data/annotations --run-tag <tag1> --run-tag <tag2>
```

Reported development metrics are computed on annotated training regions only (scribble-labeled positive vs negative pixels; Ignore/Unlabeled excluded), not whole-slide validation regions.

## One-command Stage 1 runner (readiness-gated)

After saving annotations (`s`) and confirming readiness, you can run all Stage 1 modeling/report steps with one CLI:

```bash
python scripts/run_stage1_image.py \
  --config config/base.yaml \
  --image-id <image_id> \
  --run-tag <tag> \
  --raw-dir data/raw \
  --annotations-dir data/annotations \
  --outputs-root outputs \
  --models-root models
```

What this runner does:

1. Calls shared readiness logic (`annotation_readiness.compute_annotation_readiness`) first.
2. Exits early if status is not `READY` (no model steps run).
3. Runs the Stage 1 scripts in authoritative order:
   - `extract_tiles.py`
   - `embed_vit.py`
   - `make_tile_labels.py`
   - `train_tile_head.py`
   - `train_pixel_classifier.py`
   - `run_inference.py`
   - `make_report.py`
4. Streams live step output to console and writes a combined log:
   - `outputs/reports_<tag>/stage1_runner.log`
5. Writes concise run-summary artifacts:
   - `outputs/reports_<tag>/stage1_run_summary.json`
   - `outputs/reports_<tag>/stage1_run_summary.md`
6. Refreshes per-image report history/comparison artifacts:
   - `outputs/report_history/<image_slug>/history_index.json`
   - `outputs/report_history/<image_slug>/latest_vs_previous.json`
   - `outputs/report_history/<image_slug>/latest_vs_previous.md`

Annotator history browsing reads `history_index.json` when present and backfills from existing `outputs/reports_*` runs for the current image, so older runs are selectable immediately on first launch after this change.

Runner exit codes:

- `0`: full pipeline success
- `1`: annotation artifacts found but not ready
- `2`: readiness/config/artifact error
- `3`: downstream step failure

Standard derived directories used by the runner:

- `outputs/tiles_<tag>`
- `outputs/embeddings_<tag>`
- `outputs/maps_<tag>`
- `outputs/maps_<tag>_fused`
- `outputs/masks_<tag>`
- `outputs/overlays_<tag>`
- `outputs/reports_<tag>`
- `models/tile_head_<tag>`
- `models/pixel_classifier_<tag>`

Final per-image report summaries now include an operator-facing top section with:

- `result_description`
- `evaluation_scope`
- `error_pattern`
- `next_review_focus`

Single-image reports now also include:

- `Supervision summary` (polygon counts + annotated pixel counts by class),
- `Tile supervision summary` (accepted/usable/ignored tiles, label counts, ignored reasons, selection source when available),
- `Class-specific annotated-region metrics` for `Positive_Tumor`, `Negative_Tumor`, and `NonTumor`,
- `Warnings / review focus` as deterministic supervision-quality diagnostics.

Important interpretation:

- Stage 1 inference is still binary (`Positive_Tumor` vs combined negatives).
- Class-specific report sections split `Negative_Tumor` and `NonTumor` for review only; this is not a multiclass model.
- These diagnostics are for annotated-region development review, not whole-slide validation claims.


## Shared multi-image Stage 1 runner (project mode)

Use `scripts/run_stage1_project.py` to train one shared tile head and one shared pixel classifier across a cohort of READY images.

```bash
python scripts/run_stage1_project.py   --config config/base.yaml   --project-tag cohort_a   --case pf0213="PFZ083 MOS634-PD ML1610213 20161010 PD-L1"   --case pf0229="PFZ083 MOS634-PD ML1610229 20161010 PD-L1"
```

You can also pass `--cases-file` with `alias=image_id` lines (blank lines and `#` comments supported).

Behavior summary:

- readiness is evaluated for all requested cases first;
- default: any non-READY case exits before preprocessing/training;
- `--allow-skip-not-ready`: non-READY cases are skipped with reasons;
- shared training requires at least 2 included READY cases.

Shared model artifacts:

- `models/tile_head_<project_tag>_shared/tile_head.pkl`
- `models/pixel_classifier_<project_tag>_shared/pixel_model.pkl`

Per-case run tags are derived as `<project_tag>__<alias>`, and each case still writes canonical per-image outputs:

- `outputs/maps_<project_tag>__<alias>/tile_probabilities.csv`
- `outputs/reports_<project_tag>__<alias>/report_summary.{md,json}`
- `outputs/reports_<project_tag>__<alias>/one_page_report.pdf`

Project-level outputs are written to `outputs/reports_training_<project_tag>/`:

- `training_summary.md`
- `training_summary.json`
- `stage1_project_run_summary.md`
- `stage1_project_run_summary.json`
- `stage1_project_runner.log`

Report scope remains unchanged: these rollups are annotated-region development diagnostics only, not whole-slide validation claims.
Shared training does not change the Stage 1 model family (frozen ViT embeddings + logistic tile head + random-forest pixel classifier + transparent fusion).

## Repository layout

- `config/base.yaml`
- `apps/annotator.py`
- `scripts/run_qc.py`
- `scripts/extract_tiles.py`
- `scripts/embed_vit.py`
- `scripts/make_tile_labels.py`
- `scripts/train_tile_head.py`
- `scripts/train_pixel_classifier.py`
- `scripts/run_inference.py`
- `scripts/make_report.py`
- `scripts/make_project_report.py`
- `scripts/report_metrics.py`
- `data/`
- `models/`
- `outputs/`
- `docs/specs/stage1/`

## Shared project Stage 1 panel
- Annotator now includes **Stage 1 Workflow** for shared-project runs.
- Use: save annotations (`s`), refresh project cases, run shared project Stage 1, preview project `training_summary.md`, optionally load current image shared child report.
- Project cases are auto-discovered from saved artifacts in `data/annotations` and filtered to READY cases.
- Runner writes case manifests: `outputs/reports_training_<project_tag>/stage1_project_cases.json` and `.md`.


### Consolidated Stage 1 Workflow (Project-first)
- Primary GUI action is **Train model and run verification** (shared-project runner).
- Project tags are auto-generated at click-time (`training_YYYYMMDD_HHMMSS`).
- History browsing uses dropdowns for **Project summaries** and **Current image shared reports**; selecting an item auto-loads preview.
- Runner log visibility is controlled with **Show runner log**.
- Single-image CLI (`scripts/run_stage1_image.py`) remains supported.


### UI/Display polish (2026-04-30)
- Consolidated right-hand **Stage 1 Workflow** dock is shrink-friendly: combo boxes elide long labels, status/path labels wrap, and preview/log widgets scroll internally instead of forcing dock width.
- Human-facing metric displays in markdown/UI labels are rounded to **3 decimals** for readability (for example `0.848513 -> 0.849`).
- Machine-readable JSON outputs (`report_summary.json`, `training_summary.json`) remain full precision and authoritative.


### Verification overlay (current-image shared report review)
- `scripts/make_report.py` writes `verification_overlay.png` and `verification_overlay_summary.json` when scribble labels and `positive_mask.png` are available, generated in working-image space and cropped to annotated ROI.
- `verification_overlay.png` is stored as a cropped **single-channel binary positive-mask artifact** (`positive_mask_working_crop`) and should be treated as data, not pre-tinted UI output.
- In the annotator Stage 1 Workflow panel, **Show verification mask** applies to the selected entry in **Current image shared reports** and renders one visible `verification_mask` layer using Napari-controlled styling (red, semi-transparent, nearest interpolation, crop translate offsets).
- While verification review is ON, annotation context uses light class-aware labels while prediction uses opaque class-aware labels; polygon edges remain available and prior styling is restored when review is OFF.
- The prediction mask is rendered as a darker red semi-opaque fill on top of the image for quick TP/FN/FP visual inspection in annotated regions.
- This verification layer is an annotated-region development aid, not whole-slide validation.
- Save-status feedback is hardened so pressing `s` immediately shows in-progress status and reliably updates to completion timing.
- Save-status text updates immediately on `s` and reports final completion/error state.


Verification review UX: GUI review now renders two aligned layers for annotated-region development review (not whole-slide validation): class-aware annotation labels plus a darker/more-opaque positive prediction mask.


### Hibou-B troubleshooting

- authenticate with `hf auth login`
- accept model access in the browser
- use a read token
- use `transformers>=4.53.3,<5`
- run:

```bash
python scripts/check_embedding_encoder_env.py --embedding-encoder hibou_b --try-load
```

## Embedding encoder benchmarking (PR3)

- The Stage 1 Workflow GUI now includes an **Embedding encoder** dropdown sourced from `config/base.yaml`.
- `current_timm` remains the default encoder.
- `hibou_b` can be selected from the GUI when Hugging Face access is configured on the VM.
- Project summaries, single-image shared reports, history labels, and encoder comparison summaries now expose encoder provenance more clearly.
- Metrics remain annotated-region development metrics only; they are not whole-slide validation and not clinical validation.


## Encoder benchmark delta diagnostics (PR4)

- `scripts/compare_encoder_runs.py` now compares more than aggregate metrics and keeps the existing pairwise CLI/output filenames.
- The comparison verifies encoder provenance plus per-image cache signatures and tile-manifest hashes.
- It summarizes per-image tile probability deltas (including threshold flip counts) and writes CSV deltas.
- It summarizes decoded pixel-probability-map deltas and writes absolute-delta PNGs.
- It summarizes final positive-mask deltas (XOR pixels, Dice, Jaccard) and writes XOR PNGs.
- Identical final masks can be expected when continuous values change but do not cross existing tile/pixel/fusion thresholds.
- Metrics remain annotated-region development metrics only (not whole-slide/clinical validation).
- Output JSON now includes a future-friendly benchmark schema (`benchmark_schema_version`, `runs`, `comparisons`) so later GUI multi-encoder selection can reuse this contract.

## Encoder benchmark comparison browser

- Stage 1 Workflow now discovers existing comparison artifacts under `outputs/reports_encoder_comparison_*`.
- The new **Encoder benchmark comparisons** dropdown previews `encoder_comparison_summary.md` in the existing markdown panel.
- Comparison reports are generated by `scripts/compare_encoder_runs.py`.
- Browsing/selecting a comparison is report-only UI behavior and does **not** rerun training.
- Verification viewer behavior is unchanged and still uses the selected **Current image shared reports** entry.
- Metrics shown remain annotated-region development metrics only, not clinical validation.
