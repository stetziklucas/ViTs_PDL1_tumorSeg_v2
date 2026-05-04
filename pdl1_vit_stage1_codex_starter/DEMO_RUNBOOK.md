# Demo Runbook (Stage 1 annotation save + readiness)

## Goal

Provide an operator-friendly annotation loop where save completion and readiness are explicit after pressing `s` in Napari.

## Annotator workflow

1. Launch annotator:
   ```bash
   python apps/annotator.py --config config/base.yaml --image-id <image_id>
   ```
2. Draw classed polygons.
3. Press `s`.
4. Wait for the right-dock status fields to update:
   - Save status
   - Readiness
   - Next action
5. Continue downstream steps only after reading the readiness guidance.

The annotator stays open after save (no auto-close).

## In-annotator Stage 1 run panel (current image)

The annotator right dock now includes **Stage 1 Workflow**:

- editable `Run tag`,
- `Generate new run tag`,
- `Run Stage 1` (background/non-blocking),
- `History` dropdown for any discovered prior run on this image,
- `Refresh history`,
- `Load selected report`,
- `Jump to newest` and `Jump to oldest`,
- `Show runner log` toggle (hide log to expand preview space),
- run status + explicit latest/selected run metadata + markdown preview.

Primary UX is now: **save -> run -> choose report history entry -> preview selected report**.
The panel uses in-dock markdown as the default report-reading path (no desktop-open tool required).
Latest-vs-previous artifacts are still written under `outputs/report_history/<image_slug>/`, but direct history selection is now the primary GUI review flow.

## Canonical artifacts written on `s`

- `data/annotations/{image_id}_annotation_meta.json`
- `data/annotations/roi_masks/{image_id}_roi_mask.png`
- `data/annotations/scribbles/{image_id}_scribble_labels.png`

## SSH readiness check

Run lightweight preflight (no tile/model steps):

```bash
python scripts/check_annotation_readiness.py --config config/base.yaml --image-id <image_id> --annotations-dir data/annotations
```

Machine-readable JSON output:

```bash
python scripts/check_annotation_readiness.py --config config/base.yaml --image-id <image_id> --annotations-dir data/annotations --json
```

Exit codes:

- `0` = READY
- `1` = not ready (needs more annotation)
- `2` = ERROR (missing/broken artifacts)

## One-command Stage 1 execution after readiness

Once readiness is `READY`, run the full Stage 1 pipeline for one image:

```bash
python scripts/run_stage1_image.py \
  --config config/base.yaml \
  --image-id <image_id> \
  --run-tag <tag>
```

The runner derives and uses standard suffixed directories:

- `outputs/tiles_<tag>`
- `outputs/embeddings_<tag>`
- `outputs/maps_<tag>`
- `outputs/maps_<tag>_fused`
- `outputs/masks_<tag>`
- `outputs/overlays_<tag>`
- `outputs/reports_<tag>`
- `models/tile_head_<tag>`
- `models/pixel_classifier_<tag>`

It streams progress live and writes:

- `outputs/reports_<tag>/stage1_runner.log`
- `outputs/reports_<tag>/stage1_run_summary.json`
- `outputs/reports_<tag>/stage1_run_summary.md`

Runner exit codes:

- `0`: full Stage 1 success
- `1`: annotation set exists but is not READY
- `2`: readiness/config/artifact error
- `3`: downstream pipeline step failure

For human review, start with:

- `outputs/reports_<tag>/report_summary.md`
- `outputs/reports_<tag>/report_summary.json`
- `outputs/reports_<tag>/one_page_report.pdf`
- `outputs/reports_<tag>/stage1_run_summary.md`

`report_summary.md` now includes supervision-audit sections intended for pathologist/operator development review:

- **Supervision summary**: polygon counts and annotated pixels by class.
- **Tile supervision summary**: accepted/usable/ignored tile counts, tile label counts, label reasons, and ignored-tile reasons.
- **Class-specific annotated-region metrics**: `Positive_Tumor`, `Negative_Tumor`, and `NonTumor` split from the same binary Stage 1 positive-mask outputs.
- **Warnings / review focus**: deterministic supervision-quality flags (for example low usable negative support or high ignore share).

These report sections are annotated-region diagnostics only and should not be interpreted as whole-slide clinical validation.

Each successful single-image run also refreshes per-image history artifacts under:

- `outputs/report_history/<image_slug>/history_index.json`
- `outputs/report_history/<image_slug>/latest_vs_previous.json`
- `outputs/report_history/<image_slug>/latest_vs_previous.md`

History indexing backfills from existing `outputs/reports_*` folders for the same image id, so comparison can be available before a brand-new run is launched in the current session.
The history dropdown therefore populates from already-existing runs immediately after launch (no new run required first).


## Shared project-mode training demo

After multiple images are annotation-READY, run one shared Stage 1 training pass:

```bash
python scripts/run_stage1_project.py   --config config/base.yaml   --project-tag demo_shared   --case pf0213="PFZ083 MOS634-PD ML1610213 20161010 PD-L1"   --case pf0229="PFZ083 MOS634-PD ML1610229 20161010 PD-L1"
```

Optional case list file format (`--cases-file`):

```text
# alias=image_id
pf0213=PFZ083 MOS634-PD ML1610213 20161010 PD-L1
pf0229=PFZ083 MOS634-PD ML1610229 20161010 PD-L1
```

The runner trains shared models once, then performs inference/reporting per case using run tags `<project_tag>__<alias>`.
Per-image report filenames are unchanged. Project rollups remain annotated-region development diagnostics, not whole-slide validation.
Project rollup now ingests per-image `report_summary.json` as authoritative when present, so shared child runs are not skipped solely because per-child `models/tile_head_<run_tag>` directories are absent.

## Shared-project GUI path
1. Save annotations for each image.
2. In annotator: Refresh project cases.
3. Run shared project Stage 1.
4. Load latest project summary.
5. Optionally load current-image shared child report.


### UI/Display polish (2026-04-30)
- Consolidated right-hand **Stage 1 Workflow** dock is shrink-friendly: combo boxes elide long labels, status/path labels wrap, and preview/log widgets scroll internally instead of forcing dock width.
- Human-facing metric displays in markdown/UI labels are rounded to **3 decimals** for readability (for example `0.848513 -> 0.849`).
- Machine-readable JSON outputs (`report_summary.json`, `training_summary.json`) remain full precision and authoritative.


### Verification overlay (current-image shared report review)
- `scripts/make_report.py` writes `verification_overlay.png` and `verification_overlay_summary.json` into the run overlays directory when scribble labels and `positive_mask.png` are available; generation is in working image space and cropped to annotated ROI.
- `verification_overlay.png` is a cropped **single-channel binary positive-mask artifact** (`positive_mask_working_crop`) so GUI rendering is controlled by Napari layer settings.
- This is an annotated-region model-development review aid, not whole-slide validation.
- In the annotator Stage 1 Workflow panel, **Show verification mask** applies to the currently selected entry in **Current image shared reports** and displays a clearly visible red semi-transparent `verification_mask` layer with crop translate offsets.
- During verification review, annotation polygons remain visible as lighter translucent class-colored fills (plus edges), while prediction is rendered as an opaque class-aware label mask (positive=red, negative=gold, nontumor=green, ignore=gray, outside-ROI=magenta); original polygon styling is restored when review is OFF.
- Save feedback in the Annotator panel now immediately flips to "Saving... please wait" on `s` and then to a deterministic success duration after write completion.

- Verification review now uses an ROI-cropped positive-mask overlay derived from annotated scribble extent (development review aid, not whole-slide validation).
- Save-status panel is hardened: pressing `s` immediately shows saving state, then success/error reliably.


Verification review UX: GUI review now renders two aligned layers for annotated-region development review (not whole-slide validation): class-aware annotation labels plus a darker/more-opaque positive prediction mask.
