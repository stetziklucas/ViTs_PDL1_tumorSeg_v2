# S1-003 Python annotator

## Goal

Provide a lightweight Python-owned annotator where classed polygons are the single source of truth and downstream-compatible raster artifacts are derived deterministically on save.

## Current implementation stance

- Napari desktop annotator is the current Stage 1 implementation.
- User workflow is polygon/lasso-first with required class selection.
- Metadata JSON stores polygon geometry + class assignments.

## Annotation classes

- `Positive_Tumor`
- `Negative_Tumor`
- `NonTumor`
- `Ignore`

## Inputs

- image path (`.png/.jpg/...` and `.svs` when OpenSlide is available)
- class list (defaults from config)
- optional prior annotation metadata

## Required outputs

- `data/annotations/roi_masks/{image_id}_roi_mask.png` (derived)
- `data/annotations/scribbles/{image_id}_scribble_labels.png` (derived)
- `data/annotations/{image_id}_annotation_meta.json` (canonical polygon source)

## Source-of-truth semantics

- Canonical annotation state is the classed polygon list in metadata JSON.
- `roi_mask.png` is derived from union of tumor polygons (Positive_Tumor + Negative_Tumor).
- `scribble_labels.png` is a derived rasterized class-label map from polygons.
- On save, derived outputs are regenerated deterministically from polygon metadata.

## Required metadata fields

- `image_id`
- `annotator`
- `timestamp`
- `classes`
- `notes`
- `uncertainty_comment`
- polygon payload (class + vertices)
- optional app version / git commit

## Label encoding contract

- 0 = Unlabeled
- 1 = Positive_Tumor
- 2 = Negative_Tumor
- 3 = NonTumor
- 4 = Ignore

## UX requirements

Minimum viable UX:

- load image and show polygon layer,
- class selector required before drawing,
- explicit class-control dock dependency (`magicgui`),
- fail loudly if class-control UI init fails,
- deterministic save paths,
- reload prior polygons from metadata when present.
- persistent right-dock save/readiness feedback:
  - Save status
  - Readiness
  - Next action
- save action (`s`) must keep annotator open and show explicit completion/readiness guidance.
- Stage 1 run controls are history-browser-first:
  - editable run tag + generate/run buttons,
  - per-image history dropdown with refresh/load/newest/oldest actions,
  - clearly visible selected run tag (not buried in long status text),
  - selected history provenance details (timestamp, model scope, shared model tag/training image count when present),
  - in-dock markdown preview of selected history run report,
  - show/hide runner log toggle that expands preview area when hidden,
  - latest-vs-previous artifacts remain generated on disk for compatibility but are not the primary GUI path.

## Save feedback + readiness behavior

- Press `s` to save canonical artifacts:
  - `data/annotations/{image_id}_annotation_meta.json`
  - `data/annotations/roi_masks/{image_id}_roi_mask.png`
  - `data/annotations/scribbles/{image_id}_scribble_labels.png`
- While writing artifacts, UI shows `Saving... please wait`.
- On completion, annotator runs a lightweight readiness summary on saved artifacts and updates persistent dock labels.
- Readiness is an annotation preflight only (not model quality/validation).
- Repeated `s` during an in-flight save is ignored (re-entrant guard).
- Save failures surface explicit error status without closing the window.
- Save-status panel transitions are immediate and reliable on repeated saves (`Saving...` -> `Saved successfully in X.XXs`).

## CLI preflight checker

- Script: `scripts/check_annotation_readiness.py`
- Purpose: SSH-friendly annotation readiness summary for one image id.
- Inputs:
  - `--config`
  - `--image-id`
  - `--annotations-dir`
  - `--json` (optional machine-readable output)
- Exit codes:
  - `0`: `READY`
  - `1`: not ready / needs more annotation
  - `2`: artifact `ERROR` (missing or broken artifacts)

## Dependency notes

- `.svs` loading requires `openslide-python` plus system OpenSlide libs.

## Non-goals

- collaborative editing,
- authentication,
- dense-production annotation tooling.

## Acceptance criteria

- image can be opened,
- classed polygons can be drawn/saved,
- metadata JSON is emitted,
- derived ROI/scribble rasters are emitted on save,
- saved polygons reload without geometry mismatch.

## Verification

Manual smoke test:
1. open sample image,
2. draw one polygon per class,
3. save,
4. reload,
5. confirm metadata polygon fidelity and regenerated ROI/scribble artifacts.

## Risks

- napari dependency/setup issues,
- memory pressure on very large images,
- coordinate consistency across tools.


### UI/Display polish (2026-04-30)
- Consolidated right-hand **Stage 1 Workflow** dock is shrink-friendly: combo boxes elide long labels, status/path labels wrap, and preview/log widgets scroll internally instead of forcing dock width.
- Human-facing metric displays in markdown/UI labels are rounded to **3 decimals** for readability (for example `0.848513 -> 0.849`).
- Machine-readable JSON outputs (`report_summary.json`, `training_summary.json`) remain full precision and authoritative.
- Verification review toggle (`Show verification mask`) loads ROI-cropped working-space positive-mask artifacts (`positive_mask_working_crop`) for the currently selected current-image shared report entry.
- Verification masks are rendered by Napari layer settings (strong-red semi-transparent fill, nearest interpolation, crop translate offsets); the artifact itself is single-channel mask data.
- While verification review is enabled, class polygons remain visible with lighter translucent class-colored fills and visible edges (not hidden), while model prediction uses an opaque class-aware label layer for side-by-side comparison and immediate false-positive/false-negative inspection.
- Verification overlays are development-only annotated-region review aids, not whole-slide validation signals.
