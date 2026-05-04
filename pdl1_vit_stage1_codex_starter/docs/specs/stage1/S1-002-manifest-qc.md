# S1-002 Manifest and QC

## Goal

Implement reproducible manifest creation/update plus lightweight ingest QC for Stage 1 raw images.

## Inputs

- raw image files under `data/raw/`
- config from `config/base.yaml`

## Outputs

- `data/manifests/image_manifest.csv`
- `data/qc/qc_report.csv`
- `data/qc/qc_thumbnails/`
- optional per-image JSON sidecars

## Required manifest columns

- `image_id`
- `source_url`
- `marker`
- `stain`
- `cancer_type`
- `download_date`
- `license_checked`
- `roi_status`
- `annotation_status`
- `qc_status`

## Required QC checks

For each discovered/manifested image:

- verify file load,
- record width and height,
- estimate tissue fraction,
- estimate blank/background fraction,
- emit simple issue estimates (`blur`, `pen marks`, `fold/tear`) where possible,
- write QC status + notes/error fields.

## Thumbnails

Create lightweight manual-review artifacts in `data/qc/qc_thumbnails/`.

## CLI behavior

`run_qc.py` supports:

- `--config`
- `--input-manifest` or raw-directory discovery (`--input`)
- `--output-dir`
- `--manifest-out`
- deterministic overwrite via `--force`

## Current limitation (must stay explicit)

`run_qc.py` supports `.svs` discovery for manifest/QC row creation, but SVS handling is currently **metadata-only** (presence/path/size/mtime and dimensions when OpenSlide is available). It does not perform full raster-level blur/pen/fold QC parity for whole-slide images.

## Non-goals

- pathology-grade QC scoring,
- whole-slide ingest parity,
- perfect artifact detection.

## Acceptance criteria

- manifest can be created or updated reproducibly,
- QC runs end-to-end on at least one supported sample image,
- `qc_report.csv` contains required columns,
- thumbnails are created for readable images,
- failed loads are reported cleanly.

## Verification

Smoke test on one supported image, then on available supported-image set.

## Risks

- public image heterogeneity,
- ambiguous stain interpretation,
- thumbnail generation memory issues for large images.
