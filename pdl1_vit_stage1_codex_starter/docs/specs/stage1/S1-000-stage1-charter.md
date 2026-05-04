# S1-000 Stage 1 charter

## Purpose

Deliver a credible Stage 1 PD-L1 IHC proof of concept that demonstrates a Python-owned workflow for annotation, frozen ViT-based tile context, and a fused pixel-level positive mask.

## Scope

The demo must show:

1. ingest of 1–2 public PD-L1 IHC images,
2. manifest and QC discipline,
3. lightweight in-house Python annotation,
4. frozen ViT tile embeddings,
5. lightweight tile classifier head,
6. lightweight pixel classifier using image features + tile prior,
7. fused inference outputs,
8. visual overlays, metrics, and one-page report(s).

## Deliverables per image

- `roi_mask.png`
- `tile_prob_map.png`
- `positive_mask.png`
- `overlay.png`
- `metrics.json`
- `one_page_report.pdf`

## Honest output statement

“PD-L1-positive tumor mask” means positive DAB signal constrained by:
- manually defined tumor-rich ROI, and
- a ViT-derived tumor/context prior.

This is **not** a validated tumor proportion score.

## Architecture contract

public image(s)
-> manifest + QC
-> Python annotator
-> tissue mask + ROI export
-> tile extraction
-> frozen ViT encoder
-> tile embeddings
-> tile classifier head
-> tumor/context probability map
-> pixel classifier
-> fused mask
-> overlay + metrics + report

## Non-goals

- no diagnostic claims,
- no end-to-end deep segmentation model,
- no foundation-model training,
- no broad benchmarking study,
- no scientific performance claim from 1–2 images.

## Success criteria

A successful Stage 1 result demonstrates:

- Python-owned reproducible annotation artifacts,
- a meaningful frozen ViT branch,
- visually plausible outputs,
- masks mostly constrained to tumor-rich regions,
- automatic metrics export,
- honest documentation of failure modes,
- a clean swap path to internal data later.

## Limitations to keep visible

- tiny public dataset,
- sparse scribble supervision,
- exploratory frozen encoder,
- no claim of generalization.

## Exit condition

Stage 1 is complete when the repo can produce the listed artifacts for at least one strong image and ideally one more ambiguous image using documented commands and configs.
