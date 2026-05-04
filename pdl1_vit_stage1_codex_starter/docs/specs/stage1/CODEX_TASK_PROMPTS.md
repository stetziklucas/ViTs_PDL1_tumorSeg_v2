# Copy/paste Codex prompts for Stage 1

These prompts follow the pattern:
- Goal
- Context
- Constraints
- Done when
- Verification

---

## Prompt 1 — bootstrap the repo

Please work from `docs/specs/stage1/S1-001-repo-bootstrap.md` and `AGENTS.md`.

Goal:
Bootstrap the repository for the Stage 1 PD-L1 ViT proof of concept.

Context:
The repo should support a pipeline of manifest/QC -> annotator -> tile extraction -> frozen ViT embeddings -> tile classifier -> pixel classifier -> fused inference -> report.
Use the handoff-aligned structure in the spec.

Constraints:
- Keep this PR to repo structure, docs, config, and CLI stubs only.
- Do not implement real modeling yet.
- Add only the minimum dependencies needed for the bootstrap.
- Keep filenames and directories aligned with the spec.
- Make all script entrypoints expose `--help`.

Done when:
- the repo contains the Stage 1 layout,
- `README.md` explains setup and intended execution flow,
- `config/base.yaml` exists and parses,
- each script stub has a basic CLI,
- no code falsely claims the pipeline is already implemented.

Verification:
- run `python <script> --help` for each script,
- run a small config parse smoke test,
- report exactly what was verified.

Please propose a short plan first, then implement.

---

## Prompt 2 — implement manifest + QC

Please work from `docs/specs/stage1/S1-002-manifest-qc.md` and `AGENTS.md`.

Goal:
Implement the manifest builder and basic QC pipeline for public PD-L1 IHC images.

Context:
This is Stage 1 only. The QC should be lightweight but reproducible. We need manifest discipline and thumbnail outputs even for 1–2 images.

Constraints:
- Keep the implementation simple and transparent.
- Prefer deterministic CSV outputs.
- Do not overengineer artifact detection.
- Fail cleanly on unreadable files.
- Keep the CLI explicit.

Done when:
- a manifest CSV can be created or updated,
- QC produces `qc_report.csv`,
- QC thumbnails are written,
- the required columns exist,
- the script works on at least one sample image.

Verification:
- run the QC workflow on a small sample,
- report generated files and any assumptions.

Please inspect existing repo files first, then plan, then implement the minimal complete slice.

---

## Prompt 3 — build the napari annotator

Please work from `docs/specs/stage1/S1-003-annotator.md` and `AGENTS.md`.

Goal:
Build a lightweight napari-based annotator for ROI + sparse scribbles.

Context:
Stage 1 needs a Python-owned annotation workflow that saves `roi_mask.png`, `scribble_labels.png`, and metadata JSON. Sparse scribbles are enough; dense outlines are not required.

Constraints:
- Use napari for the first implementation.
- Keep the app simple and local.
- Save outputs deterministically under `data/annotations/`.
- Document the label encoding.
- Ensure reloading previous annotations works.

Done when:
- a user can open one image,
- create an ROI,
- paint scribbles for the required classes,
- save artifacts,
- reopen and reload those artifacts.

Verification:
- provide manual smoke-test steps,
- if possible, run non-GUI verification for save/load helper functions.

Please propose the app structure first, then implement.

---

## Prompt 4 — implement tiling + embeddings cache

Please work from `docs/specs/stage1/S1-004-tiling-embedding-modeling.md` and `AGENTS.md`.

Goal:
Implement tissue masking, tile extraction, and frozen ViT embedding generation.

Context:
This is the first modeling slice. Tiles should be 224 px with default stride 112 px, filtered by tissue fraction and ROI intersection. Embeddings should be cached.

Constraints:
- Keep the encoder frozen.
- Do not add training of any deep model.
- Make coordinate handling explicit.
- Write deterministic manifests and embedding indices.
- Keep the code structured so later scripts can reuse the outputs.

Done when:
- tile manifest generation works,
- embeddings are produced and cached,
- reruns skip recomputation when appropriate,
- outputs follow the spec.

Verification:
- run a smoke test on one image,
- report tile count and embedding array shape.

Please plan first, then implement.

---

## Prompt 5 — tile labels + tile head

Please work from `docs/specs/stage1/S1-004-tiling-embedding-modeling.md` and `AGENTS.md`.

Goal:
Generate tile labels from scribbles and train the first tile classifier head.

Context:
We want a simple logistic regression on frozen ViT embeddings to produce a tile-level probability map.

Constraints:
- Use the scribble-derived tile-label rules from the spec.
- Ignore low-coverage / mixed tiles.
- Keep the first model simple and explainable.
- Export metrics and probability maps.
- Do not add an MLP unless the simple baseline clearly needs it.

Done when:
- `tile_labels.csv` is generated,
- the tile head trains,
- CV metrics are written,
- a tile probability map is exported.

Verification:
- run a smoke test on the toy dataset,
- report class counts and output paths.

Please inspect the existing outputs first, then plan, then implement.

---

## Prompt 6 — pixel classifier + fused inference

Please work from `docs/specs/stage1/S1-004-tiling-embedding-modeling.md` and `AGENTS.md`.

Goal:
Implement the pixel classifier and fused inference pipeline.

Context:
The final positive mask should combine tissue mask, ROI, tile prior, and pixel probability thresholding.

Constraints:
- Use a random forest first unless the existing code strongly suggests logistic regression is simpler.
- Feature set should include RGB, HED, Gaussian/local texture, and tile prior.
- Keep the fusion rule transparent and configurable.
- Export `positive_mask.png`, `overlay.png`, and `metrics.json`.

Done when:
- the pixel model trains on scribble-derived samples,
- pixel probability map is exported,
- fused inference exports final mask + overlay + metrics,
- output dimensions align with original image coordinates.

Verification:
- run an end-to-end smoke test on one image,
- report produced artifacts and any failure modes observed.

Please plan first, then implement.

---

## Prompt 7 — PR review request

Review this branch against `AGENTS.md` and the relevant Stage 1 spec.

Focus on:
- broken output contracts,
- coordinate or shape mismatches,
- silent config drift,
- misleading docs,
- weak verification,
- unnecessary refactors.

Report:
1. blocking issues,
2. non-blocking issues,
3. verification gaps,
4. whether the branch matches the spec.

### Shared-project annotator flow update
- Keep single-image Stage 1 panel unchanged for per-image runs.
- Added shared-project panel that shells out to `scripts/run_stage1_project.py --discover-ready-cases`.
- Project preview defaults to `outputs/reports_training_<project_tag>/training_summary.md` and can switch to current-image child report `outputs/reports_<project_tag>__<alias>/report_summary.md`.
- Auto-discovery uses saved artifacts only; no implicit save occurs before project run.
