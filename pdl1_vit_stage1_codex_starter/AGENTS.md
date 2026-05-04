# AGENTS.md

## Repository mission

Build a Stage 1 proof-of-concept pipeline for PD-L1 IHC image analysis that:

1. ingests 1–2 public PD-L1 IHC images,
2. supports Python-owned ROI + sparse scribble annotation,
3. computes frozen ViT tile embeddings,
4. trains a lightweight tile classifier head,
5. trains a lightweight Python pixel classifier, and
6. fuses tissue mask + ROI + tile prior + pixel probabilities into a reproducible PD-L1-positive tumor mask.

This repo is for a **research / screening-enrichment proof of concept**, not a diagnostic workflow.

## Working style

- Follow the specs in `docs/specs/stage1/`.
- For non-trivial work, read the relevant spec before editing code.
- Keep PRs and tasks narrow: one bounded deliverable per branch.
- Prefer incremental, reviewable changes over broad refactors.
- Preserve clear data and file contracts.
- Do not silently change output filenames, directory layout, or config keys without updating the spec and README.

## Primary architecture constraints

The Stage 1 architecture is:

public image(s)
-> manifest + QC
-> Python annotator
-> tissue mask + ROI export
-> tile extraction
-> frozen ViT encoder
-> tile embeddings
-> tile classifier head
-> tile probability map
-> pixel classifier
-> fused final mask
-> overlay + metrics + report

## Hard rules

- Do **not** replace the Stage 1 pixel classifier with a full end-to-end segmentation network.
- Do **not** train a foundation model from scratch.
- Keep the ViT encoder frozen in Stage 1.
- Keep annotation artifacts repo-owned and simple: PNG / CSV / JSON.
- Avoid introducing heavyweight infrastructure unless explicitly required by a spec.
- Avoid premature optimization.
- Do not claim scientific performance or clinical validity from 1–2 public images.

## Preferred implementation choices

- Python >= 3.10
- Use `pathlib` instead of raw string paths where practical.
- Use typed function signatures where practical.
- Prefer small pure functions for image transforms / feature extraction.
- Prefer YAML config over hard-coded constants.
- Prefer `logging` over print spam.
- Save intermediate artifacts with stable names and deterministic directory conventions.

## Repo layout expectations

Expected top-level structure:

- `README.md`
- `requirements.txt`
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
- `data/`
- `models/`
- `outputs/`
- `notebooks/`
- `docs/specs/stage1/`

## Coding conventions

- Add docstrings to public functions and CLI entrypoints.
- Keep CLIs explicit and reproducible.
- Include `--config` and `--image-id` / `--input` style arguments where relevant.
- Surface assumptions in code comments only when they are truly clarifying.
- Fail loudly on missing files, malformed config, or shape mismatches.
- For notebooks, keep them exploratory and non-authoritative; the scripts are the source of truth.

## Verification rules

When you change code, verify the smallest reasonable thing:

- targeted script help output,
- a focused unit test,
- a smoke test on a tiny sample image,
- or a dry run that proves files are created.

If you cannot run verification, say exactly what should be run and why it was not executed.

## Definition of done for any task

A task is done only if:

1. the requested code changes are implemented,
2. the relevant spec is updated if behavior changed,
3. the verification steps are run or clearly described,
4. output contracts remain explicit,
5. there are no unrelated refactors,
6. the final summary states what changed, what was verified, and any open risks.

## Review guidelines

Treat the following as high priority in review:

- broken file contracts,
- hidden config drift,
- data leakage between train/inference steps,
- silently changing image geometry or coordinate systems,
- weak reproducibility,
- misleading claims in docs or reports,
- failure to document limitations.

## Task execution pattern

For complex tasks:

1. summarize the target behavior,
2. inspect relevant files/specs,
3. propose a brief plan,
4. implement minimal changes,
5. verify,
6. report crisply.

## Prompting preference

When asked to implement from a spec, keep a checklist of:
- Goal
- Context
- Constraints
- Done when
- Verification run
