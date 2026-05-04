# S1-001 Repo bootstrap

## Goal

Create the initial repository structure, base config, documentation skeleton, and CLI stubs needed to support the Stage 1 pipeline.

## Why this exists

Codex works best when the repo already communicates structure, contracts, and expectations clearly. This spec creates the stable scaffolding every later task depends on.

## In scope

Create or update:

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
- placeholder directories under `data/`, `models/`, `outputs/`, `notebooks/`
- `docs/specs/stage1/`

## Behavior requirements

### README
Must explain:
- project purpose,
- Stage 1 scope and limitations,
- expected pipeline outputs,
- top-level repo layout,
- basic setup,
- high-level command sequence.

### requirements.txt
Must include the minimal handoff stack:
- numpy
- pandas
- scikit-image
- scikit-learn
- opencv-python
- matplotlib
- Pillow
- PyTorch
- timm or transformers
- pyyaml
- jupyter
- optional annotator dependency path (napari or streamlit/gradio)

### config/base.yaml
Must define:
- project metadata,
- tile size and stride,
- class names,
- ViT config block,
- tile-head config block,
- pixel-model config block,
- fusion config block,
- output directory names.

### Script stubs
Each CLI script must:
- parse arguments,
- load config,
- log what it intends to do,
- exit with a clear TODO / NotImplemented path if full behavior is not yet implemented.

### Directory contract
The repo layout must be explicit and stable enough that later tasks can rely on it.

## Non-goals

- implementing full QC,
- implementing annotation UX,
- implementing model training,
- choosing a final encoder.

## Acceptance criteria

- repository contains the agreed Stage 1 layout,
- `python <script> --help` works for every script stub,
- `config/base.yaml` exists and is parseable,
- README documents the intended execution flow,
- no placeholder code claims to perform completed analysis when it does not.

## Verification

Minimum:
- run `--help` on each script,
- parse `config/base.yaml` in a smoke test,
- confirm expected directories exist.

## Open questions

- napari vs streamlit/gradio for the annotator,
- exact package pins,
- exact ViT encoder selection.
