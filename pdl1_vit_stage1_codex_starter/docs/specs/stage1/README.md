# Stage 1 specs

This directory holds the **spec-first contracts** for the PD-L1 ViT proof of concept.

## How to work

1. Pick one spec.
2. Open a dedicated branch / Codex task for that spec only.
3. Implement the smallest slice that satisfies the acceptance criteria.
4. Verify locally or via Codex cloud.
5. Open a PR with the checklist from the PR template.
6. Ask Codex for review and then do human review.

## Recommended order

1. `S1-000-stage1-charter.md`
2. `S1-001-repo-bootstrap.md`
3. `S1-002-manifest-qc.md`
4. `S1-003-annotator.md`
5. `S1-004-tiling-embedding-modeling.md`

## Spec template shape

Each spec should define:

- why the work exists,
- inputs and outputs,
- implementation scope,
- non-goals,
- acceptance criteria,
- verification,
- risks / open questions.

## Governance

- Specs should describe **observable behavior**.
- ADRs or design notes can justify architecture choices.
- Code should follow specs; if reality changes, update the spec in the same PR.


### Verification overlay (current-image shared report review)
- `scripts/make_report.py` now writes `verification_overlay.png` and `verification_overlay_summary.json` into the run overlays directory when scribble labels and `positive_mask.png` are available.
- Overlay semantics are ROI-cropped positive-mask review (`positive_mask_crop`) in working image space, scoped to annotated-region extent (+ padding).
- This is an annotated-region model-development review aid, not whole-slide validation.
- In the annotator Stage 1 Workflow panel, **Show verification mask** applies to the currently selected entry in **Current image shared reports** and toggles a cropped `verification_mask` layer on/off.
