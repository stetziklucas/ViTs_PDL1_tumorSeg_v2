# Stage 1 Single-Image Development Report

image_id: PFZ083 MOS634-PD ML1610213 20161010 PD-L1
run_tag: training_20260430_000442__pfz083_mos634-pd_ml1610213_20161010_pd-l1
model_scope: shared_project_model
shared_model_tag: training_20260430_000442
training_image_count: 2
included_training_aliases: pfz083_mos634-pd_ml1610213_20161010_pd-l1, pfz083_mos634-pd_ml1610229_20161010_pd-l1

## Operator-facing summary

- result_description: Tumor-enriched PD-L1-positive mask in working image space
- evaluation_scope: Annotated-region development metrics only; not whole-slide validation
- error_pattern: Conservative / false-negative dominant
- next_review_focus: Review undercalled annotated positive tumor regions and add representative positive supervision.

Note: Metrics below are computed on annotated training regions only, not whole-slide validation.
Ignore/Unlabeled pixels are excluded.

## Development metrics

| metric | value |
| --- | ---: |
| false_positive_px | 0 |
| false_negative_px | 1891 |
| precision | 1.000000 |
| sensitivity | 0.687954 |
| f1 | 0.815133 |
| training_log_loss_total | 35.359741 |

## Supervision summary

### Polygon counts by class
- Ignore: 4
- Negative_Tumor: 6
- NonTumor: 5
- Positive_Tumor: 6

### Annotated pixel counts by class
- Unlabeled: 513925083
- Positive_Tumor: 388814
- Negative_Tumor: 86907
- NonTumor: 218676
- Ignore: 37624

## Tile supervision summary

- accepted_tile_count: 6
- usable_tile_count: 6
- ignored_tile_count: 0
- ignored_tile_share: 0.0000

### Tile label counts
- Positive_Context: 6

### Tile label_reason counts
- sparse_positive_seed: 6

### Ignored tile reasons
- none

### Selection-source counts
- none

## Class-specific annotated-region metrics

| class | annotated_px | primary_correct_px | primary_error_px | score_name | score |
| --- | ---: | ---: | ---: | --- | ---: |
| Positive_Tumor | 6060 | 4169 | 1891 | sensitivity | 0.687954 |
| Negative_Tumor | 1349 | 1349 | 0 | specificity | 1.000000 |
| NonTumor | 3417 | 3417 | 0 | specificity | 1.000000 |

## Additional machine-readable counts

- tp_px: 4169
- annotated_positive_px: 6060
- annotated_negative_px: 4766
- annotated_total_px: 10826
- training_log_loss_mean: 0.003266

## Warnings / review focus

- very low usable negative tile support

## Working-space caveat

- All masks/maps/overlay are exported in working image space (selected SVS level when applicable).
