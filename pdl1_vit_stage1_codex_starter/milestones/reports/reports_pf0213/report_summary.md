# Stage 1 Single-Image Development Report

image_id: PFZ083 MOS634-PD ML1610213 20161010 PD-L1
run_tag: pf0213

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
| false_negative_px | 720805 |
| precision | 1.000000 |
| sensitivity | 0.622002 |
| f1 | 0.766956 |
| training_log_loss_total | 269838.343750 |

## Supervision summary

### Polygon counts by class
- Negative_Tumor: 4
- NonTumor: 2
- Positive_Tumor: 5

### Annotated pixel counts by class
- Unlabeled: 257301126
- Positive_Tumor: 122037774
- Negative_Tumor: 135318204
- NonTumor: 0
- Ignore: 0

## Tile supervision summary

- accepted_tile_count: 162
- usable_tile_count: 146
- ignored_tile_count: 16
- ignored_tile_share: 0.0988

### Tile label counts
- Positive_Context: 135
- Ignore: 16
- Negative_Context: 11

### Tile label_reason counts
- dominant_positive: 135
- mixed_or_ambiguous: 16
- dominant_negative: 11

### Ignored tile reasons
- mixed_or_ambiguous: 16

### Selection-source counts
- none

## Class-specific annotated-region metrics

| class | annotated_px | primary_correct_px | primary_error_px | score_name | score |
| --- | ---: | ---: | ---: | --- | ---: |
| Positive_Tumor | 1906902 | 1186097 | 720805 | sensitivity | 0.622002 |
| Negative_Tumor | 2114129 | 2114129 | 0 | specificity | 1.000000 |
| NonTumor | 0 | 0 | 0 | specificity | 0.000000 |

## Additional machine-readable counts

- tp_px: 1186097
- annotated_positive_px: 1906902
- annotated_negative_px: 2114129
- annotated_total_px: 4021031
- training_log_loss_mean: 0.067107

## Warnings / review focus

- none

## Working-space caveat

- All masks/maps/overlay are exported in working image space (selected SVS level when applicable).
