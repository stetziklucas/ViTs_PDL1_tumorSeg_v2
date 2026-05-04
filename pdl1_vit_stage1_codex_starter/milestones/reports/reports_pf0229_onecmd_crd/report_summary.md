# Stage 1 Single-Image Development Report

image_id: PFZ083 MOS634-PD ML1610229 20161010 PD-L1
run_tag: pf0229_onecmd_crd

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
| false_negative_px | 380 |
| precision | 1.000000 |
| sensitivity | 0.852370 |
| f1 | 0.920302 |
| training_log_loss_total | 20.293556 |

## Supervision summary

### Polygon counts by class
- Ignore: 3
- Negative_Tumor: 3
- NonTumor: 3
- Positive_Tumor: 3

### Annotated pixel counts by class
- Unlabeled: 707631009
- Positive_Tumor: 165251
- Negative_Tumor: 52264
- NonTumor: 111043
- Ignore: 286073

## Tile supervision summary

- accepted_tile_count: 11
- usable_tile_count: 7
- ignored_tile_count: 4
- ignored_tile_share: 0.3636

### Tile label counts
- Positive_Context: 6
- Ignore: 4
- Negative_Context: 1

### Tile label_reason counts
- sparse_positive_seed: 6
- mixed_or_ambiguous: 4
- sparse_negative_seed: 1

### Ignored tile reasons
- mixed_or_ambiguous: 4

### Selection-source counts
- none

## Class-specific annotated-region metrics

| class | annotated_px | primary_correct_px | primary_error_px | score_name | score |
| --- | ---: | ---: | ---: | --- | ---: |
| Positive_Tumor | 2574 | 2194 | 380 | sensitivity | 0.852370 |
| Negative_Tumor | 815 | 815 | 0 | specificity | 1.000000 |
| NonTumor | 1741 | 1741 | 0 | specificity | 1.000000 |

## Additional machine-readable counts

- tp_px: 2194
- annotated_positive_px: 2574
- annotated_negative_px: 2556
- annotated_total_px: 5130
- training_log_loss_mean: 0.003956

## Warnings / review focus

- very low usable negative tile support

## Working-space caveat

- All masks/maps/overlay are exported in working image space (selected SVS level when applicable).
