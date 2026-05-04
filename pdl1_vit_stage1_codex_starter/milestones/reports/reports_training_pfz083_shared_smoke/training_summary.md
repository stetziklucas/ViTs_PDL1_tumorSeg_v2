# Stage 1 Training Development Summary

Note: Metrics are computed on annotated training regions only (scribble labels),
not on whole-slide unlabeled validation regions. Ignore/Unlabeled pixels are excluded.

## Aggregate project summary

| metric | value |
| --- | ---: |
| false_positive_px | 0 |
| false_negative_px | 0 |
| precision | 0.000000 |
| sensitivity | 0.000000 |
| f1 | 0.000000 |
| training_log_loss_total | 0.000000 |

## Aggregate class-specific annotated-region metrics

| class | annotated_px | primary_correct_px | primary_error_px | score_name | score |
| --- | ---: | ---: | ---: | --- | ---: |
| Positive_Tumor | 0 | 0 | 0 | sensitivity | 0.000000 |
| Negative_Tumor | 0 | 0 | 0 | specificity | 0.000000 |
| NonTumor | 0 | 0 | 0 | specificity | 0.000000 |

## Per-image breakdown

| run_tag | image_id | polygons(+/-tumor/non) | pixels(+/-tumor/non) | tile labels(P/N/I) | ignored tiles | false_positive_px | false_negative_px | precision | sensitivity | f1 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Images needing attention

- none

## Skipped runs

- run_tag=pfz083_shared_smoke__pf0213: missing required directories: tile_model_dir
- run_tag=pfz083_shared_smoke__pf0229: missing required directories: tile_model_dir
