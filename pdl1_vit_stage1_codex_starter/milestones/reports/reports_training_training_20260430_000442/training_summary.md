# Stage 1 Training Development Summary

Note: Metrics are computed on annotated training regions only (scribble labels),
not on whole-slide unlabeled validation regions. Ignore/Unlabeled pixels are excluded.

## Aggregate project summary

| metric | value |
| --- | ---: |
| false_positive_px | 1 |
| false_negative_px | 2271 |
| precision | 0.999843 |
| sensitivity | 0.736970 |
| f1 | 0.848513 |
| training_log_loss_total | 54.455385 |

## Aggregate class-specific annotated-region metrics

| class | annotated_px | primary_correct_px | primary_error_px | score_name | score |
| --- | ---: | ---: | ---: | --- | ---: |
| Positive_Tumor | 8634 | 6363 | 2271 | sensitivity | 0.736970 |
| Negative_Tumor | 2164 | 2163 | 1 | specificity | 0.999538 |
| NonTumor | 5158 | 5158 | 0 | specificity | 1.000000 |

## Per-image breakdown

| run_tag | image_id | polygons(+/-tumor/non) | pixels(+/-tumor/non) | tile labels(P/N/I) | ignored tiles | false_positive_px | false_negative_px | precision | sensitivity | f1 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training_20260430_000442__pfz083_mos634-pd_ml1610213_20161010_pd-l1 | PFZ083 MOS634-PD ML1610213 20161010 PD-L1 | 6/6/5 | 388814/86907/218676 | 6/0/0 | 0 | 0 | 1891 | 1.000000 | 0.687954 | 0.815133 |
| training_20260430_000442__pfz083_mos634-pd_ml1610229_20161010_pd-l1 | PFZ083 MOS634-PD ML1610229 20161010 PD-L1 | 3/3/3 | 165251/52264/111043 | 6/1/4 | 4 | 1 | 380 | 0.999544 | 0.852370 | 0.920109 |

## Images needing attention

- training_20260430_000442__pfz083_mos634-pd_ml1610213_20161010_pd-l1 | PFZ083 MOS634-PD ML1610213 20161010 PD-L1: very low usable negative tile support
- training_20260430_000442__pfz083_mos634-pd_ml1610229_20161010_pd-l1 | PFZ083 MOS634-PD ML1610229 20161010 PD-L1: very low usable negative tile support

## Skipped runs

- none
