# Stage 1 Single-Image Development Report

image_id: PFZ083 MOS634-PD ML1610229 20161010 PD-L1
run_tag: pf0229_accept

Note: Metrics below are computed on annotated training regions only, not whole-slide validation.
Ignore/Unlabeled pixels are excluded.

## Development metrics

| metric | value |
| --- | ---: |
| false_positive_px | 0 |
| false_negative_px | 505672 |
| precision | 1.000000 |
| sensitivity | 0.540990 |
| f1 | 0.702133 |
| training_log_loss_total | 34294.308594 |

## Additional machine-readable counts

- tp_px: 595987
- tn_px: 1648169
- annotated_positive_px: 1101659
- annotated_negative_px: 1648169
- annotated_total_px: 2749828
- training_log_loss_mean: 0.012471

## Working-space caveat

- All masks/maps/overlay are exported in working image space (selected SVS level when applicable).
