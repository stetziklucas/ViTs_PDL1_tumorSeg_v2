# Stage 1 Project Run Summary

project_tag: pfz083_shared_smoke
final_status: SUCCESS
started_at_utc: 2026-04-18T03:00:07+00:00
ended_at_utc: 2026-04-18T03:02:36+00:00
elapsed_seconds: 148.72

## Requested cases
- pf0213=PFZ083 MOS634-PD ML1610213 20161010 PD-L1
- pf0229=PFZ083 MOS634-PD ML1610229 20161010 PD-L1

## Included READY cases
- pf0213=PFZ083 MOS634-PD ML1610213 20161010 PD-L1
- pf0229=PFZ083 MOS634-PD ML1610229 20161010 PD-L1

## Skipped cases
- none

## Shared model artifacts
- tile_model: `models/tile_head_pfz083_shared_smoke_shared/tile_head.pkl`
- tile_metrics: `models/tile_head_pfz083_shared_smoke_shared/tile_cv_metrics.json`
- pixel_model: `models/pixel_classifier_pfz083_shared_smoke_shared/pixel_model.pkl`
- pixel_feature_spec: `models/pixel_classifier_pfz083_shared_smoke_shared/pixel_feature_spec.json`

## Step outcomes

| step | exit_code | elapsed_seconds |
| --- | ---: | ---: |
| pf0213:extract_tiles | 0 | 6.43 |
| pf0213:embed_vit | 0 | 9.06 |
| pf0213:make_tile_labels | 0 | 4.64 |
| pf0229:extract_tiles | 0 | 8.84 |
| pf0229:embed_vit | 0 | 5.79 |
| pf0229:make_tile_labels | 0 | 5.90 |
| shared_train_tile_head | 0 | 3.79 |
| shared_train_pixel_classifier | 0 | 13.32 |
| pf0213:run_inference | 0 | 24.95 |
| pf0213:make_report | 0 | 9.09 |
| pf0229:run_inference | 0 | 33.56 |
| pf0229:make_report | 0 | 10.81 |
| make_project_report | 0 | 0.49 |

## Next review files
- `outputs/reports_training_pfz083_shared_smoke/training_summary.md`
- `outputs/reports_training_pfz083_shared_smoke/training_summary.json`
- `outputs/reports_training_pfz083_shared_smoke/stage1_project_run_summary.md`
- `outputs/reports_training_pfz083_shared_smoke/stage1_project_run_summary.json`
- `outputs/reports_training_pfz083_shared_smoke/stage1_project_runner.log`
