# Stage 1 Project Run Summary

project_tag: training_20260430_000442
final_status: SUCCESS
started_at_utc: 2026-04-30T00:04:43+00:00
ended_at_utc: 2026-04-30T00:07:59+00:00
elapsed_seconds: 195.87

## Requested cases
- pfz083_mos634-pd_ml1610213_20161010_pd-l1=PFZ083 MOS634-PD ML1610213 20161010 PD-L1
- pfz083_mos634-pd_ml1610229_20161010_pd-l1=PFZ083 MOS634-PD ML1610229 20161010 PD-L1

## Included READY cases
- pfz083_mos634-pd_ml1610213_20161010_pd-l1=PFZ083 MOS634-PD ML1610213 20161010 PD-L1
- pfz083_mos634-pd_ml1610229_20161010_pd-l1=PFZ083 MOS634-PD ML1610229 20161010 PD-L1

## Skipped cases
- none

## Shared model artifacts
- tile_model: `models/tile_head_training_20260430_000442_shared/tile_head.pkl`
- tile_metrics: `models/tile_head_training_20260430_000442_shared/tile_cv_metrics.json`
- pixel_model: `models/pixel_classifier_training_20260430_000442_shared/pixel_model.pkl`
- pixel_feature_spec: `models/pixel_classifier_training_20260430_000442_shared/pixel_feature_spec.json`

## Step outcomes

| step | exit_code | elapsed_seconds |
| --- | ---: | ---: |
| pfz083_mos634-pd_ml1610213_20161010_pd-l1:extract_tiles | 0 | 6.68 |
| pfz083_mos634-pd_ml1610213_20161010_pd-l1:embed_vit | 0 | 11.94 |
| pfz083_mos634-pd_ml1610213_20161010_pd-l1:make_tile_labels | 0 | 4.38 |
| pfz083_mos634-pd_ml1610229_20161010_pd-l1:extract_tiles | 0 | 10.13 |
| pfz083_mos634-pd_ml1610229_20161010_pd-l1:embed_vit | 0 | 18.04 |
| pfz083_mos634-pd_ml1610229_20161010_pd-l1:make_tile_labels | 0 | 6.68 |
| shared_train_tile_head | 0 | 5.41 |
| shared_train_pixel_classifier | 0 | 13.89 |
| pfz083_mos634-pd_ml1610213_20161010_pd-l1:run_inference | 0 | 30.32 |
| pfz083_mos634-pd_ml1610213_20161010_pd-l1:make_report | 0 | 10.19 |
| pfz083_mos634-pd_ml1610229_20161010_pd-l1:run_inference | 0 | 42.18 |
| pfz083_mos634-pd_ml1610229_20161010_pd-l1:make_report | 0 | 13.28 |
| make_project_report | 0 | 0.70 |

## Next review files
- `outputs/reports_training_training_20260430_000442/training_summary.md`
- `outputs/reports_training_training_20260430_000442/training_summary.json`
- `outputs/reports_training_training_20260430_000442/stage1_project_run_summary.md`
- `outputs/reports_training_training_20260430_000442/stage1_project_run_summary.json`
- `outputs/reports_training_training_20260430_000442/stage1_project_runner.log`
