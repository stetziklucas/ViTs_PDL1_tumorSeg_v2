# Stage 1 Run Summary

image_id: PFZ083 MOS634-PD ML1610229 20161010 PD-L1
run_tag: pfz083_gui_20260429_201659
final_status: SUCCESS
started_at_utc: 2026-04-29T20:18:06+00:00
ended_at_utc: 2026-04-29T20:19:28+00:00
elapsed_seconds: 81.75

## Readiness at launch
- status: READY (Ready)
- summary: Annotations are sufficient to continue with Stage 1 training/inference steps.
- next action: You can continue to downstream Stage 1 CLI steps.

## Step outcomes

| step | exit_code | elapsed_seconds |
| --- | ---: | ---: |
| extract_tiles | 0 | 8.14 |
| embed_vit | 0 | 17.47 |
| make_tile_labels | 0 | 5.30 |
| train_tile_head | 0 | 3.78 |
| train_pixel_classifier | 0 | 7.38 |
| run_inference | 0 | 24.08 |
| make_report | 0 | 10.07 |

## Next review files
- `outputs/reports_pfz083_gui_20260429_201659/report_summary.md`
- `outputs/reports_pfz083_gui_20260429_201659/report_summary.json`
- `outputs/reports_pfz083_gui_20260429_201659/one_page_report.pdf`
- `outputs/reports_pfz083_gui_20260429_201659/stage1_runner.log`
