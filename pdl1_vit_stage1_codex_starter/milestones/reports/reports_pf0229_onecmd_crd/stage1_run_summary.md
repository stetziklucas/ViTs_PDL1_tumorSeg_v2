# Stage 1 Run Summary

image_id: PFZ083 MOS634-PD ML1610229 20161010 PD-L1
run_tag: pf0229_onecmd_crd
final_status: SUCCESS
started_at_utc: 2026-04-17T23:18:18+00:00
ended_at_utc: 2026-04-17T23:19:49+00:00
elapsed_seconds: 90.57

## Readiness at launch
- status: READY (Ready)
- summary: Annotations are sufficient to continue with Stage 1 training/inference steps.
- next action: You can continue to downstream Stage 1 CLI steps.

## Step outcomes

| step | exit_code | elapsed_seconds |
| --- | ---: | ---: |
| extract_tiles | 0 | 9.44 |
| embed_vit | 0 | 20.68 |
| make_tile_labels | 0 | 5.77 |
| train_tile_head | 0 | 4.58 |
| train_pixel_classifier | 0 | 8.64 |
| run_inference | 0 | 27.38 |
| make_report | 0 | 7.83 |

## Next review files
- `outputs/reports_pf0229_onecmd_crd/report_summary.md`
- `outputs/reports_pf0229_onecmd_crd/report_summary.json`
- `outputs/reports_pf0229_onecmd_crd/one_page_report.pdf`
- `outputs/reports_pf0229_onecmd_crd/stage1_runner.log`
