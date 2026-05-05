import json, tempfile, unittest
from pathlib import Path
from project_report_history import discover_project_summaries, discover_current_image_shared_reports, format_project_summary_label, format_current_image_report_label, auto_select_latest_indices

class ProjectReportHistoryTests(unittest.TestCase):
    def test_discovery_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            o=Path(d)/'outputs'; p=o/'reports_training_training_1'; p.mkdir(parents=True)
            (p/'training_summary.json').write_text(json.dumps({'project_tag':'training_1','aggregate_metrics':{'f1':0.8},'ended_at_utc':'2026-01-01T00:00:00+00:00'}))
            (p/'training_summary.md').write_text('# t')
            (p/'stage1_project_cases.json').write_text(json.dumps({'included_ready_cases':[{'alias':'a','image_id':'IMG'}],'skipped_cases':[]}))
            c=o/'reports_training_1__a'; c.mkdir(parents=True)
            (c/'report_summary.json').write_text(json.dumps({'image_id':'IMG','model_scope':'shared_project_model','shared_model_tag':'training_1','development_metrics':{'f1':0.9},'ended_at_utc':'2026-01-01T00:01:00+00:00','verification_overlay_mode':'positive_mask_working_crop','crop_y0':5,'crop_x0':7,'crop_h':20,'crop_w':21,'verification_annotation_labels_available':True,'verification_regions_available':True,'verification_regions_path':'/tmp/regions.json','verification_region_count':2,'verification_annotation_labels_path':'/tmp/ann.png','verification_prediction_labels_available':True,'verification_prediction_labels_path':'/tmp/pred.png'}))
            ov_dir=o/'overlays_training_1__a'; ov_dir.mkdir(parents=True)
            (ov_dir/'verification_overlay.png').write_bytes(b'PNG')
            (c/'report_summary.md').write_text('# c')
            pe=discover_project_summaries(o,'IMG'); ie=discover_current_image_shared_reports('IMG',o)
            self.assertEqual(pe[0]['project_tag'],'training_1'); self.assertTrue(pe[0]['current_image_included'])
            self.assertIn('agg F1 0.800', format_project_summary_label(pe[0]))
            self.assertIn('shared F1 0.900', format_current_image_report_label(ie[0]))
            self.assertTrue(ie[0]['verification_overlay_available'])
            self.assertTrue(str(ie[0]['verification_overlay_path']).endswith('verification_overlay.png'))
            self.assertEqual(ie[0]['verification_overlay_mode'], 'positive_mask_working_crop')
            self.assertEqual(ie[0]['crop_y0'], 5)
            self.assertTrue(ie[0]['verification_annotation_labels_available'])
            self.assertEqual(ie[0]['verification_annotation_labels_path'], '/tmp/ann.png')
            self.assertTrue(ie[0]['verification_prediction_labels_available'])
            self.assertEqual(ie[0]['verification_prediction_labels_path'], '/tmp/pred.png')
            self.assertEqual(ie[0]['verification_region_count'], 2)
            self.assertEqual(ie[0]['verification_regions_path'], '/tmp/regions.json')
            self.assertTrue(str(ie[0]['report_summary_json']).endswith('report_summary.json'))
            self.assertEqual(auto_select_latest_indices(pe, ie),(0,0))

if __name__=='__main__': unittest.main()
