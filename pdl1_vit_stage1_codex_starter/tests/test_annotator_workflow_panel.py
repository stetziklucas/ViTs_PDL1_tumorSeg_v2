import json
import unittest
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import numpy as np
from apps.annotator import default_project_tag, build_stage1_project_command, compact_path_label, resolve_verification_overlay_path, resolve_verification_annotation_labels_path, resolve_verification_prediction_labels_path, verification_overlay_translate, verification_mask_layer_kwargs, build_polygon_review_face_colors
from apps.verification_results_viewer import load_verification_regions, filter_verification_regions, sort_verification_regions, verification_region_label, viewer_bbox_from_region

class AnnotatorWorkflowPanelTests(unittest.TestCase):
    def test_auto_generated_project_tag(self):
        self.assertEqual(default_project_tag(datetime(2026,1,2,3,4,5,tzinfo=timezone.utc)), 'training_20260102_030405')
    def test_project_runner_command(self):
        cmd=build_stage1_project_command(config_path=Path('config/base.yaml'), project_tag='training_x', raw_dir=Path('data/raw'), annotations_dir=Path('data/annotations'), outputs_root=Path('outputs'), models_root=Path('models'))
        self.assertIn('scripts/run_stage1_project.py', ' '.join(cmd))
    def test_verification_helpers(self):
        regions=[{'class_name':'Positive_Tumor','issue':'x','review_priority':3,'score_name':'sensitivity','score':0.5,'error_px':2,'bbox_annotation_yxhw':[1,2,3,4],'center_annotation_yx':[2,3]},{'class_name':'NonTumor','issue':'y','review_priority':1,'score_name':'specificity','score':1.0,'error_px':0,'bbox_annotation_yxhw':[3,4,5,6]}]
        self.assertEqual(len(filter_verification_regions(regions,'Positive_Tumor','All')),1)
        self.assertEqual(sort_verification_regions(regions,'review_priority')[0]['review_priority'],3)
        self.assertIn('Positive_Tumor', verification_region_label(regions[0]))
        self.assertEqual(viewer_bbox_from_region(regions[0])['x'],2)
    def test_load_regions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'verification_regions.json'; p.write_text(json.dumps({'regions':[{'class_name':'A'}]}))
            self.assertEqual(len(load_verification_regions(p)),1)

if __name__=='__main__':
    unittest.main()
