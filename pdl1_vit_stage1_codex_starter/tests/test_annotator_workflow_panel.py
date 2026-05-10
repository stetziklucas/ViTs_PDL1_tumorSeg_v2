import json
import unittest
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import numpy as np
from apps.annotator import default_project_tag, build_stage1_project_command, compact_path_label, resolve_verification_overlay_path, resolve_verification_annotation_labels_path, resolve_verification_prediction_labels_path, verification_overlay_translate, verification_mask_layer_kwargs, build_polygon_review_face_colors, qt_orientation, build_verification_label_layer_kwargs, launch_napari_app
from apps.verification_results_viewer import load_verification_regions, load_verification_regions_payload, filter_verification_regions, sort_verification_regions, verification_region_label, viewer_bbox_from_region, verification_regions_message, resolve_verification_regions_path, rectangle_vertices_from_bbox_yxhw, resolve_region_image_path, label_layer_transform_from_working_crop, build_label_layer_transform_from_entry_or_payload, compute_jump_zoom, get_layer_by_name, get_display_image_shape_hw, canvas_size_wh, set_camera_center_yx

class AnnotatorWorkflowPanelTests(unittest.TestCase):
    def test_auto_generated_project_tag(self):
        self.assertEqual(default_project_tag(datetime(2026,1,2,3,4,5,tzinfo=timezone.utc)), 'training_20260102_030405')
    def test_project_runner_command(self):
        cmd=build_stage1_project_command(config_path=Path('config/base.yaml'), project_tag='training_x', raw_dir=Path('data/raw'), annotations_dir=Path('data/annotations'), outputs_root=Path('outputs'), models_root=Path('models'))
        self.assertIn('scripts/run_stage1_project.py', ' '.join(cmd))
    def test_verification_helpers(self):
        regions=[{'class_name':'Positive_Tumor','issue':'x','review_priority':3,'score_name':'sensitivity','score':0.5,'error_px':2,'bbox_annotation_yxhw':[1,2,3,4],'bbox_working_yxhw':[1,2,3,4],'center_annotation_yx':[2,3]},{'class_name':'NonTumor','issue':'y','review_priority':1,'score_name':'specificity','score':1.0,'error_px':0,'bbox_annotation_yxhw':[3,4,5,6]}]
        self.assertEqual(len(filter_verification_regions(regions,'Positive_Tumor','All')),1)
        self.assertEqual(sort_verification_regions(regions,'Highest error first')[0]['review_priority'],3)
        self.assertIn('Positive_Tumor', verification_region_label(regions[0]))
        self.assertIn('src=', verification_region_label(regions[0]))
        self.assertEqual(viewer_bbox_from_region(regions[0])['x'],2)
    def test_load_regions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'verification_regions.json'; p.write_text(json.dumps({'regions':[{'class_name':'A'}], 'working_shape_hw':[10,20]}))
            self.assertEqual(len(load_verification_regions(p)),1)
            self.assertEqual(load_verification_regions_payload(p).get("working_shape_hw"), [10,20])
    def test_regions_path_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rp=root/'reports'/'report_summary.md'; rp.parent.mkdir(); rp.write_text('x')
            p=root/'reports'/'verification_regions.json'; p.write_text('[]')
            resolved,_=resolve_verification_regions_path(verification_regions_path='verification_regions.json', report_path=rp, repo_root=root)
            self.assertEqual(resolved, p)
    def test_transform_helpers(self):
        tfm=label_layer_transform_from_working_crop((100,200),(200,400),(10,20))
        self.assertEqual(tuple(round(v,2) for v in tfm['translate']),(20.0,40.0))
        verts=rectangle_vertices_from_bbox_yxhw([1,2,3,4])
        self.assertEqual(verts[0],[1.0,2.0])

    def test_preview_path_resolution_basename(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rp=root/'verification_regions.json'; rp.write_text('{}')
            (root/'verification_regions').mkdir(); img=root/'verification_regions'/'region_0000_preview.png'; img.write_bytes(b'x')
            resolved,_=resolve_region_image_path(image_path='region_0000_preview.png', regions_json_path=rp, repo_root=root)
            self.assertEqual(resolved, img)


    def test_label_layer_kwargs_shared_transform(self):
        entry={"working_shape_hw":[100,200],"crop_y0":10,"crop_x0":20}
        pred,ann,warn=build_verification_label_layer_kwargs(entry,(200,400))
        self.assertIsNone(warn)
        self.assertEqual(pred["scale"], ann["scale"])
        self.assertEqual(pred["translate"], ann["translate"])

    def test_label_layer_kwargs_missing_metadata(self):
        pred,ann,warn=build_verification_label_layer_kwargs({},(200,400))
        self.assertIsNone(pred)
        self.assertIn("Cannot place verification labels", warn)

    def test_label_transform_falls_back_to_payload(self):
        tfm=build_label_layer_transform_from_entry_or_payload({}, {"working_shape_hw":[100,200],"crop_origin_working_yx":[10,20]}, (200,400))
        self.assertIsNone(tfm.get("warning"))
        self.assertEqual(tfm["scale"], (2.0,2.0))
        self.assertEqual(tfm["translate"], (20.0,40.0))

    def test_jump_zoom_clamp(self):
        z=compute_jump_zoom([0,0,2,2], canvas_shape_wh=(4096,4096), current_zoom=10.0)
        self.assertLessEqual(z, 0.90)
        z2=compute_jump_zoom([0,0,5000,5000], canvas_shape_wh=(320,240), current_zoom=0.3)
        self.assertGreaterEqual(z2, 0.04)
        self.assertEqual(compute_jump_zoom([0,0,100,100], canvas_shape_wh=None, current_zoom=0.42), 0.42)

    def test_canvas_size_wh_variants(self):
        class QSizeM:
            def width(self): return 800
            def height(self): return 600
        class QSizeA:
            width=700; height=500
        self.assertEqual(canvas_size_wh(type("C",(),{"size":(1024,768)})()), (1024,768))
        self.assertEqual(canvas_size_wh(type("C",(),{"size":[1200,900]})()), (1200,900))
        self.assertEqual(canvas_size_wh(type("C",(),{"size":QSizeM()})()), (800,600))
        self.assertEqual(canvas_size_wh(type("C",(),{"size":QSizeA()})()), (700,500))
        self.assertIsNone(canvas_size_wh(type("C",(),{})()))
        self.assertIsNone(canvas_size_wh(type("C",(),{"size":"bad"})()))

    def test_set_camera_center_preserves_leading_dims(self):
        cam=type("Cam",(),{"center":(0.0, 100.0, 200.0)})()
        viewer=type("V",(),{"camera":cam})()
        out=set_camera_center_yx(viewer, [10.5,20.5])
        self.assertEqual(out, (0.0, 10.5, 20.5))
        self.assertEqual(viewer.camera.center, (0.0, 10.5, 20.5))

    def test_no_local_qt_shadowing_in_viewer_open(self):
        import inspect
        src=inspect.getsource(launch_napari_app)
        open_idx = src.find("def _open_verification_results_viewer")
        self.assertNotEqual(open_idx, -1)
        open_src = src[open_idx:]
        self.assertNotIn("from qtpy.QtWidgets import QLabel", open_src)
        self.assertNotIn("from qtpy.QtGui import QPixmap", open_src)
        self.assertNotIn("getattr(canvas, \"size\", (0, 0)).width()", open_src)

    def test_get_layer_helpers_without_get(self):
        class L: 
            def __init__(self,name,data): self.name=name; self.data=data
        class Layers(list):
            def __getitem__(self,k):
                if isinstance(k,str):
                    for i in self:
                        if i.name==k: return i
                    raise KeyError(k)
                return super().__getitem__(k)
        layers=Layers([L('other',np.zeros((10,10))),L('image',np.zeros((11,12,3)))])
        self.assertEqual(get_layer_by_name(layers,'image').name,'image')
        viewer=type('V',(),{'layers':layers})()
        self.assertEqual(get_display_image_shape_hw(viewer),(11,12))
if __name__=='__main__':
    unittest.main()


class QtCompatTests(unittest.TestCase):
    def test_qt_orientation_compat(self):
        class Q1: Horizontal=object(); Vertical=object()
        class Qt1: Orientation=Q1
        class Qt2: Horizontal=object(); Vertical=object()
        self.assertIs(qt_orientation(Qt1, 'Horizontal'), Qt1.Orientation.Horizontal)
        self.assertIs(qt_orientation(Qt2, 'Vertical'), Qt2.Vertical)

    def test_zero_region_message(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'verification_regions.json'; p.write_text(json.dumps({'region_count':0,'regions':[]}))
            self.assertIn('No verification review regions were generated', verification_regions_message(p))
