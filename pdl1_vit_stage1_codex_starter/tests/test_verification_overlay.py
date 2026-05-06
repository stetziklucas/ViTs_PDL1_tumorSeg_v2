import json, tempfile, unittest
from pathlib import Path
import numpy as np
from PIL import Image
from verification_overlay import generate_verification_overlay

class VerificationOverlayTests(unittest.TestCase):
    def _run(self, meta=True):
        d=tempfile.TemporaryDirectory(); root=Path(d.name)
        scribble=np.zeros((20,30),dtype=np.uint8); scribble[2:8,3:9]=1; scribble[10:14,20:26]=2
        pred=np.zeros((20,30),dtype=np.uint8); pred[3:6,4:8]=255; pred[11:12,21:22]=255
        s=root/'s.png'; p=root/'p.png'; ov=root/'ov.png'
        Image.fromarray(scribble).save(s); Image.fromarray(pred).save(p); Image.fromarray(np.dstack([pred,pred,pred])).save(ov)
        mp=root/'m.json'
        if meta: mp.write_text(json.dumps({'polygons':[{'class_name':'Positive_Tumor','vertices':[[2,3],[2,8],[7,8],[7,3]]},{'class_name':'Negative_Tumor','vertices':[[10,20],[10,25],[13,25],[13,20]]}]}))
        out=generate_verification_overlay(image_id='IMG',run_tag='r',scribble_labels_path=s,positive_mask_path=p,output_dir=root,label_encoding={'Positive_Tumor':1,'Negative_Tumor':2,'NonTumor':3,'Ignore':4},annotation_metadata_path=mp if meta else None,overlay_base_path=ov,crop_padding_px=0)
        return d,root,out

    def test_regions_and_metrics(self):
        d,root,out=self._run(True)
        self.assertGreater(out['verification_region_count'],0)
        reg=json.loads((root/'verification_regions.json').read_text())['regions']
        pos=[r for r in reg if r['class_name']=='Positive_Tumor'][0]; neg=[r for r in reg if r['class_name']=='Negative_Tumor'][0]
        self.assertEqual(pos['score_name'],'sensitivity'); self.assertEqual(pos['error_px'], pos['annotated_px']-pos['correct_px'])
        self.assertEqual(neg['score_name'],'specificity'); self.assertEqual(neg['error_px'], neg['pred_positive_px'])
        self.assertTrue(Path(pos['preview_path']).exists()); self.assertTrue(Path(pos['thumbnail_path']).exists())
        d.cleanup()

    def test_fallback_components(self):
        d,root,out=self._run(False)
        self.assertGreater(out['verification_region_count'],0)
        reg=json.loads((root/'verification_regions.json').read_text())['regions']
        self.assertTrue(all(r['source_type']=='annotation_component' for r in reg))
        d.cleanup()

if __name__=='__main__': unittest.main()
