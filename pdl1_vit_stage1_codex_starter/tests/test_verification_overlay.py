import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from verification_overlay import generate_verification_overlay

class VerificationOverlayTests(unittest.TestCase):
    def test_regions_generated_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scribble = np.zeros((10, 12), dtype=np.uint8)
            scribble[2:5, 3:7] = 1
            scribble[6:8, 3:7] = 2
            pred = np.zeros((10, 12), dtype=np.uint8)
            pred[3:4, 4:6] = 255
            pred[6:7, 4:5] = 255
            s = root / 'scribble_labels.png'; p = root / 'positive_mask.png'; ov = root / 'overlay.png'
            Image.fromarray(scribble).save(s); Image.fromarray(pred).save(p); Image.fromarray(np.dstack([pred,pred,pred])).save(ov)
            meta = root / 'IMG_annotation_meta.json'
            meta.write_text(json.dumps({'polygons':[{'class_name':'Positive_Tumor','vertices':[[2,3],[2,6],[4,6],[4,3]]},{'class_name':'Negative_Tumor','vertices':[[6,3],[6,6],[7,6],[7,3]]}]}))
            out = generate_verification_overlay(image_id='IMG',run_tag='r1',scribble_labels_path=s,positive_mask_path=p,output_dir=root,label_encoding={'Positive_Tumor':1,'Negative_Tumor':2,'NonTumor':3,'Ignore':4},annotation_metadata_path=meta,overlay_base_path=ov,crop_padding_px=1)
            self.assertTrue(out['verification_regions_available'])
            self.assertEqual(out['verification_region_count'], 2)
            reg = json.loads((root/'verification_regions.json').read_text())['regions']
            self.assertEqual(reg[0]['score_name'],'sensitivity')
            self.assertEqual(reg[1]['score_name'],'specificity')
            self.assertTrue(Path(reg[0]['preview_path']).exists())
            self.assertTrue(Path(reg[0]['thumbnail_path']).exists())

if __name__ == '__main__':
    unittest.main()
