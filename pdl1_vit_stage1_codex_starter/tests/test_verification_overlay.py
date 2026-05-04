import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from verification_overlay import generate_verification_overlay


class VerificationOverlayTests(unittest.TestCase):
    def test_overlay_is_cropped_to_annotated_bbox_with_padding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scribble = np.zeros((10, 12), dtype=np.uint8)
            scribble[2:5, 3:7] = 1
            pred = np.zeros((10, 12), dtype=np.uint8)
            pred[3:4, 4:6] = 255
            s = root / "scribble_labels.png"
            p = root / "positive_mask.png"
            Image.fromarray(scribble).save(s)
            Image.fromarray(pred).save(p)
            out = generate_verification_overlay(
                image_id="IMG",
                run_tag="r1",
                scribble_labels_path=s,
                positive_mask_path=p,
                output_dir=root,
                label_encoding={"Positive_Tumor": 1, "Negative_Tumor": 2, "NonTumor": 3, "Ignore": 4},
                crop_padding_px=2,
            )
            overlay = np.asarray(Image.open(root / "verification_overlay.png"))
            self.assertEqual(overlay.shape, (7, 8))
            self.assertEqual(out["verification_overlay_mode"], "positive_mask_working_crop")
            self.assertEqual(out["crop_y0"], 0)
            self.assertEqual(out["crop_x0"], 1)
            self.assertEqual(out["crop_h"], 7)
            self.assertEqual(out["crop_w"], 8)
            self.assertGreater(int(np.count_nonzero(overlay)), 0)
            ann = np.asarray(Image.open(root / "verification_annotation_labels.png"))
            pred_labels = np.asarray(Image.open(root / "verification_prediction_labels.png"))
            self.assertEqual(ann.shape, overlay.shape)
            self.assertEqual(pred_labels.shape, overlay.shape)
            self.assertEqual(out["verification_annotation_labels_available"], True)
            self.assertEqual(out["verification_prediction_labels_available"], True)
            self.assertIn(1, set(np.unique(pred_labels).tolist()))
            self.assertEqual(out["prediction_label_mapping"]["pred_on_positive_tumor"], 1)
            self.assertTrue(str(out["verification_annotation_labels_path"]).endswith("verification_annotation_labels.png"))
            self.assertEqual(out["annotation_label_mapping"]["Positive_Tumor"], 1)

if __name__ == '__main__':
    unittest.main()
