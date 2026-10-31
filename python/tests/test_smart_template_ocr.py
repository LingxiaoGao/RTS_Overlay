import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from aoe4.smart_capture import CapturedFrame, CaptureRegion
from aoe4.smart_regions import HudRegionSet
from aoe4.smart_template_ocr import TemplateHudRecognizer


REGION = CaptureRegion(0, 0, 4, 4)


def make_frame(frame_id, captured_at, fill):
    pixel = bytes((fill, fill, fill, 255))
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=captured_at,
        width=4,
        height=4,
        pixels=pixel * 16,
        region=REGION,
    )


class TemplateHudRecognizerCacheTests(unittest.TestCase):
    def make_recognizer(self, root, field, kind, label):
        directory = Path(root) / kind / label
        directory.mkdir(parents=True)
        image = np.zeros((8, 4), dtype=np.uint8)
        image[:, 1:3] = 255
        cv2.imwrite(str(directory / "sample.png"), image)
        return TemplateHudRecognizer(HudRegionSet({field: REGION}), root)

    def test_identical_crop_reuses_previous_result(self):
        with tempfile.TemporaryDirectory() as root:
            recognizer = self.make_recognizer(root, "food", "digits", "1")
            glyph = np.full((8, 2), 255, dtype=np.uint8)
            with mock.patch(
                "aoe4.smart_template_ocr._extract_glyphs",
                return_value=[glyph],
            ) as extract:
                first = recognizer.recognize(make_frame(1, 0.0, 100))
                second = recognizer.recognize(make_frame(2, 0.5, 100))
            self.assertEqual(first, second)
            self.assertEqual(extract.call_count, 1)

    def test_age_reuses_result_for_one_second(self):
        with tempfile.TemporaryDirectory() as root:
            recognizer = self.make_recognizer(root, "age", "roman", "I")
            glyph = np.full((8, 2), 255, dtype=np.uint8)
            with mock.patch(
                "aoe4.smart_template_ocr._extract_glyphs",
                return_value=[glyph],
            ) as extract:
                first = recognizer.recognize(make_frame(1, 0.0, 100))
                second = recognizer.recognize(make_frame(2, 0.5, 110))
                recognizer.recognize(make_frame(3, 1.0, 120))
            self.assertEqual(first, second)
            self.assertEqual(extract.call_count, 2)


if __name__ == "__main__":
    unittest.main()
