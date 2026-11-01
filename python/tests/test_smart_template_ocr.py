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

    def test_scaled_crop_matches_reference_and_reuses_result(self):
        import aoe4.smart_template_ocr as ocr_module

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "digits" / "1"
            directory.mkdir(parents=True)
            cv2.imwrite(
                str(directory / "sample.png"),
                np.full((20, 4), 255, dtype=np.uint8),
            )

            reference_region = CaptureRegion(0, 0, 50, 28)
            active_region = CaptureRegion(0, 0, 150, 84)
            reference_bgr = np.zeros((28, 50, 3), dtype=np.uint8)
            reference_bgr[4:24, 15:19] = 255
            reference_bgra = np.dstack(
                (reference_bgr, np.full((28, 50), 255, dtype=np.uint8))
            )
            active_bgr = cv2.resize(
                reference_bgr, (150, 84), interpolation=cv2.INTER_NEAREST
            )
            active_bgra = np.dstack(
                (active_bgr, np.full((84, 150), 255, dtype=np.uint8))
            )

            reference = TemplateHudRecognizer(
                HudRegionSet({"food": reference_region}), root
            )
            active = TemplateHudRecognizer(
                HudRegionSet({"food": active_region}),
                root,
                reference_regions=HudRegionSet({"food": reference_region}),
            )
            reference_frame = CapturedFrame(
                frame_id=1,
                captured_at=0.0,
                width=50,
                height=28,
                pixels=reference_bgra.tobytes(),
                region=reference_region,
            )
            first_active_frame = CapturedFrame(
                frame_id=2,
                captured_at=0.0,
                width=150,
                height=84,
                pixels=active_bgra.tobytes(),
                region=active_region,
            )
            second_active_frame = CapturedFrame(
                frame_id=3,
                captured_at=0.5,
                width=150,
                height=84,
                pixels=active_bgra.tobytes(),
                region=active_region,
            )

            reference_result = reference.recognize(reference_frame)
            with mock.patch(
                "aoe4.smart_template_ocr._extract_glyphs",
                wraps=ocr_module._extract_glyphs,
            ) as extract:
                first = active.recognize(first_active_frame)
                second = active.recognize(second_active_frame)

            self.assertEqual(reference_result[0].text, "1")
            self.assertEqual(first[0].text, "1")
            self.assertEqual(second[0].text, "1")
            self.assertEqual(extract.call_count, 1)


if __name__ == "__main__":
    unittest.main()
