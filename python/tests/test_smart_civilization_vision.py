import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aoe4.smart_civilization_vision import CivilizationVision


def pattern(color):
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[4:36, 5:20] = color
    image[10:30, 27:46] = tuple(reversed(color))
    image[18:23, :] = (255, 255, 255)
    return image


class CivilizationVisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.english = pattern((20, 30, 220))
        self.french = pattern((220, 40, 20))
        self.villager = pattern((20, 220, 220))
        paths = {}
        for name, image in (("English", self.english), ("French", self.french)):
            path = root / f"{name}.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            paths[name] = path
        self.vision = CivilizationVision(paths)

    def test_best_flag_and_active_production_button(self):
        flag_crop = np.zeros((80, 140, 3), dtype=np.uint8)
        flag_crop[20:60, 45:95] = self.english
        buttons_crop = np.zeros((55, 78, 3), dtype=np.uint8)
        buttons_crop[:, :39] = (80, 100, 120)
        cue, active = self.vision.recognize_in_game(flag_crop, buttons_crop)
        self.assertEqual(cue.civilization, "English")
        self.assertTrue(active)

    def test_dark_or_malformed_crop_is_not_evidence(self):
        dark = np.zeros((80, 140, 3), dtype=np.uint8)
        cue, active = self.vision.recognize_in_game(dark, dark)
        self.assertIsNone(cue)
        self.assertFalse(active)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(self.vision.recognize_in_game(empty, empty), (None, False))

    def test_missing_template_rejected(self):
        with self.assertRaises(ValueError):
            CivilizationVision({"English": "/missing/flag.png"})


if __name__ == "__main__":
    unittest.main()
