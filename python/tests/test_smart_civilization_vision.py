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
        self.paths = paths
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

    def test_loading_background_matches_template_and_dark_frame_is_ignored(self):
        frame = np.tile(self.french, (8, 10, 1))
        cue = CivilizationVision({"French": self.paths["French"]}).recognize_loading(frame)
        self.assertIsNotNone(cue)
        self.assertEqual(cue.civilization, "French")
        self.assertIsNone(
            self.vision.recognize_loading(np.zeros((240, 480, 3), dtype=np.uint8))
        )

    def test_flag_hash_reuses_stable_result(self):
        flag_crop = np.zeros((80, 140, 3), dtype=np.uint8)
        flag_crop[20:60, 45:95] = self.english
        first, _ = self.vision.recognize_in_game(flag_crop, flag_crop)
        second, _ = self.vision.recognize_in_game(flag_crop.copy(), flag_crop)
        self.assertIs(first, second)

    def test_real_dragon_loading_background_beats_similar_yellow_flags(self):
        repository = Path(__file__).resolve().parents[2]
        flag_dir = repository / "docs" / "assets" / "aoe4" / "civilization_flag"
        vision = CivilizationVision(
            {
                "Order of the Dragon": flag_dir / "dra.webp",
                "Ayyubids": flag_dir / "ayy.webp",
                "Malians": flag_dir / "mal.webp",
                "Holy Roman Empire": flag_dir / "hre.webp",
            }
        )
        frame = cv2.imread(
            str(
                Path(__file__).parent
                / "fixtures"
                / "smart_session"
                / "dragon_loading.webp"
            )
        )
        cue = vision.recognize_loading(frame)
        self.assertIsNotNone(cue)
        self.assertEqual(cue.civilization, "Order of the Dragon")

    def test_real_unit_banners_use_loading_candidate_as_a_bounded_hint(self):
        repository = Path(__file__).resolve().parents[2]
        flag_dir = repository / "docs" / "assets" / "aoe4" / "civilization_flag"
        vision = CivilizationVision(
            {
                "English": flag_dir / "eng.webp",
                "French": flag_dir / "fre.webp",
                "Mongols": flag_dir / "mon.webp",
                "Order of the Dragon": flag_dir / "dra.webp",
                "Holy Roman Empire": flag_dir / "hre.webp",
                "Knights Templar": flag_dir / "kte.webp",
            }
        )
        fixture = Path(__file__).parent / "fixtures" / "smart_civilization"

        def recognize(name, preferred=None):
            flag = cv2.imread(str(fixture / f"{name}_flag.webp"))
            commands = cv2.imread(str(fixture / f"{name}_commands.webp"))
            return vision.recognize_in_game(flag, commands, preferred)[0]

        self.assertEqual(recognize("english").civilization, "English")
        self.assertEqual(
            recognize("french", "French").civilization, "French"
        )
        self.assertEqual(
            recognize("dragon", "Order of the Dragon").civilization,
            "Order of the Dragon",
        )

    def test_bright_background_without_command_panel_is_not_a_unit_banner(self):
        grass = np.full((55, 78, 3), (60, 150, 80), dtype=np.uint8)
        flag = np.full((35, 40, 3), (60, 150, 80), dtype=np.uint8)
        cue, _ = self.vision.recognize_in_game(flag, grass)
        self.assertIsNone(cue)


if __name__ == "__main__":
    unittest.main()
