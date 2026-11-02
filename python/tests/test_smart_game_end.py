from pathlib import Path
import unittest

import cv2
import numpy as np

from aoe4.smart_game_end import GameEndDetector, has_game_end_banner


FIXTURES = Path(__file__).parent / "fixtures" / "smart_session"


class GameEndDetectorTests(unittest.TestCase):
    @staticmethod
    def fixture(name):
        image = cv2.imread(str(FIXTURES / name), cv2.IMREAD_COLOR)
        if image is None:
            raise AssertionError(f"missing fixture: {name}")
        return image

    def test_real_victory_and_defeat_are_positive(self):
        self.assertTrue(has_game_end_banner(self.fixture("victory.webp")))
        self.assertTrue(has_game_end_banner(self.fixture("defeat.webp")))

    def test_real_4v4_loading_screen_is_negative(self):
        self.assertFalse(has_game_end_banner(self.fixture("hre_loading_4v4.webp")))

    def test_two_consecutive_frames_confirm_and_negative_breaks_streak(self):
        detector = GameEndDetector()
        victory = self.fixture("victory.webp")
        loading = self.fixture("hre_loading_4v4.webp")
        self.assertFalse(detector.observe(victory))
        self.assertFalse(detector.observe(loading))
        self.assertFalse(detector.observe(victory))
        self.assertTrue(detector.observe(victory))

    def test_reset_and_malformed_frames(self):
        detector = GameEndDetector()
        victory = self.fixture("victory.webp")
        self.assertFalse(detector.observe(victory))
        detector.reset()
        self.assertFalse(detector.observe(victory))
        self.assertFalse(has_game_end_banner(np.zeros((10, 10), dtype=np.uint8)))
        self.assertFalse(has_game_end_banner(np.zeros((64, 128, 3), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
