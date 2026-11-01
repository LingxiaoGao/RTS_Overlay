import json
import tempfile
import unittest
from pathlib import Path

from aoe4.smart_controller_config import SmartControllerConfig


class SmartControllerConfigFromFileTests(unittest.TestCase):
    def _write_config(self, data):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_config_loads_expected_values(self):
        data = {
            "capture_region": {"left": 0, "top": 0, "width": 100, "height": 100},
            "hud_regions": {
                "food": {"left": 10, "top": 10, "width": 20, "height": 20}
            },
            "profile_validated": True,
            "age_signal_validated": True,
        }
        path = self._write_config(data)

        config = SmartControllerConfig.from_file(path)

        self.assertEqual(config.capture_region.left, 0)
        self.assertEqual(config.capture_region.top, 0)
        self.assertEqual(config.capture_region.width, 100)
        self.assertEqual(config.capture_region.height, 100)
        self.assertEqual(config.hud_regions.regions["food"].left, 10)
        self.assertEqual(config.hud_regions.regions["food"].top, 10)
        self.assertEqual(config.hud_regions.regions["food"].width, 20)
        self.assertEqual(config.hud_regions.regions["food"].height, 20)
        self.assertTrue(config.profile_validated)
        self.assertTrue(config.age_signal_validated)
        self.assertIsNone(config.civilization_detection)

    def test_civilization_regions_load_when_contained(self):
        data = {
            "capture_region": {"left": 0, "top": 0, "width": 100, "height": 100},
            "hud_regions": {
                "food": {"left": 10, "top": 10, "width": 20, "height": 20}
            },
            "civilization_detection": {
                "flag": {"left": 30, "top": 30, "width": 20, "height": 10},
                "production_buttons": {"left": 60, "top": 40, "width": 20, "height": 20},
            },
        }
        config = SmartControllerConfig.from_file(self._write_config(data))
        self.assertEqual(config.civilization_detection.flag.left, 30)
        self.assertEqual(config.civilization_detection.production_buttons.width, 20)

    def test_civilization_region_beyond_capture_bounds_is_rejected(self):
        data = {
            "capture_region": {"left": 0, "top": 0, "width": 100, "height": 100},
            "hud_regions": {
                "food": {"left": 10, "top": 10, "width": 20, "height": 20}
            },
            "civilization_detection": {
                "flag": {"left": 90, "top": 30, "width": 20, "height": 10},
                "production_buttons": {"left": 60, "top": 40, "width": 20, "height": 20},
            },
        }
        with self.assertRaises(ValueError):
            SmartControllerConfig.from_file(self._write_config(data))

    def test_malformed_json_raises_value_error(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.json"
        path.write_text("{not valid json", encoding="utf-8")

        with self.assertRaises(ValueError):
            SmartControllerConfig.from_file(path)

    def test_unknown_hud_field_raises_value_error(self):
        data = {
            "capture_region": {"left": 0, "top": 0, "width": 100, "height": 100},
            "hud_regions": {
                "food": {
                    "left": 10,
                    "top": 10,
                    "width": 20,
                    "height": 20,
                    "bogus": 5,
                }
            },
        }
        path = self._write_config(data)

        with self.assertRaises(ValueError):
            SmartControllerConfig.from_file(path)

    def test_negative_capture_left_raises_value_error(self):
        data = {
            "capture_region": {"left": -1, "top": 0, "width": 100, "height": 100},
            "hud_regions": {},
        }
        path = self._write_config(data)

        with self.assertRaises(ValueError):
            SmartControllerConfig.from_file(path)

    def test_food_region_beyond_capture_bounds_raises_value_error(self):
        data = {
            "capture_region": {"left": 0, "top": 0, "width": 100, "height": 100},
            "hud_regions": {
                "food": {"left": 90, "top": 90, "width": 20, "height": 20}
            },
        }
        path = self._write_config(data)

        with self.assertRaises(ValueError):
            SmartControllerConfig.from_file(path)


if __name__ == "__main__":
    unittest.main()
