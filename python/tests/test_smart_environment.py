import json
import tempfile
import unittest
from pathlib import Path

from aoe4.smart_environment import (
    REQUIRED_MODULES,
    DependencyStatus,
    EnvironmentReport,
    EnvironmentValidator,
)


def _valid_config_data():
    return {
        "capture_region": {"left": 0, "top": 0, "width": 100, "height": 100},
        "hud_regions": {
            "food": {"left": 10, "top": 10, "width": 20, "height": 20}
        },
        "profile_validated": True,
        "age_signal_validated": True,
    }


class EnvironmentValidatorTests(unittest.TestCase):
    def _write_config(self, data):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _all_available(self, name):
        return object()

    def test_all_ready(self):
        path = self._write_config(_valid_config_data())
        report = EnvironmentValidator(self._all_available).validate(path)

        self.assertTrue(report.ready)
        self.assertTrue(report.config_available)
        self.assertIsNone(report.config_error)
        self.assertTrue(all(dep.available for dep in report.dependencies))

    def test_one_missing_dependency(self):
        path = self._write_config(_valid_config_data())

        def finder(name):
            if name == "mss":
                return None
            return object()

        report = EnvironmentValidator(finder).validate(path)

        self.assertFalse(report.ready)
        self.assertTrue(report.config_available)
        unavailable = [d.module for d in report.dependencies if not d.available]
        self.assertEqual(unavailable, ["mss"])

    def test_finder_exception_is_isolated(self):
        path = self._write_config(_valid_config_data())

        def finder(name):
            if name == "cv2":
                raise RuntimeError("boom")
            return object()

        report = EnvironmentValidator(finder).validate(path)

        self.assertFalse(report.ready)
        by_name = {d.module: d.available for d in report.dependencies}
        self.assertFalse(by_name["cv2"])
        self.assertTrue(by_name["PyQt5"])

    def test_missing_config(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        missing = Path(tmpdir.name) / "absent.json"

        report = EnvironmentValidator(self._all_available).validate(missing)

        self.assertFalse(report.ready)
        self.assertFalse(report.config_available)
        self.assertIsNotNone(report.config_error)

    def test_malformed_config(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.json"
        path.write_text("{not valid json", encoding="utf-8")

        report = EnvironmentValidator(self._all_available).validate(path)

        self.assertFalse(report.ready)
        self.assertFalse(report.config_available)
        self.assertIsNotNone(report.config_error)
        self.assertIn("ValueError", report.config_error)

    def test_invalid_finder_not_callable(self):
        with self.assertRaises(TypeError):
            EnvironmentValidator(module_finder=None)

    def test_deterministic_ordering(self):
        path = self._write_config(_valid_config_data())
        report = EnvironmentValidator(self._all_available).validate(path)

        self.assertEqual(
            [dep.module for dep in report.dependencies],
            list(REQUIRED_MODULES),
        )
        self.assertEqual(
            list(REQUIRED_MODULES),
            ["PyQt5", "mss", "cv2", "numpy"],
        )

    def test_report_is_immutable(self):
        report = EnvironmentReport(
            dependencies=(DependencyStatus(module="mss", available=True),),
            config_available=True,
            config_error=None,
        )
        with self.assertRaises(Exception):
            report.config_available = False
        with self.assertRaises(Exception):
            report.dependencies[0].available = False

    def test_chinese_messages_when_ready(self):
        path = self._write_config(_valid_config_data())
        report = EnvironmentValidator(self._all_available).validate(path)

        messages = report.messages_zh
        self.assertIsInstance(messages, tuple)
        self.assertTrue(any("环境已就绪" in line for line in messages))

    def test_chinese_messages_report_missing_and_config(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)

        def finder(name):
            if name == "cv2":
                return None
            return object()

        report = EnvironmentValidator(finder).validate(
            Path(tmpdir.name) / "absent.json"
        )
        messages = report.messages_zh

        self.assertIsInstance(messages, tuple)
        joined = "\n".join(messages)
        self.assertIn("cv2", joined)
        self.assertIn("配置文件", joined)


if __name__ == "__main__":
    unittest.main()
