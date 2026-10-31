"""Tests for the MSS-backed screen capture adapter."""

from __future__ import annotations

import unittest

from aoe4.smart_capture import CaptureRegion, CapturedFrame
from aoe4.smart_screen_capture import MssScreenCapture


class FakeShot:
    """Minimal stand-in for an MSS screenshot object."""

    def __init__(self, width: int, height: int, bgra: object) -> None:
        self.width = width
        self.height = height
        self.bgra = bgra


class FakeBackend:
    """Fake backend recording grab calls and scripted results."""

    def __init__(self, results: object) -> None:
        if isinstance(results, list):
            results = list(results)
        else:
            results = [results]
        self._results = results
        self.calls: list[dict] = []
        self.close_count = 0

    def grab(self, monitor: dict) -> object:
        self.calls.append(dict(monitor))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.close_count += 1


class FakeFactory:
    """Callable factory producing fresh backends and counting invocations."""

    def __init__(self, backends: list) -> None:
        self._backends = list(backends)
        self.call_count = 0

    def __call__(self) -> object:
        self.call_count += 1
        return self._backends.pop(0)


REGION = CaptureRegion(left=10, top=20, width=2, height=3)
PIXELS = bytes(range(REGION.width * REGION.height * 4))


def make_shot(bgra: object = PIXELS) -> FakeShot:
    return FakeShot(REGION.width, REGION.height, bgra)


class LazyFactoryTests(unittest.TestCase):
    def test_factory_not_called_until_capture_and_reused(self) -> None:
        backend = FakeBackend([make_shot(), make_shot()])
        factory = FakeFactory([backend])
        capture = MssScreenCapture(factory)

        self.assertEqual(factory.call_count, 0)

        capture.capture(REGION, 1, 0.5)
        self.assertEqual(factory.call_count, 1)

        capture.capture(REGION, 2, 1.5)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(backend.calls), 2)

    def test_non_callable_factory_rejected(self) -> None:
        with self.assertRaises(TypeError):
            MssScreenCapture(backend_factory=object())


class MonitorDictionaryTests(unittest.TestCase):
    def test_primary_screen_region_uses_first_real_monitor(self) -> None:
        backend = FakeBackend([])
        backend.monitors = [
            {"left": 0, "top": 0, "width": 3840, "height": 1080},
            {"left": 100, "top": 20, "width": 2560, "height": 1440},
        ]
        capture = MssScreenCapture(FakeFactory([backend]))

        self.assertEqual(
            capture.screen_region(),
            CaptureRegion(left=100, top=20, width=2560, height=1440),
        )

    def test_exact_monitor_dictionary_passed_to_grab(self) -> None:
        backend = FakeBackend([make_shot()])
        capture = MssScreenCapture(FakeFactory([backend]))

        capture.capture(REGION, 0, 0.0)

        self.assertEqual(
            backend.calls[0],
            {"left": 10, "top": 20, "width": 2, "height": 3},
        )


class FrameConversionTests(unittest.TestCase):
    def test_bgra_converted_to_bytes_and_frame_fields(self) -> None:
        backend = FakeBackend([make_shot()])
        capture = MssScreenCapture(FakeFactory([backend]))

        frame = capture.capture(REGION, 7, 2.5)

        self.assertIsInstance(frame, CapturedFrame)
        self.assertEqual(frame.frame_id, 7)
        self.assertEqual(frame.captured_at, 2.5)
        self.assertEqual(frame.width, REGION.width)
        self.assertEqual(frame.height, REGION.height)
        self.assertEqual(frame.pixels, PIXELS)
        self.assertIsInstance(frame.pixels, bytes)
        self.assertIs(frame.region, REGION)


class ValidationTests(unittest.TestCase):
    def test_dimension_mismatch_raises(self) -> None:
        shot = FakeShot(REGION.width + 1, REGION.height, PIXELS)
        capture = MssScreenCapture(FakeFactory([FakeBackend([shot])]))

        with self.assertRaises(ValueError):
            capture.capture(REGION, 0, 0.0)

    def test_invalid_buffer_length_raises(self) -> None:
        capture = MssScreenCapture(FakeFactory([FakeBackend([make_shot(b"short")])]))

        with self.assertRaises(ValueError):
            capture.capture(REGION, 0, 0.0)

    def test_invalid_region_type_raises(self) -> None:
        factory = FakeFactory([FakeBackend([make_shot()])])
        capture = MssScreenCapture(factory)

        with self.assertRaises(TypeError):
            capture.capture({"left": 0}, 0, 0.0)  # type: ignore[arg-type]

        self.assertEqual(factory.call_count, 0)


class ErrorRecoveryTests(unittest.TestCase):
    def test_grab_error_then_successful_retry(self) -> None:
        backend = FakeBackend([RuntimeError("boom"), make_shot()])
        capture = MssScreenCapture(FakeFactory([backend]))

        with self.assertRaises(RuntimeError):
            capture.capture(REGION, 0, 0.0)

        frame = capture.capture(REGION, 1, 1.0)
        self.assertEqual(frame.frame_id, 1)
        self.assertEqual(frame.pixels, PIXELS)


class CloseTests(unittest.TestCase):
    def test_close_is_idempotent_and_closes_backend_once(self) -> None:
        backend = FakeBackend([make_shot()])
        capture = MssScreenCapture(FakeFactory([backend]))
        capture.capture(REGION, 0, 0.0)

        capture.close()
        capture.close()

        self.assertEqual(backend.close_count, 1)

    def test_close_without_backend_is_noop(self) -> None:
        capture = MssScreenCapture(FakeFactory([]))
        capture.close()  # no backend was ever created

    def test_capture_after_close_creates_new_backend(self) -> None:
        first = FakeBackend([make_shot()])
        second = FakeBackend([make_shot()])
        factory = FakeFactory([first, second])
        capture = MssScreenCapture(factory)

        capture.capture(REGION, 0, 0.0)
        capture.close()
        capture.capture(REGION, 1, 1.0)

        self.assertEqual(factory.call_count, 2)
        self.assertEqual(first.close_count, 1)
        self.assertEqual(len(second.calls), 1)

    def test_context_manager_closes_backend(self) -> None:
        backend = FakeBackend([make_shot()])
        with MssScreenCapture(FakeFactory([backend])) as capture:
            self.assertIsInstance(capture, MssScreenCapture)
            capture.capture(REGION, 0, 0.0)

        self.assertEqual(backend.close_count, 1)


if __name__ == "__main__":
    unittest.main()
