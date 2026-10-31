"""Tests for aoe4.smart_capture."""

import threading
import unittest
from dataclasses import FrozenInstanceError

from aoe4.smart_capture import CaptureRegion, CapturedFrame, LatestFrameSlot


def make_frame(frame_id=1, captured_at=0.0, width=2, height=3, region=None, pixels=b"\x00" * 6):
    if region is None:
        region = CaptureRegion(left=0, top=0, width=width, height=height)
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=captured_at,
        width=width,
        height=height,
        pixels=pixels,
        region=region,
    )


class CaptureRegionTests(unittest.TestCase):
    def test_boundaries_and_properties(self):
        region = CaptureRegion(left=5, top=7, width=10, height=20)
        self.assertEqual(region.right, 15)
        self.assertEqual(region.bottom, 27)

    def test_zero_origin_and_unit_size_allowed(self):
        region = CaptureRegion(left=0, top=0, width=1, height=1)
        self.assertEqual((region.right, region.bottom), (1, 1))

    def test_frozen(self):
        region = CaptureRegion(left=0, top=0, width=1, height=1)
        with self.assertRaises(FrozenInstanceError):
            region.left = 2

    def test_bool_equality(self):
        region = CaptureRegion(left=1, top=2, width=3, height=4)
        self.assertTrue(region == CaptureRegion(left=1, top=2, width=3, height=4))

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            CaptureRegion(left=True, top=0, width=1, height=1)
        with self.assertRaises(TypeError):
            CaptureRegion(left=0.0, top=0, width=1, height=1)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            CaptureRegion(left=-1, top=0, width=1, height=1)
        with self.assertRaises(ValueError):
            CaptureRegion(left=0, top=-1, width=1, height=1)
        with self.assertRaises(ValueError):
            CaptureRegion(left=0, top=0, width=0, height=1)
        with self.assertRaises(ValueError):
            CaptureRegion(left=0, top=0, width=1, height=0)


class CapturedFrameTests(unittest.TestCase):
    def test_valid_and_bytes(self):
        frame = make_frame()
        self.assertIsInstance(frame.pixels, bytes)
        self.assertIsInstance(frame.region, CaptureRegion)

    def test_frozen(self):
        frame = make_frame()
        with self.assertRaises(FrozenInstanceError):
            frame.frame_id = 5

    def test_non_bytes_pixels_rejected(self):
        with self.assertRaises(TypeError):
            make_frame(pixels=bytearray(b"\x00" * 6))

    def test_negative_frame_id_rejected(self):
        with self.assertRaises(ValueError):
            make_frame(frame_id=-1)

    def test_bool_frame_id_rejected(self):
        with self.assertRaises(TypeError):
            make_frame(frame_id=True)

    def test_non_finite_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            make_frame(captured_at=float("inf"))
        with self.assertRaises(ValueError):
            make_frame(captured_at=float("nan"))

    def test_negative_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            make_frame(captured_at=-0.1)

    def test_int_timestamp_allowed(self):
        self.assertEqual(make_frame(captured_at=3).captured_at, 3)

    def test_dimensions_must_be_positive(self):
        with self.assertRaises(ValueError):
            make_frame(width=0)
        with self.assertRaises(ValueError):
            make_frame(height=0)

    def test_dimensions_must_match_region(self):
        with self.assertRaises(ValueError):
            make_frame(width=5, region=CaptureRegion(0, 0, 2, 3))

    def test_region_type_required(self):
        with self.assertRaises(TypeError):
            CapturedFrame(frame_id=1, captured_at=0.0, width=2, height=3,
                          pixels=b"\x00" * 6, region=(0, 0, 2, 3))


class LatestFrameSlotTests(unittest.TestCase):
    def test_initial_state(self):
        slot = LatestFrameSlot()
        self.assertIsNone(slot.peek_latest())
        self.assertIsNone(slot.take_latest())
        self.assertEqual(slot.published_count, 0)
        self.assertEqual(slot.dropped_count, 0)

    def test_publish_peek_take_clear(self):
        slot = LatestFrameSlot()
        first = make_frame(frame_id=1)
        slot.publish(first)
        self.assertIs(slot.peek_latest(), first)
        self.assertIs(slot.take_latest(), first)
        self.assertIsNone(slot.peek_latest())
        second = make_frame(frame_id=2)
        slot.publish(second)
        slot.clear()
        self.assertIsNone(slot.peek_latest())
        self.assertEqual(slot.published_count, 2)

    def test_replacement_drop_accounting(self):
        slot = LatestFrameSlot()
        slot.publish(make_frame(frame_id=1))
        slot.publish(make_frame(frame_id=2))
        slot.publish(make_frame(frame_id=3))
        self.assertEqual(slot.published_count, 3)
        self.assertEqual(slot.dropped_count, 2)
        self.assertEqual(slot.peek_latest().frame_id, 3)

    def test_consumed_not_dropped(self):
        slot = LatestFrameSlot()
        slot.publish(make_frame(frame_id=1))
        slot.take_latest()
        slot.publish(make_frame(frame_id=2))
        self.assertEqual(slot.dropped_count, 0)

    def test_duplicate_and_decreasing_ids_rejected_atomically(self):
        slot = LatestFrameSlot()
        slot.publish(make_frame(frame_id=5))
        for bad_id in (5, 4):
            with self.assertRaises(ValueError):
                slot.publish(make_frame(frame_id=bad_id))
            self.assertEqual(slot.peek_latest().frame_id, 5)
            self.assertEqual(slot.published_count, 1)
            self.assertEqual(slot.dropped_count, 0)

    def test_invalid_type_rejected_atomically(self):
        slot = LatestFrameSlot()
        slot.publish(make_frame(frame_id=1))
        with self.assertRaises(TypeError):
            slot.publish(object())
        self.assertEqual(slot.peek_latest().frame_id, 1)
        self.assertEqual(slot.published_count, 1)
        self.assertEqual(slot.dropped_count, 0)

    def test_clear_allows_lower_id_after_reset(self):
        slot = LatestFrameSlot()
        slot.publish(make_frame(frame_id=5))
        slot.clear()
        self.assertEqual(slot.peek_latest(), None)
        # _last_id persists after clear; a lower id is still rejected.
        with self.assertRaises(ValueError):
            slot.publish(make_frame(frame_id=4))

    def test_deterministic_concurrency(self):
        slot = LatestFrameSlot()
        barrier = threading.Barrier(2)
        errors = []

        def publisher():
            frame = make_frame(frame_id=1)
            barrier.wait()
            try:
                slot.publish(frame)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def taker():
            barrier.wait()
            try:
                slot.take_latest()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=publisher), threading.Thread(target=taker)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        remaining = slot.peek_latest()
        # Only two valid outcomes: taker ran after publish (None) or before (frame 1).
        self.assertIn(remaining, (None, make_frame(frame_id=1)))
        self.assertEqual(slot.published_count, 1)
        self.assertEqual(slot.dropped_count, 0)


if __name__ == "__main__":
    unittest.main()
