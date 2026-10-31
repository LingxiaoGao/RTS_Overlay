"""Tests for SmartBuildOrderSession."""

import copy
import json
import pathlib
import unittest

from aoe4.smart_build_order_session import (
    ScheduleUpdate,
    SmartBuildOrderSession,
    apply_schedule_times,
)
from aoe4.smart_pipeline import PipelineResult
from aoe4.smart_runtime import FieldState, GameEvent, GameState, RecognitionStatus


def _build_order(dependencies=(None, 2, 2, 3), include_meta=True):
    data = {"build_order": [
        {"time": "0:00", "notes": "start"},
        {"time": "5:00", "notes": "feudal"},
        {"time": "6:00", "notes": "more"},
        {"time": "10:00", "notes": "castle"},
    ]}
    if include_meta:
        data["smart_calibration"] = {
            "version": 1,
            "reference_age_times": {"2": 300, "3": 600},
            "step_age_dependencies": list(dependencies),
        }
    return data


def _result(age, game_time=300, status=RecognitionStatus.CONFIRMED,
            emitted_at=12.0, timer_status=RecognitionStatus.CONFIRMED):
    event = GameEvent(1, "age_completed", {"age": age}, (), 1.0, status,
                      emitted_at)
    state = GameState(
        game_time=FieldState(game_time, 1.0, timer_status, emitted_at, ()),
        updated_at=emitted_at,
    )
    return PipelineResult(True, {}, (), (), state, (event,))


class SmartBuildOrderSessionTests(unittest.TestCase):
    def test_missing_metadata_disabled(self):
        session = SmartBuildOrderSession.from_build_order(
            _build_order(include_meta=False)
        )
        self.assertFalse(session.enabled)
        self.assertEqual(session.times, (0, 300, 360, 600))
        self.assertEqual(session.consume(_result(2)),
                         ScheduleUpdate(False, False, (0, 300, 360, 600)))

    def test_valid_age_two_and_three_shifts(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        first = session.consume(_result(2, game_time=310))
        self.assertTrue(first.changed)
        self.assertEqual(first.times, (0, 310, 370, 600))
        second = session.consume(_result(3, game_time=615, emitted_at=13.0))
        self.assertTrue(second.changed)
        self.assertEqual(second.times, (0, 310, 370, 615))

    def test_unconfirmed_event_and_timer_rejected(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        self.assertFalse(session.consume(
            _result(2, status=RecognitionStatus.INFERRED)).changed)
        self.assertFalse(session.consume(
            _result(2, timer_status=RecognitionStatus.STALE)).changed)

    def test_timestamp_mismatch_rejected(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        result = _result(2)
        mismatched = PipelineResult(
            True, {}, (), (),
            GameState(game_time=result.state.game_time, updated_at=99.0),
            result.events,
        )
        self.assertFalse(session.consume(mismatched).changed)

    def test_duplicate_is_idempotent(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        first = session.consume(_result(2, game_time=310))
        second = session.consume(_result(2, game_time=310))
        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(second.times, first.times)

    def test_ordering_rejection_is_atomic(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        first = session.consume(_result(2, game_time=500))
        self.assertEqual(first.times, (0, 500, 560, 600))
        second = session.consume(_result(3, game_time=550, emitted_at=13.0))
        self.assertFalse(second.changed)
        self.assertEqual(session.times, first.times)

    def test_reset_restores_original_times(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        session.consume(_result(2, game_time=310))
        session.reset()
        self.assertEqual(session.times, (0, 300, 360, 600))

    def test_invalid_present_metadata(self):
        bad_version = _build_order()
        bad_version["smart_calibration"]["version"] = True
        with self.assertRaises(ValueError):
            SmartBuildOrderSession.from_build_order(bad_version)
        missing_reference = _build_order(dependencies=(None, 2, 4, 3))
        with self.assertRaises(ValueError):
            SmartBuildOrderSession.from_build_order(missing_reference)

    def test_source_data_unchanged(self):
        data = _build_order()
        before = copy.deepcopy(data)
        session = SmartBuildOrderSession.from_build_order(data)
        session.consume(_result(2, game_time=310))
        self.assertEqual(data, before)

    def test_apply_schedule_times_synchronizes_fields(self):
        steps = [
            {"time": "0:00", "time_sec": 0, "notes": "start"},
            {"time": "5:00", "time_sec": 300, "notes": "feudal"},
        ]
        result = apply_schedule_times(steps, (0, 310))
        self.assertEqual(result[0]["time_sec"], 0)
        self.assertEqual(result[0]["time"], "0:00")
        self.assertEqual(result[1]["time_sec"], 310)
        self.assertEqual(result[1]["time"], "5:10")
        self.assertEqual(result[1]["notes"], "feudal")

    def test_apply_schedule_times_does_not_mutate_inputs(self):
        steps = [{"time": "5:00", "time_sec": 300}]
        before = copy.deepcopy(steps)
        result = apply_schedule_times(steps, (310,))
        self.assertEqual(steps, before)
        self.assertIsNot(result, steps)
        self.assertIsNot(result[0], steps[0])

    def test_apply_schedule_times_length_mismatch(self):
        with self.assertRaises(ValueError):
            apply_schedule_times([{"time": "0:00", "time_sec": 0}], (0, 1))

    def test_packaged_demo_enables_smart_calibration(self):
        demo_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "demo_data" / "build_orders" / "example.json"
        )
        with demo_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        session = SmartBuildOrderSession.from_build_order(data)
        self.assertTrue(session.enabled)
        self.assertEqual(session.times, (30, 150, 300))
        update = session.consume(_result(2, game_time=330))
        self.assertTrue(update.changed)
        self.assertEqual(update.times, (30, 150, 330))


if __name__ == "__main__":
    unittest.main()
