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


def _meta_free(dependencies):
    data = {"build_order": [
        {"time": "0:00", "notes": "start"},
        {"time": "5:00", "notes": "feudal"},
        {"time": "6:00", "notes": "more"},
        {"time": "10:00", "notes": "castle"},
    ]}
    for step, notes in zip(data["build_order"], dependencies):
        if notes is not None:
            step["notes"] = notes
    return data


def _result(age, game_time=300, status=RecognitionStatus.CONFIRMED,
            emitted_at=12.0, timer_status=RecognitionStatus.CONFIRMED,
            event_type="age_completed"):
    event = GameEvent(1, event_type, {"age": age}, (), 1.0, status,
                      emitted_at)
    state = GameState(
        game_time=FieldState(game_time, 1.0, timer_status, emitted_at, ()),
        updated_at=emitted_at,
    )
    return PipelineResult(True, {}, (), (), state, (event,))


def _result_no_event(game_time=300, emitted_at=12.0):
    state = GameState(
        game_time=FieldState(game_time, 1.0, RecognitionStatus.CONFIRMED,
                             emitted_at, ()),
        updated_at=emitted_at,
    )
    return PipelineResult(True, {}, (), (), state, ())


class SmartBuildOrderSessionTests(unittest.TestCase):
    def test_disabled_when_no_timer_steps(self):
        data = {"build_order": [{"time": "0:00", "notes": "start"}]}
        data["build_order"] = []
        session = SmartBuildOrderSession.from_build_order(
            data
        )
        self.assertFalse(session.enabled)
        self.assertEqual(session.times, ())
        self.assertEqual(session.consume(_result(2)),
                         ScheduleUpdate(False, False, ()))

    def test_valid_age_two_and_three_shifts(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        first = session.consume(_result(2, game_time=310))
        self.assertTrue(first.changed)
        self.assertEqual(first.times, (0, 310, 370, 610))
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

    def test_negative_shift_rejected_atomically(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        first = session.consume(_result(2, game_time=100))
        self.assertEqual(first.times, (0, 100, 160, 400))
        second = session.consume(_result(3, game_time=150, emitted_at=13.0))
        # age 3 is at index 3 currently 400; shift to 150 pushes index 3
        # to 150, prior index 2 is 160, so 150 < 160 is a rejection.
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

    def test_metadata_free_detection(self):
        data = _meta_free(["start", "feudal", "more", "castle"])
        session = SmartBuildOrderSession.from_build_order(data)
        self.assertTrue(session.enabled)
        self.assertEqual(session.times, (0, 300, 360, 600))
        update = session.consume(_result(2, game_time=310))
        self.assertTrue(update.changed)
        self.assertEqual(update.times, (0, 310, 370, 610))

    def test_metadata_free_age_up_next_age(self):
        data = _meta_free(["start", "age up", "age 3", None])
        session = SmartBuildOrderSession.from_build_order(data)
        self.assertEqual(session.times, (0, 300, 360, 600))
        update = session.consume(_result(2, game_time=310))
        self.assertEqual(update.times, (0, 310, 370, 610))

    def test_metadata_free_aoe4_guides_aging_up_phrase(self):
        data = _meta_free(["start", "Start aging up with 4 villagers", "more", "castle"])
        session = SmartBuildOrderSession.from_build_order(data)
        update = session.consume(_result(
            2, game_time=330, event_type="age_started"
        ))
        self.assertTrue(update.changed)
        self.assertEqual(update.times, (0, 330, 390, 630))

    def test_float_game_time_accepted(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        update = session.consume(_result(2, game_time=310.0))
        self.assertTrue(update.changed)
        self.assertEqual(update.times, (0, 310, 370, 610))

    def test_no_event_after_anchor_does_not_move_schedule(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        first = session.consume(_result_no_event(game_time=305, emitted_at=13.0))
        self.assertFalse(first.changed)
        self.assertEqual(first.times, (0, 300, 360, 600))
        second = session.consume(_result_no_event(game_time=310, emitted_at=14.0))
        self.assertFalse(second.changed)
        self.assertEqual(second.times, (0, 300, 360, 600))

    def test_start_anchor_shifts_advancing_rows_and_completion_shifts_suffix(self):
        data = {"build_order": [
            {"time": "0:00", "age": 1, "notes": ["start"]},
            {"time": "2:10", "age": 1, "notes": ["age 2"]},
            {"time": "3:05", "age": 1, "notes": ["keep producing"]},
            {"time": "3:25", "age": 1, "notes": ["move villagers"]},
            {"time": "3:50", "age": 2, "notes": ["feudal complete"]},
            {"time": "4:30", "age": 2, "notes": ["attack"]},
        ]}
        session = SmartBuildOrderSession.from_build_order(data)
        started = session.consume(_result(
            2, game_time=180, event_type="age_started"
        ))
        self.assertEqual(started.times, (0, 180, 235, 255, 280, 320))
        self.assertEqual(started.times[2] - started.times[1], 55)
        self.assertEqual(started.times[3] - started.times[2], 20)

        completed = session.consume(_result(2, game_time=300, emitted_at=13.0))
        self.assertEqual(completed.times, (0, 180, 235, 255, 300, 340))

    def test_early_start_shifts_suffix_earlier_and_duplicate_is_idempotent(self):
        data = _meta_free(["start", "age 2", "more", "castle"])
        session = SmartBuildOrderSession.from_build_order(data)
        first = session.consume(_result(
            2, game_time=120, event_type="age_started"
        ))
        self.assertEqual(first.times, (0, 120, 180, 420))
        duplicate = session.consume(_result(
            2, game_time=130, emitted_at=13.0, event_type="age_started"
        ))
        self.assertFalse(duplicate.changed)
        self.assertEqual(duplicate.times, first.times)

    def test_unchanged_prefix_preserved_gaps(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        update = session.consume(_result(2, game_time=310))
        self.assertEqual(update.times[0], 0)
        self.assertEqual(update.times[2] - update.times[1], 60)
        self.assertEqual(update.times[3] - update.times[2], 240)

    def test_duplicate_confirmed_event_does_nothing(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        first = session.consume(_result(2, game_time=310))
        second = session.consume(_result(2, game_time=400, emitted_at=13.0))
        self.assertFalse(second.changed)
        self.assertEqual(second.times, first.times)

    def test_sample_before_anchor_leaves_schedule_unchanged(self):
        session = SmartBuildOrderSession.from_build_order(_build_order())
        update = session.consume(_result_no_event(game_time=50))
        self.assertFalse(update.changed)
        self.assertEqual(update.times, (0, 300, 360, 600))

if __name__ == "__main__":
    unittest.main()
