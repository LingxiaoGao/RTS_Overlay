from __future__ import annotations

import unittest

from aoe4.smart_capture import CaptureRegion, CapturedFrame
from aoe4.smart_hud_reader import HudTextEvidence
from aoe4.smart_pipeline import SmartHudPipeline
from aoe4.smart_runtime import RecognitionStatus


def make_frame(frame_id: int, captured_at: float) -> CapturedFrame:
    region = CaptureRegion(left=0, top=0, width=2, height=2)
    pixels = bytes([0, 0, 255, 255]) * 4
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=captured_at,
        width=2,
        height=2,
        pixels=pixels,
        region=region,
    )


def make_evidence(
    frame: CapturedFrame, field: str, text: str, confidence: float = 0.95
) -> HudTextEvidence:
    return HudTextEvidence(
        field=field,
        text=text,
        confidence=confidence,
        frame_id=frame.frame_id,
        observed_at=frame.captured_at,
        line_count=1,
    )


def evidence_for(
    frame: CapturedFrame, fields: dict[str, str], confidence: float = 0.95
) -> dict[str, HudTextEvidence]:
    return {
        field: make_evidence(frame, field, text, confidence)
        for field, text in fields.items()
    }


class FakeReader:
    def __init__(self) -> None:
        self.mapping: dict[str, HudTextEvidence] = {}
        self.evidence: dict[str, HudTextEvidence] = {}

    def set_frame(
        self, frame: CapturedFrame, fields: dict[str, str], confidence: float = 0.95
    ) -> None:
        self.evidence = evidence_for(frame, fields, confidence)
        self.mapping = dict(self.evidence)

    def read(self, frame: CapturedFrame) -> dict[str, HudTextEvidence]:
        return self.mapping


def status_map(state) -> dict[str, RecognitionStatus]:
    result = {}
    for name in (
        "game_time",
        "age",
        "food",
        "wood",
        "gold",
        "stone",
        "population_current",
        "population_cap",
        "idle_villagers",
        "villagers_food",
        "villagers_wood",
        "villagers_gold",
        "villagers_stone",
        "villagers_total",
    ):
        result[name] = getattr(state, name).status
    return result


class SmartPipelineConfirmationTests(unittest.TestCase):
    def test_default_policy_keeps_food_and_estimator_inferred(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader=reader)
        for index in range(3):
            frame = make_frame(index, float(index))
            reader.set_frame(frame, {"food": "100"})
            result = pipeline.process(frame)
            self.assertTrue(result.sampled)
            for obs in result.verified_observations:
                self.assertIs(obs.status, RecognitionStatus.INFERRED)
        self.assertIs(pipeline.state.food.status, RecognitionStatus.INFERRED)

    def test_five_food_samples_profile_validated(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader=reader, profile_validated=True)
        expected_verified = [
            RecognitionStatus.INFERRED,
            RecognitionStatus.INFERRED,
            RecognitionStatus.CONFIRMED,
            RecognitionStatus.CONFIRMED,
            RecognitionStatus.CONFIRMED,
        ]
        expected_state = [
            RecognitionStatus.INFERRED,
            RecognitionStatus.INFERRED,
            RecognitionStatus.UNKNOWN,
            RecognitionStatus.UNKNOWN,
            RecognitionStatus.CONFIRMED,
        ]
        for index in range(5):
            frame = make_frame(index, float(index))
            reader.set_frame(frame, {"food": "100"})
            result = pipeline.process(frame)
            self.assertTrue(result.sampled)
            raw_food = [o for o in result.observations if o.field == "food"]
            verified_food = [
                o for o in result.verified_observations if o.field == "food"
            ]
            self.assertEqual(len(raw_food), 1)
            self.assertEqual(len(verified_food), 1)
            self.assertIs(raw_food[0].status, RecognitionStatus.INFERRED)
            self.assertIs(
                verified_food[0].status, expected_verified[index]
            )
            self.assertIs(
                pipeline.state.food.status, expected_state[index]
            )

    def _feed_age(self, pipeline, reader, texts, start_frame=0, step=1.0):
        results = []
        for offset, text in enumerate(texts):
            frame_id = start_frame + offset
            captured_at = start_frame * step + offset * step
            frame = make_frame(frame_id, captured_at)
            reader.set_frame(frame, {"age": text})
            results.append(pipeline.process(frame))
        return results

    def test_age_baseline_then_completion_event(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(
            reader=reader, profile_validated=True, age_signal_validated=True
        )
        first = self._feed_age(pipeline, reader, ["1"] * 5, start_frame=0)
        for result in first:
            self.assertEqual(result.events, ())
        self.assertIs(pipeline.state.age.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(pipeline.state.age.value, 1)

        second = self._feed_age(pipeline, reader, ["2"] * 5, start_frame=5)
        events = [event for r in second for event in r.events]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "age_completed")
        self.assertIs(events[0].status, RecognitionStatus.CONFIRMED)
        self.assertEqual(events[0].payload["age"], 2)
        self.assertIs(pipeline.state.age.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(pipeline.state.age.value, 2)

    def test_age_starting_at_two_no_event(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(
            reader=reader, profile_validated=True, age_signal_validated=True
        )
        results = self._feed_age(pipeline, reader, ["2"] * 5, start_frame=0)
        for result in results:
            self.assertEqual(result.events, ())
        self.assertIs(pipeline.state.age.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(pipeline.state.age.value, 2)

    def test_age_signal_validation_false_keeps_inferred(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(
            reader=reader, profile_validated=True, age_signal_validated=False
        )
        results = self._feed_age(pipeline, reader, ["1"] * 5, start_frame=0)
        for result in results:
            age_verified = [
                o for o in result.verified_observations if o.field == "age"
            ]
            self.assertEqual(len(age_verified), 1)
            self.assertIs(age_verified[0].status, RecognitionStatus.INFERRED)
            self.assertEqual(result.events, ())
        self.assertIs(pipeline.state.age.status, RecognitionStatus.INFERRED)
        self.assertEqual(pipeline.state.age.value, 1)

    def test_changing_validation_requires_fresh_streak(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader=reader, profile_validated=True)
        for index in range(2):
            frame = make_frame(index, float(index))
            reader.set_frame(frame, {"food": "100"})
            result = pipeline.process(frame)

        pipeline.set_validation(False, False)
        frame = make_frame(2, 2.0)
        reader.set_frame(frame, {"food": "100"})
        result = pipeline.process(frame)
        food = [o for o in result.verified_observations if o.field == "food"][0]
        self.assertIs(food.status, RecognitionStatus.INFERRED)
        self.assertIs(pipeline.state.food.status, RecognitionStatus.INFERRED)

        pipeline.set_validation(True, False)
        statuses = []
        for index in range(3):
            frame = make_frame(3 + index, 3.0 + index)
            reader.set_frame(frame, {"food": "100"})
            result = pipeline.process(frame)
            food = [
                o for o in result.verified_observations if o.field == "food"
            ][0]
            statuses.append(food.status)
        self.assertEqual(
            statuses,
            [
                RecognitionStatus.INFERRED,
                RecognitionStatus.INFERRED,
                RecognitionStatus.CONFIRMED,
            ],
        )

    def test_population_mismatch_is_skipped(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader=reader, profile_validated=True)
        fields = {"population_current": "200", "population_cap": "100"}
        frames = [(0, 0.0), (2, 0.5), (4, 1.0), (6, 1.5)]
        for frame_id, captured_at in frames:
            frame = make_frame(frame_id, captured_at)
            reader.set_frame(frame, fields)
            result = pipeline.process(frame)
            self.assertTrue(result.sampled)
        self.assertEqual(pipeline.scheduler.last_sample, 1.5)
        self.assertEqual(pipeline.confirmer.last_sample_index, 3)

        bad_frame = make_frame(8, 2.0)
        reader.set_frame(bad_frame, fields)
        result = pipeline.process(bad_frame)
        self.assertTrue(result.sampled)
        self.assertFalse(
            any(
                item.field in ("population_current", "population_cap")
                for item in result.verified_observations
            )
        )
        self.assertEqual(pipeline.scheduler.last_sample, 2.0)
        self.assertEqual(pipeline.confirmer.last_sample_index, 4)

        good_frame = make_frame(10, 2.5)
        reader.set_frame(
            good_frame,
            {"population_current": "200", "population_cap": "300"},
        )
        result = pipeline.process(good_frame)
        self.assertTrue(result.sampled)
        self.assertEqual(pipeline.scheduler.last_sample, 2.5)
        self.assertEqual(pipeline.confirmer.last_sample_index, 5)

        current = [
            o for o in result.verified_observations
            if o.field == "population_current"
        ][0]
        cap = [
            o for o in result.verified_observations if o.field == "population_cap"
        ][0]
        self.assertIs(current.status, RecognitionStatus.INFERRED)
        self.assertIs(cap.status, RecognitionStatus.INFERRED)
        self.assertIs(
            result.state.population_current.status, RecognitionStatus.INFERRED
        )
        self.assertEqual(result.state.population_current.value, 200)
        self.assertIs(
            result.state.population_cap.status, RecognitionStatus.INFERRED
        )

        for frame_id, captured_at in ((12, 3.0), (14, 3.5)):
            frame = make_frame(frame_id, captured_at)
            reader.set_frame(
                frame,
                {"population_current": "200", "population_cap": "300"},
            )
            result = pipeline.process(frame)
        population = {
            item.field: item for item in result.verified_observations
            if item.field in ("population_current", "population_cap")
        }
        self.assertIs(
            population["population_current"].status,
            RecognitionStatus.CONFIRMED,
        )
        self.assertIs(
            population["population_cap"].status,
            RecognitionStatus.CONFIRMED,
        )

    def test_reset_clears_state_and_preserves_validation(self) -> None:
        reader = FakeReader()
        pipeline = SmartHudPipeline(
            reader=reader, profile_validated=True, age_signal_validated=True
        )
        for index in range(5):
            frame = make_frame(index, float(index))
            reader.set_frame(frame, {"food": "100"})
            pipeline.process(frame)
        self.assertIs(pipeline.state.food.status, RecognitionStatus.CONFIRMED)
        self.assertIsNotNone(pipeline.scheduler.last_sample)
        self.assertIsNotNone(pipeline.confirmer.last_sample_index)

        pipeline.reset()
        self.assertIsNone(pipeline.scheduler.last_sample)
        self.assertIsNone(pipeline.confirmer.last_sample_index)
        self.assertTrue(pipeline.profile_validated)
        self.assertTrue(pipeline.age_signal_validated)
        self.assertIs(pipeline.state.food.status, RecognitionStatus.UNKNOWN)

        frame = make_frame(10, 10.0)
        reader.set_frame(frame, {"food": "100"})
        result = pipeline.process(frame)
        self.assertTrue(result.sampled)
        self.assertEqual(pipeline.confirmer.last_sample_index, 0)
        food = [o for o in result.verified_observations if o.field == "food"][0]
        self.assertIs(food.status, RecognitionStatus.INFERRED)
        self.assertIs(pipeline.state.food.status, RecognitionStatus.INFERRED)


if __name__ == "__main__":
    unittest.main()
