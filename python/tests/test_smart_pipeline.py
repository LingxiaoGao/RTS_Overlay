import unittest

from aoe4.smart_capture import CaptureRegion, CapturedFrame
from aoe4.smart_hud_reader import HudTextEvidence
from aoe4.smart_pipeline import SmartHudPipeline
from aoe4.smart_runtime import EstimatorConfig, RecognitionStatus, StateEstimator
from aoe4.smart_sampling import ObservationScheduler, SamplingConfig

REGION = CaptureRegion(left=0, top=0, width=2, height=2)


def make_frame(frame_id=0, captured_at=0, region=REGION):
    pixels = bytes([10, 20, 30, 255] * (region.width * region.height))
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=captured_at,
        width=region.width,
        height=region.height,
        pixels=pixels,
        region=region,
    )


def evidence(field, text, frame, value_at=None):
    at = frame.captured_at if value_at is None else value_at
    return HudTextEvidence(
        field=field,
        text=text,
        confidence=0.9,
        frame_id=frame.frame_id,
        observed_at=at,
        line_count=1,
    )


class FakeReader:
    def __init__(self, payload=None):
        self.payload = {} if payload is None else payload
        self.calls = []

    def read(self, frame):
        self.calls.append(frame)
        return self.payload


class ReaderContractTests(unittest.TestCase):
    def test_defaults_and_properties(self):
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader)
        self.assertIsInstance(pipeline.scheduler, ObservationScheduler)
        self.assertIsInstance(pipeline.estimator, StateEstimator)
        self.assertIs(pipeline.state, pipeline.estimator.state)
        self.assertIsNone(pipeline.scheduler.last_sample)

    def test_invalid_reader_rejected(self):
        with self.assertRaises(TypeError):
            SmartHudPipeline(object())
        with self.assertRaises(TypeError):
            SmartHudPipeline(None)
        with self.assertRaises(TypeError):
            SmartHudPipeline(type("R", (), {"read": 1})())

    def test_scheduler_subclass_accepted(self):
        class Sub(ObservationScheduler):
            pass

        scheduler = Sub()
        pipeline = SmartHudPipeline(FakeReader(), scheduler=scheduler)
        self.assertIs(pipeline.scheduler, scheduler)
        with self.assertRaises(TypeError):
            SmartHudPipeline(FakeReader(), scheduler=object())

    def test_estimator_subclass_accepted(self):
        class Sub(StateEstimator):
            pass

        estimator = Sub()
        pipeline = SmartHudPipeline(FakeReader(), estimator=estimator)
        self.assertIs(pipeline.estimator, estimator)
        with self.assertRaises(TypeError):
            SmartHudPipeline(FakeReader(), estimator=object())

    def test_supplied_components_used(self):
        scheduler = ObservationScheduler(SamplingConfig(interval_seconds=2.0))
        estimator = StateEstimator(EstimatorConfig(stale_after_seconds=0.5))
        pipeline = SmartHudPipeline(FakeReader(), scheduler=scheduler, estimator=estimator)
        self.assertIs(pipeline.scheduler, scheduler)
        self.assertIs(pipeline.estimator, estimator)

    def test_capability_compatible_substitutes_are_used(self):
        class Scheduler:
            def __init__(self):
                self._inner = ObservationScheduler()

            @property
            def last_sample(self):
                return self._inner.last_sample

            def due(self, now):
                return self._inner.due(now)

            def mark_sample(self, now):
                return self._inner.mark_sample(now)

            def reset(self):
                return self._inner.reset()

        class Estimator:
            def __init__(self):
                self._inner = StateEstimator()

            @property
            def state(self):
                return self._inner.state

            def ingest(self, observations, now):
                return self._inner.ingest(observations, now)

            def mark_stale(self, now):
                return self._inner.mark_stale(now)

            def reset(self):
                return self._inner.reset()

        scheduler = Scheduler()
        estimator = Estimator()
        pipeline = SmartHudPipeline(
            FakeReader(), scheduler=scheduler, estimator=estimator
        )

        result = pipeline.process(make_frame(frame_id=0, captured_at=0))

        self.assertTrue(result.sampled)
        self.assertIs(pipeline.scheduler, scheduler)
        self.assertIs(pipeline.estimator, estimator)


class SamplingTests(unittest.TestCase):
    def test_first_frame_samples_once(self):
        frame = make_frame(frame_id=1, captured_at=0)
        reader = FakeReader({"food": evidence("food", "100", frame)})
        pipeline = SmartHudPipeline(reader)
        result = pipeline.process(frame)
        self.assertTrue(result.sampled)
        self.assertEqual(reader.calls, [frame])
        self.assertEqual(pipeline.scheduler.last_sample, 0)
        self.assertEqual(len(result.observations), 1)
        obs = result.observations[0]
        self.assertEqual(obs.status, RecognitionStatus.INFERRED)
        self.assertEqual(obs.frame_id, 1)
        self.assertEqual(obs.observed_at, 0.0)
        self.assertEqual(result.state.food.value, 100)
        self.assertEqual(result.state.food.status, RecognitionStatus.INFERRED)
        self.assertEqual(result.events, ())

    def test_default_cadence(self):
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader)
        r0 = pipeline.process(make_frame(frame_id=0, captured_at=0))
        self.assertTrue(r0.sampled)
        self.assertEqual(len(reader.calls), 1)
        r1 = pipeline.process(make_frame(frame_id=1, captured_at=0.49))
        self.assertFalse(r1.sampled)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(r1.observations, ())
        self.assertEqual(r1.events, ())
        self.assertEqual(r1.evidence, {})
        r2 = pipeline.process(make_frame(frame_id=2, captured_at=0.5))
        self.assertTrue(r2.sampled)
        self.assertEqual(len(reader.calls), 2)
        self.assertEqual(pipeline.scheduler.last_sample, 0.5)

    def test_empty_evidence_is_successful_sample(self):
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader)
        result = pipeline.process(make_frame(frame_id=0, captured_at=0))
        self.assertTrue(result.sampled)
        self.assertEqual(dict(result.evidence), {})
        self.assertEqual(result.observations, ())
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(pipeline.scheduler.last_sample, 0)


class StaleAdvanceTests(unittest.TestCase):
    def test_skipped_frame_advances_stale(self):
        frame = make_frame(frame_id=0, captured_at=0)
        reader = FakeReader({"food": evidence("food", "100", frame)})
        estimator = StateEstimator(EstimatorConfig(stale_after_seconds=0.1))
        pipeline = SmartHudPipeline(reader, estimator=estimator)
        first = pipeline.process(frame)
        self.assertTrue(first.sampled)
        self.assertEqual(pipeline.state.food.status, RecognitionStatus.INFERRED)
        skipped = pipeline.process(make_frame(frame_id=1, captured_at=0.49))
        self.assertFalse(skipped.sampled)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(pipeline.state.food.status, RecognitionStatus.STALE)
        self.assertEqual(skipped.state.food.status, RecognitionStatus.STALE)

    def test_sampled_result_is_post_mark_stale_snapshot(self):
        frame = make_frame(frame_id=0, captured_at=0)
        reader = FakeReader({"food": evidence("food", "100", frame)})
        estimator = StateEstimator(EstimatorConfig(stale_after_seconds=5.0))
        pipeline = SmartHudPipeline(reader, estimator=estimator)
        pipeline.process(frame)
        self.assertEqual(pipeline.state.food.value, 100)
        self.assertEqual(pipeline.state.food.status, RecognitionStatus.INFERRED)
        reader.payload = {}
        later = pipeline.process(make_frame(frame_id=6, captured_at=6))
        self.assertTrue(later.sampled)
        self.assertEqual(later.state.food.status, RecognitionStatus.STALE)
        self.assertEqual(later.state.food.value, 100)
        self.assertEqual(later.state, pipeline.state)
        self.assertIs(later.state, pipeline.state)


class MalformedEvidenceTests(unittest.TestCase):
    def test_non_mapping(self):
        pipeline = SmartHudPipeline(FakeReader([]))
        with self.assertRaises(TypeError):
            pipeline.process(make_frame(frame_id=0, captured_at=0))
        self.assertIsNone(pipeline.scheduler.last_sample)

    def test_retry_same_timestamp_after_failure(self):
        frame = make_frame(frame_id=0, captured_at=0)
        reader = FakeReader(["bad"])
        pipeline = SmartHudPipeline(reader)
        with self.assertRaises(TypeError):
            pipeline.process(frame)
        self.assertIsNone(pipeline.scheduler.last_sample)
        self.assertEqual(pipeline.state, StateEstimator().state)
        good = make_frame(frame_id=0, captured_at=0)
        reader.payload = {"food": evidence("food", "100", good)}
        retried = pipeline.process(good)
        self.assertTrue(retried.sampled)
        self.assertEqual(pipeline.scheduler.last_sample, 0)
        self.assertEqual(len(reader.calls), 2)

    def test_last_sample_unchanged_after_failed_due_sample(self):
        frame = make_frame(frame_id=0, captured_at=0)
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader)
        pipeline.process(frame)
        self.assertEqual(pipeline.scheduler.last_sample, 0)
        reader.payload = ["bad"]
        with self.assertRaises(TypeError):
            pipeline.process(make_frame(frame_id=1, captured_at=1))
        self.assertEqual(pipeline.scheduler.last_sample, 0)
        retried_frame = make_frame(frame_id=1, captured_at=1)
        reader.payload = {}
        retried = pipeline.process(retried_frame)
        self.assertTrue(retried.sampled)
        self.assertEqual(pipeline.scheduler.last_sample, 1)


class ResetAndInvalidFrameTests(unittest.TestCase):
    def test_reset_clears_cadence_and_estimator(self):
        frame = make_frame(frame_id=0, captured_at=0)
        reader = FakeReader({"food": evidence("food", "100", frame)})
        pipeline = SmartHudPipeline(reader)
        pipeline.process(frame)
        self.assertIsNotNone(pipeline.scheduler.last_sample)
        self.assertNotEqual(
            pipeline.state.food.status, RecognitionStatus.UNKNOWN
        )
        state = pipeline.reset()
        self.assertIsNone(pipeline.scheduler.last_sample)
        self.assertEqual(state, StateEstimator().state)
        self.assertEqual(pipeline.state, StateEstimator().state)
        self.assertEqual(pipeline.state.food.status, RecognitionStatus.UNKNOWN)
        reader.payload = {}
        next_frame = make_frame(frame_id=1, captured_at=0.1)
        result = pipeline.process(next_frame)
        self.assertTrue(result.sampled)
        self.assertEqual(reader.calls[-1], next_frame)
        self.assertEqual(pipeline.scheduler.last_sample, 0.1)

    def test_invalid_frame_does_not_call_reader(self):
        reader = FakeReader()
        pipeline = SmartHudPipeline(reader)
        with self.assertRaises(TypeError):
            pipeline.process(object())
        with self.assertRaises(TypeError):
            pipeline.process(None)
        with self.assertRaises(TypeError):
            pipeline.process({"frame_id": 0})
        self.assertEqual(reader.calls, [])


if __name__ == "__main__":
    unittest.main()
