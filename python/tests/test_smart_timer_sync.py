import unittest

from aoe4.smart_timer_sync import GameTimerSynchronizer
from aoe4.smart_hud_reader import HudTextEvidence
from aoe4.smart_pipeline import PipelineResult
from aoe4.smart_runtime import (
    FieldState, GameState, Observation, RecognitionStatus,
)


def _timer_obs(value, observed_at):
    return Observation(
        observation_id=f"timer-{observed_at}",
        field="game_time",
        value=float(value),
        confidence=1.0,
        status=RecognitionStatus.INFERRED,
        frame_id=round(observed_at * 2),
        observed_at=observed_at,
    )


def _timer_evidence(value, observed_at):
    minutes, seconds = divmod(int(value), 60)
    return HudTextEvidence(
        field="game_time",
        text=f"{minutes:02d}:{seconds:02d}",
        confidence=1.0,
        frame_id=round(observed_at * 2),
        observed_at=observed_at,
        line_count=1,
    )


def _result(game_time=None, observations=(), evidence=None, sampled=True):
    if game_time is None:
        state = GameState(updated_at=0.0)
    else:
        state = GameState(
            game_time=FieldState(
                value=game_time,
                confidence=1.0,
                status=RecognitionStatus.CONFIRMED,
                observed_at=observations[0].observed_at if observations else 0.0,
            ),
            updated_at=observations[0].observed_at if observations else 0.0,
        )
    return PipelineResult(
        sampled=sampled,
        evidence=evidence if evidence is not None else {},
        observations=tuple(observations),
        verified_observations=(),
        state=state,
        events=(),
    )


def _raw(seconds, at):
    return _result(
        game_time=None,
        observations=(_timer_obs(seconds, at),),
        evidence={"game_time": _timer_evidence(seconds, at)},
    )


def _confirmed(seconds, at):
    return _result(
        game_time=seconds,
        observations=(_timer_obs(seconds, at),),
        evidence={"game_time": _timer_evidence(seconds, at)},
    )


def _projected(result):
    field = result.state.game_time
    assert field.status == RecognitionStatus.CONFIRMED, field.status
    value = field.value
    assert isinstance(value, int), value
    return value


class TestPreAnchorCadence(unittest.TestCase):
    def test_reads_every_half_second_before_anchor(self):
        sync = GameTimerSynchronizer()
        self.assertTrue(sync.should_read(0.0))
        self.assertFalse(sync.should_read(0.25))
        self.assertTrue(sync.should_read(0.5))
        self.assertFalse(sync.should_read(0.75))
        self.assertTrue(sync.should_read(1.0))


class TestSteadyCadenceAfterAnchor(unittest.TestCase):
    def test_steady_cadence_becomes_two_seconds(self):
        sync = GameTimerSynchronizer()
        sync.process(_confirmed(10, 0.0), 0.0)
        self.assertFalse(sync.should_read(0.5))
        self.assertFalse(sync.should_read(1.5))
        self.assertTrue(sync.should_read(2.0))


class TestProjectionBetweenReads(unittest.TestCase):
    def test_intermediate_calls_project_integer_time(self):
        sync = GameTimerSynchronizer()
        result = sync.process(_confirmed(30, 0.0), 0.0)
        self.assertEqual(_projected(result), 30)
        for offset in (0.5, 1.0, 1.5):
            result = sync.process(_result(), offset)
            self.assertEqual(_projected(result), 30 + int(offset))


class TestImmediateRecalibration(unittest.TestCase):
    def test_periodic_reading_within_tolerance_recalibrates(self):
        sync = GameTimerSynchronizer()
        sync.process(_confirmed(60, 0.0), 0.0)
        result = sync.process(_confirmed(62, 2.0), 2.0)
        self.assertEqual(_projected(result), 62)
        result = sync.process(_result(), 2.5)
        self.assertEqual(_projected(result), 62)

        result = sync.process(_confirmed(63, 4.0), 4.0)
        self.assertEqual(_projected(result), 63)


class TestPauseCycle(unittest.TestCase):
    def test_two_pause_resume_cycles(self):
        sync = GameTimerSynchronizer()
        t = 0.0

        result = sync.process(_confirmed(100, t), t)
        self.assertEqual(_projected(result), 100)

        # First pause: timer OCR returns same value while wall clock advances.
        t += 2.0
        result = sync.process(_confirmed(100, t), t)
        self.assertEqual(_projected(result), 100)
        pause_anchor = t
        for step in (0.5, 1.0, 1.5, 2.0):
            sample_at = pause_anchor + step
            self.assertTrue(sync.should_read(sample_at))
            result = sync.process(_result(), sample_at)
            self.assertEqual(_projected(result), 100)
        t = pause_anchor + 2.0

        # First resume: timer advances again.
        t += 2.0
        result = sync.process(_confirmed(104, t), t)
        self.assertEqual(_projected(result), 104)
        self.assertFalse(sync.should_read(t + 0.5))
        self.assertTrue(sync.should_read(t + 2.0))

        # Normal progression.
        t += 2.0
        result = sync.process(_confirmed(106, t), t)
        self.assertEqual(_projected(result), 106)

        # Second pause.
        t += 2.0
        result = sync.process(_confirmed(106, t), t)
        self.assertEqual(_projected(result), 106)
        pause_anchor = t
        for step in (0.5, 1.0):
            sample_at = pause_anchor + step
            self.assertTrue(sync.should_read(sample_at))
            result = sync.process(_result(), sample_at)
            self.assertEqual(_projected(result), 106)
        t = pause_anchor + 1.0

        # Second resume.
        t += 2.0
        result = sync.process(_confirmed(109, t), t)
        self.assertEqual(_projected(result), 109)
        self.assertFalse(sync.should_read(t + 0.5))
        self.assertTrue(sync.should_read(t + 2.0))


class TestMissingReadingRecovery(unittest.TestCase):
    def test_missing_periodic_reading_enters_half_second_recovery(self):
        sync = GameTimerSynchronizer()
        sync.process(_confirmed(20, 0.0), 0.0)
        self.assertTrue(sync.should_read(2.0))

        result = sync.process(_result(), 2.0)
        self.assertEqual(_projected(result), 22)

        self.assertTrue(sync.should_read(2.5))
        result = sync.process(_result(), 2.5)
        self.assertEqual(_projected(result), 22)


class TestImplausibleReadingRecovery(unittest.TestCase):
    def test_implausible_then_three_coherent_raws_reanchor(self):
        sync = GameTimerSynchronizer()
        sync.process(_confirmed(50, 0.0), 0.0)

        result = sync.process(_confirmed(200, 2.0), 2.0)
        self.assertEqual(_projected(result), 52)
        self.assertTrue(sync.should_read(2.5))

        r1 = sync.process(_raw(53, 2.5), 2.5)
        self.assertEqual(_projected(r1), 52)
        r2 = sync.process(_raw(54, 3.0), 3.0)
        self.assertEqual(_projected(r2), 52)
        r3 = sync.process(_raw(55, 3.5), 3.5)
        self.assertEqual(_projected(r3), 55)
        self.assertFalse(sync.should_read(4.0))
        self.assertTrue(sync.should_read(5.5))


class TestReset(unittest.TestCase):
    def test_reset_returns_to_unanchored_half_second(self):
        sync = GameTimerSynchronizer()
        sync.process(_confirmed(40, 0.0), 0.0)
        self.assertFalse(sync.should_read(0.5))

        sync.reset()
        self.assertTrue(sync.should_read(0.5))
        self.assertFalse(sync.should_read(0.75))
        self.assertTrue(sync.should_read(1.0))


if __name__ == "__main__":
    unittest.main()
