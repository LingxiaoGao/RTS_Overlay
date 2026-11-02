from __future__ import annotations

import unittest

from aoe4.smart_runtime import (
    EstimatorConfig,
    FieldState,
    GameEvent,
    GameState,
    Observation,
    RecognitionStatus,
    StateEstimator,
)


def obs(
    field: str,
    value,
    frame_id: int,
    *,
    oid: str | None = None,
    confidence: float = 0.95,
    status: RecognitionStatus = RecognitionStatus.CONFIRMED,
    observed_at: float = 0.0,
) -> Observation:
    return Observation(
        observation_id=oid if oid is not None else f"{field}-{frame_id}",
        field=field,
        value=value,
        confidence=confidence,
        status=status,
        frame_id=frame_id,
        observed_at=observed_at,
    )


def confirm_age(
    value: int,
    frames=(10, 11, 12),
    oid_prefix: str = "age",
) -> list[Observation]:
    return [obs("age", value, f, oid=f"{oid_prefix}-{f}") for f in frames]


class InitialStateTests(unittest.TestCase):
    def test_initial_state_unknown(self) -> None:
        est = StateEstimator()
        state = est.state
        self.assertEqual(state.updated_at, 0.0)
        for name in (
            "game_time", "age", "food", "wood", "gold", "stone",
            "population_current", "population_cap", "idle_villagers",
            "villagers_food", "villagers_wood", "villagers_gold",
            "villagers_stone", "villagers_total",
        ):
            self.assertEqual(getattr(state, name), FieldState.unknown())
            self.assertIs(getattr(state, name).status, RecognitionStatus.UNKNOWN)

    def test_reset_clears_state_and_sequence(self) -> None:
        est = StateEstimator()
        # First confirmed age (2) only establishes baseline, no event.
        _, events = est.ingest(confirm_age(2), now=1.0)
        self.assertEqual(events, ())
        self.assertIs(est.state.age.status, RecognitionStatus.CONFIRMED)

        fresh = est.reset()
        self.assertEqual(fresh, est.state)
        self.assertEqual(fresh.updated_at, 0.0)
        self.assertIs(est.state.age.status, RecognitionStatus.UNKNOWN)

        # After reset, baseline age 1 then age 2 emits sequence 1.
        _, base = est.ingest(confirm_age(1, frames=(20, 21, 22), oid_prefix="b"), now=1.5)
        self.assertEqual(base, ())
        _, events2 = est.ingest(confirm_age(2, frames=(23, 24, 25), oid_prefix="c"), now=2.0)
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0].sequence, 1)
        self.assertEqual(events2[0].payload["age"], 2)


class ConfirmationTests(unittest.TestCase):
    def test_two_or_three_frames_required(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        state, events = est.ingest([obs("food", 100, 1)], now=0.5)
        self.assertIs(state.food.status, RecognitionStatus.UNKNOWN)
        self.assertEqual(events, ())

        state, events = est.ingest([obs("food", 100, 2)], now=0.6)
        self.assertIs(state.food.status, RecognitionStatus.UNKNOWN)

        state, events = est.ingest([obs("food", 100, 3)], now=0.7)
        self.assertIs(state.food.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.food.value, 100)
        self.assertEqual(state.food.evidence_ids, ("food-1", "food-2", "food-3"))

    def test_nonconsecutive_frames_continue_candidate(self) -> None:
        # Strictly increasing nonconsecutive frame IDs continue a candidate.
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest([obs("food", 100, 1)], now=0.1)
        est.ingest([obs("food", 100, 5)], now=0.2)
        state, _ = est.ingest([obs("food", 100, 9)], now=0.3)
        self.assertIs(state.food.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.food.evidence_ids, ("food-1", "food-5", "food-9"))

    def test_equal_or_backward_frames_restart(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest([obs("food", 100, 1)], now=0.1)
        est.ingest([obs("food", 100, 2)], now=0.2)
        # Equal frame re-attempts: not strictly greater; restarts.
        state, _ = est.ingest([obs("food", 100, 2, oid="food-2b")], now=0.3)
        self.assertIs(state.food.status, RecognitionStatus.UNKNOWN)
        # Backward frame also restarts.
        state, _ = est.ingest([obs("food", 100, 1, oid="food-1b")], now=0.4)
        self.assertIs(state.food.status, RecognitionStatus.UNKNOWN)

    def test_value_change_restarts_confirmation(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest([obs("food", 100, 1)], now=0.1)
        est.ingest([obs("food", 100, 2)], now=0.2)
        state, _ = est.ingest([obs("food", 200, 3)], now=0.3)
        self.assertIs(state.food.status, RecognitionStatus.UNKNOWN)
        state, _ = est.ingest([obs("food", 200, 4)], now=0.4)
        self.assertIs(state.food.status, RecognitionStatus.UNKNOWN)
        state, _ = est.ingest([obs("food", 200, 5)], now=0.5)
        self.assertIs(state.food.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.food.value, 200)

    def test_duplicate_field_frame_in_batch_rejected(self) -> None:
        est = StateEstimator()
        with self.assertRaises(ValueError):
            est.ingest(
                [obs("food", 100, 1, oid="a"), obs("food", 100, 1, oid="b")],
                now=0.5,
            )
        self.assertIs(est.state.food.status, RecognitionStatus.UNKNOWN)

    def test_inferred_and_low_confidence_never_confirm(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=2, confidence_threshold=0.8))
        frames = range(1, 6)
        for f in frames:
            est.ingest(
                [obs("food", 500, f, status=RecognitionStatus.INFERRED)],
                now=0.1 * f,
            )
        self.assertIsNot(est.state.food.status, RecognitionStatus.CONFIRMED)

        for f in frames:
            est.ingest([obs("gold", 50, f, confidence=0.5)], now=0.1 * f)
        self.assertIsNot(est.state.gold.status, RecognitionStatus.CONFIRMED)

    def test_inferred_never_emits_age_event(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=2))
        for f in (1, 2, 3):
            _, events = est.ingest(
                [obs("age", 2, f, status=RecognitionStatus.INFERRED)],
                now=0.1 * f,
            )
            self.assertEqual(events, ())


class AgeEventTests(unittest.TestCase):
    def test_baseline_then_adjacent_transitions_emit_once_each(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        # Baseline age 1: no event.
        _, b = est.ingest(confirm_age(1, frames=(1, 4, 9), oid_prefix="a1"), now=1.0)
        self.assertEqual(b, ())
        _, e2 = est.ingest(confirm_age(2, frames=(10, 14, 19), oid_prefix="a2"), now=2.0)
        _, e3 = est.ingest(confirm_age(3, frames=(20, 24, 29), oid_prefix="a3"), now=3.0)
        _, e4 = est.ingest(confirm_age(4, frames=(30, 34, 39), oid_prefix="a4"), now=4.0)
        self.assertEqual([e.sequence for e in e2 + e3 + e4], [1, 2, 3])
        self.assertEqual([e.payload["age"] for e in e2 + e3 + e4], [2, 3, 4])

    def test_age_event_payload_evidence_confidence_contract(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest(confirm_age(1, frames=(1, 2, 3), oid_prefix="a1"), now=1.0)
        _, events = est.ingest(confirm_age(2, frames=(4, 5, 6), oid_prefix="a2"), now=2.0)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_type, "age_completed")
        self.assertEqual(event.payload, {"age": 2, "frame_id": 6})
        self.assertEqual(event.evidence_ids, ("a2-4", "a2-5", "a2-6"))
        self.assertEqual(event.confidence, 0.95)
        self.assertIs(event.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(event.emitted_at, 2.0)

    def test_age_event_payload_immutable(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest(confirm_age(1, frames=(1, 2, 3), oid_prefix="a1"), now=1.0)
        _, events = est.ingest(confirm_age(2, frames=(4, 5, 6), oid_prefix="a2"), now=2.0)
        payload = events[0].payload
        with self.assertRaises(TypeError):
            payload["age"] = 99  # type: ignore[index]

    def test_confirmed_age_state_after_event(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=2))
        state, _ = est.ingest(confirm_age(3, frames=(1, 2)), now=1.5)
        self.assertEqual(state.age.value, 3)
        self.assertEqual(state.age.confidence, 0.95)
        self.assertIs(state.age.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.age.evidence_ids, ("age-1", "age-2"))


class AgeTransitionSemanticsTests(unittest.TestCase):
    def test_starting_at_confirmed_age_2_emits_no_event(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        _, events = est.ingest(confirm_age(2, frames=(1, 5, 9), oid_prefix="a2"), now=1.0)
        self.assertEqual(events, ())
        self.assertEqual(est.state.age.value, 2)

    def test_forward_jump_1_to_3_establishes_baseline_without_event(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest(confirm_age(1, frames=(1, 5, 9), oid_prefix="a1"), now=1.0)
        # Jump over age 2 directly to 3: no event, new baseline is 3.
        _, jump = est.ingest(confirm_age(3, frames=(10, 14, 19), oid_prefix="a3"), now=2.0)
        self.assertEqual(jump, ())
        self.assertEqual(est.state.age.value, 3)
        # Later adjacent 3->4 emits exactly one event, sequence 1.
        _, after = est.ingest(confirm_age(4, frames=(20, 24, 29), oid_prefix="a4"), now=3.0)
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0].sequence, 1)
        self.assertEqual(after[0].payload["age"], 4)

    def test_backward_confirmed_age_marks_stale_and_preserves_prior(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest(confirm_age(1, frames=(1, 5, 9), oid_prefix="a1"), now=1.0)
        _, e2 = est.ingest(confirm_age(2, frames=(10, 14, 19), oid_prefix="a2"), now=2.0)
        self.assertEqual(len(e2), 1)
        state, back = est.ingest(confirm_age(1, frames=(20, 24, 29), oid_prefix="a1b"), now=3.0)
        self.assertEqual(back, ())
        self.assertEqual(state.age.value, 2)
        self.assertIs(state.age.status, RecognitionStatus.STALE)


class GameTimeTests(unittest.TestCase):
    def test_game_time_same_frame_confirmation(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        state, _ = est.ingest([obs("game_time", 10.0, 1)], now=0.1)
        self.assertIs(state.game_time.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.game_time.value, 10.0)
        self.assertEqual(state.game_time.evidence_ids, ("game_time-1",))

    def test_game_time_plausible_progression_stays_confirmed(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=3))
        est.ingest([obs("game_time", 10.0, 1, observed_at=0.1)], now=0.1)
        state, _ = est.ingest(
            [obs("game_time", 11.0, 4, observed_at=1.1)], now=1.1
        )
        self.assertIs(state.game_time.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.game_time.value, 11.0)

        # Later 12.0/13.0 progression remains confirmed (no freeze required).
        state, _ = est.ingest(
            [obs("game_time", 12.0, 12, observed_at=2.1)], now=2.1
        )
        self.assertIs(state.game_time.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.game_time.value, 12.0)
        state, _ = est.ingest(
            [obs("game_time", 13.0, 15, observed_at=3.1)], now=3.1
        )
        self.assertIs(state.game_time.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(state.game_time.value, 13.0)

    def test_game_time_backward_rejected_preserving_last(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=2))
        est.ingest([obs("game_time", 10.0, 1)], now=0.1)
        state, _ = est.ingest([obs("game_time", 8.0, 3)], now=0.3)
        self.assertEqual(state.game_time.value, 10.0)
        self.assertIs(state.game_time.status, RecognitionStatus.STALE)
        self.assertEqual(state.game_time.evidence_ids, ("game_time-1",))

    def test_game_time_large_jump_rejected_preserving_last(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=2))
        est.ingest([obs("game_time", 10.0, 1)], now=0.1)
        state, _ = est.ingest([obs("game_time", 50.0, 3)], now=0.3)
        self.assertEqual(state.game_time.value, 10.0)
        self.assertIs(state.game_time.status, RecognitionStatus.STALE)
        self.assertEqual(state.game_time.evidence_ids, ("game_time-1",))

    def test_game_time_recovers_after_missing_evidence(self) -> None:
        est = StateEstimator(
            EstimatorConfig(timer_max_forward_jump_seconds=2.0)
        )
        est.ingest(
            [obs("game_time", 6.0, 1, observed_at=16.0)], now=16.0
        )
        stale, _ = est.ingest(
            [
                obs(
                    "game_time",
                    6.0,
                    2,
                    confidence=0.5,
                    observed_at=16.5,
                )
            ],
            now=16.5,
        )
        self.assertIs(stale.game_time.status, RecognitionStatus.STALE)

        recovered, _ = est.ingest(
            [obs("game_time", 11.0, 3, observed_at=21.0)], now=21.0
        )
        self.assertIs(recovered.game_time.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(recovered.game_time.value, 11.0)


class ValidationTests(unittest.TestCase):
    def test_invalid_observation_values_rejected(self) -> None:
        est = StateEstimator()
        cases = [
            obs("food", -1, 1),
            obs("age", 0, 1),
            obs("age", 5, 1),
            obs("game_time", -1.0, 1),
            obs("food", "100", 1),  # type: ignore[arg-type]
        ]
        for bad in cases:
            with self.assertRaises((TypeError, ValueError)):
                est.ingest([bad], now=0.5)
        self.assertIs(est.state.food.status, RecognitionStatus.UNKNOWN)

    def test_invalid_status_confidence_timestamps_rejected(self) -> None:
        est = StateEstimator()
        with self.assertRaises(TypeError):
            est.ingest(
                [obs("food", 100, 1, status="confirmed")],  # type: ignore[arg-type]
                now=0.5,
            )
        with self.assertRaises(ValueError):
            est.ingest([obs("food", 100, 1, confidence=1.5)], now=0.5)
        with self.assertRaises(ValueError):
            est.ingest([obs("food", 100, 1, observed_at=-1.0)], now=0.5)
        with self.assertRaises((TypeError, ValueError)):
            est.ingest([obs("food", 100, -1)], now=0.5)

    def test_unknown_field_rejected(self) -> None:
        est = StateEstimator()
        with self.assertRaises(ValueError):
            est.ingest([obs("bananas", 1, 1)], now=0.5)

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EstimatorConfig(confirmation_frames=1)
        with self.assertRaises(TypeError):
            EstimatorConfig(confirmation_frames=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EstimatorConfig(confirmation_frames=2, timer_max_forward_jump_seconds=0)
        with self.assertRaises(ValueError):
            EstimatorConfig(confirmation_frames=2, stale_after_seconds=0)
        with self.assertRaises(ValueError):
            EstimatorConfig(confirmation_frames=2, confidence_threshold=1.5)
        with self.assertRaises(TypeError):
            StateEstimator("nope")  # type: ignore[arg-type]


class AtomicityTests(unittest.TestCase):
    def test_invalid_batch_is_atomic(self) -> None:
        est = StateEstimator(EstimatorConfig(confirmation_frames=2))
        # Valid baseline age 1 (no event) then valid transition to age 2 (seq 1).
        est.ingest(confirm_age(1, frames=(1, 2), oid_prefix="a1"), now=0.5)
        _, first = est.ingest(confirm_age(2, frames=(3, 4), oid_prefix="a2"), now=1.0)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].sequence, 1)
        self.assertEqual(first[0].payload["age"], 2)

        state_before = est.state

        # Population inversion should reject whole batch, including the
        # age-3 confirmation and any candidate progress.
        batch = [
            obs("age", 3, 5, oid="a3-5"),
            obs("age", 3, 6, oid="a3-6"),
            obs("population_current", 10, 7, oid="pc-7"),
            obs("population_current", 10, 8, oid="pc-8"),
            obs("population_cap", 5, 9, oid="pcap-9"),
            obs("population_cap", 5, 10, oid="pcap-10"),
        ]
        with self.assertRaises(ValueError):
            est.ingest(batch, now=2.0)

        self.assertEqual(est.state, state_before)

        # Candidate progress must be unchanged: single new age-3 frame does
        # not complete confirmation (prior was 5,6; now 9 is not 3 consecutive).
        _, events = est.ingest([obs("age", 3, 9, oid="a3-9")], now=2.5)
        self.assertEqual(events, ())

        # Next valid adjacent transition 2->3 must receive sequence 2.
        state, events = est.ingest(
            [obs("age", 3, 10, oid="a3-10"), obs("age", 3, 11, oid="a3-11")],
            now=3.0,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].sequence, 2)
        self.assertEqual(events[0].payload["age"], 3)
        self.assertEqual(state.age.value, 3)


class MarkStaleTests(unittest.TestCase):
    def test_mark_stale_uses_observed_at_and_preserves_fields(self) -> None:
        est = StateEstimator(
            EstimatorConfig(confirmation_frames=2, stale_after_seconds=5.0)
        )
        est.ingest([obs("food", 100, 1)], now=0.0)
        state, _ = est.ingest([obs("food", 100, 2)], now=1.0)
        self.assertIs(state.food.status, RecognitionStatus.CONFIRMED)
        prev = state

        # 4s since observed_at: unchanged.
        state = est.mark_stale(now=5.0)
        self.assertIs(state.food.status, RecognitionStatus.CONFIRMED)
        self.assertIs(prev, prev)

        # 10s since observed_at -> stale, value/confidence/evidence preserved.
        state = est.mark_stale(now=11.0)
        self.assertIs(state.food.status, RecognitionStatus.STALE)
        self.assertEqual(state.food.value, 100)
        self.assertEqual(state.food.confidence, 0.95)
        self.assertEqual(state.food.evidence_ids, ("food-1", "food-2"))
        self.assertEqual(state.updated_at, 11.0)

        # Previous snapshot unchanged.
        self.assertIs(prev.food, prev.food)
        self.assertIs(prev.food.status, RecognitionStatus.CONFIRMED)
        self.assertEqual(prev.updated_at, 1.0)


class ImmutabilityTests(unittest.TestCase):
    def test_public_dataclasses_frozen(self) -> None:
        est = StateEstimator()
        state = est.state
        with self.assertRaises(Exception):
            state.updated_at = 1.0  # type: ignore[misc]
        with self.assertRaises(Exception):
            est.state.food = FieldState()  # type: ignore[misc]
        with self.assertRaises(Exception):
            Observation(  # type: ignore[misc]
                observation_id="x", field="food", value=1,
                confidence=0.9, status=RecognitionStatus.CONFIRMED,
                frame_id=1, observed_at=0.0,
            ).value = 2
        with self.assertRaises(Exception):
            EstimatorConfig().confirmation_frames = 5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
