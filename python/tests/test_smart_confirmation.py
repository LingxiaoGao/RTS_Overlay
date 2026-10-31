import math
import unittest

from aoe4.smart_confirmation import ConfirmationConfig, HudEvidenceConfirmer
from aoe4.smart_hud_reader import HudTextEvidence
from aoe4.smart_runtime import Observation, RecognitionStatus


def ev(field, text, confidence, frame_id, observed_at):
    return HudTextEvidence(
        field=field,
        text=text,
        confidence=confidence,
        frame_id=frame_id,
        observed_at=observed_at,
        line_count=1,
    )


def obs(oid, field, value, confidence, frame_id, observed_at):
    return Observation(
        observation_id=oid,
        field=field,
        value=value,
        confidence=confidence,
        status=RecognitionStatus.INFERRED,
        frame_id=frame_id,
        observed_at=observed_at,
    )


def sample(confirmer, evidence, observations, sample_index, profile_validated=True,
           age_signal_validated=False):
    return confirmer.accept(
        evidence,
        observations,
        sample_index,
        profile_validated=profile_validated,
        age_signal_validated=age_signal_validated,
    )


def food_frame(frame_id, at, value=100, ev_conf=0.9, obs_conf=0.8, field="food",
               text="100", oid_prefix="o"):
    evidence = {field: ev(field, text, ev_conf, frame_id, at)}
    observations = [obs(f"{oid_prefix}{frame_id}", field, value, obs_conf, frame_id, float(at))]
    return evidence, observations


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = ConfirmationConfig()
        self.assertEqual(config.consecutive_samples, 3)
        self.assertEqual(config.min_evidence_confidence, 0.8)
        self.assertEqual(config.timer_max_forward_jump_seconds, 2.0)

    def test_bool_rejected_for_samples(self):
        with self.assertRaises(TypeError):
            ConfirmationConfig(consecutive_samples=True)

    def test_bool_rejected_for_confidence(self):
        with self.assertRaises(TypeError):
            ConfirmationConfig(min_evidence_confidence=False)

    def test_bool_rejected_for_timer_jump(self):
        with self.assertRaises(TypeError):
            ConfirmationConfig(timer_max_forward_jump_seconds=True)

    def test_samples_below_two(self):
        with self.assertRaises(ValueError):
            ConfirmationConfig(consecutive_samples=1)
        with self.assertRaises(ValueError):
            ConfirmationConfig(consecutive_samples=0)

    def test_nonfinite_confidence(self):
        with self.assertRaises(ValueError):
            ConfirmationConfig(min_evidence_confidence=math.nan)
        with self.assertRaises(ValueError):
            ConfirmationConfig(min_evidence_confidence=math.inf)

    def test_confidence_out_of_range(self):
        with self.assertRaises(ValueError):
            ConfirmationConfig(min_evidence_confidence=-0.1)
        with self.assertRaises(ValueError):
            ConfirmationConfig(min_evidence_confidence=1.1)

    def test_timer_jump_nonfinite(self):
        with self.assertRaises(ValueError):
            ConfirmationConfig(timer_max_forward_jump_seconds=math.inf)

    def test_timer_jump_nonpositive(self):
        with self.assertRaises(ValueError):
            ConfirmationConfig(timer_max_forward_jump_seconds=0.0)
        with self.assertRaises(ValueError):
            ConfirmationConfig(timer_max_forward_jump_seconds=-1.0)


class ConstructorTests(unittest.TestCase):
    def test_defaults_and_properties(self):
        confirmer = HudEvidenceConfirmer()
        self.assertIsInstance(confirmer.config, ConfirmationConfig)
        self.assertIsNone(confirmer.last_sample_index)

    def test_custom_config_used(self):
        config = ConfirmationConfig(consecutive_samples=2)
        confirmer = HudEvidenceConfirmer(config)
        self.assertIs(confirmer.config, config)

    def test_bad_config_type(self):
        with self.assertRaises(TypeError):
            HudEvidenceConfirmer(config=object())

    def test_reset_clears_state(self):
        confirmer = HudEvidenceConfirmer()
        e, o = food_frame(1, 1)
        sample(confirmer, e, o, 0)
        self.assertEqual(confirmer.last_sample_index, 0)
        confirmer.reset()
        self.assertIsNone(confirmer.last_sample_index)
        e, o = food_frame(2, 2)
        result = sample(confirmer, e, o, 0)
        self.assertEqual(result[0].status, RecognitionStatus.INFERRED)


class BasicAcceptanceTests(unittest.TestCase):
    def test_confidence_and_value(self):
        confirmer = HudEvidenceConfirmer()
        e, o = food_frame(1, 1, value=100, ev_conf=0.9, obs_conf=0.8)
        result = sample(confirmer, e, o, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.8)
        self.assertEqual(result[0].status, RecognitionStatus.INFERRED)

    def test_output_order_and_fields_preserved(self):
        confirmer = HudEvidenceConfirmer()
        e = {
            "food": ev("food", "100", 0.9, 5, 2),
            "wood": ev("wood", "50", 0.9, 5, 2),
        }
        o = [
            obs("food-1", "food", 100, 0.8, 5, 2.0),
            obs("wood-1", "wood", 50, 0.8, 5, 2.0),
        ]
        result = sample(confirmer, e, o, 0)
        self.assertEqual([r.observation_id for r in result], ["food-1", "wood-1"])
        self.assertEqual([r.field for r in result], ["food", "wood"])
        self.assertEqual([r.value for r in result], [100, 50])
        self.assertEqual([r.frame_id for r in result], [5, 5])
        self.assertEqual([r.observed_at for r in result], [2.0, 2.0])
        for out in result:
            self.assertEqual(out.status, RecognitionStatus.INFERRED)


class StreakTests(unittest.TestCase):
    def test_confirms_only_on_third_index(self):
        confirmer = HudEvidenceConfirmer()
        statuses = []
        for index, frame in enumerate([10, 20, 30]):
            e, o = food_frame(frame, frame, oid_prefix="f")
            out = sample(confirmer, e, o, index)
            statuses.append(out[0].status)
        self.assertEqual(
            statuses,
            [
                RecognitionStatus.INFERRED,
                RecognitionStatus.INFERRED,
                RecognitionStatus.CONFIRMED,
            ],
        )
        self.assertEqual(confirmer.last_sample_index, 2)

    def test_frame_ids_preserved(self):
        confirmer = HudEvidenceConfirmer()
        frames = [10, 20, 30]
        last = None
        for index, frame in enumerate(frames):
            e, o = food_frame(frame, frame)
            last = sample(confirmer, e, o, index)
        self.assertEqual(last[0].frame_id, 30)

    def test_gap_in_index_restarts(self):
        confirmer = HudEvidenceConfirmer()
        e, o = food_frame(1, 1)
        sample(confirmer, e, o, 0)
        e, o = food_frame(2, 2)
        sample(confirmer, e, o, 2)
        out = sample(confirmer, *(food_frame(3, 3)), sample_index=3)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)

    def test_omitting_food_breaks_streak(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *food_frame(1, 1), sample_index=0)
        sample(confirmer, *food_frame(2, 2), sample_index=1)
        wood_ev = {"wood": ev("wood", "50", 0.9, 3, 3)}
        wood_obs = [obs("w3", "wood", 50, 0.8, 3, 3.0)]
        sample(confirmer, wood_ev, wood_obs, 2)
        out = sample(confirmer, *food_frame(4, 4), sample_index=3)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)
        out2 = sample(confirmer, *food_frame(5, 5), sample_index=4)
        self.assertEqual(out2[0].status, RecognitionStatus.INFERRED)
        out3 = sample(confirmer, *food_frame(6, 6), sample_index=5)
        self.assertEqual(out3[0].status, RecognitionStatus.CONFIRMED)

    def test_empty_sample_breaks_streak_and_advances(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *food_frame(1, 1), sample_index=0)
        sample(confirmer, *food_frame(2, 2), sample_index=1)
        empty = sample(confirmer, {}, [], 2)
        self.assertEqual(empty, ())
        self.assertEqual(confirmer.last_sample_index, 2)
        out = sample(confirmer, *food_frame(3, 3), sample_index=3)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)
        out2 = sample(confirmer, *food_frame(4, 4), sample_index=4)
        self.assertEqual(out2[0].status, RecognitionStatus.INFERRED)
        out3 = sample(confirmer, *food_frame(5, 5), sample_index=5)
        self.assertEqual(out3[0].status, RecognitionStatus.CONFIRMED)

    def test_gate_open_later_needs_fresh_streak(self):
        confirmer = HudEvidenceConfirmer()
        for index in range(4):
            e, o = food_frame(index, index)
            out = sample(confirmer, e, o, index, profile_validated=False)
            self.assertEqual(out[0].status, RecognitionStatus.INFERRED)
        out = sample(confirmer, *food_frame(10, 10), sample_index=10,
                     profile_validated=True)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)

    def test_low_combined_confidence_breaks(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *food_frame(1, 1), sample_index=0)
        sample(confirmer, *food_frame(2, 2), sample_index=1)
        e = {"food": ev("food", "100", 0.5, 3, 3)}
        o = [obs("o3", "food", 100, 0.8, 3, 3.0)]
        out = sample(confirmer, e, o, 2)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)
        self.assertEqual(out[0].confidence, 0.5)
        out2 = sample(confirmer, *food_frame(4, 4), sample_index=3)
        self.assertEqual(out2[0].status, RecognitionStatus.INFERRED)


class AgeTests(unittest.TestCase):
    @staticmethod
    def age_frame(frame_id, at, value=2):
        e = {"age": ev("age", str(value), 0.9, frame_id, at)}
        o = [obs(f"a{frame_id}", "age", value, 0.8, frame_id, float(at))]
        return e, o

    def test_age_requires_both_flags(self):
        confirmer = HudEvidenceConfirmer()
        for index in range(3):
            out = sample(confirmer, *self.age_frame(index, index), sample_index=index,
                         profile_validated=True, age_signal_validated=False)
            self.assertEqual(out[0].status, RecognitionStatus.INFERRED)

    def test_age_confirms_with_both_flags(self):
        confirmer = HudEvidenceConfirmer()
        statuses = []
        for index in range(3):
            out = sample(confirmer, *self.age_frame(index, index), sample_index=index,
                         profile_validated=True, age_signal_validated=True)
            statuses.append(out[0].status)
        self.assertEqual(statuses[-1], RecognitionStatus.CONFIRMED)

    def test_age_flag_gate_later_needs_fresh_streak(self):
        confirmer = HudEvidenceConfirmer()
        for index in range(3):
            sample(confirmer, *self.age_frame(index, index), sample_index=index,
                   profile_validated=True, age_signal_validated=False)
        out = sample(confirmer, *self.age_frame(5, 5), sample_index=5,
                     profile_validated=True, age_signal_validated=True)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)


class TimerTests(unittest.TestCase):
    @staticmethod
    def timer_frame(frame_id, value, at):
        e = {"game_time": ev("game_time", str(value), 0.9, frame_id, at)}
        o = [obs(f"t{frame_id}", "game_time", value, 0.8, frame_id, float(at))]
        return e, o

    def test_timer_progression_confirms(self):
        confirmer = HudEvidenceConfirmer()
        values = [10.0, 11.0, 11.0]
        statuses = []
        for index, value in enumerate(values):
            out = sample(confirmer, *self.timer_frame(index, value, index), sample_index=index)
            statuses.append(out[0].status)
        self.assertEqual(statuses[-1], RecognitionStatus.CONFIRMED)

    def test_timer_backward_restarts(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *self.timer_frame(0, 10.0, 0), sample_index=0)
        sample(confirmer, *self.timer_frame(1, 11.0, 1), sample_index=1)
        out = sample(confirmer, *self.timer_frame(2, 10.5, 2), sample_index=2)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)

    def test_timer_over_limit_jump_restarts(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *self.timer_frame(0, 10.0, 0), sample_index=0)
        sample(confirmer, *self.timer_frame(1, 11.0, 1), sample_index=1)
        out = sample(confirmer, *self.timer_frame(2, 20.0, 2), sample_index=2)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)

    def test_timer_sample_gap_restarts(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *self.timer_frame(0, 10.0, 0), sample_index=0)
        sample(confirmer, *self.timer_frame(1, 11.0, 1), sample_index=1)
        out = sample(confirmer, *self.timer_frame(2, 12.0, 2), sample_index=3)
        self.assertEqual(out[0].status, RecognitionStatus.INFERRED)


class ValidationTests(unittest.TestCase):
    def test_invalid_flags(self):
        confirmer = HudEvidenceConfirmer()
        e, o = food_frame(0, 0)
        with self.assertRaises(TypeError):
            confirmer.accept(e, o, 0, profile_validated=1, age_signal_validated=False)
        with self.assertRaises(TypeError):
            confirmer.accept(e, o, 0, profile_validated=True, age_signal_validated=0)

    def test_reused_and_backward_index(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *food_frame(1, 1), sample_index=2)
        with self.assertRaises(ValueError):
            sample(confirmer, *food_frame(2, 2), sample_index=2)
        with self.assertRaises(ValueError):
            sample(confirmer, *food_frame(3, 3), sample_index=1)
        self.assertEqual(confirmer.last_sample_index, 2)


class AtomicityTests(unittest.TestCase):
    def test_failed_index_validation_atomic(self):
        confirmer = HudEvidenceConfirmer()
        sample(confirmer, *food_frame(1, 1), sample_index=0)
        sample(confirmer, *food_frame(2, 2), sample_index=1)
        with self.assertRaises(ValueError):
            sample(confirmer, *food_frame(3, 3), sample_index=1)
        self.assertEqual(confirmer.last_sample_index, 1)
        out = sample(confirmer, *food_frame(3, 3), sample_index=2)
        self.assertEqual(out[0].status, RecognitionStatus.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
