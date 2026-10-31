import unittest

from aoe4.smart_hud import (
    HudSampler,
    parse_age,
    parse_game_time,
    parse_hud_text,
    parse_resource,
)
from aoe4.smart_sampling import SamplingConfig
from aoe4.smart_runtime import (
    EstimatorConfig,
    FieldState,
    Observation,
    RecognitionStatus,
    StateEstimator,
)

OBS_ID = "obs-1"
FRAME = 5
AT = 10.0


def _ctx():
    return OBS_ID, FRAME, AT


class GameTimeParsingTests(unittest.TestCase):
    def test_mm_ss(self):
        obs = parse_game_time("12:34", *_ctx())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.field, "game_time")
        self.assertEqual(obs.value, 12 * 60 + 34)
        self.assertIsInstance(obs.value, float)

    def test_h_mm_ss(self):
        obs = parse_game_time("1:02:03", *_ctx())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.value, 3600 + 2 * 60 + 3)

    def test_zero_time(self):
        obs = parse_game_time("00:00", *_ctx())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.value, 0.0)

    def test_bad_time(self):
        for text in ("", "hello", "12", "12:60:00x", "abc:def", "1:2:3:4"):
            with self.subTest(text=text):
                self.assertIsNone(parse_game_time(text, *_ctx()))

    def test_seconds_bounds(self):
        self.assertIsNone(parse_game_time("12:60", *_ctx()))
        self.assertIsNone(parse_game_time("0:75", *_ctx()))
        self.assertIsNotNone(parse_game_time("12:59", *_ctx()))

    def test_minutes_bounds_with_hours(self):
        self.assertIsNone(parse_game_time("1:60:00", *_ctx()))
        self.assertIsNone(parse_game_time("2:99:01", *_ctx()))
        self.assertIsNotNone(parse_game_time("2:59:59", *_ctx()))

    def test_non_string_time(self):
        self.assertIsNone(parse_game_time(None, *_ctx()))
        self.assertIsNone(parse_game_time(123, *_ctx()))


class AgeParsingTests(unittest.TestCase):
    def test_all_age_forms(self):
        cases = {
            "age i": 1,
            "Age II": 2,
            "age III": 3,
            "age iv": 4,
            "dark": 1,
            "Feudal Age": 2,
            "castle": 3,
            "imperial age": 4,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                obs = parse_age(text, *_ctx())
                self.assertIsNotNone(obs)
                self.assertEqual(obs.field, "age")
                self.assertEqual(obs.value, expected)

    def test_bad_age(self):
        for text in ("", "age v", "medieval", "age", "IIV", None):
            with self.subTest(text=text):
                self.assertIsNone(parse_age(text, *_ctx()))


class ResourceParsingTests(unittest.TestCase):
    def test_plain(self):
        obs = parse_resource("food", "1250", *_ctx())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.field, "food")
        self.assertEqual(obs.value, 1250)

    def test_comma_thousands(self):
        obs = parse_resource("wood", "1,250", *_ctx())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.value, 1250)

    def test_period_thousands(self):
        obs = parse_resource("gold", "2.500", *_ctx())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.value, 2500)

    def test_zero_value(self):
        obs = parse_resource("stone", "0", *_ctx())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.value, 0)

    def test_reject_negatives(self):
        for text in ("-5", "-1,250", "12-"):
            with self.subTest(text=text):
                self.assertIsNone(parse_resource("food", text, *_ctx()))

    def test_reject_ambiguous_decimals(self):
        for text in ("1.25", "12.5", "1.2", "1.2345", "1,25", "1,2,3", "1,234,567"):
            with self.subTest(text=text):
                self.assertIsNone(parse_resource("food", text, *_ctx()))

    def test_reject_unknown_field(self):
        self.assertIsNone(parse_resource("not_a_field", "10", *_ctx()))

    def test_reject_plus_sign(self):
        self.assertIsNone(parse_resource("food", "+5", *_ctx()))


class ContextValidationTests(unittest.TestCase):
    def test_invalid_context(self):
        self.assertIsNone(parse_resource("food", "10", "", FRAME, AT))
        self.assertIsNone(parse_resource("food", "10", None, FRAME, AT))
        self.assertIsNone(parse_resource("food", "10", OBS_ID, -1, AT))
        self.assertIsNone(parse_resource("food", "10", OBS_ID, "5", AT))
        self.assertIsNone(parse_resource("food", "10", OBS_ID, True, AT))
        self.assertIsNone(parse_resource("food", "10", OBS_ID, FRAME, "now"))
        self.assertIsNone(parse_resource("food", "10", OBS_ID, FRAME, None))

    def test_invalid_context_game_time_and_age(self):
        self.assertIsNone(parse_game_time("1:00", "", FRAME, AT))
        self.assertIsNone(parse_age("dark", OBS_ID, -1, AT))


class HudSamplerTests(unittest.TestCase):
    def test_first_sample_and_half_second_boundary(self):
        sampler = HudSampler()
        self.assertEqual(len(sampler.sample(0.0, {"food": "100"}, FRAME)), 1)
        self.assertEqual(sampler.sample(0.49, {"food": "100"}, FRAME), ())
        self.assertEqual(len(sampler.sample(0.5, {"food": "100"}, FRAME)), 1)

    def test_parse_failure_does_not_consume_cadence(self):
        sampler = HudSampler(SamplingConfig(interval_seconds=0.5))
        class ExplodingMapping(dict):
            def __contains__(self, key):
                return True

            def __getitem__(self, key):
                raise TypeError("broken OCR mapping")

        with self.assertRaises(TypeError):
            sampler.sample(1.0, ExplodingMapping(food="100"), FRAME)
        self.assertIsNone(sampler.scheduler.last_sample)
        self.assertEqual(len(sampler.sample(1.0, {"food": "100"}, FRAME)), 1)


class ParseHudTextTests(unittest.TestCase):
    def test_deterministic_order(self):
        mapping = {
            "villagers_total": "20",
            "food": "1,000",
            "age": "feudal",
            "game_time": "10:30",
            "wood": "500",
        }
        result = parse_hud_text(mapping, FRAME, AT)
        self.assertEqual(
            [o.field for o in result],
            ["game_time", "age", "food", "wood", "villagers_total"],
        )

    def test_ids_generated(self):
        result = parse_hud_text({"food": "100", "wood": "200"}, FRAME, AT)
        self.assertEqual([o.observation_id for o in result], ["food-5", "wood-5"])

    def test_deterministic_repeat(self):
        mapping = {"game_time": "1:00", "food": "10", "age": "3"}
        first = parse_hud_text(mapping, FRAME, AT)
        second = parse_hud_text(mapping, FRAME, AT)
        self.assertEqual(first, second)

    def test_all_outputs_inferred_never_confirmed(self):
        result = parse_hud_text(
            {
                "game_time": "5:00",
                "age": "ii",
                "food": "1,000",
                "wood": "300",
                "gold": "400",
                "stone": "0",
                "population_current": "10",
                "population_cap": "50",
                "idle_villagers": "2",
                "villagers_food": "6",
                "villagers_wood": "4",
                "villagers_gold": "3",
                "villagers_stone": "1",
                "villagers_total": "14",
            },
            FRAME,
            AT,
        )
        self.assertEqual(len(result), 14)
        for obs in result:
            self.assertIsInstance(obs, Observation)
            self.assertIs(obs.status, RecognitionStatus.INFERRED)
            self.assertNotEqual(obs.status, RecognitionStatus.CONFIRMED)
            self.assertLessEqual(obs.confidence, 1.0)
            self.assertGreaterEqual(obs.confidence, 0.0)

    def test_invalid_entries_skipped(self):
        result = parse_hud_text({"food": "bad", "wood": "10"}, FRAME, AT)
        self.assertEqual([o.field for o in result], ["wood"])

    def test_non_mapping(self):
        self.assertEqual(parse_hud_text(None, FRAME, AT), ())
        self.assertEqual(parse_hud_text([("food", "1")], FRAME, AT), ())


class EstimatorIntegrationTests(unittest.TestCase):
    def _obs(self, field, value, frame, status=RecognitionStatus.INFERRED):
        return Observation(
            observation_id=f"{field}-{frame}",
            field=field,
            value=value,
            confidence=0.8,
            status=status,
            frame_id=frame,
            observed_at=float(frame),
        )

    def test_inferred_frames_do_not_confirm(self):
        config = EstimatorConfig(confirmation_frames=3, confidence_threshold=0.8)
        estimator = StateEstimator(config)
        for frame in range(1, 6):
            observations = parse_hud_text({"food": "100"}, frame, float(frame))
            state, events = estimator.ingest(observations, now=float(frame))
            self.assertEqual(events, ())
            self.assertIs(state.food.status, RecognitionStatus.INFERRED)
            self.assertEqual(state.food.value, 100)

    def test_inferred_game_time_does_not_confirm(self):
        config = EstimatorConfig(confirmation_frames=3)
        estimator = StateEstimator(config)
        observations = tuple(
            parse_hud_text({"game_time": "0:10"}, frame, float(frame))[0]
            for frame in range(1, 5)
        )
        state, _ = estimator.ingest(observations, now=4.0)
        self.assertNotEqual(state.game_time.status, RecognitionStatus.CONFIRMED)

    def test_confirmed_requires_confirmed_status(self):
        config = EstimatorConfig(confirmation_frames=2)
        estimator = StateEstimator(config)
        inferred = self._obs("food", 100, 1, RecognitionStatus.INFERRED)
        confirmed = self._obs("food", 100, 2, RecognitionStatus.CONFIRMED)
        state, _ = estimator.ingest((inferred,), now=1.0)
        self.assertIs(state.food.status, RecognitionStatus.INFERRED)
        state, _ = estimator.ingest((confirmed,), now=2.0)
        self.assertIsNot(state.food.status, RecognitionStatus.INFERRED)

    def test_out_of_band_confidence_not_confirmed(self):
        config = EstimatorConfig(confirmation_frames=2, confidence_threshold=0.9)
        estimator = StateEstimator(config)
        observations = (
            self._obs("food", 100, 1, RecognitionStatus.CONFIRMED),
            self._obs("food", 100, 2, RecognitionStatus.CONFIRMED),
        )
        state, _ = estimator.ingest(observations, now=2.0)
        self.assertNotEqual(state.food.status, RecognitionStatus.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
