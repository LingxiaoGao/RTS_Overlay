import unittest
from dataclasses import FrozenInstanceError
from typing import List, Mapping, Tuple

from aoe4.smart_capture import CapturedFrame, CaptureRegion
from aoe4.smart_hud_reader import HudTextEvidence, HudOcrReader, text_mapping
from aoe4.smart_ocr import OcrLine
from aoe4.smart_regions import HudRegionSet
from aoe4.smart_runtime import FIELD_NAMES


def make_frame(width=4, height=2, frame_id=7, captured_at=1000.0):
    pixels = bytes(range(width * height * 4))
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=captured_at,
        width=width,
        height=height,
        pixels=pixels,
        region=CaptureRegion(left=0, top=0, width=width, height=height),
    )


def make_line(text, confidence, top_y, left_x, size=1.0):
    box = (
        (left_x, top_y),
        (left_x + size, top_y),
        (left_x + size, top_y + size),
        (left_x, top_y + size),
    )
    return OcrLine(text=text, confidence=confidence, box=box)


class ScriptedRecognizer:
    def __init__(self, responses):
        self._responses = responses
        self.calls: List[CapturedFrame] = []
        self._cursor = 0

    def recognize(self, frame):
        self.calls.append(frame)
        if self._cursor >= len(self._responses):
            raise AssertionError("recognizer called more times than scripted")
        response = self._responses[self._cursor]
        self._cursor += 1
        return response


def make_region_set():
    return HudRegionSet(
        {
            "game_time": CaptureRegion(left=1, top=0, width=2, height=2),
            "food": CaptureRegion(left=0, top=1, width=2, height=1),
        }
    )


def make_evidence(**overrides):
    data = {
        "field": "food",
        "text": "  120  ",
        "confidence": 0.5,
        "frame_id": 3,
        "observed_at": 12,
        "line_count": 1,
    }
    data.update(overrides)
    return HudTextEvidence(**data)


class HudTextEvidenceTests(unittest.TestCase):
    def test_construction(self):
        ev = HudTextEvidence(
            field="food",
            text="123",
            confidence=0.75,
            frame_id=2,
            observed_at=5,
            line_count=2,
        )
        self.assertEqual(ev.field, "food")
        self.assertEqual(ev.text, "123")
        self.assertEqual(ev.confidence, 0.75)
        self.assertEqual(ev.frame_id, 2)
        self.assertEqual(ev.observed_at, 5)
        self.assertEqual(ev.line_count, 2)

    def test_frozen(self):
        ev = make_evidence()
        with self.assertRaises(FrozenInstanceError):
            ev.text = "x"
        with self.assertRaises(FrozenInstanceError):
            ev.field = "gold"


class HudOcrReaderConstructorTests(unittest.TestCase):
    def test_rejects_invalid_region_set(self):
        with self.assertRaises(TypeError):
            HudOcrReader(region_set={}, recognizer=ScriptedRecognizer([]))

    def test_rejects_missing_recognize(self):
        with self.assertRaises(TypeError):
            HudOcrReader(region_set=make_region_set(), recognizer=object())

    def test_rejects_noncallable_recognize(self):
        class Bad:
            recognize = 5

        with self.assertRaises(TypeError):
            HudOcrReader(region_set=make_region_set(), recognizer=Bad())

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            HudOcrReader(make_region_set(), ScriptedRecognizer([]), min_confidence=1.5)
        with self.assertRaises(TypeError):
            HudOcrReader(make_region_set(), ScriptedRecognizer([]), min_confidence=True)

    def test_accepts_zero_and_one(self):
        self.assertEqual(
            HudOcrReader(make_region_set(), ScriptedRecognizer([]), 0.0).min_confidence,
            0.0,
        )
        self.assertEqual(
            HudOcrReader(make_region_set(), ScriptedRecognizer([]), 1).min_confidence,
            1.0,
        )


class HudOcrReaderReadTests(unittest.TestCase):
    def test_shared_population_crop_is_recognized_once_and_split(self):
        region = CaptureRegion(left=0, top=0, width=4, height=2)
        regions = HudRegionSet(
            {"population_current": region, "population_cap": region}
        )
        rec = ScriptedRecognizer([[make_line("83/120", 0.9, 0, 0)]])

        evidence = HudOcrReader(regions, rec).read(make_frame())

        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(evidence["population_current"].text, "83")
        self.assertEqual(evidence["population_cap"].text, "120")

    def test_crop_and_recognize_calls_with_correct_frames(self):
        frame = make_frame()
        rec = ScriptedRecognizer([(), ()])
        reader = HudOcrReader(make_region_set(), rec)
        result = reader.read(frame)
        self.assertEqual(result, {})
        self.assertEqual(len(rec.calls), 2)
        self.assertEqual(tuple(result.keys()), ())
        first = rec.calls[0]
        second = rec.calls[1]
        self.assertEqual(first.width, 2)
        self.assertEqual(first.height, 2)
        self.assertEqual(first.pixels, bytes(range(4, 12)) + bytes(range(20, 28)))
        self.assertEqual(second.width, 2)
        self.assertEqual(second.height, 1)
        self.assertEqual(second.pixels, bytes(range(16, 24)))
        self.assertEqual(
            second.region, CaptureRegion(left=0, top=1, width=2, height=1)
        )

    def test_recognize_order_matches_field_names(self):
        rec = ScriptedRecognizer([(), ()])
        HudOcrReader(make_region_set(), rec).read(make_frame())
        names = [n for n in FIELD_NAMES if n in ("game_time", "food")]
        self.assertEqual(names, ["game_time", "food"])
        self.assertEqual(len(rec.calls), len(names))

    def test_sorting_and_aggregation(self):
        frame = make_frame(frame_id=11, captured_at=99.5)
        lines = [
            make_line("c", 0.9, top_y=10, left_x=5),
            make_line("a", 0.4, top_y=5, left_x=9),
            make_line("b", 0.6, top_y=5, left_x=2),
            make_line("a2", 0.7, top_y=5, left_x=9),
        ]
        rec = ScriptedRecognizer([lines, ()])
        ev = HudOcrReader(make_region_set(), rec).read(frame)
        self.assertIn("game_time", ev)
        self.assertNotIn("food", ev)
        item = ev["game_time"]
        self.assertEqual(item.text, "b a a2 c")
        self.assertAlmostEqual(item.confidence, 0.4)
        self.assertEqual(item.line_count, 4)
        self.assertEqual(item.frame_id, 11)
        self.assertAlmostEqual(item.observed_at, 99.5)

    def test_threshold_filters_and_boundary(self):
        lines = [
            make_line("keep", 0.5, top_y=0, left_x=0),
            make_line("drop", 0.49, top_y=1, left_x=0),
        ]
        rec = ScriptedRecognizer([lines, ()])
        ev = HudOcrReader(make_region_set(), rec, min_confidence=0.5).read(make_frame())
        self.assertEqual(ev["game_time"].text, "keep")
        self.assertEqual(ev["game_time"].line_count, 1)

    def test_empty_generator_and_tuple_results(self):
        def gen():
            yield from ()

        rec = ScriptedRecognizer([gen(), ()])
        ev = HudOcrReader(make_region_set(), rec).read(make_frame())
        self.assertEqual(dict(ev), {})

    def test_invalid_result_string(self):
        rec = ScriptedRecognizer(["bad", ()])
        with self.assertRaises(TypeError):
            HudOcrReader(make_region_set(), rec).read(make_frame())

    def test_invalid_result_noniterable(self):
        rec = ScriptedRecognizer([42, ()])
        with self.assertRaises(TypeError):
            HudOcrReader(make_region_set(), rec).read(make_frame())

    def test_invalid_result_item(self):
        rec = ScriptedRecognizer([(object(),), ()])
        with self.assertRaises(TypeError):
            HudOcrReader(make_region_set(), rec).read(make_frame())

    def test_wrong_frame_type(self):
        with self.assertRaises(TypeError):
            HudOcrReader(make_region_set(), ScriptedRecognizer([])).read(object())

    def test_recognizer_exception_propagates(self):
        class Boom(Exception):
            pass

        class Raiser:
            def recognize(self, frame):
                raise Boom("nope")

        with self.assertRaises(Boom):
            HudOcrReader(make_region_set(), Raiser()).read(make_frame())



class TextMappingTests(unittest.TestCase):
    def test_valid_order(self):
        ev1 = make_evidence(field="game_time", text="1:00")
        ev2 = make_evidence(field="food", text="200")
        evidence = {"game_time": ev1, "food": ev2}
        mapped = text_mapping(evidence)
        self.assertEqual(list(mapped.items()), [("game_time", "1:00"), ("food", "200")])

    def test_empty_mapping(self):
        mapped = text_mapping({})
        self.assertEqual(dict(mapped), {})



if __name__ == "__main__":
    unittest.main()
