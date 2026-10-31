import math
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from aoe4.smart_capture import CaptureRegion, CapturedFrame
from aoe4.smart_ocr import OcrLine, RapidOcrRecognizer


def make_frame(frame_id=1):
    region = CaptureRegion(left=0, top=0, width=1, height=1)
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=0.5,
        width=1,
        height=1,
        pixels=b"\x00\x00\x00\xff",
        region=region,
    )


def make_box(x=0.0, y=0.0):
    return ((x, y), (x + 1.0, y), (x + 1.0, y + 1.0), (x, y + 1.0))


class FakeScalar:
    """Float-convertible object accepted by _finite_float."""

    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)


class BadScalar:
    def __float__(self):
        raise TypeError("no float")


class FakeEngine:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.closed = 0

    def __call__(self, image):
        self.calls.append(image)
        return self.result

    def close(self):
        self.closed += 1


def ok_result(txts=None, scores=None, boxes=None):
    if txts is None:
        txts = ["hello"]
    if scores is None:
        scores = [0.9]
    if boxes is None:
        boxes = [make_box()]
    return SimpleNamespace(txts=txts, scores=scores, boxes=boxes)


class OcrLineTests(unittest.TestCase):
    def test_valid_line_is_frozen_and_normalized(self):
        line = OcrLine("text", 0.4, ((0, 1), (2, 3), (4, 5), (6, 7)))
        self.assertEqual(line.text, "text")
        self.assertEqual(line.confidence, 0.4)
        self.assertIsInstance(line.confidence, float)
        self.assertIsInstance(line.box, tuple)
        self.assertIsInstance(line.box[0], tuple)
        self.assertIsInstance(line.box[0][0], float)
        with self.assertRaises(FrozenInstanceError):
            line.text = "other"

    def test_invalid_text(self):
        for text in ("", "   ", None, 5):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    OcrLine(text, 0.5, make_box())

    def test_invalid_confidence(self):
        for conf in (True, False, math.nan, math.inf, -0.1, 1.1, "x", None, BadScalar()):
            with self.subTest(conf=conf):
                with self.assertRaises(ValueError):
                    OcrLine("t", conf, make_box())

    def test_malformed_boxes(self):
        bad_boxes = (
            "abcd",
            b"abcd",
            5,
            ((0, 0), (1, 1), (2, 2)),
            ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4)),
            ("pt", (1, 1), (2, 2), (3, 3)),
            (5, (1, 1), (2, 2), (3, 3)),
            ((0,), (1, 1), (2, 2), (3, 3)),
            ((0, 0, 0), (1, 1), (2, 2), (3, 3)),
            ((math.inf, 0), (1, 1), (2, 2), (3, 3)),
            ((0, 0), (1, 1), (2, 2), (3, 3.5e309 * 0 if False else (4, 5))),
        )
        for box in bad_boxes:
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    OcrLine("t", 0.5, box)

    def test_accepts_generators_and_custom_scalars(self):
        gen = (pair for pair in ((0, 0), (1, 0), (1, 1), (0, 1)))
        line = OcrLine("t", FakeScalar(0.25), gen)
        self.assertEqual(line.confidence, 0.25)
        self.assertIsInstance(line.confidence, float)
        self.assertEqual(line.box, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        box_with_scalars = tuple(
            (FakeScalar(x), FakeScalar(y))
            for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))
        )
        line2 = OcrLine("t", 0.5, box_with_scalars)
        self.assertEqual(line2.box, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        self.assertTrue(all(isinstance(c, float) for p in line2.box for c in p))


class FactoryTests(unittest.TestCase):
    def test_invalid_factories(self):
        with self.assertRaises(TypeError):
            RapidOcrRecognizer(engine_factory=5)
        with self.assertRaises(TypeError):
            RapidOcrRecognizer(image_factory="x")

    def test_default_engine_factory_limits_threads(self):
        import sys
        from unittest import mock

        from aoe4.smart_ocr import _default_engine_factory

        fake_main = mock.MagicMock()
        fake_main.RapidOCR = mock.MagicMock(return_value="engine")
        with mock.patch.dict(
            sys.modules,
            {"rapidocr": mock.MagicMock(), "rapidocr.main": fake_main},
        ):
            self.assertEqual(_default_engine_factory(), "engine")

        fake_main.RapidOCR.assert_called_once_with(
            params={
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

    def test_lazy_creation_and_reuse(self):
        created = []

        def ef():
            eng = FakeEngine(ok_result())
            created.append(eng)
            return eng

        sentinel = object()
        rec = RapidOcrRecognizer(engine_factory=ef, image_factory=lambda f: sentinel)
        self.assertEqual(created, [])
        frame = make_frame()
        rec.recognize(frame)
        rec.recognize(frame)
        self.assertEqual(len(created), 1)
        self.assertEqual(len(created[0].calls), 2)
        self.assertIs(created[0].calls[0], sentinel)

    def test_image_factory_receives_exact_frame(self):
        seen = []

        def imgf(frame):
            seen.append(frame)
            return "IMG"

        eng = FakeEngine(ok_result())
        rec = RapidOcrRecognizer(engine_factory=lambda: eng, image_factory=imgf)
        frame = make_frame()
        rec.recognize(frame)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], frame)
        self.assertIs(eng.calls[0], "IMG")

    def test_engine_factory_returning_none(self):
        rec = RapidOcrRecognizer(
            engine_factory=lambda: None, image_factory=lambda f: object()
        )
        with self.assertRaises(RuntimeError):
            rec.recognize(make_frame())

    def test_non_frame_rejected(self):
        rec = RapidOcrRecognizer(
            engine_factory=lambda: FakeEngine(ok_result()),
            image_factory=lambda f: object(),
        )
        with self.assertRaises(TypeError):
            rec.recognize({"not": "frame"})


class RecognizeTests(unittest.TestCase):
    def recognize(self, result):
        eng = FakeEngine(result)
        rec = RapidOcrRecognizer(engine_factory=lambda: eng, image_factory=lambda f: "img")
        return rec.recognize(make_frame()), eng

    def test_order_and_normalization(self):
        result = SimpleNamespace(
            txts=["a", "b"],
            scores=[FakeScalar(0.1), 0.8],
            boxes=[
                ((0, 0), (1, 0), (1, 1), (0, 1)),
                tuple((FakeScalar(i), FakeScalar(i + 1)) for i in range(4)),
            ],
        )
        lines, _ = self.recognize(result)
        self.assertEqual([l.text for l in lines], ["a", "b"])
        self.assertEqual([l.confidence for l in lines], [0.1, 0.8])
        self.assertIsInstance(lines[0].confidence, float)
        self.assertEqual(lines[0].box[0], (0.0, 0.0))
        self.assertIsInstance(lines[1].box[0][0], float)

    def test_all_none_is_empty(self):
        lines, eng = self.recognize(SimpleNamespace(txts=None, scores=None, boxes=None))
        self.assertEqual(lines, ())
        self.assertEqual(len(eng.calls), 1)

    def test_partial_none(self):
        for result in (
            SimpleNamespace(txts=["a"], scores=None, boxes=[make_box()]),
            SimpleNamespace(txts=None, scores=[0.5], boxes=[make_box()]),
            SimpleNamespace(txts=["a"], scores=[0.5], boxes=None),
        ):
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    self.recognize(result)

    def test_missing_attribute(self):
        for name in ("txts", "scores", "boxes"):
            fields = {"txts": ["a"], "scores": [0.5], "boxes": [make_box()]}
            del fields[name]
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.recognize(SimpleNamespace(**fields))

    def test_noniterable_and_string_containers(self):
        for field in ("txts", "scores", "boxes"):
            for bad in ("str", 5):
                fields = {"txts": ["a"], "scores": [0.5], "boxes": [make_box()]}
                fields[field] = bad
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValueError):
                        self.recognize(SimpleNamespace(**fields))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            self.recognize(ok_result(txts=["a", "b"], scores=[0.5], boxes=[make_box()]))

    def test_generator_containers(self):
        result = SimpleNamespace(
            txts=(t for t in ["z"]),
            scores=(s for s in (FakeScalar(0.7),)),
            boxes=(b for b in [make_box()]),
        )
        lines, _ = self.recognize(result)
        self.assertEqual(lines[0].text, "z")
        self.assertEqual(lines[0].confidence, 0.7)

    def test_malformed_text(self):
        for txt in ("", "  ", 5, None):
            with self.subTest(txt=txt):
                with self.assertRaises(ValueError):
                    self.recognize(ok_result(txts=[txt]))

    def test_malformed_score(self):
        for score in (True, math.nan, math.inf, -0.5, 1.5, BadScalar()):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    self.recognize(ok_result(scores=[score]))

    def test_malformed_box(self):
        for box in ("bad", 3, ((0, 0), (1, 1))):
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    self.recognize(ok_result(boxes=[box]))


class LifecycleTests(unittest.TestCase):
    def test_idempotent_close_and_recreation(self):
        engines = []

        def ef():
            eng = FakeEngine(ok_result())
            engines.append(eng)
            return eng

        rec = RapidOcrRecognizer(engine_factory=ef, image_factory=lambda f: "img")
        frame = make_frame()
        rec.recognize(frame)
        rec.close()
        rec.close()
        self.assertEqual(engines[0].closed, 1)
        rec.recognize(frame)
        self.assertEqual(len(engines), 2)

    def test_close_without_creation(self):
        rec = RapidOcrRecognizer(
            engine_factory=lambda: self.fail("should not create"),
            image_factory=lambda f: object(),
        )
        rec.close()

    def test_engine_without_close(self):
        class Bare:
            def __call__(self, image):
                return SimpleNamespace(txts=None, scores=None, boxes=None)

        rec = RapidOcrRecognizer(engine_factory=Bare, image_factory=lambda f: "img")
        rec.recognize(make_frame())
        rec.close()

    def test_context_manager_closes(self):
        eng = FakeEngine(ok_result())
        with RapidOcrRecognizer(engine_factory=lambda: eng, image_factory=lambda f: "i") as rec:
            self.assertIsInstance(rec, RapidOcrRecognizer)
            rec.recognize(make_frame())
        self.assertEqual(eng.closed, 1)
        rec.close()
        self.assertEqual(eng.closed, 1)


if __name__ == "__main__":
    unittest.main()
