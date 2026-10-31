"""Synchronous lazy offline OCR adapter for captured frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from aoe4.smart_capture import CapturedFrame


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got bool")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a number, got {type(value).__name__}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_box(box: object) -> Tuple[Tuple[float, float], ...]:
    if isinstance(box, (str, bytes)) or not hasattr(box, "__iter__"):
        raise ValueError(f"box must be a non-string iterable, got {type(box).__name__}")
    try:
        points = list(box)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"box must be iterable, got {type(box).__name__}") from exc
    if len(points) != 4:
        raise ValueError(f"box must contain exactly 4 points, got {len(points)}")
    normalized = []
    for index, point in enumerate(points):
        if isinstance(point, (str, bytes)) or not hasattr(point, "__iter__"):
            raise ValueError(f"box point {index} must be a pair")
        try:
            coords = list(point)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"box point {index} must be iterable") from exc
        if len(coords) != 2:
            raise ValueError(f"box point {index} must have 2 coordinates, got {len(coords)}")
        x = _finite_float(coords[0], f"box point {index} x")
        y = _finite_float(coords[1], f"box point {index} y")
        normalized.append((x, y))
    return tuple(normalized)


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a nonempty string")
        confidence = _finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "box", _normalize_box(self.box))


def _default_image_factory(frame: CapturedFrame) -> object:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for default image conversion") from exc
    return Image.frombytes(
        "RGBA", (frame.width, frame.height), frame.pixels, "raw", "BGRA"
    ).convert("RGB")


def _default_engine_factory() -> object:
    try:
        # RapidOCR exposes this through a module-level lazy import. Import the
        # implementation directly so standalone compilers can see the module.
        from rapidocr.main import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            f"rapidocr import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return RapidOCR(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": 1,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


class RapidOcrRecognizer:
    """Lazily create and reuse a RapidOCR engine to recognize frame text."""

    def __init__(
        self,
        engine_factory: Optional[Callable[[], object]] = None,
        image_factory: Optional[Callable[[CapturedFrame], object]] = None,
    ) -> None:
        if engine_factory is not None and not callable(engine_factory):
            raise TypeError("engine_factory must be callable")
        if image_factory is not None and not callable(image_factory):
            raise TypeError("image_factory must be callable")
        self._engine_factory: Callable[[], object] = (
            _default_engine_factory if engine_factory is None else engine_factory
        )
        self._image_factory: Callable[[CapturedFrame], object] = (
            _default_image_factory if image_factory is None else image_factory
        )
        self._engine: Optional[object] = None

    def _get_engine(self) -> object:
        engine = self._engine
        if engine is None:
            engine = self._engine_factory()
            if engine is None:
                raise RuntimeError("engine_factory returned None")
            self._engine = engine
        return engine

    @staticmethod
    def _extract_fields(result: object) -> Tuple[object, object, object]:
        fields = []
        for name in ("txts", "scores", "boxes"):
            try:
                value = getattr(result, name)
            except AttributeError as exc:
                raise ValueError(f"OCR result is missing field {name}") from exc
            fields.append(value)
        return fields[0], fields[1], fields[2]

    @staticmethod
    def _as_sequence(value: object, name: str) -> list:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            raise ValueError(f"{name} must be a non-string iterable")
        try:
            return list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{name} must be iterable") from exc

    def recognize(self, frame: CapturedFrame) -> Tuple[OcrLine, ...]:
        if not isinstance(frame, CapturedFrame):
            raise TypeError(f"frame must be a CapturedFrame, got {type(frame).__name__}")
        engine = self._get_engine()
        image = self._image_factory(frame)
        result = engine(image)  # type: ignore[operator]
        txts, scores, boxes = self._extract_fields(result)
        if txts is None and scores is None and boxes is None:
            return ()
        if txts is None or scores is None or boxes is None:
            raise ValueError("OCR result must provide all of txts, scores, and boxes")
        text_values = self._as_sequence(txts, "txts")
        score_values = self._as_sequence(scores, "scores")
        box_values = self._as_sequence(boxes, "boxes")
        if not (len(text_values) == len(score_values) == len(box_values)):
            raise ValueError(
                "OCR result field lengths differ: "
                f"txts={len(text_values)}, scores={len(score_values)}, "
                f"boxes={len(box_values)}"
            )
        lines = [
            OcrLine(text=text, confidence=score, box=box)
            for text, score, box in zip(text_values, score_values, box_values)
        ]
        return tuple(lines)

    def close(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            closer = getattr(engine, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> "RapidOcrRecognizer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False
