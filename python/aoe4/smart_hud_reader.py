"""HUD OCR reader orchestration over fixed regions and recognizer output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple

from aoe4.smart_capture import CapturedFrame
from aoe4.smart_ocr import OcrLine
from aoe4.smart_regions import HudRegionSet
from aoe4.smart_runtime import FIELD_NAMES


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_confidence(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


@dataclass(frozen=True)
class HudTextEvidence:
    """Aggregated OCR evidence for one HUD field in one frame."""

    field: str
    text: str
    confidence: float
    frame_id: int
    observed_at: object
    line_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.field, str):
            raise TypeError(f"field must be a str, got {type(self.field).__name__}")
        if self.field not in FIELD_NAMES:
            raise ValueError(f"unknown field name: {self.field!r}")
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str, got {type(self.text).__name__}")
        stripped = self.text.strip()
        if not stripped:
            raise ValueError("text must be a nonempty string")
        object.__setattr__(self, "text", stripped)
        object.__setattr__(
            self, "confidence", _require_confidence(self.confidence, "confidence")
        )
        frame_id = _require_int(self.frame_id, "frame_id")
        if frame_id < 0:
            raise ValueError("frame_id must be nonnegative")
        observed_at = self.observed_at
        if isinstance(observed_at, bool) or not isinstance(observed_at, (int, float)):
            raise TypeError(
                f"observed_at must be a number, got {type(observed_at).__name__}"
            )
        observed = float(observed_at)
        if not math.isfinite(observed):
            raise ValueError("observed_at must be finite")
        if observed < 0:
            raise ValueError("observed_at must be nonnegative")
        line_count = _require_int(self.line_count, "line_count")
        if line_count <= 0:
            raise ValueError("line_count must be positive")


def _top_y(line: OcrLine) -> float:
    return min(point[1] for point in line.box)


def _left_x(line: OcrLine) -> float:
    return min(point[0] for point in line.box)


def _validated_lines(result: object) -> List[OcrLine]:
    if isinstance(result, (str, bytes)) or not isinstance(result, Iterable):
        raise TypeError(
            f"recognizer result must be an iterable of OcrLine, "
            f"got {type(result).__name__}"
        )
    lines = list(result)
    for index, line in enumerate(lines):
        if not isinstance(line, OcrLine):
            raise TypeError(
                f"recognizer result item {index} must be an OcrLine, "
                f"got {type(line).__name__}"
            )
    return lines


class HudOcrReader:
    """Crop fixed HUD regions and aggregate recognizable text per field."""

    def __init__(
        self,
        region_set: HudRegionSet,
        recognizer: object,
        min_confidence: float = 0.0,
    ) -> None:
        if not isinstance(region_set, HudRegionSet):
            raise TypeError(
                f"region_set must be a HudRegionSet, got {type(region_set).__name__}"
            )
        recognize = getattr(recognizer, "recognize", None)
        if not callable(recognize):
            raise TypeError("recognizer must expose a callable recognize(frame)")
        self._region_set = region_set
        self._recognizer = recognizer
        self._min_confidence = _require_confidence(min_confidence, "min_confidence")

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    def read(self, frame: CapturedFrame) -> Mapping[str, HudTextEvidence]:
        if not isinstance(frame, CapturedFrame):
            raise TypeError(
                f"frame must be a CapturedFrame, got {type(frame).__name__}"
            )
        cropped = self._region_set.crop(frame)
        evidence = {}
        for field, region_frame in cropped.items():
            result = self._recognizer.recognize(region_frame)
            lines = _validated_lines(result)
            indexed: List[Tuple[float, float, int, OcrLine]] = []
            for index, line in enumerate(lines):
                if line.confidence < self._min_confidence:
                    continue
                indexed.append((_top_y(line), _left_x(line), index, line))
            if not indexed:
                continue
            indexed.sort(key=lambda item: (item[0], item[1], item[2]))
            retained = [item[3] for item in indexed]
            text = " ".join(line.text.strip() for line in retained)
            confidence = min(line.confidence for line in retained)
            evidence[field] = HudTextEvidence(
                field=field,
                text=text,
                confidence=confidence,
                frame_id=frame.frame_id,
                observed_at=frame.captured_at,
                line_count=len(retained),
            )
        return evidence


def text_mapping(evidence: Mapping[str, HudTextEvidence]) -> Mapping[str, str]:
    """Return a field -> text mapping of validated evidence."""
    return {key: item.text for key, item in evidence.items()}
