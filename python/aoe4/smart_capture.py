"""Thread-safe latest-frame capture primitives."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Union

Number = Union[int, float]


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_number(value: object, name: str) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        left = _require_int(self.left, "left")
        top = _require_int(self.top, "top")
        width = _require_int(self.width, "width")
        height = _require_int(self.height, "height")
        if left < 0 or top < 0:
            raise ValueError("region origin must be nonnegative")
        if width <= 0 or height <= 0:
            raise ValueError("region size must be positive")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    captured_at: Number
    width: int
    height: int
    pixels: bytes
    region: CaptureRegion

    def __post_init__(self) -> None:
        frame_id = _require_int(self.frame_id, "frame_id")
        captured_at = _require_number(self.captured_at, "captured_at")
        width = _require_int(self.width, "width")
        height = _require_int(self.height, "height")
        if frame_id < 0:
            raise ValueError("frame_id must be nonnegative")
        if not math.isfinite(captured_at) or captured_at < 0:
            raise ValueError("captured_at must be finite and nonnegative")
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not isinstance(self.region, CaptureRegion):
            raise TypeError("region must be a CaptureRegion")
        if width != self.region.width or height != self.region.height:
            raise ValueError("dimensions must match region")
        if not isinstance(self.pixels, bytes):
            raise TypeError("pixels must be bytes")


class LatestFrameSlot:
    """Hold the newest unread frame behind a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[CapturedFrame] = None
        self._last_id: Optional[int] = None
        self._published = 0
        self._dropped = 0

    def publish(self, frame: CapturedFrame) -> None:
        if not isinstance(frame, CapturedFrame):
            raise TypeError(f"frame must be a CapturedFrame, got {type(frame).__name__}")
        with self._lock:
            if self._last_id is not None and frame.frame_id <= self._last_id:
                raise ValueError(f"frame_id {frame.frame_id} must exceed {self._last_id}")
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame
            self._last_id = frame.frame_id
            self._published += 1

    def take_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    def peek_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._frame

    def clear(self) -> None:
        with self._lock:
            self._frame = None

    @property
    def published_count(self) -> int:
        with self._lock:
            return self._published

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped
