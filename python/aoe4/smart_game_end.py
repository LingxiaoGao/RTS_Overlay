"""Lightweight AoE4 victory/defeat banner detection."""

from __future__ import annotations

import cv2
import numpy as np


def has_game_end_banner(frame_bgr: np.ndarray) -> bool:
    """Return whether the fixed central AoE4 result banner is visible."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        return False
    if frame_bgr.dtype != np.uint8 or frame_bgr.shape[2] != 3:
        return False
    height, width = frame_bgr.shape[:2]
    if height < 32 or width < 64:
        return False

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    middle = hsv[int(0.34 * height) : int(0.70 * height), int(0.10 * width) : int(0.90 * width)]
    dark_navy = (
        (middle[:, :, 0] >= 95)
        & (middle[:, :, 0] <= 125)
        & (middle[:, :, 1] >= 35)
        & (middle[:, :, 2] <= 140)
    )
    row_coverage = dark_navy.mean(axis=1)
    qualifying = row_coverage >= 0.80
    longest = current = 0
    for present in qualifying:
        current = current + 1 if present else 0
        longest = max(longest, current)
    return longest >= max(3, round(height * 0.09)) and float(row_coverage.max()) >= 0.90


class GameEndDetector:
    """Confirm a result banner across adjacent 500 ms capture frames."""

    def __init__(self, required_streak: int = 2) -> None:
        if required_streak < 1:
            raise ValueError("required_streak must be >= 1")
        self._required_streak = required_streak
        self._streak = 0

    def observe(self, frame_bgr: np.ndarray) -> bool:
        if has_game_end_banner(frame_bgr):
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self._required_streak

    def reset(self) -> None:
        self._streak = 0
