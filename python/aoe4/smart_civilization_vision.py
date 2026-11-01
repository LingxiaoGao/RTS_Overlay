"""Bounded OpenCV recognizer for in-game civilization cues."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from aoe4.smart_civilization import CivilizationCue


FLAG_MATCH_THRESHOLD = 0.45
MATCH_SCALES = (0.5, 0.75, 1.0)
PRODUCTION_BUTTON_MIN_VALUE = 70.0


class CivilizationVision:
    def __init__(
        self,
        flag_templates: Mapping[str, str | Path],
    ) -> None:
        if not flag_templates:
            raise ValueError("flag_templates must be nonempty")
        self._flag_templates = {
            civ: _load_template(path) for civ, path in flag_templates.items()
        }

    def recognize_in_game(
        self, flag_bgr: np.ndarray, production_buttons_bgr: np.ndarray
    ) -> tuple[CivilizationCue | None, bool]:
        return self._match_flag(flag_bgr), self._production_button_active(
            production_buttons_bgr
        )

    def _match_flag(self, crop: np.ndarray) -> CivilizationCue | None:
        if not _valid_crop(crop):
            return None
        best_civ = None
        best_score = 0.0
        for civilization, template in self._flag_templates.items():
            score = _best_template_score(template, crop, crop.shape[0])
            if score > best_score:
                best_score = score
                best_civ = civilization
        if best_civ is None or best_score < FLAG_MATCH_THRESHOLD:
            return None
        return CivilizationCue(best_civ, min(1.0, max(0.0, best_score)))

    def _production_button_active(self, crop: np.ndarray) -> bool:
        if not _valid_crop(crop):
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        midpoint = crop.shape[1] // 2
        if midpoint == 0:
            return False
        return any(
            float(cell[:, :, 2].mean()) >= PRODUCTION_BUTTON_MIN_VALUE
            for cell in (hsv[:, :midpoint], hsv[:, midpoint:])
            if cell.size
        )


def _load_template(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"cannot load template: {path!r}")
    return image


def _valid_crop(crop: np.ndarray) -> bool:
    return (
        isinstance(crop, np.ndarray)
        and crop.ndim == 3
        and crop.shape[2] == 3
        and crop.dtype == np.uint8
        and crop.shape[0] > 0
        and crop.shape[1] > 0
    )


def _best_template_score(
    template: np.ndarray, crop: np.ndarray, target_height: int
) -> float:
    crop_h, crop_w = crop.shape[:2]
    best = 0.0
    for scale in MATCH_SCALES:
        height = max(1, int(round(target_height * scale)))
        width = max(1, int(round(height * template.shape[1] / template.shape[0])))
        if height > crop_h or width > crop_w:
            continue
        resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(crop, resized, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, _ = cv2.minMaxLoc(result)
        best = max(best, float(max_value))
    return best
