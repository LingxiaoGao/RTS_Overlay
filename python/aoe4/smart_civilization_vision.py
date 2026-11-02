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
COMMAND_PANEL_MIN_DARK_FRACTION = 0.45
FLAG_PALETTE_WEIGHT = 0.50
PREFERRED_CIVILIZATION_SCORE_RATIO = 0.85
LOADING_MATCH_THRESHOLD = 0.30
LOADING_LEAD_RATIO = 1.25
LOADING_DARK_FOREGROUND_THRESHOLD = 0.05
LOADING_YELLOW_BACKGROUND_THRESHOLD = 0.70
LOADING_MISSING_DARK_PENALTY = 0.40
LOADING_RED_FOREGROUND_THRESHOLD = 0.10
LOADING_MISSING_RED_THRESHOLD = 0.03
LOADING_EXTRA_RED_PENALTY = 0.40


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
        self._flag_histograms = {
            civ: _hs_histogram(template) for civ, template in self._flag_templates.items()
        }
        self._flag_dark_fraction = {
            civ: _dark_fraction(template) for civ, template in self._flag_templates.items()
        }
        self._flag_red_fraction = {
            civ: _red_fraction(template) for civ, template in self._flag_templates.items()
        }
        self._flag_palettes = {
            civ: _flag_palette(template) for civ, template in self._flag_templates.items()
        }
        self._last_flag_hash: int | None = None
        self._last_flag_result: CivilizationCue | None = None

    def recognize_in_game(
        self,
        flag_bgr: np.ndarray,
        production_buttons_bgr: np.ndarray,
        preferred_civilization: str | None = None,
    ) -> tuple[CivilizationCue | None, bool]:
        active = self._production_button_active(production_buttons_bgr)
        if not self._command_panel_present(production_buttons_bgr):
            return None, active
        return self._match_flag(flag_bgr, preferred_civilization), active

    def _match_flag(
        self, crop: np.ndarray, preferred_civilization: str | None = None
    ) -> CivilizationCue | None:
        if not _valid_crop(crop):
            return None
        fingerprint = _difference_hash(crop)
        if self._last_flag_hash is not None:
            if bin(fingerprint ^ self._last_flag_hash).count("1") <= 4:
                return self._last_flag_result
        scores = []
        palette_width = max(1, int(round(crop.shape[1] * 0.40)))
        observed_palette = _flag_palette(crop[:, :palette_width])
        for civilization, template in self._flag_templates.items():
            shape_score = _best_template_score(template, crop, crop.shape[0])
            palette_score = _palette_similarity(
                observed_palette, self._flag_palettes[civilization]
            )
            score = (
                FLAG_PALETTE_WEIGHT * palette_score
                + (1.0 - FLAG_PALETTE_WEIGHT) * shape_score
            )
            scores.append((civilization, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        best_civ, best_score = scores[0]
        if preferred_civilization is not None:
            preferred = next(
                (item for item in scores if item[0] == preferred_civilization), None
            )
            if (
                preferred is not None
                and preferred[1] >= FLAG_MATCH_THRESHOLD
                and preferred[1]
                >= best_score * PREFERRED_CIVILIZATION_SCORE_RATIO
            ):
                best_civ, best_score = preferred
        if best_civ is None or best_score < FLAG_MATCH_THRESHOLD:
            result = None
        else:
            second = max(
                (score for civilization, score in scores if civilization != best_civ),
                default=0.0,
            )
            confidence = max(0.0, min(1.0, 1.0 - second / max(best_score, 1e-6)))
            result = CivilizationCue(best_civ, confidence)
        self._last_flag_hash = fingerprint
        self._last_flag_result = result
        return result

    def recognize_loading(self, frame_bgr: np.ndarray) -> CivilizationCue | None:
        """Return a provisional civilization from the broad loading background."""
        if not _valid_crop(frame_bgr):
            return None
        region = frame_bgr[:, frame_bgr.shape[1] // 2 :]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Dark player nameplates are overlays rather than civilization evidence.
        visible_background = (hsv[:, :, 2] >= 85).astype(np.uint8) * 255
        histogram = _hs_histogram(region, visible_background)
        # The upper-right loading background is free of player cards.  Keep a
        # tiny spatial cue that separates yellow flags with a dark emblem
        # (for example the Order of the Dragon) from yellow flags whose emblem
        # is coloured (for example the Ayyubids).  This is a few thresholded
        # pixel counts, not another OCR/model pass.
        background = frame_bgr[
            : max(1, int(frame_bgr.shape[0] * 0.40)), frame_bgr.shape[1] // 2 :
        ]
        background_dark = _dark_fraction(background)
        background_yellow = _yellow_fraction(background)
        background_red = _red_fraction(background)

        def loading_score(civilization: str, template: np.ndarray) -> float:
            score = float(
                cv2.compareHist(
                    histogram,
                    self._flag_histograms[civilization],
                    cv2.HISTCMP_INTERSECT,
                )
            )
            if (
                background_yellow >= LOADING_YELLOW_BACKGROUND_THRESHOLD
                and background_dark >= LOADING_DARK_FOREGROUND_THRESHOLD
                and self._flag_dark_fraction[civilization]
                < LOADING_DARK_FOREGROUND_THRESHOLD
            ):
                score *= LOADING_MISSING_DARK_PENALTY
            if (
                background_red < LOADING_MISSING_RED_THRESHOLD
                and self._flag_red_fraction[civilization]
                >= LOADING_RED_FOREGROUND_THRESHOLD
            ):
                score *= LOADING_EXTRA_RED_PENALTY
            return score

        scores = sorted(
            (
                (
                    civilization,
                    loading_score(civilization, template),
                )
                for civilization, template in self._flag_histograms.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not scores:
            return None
        best_civ, best = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        ratio = best / max(second, 1e-6)
        if best < LOADING_MATCH_THRESHOLD or ratio < LOADING_LEAD_RATIO:
            return None
        return CivilizationCue(best_civ, min(1.0, 1.0 - second / best))

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

    def _command_panel_present(self, crop: np.ndarray) -> bool:
        return _valid_crop(crop) and _dark_fraction(crop) >= COMMAND_PANEL_MIN_DARK_FRACTION


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


def _difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def _hs_histogram(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], mask, [18, 16], [0, 180, 0, 256]
    )
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram


def _dark_fraction(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2] < 85))


def _yellow_fraction(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = (
        (hsv[:, :, 0] > 18)
        & (hsv[:, :, 0] < 40)
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 100)
    )
    return float(np.mean(yellow))


def _red_fraction(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 90)
        & (hsv[:, :, 2] > 80)
    )
    return float(np.mean(red))


def _flag_palette(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    visible = value > 50
    bins = (
        (((hue < 10) | (hue >= 170)) & (saturation > 55)),
        (((hue >= 10) & (hue < 35)) & (saturation > 55)),
        (((hue >= 35) & (hue < 85)) & (saturation > 55)),
        (((hue >= 85) & (hue < 130)) & (saturation > 55)),
        (((hue >= 130) & (hue < 170)) & (saturation > 55)),
        ((saturation <= 55) & (value > 95)),
    )
    palette = np.asarray([np.count_nonzero(mask & visible) for mask in bins], dtype=float)
    total = float(palette.sum())
    return palette / total if total > 0 else palette


def _palette_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if not np.any(left) or not np.any(right):
        return 0.0
    return max(0.0, 1.0 - float(np.abs(left - right).sum()) / 2.0)


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
