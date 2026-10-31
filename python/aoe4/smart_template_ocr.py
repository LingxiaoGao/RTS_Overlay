"""Template-matching OCR recognizer for fixed HUD regions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from aoe4.smart_capture import CapturedFrame
from aoe4.smart_ocr import OcrLine
from aoe4.smart_regions import HudRegionSet

_ALPHABET: Dict[str, Tuple[str, ...]] = {
    "food": tuple("0123456789"),
    "wood": tuple("0123456789"),
    "gold": tuple("0123456789"),
    "stone": tuple("0123456789"),
    "population_current": tuple("0123456789") + ("/",),
    "population_cap": tuple("0123456789") + ("/",),
    "game_time": tuple("0123456789") + (":",),
    "age": ("I", "V"),
}

_LABEL_KIND: Dict[str, str] = {
    **{str(d): "digits" for d in range(10)},
    ":": "punctuation",
    "/": "punctuation",
    "I": "roman",
    "V": "roman",
}

_GLYPH_W = 20
_GLYPH_H = 28


def _bgra_to_bgr(pixels: bytes, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(arr[:, :, :3])


def _foreground_mask(bgr: np.ndarray, field: str) -> np.ndarray:
    """Return a binary mask for the known HUD text colour of ``field``."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    white = (saturation <= 100) & (value >= 170)
    orange = (hue <= 30) & (saturation >= 70) & (value >= 120)
    if field == "age":
        gold = (hue <= 30) & (saturation >= 30) & (value >= 180)
        pale_gold = (saturation <= 80) & (value >= 160)
        foreground = gold | pale_gold
    elif field == "game_time":
        foreground = (saturation <= 100) & (value >= 160)
    elif field in ("population_current", "population_cap"):
        population_white = (saturation <= 110) & (value >= 145)
        foreground = population_white | orange
    else:
        foreground = white
    mask = foreground.astype(np.uint8) * 255
    if field == "game_time":
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 1), np.uint8))
    return mask


def _extract_glyphs(mask: np.ndarray, field: Optional[str] = None) -> List[np.ndarray]:
    """Extract left-to-right glyph crops from a binary mask."""
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape[:2]
    min_h = 3 if field == "game_time" else max(4, int(round(height * 0.3)))
    min_area = 3
    comps: List[Tuple[int, int, int, int]] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        if h < min_h or h > height:
            continue
        if w <= 0 or w > width:
            continue
        if field == "game_time" and (w < 3 or y < 4 or x < 2):
            continue
        if field == "age" and (h < 8 or w < 2 or w > 16 or y < 7):
            continue
        comps.append((int(x), int(y), int(w), int(h)))
    comps.sort(key=lambda c: c[0])
    if field == "game_time":
        merged: List[Tuple[int, int, int, int]] = []
        for box in comps:
            if merged:
                px, py, pw, ph = merged[-1]
                x, y, w, h = box
                overlap = min(px + pw, x + w) - max(px, x)
                if overlap > 0 and max(px + pw, x + w) - min(px, x) <= 8:
                    left, top = min(px, x), min(py, y)
                    right, bottom = max(px + pw, x + w), max(py + ph, y + h)
                    merged[-1] = (left, top, right - left, bottom - top)
                    continue
            merged.append(box)
        comps = [box for box in merged if box[3] >= 8]
    glyphs: List[np.ndarray] = []
    for x, y, w, h in comps:
        crop = mask[y:y + h, x:x + w]
        if w > 8:
            columns = np.count_nonzero(crop, axis=0)
            candidates = columns[3:w - 3]
            if candidates.size:
                split = int(np.argmin(candidates)) + 3
                if columns[split] <= max(2, h // 4):
                    for part in (crop[:, :split], crop[:, split:]):
                        occupied = np.argwhere(part > 0)
                        if occupied.size:
                            y0, x0 = occupied.min(axis=0)
                            y1, x1 = occupied.max(axis=0) + 1
                            glyphs.append(part[y0:y1, x0:x1])
                    continue
        glyphs.append(crop)
    return glyphs


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    """Scale ``glyph`` preserving aspect ratio into a centered 20x28 canvas."""
    h, w = glyph.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((_GLYPH_H, _GLYPH_W), dtype=np.uint8)
    scale = min(_GLYPH_W / w, _GLYPH_H / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_GLYPH_H, _GLYPH_W), dtype=np.uint8)
    ox = (_GLYPH_W - new_w) // 2
    oy = (_GLYPH_H - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    union = np.count_nonzero(a_bin | b_bin)
    if union == 0:
        return 0.0
    inter = np.count_nonzero(a_bin & b_bin)
    return float(inter) / float(union)


def _load_kind(directory: Path) -> Dict[str, List[np.ndarray]]:
    templates: Dict[str, List[np.ndarray]] = {}
    if not directory.is_dir():
        return templates
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir():
            continue
        label = {"slash": "/", "colon": ":"}.get(sub.name, sub.name)
        images: List[np.ndarray] = []
        for path in sorted(sub.glob("*.png")):
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            images.append(_normalize_glyph(raw))
        if images:
            templates[label] = images
    return templates


class TemplateHudRecognizer:
    """Match HUD glyphs against on-disk glyph templates."""

    def __init__(
        self,
        regions: HudRegionSet,
        template_dir: Union[str, Path],
    ) -> None:
        if not isinstance(regions, HudRegionSet):
            raise TypeError(
                f"regions must be a HudRegionSet, got {type(regions).__name__}"
            )
        root = Path(template_dir)
        if not root.is_dir():
            raise ValueError(f"template_dir does not exist: {root}")
        self._field_regions = dict(regions.regions.items())
        self._templates: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for kind in ("digits", "punctuation", "roman"):
            self._templates[kind] = _load_kind(root / kind)

    def _templates_for(self, field: str) -> Dict[str, List[np.ndarray]]:
        alphabet = _ALPHABET.get(field)
        if alphabet is None:
            return {}
        selected: Dict[str, List[np.ndarray]] = {}
        for label in alphabet:
            kind = _LABEL_KIND.get(label)
            if kind is None:
                continue
            entries = self._templates.get(kind, {}).get(label)
            if entries:
                selected[label] = entries
        return selected

    def _match(self, glyph: np.ndarray, templates: Dict[str, List[np.ndarray]]) -> Tuple[Optional[str], float]:
        best_label: Optional[str] = None
        best_score = -1.0
        for label, images in templates.items():
            for image in images:
                score = _iou(glyph, image)
                if score > best_score:
                    best_score = score
                    best_label = label
        if best_label is None or best_score < 0.3:
            return None, 0.0
        return best_label, best_score

    def recognize(self, frame: CapturedFrame) -> Tuple[OcrLine, ...]:
        if not isinstance(frame, CapturedFrame):
            raise TypeError(
                f"frame must be a CapturedFrame, got {type(frame).__name__}"
            )
        field: Optional[str] = None
        for name, region in self._field_regions.items():
            if region == frame.region:
                field = name
                break
        if field is None:
            return ()
        templates = self._templates_for(field)
        if not templates:
            return ()
        bgr = _bgra_to_bgr(frame.pixels, frame.width, frame.height)
        mask = _foreground_mask(bgr, field)
        glyphs = _extract_glyphs(mask, field)
        if not glyphs:
            return ()
        labels: List[str] = []
        scores: List[float] = []
        for glyph in glyphs:
            normalized = _normalize_glyph(glyph)
            label, score = self._match(normalized, templates)
            if label is None:
                continue
            labels.append(label)
            scores.append(score)
        if not labels:
            return ()
        text = "".join(labels)
        if field == "game_time" and len(text) == 4 and text.isdigit():
            text = text[:2] + ":" + text[2:]
        elif field == "age":
            text = "IV" if "V" in text else "I" * min(3, text.count("I"))
            if not text:
                return ()
        confidence = float(sum(scores) / len(scores))
        confidence = max(0.0, min(1.0, confidence))
        box = (
            (0.0, 0.0),
            (float(frame.width), 0.0),
            (float(frame.width), float(frame.height)),
            (0.0, float(frame.height)),
        )
        return (OcrLine(text=text, confidence=confidence, box=box),)

    def close(self) -> None:
        return None
