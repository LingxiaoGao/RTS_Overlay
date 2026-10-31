"""Offline evaluator for TemplateHudRecognizer on an AoE4 video.

Hobby-scale utility: decode a video sequentially, sample frames nearest a
fixed interval, crop the configured HUD regions, and recognise the numeric
fields once per sample. Writes ``results.csv`` and ``summary.txt``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aoe4.smart_capture import CapturedFrame, CaptureRegion  # noqa: E402
from aoe4.smart_controller_config import SmartControllerConfig  # noqa: E402
from aoe4.smart_regions import crop_frame, scale_regions  # noqa: E402
from aoe4.smart_template_ocr import TemplateHudRecognizer  # noqa: E402

_FIELDS = (
    "game_time",
    "age",
    "food",
    "wood",
    "gold",
    "stone",
    "population_current",
)
_RESOURCE_FIELDS = ("food", "wood", "gold", "stone")
_POP_RE = re.compile(r"^([0-9]{1,3})/([0-9]{1,3})$")
_TIME_RE = re.compile(r"^([0-9]{2}):([0-9]{2})$")
_RES_RE = re.compile(r"^[0-9]{1,4}$")

# Standard age-up costs: (food, gold) from current age to next.
_AGE_COSTS = {
    2: (400, 200),
    3: (1200, 600),
    4: (2400, 1200),
}
_DROP_TOLERANCE = 0.30
_CANDIDATE_WINDOW_SECONDS = 180.0
_RECENT_FRAMES = 10
_MIN_VOTES = 6


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TemplateHudRecognizer on an AoE4 video.",
    )
    parser.add_argument("video", help="path to the input video")
    parser.add_argument(
        "--config",
        default="python/demo_data/smart_hud_config_1280x720.json",
        help="path to the smart controller config JSON",
    )
    parser.add_argument(
        "--templates",
        default="python/demo_data/hud_templates_1280x720",
        help="path to the glyph template directory",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.5,
        help="sampling interval in seconds",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory (default: derived from video name)",
    )
    return parser.parse_args(argv)


def _is_valid(field: str, text: str) -> bool:
    if not text:
        return False
    if field in _RESOURCE_FIELDS:
        return _RES_RE.match(text) is not None
    if field == "population_current":
        match = _POP_RE.match(text)
        if match is None:
            return False
        current, cap = int(match.group(1)), int(match.group(2))
        return current <= cap <= 999
    if field == "game_time":
        match = _TIME_RE.match(text)
        if match is None:
            return False
        return int(match.group(1)) < 60
    if field == "age":
        return set(text) <= {"I", "V"} and 1 <= _normalize_age(text) <= 4
    return False


def _normalize_age(text: str) -> int:
    """Map a raw age string to 1..4. Any ``V`` means 4, else count ``I``."""
    if "V" in text:
        return 4
    return max(0, min(3, text.count("I")))


def _parse_int(text: str) -> Optional[int]:
    if not text or not _RES_RE.match(text):
        return None
    return int(text)


def _parse_time_seconds(text: str) -> Optional[float]:
    match = _TIME_RE.match(text)
    if match is None:
        return None
    minutes, seconds = int(match.group(1)), int(match.group(2))
    if seconds >= 60:
        return None
    return minutes * 60.0 + seconds


@dataclass
class _Row:
    index: int
    frame_id: int
    timestamp: float
    fields: Dict[str, str]
    recognize_ms: float
    field_ms: Dict[str, float]


@dataclass
class _AgeTracker:
    current_age: int = 1
    candidates: List[Tuple[float, int, int, int]] = field(default_factory=list)
    transitions: List[Tuple[float, int]] = field(default_factory=list)
    candidate_age: Optional[int] = None
    candidate_at: Optional[float] = None
    recent_visual: deque = field(default_factory=lambda: deque(maxlen=_RECENT_FRAMES))

    def observe(
        self,
        timestamp: float,
        prev: Optional[Dict[str, Optional[int]]],
        cur: Dict[str, Optional[int]],
        visual_age: int,
    ) -> None:
        """Record candidates and confirm age transitions from resource drops."""
        next_age = self.current_age + 1
        if prev is not None and next_age in _AGE_COSTS:
            cost_food, cost_gold = _AGE_COSTS[next_age]
            before_food, after_food = prev.get("food"), cur.get("food")
            before_gold, after_gold = prev.get("gold"), cur.get("gold")
            if (
                _matches_drop(before_food, after_food, cost_food)
                and _matches_drop(before_gold, after_gold, cost_gold)
            ):
                food_drop = int(before_food) - int(after_food)  # type: ignore[arg-type]
                gold_drop = int(before_gold) - int(after_gold)  # type: ignore[arg-type]
                self.candidates.append((timestamp, next_age, food_drop, gold_drop))
                self.candidate_age = next_age
                self.candidate_at = timestamp
                self.recent_visual.clear()

        self.recent_visual.append(visual_age)
        if self.candidate_age is not None and self.candidate_at is not None:
            if timestamp - self.candidate_at > _CANDIDATE_WINDOW_SECONDS:
                self.candidate_age = None
                self.candidate_at = None
                self.recent_visual.clear()
            elif self.recent_visual.count(self.candidate_age) >= _MIN_VOTES:
                next_age = self.candidate_age
                self.current_age = next_age
                self.transitions.append((timestamp, next_age))
                self.candidate_age = None
                self.candidate_at = None
                self.recent_visual.clear()


def _matches_drop(
    before: Optional[int], after: Optional[int], expected: int
) -> bool:
    if before is None or after is None:
        return False
    drop = before - after
    if drop <= 0:
        return False
    low = expected * (1.0 - _DROP_TOLERANCE)
    high = expected * (1.0 + _DROP_TOLERANCE)
    return low <= drop <= high


def _nearest_frame_indices(
    total_frames: int, fps: float, interval_seconds: float
) -> List[Tuple[int, float]]:
    if fps <= 0:
        return []
    samples: List[Tuple[int, float]] = []
    index = 0
    while True:
        timestamp = index * interval_seconds
        target_frame = int(round(timestamp * fps))
        if target_frame >= total_frames:
            break
        samples.append((target_frame, target_frame / fps))
        index += 1
    return samples


def _decoded_samples(
    cap: cv2.VideoCapture, samples: List[Tuple[int, float]]
) -> "list":
    """Yield ``(frame_id, timestamp, bgr)`` by decoding sequentially."""
    wanted = {frame_index: (frame_index, ts) for frame_index, ts in samples}
    if not wanted:
        return
    max_target = max(wanted)
    for frame_index in range(max_target + 1):
        ok, frame = cap.read()
        if not ok:
            return
        if frame_index in wanted:
            _, ts = wanted[frame_index]
            yield frame_index, ts, frame


def _to_captured(frame_id: int, timestamp: float, bgr: np.ndarray) -> CapturedFrame:
    height, width = bgr.shape[:2]
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    target = CaptureRegion(0, 0, width, height)
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=timestamp,
        width=width,
        height=height,
        pixels=bgra.tobytes(),
        region=target,
    )


def _process_frame(
    frame: CapturedFrame,
    recognizer: TemplateHudRecognizer,
    regions,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    results: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    for name, crop_region in regions.regions.items():
        if name not in _FIELDS:
            continue
        crop = crop_frame(frame, crop_region)
        started = time.perf_counter()
        lines = recognizer.recognize(crop)
        timings[name] = (time.perf_counter() - started) * 1000.0
        results[name] = lines[0].text if lines else ""
    return results, timings


def _summarize(rows: List[_Row]) -> str:
    lines: List[str] = []
    lines.append(f"samples: {len(rows)}")
    lines.append("")
    lines.append("field,n samples,nonempty,valid,valid_pct,avg_recognition_ms")
    for name in _FIELDS:
        total = len(rows)
        nonempty = 0
        valid = 0
        total_ms = 0.0
        for row in rows:
            text = row.fields.get(name, "")
            if text:
                nonempty += 1
                if _is_valid(name, text):
                    valid += 1
            total_ms += row.field_ms.get(name, 0.0)
        valid_pct = (100.0 * valid / total) if total else 0.0
        avg_ms = (total_ms / total) if total else 0.0
        lines.append(
            f"{name}: {total} samples, {nonempty} nonempty, "
            f"{valid} syntactically valid, {valid_pct:.1f}% valid, "
            f"{avg_ms:.2f} ms avg recognition"
        )
    lines.append("")
    return "\n".join(lines)


def _evaluate_temporal(
    rows: List[_Row],
) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int, int, int]]]:
    tracker = _AgeTracker()
    prev: Optional[Dict[str, Optional[int]]] = None
    prev_at: Optional[float] = None
    for row in rows:
        cur: Dict[str, Optional[int]] = {
            name: _parse_int(row.fields.get(name, ""))
            for name in ("food", "gold")
        }
        usable_prev = (
            prev
            if prev_at is not None and row.timestamp - prev_at <= 2.0
            else None
        )
        tracker.observe(
            row.timestamp,
            usable_prev,
            cur,
            _normalize_age(row.fields.get("age", "")),
        )
        if cur["food"] is not None and cur["gold"] is not None:
            prev = cur
            prev_at = row.timestamp
    return tracker.transitions, tracker.candidates


def _write_csv(path: Path, rows: List[_Row]) -> None:
    header = ["index", "frame_id", "timestamp", *_FIELDS, "recognize_ms"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    row.index,
                    row.frame_id,
                    f"{row.timestamp:.3f}",
                    *[row.fields.get(name, "") for name in _FIELDS],
                    f"{row.recognize_ms:.3f}",
                ]
            )


def _write_summary(
    path: Path,
    rows: List[_Row],
    transitions: List[Tuple[float, int]],
    candidates: List[Tuple[float, int, int, int]],
) -> None:
    lines: List[str] = []
    lines.extend(_summarize(rows).splitlines())
    lines.append("")
    lines.append("age candidates (adjacent-frame resource drops):")
    if candidates:
        for ts, age, food_drop, gold_drop in candidates:
            lines.append(
                f"  t={ts:.3f}s -> age {age}: food -{food_drop}, gold -{gold_drop}"
            )
    else:
        lines.append("  none")
    lines.append("")
    lines.append("confirmed age transitions:")
    if transitions:
        for ts, age in transitions:
            lines.append(f"  t={ts:.3f}s -> age {age}")
    else:
        lines.append("  none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    templates_path = Path(args.templates)
    if not templates_path.is_dir():
        raise SystemExit(f"templates not found: {templates_path}")
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")

    config = SmartControllerConfig.from_file(config_path)

    output_dir = (
        Path(args.output)
        if args.output is not None
        else video_path.with_name(video_path.stem + "_ocr_eval")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if total_frames <= 0 or fps <= 0:
            raise SystemExit(f"video has no decodable frames: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target = CaptureRegion(0, 0, width, height)
        scaled = scale_regions(config.hud_regions, config.capture_region, target)
        recognizer = TemplateHudRecognizer(scaled, templates_path)

        samples = _nearest_frame_indices(total_frames, fps, args.interval_seconds)
        rows: List[_Row] = []
        for index, (frame_id, timestamp, bgr) in enumerate(
            _decoded_samples(cap, samples)
        ):
            captured = _to_captured(frame_id, timestamp, bgr)
            started = time.perf_counter()
            fields, field_ms = _process_frame(captured, recognizer, scaled)
            recognize_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                _Row(
                    index=index,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    fields=fields,
                    recognize_ms=recognize_ms,
                    field_ms=field_ms,
                )
            )
    finally:
        cap.release()

    transitions, candidates = _evaluate_temporal(rows)

    _write_csv(output_dir / "results.csv", rows)
    _write_summary(
        output_dir / "summary.txt",
        rows,
        transitions,
        candidates,
    )
    print(f"wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
