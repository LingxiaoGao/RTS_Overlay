"""Hard-fail replay acceptance for the three labelled AoE4 recordings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from aoe4.smart_capture import CapturedFrame, CaptureRegion
from aoe4.smart_controller import SmartHudController
from aoe4.smart_controller_config import SmartControllerConfig


@dataclass(frozen=True)
class Case:
    name: str
    expected_civilization: str
    path: Path
    max_seconds: float


class ReplayClock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class ReplayCapture:
    def __init__(self, width: int, height: int) -> None:
        self.region = CaptureRegion(0, 0, width, height)
        self.bgr = None

    def screen_region(self) -> CaptureRegion:
        return self.region

    def capture(self, region, frame_id, captured_at) -> CapturedFrame:
        if self.bgr is None:
            raise RuntimeError("recording frame was not supplied")
        bgra = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2BGRA)
        return CapturedFrame(
            frame_id,
            captured_at,
            bgra.shape[1],
            bgra.shape[0],
            bgra.tobytes(),
            self.region,
        )

    def close(self) -> None:
        return None


def _replay(case: Case, root: Path) -> dict:
    video = cv2.VideoCapture(str(case.path))
    if not video.isOpened():
        raise AssertionError(f"{case.name}: cannot open {case.path}")
    fps = float(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise AssertionError(f"{case.name}: invalid video metadata")

    clock = ReplayClock()
    capture = ReplayCapture(width, height)
    controller = SmartHudController(
        SmartControllerConfig.from_file(
            root / "python/demo_data/smart_hud_config_1280x720.json"
        ),
        capture=capture,
        template_dir=root / "python/demo_data/hud_templates_1280x720",
        game_assets_dir=root / "docs/assets/aoe4",
        clock=clock,
    )
    locked = []
    candidates = []
    age_events = []
    frame_index = 0
    next_sample = 0
    try:
        while True:
            ok, bgr = video.read()
            if not ok:
                break
            video_time = frame_index / fps
            if video_time > case.max_seconds:
                break
            if frame_index >= next_sample:
                clock.now = video_time
                capture.bgr = bgr
                result = controller.poll()
                decision = controller.detected_civilization
                if decision is not None:
                    item = (
                        round(video_time, 3),
                        decision.civilization,
                        round(decision.confidence, 4),
                        decision.source,
                    )
                    if not candidates or candidates[-1][1:] != item[1:]:
                        candidates.append(item)
                    if controller.civilization_locked:
                        if not locked or locked[-1][1:] != item[1:]:
                            locked.append(item)
                for event in result.events:
                    if event.event_type in ("age_started", "age_completed"):
                        age_events.append(
                            (
                                round(video_time, 3),
                                event.event_type,
                                event.payload.get("age"),
                                result.state.game_time.value,
                            )
                        )
                next_sample = round(next_sample + fps * 0.5)
            frame_index += 1
    finally:
        video.release()
        controller.close()

    if not locked:
        raise AssertionError(f"{case.name}: civilization never locked; {candidates=}")
    wrong = [entry for entry in locked if entry[1] != case.expected_civilization]
    if wrong:
        raise AssertionError(
            f"{case.name}: expected {case.expected_civilization}, wrong locks: {wrong}"
        )
    return {"candidate_changes": candidates, "locked_changes": locked, "age_events": age_events}


def _assert_english_ages(events: list) -> None:
    expected = (
        ("age_started", 2, (147.0, 150.0), 138),
        ("age_completed", 2, (228.0, 231.0), 218),
        ("age_started", 3, (575.0, 578.0), 565),
        ("age_completed", 3, (602.0, 606.0), 593),
        ("age_started", 4, (896.0, 899.0), 886),
        ("age_completed", 4, (937.0, 941.0), 928),
    )
    actual = [(kind, age, at, game_time) for at, kind, age, game_time in events]
    if len(actual) != len(expected):
        raise AssertionError(f"English: expected six age events, got {actual}")
    for actual_item, expected_item in zip(actual, expected):
        kind, age, at, game_time = actual_item
        expected_kind, expected_age, video_window, expected_game_time = expected_item
        if (kind, age) != (expected_kind, expected_age):
            raise AssertionError(f"English: wrong age event order: {actual}")
        if not video_window[0] <= at <= video_window[1]:
            raise AssertionError(
                f"English: {kind} age {age} at {at}, expected {video_window}"
            )
        if not isinstance(game_time, (int, float)) or abs(game_time - expected_game_time) > 2:
            raise AssertionError(
                f"English: {kind} age {age} game time {game_time}, "
                f"expected {expected_game_time} +/- 2"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--english-video", type=Path, required=True)
    parser.add_argument("--french-video", type=Path, required=True)
    parser.add_argument("--dragon-video", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    cases = (
        Case("English", "English", args.english_video, 1200.0),
        Case("French", "French", args.french_video, 70.0),
        Case("Dragon", "Order of the Dragon", args.dragon_video, 70.0),
    )
    report = {case.name: _replay(case, args.project_root) for case in cases}
    _assert_english_ages(report["English"]["age_events"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
