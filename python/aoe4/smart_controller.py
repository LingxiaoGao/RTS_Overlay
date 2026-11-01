"""Runtime controller for smart HUD capture, recognition, and pipeline execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from aoe4.aoe4_civ_icon import aoe4_civilization_icon
from aoe4.smart_capture import CaptureRegion
from aoe4.smart_civilization import CivilizationDecision, CivilizationTracker
from aoe4.smart_controller_config import SmartControllerConfig
from aoe4.smart_hud_reader import HudOcrReader
from aoe4.smart_pipeline import PipelineResult, SmartHudPipeline
from aoe4.smart_regions import HudRegionSet, crop_frame, scale_regions
from aoe4.smart_runtime import RecognitionStatus
from aoe4.smart_screen_capture import MssScreenCapture
from aoe4.smart_template_ocr import TemplateHudRecognizer


_FLAG_CODE_OVERRIDES = {
    "Golden Horde": "goh",
    "Knights Templar": "kte",
    "Order of the Dragon": "dra",
}


@dataclass(frozen=True)
class SafePollResult:
    """Result of a safe poll with error details."""

    result: PipelineResult | None
    error: str | None
    consecutive_failures: int


class SmartHudController:
    """Coordinate capture, OCR recognition, and pipeline processing."""

    def __init__(
        self,
        config: SmartControllerConfig,
        capture: object | None = None,
        recognizer: object | None = None,
        template_dir: str | Path | None = None,
        civilization_vision: object | None = None,
        civilization_tracker: CivilizationTracker | None = None,
        game_assets_dir: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._capture = capture if capture is not None else MssScreenCapture()
        self._clock = clock
        self._frame_id = 0
        self._consecutive_failures = 0
        self._closed = False
        self._close_error: Exception | None = None
        self._civilization_started = False
        self._civilization_decision: CivilizationDecision | None = None
        self._civilization_tracker = civilization_tracker or CivilizationTracker()

        if not callable(getattr(self._capture, "capture", None)):
            raise TypeError("capture must expose a callable capture method")
        if not callable(self._clock):
            raise TypeError("clock must be callable")

        self._regions = self._config.hud_regions
        self._capture_regions = dict(self._regions.regions)
        self._capture_groups: tuple[
            tuple[CaptureRegion, tuple[str, ...]], ...
        ] = ()
        screen_region = getattr(self._capture, "screen_region", None)
        if callable(screen_region):
            target = screen_region()
            self._regions = scale_regions(
                self._config.hud_regions, self._config.capture_region, target
            )
            self._capture_regions = dict(self._regions.regions)
            civ_config = self._config.civilization_detection
            if civ_config is not None:
                self._capture_regions["civilization_flag"] = self._scale_region(
                    civ_config.flag, self._config.capture_region, target
                )
                self._capture_regions["civilization_production_buttons"] = self._scale_region(
                    civ_config.production_buttons, self._config.capture_region, target
                )
            self._capture_groups = self._make_capture_groups(self._capture_regions)

        self._civilization_vision = civilization_vision
        if self._civilization_vision is None and game_assets_dir is not None:
            assets = Path(game_assets_dir)
            flag_dir = assets / "civilization_flag"
            templates = {
                civilization: flag_dir / (
                    _FLAG_CODE_OVERRIDES.get(civilization, values[0].lower())
                    + ".webp"
                )
                for civilization, values in aoe4_civilization_icon.items()
                if (
                    flag_dir
                    / (
                        _FLAG_CODE_OVERRIDES.get(civilization, values[0].lower())
                        + ".webp"
                    )
                ).is_file()
            }
            if templates:
                from aoe4.smart_civilization_vision import CivilizationVision

                self._civilization_vision = CivilizationVision(templates)

        if recognizer is None:
            if template_dir is None:
                template_dir = (
                    Path(__file__).resolve().parents[1]
                    / "demo_data"
                    / "hud_templates_1280x720"
                )
            recognizer = TemplateHudRecognizer(
                self._regions, template_dir, self._config.hud_regions
            )
        self._recognizer = recognizer
        if not callable(getattr(self._recognizer, "recognize", None)):
            raise TypeError("recognizer must expose a callable recognize method")

        reader = HudOcrReader(self._regions, self._recognizer)
        self._pipeline = SmartHudPipeline(
            reader,
            profile_validated=self._config.profile_validated,
            age_signal_validated=self._config.age_signal_validated,
        )

    @staticmethod
    def _make_capture_groups(
        regions: Mapping[str, CaptureRegion],
    ) -> tuple[tuple[CaptureRegion, tuple[str, ...]], ...]:
        top_names = tuple(
            name for name in ("game_time", "age") if name in regions
        )
        bottom_names = tuple(
            name for name in regions if name not in top_names
        )
        groups = []
        for names in (top_names, bottom_names):
            if not names:
                continue
            selected = [regions[name] for name in names]
            left = min(region.left for region in selected)
            top = min(region.top for region in selected)
            right = max(region.right for region in selected)
            bottom = max(region.bottom for region in selected)
            groups.append(
                (CaptureRegion(left, top, right - left, bottom - top), names)
            )
        return tuple(groups)

    @staticmethod
    def _scale_region(
        region: CaptureRegion,
        reference: CaptureRegion,
        target: CaptureRegion,
    ) -> CaptureRegion:
        left = target.left + round(
            (region.left - reference.left) * target.width / reference.width
        )
        top = target.top + round(
            (region.top - reference.top) * target.height / reference.height
        )
        right = target.left + round(
            (region.right - reference.left) * target.width / reference.width
        )
        bottom = target.top + round(
            (region.bottom - reference.top) * target.height / reference.height
        )
        return CaptureRegion(left, top, max(1, right - left), max(1, bottom - top))

    def poll(self) -> PipelineResult:
        timestamp = self._clock()
        if self._capture_groups:
            crops = {}
            for group_region, names in self._capture_groups:
                group = self._capture.capture(
                    group_region, self._frame_id, timestamp
                )
                for name in names:
                    crops[name] = crop_frame(group, self._capture_regions[name])
            self._frame_id += 1
            hud_crops = {
                name: crops[name] for name in self._regions.regions if name in crops
            }
            result = self._pipeline.process_crops(hud_crops)
            self._observe_civilization(result, crops, timestamp)
            return result
        frame = self._capture.capture(
            self._config.capture_region, self._frame_id, timestamp
        )
        self._frame_id += 1
        result = self._pipeline.process(frame)
        return result

    def _observe_civilization(
        self,
        result: PipelineResult,
        crops: Mapping[str, object],
        observed_at: float,
    ) -> None:
        if self._civilization_vision is None or self._civilization_decision is not None:
            return
        game_time = result.state.game_time
        seconds = game_time.value
        if (
            game_time.status is not RecognitionStatus.CONFIRMED
            or not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or seconds < 0
            or seconds > 30
        ):
            return
        if not self._civilization_started:
            self._civilization_tracker.start_game(observed_at - float(seconds))
            self._civilization_started = True
        flag = crops.get("civilization_flag")
        production_buttons = crops.get("civilization_production_buttons")
        if flag is None or production_buttons is None:
            return
        flag_bgr = self._frame_bgr(flag)
        buttons_bgr = self._frame_bgr(production_buttons)
        cue, active = self._civilization_vision.recognize_in_game(
            flag_bgr, buttons_bgr
        )
        decision = self._civilization_tracker.observe_in_game(
            cue, active, observed_at
        )
        if decision is not None:
            self._civilization_decision = decision

    @staticmethod
    def _frame_bgr(frame: object):
        import numpy as np

        pixels = np.frombuffer(frame.pixels, dtype=np.uint8)
        bgra = pixels.reshape((frame.height, frame.width, 4))
        return bgra[:, :, :3].copy()

    @property
    def detected_civilization(self) -> CivilizationDecision | None:
        return self._civilization_decision

    def poll_safe(self) -> SafePollResult:
        try:
            result = self.poll()
            self._consecutive_failures = 0
            return SafePollResult(
                result=result,
                error=None,
                consecutive_failures=0,
            )
        except Exception as exc:
            self._consecutive_failures += 1
            return SafePollResult(
                result=None,
                error=f"{type(exc).__name__}: {exc}",
                consecutive_failures=self._consecutive_failures,
            )

    def reset(self) -> None:
        self._frame_id = 0
        self._consecutive_failures = 0
        self._pipeline.reset()
        self._civilization_tracker.reset()
        self._civilization_started = False
        self._civilization_decision = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None

        close_capture = getattr(self._capture, "close", None)
        if callable(close_capture):
            try:
                close_capture()
            except Exception as exc:
                first_error = exc

        close_recognizer = getattr(self._recognizer, "close", None)
        if callable(close_recognizer):
            try:
                close_recognizer()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            self._close_error = first_error
            raise first_error

    def __enter__(self) -> "SmartHudController":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
