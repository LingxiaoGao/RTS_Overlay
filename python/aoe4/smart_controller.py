"""Runtime controller for smart HUD capture, recognition, and pipeline execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aoe4.smart_capture import CaptureRegion
from aoe4.smart_controller_config import SmartControllerConfig
from aoe4.smart_hud_reader import HudOcrReader
from aoe4.smart_pipeline import PipelineResult, SmartHudPipeline
from aoe4.smart_regions import HudRegionSet, crop_frame, scale_regions
from aoe4.smart_screen_capture import MssScreenCapture
from aoe4.smart_template_ocr import TemplateHudRecognizer


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._capture = capture if capture is not None else MssScreenCapture()
        self._clock = clock
        self._frame_id = 0
        self._consecutive_failures = 0
        self._closed = False
        self._close_error: Exception | None = None

        if not callable(getattr(self._capture, "capture", None)):
            raise TypeError("capture must expose a callable capture method")
        if not callable(self._clock):
            raise TypeError("clock must be callable")

        self._regions = self._config.hud_regions
        self._capture_groups: tuple[
            tuple[CaptureRegion, tuple[str, ...]], ...
        ] = ()
        screen_region = getattr(self._capture, "screen_region", None)
        if callable(screen_region):
            target = screen_region()
            self._regions = scale_regions(
                self._config.hud_regions, self._config.capture_region, target
            )
            self._capture_groups = self._make_capture_groups(self._regions)

        if recognizer is None:
            if template_dir is None:
                template_dir = (
                    Path(__file__).resolve().parents[1]
                    / "demo_data"
                    / "hud_templates_1280x720"
                )
            recognizer = TemplateHudRecognizer(self._regions, template_dir)
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
        regions: HudRegionSet,
    ) -> tuple[tuple[CaptureRegion, tuple[str, ...]], ...]:
        top_names = tuple(
            name for name in ("game_time", "age") if name in regions.regions
        )
        bottom_names = tuple(
            name for name in regions.regions if name not in top_names
        )
        groups = []
        for names in (top_names, bottom_names):
            if not names:
                continue
            selected = [regions.regions[name] for name in names]
            left = min(region.left for region in selected)
            top = min(region.top for region in selected)
            right = max(region.right for region in selected)
            bottom = max(region.bottom for region in selected)
            groups.append(
                (CaptureRegion(left, top, right - left, bottom - top), names)
            )
        return tuple(groups)

    def poll(self) -> PipelineResult:
        timestamp = self._clock()
        if self._capture_groups:
            crops = {}
            for group_region, names in self._capture_groups:
                group = self._capture.capture(
                    group_region, self._frame_id, timestamp
                )
                for name in names:
                    crops[name] = crop_frame(
                        group, self._regions.regions[name]
                    )
            self._frame_id += 1
            return self._pipeline.process_crops(crops)
        frame = self._capture.capture(
            self._config.capture_region, self._frame_id, timestamp
        )
        self._frame_id += 1
        return self._pipeline.process(frame)

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
