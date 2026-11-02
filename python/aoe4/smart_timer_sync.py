"""Game timer synchronizer: OCR cadence gating and between-read projection."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from aoe4.smart_hud_reader import HudTextEvidence
from aoe4.smart_pipeline import PipelineResult
from aoe4.smart_runtime import (
    FieldState,
    GameState,
    Observation,
    RecognitionStatus,
)

_UNANCHORED_CADENCE = 0.5
_RUNNING_CADENCE = 2.0
_PAUSE_CADENCE = 0.5
_RECOVERY_CADENCE = 0.5
_RECALIBRATION_TOLERANCE = 2.0
_RECOVERY_STEP_MAX_FORWARD = 2.0
_RECOVERY_COHERENT_COUNT = 3
_HIGH_CONFIDENCE = 0.8


def _validate_now(now: object, previous: float) -> float:
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise TypeError(f"now must be numeric, got {type(now).__name__}")
    value = float(now)
    if not math.isfinite(value):
        raise ValueError("now must be finite")
    if value < previous:
        raise ValueError("now must be nondecreasing")
    return value


class GameTimerSynchronizer:
    """Gate OCR reads and project the confirmed game timer between reads."""

    def __init__(self) -> None:
        self._last_now = 0.0
        self._last_read = 0.0
        self._first_call = True
        self.reset()

    def reset(self) -> None:
        self._first_call = True
        self._pending_read_at: float | None = None
        self._anchored = False
        self._anchor_value = 0.0
        self._anchor_now = 0.0
        self._paused = False
        self._pause_anchor = 0.0
        self._recovering = False
        self._recovery_value = 0.0
        self._recovery_candidates: list[tuple[float, float]] = []
        self._last_read = self._last_now

    def should_read(self, now: float) -> bool:
        now = _validate_now(now, self._last_now)
        self._last_now = now
        if self._first_call:
            self._first_call = False
            self._last_read = now
            self._pending_read_at = now
            return True
        if not self._anchored:
            cadence = _UNANCHORED_CADENCE
        elif self._paused:
            cadence = _PAUSE_CADENCE
        elif self._recovering:
            cadence = _RECOVERY_CADENCE
        else:
            cadence = _RUNNING_CADENCE
        if now - self._last_read >= cadence:
            self._last_read = now
            self._pending_read_at = now
            return True
        return False

    def process(self, result: PipelineResult, now: float) -> PipelineResult:
        now = _validate_now(now, self._last_now)
        self._last_now = now
        self._current_result = result
        self._current_state = result.state
        requested_read = (
            self._pending_read_at is not None
            and abs(self._pending_read_at - now) <= 1e-6
        ) or "game_time" in result.evidence
        self._pending_read_at = None

        field = result.state.game_time
        confirmed_value: Optional[float] = None
        if (
            field.status is RecognitionStatus.CONFIRMED
            and isinstance(field.value, (int, float))
            and not isinstance(field.value, bool)
        ):
            confirmed_value = float(field.value)

        if not self._anchored:
            if confirmed_value is not None:
                self._first_call = False
                self._anchor(confirmed_value, now)
                return self._emit(now)
            return result

        if self._recovering:
            self._observe_recovery(result, now)
            if self._anchored:
                return self._emit(now)
            return self._emit(now, projected=self._project(now))

        if self._paused:
            if confirmed_value is not None and confirmed_value > self._anchor_value:
                self._resume(confirmed_value, now)
                return self._emit(now)
            if confirmed_value is not None:
                self._anchor_now = now
            return self._emit(now, projected=self._anchor_value)

        if confirmed_value is None:
            due_without_query = now - self._last_read >= _RUNNING_CADENCE
            if requested_read or due_without_query:
                self._enter_recovery(now)
                return self._emit(now, projected=self._project(now))
            return self._emit(now)

        predicted = self._project(now)
        if abs(confirmed_value - predicted) <= _RECALIBRATION_TOLERANCE:
            if confirmed_value <= self._anchor_value:
                self._enter_pause(confirmed_value, now)
                return self._emit(now, projected=self._anchor_value)
            self._anchor(confirmed_value, now)
            return self._emit(now)

        self._enter_recovery(now)
        return self._emit(now, projected=self._project(now))

    def _anchor(self, value: float, now: float) -> None:
        self._anchored = True
        self._anchor_value = value
        self._anchor_now = now
        self._paused = False
        self._recovering = False
        self._recovery_candidates = []
        self._last_read = now

    def _resume(self, value: float, now: float) -> None:
        self._anchor(value, now)

    def _enter_pause(self, value: float, now: float) -> None:
        self._paused = True
        self._pause_anchor = now
        self._anchor_value = value
        self._anchor_now = now
        self._last_read = now

    def _enter_recovery(self, now: float) -> None:
        self._recovery_value = self._project(now)
        self._recovering = True
        self._paused = False
        self._recovery_candidates = []
        self._last_read = now

    def _project(self, now: float) -> float:
        if self._paused:
            return self._anchor_value
        if self._recovering:
            return self._recovery_value
        elapsed = max(0.0, now - self._anchor_now)
        return self._anchor_value + elapsed

    def _observe_recovery(self, result: PipelineResult, now: float) -> None:
        observation = None
        for candidate in result.observations:
            if isinstance(candidate, Observation) and candidate.field == "game_time":
                observation = candidate
                break
        if observation is None:
            self._recovery_candidates = []
            return
        evidence = result.evidence.get("game_time")
        if not isinstance(evidence, HudTextEvidence):
            self._recovery_candidates = []
            return
        if (
            observation.confidence < _HIGH_CONFIDENCE
            or evidence.confidence < _HIGH_CONFIDENCE
        ):
            self._recovery_candidates = []
            return
        if not isinstance(observation.value, (int, float)) or isinstance(
            observation.value, bool
        ):
            self._recovery_candidates = []
            return
        value = float(observation.value)
        at = float(observation.observed_at)

        candidates = self._recovery_candidates
        if candidates:
            last_value, last_at = candidates[-1]
            if value < last_value:
                candidates = []
            elif at > last_at:
                if value - last_value > _RECOVERY_STEP_MAX_FORWARD * (at - last_at) / 0.5:
                    candidates = []
        candidates.append((value, at))
        if len(candidates) > _RECOVERY_COHERENT_COUNT:
            candidates = candidates[-_RECOVERY_COHERENT_COUNT:]
        self._recovery_candidates = candidates
        if len(candidates) >= _RECOVERY_COHERENT_COUNT:
            self._anchor(value, now)

    def _emit(self, now: float, projected: Optional[float] = None, anchor_value: Optional[float] = None) -> PipelineResult:
        if anchor_value is not None:
            value = int(anchor_value)
        elif projected is not None:
            value = int(projected)
        else:
            value = int(self._project(now))
        if value < 0:
            value = 0
        state = self._state_with(self._current_state, value, now)
        return replace(self._current_result, state=state)

    @staticmethod
    def _state_with(state: GameState, value: int, now: float) -> GameState:
        return replace(
            state,
            game_time=FieldState(
                value=value,
                confidence=state.game_time.confidence,
                status=RecognitionStatus.CONFIRMED,
                observed_at=now,
                evidence_ids=state.game_time.evidence_ids,
            ),
            updated_at=now,
        )
