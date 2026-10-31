from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from aoe4.smart_hud_reader import HudTextEvidence
from aoe4.smart_runtime import Observation, RecognitionStatus


def _check_int(
    value: object, name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _check_float(
    value: object,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _check_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value




@dataclass(frozen=True)
class ConfirmationConfig:
    consecutive_samples: int = 3
    min_evidence_confidence: float = 0.8
    timer_max_forward_jump_seconds: float = 2.0

    def __post_init__(self) -> None:
        samples = _check_int(
            self.consecutive_samples, "consecutive_samples", minimum=2
        )
        if samples != self.consecutive_samples:
            raise ValueError("consecutive_samples must be an int")
        _check_float(
            self.min_evidence_confidence,
            "min_evidence_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        _check_float(
            self.timer_max_forward_jump_seconds,
            "timer_max_forward_jump_seconds",
            minimum=0.0,
        )
        if self.timer_max_forward_jump_seconds <= 0:
            raise ValueError("timer_max_forward_jump_seconds must be > 0")


@dataclass(frozen=True)
class _Streak:
    value: object
    count: int
    last_sample_index: int


_Checkpoint = Tuple[Optional[int], Dict[str, _Streak]]


def _same_value(left: object, right: object) -> bool:
    return left == right


class HudEvidenceConfirmer:
    def __init__(self, config: ConfirmationConfig | None = None) -> None:
        if config is None:
            config = ConfirmationConfig()
        if not isinstance(config, ConfirmationConfig):
            raise TypeError("config must be a ConfirmationConfig")
        self._config = config
        self._streaks: dict[str, _Streak] = {}
        self._last_sample_index: int | None = None

    @property
    def config(self) -> ConfirmationConfig:
        return self._config

    @property
    def last_sample_index(self) -> int | None:
        return self._last_sample_index

    def reset(self) -> None:
        self._streaks = {}
        self._last_sample_index = None

    def checkpoint(self) -> _Checkpoint:
        return (self._last_sample_index, dict(self._streaks))

    def restore(self, checkpoint: _Checkpoint) -> None:
        last_sample_index, streaks = checkpoint
        self._streaks = streaks
        self._last_sample_index = last_sample_index

    def accept(
        self,
        evidence: Mapping[str, HudTextEvidence],
        observations: Iterable[Observation],
        sample_index: int,
        profile_validated: bool,
        age_signal_validated: bool = False,
    ) -> tuple[Observation, ...]:
        materialized = tuple(observations)
        _check_int(sample_index, "sample_index", minimum=0)
        _check_bool(profile_validated, "profile_validated")
        _check_bool(age_signal_validated, "age_signal_validated")
        if self._last_sample_index is not None:
            if sample_index <= self._last_sample_index:
                raise ValueError(
                    "sample_index must be strictly greater than last accepted"
                )

        current_fields = {obs.field for obs in materialized}
        working: dict[str, _Streak] = {
            field: streak
            for field, streak in self._streaks.items()
            if field in current_fields
        }

        threshold = self._config.consecutive_samples
        jump_limit = self._config.timer_max_forward_jump_seconds
        min_confidence = self._config.min_evidence_confidence
        outputs: list[Observation] = []

        for obs in materialized:
            item = evidence[obs.field]
            combined = min(float(obs.confidence), float(item.confidence))
            eligible = profile_validated and combined >= min_confidence
            if obs.field == "age" and not age_signal_validated:
                eligible = False

            if not eligible:
                working.pop(obs.field, None)
                outputs.append(
                    Observation(
                        observation_id=obs.observation_id,
                        field=obs.field,
                        value=obs.value,
                        confidence=combined,
                        status=RecognitionStatus.INFERRED,
                        frame_id=obs.frame_id,
                        observed_at=obs.observed_at,
                    )
                )
                continue

            prior = working.get(obs.field)
            count = 1
            if prior is not None and prior.last_sample_index == sample_index - 1:
                if obs.field == "game_time":
                    if (
                        isinstance(prior.value, float)
                        and isinstance(obs.value, float)
                    ):
                        delta = obs.value - prior.value
                        if 0.0 <= delta <= jump_limit:
                            count = prior.count + 1
                else:
                    if _same_value(prior.value, obs.value):
                        count = prior.count + 1

            if count > threshold:
                count = threshold
            working[obs.field] = _Streak(
                value=obs.value,
                count=count,
                last_sample_index=sample_index,
            )
            status = (
                RecognitionStatus.CONFIRMED
                if count >= threshold
                else RecognitionStatus.INFERRED
            )
            outputs.append(
                Observation(
                    observation_id=obs.observation_id,
                    field=obs.field,
                    value=obs.value,
                    confidence=combined,
                    status=status,
                    frame_id=obs.frame_id,
                    observed_at=obs.observed_at,
                )
            )

        self._streaks = working
        self._last_sample_index = sample_index
        return tuple(outputs)
