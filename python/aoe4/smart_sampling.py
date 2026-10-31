"""Deterministic sampling cadence helpers (stdlib only)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SamplingConfig:
    interval_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")


class ObservationScheduler:
    def __init__(self, config: SamplingConfig = SamplingConfig()) -> None:
        self._config = config
        self._last_sample: Optional[float] = None

    @property
    def config(self) -> SamplingConfig:
        return self._config

    @property
    def last_sample(self) -> Optional[float]:
        return self._last_sample

    def due(self, now: float) -> bool:
        self._validate_timestamp(now)
        return (
            self._last_sample is None
            or now - self._last_sample + 1e-6 >= self._config.interval_seconds
        )

    def mark_sample(self, now: float) -> None:
        self._validate_timestamp(now)
        if self._last_sample is not None and now < self._last_sample:
            raise ValueError("timestamp went backward")
        self._last_sample = now

    def reset(self) -> None:
        self._last_sample = None

    @staticmethod
    def _validate_timestamp(now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("timestamp must be finite")
