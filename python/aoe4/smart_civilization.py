"""Pure-Python civilization confirmation state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


_VALID_SOURCES = ("loading", "in_game", "combined")


def _validate_cue(civilization: str, confidence: float) -> None:
    if not civilization:
        raise ValueError("civilization must be nonempty")
    if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be finite and within [0, 1]")


@dataclass(frozen=True)
class CivilizationCue:
    civilization: str
    confidence: float

    def __post_init__(self) -> None:
        _validate_cue(self.civilization, self.confidence)


@dataclass(frozen=True)
class CivilizationDecision:
    civilization: str
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"invalid source: {self.source!r}")


class CivilizationTracker:
    def __init__(
        self,
        required_streak: int = 3,
        lock_deadline_seconds: float = 30.0,
    ) -> None:
        if required_streak < 1:
            raise ValueError("required_streak must be >= 1")
        if not math.isfinite(lock_deadline_seconds) or lock_deadline_seconds <= 0:
            raise ValueError("lock_deadline_seconds must be finite and positive")
        self._required_streak = required_streak
        self._lock_deadline_seconds = lock_deadline_seconds
        self._reset_state()

    def _reset_state(self) -> None:
        self._loading_cue: Optional[CivilizationCue] = None
        self._started_at: Optional[float] = None
        self._streak_civ: Optional[str] = None
        self._streak_count = 0
        self._decision: Optional[CivilizationDecision] = None

    @property
    def decision(self) -> Optional[CivilizationDecision]:
        return self._decision

    def observe_loading(self, cue: CivilizationCue) -> None:
        self._reset_state()
        self._loading_cue = cue

    def start_game(self, started_at: float) -> None:
        self._started_at = started_at

    def observe_in_game(
        self,
        flag: Optional[CivilizationCue],
        production_button_active: bool,
        observed_at: float,
    ) -> Optional[CivilizationDecision]:
        if self._decision is not None:
            return self._decision

        if not production_button_active or flag is None:
            self._streak_civ = None
            self._streak_count = 0
            return None

        if flag.civilization == self._streak_civ:
            self._streak_count += 1
        else:
            self._streak_civ = flag.civilization
            self._streak_count = 1

        if self._streak_count < self._required_streak:
            return None
        if self._started_at is None:
            return None
        if observed_at - self._started_at > self._lock_deadline_seconds:
            return None

        self._decision = self._combine(flag)
        return self._decision

    def _combine(self, in_game: CivilizationCue) -> CivilizationDecision:
        loading = self._loading_cue
        if loading is None:
            return CivilizationDecision(
                in_game.civilization, in_game.confidence, "in_game"
            )
        if loading.civilization == in_game.civilization:
            return CivilizationDecision(
                in_game.civilization,
                max(loading.confidence, in_game.confidence),
                "combined",
            )
        if in_game.confidence >= loading.confidence:
            return CivilizationDecision(
                in_game.civilization, in_game.confidence, "in_game"
            )
        return CivilizationDecision(
            loading.civilization, loading.confidence, "loading"
        )

    def reset(self) -> None:
        self._reset_state()
