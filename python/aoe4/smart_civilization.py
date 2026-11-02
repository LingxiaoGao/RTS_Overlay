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
        required_streak: int = 4,
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
        self._counts: dict[str, int] = {}
        self._best_confidence: dict[str, float] = {}
        self._decision: Optional[CivilizationDecision] = None

    @property
    def decision(self) -> Optional[CivilizationDecision]:
        return self._decision

    @property
    def candidate(self) -> Optional[CivilizationDecision]:
        if self._decision is not None:
            return self._decision
        if self._loading_cue is None:
            return None
        return CivilizationDecision(
            self._loading_cue.civilization,
            self._loading_cue.confidence,
            "loading",
        )

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

        # Any selected own unit carries the civilization banner. Q/W may be
        # unavailable or grey while a unit is queued, so button brightness is
        # only supplementary evidence and never blocks a valid flag sample.
        if flag is None:
            return None

        if self._started_at is None:
            self._started_at = observed_at
        if observed_at - self._started_at >= self._lock_deadline_seconds:
            return self.finalize(observed_at)

        self._counts[flag.civilization] = self._counts.get(flag.civilization, 0) + 1
        self._best_confidence[flag.civilization] = max(
            flag.confidence, self._best_confidence.get(flag.civilization, 0.0)
        )
        return None

    def finalize(self, observed_at: float) -> Optional[CivilizationDecision]:
        """Lock the 30-second plurality after two two-hit confirmations."""
        if self._decision is not None:
            return self._decision
        if self._started_at is None:
            return None
        if observed_at - self._started_at < self._lock_deadline_seconds:
            return None
        eligible = [
            civilization
            for civilization, count in self._counts.items()
            if count >= self._required_streak
        ]
        if not eligible:
            return None
        winner = max(
            eligible,
            key=lambda civilization: (
                self._counts[civilization],
                self._best_confidence[civilization],
                civilization,
            ),
        )
        self._decision = self._combine(
            CivilizationCue(winner, self._best_confidence[winner])
        )
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
