"""Age-anchor build-order calibration.

Standalone, dependency-free (standard library only) module that adjusts the
timings of build-order steps using *confirmed* age-up anchors. No HUD
recognition is performed here; observations are supplied by callers.

Semantics
---------
An anchor ties an *actual* age-up time to the *reference* time the build order
originally predicted for that age. For a step whose ``age_dependency`` equals
the anchored age, the adjusted time is::

    adjusted = original_time + actual_time - reference_time

Steps with no matching ``age_dependency`` are left unchanged. Each age (2..4)
may be anchored at most once, and only with ``status == 'confirmed'``. Adjusted
times may legitimately be negative; the original data is never mutated.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

MIN_AGE = 2
MAX_AGE = 4
CONFIRMED = 'confirmed'
REJECTED_STATUSES = ('inferred', 'unknown', 'stale')


def _validate_age(age: int) -> int:
    if isinstance(age, bool) or not isinstance(age, int):
        raise TypeError(f'age must be an int, got {type(age).__name__}')
    if age < MIN_AGE or age > MAX_AGE:
        raise ValueError(f'age must be in [{MIN_AGE}, {MAX_AGE}], got {age}')
    return age


def _validate_seconds(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an int, got {type(value).__name__}')
    if value < 0:
        raise ValueError(f'{name} must be nonnegative, got {value}')
    return value


@dataclass(frozen=True)
class BuildOrderStep:
    """An immutable build-order step.

    ``original_time`` is nonnegative seconds. ``age_dependency`` is ``None`` for
    an independent step, otherwise the age (2..4) the step depends on.
    """

    id: str
    original_time: int
    age_dependency: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('id must be a nonempty string')
        _validate_seconds(self.original_time, 'original_time')
        if self.age_dependency is not None:
            _validate_age(self.age_dependency)


@dataclass(frozen=True)
class CalibrationAnchor:
    """An immutable observation tying an actual age-up time to a reference."""

    age: int
    actual_time: int
    reference_time: int

    def __post_init__(self) -> None:
        _validate_age(self.age)
        _validate_seconds(self.actual_time, 'actual_time')
        _validate_seconds(self.reference_time, 'reference_time')

    @property
    def offset(self) -> int:
        return self.actual_time - self.reference_time


class CalibrationEngine:
    """Applies confirmed age anchors to a fixed list of build-order steps."""

    def __init__(self, steps: Iterable[BuildOrderStep]) -> None:
        steps = tuple(steps)
        seen = set()
        for step in steps:
            if not isinstance(step, BuildOrderStep):
                raise TypeError('steps must contain BuildOrderStep instances')
            if step.id in seen:
                raise ValueError(f'duplicate step id: {step.id!r}')
            seen.add(step.id)
        self._steps: Tuple[BuildOrderStep, ...] = steps
        self._anchors: Dict[int, CalibrationAnchor] = {}

    @property
    def steps(self) -> Tuple[BuildOrderStep, ...]:
        """Steps in original order (immutable)."""
        return self._steps

    @property
    def anchors(self) -> Dict[int, CalibrationAnchor]:
        """Mapping of age -> accepted anchor (a copy)."""
        return dict(self._anchors)

    def apply_anchor(self, anchor: CalibrationAnchor, status: str) -> bool:
        """Apply ``anchor`` if ``status`` is 'confirmed' and age is unanchored.

        Returns ``True`` on success, ``False`` on rejection. Rejection never
        mutates engine state. Only one anchor per age is accepted; subsequent
        calls for the same age (repeated or conflicting) are rejected.
        """
        if not isinstance(anchor, CalibrationAnchor):
            raise TypeError('anchor must be a CalibrationAnchor')
        if status != CONFIRMED:
            return False
        if anchor.age in self._anchors:
            return False
        self._anchors[anchor.age] = anchor
        return True

    def adjusted_time(self, step: BuildOrderStep) -> int:
        """Adjusted time for ``step``; unchanged unless its age is anchored."""
        anchor = self._anchors.get(step.age_dependency) \
            if step.age_dependency is not None else None
        if anchor is None:
            return step.original_time
        return step.original_time + anchor.offset

    @property
    def adjusted_times(self) -> Dict[str, int]:
        """Mapping of step id -> adjusted time, recomputed from originals."""
        return {step.id: self.adjusted_time(step) for step in self._steps}
