"""Age-anchor calibration driven by confirmed pipeline results.

Consumes :class:`PipelineResult` samples, pairing confirmed ``age_completed``
events with fresh confirmed game time. Only the adjusted ``times`` tuple is
exposed; source build-order and step dictionaries are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from aoe4.smart_pipeline import PipelineResult
from aoe4.smart_runtime import RecognitionStatus
from common.build_order_tools import get_build_order_timer_steps


_REFERENCE_KEYS = ("2", "3", "4")
_SPECIFIC_MARKERS = (
    (2, ("age_2", "age 2", "feudal", "封建")),
    (3, ("age_3", "age 3", "castle", "城堡")),
    (4, ("age_4", "age 4", "imperial", "帝王")),
)
_AGE_UP_MARKERS = (
    "age up",
    "age-up",
    "aging up",
    "advance to",
    "advancing to",
    "升本",
    "升级时代",
)


@dataclass(frozen=True)
class ScheduleUpdate:
    """Result of consuming one pipeline sample."""

    enabled: bool
    changed: bool
    times: Tuple[int, ...]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def apply_schedule_times(steps: Any, times: Any) -> list:
    """Return shallow copies of ``steps`` with synchronized calibrated times.

    ``steps`` are validated timer-step mappings and ``times`` are validated
    integer seconds of equal length. The input list and mappings are never
    mutated.
    """
    if len(steps) != len(times):
        raise ValueError(
            "times length (%d) does not match timer step count (%d)"
            % (len(times), len(steps))
        )
    result = []
    for step, seconds in zip(steps, times):
        copied = dict(step)
        copied["time_sec"] = seconds
        copied["time"] = "%d:%02d" % (seconds // 60, seconds % 60)
        result.append(copied)
    return result


def _parse_metadata(
    data: Mapping[str, Any]
) -> Optional[Tuple[Dict[int, int], Tuple[Optional[int], ...]]]:
    """Validate present ``smart_calibration`` metadata.

    Returns ``None`` when metadata is absent, otherwise a ``(references,
    dependencies)`` pair. Raises ``ValueError`` on invalid present metadata.
    """
    if not isinstance(data, Mapping):
        raise ValueError("build order data must be a mapping")
    if "smart_calibration" not in data:
        return None
    meta = data["smart_calibration"]
    if not isinstance(meta, Mapping):
        raise ValueError("smart_calibration must be a mapping")

    version = meta.get("version")
    if _is_int(version) is False or version != 1:
        raise ValueError("smart_calibration.version must be the integer 1")

    raw_refs = meta.get("reference_age_times")
    if not isinstance(raw_refs, Mapping):
        raise ValueError("smart_calibration.reference_age_times must be a mapping")
    references: Dict[int, int] = {}
    for key, value in raw_refs.items():
        if key not in _REFERENCE_KEYS:
            raise ValueError(
                "smart_calibration.reference_age_times has unexpected key: %r" % (key,)
            )
        if not _is_int(value) or value < 0:
            raise ValueError(
                "smart_calibration.reference_age_times[%r] must be a nonnegative integer" % (key,)
            )
        references[int(key)] = value

    raw_deps = meta.get("step_age_dependencies")
    if not isinstance(raw_deps, list):
        raise ValueError("smart_calibration.step_age_dependencies must be a list")
    dependencies: list = []
    for index, dep in enumerate(raw_deps):
        if dep is None:
            dependencies.append(None)
            continue
        if not _is_int(dep) or dep < 2 or dep > 4:
            raise ValueError(
                "smart_calibration.step_age_dependencies[%d] must be null or an integer age in [2, 4]" % (index,)
            )
        if dep not in references:
            raise ValueError(
                "smart_calibration.step_age_dependencies[%d] references age %d without a reference time" % (index, dep)
            )
        dependencies.append(dep)

    return references, tuple(dependencies)


def _flatten_notes(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return " ".join(item for item in value if isinstance(item, str)).lower()
    return ""


def _detect_markers(steps: Tuple[Mapping[str, Any], ...]):
    """Detect metadata-free anchors from step notes.

    Returns ``(anchors, references)`` where ``anchors`` maps age to
    ``(step_index, reference_time)`` and ``references`` is by age.
    """
    anchors: Dict[int, Tuple[int, int]] = {}
    references: Dict[int, int] = {}
    used_ages: set = set()
    next_age = 2
    for index, step in enumerate(steps):
        text = _flatten_notes(step.get("notes"))
        specific = None
        for age, markers in _SPECIFIC_MARKERS:
            if any(marker in text for marker in markers):
                specific = age
                break
        if specific is not None:
            if specific in anchors:
                continue
            anchors[specific] = (index, int(step["time_sec"]))
            references[specific] = int(step["time_sec"])
            used_ages.add(specific)
            continue
        if any(marker in text for marker in _AGE_UP_MARKERS):
            while next_age in used_ages or next_age in anchors:
                next_age += 1
            if next_age > 4:
                continue
            anchors[next_age] = (index, int(step["time_sec"]))
            references[next_age] = int(step["time_sec"])
            used_ages.add(next_age)
    return anchors, references


class SmartBuildOrderSession:
    """Consume pipeline results and adjust build-order times via age anchors."""

    def __init__(
        self,
        enabled: bool,
        original_times: Tuple[int, ...],
        references: Dict[int, int],
        anchors: Dict[int, Tuple[int, int]],
        completion_anchors: Optional[Dict[int, int]] = None,
    ) -> None:
        self._enabled = enabled
        self._original_times = original_times
        self._references = references
        self._anchor_by_age = anchors
        self._completion_anchor_by_age = completion_anchors or {
            age: anchor[0] for age, anchor in anchors.items()
        }
        self._current_times = original_times
        self._started: set = set()
        self._completed: set = set()

    @classmethod
    def from_build_order(cls, data: Mapping[str, Any]) -> "SmartBuildOrderSession":
        steps = get_build_order_timer_steps(data)
        if not steps:
            return cls(False, (), {}, {})
        original_times = tuple(int(step["time_sec"]) for step in steps)

        parsed = _parse_metadata(data)
        if parsed is None:
            anchors, references = _detect_markers(steps)
            completions = cls._completion_anchors(steps, anchors)
            return cls(True, original_times, references, anchors, completions)

        references, dependencies = parsed
        if len(dependencies) != len(steps):
            raise ValueError(
                "smart_calibration.step_age_dependencies length (%d) does not match "
                "timer step count (%d)" % (len(dependencies), len(steps))
            )

        anchors: Dict[int, Tuple[int, int]] = {}
        for index, step in enumerate(steps):
            age = dependencies[index]
            if age is None:
                continue
            if age in anchors:
                continue
            if age not in references:
                continue
            anchors[age] = (index, int(step["time_sec"]))
        completions = cls._completion_anchors(steps, anchors)
        return cls(True, original_times, references, anchors, completions)

    @staticmethod
    def _completion_anchors(
        steps: Tuple[Mapping[str, Any], ...],
        starts: Dict[int, Tuple[int, int]],
    ) -> Dict[int, int]:
        completions: Dict[int, int] = {}
        for age, (start_index, _reference) in starts.items():
            completion = start_index
            for index in range(start_index + 1, len(steps)):
                value = steps[index].get("age")
                if _is_int(value) and value == age:
                    completion = index
                    break
            completions[age] = completion
        return completions

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def times(self) -> Tuple[int, ...]:
        return self._current_times

    def _shift(self, index: int, delta: int) -> Optional[Tuple[int, ...]]:
        proposed = (
            self._current_times[:index]
            + tuple(t + delta for t in self._current_times[index:])
        )
        for left, right in zip(proposed, proposed[1:]):
            if right < left:
                return None
        return proposed

    @staticmethod
    def _coherent_time(result: PipelineResult, event: Any) -> Optional[int]:
        game_time = result.state.game_time
        if game_time.status is not RecognitionStatus.CONFIRMED:
            return None
        value = game_time.value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if value != value or value < 0:
            return None
        if value != int(value):
            return None
        seconds = int(value)
        if not (
            game_time.observed_at == result.state.updated_at == event.emitted_at
        ):
            return None
        return seconds

    def consume(self, result: PipelineResult) -> ScheduleUpdate:
        if not self._enabled:
            return ScheduleUpdate(enabled=False, changed=False, times=self._original_times)

        entry_times = self._current_times
        committed = True
        new_times = entry_times
        new_started = set(self._started)
        new_completed = set(self._completed)

        for event in result.events:
            if event.event_type not in ("age_started", "age_completed"):
                continue
            if event.status is not RecognitionStatus.CONFIRMED:
                continue
            age = event.payload.get("age")
            if not _is_int(age) or age not in self._anchor_by_age:
                continue
            seen = new_started if event.event_type == "age_started" else new_completed
            if age in seen:
                continue
            seconds = self._coherent_time(result, event)
            if seconds is None:
                continue
            index = (
                self._anchor_by_age[age][0]
                if event.event_type == "age_started"
                else self._completion_anchor_by_age.get(
                    age, self._anchor_by_age[age][0]
                )
            )
            delta = seconds - new_times[index]
            proposed = self._shift_with(new_times, index, delta)
            if proposed is None:
                committed = False
                new_times = entry_times
                new_started = set(self._started)
                new_completed = set(self._completed)
                continue
            new_times = proposed
            seen.add(age)

        if not committed:
            return ScheduleUpdate(enabled=True, changed=False, times=self._current_times)

        self._current_times = new_times
        self._started = new_started
        self._completed = new_completed
        return ScheduleUpdate(
            enabled=True,
            changed=new_times != entry_times,
            times=self._current_times,
        )

    @staticmethod
    def _shift_with(
        times: Tuple[int, ...], index: int, delta: int
    ) -> Optional[Tuple[int, ...]]:
        proposed = times[:index] + tuple(t + delta for t in times[index:])
        for left, right in zip(proposed, proposed[1:]):
            if right < left:
                return None
        return proposed

    def reset(self) -> None:
        self._current_times = self._original_times
        self._started = set()
        self._completed = set()
