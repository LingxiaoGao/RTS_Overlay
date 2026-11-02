"""Age-anchor calibration driven by confirmed pipeline results.

Consumes :class:`PipelineResult` samples, pairing confirmed ``age_completed``
events with fresh confirmed game time. Only the adjusted ``times`` tuple is
exposed; source build-order and step dictionaries are never mutated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from aoe4.smart_pipeline import PipelineResult
from aoe4.smart_runtime import RecognitionStatus
from common.build_order_tools import build_order_time_to_sec, get_build_order_timer_steps


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


def get_available_timer_steps(data: Mapping[str, Any]) -> list:
    """Return complete or partial monotonic timed rows with source indices.

    AoE4 Guides may mix timed and untimed instructions. Untimed rows remain in
    the guide display, while the timed subset drives highlighting and age
    calibration.
    """
    complete = get_build_order_timer_steps(data)
    if complete:
        return [dict(step, _source_index=index) for index, step in enumerate(complete)]

    raw_steps = data.get("build_order") if isinstance(data, Mapping) else None
    if not isinstance(raw_steps, list):
        return []
    timed = []
    last_time = -1
    for source_index, step in enumerate(raw_steps):
        if not isinstance(step, Mapping) or not isinstance(step.get("notes"), list):
            continue
        value = step.get("time")
        if not isinstance(value, str):
            continue
        seconds = build_order_time_to_sec(value.strip().lstrip("~").strip())
        if seconds < 0:
            continue
        if seconds < last_time:
            return []
        copied = dict(step)
        copied["time_sec"] = seconds
        copied["_source_index"] = source_index
        timed.append(copied)
        last_time = seconds
    return timed


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


def _anchor_score(text: str) -> int:
    """Prefer an age-up action over passive mentions of the target age."""
    actions = (
        "start",
        "advance",
        "advancing",
        "aging up",
        "age up",
        "升本",
        "升级时代",
    )
    if any(action in text for action in actions):
        return 3
    if re.search(r"(?:go\s+)?@age/age_[234]\.webp@\s*(?:with|using)", text):
        return 3
    if "until" in text:
        return 0
    return 1


def _detect_markers(steps: Tuple[Mapping[str, Any], ...]):
    """Detect metadata-free anchors from step notes.

    Returns ``(anchors, references)`` where ``anchors`` maps age to
    ``(step_index, reference_time)`` and ``references`` is by age.
    """
    anchors: Dict[int, Tuple[int, int]] = {}
    references: Dict[int, int] = {}
    scores: Dict[int, int] = {}
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
            score = _anchor_score(text)
            if specific in anchors and score <= scores.get(specific, -1):
                continue
            anchors[specific] = (index, int(step["time_sec"]))
            references[specific] = int(step["time_sec"])
            scores[specific] = score
            used_ages.add(specific)
            continue
        if any(marker in text for marker in _AGE_UP_MARKERS):
            while next_age in used_ages or next_age in anchors:
                next_age += 1
            if next_age > 4:
                continue
            anchors[next_age] = (index, int(step["time_sec"]))
            references[next_age] = int(step["time_sec"])
            scores[next_age] = 0
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
        self._start_context_by_age: Dict[
            int, Tuple[int, Tuple[int, ...]]
        ] = {}

    @classmethod
    def from_build_order(cls, data: Mapping[str, Any]) -> "SmartBuildOrderSession":
        steps = get_available_timer_steps(data)
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
        new_start_context = dict(self._start_context_by_age)

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
            start_index = self._anchor_by_age[age][0]
            completion_index = self._completion_anchor_by_age.get(
                age, start_index
            )
            if event.event_type == "age_started":
                reference_times = new_times
                delta = seconds - reference_times[start_index]
                proposed = self._shift_with(
                    reference_times, start_index, delta
                )
            else:
                context = new_start_context.get(age)
                if context is not None and completion_index > start_index:
                    actual_start, reference_times = context
                    proposed = self._fit_age_interval(
                        reference_times,
                        start_index,
                        completion_index,
                        actual_start,
                        seconds,
                    )
                else:
                    delta = seconds - new_times[completion_index]
                    proposed = self._shift_with(
                        new_times, completion_index, delta
                    )
            if proposed is None:
                committed = False
                new_times = entry_times
                new_started = set(self._started)
                new_completed = set(self._completed)
                new_start_context = dict(self._start_context_by_age)
                continue
            new_times = proposed
            seen.add(age)
            if event.event_type == "age_started":
                new_start_context[age] = (seconds, reference_times)

        if not committed:
            return ScheduleUpdate(enabled=True, changed=False, times=self._current_times)

        self._current_times = new_times
        self._started = new_started
        self._completed = new_completed
        self._start_context_by_age = new_start_context
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

    @staticmethod
    def _fit_age_interval(
        reference_times: Tuple[int, ...],
        start_index: int,
        completion_index: int,
        actual_start: int,
        actual_completion: int,
    ) -> Optional[Tuple[int, ...]]:
        """Fit construction rows between two observed age-up anchors.

        Rows from the age-up action through the completed-age separator keep
        their relative position inside the construction interval. Rows after
        completion retain their original gaps from the completion anchor.
        """
        if actual_completion < actual_start:
            return None
        reference_start = reference_times[start_index]
        reference_completion = reference_times[completion_index]
        reference_span = reference_completion - reference_start
        if reference_span <= 0:
            return None

        actual_span = actual_completion - actual_start
        proposed = list(reference_times[:start_index])
        for index in range(start_index, completion_index + 1):
            offset = reference_times[index] - reference_start
            scaled = (offset * actual_span + reference_span // 2) // reference_span
            proposed.append(actual_start + scaled)
        completion_delta = actual_completion - reference_completion
        proposed.extend(
            value + completion_delta
            for value in reference_times[completion_index + 1:]
        )
        result = tuple(proposed)
        for left, right in zip(result, result[1:]):
            if right < left:
                return None
        return result

    def reset(self) -> None:
        self._current_times = self._original_times
        self._started = set()
        self._completed = set()
        self._start_context_by_age = {}
