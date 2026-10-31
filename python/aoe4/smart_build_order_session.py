"""Age-anchor calibration driven by confirmed pipeline results.

Thin session wrapper around :class:`CalibrationEngine` that consumes
:class:`PipelineResult` samples, pairing confirmed ``age_completed`` events with
fresh confirmed game time from the same sample. Only the adjusted ``times``
tuple is exposed; source build-order and step dictionaries are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from aoe4.smart_calibration import (
    CalibrationAnchor,
    CalibrationEngine,
    BuildOrderStep,
    CONFIRMED,
)
from aoe4.smart_pipeline import PipelineResult
from aoe4.smart_runtime import RecognitionStatus
from common.build_order_tools import get_build_order_timer_steps


_REFERENCE_KEYS = ("2", "3", "4")


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


def _parse_metadata(data: Mapping[str, Any]) -> Optional[Tuple[Dict[int, int], Tuple[Optional[int], ...]]]:
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


class SmartBuildOrderSession:
    """Consume pipeline results and adjust build-order times via age anchors."""

    def __init__(
        self,
        engine: Optional[CalibrationEngine],
        original_times: Tuple[int, ...],
        references: Dict[int, int],
    ) -> None:
        self._engine = engine
        self._original_times = original_times
        self._references = references

    @classmethod
    def from_build_order(cls, data: Mapping[str, Any]) -> "SmartBuildOrderSession":
        steps = get_build_order_timer_steps(data)
        original_times = tuple(int(step["time_sec"]) for step in steps)

        parsed = _parse_metadata(data)
        if parsed is None:
            return cls(None, original_times, {})

        references, dependencies = parsed
        if len(dependencies) != len(steps):
            raise ValueError(
                "smart_calibration.step_age_dependencies length (%d) does not match "
                "timer step count (%d)" % (len(dependencies), len(steps))
            )

        build_steps = []
        for index, step in enumerate(steps):
            build_steps.append(
                BuildOrderStep(
                    id=str(index),
                    original_time=int(step["time_sec"]),
                    age_dependency=dependencies[index],
                )
            )
        engine = CalibrationEngine(build_steps)
        return cls(engine, original_times, references)

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    @property
    def times(self) -> Tuple[int, ...]:
        if self._engine is None:
            return self._original_times
        adjusted = self._engine.adjusted_times
        return tuple(adjusted[step.id] for step in self._engine.steps)

    def _preview(self, anchors: Dict[int, CalibrationAnchor]) -> Tuple[int, ...]:
        assert self._engine is not None
        preview_engine = CalibrationEngine(self._engine.steps)
        for anchor in anchors.values():
            preview_engine.apply_anchor(anchor, CONFIRMED)
        adjusted = preview_engine.adjusted_times
        return tuple(adjusted[step.id] for step in self._engine.steps)

    def consume(self, result: PipelineResult) -> ScheduleUpdate:
        if self._engine is None:
            return ScheduleUpdate(enabled=False, changed=False, times=self._original_times)

        for event in result.events:
            anchor = self._match_anchor(result, event)
            if anchor is None:
                continue

            anchors = dict(self._engine.anchors)
            if anchor.age in anchors:
                continue
            anchors[anchor.age] = anchor

            preview = self._preview(anchors)
            if any(right < left for left, right in zip(preview, preview[1:])):
                continue

            self._engine.apply_anchor(anchor, CONFIRMED)
            return ScheduleUpdate(enabled=True, changed=True, times=self.times)

        return ScheduleUpdate(enabled=True, changed=False, times=self.times)

    def _match_anchor(self, result: PipelineResult, event: Any) -> Optional[CalibrationAnchor]:
        if event.event_type != "age_completed":
            return None
        if event.status is not RecognitionStatus.CONFIRMED:
            return None

        age = event.payload.get("age")
        if not _is_int(age):
            return None
        if age not in self._references:
            return None

        game_time = result.state.game_time
        if game_time.status is not RecognitionStatus.CONFIRMED:
            return None
        if not _is_int(game_time.value) or game_time.value < 0:
            return None
        if not (
            game_time.observed_at == result.state.updated_at == event.emitted_at
        ):
            return None

        return CalibrationAnchor(
            age=age,
            actual_time=game_time.value,
            reference_time=self._references[age],
        )

    def reset(self) -> None:
        if self._engine is None:
            return
        steps = self._engine.steps
        self._engine = CalibrationEngine(steps)
