"""Smart HUD pipeline: cadence-gated OCR read, parse, confirm, and state ingest."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from aoe4.smart_capture import CapturedFrame
from aoe4.smart_confirmation import HudEvidenceConfirmer
from aoe4.smart_hud import parse_hud_text
from aoe4.smart_hud_reader import HudTextEvidence, text_mapping
from aoe4.smart_runtime import (
    GameEvent,
    GameState,
    Observation,
    RecognitionStatus,
    StateEstimator,
)
from aoe4.smart_sampling import ObservationScheduler


_AGE_VISUAL_STREAK = 4


@dataclass(frozen=True)
class PipelineResult:
    sampled: bool
    evidence: Mapping[str, HudTextEvidence]
    observations: tuple[Observation, ...]
    verified_observations: tuple[Observation, ...]
    state: GameState
    events: tuple[GameEvent, ...]


class _AgeSignal:
    """Confirm visual age advances only after the matching resource spend."""

    _COSTS = {2: (400, 200), 3: (1200, 600), 4: (2400, 1200)}
    _ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.current = 1
        self.candidate: int | None = None
        self.candidate_at = 0.0
        self.previous: tuple[int, int] | None = None
        self.previous_at = 0.0
        self.recent: deque[tuple[int, int]] = deque(maxlen=10)
        self.visual = deque(maxlen=10)
        self.visual_streak = 0

    @staticmethod
    def _number(text: object) -> int | None:
        return int(text) if isinstance(text, str) and text.isdigit() else None

    @staticmethod
    def _visual_age(text: object) -> int:
        if not isinstance(text, str):
            return 0
        if "V" in text:
            return 4
        return min(3, text.count("I"))

    def observe(
        self, texts: Mapping[str, str], now: float
    ) -> tuple[str, int | None, int | None]:
        started_age: int | None = None
        completed_age: int | None = None
        food = self._number(texts.get("food"))
        gold = self._number(texts.get("gold"))
        if food is not None and gold is not None:
            # Compare against the recent high-water food/gold pair so a spend
            # spread over several 500ms samples is still detected, while
            # ordinary income between samples is tolerated.
            sample = (food, gold)
            if self.previous is not None and now - self.previous_at > 2.0:
                self.recent.clear()
            peak_food = max((pair[0] for pair in self.recent), default=food)
            peak_gold = max((pair[1] for pair in self.recent), default=gold)
            next_age = self.current + 1
            if next_age in self._COSTS:
                expected_food, expected_gold = self._COSTS[next_age]
                food_drop = peak_food - food
                gold_drop = peak_gold - gold
                if (
                    0.7 * expected_food <= food_drop <= 1.3 * expected_food
                    and 0.7 * expected_gold <= gold_drop <= 1.3 * expected_gold
                ):
                    if self.candidate != next_age:
                        self.visual.clear()
                        started_age = next_age
                    self.candidate = next_age
                    self.candidate_at = now
            self.previous = sample
            self.previous_at = now
            self.recent.append(sample)

        visual_age = self._visual_age(texts.get("age"))
        self.visual.append(visual_age)
        if self.candidate is not None:
            if now - self.candidate_at > 180.0:
                self.candidate = None
                self.visual.clear()
                self.visual_streak = 0
            elif visual_age == self.candidate:
                self.visual_streak += 1
            else:
                self.visual_streak = 0
            if (
                self.candidate is not None
                and self.visual_streak >= _AGE_VISUAL_STREAK
            ):
                self.current = self.candidate
                completed_age = self.current
                self.candidate = None
                self.visual.clear()
                self.visual_streak = 0
        return self._ROMAN[self.current], started_age, completed_age


def _coherent_population(observations: tuple[Observation, ...]) -> tuple[Observation, ...]:
    population = {
        item.field: item
        for item in observations
        if item.field in ("population_current", "population_cap")
    }
    if not population:
        return observations
    current = population.get("population_current")
    cap = population.get("population_cap")
    coherent = (
        current is not None
        and cap is not None
        and current.value <= cap.value
    )
    if coherent:
        return observations
    return tuple(
        item
        for item in observations
        if item.field not in ("population_current", "population_cap")
    )



def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_methods(component: object, name: str, methods: tuple[str, ...]) -> None:
    for method in methods:
        if not callable(getattr(component, method, None)):
            raise TypeError(f"{name} must expose a callable {method} method")


class SmartHudPipeline:
    """Orchestrate scheduler, reader, parser, confirmer, and estimator."""

    def __init__(
        self,
        reader: object,
        scheduler: ObservationScheduler | None = None,
        estimator: StateEstimator | None = None,
        confirmer: HudEvidenceConfirmer | None = None,
        profile_validated: bool = False,
        age_signal_validated: bool = False,
    ) -> None:
        read = getattr(reader, "read", None)
        if not callable(read):
            raise TypeError("reader must expose a callable read(frame)")
        self._reader = reader
        if scheduler is None:
            scheduler = ObservationScheduler()
        _require_methods(scheduler, "scheduler", ("due", "mark_sample", "reset"))
        if estimator is None:
            estimator = StateEstimator()
        _require_methods(estimator, "estimator", ("ingest", "mark_stale", "reset"))
        if not hasattr(estimator, "state"):
            raise TypeError("estimator must expose state")
        if confirmer is None:
            confirmer = HudEvidenceConfirmer()
        _require_methods(
            confirmer, "confirmer", ("checkpoint", "accept", "restore", "reset")
        )
        self._scheduler = scheduler
        self._estimator = estimator
        self._confirmer = confirmer
        self._profile_validated = _require_bool(
            profile_validated, "profile_validated"
        )
        self._age_signal_validated = _require_bool(
            age_signal_validated, "age_signal_validated"
        )
        self._age_signal = _AgeSignal()
        self._emitted_age_completions: set[int] = set()
        self._sample_index = 0

    @property
    def scheduler(self) -> ObservationScheduler:
        return self._scheduler

    @property
    def estimator(self) -> StateEstimator:
        return self._estimator

    @property
    def confirmer(self) -> HudEvidenceConfirmer:
        return self._confirmer

    @property
    def profile_validated(self) -> bool:
        return self._profile_validated

    @property
    def age_signal_validated(self) -> bool:
        return self._age_signal_validated

    @property
    def state(self) -> GameState:
        return self._estimator.state

    def set_validation(
        self,
        profile_validated: bool,
        age_signal_validated: bool = False,
    ) -> None:
        profile = _require_bool(profile_validated, "profile_validated")
        age = _require_bool(age_signal_validated, "age_signal_validated")
        if profile != self._profile_validated or age != self._age_signal_validated:
            self._profile_validated = profile
            self._age_signal_validated = age
            self._confirmer.reset()

    def process(self, frame: CapturedFrame) -> PipelineResult:
        if not isinstance(frame, CapturedFrame):
            raise TypeError(
                f"frame must be a CapturedFrame, got {type(frame).__name__}"
            )
        return self._process_sample(
            frame.frame_id,
            frame.captured_at,
            lambda: self._reader.read(frame),
        )

    def process_crops(
        self, crops: Mapping[str, CapturedFrame]
    ) -> PipelineResult:
        if not crops:
            raise ValueError("crops must be nonempty")
        first = next(iter(crops.values()))
        return self._process_sample(
            first.frame_id,
            first.captured_at,
            lambda: self._reader.read_crops(crops),
            attempted_fields=set(crops),
        )

    def _process_sample(
        self, frame_id: int, now: float, read_evidence, attempted_fields=None
    ) -> PipelineResult:

        if not self._scheduler.due(now):
            state = self._estimator.mark_stale(now)
            return PipelineResult(
                sampled=False,
                evidence={},
                observations=(),
                verified_observations=(),
                state=state,
                events=(),
            )

        raw = read_evidence()
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"reader.read must return a Mapping, got {type(raw).__name__}"
            )
        evidence = raw

        texts = dict(text_mapping(evidence))
        age_started = None
        age_completed = None
        if "food" in texts and "gold" in texts:
            normalized_age, age_started, age_completed = self._age_signal.observe(
                texts, now
            )
            if "age" in texts:
                texts["age"] = normalized_age
        observations = _coherent_population(parse_hud_text(texts, frame_id, now))

        checkpoint = self._confirmer.checkpoint()
        try:
            verified = self._confirmer.accept(
                evidence,
                observations,
                self._sample_index,
                self._profile_validated,
                self._age_signal_validated,
                attempted_fields=attempted_fields,
            )
            verified = _coherent_population(verified)
            state, events = self._estimator.ingest(verified, now)
            events = tuple(
                event
                for event in events
                if not (
                    event.event_type == "age_completed"
                    and event.payload.get("age") in self._emitted_age_completions
                )
            )
            if age_started is not None:
                started_event = GameEvent(
                    sequence=0,
                    event_type="age_started",
                    payload={"age": age_started, "frame_id": frame_id},
                    evidence_ids=(),
                    confidence=1.0,
                    status=RecognitionStatus.CONFIRMED,
                    emitted_at=now,
                )
                events = (started_event,) + events
            if age_completed is not None:
                self._emitted_age_completions.add(age_completed)
                completed_event = GameEvent(
                    sequence=0,
                    event_type="age_completed",
                    payload={"age": age_completed, "frame_id": frame_id},
                    evidence_ids=(),
                    confidence=1.0,
                    status=RecognitionStatus.CONFIRMED,
                    emitted_at=now,
                )
                events = events + (completed_event,)
            state = self._estimator.mark_stale(now)
            self._scheduler.mark_sample(now)
        except Exception:
            self._confirmer.restore(checkpoint)
            raise
        self._sample_index += 1

        return PipelineResult(
            sampled=True,
            evidence=evidence,
            observations=observations,
            verified_observations=verified,
            state=state,
            events=events,
        )

    def reset(self) -> GameState:
        self._scheduler.reset()
        self._confirmer.reset()
        self._age_signal.reset()
        self._emitted_age_completions.clear()
        self._sample_index = 0
        return self._estimator.reset()
