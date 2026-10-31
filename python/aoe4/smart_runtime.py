from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

FIELD_NAMES = (
    "game_time",
    "age",
    "food",
    "wood",
    "gold",
    "stone",
    "population_current",
    "population_cap",
    "idle_villagers",
    "villagers_food",
    "villagers_wood",
    "villagers_gold",
    "villagers_stone",
    "villagers_total",
)
INT_FIELDS = FIELD_NAMES[1:]
AGE_LIMIT = 4
AGE_EVENTS = (2, 3, 4)


class RecognitionStatus(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"
    STALE = "stale"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("payload keys must be nonempty strings")
            out[key] = _freeze(item)
        return MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise TypeError("set payload values are not supported")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported payload value: {type(value).__name__}")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    field: str
    value: Any
    confidence: float
    status: RecognitionStatus
    frame_id: int
    observed_at: float


@dataclass(frozen=True)
class GameEvent:
    sequence: int
    event_type: str
    payload: Mapping[str, Any]
    evidence_ids: tuple[str, ...]
    confidence: float
    status: RecognitionStatus
    emitted_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))


@dataclass(frozen=True)
class FieldState:
    value: Any = None
    confidence: float = 0.0
    status: RecognitionStatus = RecognitionStatus.UNKNOWN
    observed_at: float = 0.0
    evidence_ids: tuple[str, ...] = ()

    @staticmethod
    def unknown() -> "FieldState":
        return FieldState()


@dataclass(frozen=True)
class GameState:
    game_time: FieldState = FieldState()
    age: FieldState = FieldState()
    food: FieldState = FieldState()
    wood: FieldState = FieldState()
    gold: FieldState = FieldState()
    stone: FieldState = FieldState()
    population_current: FieldState = FieldState()
    population_cap: FieldState = FieldState()
    idle_villagers: FieldState = FieldState()
    villagers_food: FieldState = FieldState()
    villagers_wood: FieldState = FieldState()
    villagers_gold: FieldState = FieldState()
    villagers_stone: FieldState = FieldState()
    villagers_total: FieldState = FieldState()
    updated_at: float = 0.0


@dataclass(frozen=True)
class EstimatorConfig:
    confirmation_frames: int = 3
    stale_after_seconds: float = 5.0
    timer_max_forward_jump_seconds: float = 2.0
    confidence_threshold: float = 0.8

    def __post_init__(self) -> None:
        if isinstance(self.confirmation_frames, bool) or not isinstance(self.confirmation_frames, int):
            raise TypeError("confirmation_frames must be int")
        if self.confirmation_frames < 2:
            raise ValueError("confirmation_frames must be >= 2")
        if isinstance(self.stale_after_seconds, bool) or not isinstance(self.stale_after_seconds, (int, float)):
            raise TypeError("stale_after_seconds must be numeric")
        if not math.isfinite(self.stale_after_seconds) or self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be > 0")
        if isinstance(self.timer_max_forward_jump_seconds, bool) or not isinstance(
            self.timer_max_forward_jump_seconds, (int, float)
        ):
            raise TypeError("timer_max_forward_jump_seconds must be numeric")
        if (
            not math.isfinite(self.timer_max_forward_jump_seconds)
            or self.timer_max_forward_jump_seconds <= 0
        ):
            raise ValueError("timer_max_forward_jump_seconds must be > 0")
        if isinstance(self.confidence_threshold, bool) or not isinstance(
            self.confidence_threshold, (int, float)
        ):
            raise TypeError("confidence_threshold must be numeric")
        if not math.isfinite(self.confidence_threshold):
            raise ValueError("confidence_threshold must be finite")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")


@dataclass(frozen=True)
class _Candidate:
    value: Any
    last_frame: int
    observed_at: float
    evidence_ids: tuple[str, ...]
    confidences: tuple[float, ...]


_REJECTED = object()


def _check_int(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value < minimum:
        raise ValueError(f"{name} below minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} above maximum")
    return value


def _check_float(
    value: Any, name: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} below minimum")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} above maximum")
    return number


def _validate_field_value(name: str, value: Any) -> Any:
    if name == "game_time":
        return _check_float(value, name, minimum=0.0)
    if name == "age":
        return _check_int(value, name, minimum=1, maximum=AGE_LIMIT)
    return _check_int(value, name, minimum=0)


def _validate_observation(obs: Observation) -> None:
    if not isinstance(obs, Observation):
        raise TypeError("each observation must be an Observation")
    if not isinstance(obs.observation_id, str) or not obs.observation_id:
        raise ValueError("observation_id must be a nonempty string")
    if not isinstance(obs.field, str) or obs.field not in FIELD_NAMES:
        raise ValueError(f"unknown field: {obs.field!r}")
    _check_int(obs.frame_id, "frame_id")
    if not isinstance(obs.status, RecognitionStatus):
        raise TypeError("status must be RecognitionStatus")
    _check_float(obs.confidence, "confidence", minimum=0.0, maximum=1.0)
    _check_float(obs.observed_at, "observed_at", minimum=0.0)
    _validate_field_value(obs.field, obs.value)


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


class StateEstimator:
    def __init__(self, config: EstimatorConfig | None = None) -> None:
        if config is None:
            config = EstimatorConfig()
        if not isinstance(config, EstimatorConfig):
            raise TypeError("config must be an EstimatorConfig")
        self._config = config
        self._reset_internal(now=0.0)

    def _reset_internal(self, now: float) -> GameState:
        self._state = GameState(updated_at=now)
        self._candidates: dict[str, _Candidate] = {}
        self._accepted_timer: _Candidate | None = None
        self._confirmed_age: int | None = None
        self._emitted_ages: set[int] = set()
        self._sequence = 0
        return self._state

    @property
    def state(self) -> GameState:
        return self._state

    def reset(self) -> GameState:
        return self._reset_internal(now=0.0)

    def mark_stale(self, now: float) -> GameState:
        _check_float(now, "now", minimum=0.0)
        current = self._state
        updates: dict[str, FieldState] = {}
        for name in FIELD_NAMES:
            field_state = getattr(current, name)
            if field_state.status in (RecognitionStatus.UNKNOWN, RecognitionStatus.STALE):
                continue
            if now - field_state.observed_at > self._config.stale_after_seconds:
                updates[name] = FieldState(
                    value=field_state.value,
                    confidence=field_state.confidence,
                    status=RecognitionStatus.STALE,
                    observed_at=field_state.observed_at,
                    evidence_ids=field_state.evidence_ids,
                )
        snapshot = GameState(
            updated_at=now,
            **{name: updates.get(name, getattr(current, name)) for name in FIELD_NAMES},
        )
        self._state = snapshot
        return snapshot

    def ingest(
        self, iterable: Iterable[Observation], now: float
    ) -> tuple[GameState, tuple[GameEvent, ...]]:
        _check_float(now, "now", minimum=0.0)
        if isinstance(iterable, (str, bytes)):
            raise TypeError("observations must be an iterable of Observation")
        try:
            materialized = tuple(iterable)
        except TypeError as exc:
            raise TypeError("observations must be iterable") from exc

        seen: set[tuple[str, int]] = set()
        for obs in materialized:
            _validate_observation(obs)
            key = (obs.field, obs.frame_id)
            if key in seen:
                raise ValueError(f"duplicate (field, frame_id): {key}")
            seen.add(key)

        ordered = sorted(materialized, key=lambda o: (o.frame_id, o.field, o.observation_id))

        working: dict[str, FieldState] = {name: getattr(self._state, name) for name in FIELD_NAMES}
        candidates: dict[str, _Candidate] = dict(self._candidates)
        accepted = self._accepted_timer
        confirmed_age = self._confirmed_age
        emitted = set(self._emitted_ages)
        sequence = self._sequence
        events: list[GameEvent] = []

        for obs in ordered:
            name = obs.field
            confirmed_ok = (
                obs.status is RecognitionStatus.CONFIRMED
                and obs.confidence >= self._config.confidence_threshold
            )
            prior = candidates.get(name)
            candidate: _Candidate | None = None

            if confirmed_ok:
                if (
                    prior is not None
                    and obs.frame_id > prior.last_frame
                    and (
                        _same_value(prior.value, obs.value)
                        or (
                            name == "game_time"
                            and isinstance(prior.value, float)
                            and isinstance(obs.value, float)
                            and 0.0 <= obs.value - prior.value <= self._config.timer_max_forward_jump_seconds
                        )
                    )
                ):
                    candidate = _Candidate(
                        value=obs.value,
                        last_frame=obs.frame_id,
                        observed_at=obs.observed_at,
                        evidence_ids=prior.evidence_ids + (obs.observation_id,),
                        confidences=prior.confidences + (float(obs.confidence),),
                    )
                else:
                    candidate = _Candidate(
                        value=obs.value,
                        last_frame=obs.frame_id,
                        observed_at=obs.observed_at,
                        evidence_ids=(obs.observation_id,),
                        confidences=(float(obs.confidence),),
                    )

            if name == "game_time":
                if candidate is not None:
                    candidates[name] = candidate
                result = self._apply_timer(obs, candidate, working, accepted, now)
                if result is _REJECTED:
                    candidates.pop(name, None)
                elif result is not None:
                    accepted = result
                elif candidate is None:
                    candidates.pop(name, None)
                continue

            if candidate is None:
                candidates.pop(name, None)
                if obs.status is RecognitionStatus.INFERRED:
                    working[name] = FieldState(
                        value=obs.value,
                        confidence=float(obs.confidence),
                        status=RecognitionStatus.INFERRED,
                        observed_at=obs.observed_at,
                        evidence_ids=(obs.observation_id,),
                    )
                else:
                    prior_state = working[name]
                    if prior_state.status is RecognitionStatus.CONFIRMED:
                        working[name] = FieldState(
                            value=prior_state.value,
                            confidence=prior_state.confidence,
                            status=RecognitionStatus.STALE,
                            observed_at=prior_state.observed_at,
                            evidence_ids=prior_state.evidence_ids,
                        )
                continue

            candidates[name] = candidate
            if len(candidate.evidence_ids) >= self._config.confirmation_frames:
                if name == "age":
                    if confirmed_age is None:
                        confirmed_age = candidate.value
                    elif candidate.value < confirmed_age:
                        prior_state = working[name]
                        working[name] = FieldState(
                            value=prior_state.value,
                            confidence=prior_state.confidence,
                            status=RecognitionStatus.STALE,
                            observed_at=prior_state.observed_at,
                            evidence_ids=prior_state.evidence_ids,
                        )
                        candidates.pop(name, None)
                        continue
                    else:
                        newly = candidate.value
                        if (
                            newly == confirmed_age + 1
                            and newly in AGE_EVENTS
                            and newly not in emitted
                        ):
                            sequence += 1
                            emitted.add(newly)
                            events.append(
                                GameEvent(
                                    sequence=sequence,
                                    event_type="age_completed",
                                    payload={"age": newly, "frame_id": candidate.last_frame},
                                    evidence_ids=candidate.evidence_ids,
                                    confidence=float(candidate.confidences[-1]),
                                    status=RecognitionStatus.CONFIRMED,
                                    emitted_at=now,
                                )
                            )
                        confirmed_age = newly
                working[name] = FieldState(
                    value=candidate.value,
                    confidence=float(candidate.confidences[-1]),
                    status=RecognitionStatus.CONFIRMED,
                    observed_at=candidate.observed_at,
                    evidence_ids=candidate.evidence_ids,
                )
            else:
                prior_state = working[name]
                status = prior_state.status
                if status is RecognitionStatus.INFERRED:
                    status = RecognitionStatus.UNKNOWN
                working[name] = FieldState(
                    value=prior_state.value,
                    confidence=prior_state.confidence,
                    status=status,
                    observed_at=prior_state.observed_at,
                    evidence_ids=prior_state.evidence_ids,
                )

        self._validate_population(working)
        new_state = GameState(updated_at=now, **working)
        self._state = new_state
        self._candidates = candidates
        self._accepted_timer = accepted
        self._confirmed_age = confirmed_age
        self._emitted_ages = emitted
        self._sequence = sequence
        return new_state, tuple(events)

    def _apply_timer(
        self,
        obs: Observation,
        candidate: _Candidate | None,
        working: dict[str, FieldState],
        accepted: _Candidate | None,
        now: float,
    ) -> _Candidate | object | None:
        if candidate is None:
            prior_state = working["game_time"]
            if prior_state.status is RecognitionStatus.CONFIRMED:
                working["game_time"] = FieldState(
                    value=prior_state.value,
                    confidence=prior_state.confidence,
                    status=RecognitionStatus.STALE,
                    observed_at=prior_state.observed_at,
                    evidence_ids=prior_state.evidence_ids,
                )
            return None

        if len(candidate.evidence_ids) < self._config.confirmation_frames:
            return None

        if accepted is None:
            working["game_time"] = FieldState(
                value=candidate.value,
                confidence=float(candidate.confidences[-1]),
                status=RecognitionStatus.CONFIRMED,
                observed_at=candidate.observed_at,
                evidence_ids=candidate.evidence_ids,
            )
            return candidate

        delta = candidate.value - accepted.value
        if delta < 0 or delta > self._config.timer_max_forward_jump_seconds:
            working["game_time"] = FieldState(
                value=accepted.value,
                confidence=float(accepted.confidences[-1]),
                status=RecognitionStatus.STALE,
                observed_at=accepted.observed_at,
                evidence_ids=accepted.evidence_ids,
            )
            return _REJECTED

        working["game_time"] = FieldState(
            value=candidate.value,
            confidence=float(candidate.confidences[-1]),
            status=RecognitionStatus.CONFIRMED,
            observed_at=candidate.observed_at,
            evidence_ids=candidate.evidence_ids,
        )
        return candidate

    @staticmethod
    def _validate_population(working: Mapping[str, FieldState]) -> None:
        current = working["population_current"]
        cap = working["population_cap"]
        if current.status is RecognitionStatus.CONFIRMED and cap.status is RecognitionStatus.CONFIRMED:
            if current.value > cap.value:
                raise ValueError("population_current exceeds population_cap")
