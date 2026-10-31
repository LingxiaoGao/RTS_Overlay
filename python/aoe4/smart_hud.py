from __future__ import annotations

import re
from typing import Mapping

from aoe4.smart_runtime import Observation, RecognitionStatus
from aoe4.smart_sampling import ObservationScheduler, SamplingConfig

_MAX_CONFIDENCE = 0.85
_INFERRED_CONFIDENCE = 0.8

_GAME_TIME_RE = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})$")
_AGE_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
_AGE_WORD = {"dark": 1, "feudal": 2, "castle": 3, "imperial": 4}
_AGE_NUMERIC = {"1": 1, "2": 2, "3": 3, "4": 4}
_AGE_RE = re.compile(
    r"^(?:age\s*)?(i{1,3}|iv)$|^(dark|feudal|castle|imperial)(?:\s*age)?$"
)

_RESOURCE_FIELDS = (
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
_FIELD_ORDER = ("game_time", "age") + _RESOURCE_FIELDS

_SEPARATOR_MAP = {
    "\u00a0": " ",
    "\u2007": " ",
    "\u202f": " ",
    "\u2009": " ",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_context(observation_id: str, frame_id: int, observed_at: float) -> bool:
    if not isinstance(observation_id, str) or not observation_id:
        return False
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        return False
    if frame_id < 0:
        return False
    if not _is_number(observed_at):
        return False
    return True


def _check_observed_at(observed_at: float) -> float:
    if isinstance(observed_at, bool) or not isinstance(observed_at, (int, float)):
        raise TypeError("observed_at must be numeric")
    return float(observed_at)


def _make(
    observation_id: str,
    field: str,
    value: object,
    confidence: float,
    frame_id: int,
    observed_at: float,
) -> Observation:
    if confidence > _MAX_CONFIDENCE:
        confidence = _MAX_CONFIDENCE
    return Observation(
        observation_id=observation_id,
        field=field,
        value=value,
        confidence=confidence,
        status=RecognitionStatus.INFERRED,
        frame_id=frame_id,
        observed_at=observed_at,
    )


def _normalize_text(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    for source, target in _SEPARATOR_MAP.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def parse_game_time(
    text: str,
    observation_id: str,
    frame_id: int,
    observed_at: float,
) -> Observation | None:
    if not _validate_context(observation_id, frame_id, observed_at):
        return None
    normalized = _normalize_text(text)
    if normalized is None or not normalized:
        return None
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace(".", ":").replace(";", ":").replace(":", ":")
    match = _GAME_TIME_RE.match(normalized)
    if match is None:
        return None
    hours_part, minutes_part, seconds_part = match.groups()
    seconds = int(seconds_part)
    minutes = int(minutes_part)
    hours = int(hours_part) if hours_part is not None else 0
    if seconds >= 60:
        return None
    if hours_part is not None and minutes >= 60:
        return None
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds < 0:
        return None
    value = float(total_seconds)
    return _make(
        observation_id,
        "game_time",
        value,
        _INFERRED_CONFIDENCE,
        frame_id,
        _check_observed_at(observed_at),
    )


def parse_age(
    text: str,
    observation_id: str,
    frame_id: int,
    observed_at: float,
) -> Observation | None:
    if not _validate_context(observation_id, frame_id, observed_at):
        return None
    normalized = _normalize_text(text)
    if normalized is None or not normalized:
        return None
    normalized = normalized.lower()
    normalized = " ".join(normalized.split())
    if normalized in _AGE_NUMERIC:
        return _make(
            observation_id,
            "age",
            _AGE_NUMERIC[normalized],
            _INFERRED_CONFIDENCE,
            frame_id,
            _check_observed_at(observed_at),
        )
    match = _AGE_RE.match(normalized)
    if match is None:
        return None
    roman, word = match.groups()
    if roman is not None:
        age = _AGE_ROMAN.get(roman)
    else:
        age = _AGE_WORD.get(word)
    if age is None:
        return None
    return _make(
        observation_id,
        "age",
        age,
        _INFERRED_CONFIDENCE,
        frame_id,
        _check_observed_at(observed_at),
    )


def _parse_count(text: object) -> int | None:
    normalized = _normalize_text(text)
    if normalized is None or not normalized:
        return None
    normalized = normalized.replace(" ", "").replace("\u00a0", "")
    if not normalized:
        return None
    if "." in normalized and normalized.count(".") > 1:
        return None
    if "." in normalized and "," in normalized:
        return None
    if "." in normalized:
        head, tail = normalized.split(".", 1)
        if not head or not tail:
            return None
        if "." in head or "." in tail:
            return None
        if not head.isdigit() or not tail.isdigit():
            return None
        if len(tail) == 3 and len(head) >= 1:
            return int(head + tail)
        return None
    if "," in normalized:
        groups = normalized.split(",")
        if not all(group.isdigit() and group for group in groups):
            return None
        if len(groups) == 1:
            return None
        if len(groups) > 2:
            return None
        if len(groups[1]) != 3:
            return None
        return int(groups[0] + groups[1])
    if not normalized.isdigit():
        return None
    return int(normalized)


def parse_resource(
    field: str,
    text: str,
    observation_id: str,
    frame_id: int,
    observed_at: float,
) -> Observation | None:
    if field not in _RESOURCE_FIELDS:
        return None
    if not _validate_context(observation_id, frame_id, observed_at):
        return None
    normalized = _normalize_text(text)
    if normalized is None:
        return None
    if normalized.startswith("-") or normalized.endswith("-"):
        return None
    if "+" in normalized:
        return None
    value = _parse_count(normalized)
    if value is None:
        return None
    return _make(
        observation_id,
        field,
        value,
        _INFERRED_CONFIDENCE,
        frame_id,
        _check_observed_at(observed_at),
    )


def parse_hud_text(
    mapping: Mapping[str, object],
    frame_id: int,
    observed_at: float,
) -> tuple[Observation, ...]:
    if not isinstance(mapping, Mapping):
        return ()
    now = _check_observed_at(observed_at)
    results: list[Observation] = []
    for field in _FIELD_ORDER:
        if field not in mapping:
            continue
        raw = mapping[field]
        observation_id = f"{field}-{frame_id}"
        if field == "game_time":
            parsed = parse_game_time(raw, observation_id, frame_id, now)
        elif field == "age":
            parsed = parse_age(raw, observation_id, frame_id, now)
        else:
            parsed = parse_resource(field, raw, observation_id, frame_id, now)
        if parsed is not None:
            results.append(parsed)
    return tuple(results)


class HudSampler:
    """Cadence-gated wrapper around :func:`parse_hud_text`."""

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self._scheduler = ObservationScheduler(config or SamplingConfig())

    @property
    def scheduler(self) -> ObservationScheduler:
        return self._scheduler

    def sample(self, now: float, text: Mapping[str, object], frame_id: int) -> tuple[Observation, ...]:
        if not self._scheduler.due(now):
            return ()
        observations = parse_hud_text(text, frame_id, now)
        self._scheduler.mark_sample(now)
        return observations
