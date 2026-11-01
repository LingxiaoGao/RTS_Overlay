"""Pure helpers for laying out AoE4 build-order age blocks.

No Qt or UI imports: given timer steps (mappings with ``notes`` and optional
integer ``age``) and the source current row index, produce separators to
insert before source rows and the remapped display current row.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


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


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _flatten_notes(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return " ".join(item for item in value if isinstance(item, str)).lower()
    return ""


def _find_starts(steps: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    """Return the first source row index for each age that has a start marker."""
    starts: Dict[int, int] = {}
    used_ages: set = set()
    next_age = 2
    for index, step in enumerate(steps):
        text = _flatten_notes(step.get("notes"))
        specific: Optional[int] = None
        for age, markers in _SPECIFIC_MARKERS:
            if any(marker in text for marker in markers):
                specific = age
                break
        if specific is not None:
            if specific in starts:
                continue
            starts[specific] = index
            used_ages.add(specific)
            continue
        if any(marker in text for marker in _AGE_UP_MARKERS):
            while next_age in used_ages or next_age in starts:
                next_age += 1
            if next_age > 4:
                continue
            starts[next_age] = index
            used_ages.add(next_age)
    return starts


def build_age_block_layout(
    steps: Sequence[Mapping[str, Any]], current_row: int
) -> Tuple[Dict[int, Tuple[Tuple[str, int], ...]], int]:
    """Compute age block separators and the remapped display current row.

    ``separators`` maps a source row index to a tuple of ``(kind, age)``
    entries inserted before that row. ``display_current_row`` is the source
    current row shifted by the number of separators inserted before or at it.
    """
    starts = _find_starts(steps)

    entries: Dict[int, List[Tuple[str, int]]] = {}

    def _add(row: int, kind: str, age: int) -> None:
        entries.setdefault(row, []).append((kind, age))

    for age, start_index in starts.items():
        _add(start_index, "start", age)
        completion: Optional[int] = None
        for index in range(start_index + 1, len(steps)):
            value = steps[index].get("age")
            if _is_int(value) and value == age:
                completion = index
                break
        if completion is not None:
            _add(completion, "completion", age)

    # Completion-only fallback: first row where the integer age changes to 2/3/4.
    previous_age: Optional[int] = None
    emitted_completions = set()
    for age, start_index in starts.items():
        for index in range(start_index + 1, len(steps)):
            value = steps[index].get("age")
            if _is_int(value) and value == age:
                emitted_completions.add(age)
                break

    for index, step in enumerate(steps):
        value = step.get("age")
        if not _is_int(value) or value not in (2, 3, 4):
            continue
        if value == previous_age:
            continue
        previous_age = value
        if value in starts:
            continue
        if value in emitted_completions:
            continue
        emitted_completions.add(value)
        _add(index, "completion", value)

    separators: Dict[int, Tuple[Tuple[str, int], ...]] = {
        row: tuple(items) for row, items in entries.items()
    }

    display_current_row = current_row
    for row, items in separators.items():
        if row <= current_row:
            display_current_row += len(items)

    return separators, display_current_row
