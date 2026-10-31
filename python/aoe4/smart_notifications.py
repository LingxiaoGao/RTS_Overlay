from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

from aoe4.smart_advisor import Advice, AdvicePriority

_PRIORITY_ORDER = {
    AdvicePriority.HIGH: 0,
    AdvicePriority.MEDIUM: 1,
    AdvicePriority.LOW: 2,
}


@dataclass(frozen=True)
class Notification:
    key: str
    priority: AdvicePriority
    message: str
    created_at: float


def _check_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _check_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


@dataclass(frozen=True)
class NotificationConfig:
    cooldown_seconds: float = 15.0
    max_per_batch: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cooldown_seconds",
            _check_positive_float(self.cooldown_seconds, "cooldown_seconds"),
        )
        object.__setattr__(
            self,
            "max_per_batch",
            _check_positive_int(self.max_per_batch, "max_per_batch"),
        )


def _check_now(now: object) -> float:
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise TypeError("now must be a number")
    result = float(now)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("now must be finite and nonnegative")
    return result


class NotificationManager:
    def __init__(self, config: NotificationConfig | None = None) -> None:
        if config is None:
            config = NotificationConfig()
        if not isinstance(config, NotificationConfig):
            raise TypeError("config must be a NotificationConfig")
        self._config = config
        self._last_emitted: dict[str, float] = {}

    def publish(
        self, advice: Iterable[Advice], now: float
    ) -> Tuple[Notification, ...]:
        current = _check_now(now)
        items: list[Advice] = []
        for item in advice:
            if not isinstance(item, Advice):
                raise TypeError("advice items must be Advice instances")
            items.append(item)

        ordered = sorted(
            items, key=lambda item: _PRIORITY_ORDER.get(item.priority, 99)
        )

        selected: list[Advice] = []
        seen: set[str] = set()
        for item in ordered:
            if item.key in seen:
                continue
            seen.add(item.key)
            last = self._last_emitted.get(item.key)
            if last is not None and current - last < self._config.cooldown_seconds:
                continue
            selected.append(item)

        emitted = selected[: self._config.max_per_batch]
        notifications: list[Notification] = []
        for item in emitted:
            self._last_emitted[item.key] = current
            notifications.append(
                Notification(
                    key=item.key,
                    priority=item.priority,
                    message=item.message,
                    created_at=current,
                )
            )
        return tuple(notifications)

    def reset(self) -> None:
        self._last_emitted.clear()
