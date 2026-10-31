from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from aoe4.smart_runtime import GameState, RecognitionStatus

_RESOURCES = ("food", "wood", "gold", "stone")
_VILLAGER_FIELDS = {
    "food": "villagers_food",
    "wood": "villagers_wood",
    "gold": "villagers_gold",
    "stone": "villagers_stone",
}


class AdvicePriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Advice:
    key: str
    priority: AdvicePriority
    message: str


def _copy_int_mapping(
    value: Any, name: str, *, allow_negative: bool
) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        if key not in _RESOURCES:
            raise ValueError(f"unknown {name} resource: {key!r}")
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name}[{key!r}] must be int")
        if not allow_negative and item < 0:
            raise ValueError(f"{name}[{key!r}] must be >= 0")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True)
class EconomyTarget:
    villagers: Mapping[str, int]
    stockpiles: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "villagers",
            _copy_int_mapping(self.villagers, "villagers", allow_negative=False),
        )
        object.__setattr__(
            self,
            "stockpiles",
            _copy_int_mapping(self.stockpiles, "stockpiles", allow_negative=False),
        )

    @classmethod
    def from_build_order_step(cls, step: Any) -> "EconomyTarget":
        if not isinstance(step, Mapping):
            raise TypeError("build order step must be a mapping")

        villagers: dict[str, int] = {}
        resources = step.get("resources")
        if resources is not None:
            if not isinstance(resources, Mapping):
                raise TypeError("resources must be a mapping")
            for name in _RESOURCES:
                if name not in resources:
                    continue
                value = resources[name]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"resources[{name!r}] must be int")
                if value < 0:
                    continue
                villagers[name] = value

        stockpiles: dict[str, int] = {}
        smart = step.get("smart_resource_targets")
        if smart is not None:
            if not isinstance(smart, Mapping):
                raise TypeError("smart_resource_targets must be a mapping")
            for key, value in smart.items():
                if not isinstance(key, str) or key not in _RESOURCES:
                    raise ValueError(f"unknown smart resource target key: {key!r}")
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"smart_resource_targets[{key!r}] must be int")
                if value < 0:
                    raise ValueError(
                        f"smart_resource_targets[{key!r}] must be >= 0"
                    )
                stockpiles[key] = value

        return cls(villagers=villagers, stockpiles=stockpiles)


def _check_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class AdvisorConfig:
    villager_difference_threshold: int = 2
    population_headroom: int = 2
    stockpile_tolerance: int = 50

    def __post_init__(self) -> None:
        _check_int(
            self.villager_difference_threshold,
            "villager_difference_threshold",
            1,
        )
        _check_int(self.population_headroom, "population_headroom", 0)
        _check_int(self.stockpile_tolerance, "stockpile_tolerance", 0)


def _confirmed_value(state: GameState, name: str) -> Any:
    field_state = getattr(state, name)
    if field_state.status is not RecognitionStatus.CONFIRMED:
        return None
    return field_state.value


class AdvisorEngine:
    def __init__(self, config: AdvisorConfig | None = None) -> None:
        if config is None:
            config = AdvisorConfig()
        if not isinstance(config, AdvisorConfig):
            raise TypeError("config must be an AdvisorConfig")
        self._config = config

    def evaluate(
        self, state: GameState, target: EconomyTarget
    ) -> tuple[Advice, ...]:
        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        if not isinstance(target, EconomyTarget):
            raise TypeError("target must be an EconomyTarget")

        advice: list[Advice] = []
        config = self._config

        idle = _confirmed_value(state, "idle_villagers")
        if idle is not None and idle > 0:
            advice.append(
                Advice(
                    key="idle_villagers",
                    priority=AdvicePriority.HIGH,
                    message=f"有 {idle} 个闲置村民，请安排工作。",
                )
            )

        cap = _confirmed_value(state, "population_cap")
        current = _confirmed_value(state, "population_current")
        if (
            cap is not None
            and current is not None
            and cap >= current
            and cap - current <= config.population_headroom
        ):
            advice.append(
                Advice(
                    key="population_cap",
                    priority=AdvicePriority.HIGH,
                    message=(
                        f"人口接近上限（{current}/{cap}），请建造房屋。"
                    ),
                )
            )

        for resource in _RESOURCES:
            if resource not in target.villagers:
                continue
            current_villagers = _confirmed_value(
                state, _VILLAGER_FIELDS[resource]
            )
            if current_villagers is None:
                continue
            target_villagers = target.villagers[resource]
            diff = target_villagers - current_villagers
            if abs(diff) < config.villager_difference_threshold:
                continue
            label = _RESOURCE_LABELS[resource]
            if diff > 0:
                message = f"请为{label}增加 {diff} 个村民。"
            else:
                message = f"请从{label}移除 {-diff} 个村民。"
            advice.append(
                Advice(
                    key=f"villagers_{resource}",
                    priority=AdvicePriority.MEDIUM,
                    message=message,
                )
            )

        for resource in _RESOURCES:
            if resource not in target.stockpiles:
                continue
            current_stock = _confirmed_value(state, resource)
            if current_stock is None:
                continue
            target_stock = target.stockpiles[resource]
            label = _RESOURCE_LABELS[resource]
            if current_stock < target_stock - config.stockpile_tolerance:
                advice.append(
                    Advice(
                        key=f"stockpile_{resource}_low",
                        priority=AdvicePriority.MEDIUM,
                        message=(
                            f"{label}储备偏低（{current_stock}/{target_stock}），"
                            "请增加采集。"
                        ),
                    )
                )
            elif current_stock > target_stock + config.stockpile_tolerance:
                advice.append(
                    Advice(
                        key=f"stockpile_{resource}_high",
                        priority=AdvicePriority.LOW,
                        message=(
                            f"{label}储备偏高（{current_stock}/{target_stock}），"
                            "可调整采集或消耗。"
                        ),
                    )
                )

        return tuple(advice)


_RESOURCE_LABELS = {
    "food": "食物",
    "wood": "木材",
    "gold": "黄金",
    "stone": "石料",
}
