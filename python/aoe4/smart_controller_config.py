"""Load and validate smart controller configuration from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aoe4.smart_capture import CaptureRegion
from aoe4.smart_regions import HudRegionSet


_REQUIRED_KEYS = frozenset({"capture_region", "hud_regions"})
_OPTIONAL_KEYS = frozenset({"profile_validated", "age_signal_validated"})
_ALLOWED_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS
_REGION_KEYS = frozenset({"left", "top", "width", "height"})


def _load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {path!r}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config {path!r} must be a JSON object")
    return data


def _parse_region(value: Any, path: str) -> CaptureRegion:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path} must be a nonempty object")
    if set(value) != _REGION_KEYS:
        raise ValueError(f"{path} must have exactly keys {sorted(_REGION_KEYS)}")
    coords = {
        key: value[key]
        for key in _REGION_KEYS
    }
    for key, val in coords.items():
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{path}.{key} must be an integer")
    try:
        return CaptureRegion(**coords)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {path}: {exc}") from exc


def _parse_hud_regions(value: Any, path: str) -> HudRegionSet:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path} must be a nonempty object")
    regions: dict[str, CaptureRegion] = {}
    for name, region_value in value.items():
        if not isinstance(name, str):
            raise ValueError(f"{path} keys must be strings")
        region = _parse_region(region_value, f"{path}.{name}")
        regions[name] = region
    try:
        return HudRegionSet(regions)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {path}: {exc}") from exc


def _check_hud_contained(
    capture_region: CaptureRegion,
    hud_regions: HudRegionSet,
) -> None:
    for name, region in hud_regions.regions.items():
        if (
            region.left < capture_region.left
            or region.top < capture_region.top
            or region.right > capture_region.right
            or region.bottom > capture_region.bottom
        ):
            raise ValueError(
                f"HUD region {name!r} must be contained in capture_region"
            )


@dataclass(frozen=True)
class SmartControllerConfig:
    capture_region: CaptureRegion
    hud_regions: HudRegionSet
    profile_validated: bool = False
    age_signal_validated: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "SmartControllerConfig":
        data = _load_json_object(path)
        unknown = set(data) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        missing = _REQUIRED_KEYS - set(data)
        if missing:
            raise ValueError(f"missing config keys: {sorted(missing)}")

        capture_region = _parse_region(data["capture_region"], "capture_region")
        hud_regions = _parse_hud_regions(data["hud_regions"], "hud_regions")
        _check_hud_contained(capture_region, hud_regions)

        profile_validated = data.get("profile_validated", False)
        age_signal_validated = data.get("age_signal_validated", False)
        for key, value in (
            ("profile_validated", profile_validated),
            ("age_signal_validated", age_signal_validated),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")

        return cls(
            capture_region=capture_region,
            hud_regions=hud_regions,
            profile_validated=profile_validated,
            age_signal_validated=age_signal_validated,
        )
