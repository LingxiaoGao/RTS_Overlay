"""Crop immutable BGRA frames to fixed HUD regions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from aoe4.smart_capture import CapturedFrame, CaptureRegion
from aoe4.smart_runtime import FIELD_NAMES

_BYTES_PER_PIXEL = 4


def crop_frame(frame: CapturedFrame, region: CaptureRegion) -> CapturedFrame:
    """Return a copy of ``frame`` restricted to global ``region``."""
    if not isinstance(frame, CapturedFrame):
        raise TypeError(f"frame must be a CapturedFrame, got {type(frame).__name__}")
    if not isinstance(region, CaptureRegion):
        raise TypeError(f"region must be a CaptureRegion, got {type(region).__name__}")

    source = frame.region
    if (
        region.left < source.left
        or region.top < source.top
        or region.right > source.right
        or region.bottom > source.bottom
    ):
        raise ValueError("region must be fully contained in frame.region")

    expected = frame.width * frame.height * _BYTES_PER_PIXEL
    if not isinstance(frame.pixels, bytes):
        raise ValueError("frame.pixels must be bytes")
    if len(frame.pixels) != expected:
        raise ValueError(
            f"frame.pixels length {len(frame.pixels)} != expected {expected}"
        )

    stride = frame.width * _BYTES_PER_PIXEL
    x_offset = (region.left - source.left) * _BYTES_PER_PIXEL
    row_bytes = region.width * _BYTES_PER_PIXEL
    y_start = region.top - source.top

    chunks: list[bytes] = []
    for row in range(y_start, y_start + region.height):
        start = row * stride + x_offset
        chunks.append(frame.pixels[start : start + row_bytes])

    return CapturedFrame(
        frame_id=frame.frame_id,
        captured_at=frame.captured_at,
        width=region.width,
        height=region.height,
        pixels=b"".join(chunks),
        region=region,
    )


@dataclass(frozen=True)
class HudRegionSet:
    """Immutable mapping of HUD field names to capture regions."""

    _regions: Mapping[str, CaptureRegion]

    def __init__(self, regions: Mapping[str, CaptureRegion]) -> None:
        if not isinstance(regions, Mapping):
            raise TypeError(f"regions must be a Mapping, got {type(regions).__name__}")
        if not regions:
            raise ValueError("regions must be nonempty")

        ordered: dict[str, CaptureRegion] = {}
        for name, region in regions.items():
            if not isinstance(name, str):
                raise TypeError(f"region key must be str, got {type(name).__name__}")
            if name not in FIELD_NAMES:
                raise ValueError(f"unknown field name: {name!r}")
            if not isinstance(region, CaptureRegion):
                raise TypeError(
                    f"region for {name!r} must be a CaptureRegion, "
                    f"got {type(region).__name__}"
                )
            ordered[name] = region

        ordered = {name: ordered[name] for name in FIELD_NAMES if name in ordered}
        object.__setattr__(self, "_regions", MappingProxyType(ordered))

    @property
    def regions(self) -> Mapping[str, CaptureRegion]:
        return self._regions

    def crop(self, frame: CapturedFrame) -> Mapping[str, CapturedFrame]:
        if not isinstance(frame, CapturedFrame):
            raise TypeError(
                f"frame must be a CapturedFrame, got {type(frame).__name__}"
            )
        cropped = {name: crop_frame(frame, region) for name, region in self._regions.items()}
        return cropped
