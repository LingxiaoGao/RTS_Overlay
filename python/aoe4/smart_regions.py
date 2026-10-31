"""Crop immutable BGRA frames to fixed HUD regions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from aoe4.smart_capture import CapturedFrame, CaptureRegion
from aoe4.smart_runtime import FIELD_NAMES

_BYTES_PER_PIXEL = 4


def _scale_edge(start: int, size: int, source_total: int, target_start: int,
                target_total: int) -> tuple[int, int]:
    """Scale ``start``/``size`` from source into target coordinates.

    Uses independent x/y handled by the caller; here the work is purely on a
    single axis. Rounding is deterministic: scaled edges are computed as
    ``round(v * target_total / source_total)`` via integer arithmetic. A
    positive source region remains at least one pixel after downscaling.
    """
    # Round-half-up integer scaling of both edges, then derive size from the
    # scaled edges so adjacent tiles stay contiguous where possible.
    numerator_start = start * target_total
    numerator_end = (start + size) * target_total
    scaled_start = (numerator_start + source_total // 2) // source_total
    scaled_end = (numerator_end + source_total // 2) // source_total

    scaled_size = scaled_end - scaled_start
    if scaled_size < 1:
        scaled_size = 1

    return target_start + scaled_start, scaled_size


def scale_regions(
    regions: HudRegionSet,
    reference: CaptureRegion,
    target: CaptureRegion,
) -> HudRegionSet:
    """Scale ``regions`` from ``reference`` into ``target`` capture space.

    Global frame origins are preserved: output regions are expressed in the
    global coordinate space of ``target`` (i.e. offset by ``target.left/top``).
    """
    if not isinstance(regions, HudRegionSet):
        raise TypeError(
            f"regions must be a HudRegionSet, got {type(regions).__name__}"
        )
    if not isinstance(reference, CaptureRegion):
        raise TypeError(
            f"reference must be a CaptureRegion, got {type(reference).__name__}"
        )
    if not isinstance(target, CaptureRegion):
        raise TypeError(
            f"target must be a CaptureRegion, got {type(target).__name__}"
        )

    scaled: dict[str, CaptureRegion] = {}
    for name, region in regions.regions.items():
        rel_left = region.left - reference.left
        rel_top = region.top - reference.top
        left, width = _scale_edge(
            rel_left, region.width, reference.width, target.left, target.width
        )
        top, height = _scale_edge(
            rel_top, region.height, reference.height, target.top, target.height
        )
        scaled[name] = CaptureRegion(left, top, width, height)

    return HudRegionSet(scaled)


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
