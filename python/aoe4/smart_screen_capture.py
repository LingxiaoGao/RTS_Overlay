"""Synchronous MSS-backed screen capture adapter."""

from __future__ import annotations

from typing import Callable, Optional

from aoe4.smart_capture import CaptureRegion, CapturedFrame, Number

BackendFactory = Callable[[], object]


class MssScreenCapture:
    """Grab screen regions via MSS in a small, testable adapter."""

    def __init__(self, backend_factory: Optional[BackendFactory] = None) -> None:
        if backend_factory is not None and not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self._backend_factory = backend_factory
        self._backend: Optional[object] = None

    def _make_backend(self) -> object:
        if self._backend_factory is not None:
            return self._backend_factory()
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError(
                "MSS is required for screen capture; install mss==10.2.0"
            ) from exc
        return mss.mss()

    def _get_backend(self) -> object:
        if self._backend is None:
            self._backend = self._make_backend()
        return self._backend

    def screen_region(self) -> CaptureRegion:
        """Return the primary monitor bounds used for HUD coordinate scaling."""
        monitor = self._get_backend().monitors[1]
        return CaptureRegion(
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
        )

    def capture(
        self, region: CaptureRegion, frame_id: int, captured_at: Number
    ) -> CapturedFrame:
        if not isinstance(region, CaptureRegion):
            raise TypeError(
                f"region must be a CaptureRegion, got {type(region).__name__}"
            )
        backend = self._get_backend()
        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        shot = backend.grab(monitor)
        width = shot.width
        height = shot.height
        if width != region.width or height != region.height:
            raise ValueError(
                f"screenshot dimensions {width}x{height} do not match region "
                f"{region.width}x{region.height}"
            )
        pixels = bytes(shot.bgra)
        expected = width * height * 4
        if len(pixels) != expected:
            raise ValueError(f"screenshot buffer length {len(pixels)} != {expected}")
        return CapturedFrame(
            frame_id=frame_id,
            captured_at=captured_at,
            width=width,
            height=height,
            pixels=pixels,
            region=region,
        )

    def close(self) -> None:
        backend = self._backend
        if backend is None:
            return
        self._backend = None
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "MssScreenCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
