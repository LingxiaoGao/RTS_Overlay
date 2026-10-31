"""Extract HUD crops from an AoE4 video for offline testing.

Hobby-scale utility: sample frames from a video at a fixed interval, convert
each frame to the runtime BGRA layout, scale the configured HUD regions onto
the video's frame size, and write every cropped field as a PNG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from aoe4.smart_capture import CapturedFrame, CaptureRegion  # noqa: E402
from aoe4.smart_controller_config import SmartControllerConfig  # noqa: E402
from aoe4.smart_regions import scale_regions  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AoE4 HUD region crops from a video file.",
    )
    parser.add_argument("video", help="path to the input video")
    parser.add_argument(
        "--config",
        default="python/demo_data/smart_hud_config.json",
        help="path to the smart controller config JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory (default: derived from video name)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="sampling interval in seconds",
    )
    return parser.parse_args(argv)


def _write_png(frame: CapturedFrame, path: Path) -> None:
    image = memoryview(frame.pixels)
    bgr = cv2.cvtColor(
        __import__("numpy").asarray(image, dtype="uint8").reshape(
            frame.height, frame.width, 4
        ),
        cv2.COLOR_BGRA2BGR,
    )
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write {path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")

    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")

    config = SmartControllerConfig.from_file(config_path)

    output_dir = (
        Path(args.output)
        if args.output is not None
        else video_path.with_suffix("")
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise SystemExit(f"video has no frames: {video_path}")

    sample_index = 0
    written = 0
    while True:
        timestamp = sample_index * args.interval_seconds
        if duration > 0 and timestamp > duration:
            break
        msec = timestamp * 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, msec)
        ok, frame = cap.read()
        if not ok:
            break

        height, width = frame.shape[:2]
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        target = CaptureRegion(0, 0, width, height)
        captured = CapturedFrame(
            frame_id=sample_index,
            captured_at=timestamp,
            width=width,
            height=height,
            pixels=bgra.tobytes(),
            region=target,
        )

        scaled = scale_regions(config.hud_regions, config.capture_region, target)
        for field, crop in scaled.crop(captured).items():
            field_dir = output_dir / field
            field_dir.mkdir(parents=True, exist_ok=True)
            name = f"{int(round(timestamp * 1000)):010d}.png"
            _write_png(crop, field_dir / name)
            written += 1

        sample_index += 1

    cap.release()
    print(f"wrote {written} crops to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
