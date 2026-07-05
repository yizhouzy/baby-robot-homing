"""Record a plain Raspberry Pi camera video with Picamera2.

This script does not touch the Robohat or servos. Use it to verify camera
hardware, focus, exposure, and file writing before adding HSV processing.
"""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.baby_hardware import make_run_dir, write_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a camera-only video.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--filename", default="camera_video.mp4")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir("record_camera_video", args.output_dir)
    video_path = run_dir / args.filename

    from picamera2 import Picamera2

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (args.width, args.height)},
    )
    camera.configure(config)

    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/record_camera_video.py",
            "duration": args.duration,
            "width": args.width,
            "height": args.height,
            "video_path": str(video_path),
            "output_dir": str(run_dir),
        },
    )

    print(f"Recording {args.duration:.1f}s video to {video_path}")
    try:
        camera.start_and_record_video(str(video_path), duration=args.duration)
    finally:
        camera.stop()

    print(f"Done. Results: {run_dir}")


if __name__ == "__main__":
    main()
