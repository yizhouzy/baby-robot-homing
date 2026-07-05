"""Capture Pi camera frames and log the orange HSV target features."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.baby_hardware import append_csv_row, make_run_dir, write_metadata


ORANGE_HSV_RANGES = (
    ((5, 70, 50), (20, 255, 255)),
)


def isolate_orange(frame: np.ndarray) -> np.ndarray:
    """Return a binary mask for the orange target color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = None
    for lower, upper in ORANGE_HSV_RANGES:
        range_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)
    return mask


def analyze_sections(mask: np.ndarray) -> list[float]:
    """Convert the mask to the 7 vision features used by the behavior code."""
    h, w = mask.shape
    strips = [
        float(cv2.countNonZero(section)) / section.size
        for section in np.array_split(mask, 5, axis=1)
    ]
    total_pixels = cv2.countNonZero(mask)
    if total_pixels > 0:
        moments = cv2.moments(mask)
        centroid_x = (moments["m10"] / moments["m00"] / w) * 2.0 - 1.0
    else:
        centroid_x = 0.0
    area = float(total_pixels) / float(h * w)
    return strips + [float(centroid_x), area]


VISION_CSV_FIELDS = [
    "elapsed_s",
    "fps",
    "section_0",
    "section_1",
    "section_2",
    "section_3",
    "section_4",
    "centroid_x",
    "area",
    "orange_pixels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Picamera2 with the HSV mask.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def open_camera(width: int, height: int):
    """Create a Picamera2 RGB stream at the requested resolution."""
    from picamera2 import Picamera2

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
    )
    camera.configure(config)
    camera.start()
    time.sleep(0.5)
    return camera


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir("test_camera_hsv", args.output_dir)
    raw_dir = run_dir / "raw_frames"
    mask_dir = run_dir / "mask_frames"
    if args.save_frames:
        raw_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

    vision_csv = run_dir / "vision_samples.csv"
    raw_video = None
    mask_video = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        raw_video = cv2.VideoWriter(
            str(run_dir / "raw_video.mp4"),
            fourcc,
            args.hz,
            (args.width, args.height),
        )
        mask_video = cv2.VideoWriter(
            str(run_dir / "mask_video.mp4"),
            fourcc,
            args.hz,
            (args.width, args.height),
        )

    camera = open_camera(args.width, args.height)
    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/test_camera_hsv.py",
            "duration": args.duration,
            "hz": args.hz,
            "width": args.width,
            "height": args.height,
            "save_frames": args.save_frames,
            "save_video": args.save_video,
            "output_dir": str(run_dir),
        },
    )

    print(f"Vision log: {vision_csv}")
    print("The HSV mask is implemented locally so this script does not need ARIEL.")

    period_s = 1.0 / args.hz
    start = time.monotonic()
    frame_count = 0
    try:
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            elapsed_s = loop_start - start
            frame = camera.capture_array()
            mask = isolate_orange(frame)
            vision = analyze_sections(mask)
            frame_count += 1
            fps = frame_count / max(elapsed_s, 1e-6)
            orange_pixels = int(cv2.countNonZero(mask))

            append_csv_row(
                vision_csv,
                VISION_CSV_FIELDS,
                {
                    "elapsed_s": f"{elapsed_s:.6f}",
                    "fps": f"{fps:.6f}",
                    "section_0": f"{vision[0]:.6f}",
                    "section_1": f"{vision[1]:.6f}",
                    "section_2": f"{vision[2]:.6f}",
                    "section_3": f"{vision[3]:.6f}",
                    "section_4": f"{vision[4]:.6f}",
                    "centroid_x": f"{vision[5]:.6f}",
                    "area": f"{vision[6]:.6f}",
                    "orange_pixels": orange_pixels,
                },
            )

            if args.save_frames:
                cv2.imwrite(
                    str(raw_dir / f"frame_{frame_count:05d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(str(mask_dir / f"mask_{frame_count:05d}.png"), mask)

            if raw_video is not None and mask_video is not None:
                raw_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                mask_video.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            print(
                f"t={elapsed_s:6.2f}s  fps={fps:5.1f}  "
                f"area={vision[6]:7.4f}  centroid={vision[5]:+6.3f}"
            )
            time.sleep(max(0.0, period_s - (time.monotonic() - loop_start)))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.stop()
        if raw_video is not None:
            raw_video.release()
        if mask_video is not None:
            mask_video.release()

    print(f"Done. Results: {run_dir}")


if __name__ == "__main__":
    main()
