"""Record raw camera vision and HSV mask without moving the robot.

Use this script to tune the orange HSV thresholds before running a gait. The
saved video has the raw robot-camera view on the left and the binary mask on
the right, matching the diagnostic layout used by
``record_vision_during_gaits.py``.
"""
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
from hardware.test.test_camera_hsv import analyze_sections, open_camera


VISION_MASK_CSV_FIELDS = [
    "elapsed_s",
    "fps",
    "bearing",
    "area",
    "orange_pixels",
    "section_0",
    "section_1",
    "section_2",
    "section_3",
    "section_4",
    "center_h",
    "center_s",
    "center_v",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record camera + HSV mask without gait.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--camera-order",
        choices=["rgb", "bgr"],
        default="bgr",
        help="Channel order returned by Picamera2 capture_array(). Use bgr if blue/orange look swapped.",
    )
    parser.add_argument("--lower-hsv", type=int, nargs=3, default=(5, 70, 50))
    parser.add_argument("--upper-hsv", type=int, nargs=3, default=(20, 255, 255))
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--filename", default="vision_mask_only.mp4")
    parser.add_argument("--raw-filename", default="raw_robot_vision.mp4")
    parser.add_argument("--mask-filename", default="mask_video.mp4")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def camera_frame_to_rgb(frame: np.ndarray, camera_order: str) -> np.ndarray:
    """Normalize camera frames before HSV thresholding and video writing."""
    if camera_order == "rgb":
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def isolate_hsv(frame_rgb: np.ndarray, lower_hsv, upper_hsv) -> np.ndarray:
    """Threshold one RGB frame with the currently tested HSV bounds."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(
        hsv,
        np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8),
    )


def center_hsv(frame_rgb: np.ndarray) -> tuple[int, int, int]:
    """Measure the HSV value at the center patch for quick threshold tuning."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    patch = hsv[h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4]
    mean_hsv = patch.reshape(-1, 3).mean(axis=0)
    return int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])


def draw_mask_diagnostics(mask: np.ndarray, bearing: float) -> np.ndarray:
    """Draw section boundaries and the measured bearing on the mask panel."""
    h, w = mask.shape
    mask_view = np.zeros((h, w, 3), dtype=np.uint8)
    mask_view[mask > 0] = (0, 140, 255)
    for section in range(1, 5):
        x = int(round(section * w / 5))
        cv2.line(mask_view, (x, 0), (x, h - 1), (90, 90, 90), 1)
    centroid_x = int(round(((bearing + 1.0) * 0.5) * (w - 1)))
    cv2.line(mask_view, (centroid_x, 0), (centroid_x, h - 1), (255, 255, 255), 1)
    return mask_view


def compose_video_frame(
    camera_rgb: np.ndarray,
    mask: np.ndarray,
    elapsed_s: float,
    vision: list[float],
    hsv_center: tuple[int, int, int],
    lower_hsv,
    upper_hsv,
) -> np.ndarray:
    """Build one side-by-side BGR frame for OpenCV VideoWriter."""
    raw_bgr = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = draw_mask_diagnostics(mask, float(vision[5]))
    frame = np.concatenate([raw_bgr, mask_bgr], axis=1)
    h, w = camera_rgb.shape[:2]

    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), markerSize=12, thickness=1)
    cv2.putText(frame, "robot vision", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, "HSV mask", (w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"t={elapsed_s:05.2f}s  bearing={vision[5]:+.3f}  area={vision[6]:.4f}",
        (10, h - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"center HSV={hsv_center}  range={tuple(lower_hsv)}-{tuple(upper_hsv)}",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return frame


def write_vision_row(
    csv_path: Path,
    elapsed_s: float,
    fps: float,
    vision: list[float],
    orange_pixels: int,
    hsv_center: tuple[int, int, int],
) -> None:
    append_csv_row(
        csv_path,
        VISION_MASK_CSV_FIELDS,
        {
            "elapsed_s": f"{elapsed_s:.6f}",
            "fps": f"{fps:.6f}",
            "bearing": f"{vision[5]:.6f}",
            "area": f"{vision[6]:.6f}",
            "orange_pixels": orange_pixels,
            "section_0": f"{vision[0]:.6f}",
            "section_1": f"{vision[1]:.6f}",
            "section_2": f"{vision[2]:.6f}",
            "section_3": f"{vision[3]:.6f}",
            "section_4": f"{vision[4]:.6f}",
            "center_h": hsv_center[0],
            "center_s": hsv_center[1],
            "center_v": hsv_center[2],
        },
    )


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir("record_vision_mask_only", args.output_dir)
    video_path = run_dir / args.filename
    raw_video_path = run_dir / args.raw_filename
    mask_video_path = run_dir / args.mask_filename
    vision_csv = run_dir / "vision_mask_samples.csv"
    raw_dir = run_dir / "raw_frames"
    mask_dir = run_dir / "mask_frames"
    if args.save_frames:
        raw_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/record_vision_mask_only.py",
            "duration": args.duration,
            "video_fps": args.video_fps,
            "width": args.width,
            "height": args.height,
            "camera_order": args.camera_order,
            "lower_hsv": list(args.lower_hsv),
            "upper_hsv": list(args.upper_hsv),
            "save_frames": args.save_frames,
            "video_path": str(video_path),
            "raw_video_path": str(raw_video_path),
            "mask_video_path": str(mask_video_path),
            "vision_csv": str(vision_csv),
            "output_dir": str(run_dir),
        },
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, args.video_fps, (args.width * 2, args.height))
    raw_video = cv2.VideoWriter(str(raw_video_path), fourcc, args.video_fps, (args.width, args.height))
    mask_video = cv2.VideoWriter(str(mask_video_path), fourcc, args.video_fps, (args.width, args.height))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    camera = open_camera(args.width, args.height)
    print(f"Results: {run_dir}")
    print(f"Combined video: {video_path}")
    print(f"Raw video:      {raw_video_path}")
    print(f"Mask video:     {mask_video_path}")
    print(f"CSV:            {vision_csv}")
    print(f"HSV:     lower={tuple(args.lower_hsv)} upper={tuple(args.upper_hsv)}")
    print(f"Camera order: {args.camera_order}")

    period_s = 1.0 / args.video_fps
    start = time.monotonic()
    frame_count = 0
    try:
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            elapsed_s = loop_start - start
            camera_rgb = camera_frame_to_rgb(camera.capture_array(), args.camera_order)
            mask = isolate_hsv(camera_rgb, args.lower_hsv, args.upper_hsv)
            vision = analyze_sections(mask)
            hsv_center = center_hsv(camera_rgb)
            orange_pixels = int(cv2.countNonZero(mask))
            frame_count += 1
            fps = frame_count / max(elapsed_s, 1e-6)

            video.write(
                compose_video_frame(
                    camera_rgb,
                    mask,
                    elapsed_s,
                    vision,
                    hsv_center,
                    args.lower_hsv,
                    args.upper_hsv,
                ),
            )
            raw_video.write(cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR))
            mask_video.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            write_vision_row(vision_csv, elapsed_s, fps, vision, orange_pixels, hsv_center)

            if args.save_frames:
                cv2.imwrite(str(raw_dir / f"frame_{frame_count:05d}.png"), cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_dir / f"mask_{frame_count:05d}.png"), mask)

            print(
                f"t={elapsed_s:6.2f}s  fps={fps:5.1f}  "
                f"bearing={vision[5]:+6.3f}  area={vision[6]:7.4f}  "
                f"center_hsv={hsv_center}"
            )
            time.sleep(max(0.0, period_s - (time.monotonic() - loop_start)))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.stop()
        video.release()
        raw_video.release()
        mask_video.release()

    print(f"Done. Results: {run_dir}")


if __name__ == "__main__":
    main()
