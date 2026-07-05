"""Record real camera vision while the robot runs spin and forward gaits.

The output video is a side-by-side diagnostic:

* left: the Raspberry Pi camera RGB image from the robot
* right: the orange HSV mask used by the behavior-tree vision logic

The CSV log stores the same bearing and area features used by the behavior
code, plus the gait phase and commanded servo angles. This script is only for
testing vision during motion; it does not make autonomous decisions from the
camera.
"""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blocks.cpg_runtime import load_gait_runtime
from hardware.baby_hardware import (
    DEFAULT_SERVO_MAPPINGS,
    BabyRobotHardware,
    append_csv_row,
    make_run_dir,
    mapping_metadata,
    write_metadata,
)
from hardware.test.run_gait_no_camera import (
    DEFAULT_GAIT_MODELS,
    infer_meta_path,
    resolve_existing_path,
)
from hardware.test.test_camera_hsv import analyze_sections, isolate_orange, open_camera


DEFAULT_JOINT_SCALES = (0.9, 0.9, 0.9, 1.0, 0.5, 0.5, 1.0, 1.0)
DEFAULT_SPIN_GAIT = "left"
DEFAULT_FORWARD_GAIT = "forward"

VISION_GAIT_CSV_FIELDS = [
    "elapsed_s",
    "phase",
    "phase_elapsed_s",
    "fps",
    "bearing",
    "area",
    "orange_pixels",
    "section_0",
    "section_1",
    "section_2",
    "section_3",
    "section_4",
    *[f"joint_{i}_rad" for i in range(8)],
    *[f"servo_{i}_command_deg" for i in range(8)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record raw and masked camera video while running spin then forward gait.",
    )
    parser.add_argument("--spin-gait", choices=["left", "right"], default=DEFAULT_SPIN_GAIT)
    parser.add_argument("--spin-duration", type=float, default=10.0)
    parser.add_argument("--forward-duration", type=float, default=10.0)
    parser.add_argument("--spin-model", type=Path, default=None)
    parser.add_argument("--spin-meta", type=Path, default=None)
    parser.add_argument("--forward-model", type=Path, default=None)
    parser.add_argument("--forward-meta", type=Path, default=None)
    parser.add_argument("--control-hz", type=float, default=25.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--camera-order",
        choices=["rgb", "bgr"],
        default="bgr",
        help="Channel order returned by Picamera2 capture_array(). Use bgr if blue/orange look swapped.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--joint-scales", type=float, nargs=8, default=DEFAULT_JOINT_SCALES)
    parser.add_argument("--max-joint-deg", type=float, default=90.0)
    parser.add_argument("--neutral-hold-s", type=float, default=1.0)
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--delay-mode", action="store_true")
    parser.add_argument("--filename", default="vision_during_gaits.mp4")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_model_and_meta(
    gait: str,
    model_arg: Path | None,
    meta_arg: Path | None,
) -> tuple[Path, Path]:
    if model_arg is None:
        model_path, meta_path = DEFAULT_GAIT_MODELS[gait]
    else:
        model_path = model_arg
        meta_path = meta_arg if meta_arg is not None else infer_meta_path(model_path)
    return resolve_existing_path(model_path, ".npy"), resolve_existing_path(meta_path, ".npz")


def reset_network(network) -> None:
    """Start each gait from the same CPG state used after loading its weights."""
    network.cpg.reset()


def next_action(network, speed: float, amplitude_scale: float, joint_scales, max_joint_deg: float):
    action = network.forward(turn=0.0, speed=speed)
    action = action * float(amplitude_scale)
    action = action * np.asarray(joint_scales, dtype=np.float32)
    max_joint_rad = math.radians(max_joint_deg)
    return action.clip(-max_joint_rad, max_joint_rad)


def camera_frame_to_rgb(frame: np.ndarray, camera_order: str) -> np.ndarray:
    """Normalize the camera array before HSV thresholding and display."""
    if camera_order == "rgb":
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_mask_diagnostics(mask: np.ndarray, bearing: float) -> np.ndarray:
    """Make the binary mask easier to inspect in the saved video."""
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
    phase: str,
    elapsed_s: float,
    phase_elapsed_s: float,
    bearing: float,
    area: float,
) -> np.ndarray:
    raw_bgr = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = draw_mask_diagnostics(mask, bearing)
    frame = np.concatenate([raw_bgr, mask_bgr], axis=1)

    cv2.putText(frame, "robot vision", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        frame,
        "HSV mask",
        (camera_rgb.shape[1] + 10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"{phase}  t={elapsed_s:05.2f}s  phase={phase_elapsed_s:05.2f}s",
        (10, camera_rgb.shape[0] - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"bearing={bearing:+.3f}  area={area:.4f}",
        (10, camera_rgb.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return frame


def write_vision_row(
    csv_path: Path,
    elapsed_s: float,
    phase: str,
    phase_elapsed_s: float,
    fps: float,
    vision: list[float],
    orange_pixels: int,
    action,
    robot: BabyRobotHardware,
) -> None:
    row = {
        "elapsed_s": f"{elapsed_s:.6f}",
        "phase": phase,
        "phase_elapsed_s": f"{phase_elapsed_s:.6f}",
        "fps": f"{fps:.6f}",
        "bearing": f"{vision[5]:.6f}",
        "area": f"{vision[6]:.6f}",
        "orange_pixels": orange_pixels,
        "section_0": f"{vision[0]:.6f}",
        "section_1": f"{vision[1]:.6f}",
        "section_2": f"{vision[2]:.6f}",
        "section_3": f"{vision[3]:.6f}",
        "section_4": f"{vision[4]:.6f}",
    }
    row.update({f"joint_{i}_rad": f"{value:.6f}" for i, value in enumerate(action)})
    row.update(
        {
            f"servo_{i}_command_deg": f"{value:.6f}"
            for i, value in enumerate(robot.last_command_degrees)
        },
    )
    append_csv_row(csv_path, VISION_GAIT_CSV_FIELDS, row)


def run_phase(
    *,
    phase: str,
    duration_s: float,
    network,
    robot: BabyRobotHardware,
    camera,
    video,
    csv_path: Path,
    args: argparse.Namespace,
    global_start_s: float,
    frame_count: int,
    next_frame_s: float,
    next_print_s: float,
) -> tuple[int, float, float]:
    reset_network(network)
    control_period_s = 1.0 / args.control_hz
    video_period_s = 1.0 / args.video_fps
    phase_start_s = time.monotonic()

    while time.monotonic() - phase_start_s < duration_s:
        loop_start_s = time.monotonic()
        elapsed_s = loop_start_s - global_start_s
        phase_elapsed_s = loop_start_s - phase_start_s

        action = next_action(
            network,
            args.speed,
            args.amplitude_scale,
            args.joint_scales,
            args.max_joint_deg,
        )
        robot.set_joint_angles(action)

        if elapsed_s >= next_frame_s:
            camera_rgb = camera_frame_to_rgb(camera.capture_array(), args.camera_order)
            mask = isolate_orange(camera_rgb)
            vision = analyze_sections(mask)
            frame_count += 1
            fps = frame_count / max(elapsed_s, 1e-6)
            orange_pixels = int(cv2.countNonZero(mask))
            video.write(
                compose_video_frame(
                    camera_rgb,
                    mask,
                    phase,
                    elapsed_s,
                    phase_elapsed_s,
                    float(vision[5]),
                    float(vision[6]),
                ),
            )
            write_vision_row(
                csv_path,
                elapsed_s,
                phase,
                phase_elapsed_s,
                fps,
                vision,
                orange_pixels,
                action,
                robot,
            )
            next_frame_s += video_period_s

        if elapsed_s >= next_print_s:
            print(
                f"t={elapsed_s:6.2f}s  {phase:7s}  "
                f"bearing={vision[5]:+6.3f}  area={vision[6]:7.4f}  "
                f"servo0={robot.last_command_degrees[0]:6.2f}deg",
            )
            next_print_s += 1.0

        time.sleep(max(0.0, control_period_s - (time.monotonic() - loop_start_s)))

    return frame_count, next_frame_s, next_print_s


def main() -> None:
    args = parse_args()
    spin_model, spin_meta = resolve_model_and_meta(args.spin_gait, args.spin_model, args.spin_meta)
    forward_model, forward_meta = resolve_model_and_meta(
        DEFAULT_FORWARD_GAIT,
        args.forward_model,
        args.forward_meta,
    )
    spin_network, spin_metadata, spin_weight_format = load_gait_runtime(spin_model, spin_meta)
    forward_network, forward_metadata, forward_weight_format = load_gait_runtime(
        forward_model,
        forward_meta,
    )

    run_dir = make_run_dir("record_vision_during_gaits", args.output_dir)
    video_path = run_dir / args.filename
    vision_csv = run_dir / "vision_gait_samples.csv"
    direct_mode = not args.delay_mode

    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/record_vision_during_gaits.py",
            "spin_gait": args.spin_gait,
            "spin_duration": args.spin_duration,
            "forward_duration": args.forward_duration,
            "speed": args.speed,
            "amplitude_scale": args.amplitude_scale,
            "joint_scales": list(args.joint_scales),
            "max_joint_deg": args.max_joint_deg,
            "control_hz": args.control_hz,
            "video_fps": args.video_fps,
            "width": args.width,
            "height": args.height,
            "camera_order": args.camera_order,
            "spin_model": str(spin_model),
            "spin_meta": str(spin_meta),
            "spin_weight_format": spin_weight_format,
            "spin_num_joints": int(spin_metadata["num_joints"]),
            "forward_model": str(forward_model),
            "forward_meta": str(forward_meta),
            "forward_weight_format": forward_weight_format,
            "forward_num_joints": int(forward_metadata["num_joints"]),
            "direct_mode": direct_mode,
            "robohat_camera_enabled": False,
            "delay_s": args.delay_s,
            "neutral_hold_s": args.neutral_hold_s,
            "video_path": str(video_path),
            "vision_csv": str(vision_csv),
            "mappings": mapping_metadata(DEFAULT_SERVO_MAPPINGS),
            "output_dir": str(run_dir),
        },
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        str(video_path),
        fourcc,
        args.video_fps,
        (args.width * 2, args.height),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    robot = BabyRobotHardware(
        direct_mode=direct_mode,
        delay_s=args.delay_s,
        enable_camera=False,
    )
    camera = open_camera(args.width, args.height)

    print(f"Results: {run_dir}")
    print(f"Video:   {video_path}")
    print(f"CSV:     {vision_csv}")
    print(f"Spin:    {args.spin_gait} for {args.spin_duration:.1f}s")
    print(f"Forward: forward for {args.forward_duration:.1f}s")
    print(f"Camera order: {args.camera_order}")
    print(f"Joint scales: {list(args.joint_scales)}")

    frame_count = 0
    next_frame_s = 0.0
    next_print_s = 0.0
    try:
        print(f"Moving to neutral for {args.neutral_hold_s:.1f}s before recording.")
        robot.neutral()
        time.sleep(args.neutral_hold_s)
        global_start_s = time.monotonic()

        frame_count, next_frame_s, next_print_s = run_phase(
            phase=args.spin_gait,
            duration_s=args.spin_duration,
            network=spin_network,
            robot=robot,
            camera=camera,
            video=video,
            csv_path=vision_csv,
            args=args,
            global_start_s=global_start_s,
            frame_count=frame_count,
            next_frame_s=next_frame_s,
            next_print_s=next_print_s,
        )
        run_phase(
            phase=DEFAULT_FORWARD_GAIT,
            duration_s=args.forward_duration,
            network=forward_network,
            robot=robot,
            camera=camera,
            video=video,
            csv_path=vision_csv,
            args=args,
            global_start_s=global_start_s,
            frame_count=frame_count,
            next_frame_s=next_frame_s,
            next_print_s=next_print_s,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"Returning to neutral for {args.neutral_hold_s:.1f}s.")
        robot.neutral()
        time.sleep(args.neutral_hold_s)
        robot.close()
        camera.stop()
        video.release()

    print(f"Done. Results: {run_dir}")


if __name__ == "__main__":
    main()
