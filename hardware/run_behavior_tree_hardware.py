"""Run the camera-driven behavior policy on the physical baby robot.

This is the hardware counterpart of ``experiments/demo_behavior_tree.py``. It
uses selected CPG checkpoints, the Pi camera HSV mask, and the Robohat adapter.

Battery logic follows the simulator convention:
* battery > threshold: high-battery behavior, default IDLE
* battery <= threshold: low-battery homing behavior starts

On hardware, battery is the Robohat ``accu capacity`` percentage, not the raw
pack voltage. Voltage and Robohat status are still logged for diagnosis. If
``--battery-threshold-percent`` is omitted, the script waits for a valid current
capacity and sets the threshold to 100%, so homing starts immediately and is not
interrupted by small capacity-estimator jumps. To make the robot wait in
high-battery IDLE, pass a threshold below the current capacity.
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
from robot_control.artifacts import (
    DEFAULT_GAIT_MODELS,
    infer_checkpoint_meta_path,
    resolve_existing_path,
)
from robot_control.vision import analyze_sections, open_camera
from hardware.baby_hardware import (
    BATTERY_CSV_FIELDS,
    DEFAULT_SERVO_MAPPINGS,
    BabyRobotHardware,
    append_csv_row,
    battery_sample_row,
    make_run_dir,
    mapping_metadata,
    write_metadata,
)


BEHAVIOR_CSV_FIELDS = [
    "elapsed_s",
    "loop_dt_s",
    "mode",
    "selected_gait",
    "battery_v",
    "battery_percentage",
    "battery_status",
    "battery_threshold_percent",
    "fps",
    "visible",
    "reached",
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
    parser = argparse.ArgumentParser(description="Run hardware camera behavior tree.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means run until Ctrl-C.")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--camera-order", choices=["rgb", "bgr"], default="bgr")
    parser.add_argument("--lower-hsv", type=int, nargs=3, default=(157, 210, 230))
    parser.add_argument("--upper-hsv", type=int, nargs=3, default=(180, 255, 255))
    parser.add_argument("--visibility-threshold", type=float, default=0.0015)
    parser.add_argument("--reach-vision-area", type=float, default=0.14)
    parser.add_argument("--bearing-threshold", type=float, default=0.15)
    parser.add_argument("--lost-vision-hold-s", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=None, help="Legacy lost-vision hold in control frames.")
    parser.add_argument("--default-search-gait", choices=["left", "right"], default="left")
    parser.add_argument("--invert-bearing", action="store_true")
    parser.add_argument("--recenter-during-approach", action="store_true", default=False)
    parser.add_argument("--no-recenter-during-approach", dest="recenter_during_approach", action="store_false")
    parser.add_argument("--forward-speed", type=float, default=0.60)
    parser.add_argument("--search-speed", type=float, default=0.25)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--max-joint-deg", type=float, default=90.0)
    parser.add_argument("--forward-model", type=Path, default=None)
    parser.add_argument("--forward-meta", type=Path, default=None)
    parser.add_argument("--left-model", type=Path, default=None)
    parser.add_argument("--left-meta", type=Path, default=None)
    parser.add_argument("--right-model", type=Path, default=None)
    parser.add_argument("--right-meta", type=Path, default=None)
    parser.add_argument("--battery-threshold-percent", type=float, default=None)
    parser.add_argument("--start-margin-percent", type=float, default=100.0)
    parser.add_argument("--min-valid-battery-percent", type=float, default=1.0)
    parser.add_argument("--battery-init-timeout-s", type=float, default=8.0)
    parser.add_argument("--idle-measure-s", type=float, default=0.0)
    parser.add_argument("--high-battery-behavior", choices=["idle", "search"], default="idle")
    parser.add_argument("--stop-when-reached", action="store_true", default=True)
    parser.add_argument("--continue-after-reached", dest="stop_when_reached", action="store_false")
    parser.add_argument("--neutral-hold-s", type=float, default=1.0)
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--delay-mode", action="store_true")
    parser.add_argument("--filename", default="behavior_tree_hardware.mp4")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_model_and_meta(gait: str, model_arg: Path | None, meta_arg: Path | None) -> tuple[Path, Path]:
    if model_arg is None:
        model_path, meta_path = DEFAULT_GAIT_MODELS[gait]
    else:
        model_path = model_arg
        meta_path = meta_arg if meta_arg is not None else infer_checkpoint_meta_path(model_path)
    return resolve_existing_path(model_path, ".npy"), resolve_existing_path(meta_path, ".npz")


def camera_frame_to_rgb(frame: np.ndarray, camera_order: str) -> np.ndarray:
    if camera_order == "rgb":
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def isolate_hsv(frame_rgb: np.ndarray, lower_hsv, upper_hsv) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(
        hsv,
        np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8),
    )


def draw_mask_diagnostics(mask: np.ndarray, bearing: float) -> np.ndarray:
    h, w = mask.shape
    mask_view = np.zeros((h, w, 3), dtype=np.uint8)
    mask_view[mask > 0] = (0, 140, 255)
    for section in range(1, 5):
        x = int(round(section * w / 5))
        cv2.line(mask_view, (x, 0), (x, h - 1), (90, 90, 90), 1)
    centroid_x = int(round(((bearing + 1.0) * 0.5) * (w - 1)))
    cv2.line(mask_view, (centroid_x, 0), (centroid_x, h - 1), (255, 255, 255), 1)
    return mask_view


def compose_video_frame(camera_rgb: np.ndarray, mask: np.ndarray, row: dict) -> np.ndarray:
    raw_bgr = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = draw_mask_diagnostics(mask, float(row["bearing"]))
    frame = np.concatenate([raw_bgr, mask_bgr], axis=1)
    h, w = camera_rgb.shape[:2]
    cv2.putText(frame, "robot vision", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, "HSV mask", (w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        frame,
        (
            f"{row['mode']}  t={float(row['elapsed_s']):05.2f}s  "
            f"battery={int(row['battery_percentage'])}%"
        ),
        (10, h - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"bearing={float(row['bearing']):+.3f}  area={float(row['area']):.4f}",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return frame


def reset_network(network) -> None:
    network.cpg.reset()


def gait_action(network, speed: float, args: argparse.Namespace) -> np.ndarray:
    action = network.forward(turn=0.0, speed=speed)
    action = action * float(args.amplitude_scale)
    max_joint_rad = math.radians(args.max_joint_deg)
    return action.clip(-max_joint_rad, max_joint_rad)


def search_gait_from_bearing(bearing: float, args: argparse.Namespace) -> str:
    signed_bearing = -bearing if args.invert_bearing else bearing
    if abs(signed_bearing) <= args.bearing_threshold:
        return args.default_search_gait
    return "right" if signed_bearing > 0.0 else "left"


class HardwareVisionPolicy:
    def __init__(
        self,
        networks: dict[str, object],
        args: argparse.Namespace,
        battery_threshold_percent: float,
    ) -> None:
        self.networks = networks
        self.args = args
        self.battery_threshold_percent = battery_threshold_percent
        if args.patience is None:
            self.lost_frame_limit = max(1, int(round(args.lost_vision_hold_s * args.control_hz)))
        else:
            self.lost_frame_limit = args.patience
        self.initial_search_complete = False
        self.lost_frames = 0

    def reset(self) -> None:
        for network in self.networks.values():
            reset_network(network)
        self.initial_search_complete = False
        self.lost_frames = 0

    def decide(self, vision: list[float], battery_percentage: float) -> tuple[np.ndarray, str, str]:
        area = float(vision[6])
        bearing = float(vision[5])
        visible = area >= self.args.visibility_threshold

        if battery_percentage > self.battery_threshold_percent:
            if self.args.high_battery_behavior == "idle":
                return np.zeros(8, dtype=np.float32), "IDLE", "none"
            gait = self.args.default_search_gait
            return gait_action(self.networks[gait], self.args.search_speed, self.args), "SEARCHING", gait

        if area >= self.args.reach_vision_area:
            return np.zeros(8, dtype=np.float32), "STOPPED", "none"

        if visible and abs(bearing) <= self.args.bearing_threshold:
            self.initial_search_complete = True
            self.lost_frames = 0
            return gait_action(self.networks["forward"], self.args.forward_speed, self.args), "APPROACHING", "forward"

        if visible and (not self.initial_search_complete or self.args.recenter_during_approach):
            gait = search_gait_from_bearing(bearing, self.args)
            self.lost_frames = 0
            return gait_action(self.networks[gait], self.args.search_speed, self.args), "SEARCHING", gait

        if self.initial_search_complete:
            self.lost_frames += 1
            if self.lost_frames < self.lost_frame_limit:
                return gait_action(self.networks["forward"], self.args.forward_speed, self.args), "APPROACHING", "forward"

        gait = self.args.default_search_gait
        return gait_action(self.networks[gait], self.args.search_speed, self.args), "SEARCHING", gait


def make_behavior_row(
    *,
    elapsed_s: float,
    loop_dt_s: float,
    mode: str,
    selected_gait: str,
    battery_v: float,
    battery_percentage: int,
    battery_status: str,
    threshold_percent: float,
    fps: float,
    vision: list[float],
    orange_pixels: int,
    action,
    robot: BabyRobotHardware,
    args: argparse.Namespace,
) -> dict:
    row = {
        "elapsed_s": f"{elapsed_s:.6f}",
        "loop_dt_s": f"{loop_dt_s:.6f}",
        "mode": mode,
        "selected_gait": selected_gait,
        "battery_v": f"{battery_v:.6f}",
        "battery_percentage": battery_percentage,
        "battery_status": battery_status,
        "battery_threshold_percent": f"{threshold_percent:.6f}",
        "fps": f"{fps:.6f}",
        "visible": int(float(vision[6]) >= args.visibility_threshold),
        "reached": int(float(vision[6]) >= args.reach_vision_area),
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
    return row


def measure_idle_battery(
    robot: BabyRobotHardware,
    duration_s: float,
    start_time: float,
    battery_csv: Path,
    samples,
):
    last_sample = robot.read_battery(start_time, samples)
    append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(last_sample))
    idle_start = time.monotonic()
    while time.monotonic() - idle_start < duration_s:
        sample = robot.read_battery(start_time, samples)
        last_sample = sample
        append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(sample))
        print(
            f"idle battery t={sample.elapsed_s:5.1f}s  "
            f"capacity={sample.percentage}%  voltage={sample.voltage_v:.3f}V  "
            f"status={sample.status}  drain={sample.drain_mv_per_min:.2f}mV/min",
        )
        time.sleep(1.0)
    return last_sample


def read_startup_battery(
    robot: BabyRobotHardware,
    start_time: float,
    battery_csv: Path,
    samples,
    args: argparse.Namespace,
):
    deadline = time.monotonic() + args.battery_init_timeout_s
    sample = robot.read_battery(start_time, samples)
    append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(sample))
    while sample.percentage < args.min_valid_battery_percent and time.monotonic() < deadline:
        print(
            f"Waiting for battery monitor: {sample.percentage}% "
            f"(< {args.min_valid_battery_percent:.1f}%)  "
            f"voltage={sample.voltage_v:.3f}V  status={sample.status}",
        )
        time.sleep(0.5)
        sample = robot.read_battery(start_time, samples)
        append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(sample))
    return sample


def main() -> None:
    args = parse_args()
    forward_model, forward_meta = resolve_model_and_meta("forward", args.forward_model, args.forward_meta)
    left_model, left_meta = resolve_model_and_meta("left", args.left_model, args.left_meta)
    right_model, right_meta = resolve_model_and_meta("right", args.right_model, args.right_meta)

    forward_network, forward_metadata, forward_format = load_gait_runtime(forward_model, forward_meta)
    left_network, left_metadata, left_format = load_gait_runtime(left_model, left_meta)
    right_network, right_metadata, right_format = load_gait_runtime(right_model, right_meta)

    run_dir = make_run_dir("run_behavior_tree_hardware", args.output_dir)
    behavior_csv = run_dir / "behavior_samples.csv"
    battery_csv = run_dir / "battery_samples.csv"
    video_path = run_dir / args.filename
    direct_mode = not args.delay_mode

    robot = BabyRobotHardware(
        direct_mode=direct_mode,
        delay_s=args.delay_s,
        enable_camera=False,
    )
    camera = open_camera(args.width, args.height)
    battery_samples = []
    battery_start = time.monotonic()
    initial_battery = read_startup_battery(
        robot,
        battery_start,
        battery_csv,
        battery_samples,
        args,
    )
    measured_percentage = initial_battery.percentage

    print(
        f"Initial battery: {initial_battery.percentage}%  "
        f"{initial_battery.voltage_v:.3f}V  {initial_battery.status}",
    )
    print(f"Moving to neutral for {args.neutral_hold_s:.1f}s.")
    robot.neutral()
    time.sleep(args.neutral_hold_s)
    if args.idle_measure_s > 0.0:
        idle_battery = measure_idle_battery(
            robot,
            args.idle_measure_s,
            battery_start,
            battery_csv,
            battery_samples,
        )
        measured_percentage = idle_battery.percentage

    if (
        args.battery_threshold_percent is None
        and measured_percentage >= args.min_valid_battery_percent
    ):
        battery_threshold_percent = min(100.0, measured_percentage + args.start_margin_percent)
        battery_threshold_source = "initial_plus_margin"
    elif args.battery_threshold_percent is None:
        battery_threshold_percent = 101.0
        battery_threshold_source = "battery_unready_forced_start"
    else:
        battery_threshold_percent = args.battery_threshold_percent
        battery_threshold_source = "cli"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        str(video_path),
        fourcc,
        args.video_fps,
        (args.width * 2, args.height),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    networks = {
        "forward": forward_network,
        "left": left_network,
        "right": right_network,
    }
    policy = HardwareVisionPolicy(networks, args, battery_threshold_percent)
    policy.reset()

    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/run_behavior_tree_hardware.py",
            "duration": args.duration,
            "control_hz": args.control_hz,
            "video_fps": args.video_fps,
            "width": args.width,
            "height": args.height,
            "camera_order": args.camera_order,
            "lower_hsv": list(args.lower_hsv),
            "upper_hsv": list(args.upper_hsv),
            "visibility_threshold": args.visibility_threshold,
            "reach_vision_area": args.reach_vision_area,
            "bearing_threshold": args.bearing_threshold,
            "lost_vision_hold_s": args.lost_vision_hold_s,
            "lost_vision_hold_frames": policy.lost_frame_limit,
            "patience": args.patience,
            "default_search_gait": args.default_search_gait,
            "invert_bearing": args.invert_bearing,
            "recenter_during_approach": args.recenter_during_approach,
            "forward_speed": args.forward_speed,
            "search_speed": args.search_speed,
            "amplitude_scale": args.amplitude_scale,
            "max_joint_deg": args.max_joint_deg,
            "battery_initial_v": initial_battery.voltage_v,
            "battery_initial_percentage": initial_battery.percentage,
            "battery_initial_status": initial_battery.status,
            "battery_threshold_percent": battery_threshold_percent,
            "battery_threshold_source": battery_threshold_source,
            "start_margin_percent": args.start_margin_percent,
            "min_valid_battery_percent": args.min_valid_battery_percent,
            "battery_init_timeout_s": args.battery_init_timeout_s,
            "idle_measure_s": args.idle_measure_s,
            "high_battery_behavior": args.high_battery_behavior,
            "stop_when_reached": args.stop_when_reached,
            "forward_model": str(forward_model),
            "forward_meta": str(forward_meta),
            "forward_weight_format": forward_format,
            "left_model": str(left_model),
            "left_meta": str(left_meta),
            "left_weight_format": left_format,
            "right_model": str(right_model),
            "right_meta": str(right_meta),
            "right_weight_format": right_format,
            "forward_num_joints": int(forward_metadata["num_joints"]),
            "left_num_joints": int(left_metadata["num_joints"]),
            "right_num_joints": int(right_metadata["num_joints"]),
            "direct_mode": direct_mode,
            "delay_s": args.delay_s,
            "neutral_hold_s": args.neutral_hold_s,
            "video_path": str(video_path),
            "behavior_csv": str(behavior_csv),
            "battery_csv": str(battery_csv),
            "mappings": mapping_metadata(DEFAULT_SERVO_MAPPINGS),
            "output_dir": str(run_dir),
        },
    )

    print(f"Results: {run_dir}")
    print(f"Video:   {video_path}")
    print(f"CSV:     {behavior_csv}")
    print(f"Battery: {battery_csv}")
    print(f"Battery threshold: {battery_threshold_percent:.1f}%")
    if measured_percentage > battery_threshold_percent:
        print("Starting in high-battery IDLE until the pack drops below threshold.")
    else:
        print("Starting low-battery homing behavior immediately.")
    if args.duration > 0.0:
        print(f"Duration: {args.duration:.1f}s")
    else:
        print("Duration: run until Ctrl-C")
    print(f"Forward speed: {args.forward_speed:.2f}  Search speed: {args.search_speed:.2f}")
    print(f"HSV lower={tuple(args.lower_hsv)} upper={tuple(args.upper_hsv)}")
    print(f"Reach vision area: {args.reach_vision_area:.3f}")
    print(
        f"Lost vision hold: {policy.lost_frame_limit} frames "
        f"({policy.lost_frame_limit / args.control_hz:.2f}s at {args.control_hz:.1f}Hz)",
    )

    control_period_s = 1.0 / args.control_hz
    video_period_s = 1.0 / args.video_fps
    next_video_s = 0.0
    next_print_s = 0.0
    frame_count = 0

    try:
        start = time.monotonic()
        previous_loop = start
        while args.duration <= 0.0 or time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            elapsed_s = loop_start - start
            loop_dt_s = loop_start - previous_loop
            previous_loop = loop_start

            camera_rgb = camera_frame_to_rgb(camera.capture_array(), args.camera_order)
            mask = isolate_hsv(camera_rgb, args.lower_hsv, args.upper_hsv)
            vision = analyze_sections(mask)
            orange_pixels = int(cv2.countNonZero(mask))
            battery = robot.read_battery(battery_start, battery_samples)
            append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(battery))
            action, mode, selected_gait = policy.decide(vision, battery.percentage)
            robot.set_joint_angles(action)

            frame_count += 1
            fps = frame_count / max(elapsed_s, 1e-6)
            row = make_behavior_row(
                elapsed_s=elapsed_s,
                loop_dt_s=loop_dt_s,
                mode=mode,
                selected_gait=selected_gait,
                battery_v=battery.voltage_v,
                battery_percentage=battery.percentage,
                battery_status=battery.status,
                threshold_percent=battery_threshold_percent,
                fps=fps,
                vision=vision,
                orange_pixels=orange_pixels,
                action=action,
                robot=robot,
                args=args,
            )
            append_csv_row(behavior_csv, BEHAVIOR_CSV_FIELDS, row)

            if elapsed_s >= next_video_s:
                video.write(compose_video_frame(camera_rgb, mask, row))
                next_video_s += video_period_s

            if elapsed_s >= next_print_s:
                print(
                    f"t={elapsed_s:6.2f}s  {mode:11s}  gait={selected_gait:7s}  "
                    f"battery={battery.percentage:3d}%  {battery.voltage_v:.3f}V  "
                    f"bearing={vision[5]:+6.3f}  area={vision[6]:7.4f}",
                )
                next_print_s += 1.0

            if mode == "STOPPED" and args.stop_when_reached:
                print("Target reached by vision area threshold; stopping.")
                break

            time.sleep(max(0.0, control_period_s - (time.monotonic() - loop_start)))
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
