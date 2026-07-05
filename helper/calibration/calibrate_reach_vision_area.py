"""Calibrate reach_vision_area from the robot camera at a fixed target distance.

Examples:
    uv run --project ariel python helper/calibrate_reach_vision_area.py
    uv run --project ariel python helper/calibrate_reach_vision_area.py --distance 0.25
    uv run --project ariel python helper/calibrate_reach_vision_area.py --sweep
"""
# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import sys

import cv2
import mujoco
import numpy as np
from rich.console import Console
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ariel.simulation.environments import SimpleFlatWorld
from blocks.baby_robot import baby_robot

from robot_control.rendering import add_traffic_cone_target
from robot_control.vision import find_robot_camera, sample_target_vision


OUTPUT_DIR = Path("results/vision_calibration")


@dataclass(frozen=True)
class CalibrationSample:
    target_angle_deg: float
    target_yaw_deg: float
    target_pos: np.ndarray
    planar_distance: float
    orange_pixels: int
    total_pixels: int
    area: float
    bearing: float
    frame_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure orange target mask area at a fixed robot-target distance."
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=3.0,
        help="Planar robot-target distance in meters.",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=0.0,
        help="Target X position. Defaults to centered in front of the robot.",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=None,
        help="Target Y position. Defaults to -distance, the robot's forward direction.",
    )
    parser.add_argument("--target-z", type=float, default=0.1)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep target bearing angles and target base yaw orientations at the fixed distance.",
    )
    parser.add_argument(
        "--target-angles-deg",
        type=str,
        default="-45,-30,-15,0,15,30,45",
        help="Comma-separated target bearing angles around robot forward for --sweep.",
    )
    parser.add_argument(
        "--target-yaws-deg",
        type=str,
        default="0,45",
        help="Comma-separated target base yaw angles for --sweep; 45 means base diagonal faces robot.",
    )
    parser.add_argument(
        "--center-angle-threshold-deg",
        type=float,
        default=15.0,
        help="Angles within this absolute target bearing count as centered for recommended reach area.",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    return parser.parse_args()


def build_calibration_scene(target_pos):
    world = SimpleFlatWorld()
    add_traffic_cone_target(world, "reach_target", target_pos)
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mocap_id = model.body("reach_target").mocapid[0]
    cam_name = find_robot_camera(model)
    return model, data, mocap_id, cam_name


def save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def target_pos_from_angle(distance: float, angle_deg: float, z: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.asarray(
        [
            distance * np.sin(angle),
            -distance * np.cos(angle),
            z,
        ],
        dtype=np.float32,
    )


def yaw_quat(yaw_deg: float) -> np.ndarray:
    half = np.deg2rad(yaw_deg) / 2.0
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def measure_sample(
    model,
    data,
    mocap_id: int,
    cam_name: str | None,
    renderer: mujoco.Renderer,
    target_pos: np.ndarray,
    target_angle_deg: float,
    target_yaw_deg: float,
    output_dir: Path,
    timestamp: str,
) -> CalibrationSample:
    data.mocap_pos[mocap_id] = target_pos
    data.mocap_quat[mocap_id] = yaw_quat(target_yaw_deg)
    mujoco.mj_forward(model, data)

    camera_frame, mask, vision = sample_target_vision(renderer, data, cam_name)
    orange_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    area = float(vision[6])
    bearing = float(vision[5])
    planar_distance = float(np.linalg.norm(data.qpos[:2] - target_pos[:2]))

    label = f"angle_{target_angle_deg:+06.1f}_yaw_{target_yaw_deg:+06.1f}".replace(".", "p")
    frame_path = output_dir / f"reach_camera_{timestamp}_{label}.png"
    mask_path = output_dir / f"reach_mask_{timestamp}_{label}.png"
    save_frame(frame_path, camera_frame)
    save_mask(mask_path, mask)
    return CalibrationSample(
        target_angle_deg=target_angle_deg,
        target_yaw_deg=target_yaw_deg,
        target_pos=target_pos.copy(),
        planar_distance=planar_distance,
        orange_pixels=orange_pixels,
        total_pixels=total_pixels,
        area=area,
        bearing=bearing,
        frame_path=frame_path,
        mask_path=mask_path,
    )


def save_samples(
    samples: list[CalibrationSample],
    output_dir: Path,
    timestamp: str,
    cam_name: str | None,
    center_angle_threshold_deg: float,
) -> Path:
    data_path = output_dir / f"reach_vision_area_{timestamp}.npz"
    centered_samples = [
        sample for sample in samples
        if abs(sample.target_angle_deg) <= center_angle_threshold_deg
    ]
    np.savez(
        str(data_path),
        camera_name="" if cam_name is None else cam_name,
        target_angle_deg=np.asarray([sample.target_angle_deg for sample in samples], dtype=np.float32),
        target_yaw_deg=np.asarray([sample.target_yaw_deg for sample in samples], dtype=np.float32),
        target_pos=np.asarray([sample.target_pos for sample in samples], dtype=np.float32),
        planar_distance=np.asarray([sample.planar_distance for sample in samples], dtype=np.float32),
        orange_pixels=np.asarray([sample.orange_pixels for sample in samples], dtype=np.int32),
        total_pixels=np.asarray([sample.total_pixels for sample in samples], dtype=np.int32),
        reach_vision_area=np.asarray([sample.area for sample in samples], dtype=np.float32),
        bearing=np.asarray([sample.bearing for sample in samples], dtype=np.float32),
        frame_path=np.asarray([str(sample.frame_path) for sample in samples]),
        mask_path=np.asarray([str(sample.mask_path) for sample in samples]),
        center_angle_threshold_deg=float(center_angle_threshold_deg),
        recommended_centered_reach_vision_area=min(sample.area for sample in centered_samples),
        all_angle_min_reach_vision_area=min(sample.area for sample in samples),
    )
    return data_path


def main() -> None:
    install()
    console = Console()
    args = parse_args()

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if args.sweep:
        target_angles = parse_float_list(args.target_angles_deg)
        target_yaws = parse_float_list(args.target_yaws_deg)
        initial_pos = target_pos_from_angle(args.distance, target_angles[0], args.target_z)
    else:
        target_y = -float(args.distance) if args.target_y is None else float(args.target_y)
        initial_pos = np.asarray([args.target_x, target_y, args.target_z], dtype=np.float32)
        target_angles = [float(np.rad2deg(np.arctan2(initial_pos[0], -initial_pos[1])))]
        target_yaws = [0.0]

    model, data, mocap_id, cam_name = build_calibration_scene(initial_pos)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    samples = []
    for target_angle in target_angles:
        target_pos = (
            target_pos_from_angle(args.distance, target_angle, args.target_z)
            if args.sweep
            else initial_pos
        )
        for target_yaw in target_yaws:
            samples.append(
                measure_sample(
                    model,
                    data,
                    mocap_id,
                    cam_name,
                    renderer,
                    target_pos,
                    target_angle,
                    target_yaw,
                    output_dir,
                    timestamp,
                )
            )
    renderer.close()

    data_path = save_samples(
        samples,
        output_dir,
        timestamp,
        cam_name,
        center_angle_threshold_deg=args.center_angle_threshold_deg,
    )
    areas = np.asarray([sample.area for sample in samples], dtype=np.float32)
    min_index = int(np.argmin(areas))
    centered_indices = [
        index for index, sample in enumerate(samples)
        if abs(sample.target_angle_deg) <= args.center_angle_threshold_deg
    ]
    centered_areas = areas[centered_indices]
    centered_min_local = int(np.argmin(centered_areas))
    centered_min_index = centered_indices[centered_min_local]
    median_area = float(np.median(areas))
    max_area = float(np.max(areas))

    console.log(f"Robot camera: {cam_name}")
    console.log(f"Distance: {args.distance:.4f} m")
    console.log(f"Samples: {len(samples)}")
    for sample in samples:
        console.log(
            f"angle={sample.target_angle_deg:+5.1f} deg "
            f"yaw={sample.target_yaw_deg:+5.1f} deg "
            f"area={sample.area:.6f} "
            f"pixels={sample.orange_pixels}/{sample.total_pixels} "
            f"bearing={sample.bearing:+.4f}"
        )
    console.log(
        f"Recommended centered reach_vision_area: {areas[centered_min_index]:.6f} "
        f"(minimum for |angle| <= {args.center_angle_threshold_deg:.1f} deg; "
        f"angle={samples[centered_min_index].target_angle_deg:+.1f}, "
        f"yaw={samples[centered_min_index].target_yaw_deg:+.1f})"
    )
    console.log(
        f"All-angle minimum: {areas[min_index]:.6f} "
        f"(peripheral view; angle={samples[min_index].target_angle_deg:+.1f}, "
        f"yaw={samples[min_index].target_yaw_deg:+.1f})"
    )
    console.log(f"Median area: {median_area:.6f}  Max area: {max_area:.6f}")
    console.log(f"Data -> {data_path}")


if __name__ == "__main__":
    main()
