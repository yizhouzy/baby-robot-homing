"""Record simulation behavior-tree demos for the real-world test conditions."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.demo_behavior_tree import (
    VisionSwitchBehaviorPolicy,
    resolve_model_and_meta,
)
from robot_control.behavior import BehaviorDemoConfig, LearnedGaitProvider, distance_to_station
from robot_control.config_gait import make_run_id
from robot_control.controllers import load_gait_network
from robot_control.rendering import (
    build_behavior_scene,
    draw_behavior_overlays,
    yaw_from_qpos,
)
from robot_control.artifacts import DEFAULT_GAIT_MODELS
from robot_control.vision import analyze_sections
from helper.plots.real_world_exp_data import DEFAULT_PLOT_DIR


OUTPUT_ROOT = Path("results/4_respective_sim_demos")
PLOT_TRACK_DIR = DEFAULT_PLOT_DIR / "simulation_trajectories"

CONDITIONS = {
    "mat_front": {
        "condition": "Mat front target",
        "station_pos": (0.0, -2.0, 0.1),
        "friction_scale": 1.0,
        "plot_filename": "mat_front_target.csv",
    },
    "mat_right": {
        "condition": "Mat 90 deg right target",
        "station_pos": (-2.0, 0.0, 0.1),
        "friction_scale": 1.0,
        "plot_filename": "mat_90_deg_right_target.csv",
    },
    "mat_back": {
        "condition": "Mat back target",
        "station_pos": (0.0, 2.0, 0.1),
        "friction_scale": 1.0,
        "plot_filename": "mat_back_target.csv",
    },
    "floor_front": {
        "condition": "Floor front target",
        "station_pos": (0.0, -2.0, 0.1),
        "friction_scale": 0.5,
        "plot_filename": "floor_front_target.csv",
    },
    "grass_front": {
        "condition": "Grass front target",
        "station_pos": (0.0, -2.0, 0.1),
        "friction_scale": 1.5,
        "plot_filename": "grass_front_target.csv",
    },
}

CSV_FIELDS = [
    "condition",
    "time_s",
    "x_m",
    "y_m",
    "yaw_rad",
    "target_x_m",
    "target_y_m",
    "distance_m",
    "mode",
    "selected_gait",
    "friction_scale",
    "vision_model",
    "vision_area",
    "bearing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--plot-track-dir", "--output-dir", dest="plot_track_dir", type=Path, default=PLOT_TRACK_DIR)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--vision-source", choices=["rendered-hsv", "geometric"], default="rendered-hsv")
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=(140, 45, 45))
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=(180, 255, 255))
    parser.add_argument("--camera-fov-deg", type=float, default=90.0)
    parser.add_argument("--target-area-scale", type=float, default=0.04)
    parser.add_argument("--visibility-threshold", type=float, default=0.0015)
    parser.add_argument("--reach-vision-area", type=float, default=0.14)
    parser.add_argument("--bearing-threshold", type=float, default=0.15)
    parser.add_argument("--lost-vision-hold-s", type=float, default=2.0)
    parser.add_argument("--default-search-gait", choices=["left", "right"], default="left")
    parser.add_argument("--invert-bearing", action="store_true")
    parser.add_argument("--recenter-during-approach", action="store_true", default=False)
    parser.add_argument("--forward-speed", type=float, default=0.6)
    parser.add_argument("--search-speed", type=float, default=0.25)
    parser.add_argument("--forward-model", type=str, default=str(DEFAULT_GAIT_MODELS["forward"][0]))
    parser.add_argument("--forward-meta", type=str, default=None)
    parser.add_argument("--left-model", type=str, default=str(DEFAULT_GAIT_MODELS["left"][0]))
    parser.add_argument("--left-meta", type=str, default=None)
    parser.add_argument("--right-model", type=str, default=str(DEFAULT_GAIT_MODELS["right"][0]))
    parser.add_argument("--right-meta", type=str, default=None)
    return parser.parse_args()


def apply_floor_friction(model, friction_scale: float) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name == "floor":
            model.geom_friction[geom_id] *= float(friction_scale)


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def geometric_vision(data, station_pos, args: argparse.Namespace) -> list[float]:
    dx = float(station_pos[0] - data.qpos[0])
    dy = float(station_pos[1] - data.qpos[1])
    distance = math.hypot(dx, dy)
    yaw = yaw_from_qpos(data)
    target_angle = math.atan2(-dx, -dy)
    relative_angle = wrap_pi(target_angle - yaw)
    half_fov = math.radians(args.camera_fov_deg) * 0.5
    visible = abs(relative_angle) <= half_fov
    bearing = float(np.clip(relative_angle / half_fov, -1.0, 1.0)) if visible else 0.0
    area = min(0.25, args.target_area_scale / max(distance * distance, 1e-6)) if visible else 0.0
    sections = [0.0] * 5
    if visible:
        section_index = int(np.clip(math.floor((bearing + 1.0) * 2.5), 0, 4))
        sections[section_index] = area
    return sections + [bearing, area]


def rendered_hsv_vision(renderer: mujoco.Renderer, data, cam_name: str | None, args: argparse.Namespace):
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    empty_mask = np.zeros((240, 320), dtype=np.uint8)
    if cam_name is None:
        return empty_frame, empty_mask, [0.0] * 7
    renderer.update_scene(data, camera=cam_name)
    camera_frame = renderer.render()
    hsv = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray(args.hsv_lower, dtype=np.uint8),
        np.asarray(args.hsv_upper, dtype=np.uint8),
    )
    return camera_frame, mask, analyze_sections(mask)


def load_policy(args: argparse.Namespace, model, control_dt: float) -> VisionSwitchBehaviorPolicy:
    forward_model, forward_meta = resolve_model_and_meta(args.forward_model, args.forward_meta)
    left_model, left_meta = resolve_model_and_meta(args.left_model, args.left_meta)
    right_model, right_meta = resolve_model_and_meta(args.right_model, args.right_meta)
    forward_net, _, _ = load_gait_network(forward_model, forward_meta)
    left_net, _, _ = load_gait_network(left_model, left_meta)
    right_net, _, _ = load_gait_network(right_model, right_meta)
    providers = {
        "forward": LearnedGaitProvider(forward_net),
        "left": LearnedGaitProvider(left_net),
        "right": LearnedGaitProvider(right_net),
    }
    dummy_config = BehaviorDemoConfig(
        duration=args.duration,
        battery_threshold=1.0,
        station_pos=(0.0, -2.0, 0.1),
        reach_radius=0.25,
        reach_vision_area=args.reach_vision_area,
        explore_speed=args.forward_speed,
        search_speed=args.search_speed,
        output_dir=args.output_root,
        timestamp=make_run_id(0),
    )
    lost_frame_limit = max(1, int(round(args.lost_vision_hold_s / control_dt)))
    return VisionSwitchBehaviorPolicy(
        providers,
        dummy_config,
        visibility_threshold=args.visibility_threshold,
        bearing_threshold=args.bearing_threshold,
        lost_frame_limit=lost_frame_limit,
        high_battery_behavior="idle",
        forward_speed=args.forward_speed,
        search_speed=args.search_speed,
        default_search_gait=args.default_search_gait,
        invert_bearing=args.invert_bearing,
        recenter_during_approach=args.recenter_during_approach,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_condition(slug: str, args: argparse.Namespace) -> None:
    spec = CONDITIONS[slug]
    run_dir = args.output_root / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    config = BehaviorDemoConfig(
        duration=args.duration,
        battery_threshold=1.0,
        station_pos=spec["station_pos"],
        reach_radius=0.25,
        reach_vision_area=args.reach_vision_area,
        explore_speed=args.forward_speed,
        search_speed=args.search_speed,
        output_dir=run_dir,
        timestamp=slug,
    )
    model, data, cam_name, mocap_id = build_behavior_scene(config)
    apply_floor_friction(model, float(spec["friction_scale"]))
    forward_meta_path = resolve_model_and_meta(args.forward_model, args.forward_meta)[1]
    forward_meta = np.load(forward_meta_path, allow_pickle=True)
    control_dt = float(forward_meta["dt"])
    control_freq = max(1, int(round(control_dt / model.opt.timestep)))
    policy = load_policy(args, model, control_dt)
    policy.config = config
    policy.reset()
    mujoco.mj_resetData(model, data)
    data.mocap_pos[mocap_id] = config.station_pos

    ctrl_renderer = mujoco.Renderer(model, height=240, width=320)
    vid_renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    video_path = run_dir / f"{slug}_behavior_tree_sim.mp4"
    video = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(video_path), fourcc, args.video_fps, (args.width, args.height))

    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (args.video_fps * dt))))
    battery = 1.0
    drain = dt / max(config.duration, 1e-6)
    current_action = np.zeros(model.nu)
    current_mode = "SEARCHING"
    current_gait = args.default_search_gait
    current_camera_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    current_mask = np.zeros((240, 320), dtype=np.uint8)
    current_vision = [0.0] * 7
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    viz = mujoco.MjvOption()
    viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    rows = []

    while data.time < config.duration:
        for _ in range(steps_per_frame):
            data.mocap_pos[mocap_id] = config.station_pos
            step = int(np.ceil(data.time / dt))
            if step % control_freq == 0:
                current_camera_frame, current_mask, rendered_vision = rendered_hsv_vision(
                    ctrl_renderer,
                    data,
                    cam_name,
                    args,
                )
                current_vision = (
                    rendered_vision
                    if args.vision_source == "rendered-hsv"
                    else geometric_vision(data, config.station_pos, args)
                )
                current_action, current_mode, current_gait = policy.decide_with_gait(
                    current_vision,
                    battery,
                    model,
                    data,
                )
            data.ctrl[:] = current_action
            mujoco.mj_step(model, data)
            battery = max(0.0, battery - drain)

        vid_renderer.update_scene(data, scene_option=viz, camera=camera_id)
        frame = vid_renderer.render().copy()
        draw_behavior_overlays(
            frame,
            battery,
            current_mode,
            current_vision,
            current_camera_frame,
            current_mask,
            data,
            config,
        )
        if video is not None:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        rows.append({
            "condition": spec["condition"],
            "time_s": f"{float(data.time):.6f}",
            "x_m": f"{float(data.qpos[0]):.6f}",
            "y_m": f"{float(data.qpos[1]):.6f}",
            "yaw_rad": f"{yaw_from_qpos(data):.6f}",
            "target_x_m": f"{float(config.station_pos[0]):.6f}",
            "target_y_m": f"{float(config.station_pos[1]):.6f}",
            "distance_m": f"{distance_to_station(data, config.station_pos):.6f}",
            "mode": current_mode,
            "selected_gait": current_gait,
            "friction_scale": f"{float(spec['friction_scale']):.6f}",
            "vision_model": args.vision_source,
            "vision_area": f"{float(current_vision[6]):.6f}",
            "bearing": f"{float(current_vision[5]):.6f}",
        })
        if current_mode == "STOPPED":
            break

    if video is not None:
        video.release()
    ctrl_renderer.close()
    vid_renderer.close()

    csv_path = run_dir / f"{slug}_trajectory.csv"
    write_csv(csv_path, rows)
    plot_csv_path = args.plot_track_dir / spec["plot_filename"]
    write_csv(plot_csv_path, rows)
    metadata = {
        "condition": spec["condition"],
        "station_pos": list(spec["station_pos"]),
        "friction_scale": spec["friction_scale"],
        "duration": args.duration,
        "vision_source": args.vision_source,
        "hsv_lower": list(args.hsv_lower),
        "hsv_upper": list(args.hsv_upper),
        "camera_fov_deg": args.camera_fov_deg,
        "target_area_scale": args.target_area_scale,
        "forward_speed": args.forward_speed,
        "search_speed": args.search_speed,
        "visibility_threshold": args.visibility_threshold,
        "reach_vision_area": args.reach_vision_area,
        "bearing_threshold": args.bearing_threshold,
        "lost_vision_hold_s": args.lost_vision_hold_s,
        "default_search_gait": args.default_search_gait,
        "invert_bearing": args.invert_bearing,
        "recenter_during_approach": args.recenter_during_approach,
        "trajectory_csv": str(csv_path),
        "plot_track_csv": str(plot_csv_path),
        "video": "" if args.no_video else str(video_path),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Saved {spec['condition']} -> {run_dir}")


def main() -> None:
    args = parse_args()
    slugs = CONDITIONS if args.condition == "all" else [args.condition]
    for slug in slugs:
        run_condition(slug, args)


if __name__ == "__main__":
    main()
