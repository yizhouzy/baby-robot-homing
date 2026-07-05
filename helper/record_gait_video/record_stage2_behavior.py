"""Record low-battery Stage 2 homing/search behavior for one or more targets.

Examples:
    uv run --project ariel python helper/record_stage2_behavior.py --duration-per-target 2
    uv run --project ariel python helper/record_stage2_behavior.py --targets "0,-2,0.1;1,-2,0.1"
"""
# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from rich.console import Console
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import VideoRecorder
from blocks.baby_robot import baby_robot

from robot_control.artifacts import find_latest_best_model, infer_meta_path, resolve_model_path
from robot_control.behavior import BATTERY_THRESHOLD, BehaviorDemoConfig
from robot_control.controllers import load_gait_network
from robot_control.rendering import add_traffic_cone_target, draw_behavior_overlays, yaw_from_qpos
from robot_control.turning import (
    DEFAULT_BEARING_GAIN,
    DEFAULT_LOW_PASS_ALPHA,
    DEFAULT_REACH_VISION_AREA,
    DEFAULT_SEARCH_TURN,
    DEFAULT_VISIBILITY_THRESHOLD,
    HandCodedTurnProvider,
    LearnedTurnProvider,
    load_turn_checkpoint,
    parse_turn_gains,
)
from robot_control.vision import find_robot_camera, sample_target_vision


@dataclass(frozen=True)
class Stage2Sample:
    target_index: int
    time: float
    x: float
    y: float
    yaw: float
    battery: float
    mode: str
    distance: float
    bearing: float
    visible: float
    size: float
    turn: float
    speed: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Stage 2 low-battery behavior.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--controller", choices=["hand-coded", "learned"], default="hand-coded")
    parser.add_argument("--turn-model", type=str, default=None)
    parser.add_argument("--turn-meta", type=str, default=None)
    parser.add_argument("--targets", type=str, default="0,-2,0.1;1,-2,0.1;-1,-2,0.1")
    parser.add_argument("--duration-per-target", type=float, default=60.0)
    parser.add_argument("--gains", type=str, default=None)
    parser.add_argument("--bearing-k", type=float, default=DEFAULT_BEARING_GAIN)
    parser.add_argument("--search-turn", type=float, default=DEFAULT_SEARCH_TURN)
    parser.add_argument("--low-pass-alpha", type=float, default=DEFAULT_LOW_PASS_ALPHA)
    parser.add_argument("--visibility-threshold", type=float, default=DEFAULT_VISIBILITY_THRESHOLD)
    parser.add_argument("--reach-vision-area", type=float, default=DEFAULT_REACH_VISION_AREA)
    parser.add_argument("--reach-radius", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default="results/stage2_behavior")
    return parser.parse_args()


def parse_targets(value: str) -> list[np.ndarray]:
    targets = []
    for chunk in value.split(";"):
        coords = [float(part.strip()) for part in chunk.split(",")]
        if len(coords) != 3:
            raise ValueError("--targets entries must be x,y,z triples.")
        targets.append(np.asarray(coords, dtype=np.float32))
    return targets


def resolve_gait(args: argparse.Namespace):
    model_path = resolve_model_path(args.model) if args.model is not None else find_latest_best_model()
    try:
        meta_path = infer_meta_path(model_path, args.meta)
    except FileNotFoundError:
        meta_path = None
    return model_path, meta_path


def build_stage2_scene(target_pos):
    world = SimpleFlatWorld()
    add_traffic_cone_target(world, "charging_station", target_pos)
    planar_distance = float(np.linalg.norm(np.asarray(target_pos[:2], dtype=np.float32)))
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[target_pos[0] / 2.0, target_pos[1] / 2.0, max(5.0, 1.8 * planar_distance)],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data, model.body("charging_station").mocapid[0], find_robot_camera(model)


def make_provider(args, gait_net, turn_checkpoint):
    if args.controller == "learned":
        turn_net, gains, _ = turn_checkpoint
        return LearnedTurnProvider(
            gait_net,
            turn_net,
            gains,
            low_pass_alpha=args.low_pass_alpha,
            visibility_threshold=args.visibility_threshold,
            reach_vision_area=args.reach_vision_area,
        )
    return HandCodedTurnProvider(
        gait_net,
        gains=parse_turn_gains(args.gains),
        bearing_gain=args.bearing_k,
        search_turn=args.search_turn,
        low_pass_alpha=args.low_pass_alpha,
        visibility_threshold=args.visibility_threshold,
        reach_vision_area=args.reach_vision_area,
    )


def mode_from_command(command, distance: float, reach_radius: float) -> str:
    if distance <= reach_radius:
        return "STOPPED"
    if command.features.visible:
        return "APPROACHING"
    return "SEARCHING"


def record_target_episode(
    args,
    console: Console,
    gait_net,
    turn_checkpoint,
    target_index: int,
    target_pos: np.ndarray,
    output_dir: Path,
    timestamp: str,
) -> list[Stage2Sample]:
    model, data, mocap_id, cam_name = build_stage2_scene(target_pos)
    provider = make_provider(args, gait_net, turn_checkpoint)
    provider.reset()
    data.mocap_pos[mocap_id] = target_pos
    control_freq = max(1, int(round(float(gait_net.cpg.dt) / model.opt.timestep)))

    ctrl_renderer = mujoco.Renderer(model, height=24, width=32)
    vid_renderer = mujoco.Renderer(model, height=480, width=640)
    video_recorder = VideoRecorder(
        file_name=f"stage2_behavior_target{target_index}_{timestamp}",
        output_folder=str(output_dir / "videos"),
    )

    config = BehaviorDemoConfig(
        duration=args.duration_per_target,
        battery_threshold=BATTERY_THRESHOLD,
        station_pos=tuple(float(value) for value in target_pos),
        reach_radius=args.reach_radius,
        reach_vision_area=args.reach_vision_area,
        explore_speed=0.0,
        search_speed=0.0,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    viz = mujoco.MjvOption()
    viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (30 * dt))))
    battery = BATTERY_THRESHOLD - 0.01
    drain = battery / max(args.duration_per_target, dt)
    current_action = np.zeros(model.nu, dtype=np.float32)
    current_mode = "SEARCHING"
    current_camera_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    current_mask = np.zeros((24, 32), dtype=np.uint8)
    current_vision = [0.0] * 7
    current_command = None
    samples = []

    console.log(f"Recording target {target_index}: {target_pos.tolist()} camera={cam_name}")
    while data.time < args.duration_per_target:
        for _ in range(steps_per_frame):
            data.mocap_pos[mocap_id] = target_pos
            step = int(np.ceil(data.time / dt))
            if step % control_freq == 0:
                current_camera_frame, current_mask, current_vision = sample_target_vision(
                    ctrl_renderer,
                    data,
                    cam_name,
                )
                distance = float(np.linalg.norm(data.qpos[:2] - target_pos[:2]))
                current_command = provider.compute(model, current_vision)
                current_mode = mode_from_command(current_command, distance, args.reach_radius)
                current_action = (
                    np.zeros(model.nu, dtype=np.float32)
                    if current_mode == "STOPPED"
                    else current_command.action
                )

            data.ctrl[:] = current_action
            mujoco.mj_step(model, data)
            battery = max(0.0, battery - drain * dt)

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
        video_recorder.write(frame=frame)
        distance = float(np.linalg.norm(data.qpos[:2] - target_pos[:2]))
        if current_command is None:
            command = provider.compute(model, current_vision)
        else:
            command = current_command
        samples.append(Stage2Sample(
            target_index=target_index,
            time=float(data.time),
            x=float(data.qpos[0]),
            y=float(data.qpos[1]),
            yaw=yaw_from_qpos(data),
            battery=float(battery),
            mode=current_mode,
            distance=distance,
            bearing=command.features.bearing,
            visible=command.features.visible,
            size=command.features.size,
            turn=command.turn,
            speed=command.speed,
        ))

    video_recorder.release()
    ctrl_renderer.close()
    vid_renderer.close()
    return samples


def save_samples(samples: list[Stage2Sample], targets: list[np.ndarray], output_dir: Path, timestamp: str) -> Path:
    sample_path = output_dir / f"stage2_behavior_samples_{timestamp}.npz"
    np.savez(
        str(sample_path),
        target_positions=np.asarray(targets, dtype=np.float32),
        target_index=np.asarray([sample.target_index for sample in samples], dtype=np.int32),
        time=np.asarray([sample.time for sample in samples], dtype=np.float32),
        x=np.asarray([sample.x for sample in samples], dtype=np.float32),
        y=np.asarray([sample.y for sample in samples], dtype=np.float32),
        yaw=np.asarray([sample.yaw for sample in samples], dtype=np.float32),
        battery=np.asarray([sample.battery for sample in samples], dtype=np.float32),
        mode=np.asarray([sample.mode for sample in samples]),
        distance=np.asarray([sample.distance for sample in samples], dtype=np.float32),
        bearing=np.asarray([sample.bearing for sample in samples], dtype=np.float32),
        visible=np.asarray([sample.visible for sample in samples], dtype=np.float32),
        size=np.asarray([sample.size for sample in samples], dtype=np.float32),
        turn=np.asarray([sample.turn for sample in samples], dtype=np.float32),
        speed=np.asarray([sample.speed for sample in samples], dtype=np.float32),
    )
    return sample_path


def plot_samples(samples: list[Stage2Sample], targets: list[np.ndarray], output_dir: Path, timestamp: str) -> Path:
    fig, (ax_path, ax_dist) = plt.subplots(1, 2, figsize=(16, 7))
    for target_index, target in enumerate(targets):
        target_samples = [sample for sample in samples if sample.target_index == target_index]
        xs = [sample.x for sample in target_samples]
        ys = [sample.y for sample in target_samples]
        times = [sample.time for sample in target_samples]
        distances = [sample.distance for sample in target_samples]
        ax_path.plot(xs, ys, lw=2, label=f"target {target_index}")
        if xs:
            ax_path.plot(xs[0], ys[0], "go", markersize=8)
        ax_path.plot(target[0], target[1], "r*", markersize=14)
        ax_dist.plot(times, distances, lw=2, label=f"target {target_index}")
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("X (m)")
    ax_path.set_ylabel("Y (m)")
    ax_path.set_title("Stage 2 Behavior Trajectories")
    ax_path.legend()
    ax_path.grid(True, alpha=0.3)
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Distance (m)")
    ax_dist.set_title("Distance Over Time")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / f"stage2_behavior_summary_{timestamp}.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    install()
    console = Console()
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    gait_model_path, gait_meta_path = resolve_gait(args)
    gait_net, _, gait_format = load_gait_network(gait_model_path, gait_meta_path)
    turn_checkpoint = None
    if args.controller == "learned":
        turn_checkpoint = load_turn_checkpoint(Path(args.turn_model), Path(args.turn_meta) if args.turn_meta else None)

    console.log(f"Loaded gait {gait_format}: {gait_model_path}")
    targets = parse_targets(args.targets)
    all_samples = []
    for target_index, target_pos in enumerate(targets):
        all_samples.extend(record_target_episode(
            args,
            console,
            gait_net,
            turn_checkpoint,
            target_index,
            target_pos,
            output_dir,
            timestamp,
        ))

    sample_path = save_samples(all_samples, targets, output_dir, timestamp)
    plot_path = plot_samples(all_samples, targets, output_dir, timestamp)
    console.log(f"Saved samples -> {sample_path}")
    console.log(f"Saved plot -> {plot_path}")
    console.log(f"Videos -> {output_dir / 'videos'}")


if __name__ == "__main__":
    main()
