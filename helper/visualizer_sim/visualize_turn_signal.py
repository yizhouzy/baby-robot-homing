"""Interactively test the Stage 2 hand-coded turn signal.

Examples:
    uv run --project ariel python helper/visualize_turn_signal.py
    uv run --project ariel python helper/visualize_turn_signal.py --headless --duration 2
    uv run --project ariel python helper/visualize_turn_signal.py --gains=-0.5,-0.5,-0.5,0.8,0,0,-0.6,0.6
"""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.tracker import Tracker
from blocks.baby_robot import baby_robot

from robot_control.artifacts import find_latest_best_model, infer_meta_path, resolve_model_path
from robot_control.controllers import load_gait_network
from robot_control.rendering import add_traffic_cone_target
from robot_control.turning import (
    DEFAULT_BEARING_GAIN,
    DEFAULT_LOW_PASS_ALPHA,
    DEFAULT_REACH_VISION_AREA,
    DEFAULT_SEARCH_TURN,
    DEFAULT_VISIBILITY_THRESHOLD,
    HandCodedTurnProvider,
    parse_turn_gains,
)
from robot_control.vision import find_robot_camera, sample_target_vision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage 2 hand-coded turning.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--target-x", type=float, default=2.0)
    parser.add_argument("--target-y", type=float, default=-2.0)
    parser.add_argument("--target-z", type=float, default=0.1)
    parser.add_argument("--gains", type=str, default=None)
    parser.add_argument("--bearing-k", type=float, default=DEFAULT_BEARING_GAIN)
    parser.add_argument("--search-turn", type=float, default=DEFAULT_SEARCH_TURN)
    parser.add_argument("--low-pass-alpha", type=float, default=DEFAULT_LOW_PASS_ALPHA)
    parser.add_argument("--visibility-threshold", type=float, default=DEFAULT_VISIBILITY_THRESHOLD)
    parser.add_argument("--reach-vision-area", type=float, default=DEFAULT_REACH_VISION_AREA)
    parser.add_argument("--log-interval", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def build_turn_scene(target_pos):
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    add_traffic_cone_target(world, "turn_target", target_pos)
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[3.2, -0.5, 1.4],
        xyaxes=[0.1544, 0.988, 0.0, -0.3557, 0.0556, 0.9329],
    )
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data, model.body("turn_target").mocapid[0], find_robot_camera(model), world.spec


def resolve_gait(args: argparse.Namespace):
    model_path = resolve_model_path(args.model) if args.model is not None else find_latest_best_model()
    try:
        meta_path = infer_meta_path(model_path, args.meta)
    except FileNotFoundError:
        meta_path = None
    return model_path, meta_path


def projected_target_vision(model, data, cam_name: str | None, target_pos, reach_vision_area: float):
    if cam_name is None:
        return [0.0] * 7
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_xmat = data.cam_xmat[cam_id].reshape(3, 3)
    local = cam_xmat.T @ (np.asarray(target_pos, dtype=np.float32) - cam_pos)
    depth = -float(local[2])
    if depth <= 0.0:
        return [0.0] * 7

    vertical_tan = np.tan(np.deg2rad(float(model.cam_fovy[cam_id])) / 2.0)
    horizontal_tan = vertical_tan * (32.0 / 24.0)
    bearing = float(np.clip(local[0] / (depth * horizontal_tan), -1.0, 1.0))
    vertical = float(local[1] / (depth * vertical_tan))
    visible = abs(bearing) <= 1.0 and abs(vertical) <= 1.0
    raw_area = float(np.clip(0.04 / (depth * depth), 0.0, reach_vision_area)) if visible else 0.0
    return [0.0, 0.0, 0.0, 0.0, 0.0, bearing if visible else 0.0, raw_area]


def run_visualizer(args: argparse.Namespace, console: Console) -> None:
    target_pos = np.asarray([args.target_x, args.target_y, args.target_z], dtype=np.float32)
    model_path, meta_path = resolve_gait(args)
    model, data, mocap_id, cam_name, spec = build_turn_scene(target_pos)
    gait_net, meta, weight_format = load_gait_network(model_path, meta_path)
    control_freq = max(1, int(round(float(meta["dt"]) / model.opt.timestep)))
    provider = HandCodedTurnProvider(
        gait_net,
        gains=parse_turn_gains(args.gains),
        bearing_gain=args.bearing_k,
        search_turn=args.search_turn,
        low_pass_alpha=args.low_pass_alpha,
        visibility_threshold=args.visibility_threshold,
        reach_vision_area=args.reach_vision_area,
    )
    provider.reset()

    console.log(f"Loaded {weight_format}: {model_path}")
    console.log(f"Robot camera: {cam_name}")
    console.log(f"Control update every {control_freq} physics steps")

    current_action = np.zeros(model.nu, dtype=np.float32)
    step = 0
    last_log_time = -args.log_interval
    data.mocap_pos[mocap_id] = target_pos

    def step_once(renderer: mujoco.Renderer) -> None:
        nonlocal current_action, step, last_log_time
        data.mocap_pos[mocap_id] = target_pos
        if step % control_freq == 0:
            _, _, vision = sample_target_vision(renderer, data, cam_name)
            command = provider.compute(model, vision)
            current_action = command.action
            if data.time - last_log_time >= args.log_interval:
                distance = float(np.linalg.norm(data.qpos[:2] - target_pos[:2]))
                console.log(
                    f"t={data.time:6.2f}s "
                    f"bearing={command.features.bearing:+.2f} "
                    f"visible={command.features.visible:.0f} "
                    f"size={command.features.size:.2f} "
                    f"turn={command.turn:+.2f} "
                    f"speed={command.speed:.2f} "
                    f"dist={distance:.2f}m"
                )
                last_log_time = float(data.time)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

    def turn_callback(callback_model: mujoco.MjModel, callback_data: mujoco.MjData):
        nonlocal last_log_time
        callback_data.mocap_pos[mocap_id] = target_pos
        vision = projected_target_vision(
            callback_model,
            callback_data,
            cam_name,
            target_pos,
            args.reach_vision_area,
        )
        command = provider.compute(callback_model, vision)
        if callback_data.time - last_log_time >= args.log_interval:
            distance = float(np.linalg.norm(callback_data.qpos[:2] - target_pos[:2]))
            console.log(
                f"t={callback_data.time:6.2f}s "
                f"bearing={command.features.bearing:+.2f} "
                f"visible={command.features.visible:.0f} "
                f"size={command.features.size:.2f} "
                f"turn={command.turn:+.2f} "
                f"speed={command.speed:.2f} "
                f"dist={distance:.2f}m"
            )
            last_log_time = float(callback_data.time)
        return command.action

    if args.headless:
        renderer = mujoco.Renderer(model, height=24, width=32)
        while data.time < args.duration:
            step_once(renderer)
        renderer.close()
    else:
        controller = Controller(
            controller_callback_function=turn_callback,
            time_steps_per_ctrl_step=control_freq,
            time_steps_per_save=500,
            alpha=1.0,
            tracker=build_tracker(spec, data),
        )
        try:
            mujoco.set_mjcb_control(controller.set_control)
            viewer.launch(model=model, data=data)
        finally:
            mujoco.set_mjcb_control(None)
    console.log("Visualizer run finished.")


def build_tracker(spec, data) -> Tracker:
    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(spec, data)
    return tracker


def main() -> None:
    install()
    run_visualizer(parse_args(), Console())


if __name__ == "__main__":
    main()
