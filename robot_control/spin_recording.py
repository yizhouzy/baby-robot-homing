"""Video recording utilities for Phase 2 spin CPG replays."""
from __future__ import annotations

import os
from pathlib import Path
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from rich.console import Console

from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import VideoRecorder
from blocks.baby_robot import baby_robot
from robot_control.controllers import CPGGaitController, sanitize_action
from robot_control.rendering import yaw_from_qpos


def infer_spin_run_id(model_path: Path) -> str:
    stem = model_path.stem
    if stem.startswith("spin_best_"):
        return stem.removeprefix("spin_best_")
    if stem.startswith("spin_ckpt_"):
        return stem.removeprefix("spin_ckpt_").rsplit("_gen", maxsplit=1)[0]
    return model_path.parent.name


def build_spin_scene(camera_height: float):
    world = SimpleFlatWorld()
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, 0, camera_height],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def core_heading(data) -> float:
    xmat = data.geom("robot1_core").xmat.reshape(3, 3)
    forward_2d = (xmat @ np.array([0.0, -1.0, 0.0]))[:2]
    return float(np.arctan2(forward_2d[1], forward_2d[0]))


def orientation_reward(heading_history, xy_history) -> float:
    headings = np.unwrap(heading_history)
    total_rotation = abs(headings[-1] - headings[0])
    displacement = np.linalg.norm(
        np.array(xy_history[-1]) - np.array(xy_history[0])
    )
    return float(total_rotation / (1.0 + displacement))


def draw_spin_overlay(
    frame,
    data,
    heading_history: list[float],
    xy_history: list[tuple[float, float]],
) -> None:
    reward = orientation_reward(heading_history, xy_history)
    drift = float(np.linalg.norm(np.asarray(xy_history[-1]) - np.asarray(xy_history[0])))
    yaw = yaw_from_qpos(data)
    lines = [
        f"t: {data.time:5.2f}s",
        f"yaw: {np.rad2deg(yaw):7.2f} deg",
        f"reward: {reward:8.2f}",
        f"drift: {drift:6.3f} m",
        f"xy: ({data.qpos[0]:+.3f}, {data.qpos[1]:+.3f})",
    ]
    for idx, text in enumerate(lines):
        y = 28 + idx * 26
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)


def record_spin_video(
    network: CPGGaitController,
    model_path: Path,
    duration: float,
    output_dir: Path,
    camera_height: float,
    width: int,
    height: int,
    fps: int,
    overlay: bool,
    console: Console,
) -> Path | None:
    model, data = build_spin_scene(camera_height)
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        console.log(
            "Video renderer unavailable. On headless Linux, install/enable EGL "
            "or run with MUJOCO_GL=osmesa if OSMesa is available. "
            f"Renderer error: {exc}",
            style="yellow",
        )
        return None
    run_id = infer_spin_run_id(model_path)
    video_name = f"spin_demo_{run_id}"
    before_recording = set(output_dir.glob(f"{video_name}*"))
    video_recorder = VideoRecorder(
        file_name=video_name,
        output_folder=str(output_dir),
        width=width,
        height=height,
        fps=fps,
    )
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    viz = mujoco.MjvOption()
    viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    control_freq = max(1, int(round(float(network.cpg.dt) / model.opt.timestep)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    current_ctrl = np.zeros(model.nu, dtype=np.float32)
    xy_history = [(float(data.qpos[0]), float(data.qpos[1]))]
    heading_history = [core_heading(data)]
    step = 0
    network.reset_hidden()

    try:
        while data.time < duration:
            for _ in range(steps_per_frame):
                if step % control_freq == 0:
                    current_ctrl = sanitize_action(network.forward(0.0, 1.0), model)
                data.ctrl[:] = current_ctrl
                mujoco.mj_step(model, data)
                xy_history.append((float(data.qpos[0]), float(data.qpos[1])))
                heading_history.append(core_heading(data))
                step += 1

            renderer.update_scene(data, scene_option=viz, camera=camera_id)
            frame = renderer.render().copy()
            if overlay:
                draw_spin_overlay(frame, data, heading_history, xy_history)
            video_recorder.write(frame=frame)
    finally:
        video_recorder.release()
        renderer.close()

    candidates = sorted(
        set(output_dir.glob(f"{video_name}*")) - before_recording,
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        console.log("Video recorder finished but no output file was created.", style="yellow")
        return None
    video_path = candidates[-1]
    console.log(f"Video saved -> {video_path}", style="green")
    console.log(f"Final reward: {orientation_reward(heading_history, xy_history):.4f}")
    return video_path
