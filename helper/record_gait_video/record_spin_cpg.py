"""Record a Phase 2 spin CPG replay from a saved checkpoint.

Examples:
    uv run --project ariel python helper/record_spin_cpg.py
    uv run --project ariel python helper/record_spin_cpg.py --dur 10
    uv run --project ariel python helper/record_spin_cpg.py \
        --model results/turn_cpg/<run_id>/spin_best_<run_id>.npy \
        --meta results/turn_cpg/<run_id>/spin_meta_<run_id>.npz
"""
# ruff: noqa: E402
from __future__ import annotations

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
from ariel.utils.renderers import VideoRecorder
from blocks.baby_robot import baby_robot
from robot_control.controllers import CPGGaitController, load_gait_network, sanitize_action
from robot_control.rendering import yaw_from_qpos


RESULTS_DIR = Path("results/turn_cpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a Phase 2 spin CPG replay.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--dur", type=float, default=10.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--camera-height", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def find_latest_spin_model(results_dir: Path = RESULTS_DIR) -> Path:
    run_dirs = sorted(
        [path for path in results_dir.glob("*") if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    for run_dir in reversed(run_dirs):
        run_id = run_dir.name
        preferred = run_dir / f"spin_best_{run_id}.npy"
        if preferred.exists():
            return preferred
        candidates = sorted(run_dir.glob("spin_best_*.npy"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"No spin_best checkpoint found under {results_dir}.")


def infer_run_id(model_path: Path) -> str:
    stem = model_path.stem
    if stem.startswith("spin_best_"):
        return stem.removeprefix("spin_best_")
    if stem.startswith("spin_ckpt_"):
        return stem.removeprefix("spin_ckpt_").rsplit("_gen", maxsplit=1)[0]
    return model_path.parent.name


def infer_run_dir(model_path: Path) -> Path:
    if model_path.parent.name == "checkpoints":
        return model_path.parent.parent
    return model_path.parent


def infer_meta_path(model_path: Path, requested: str | None) -> Path | None:
    if requested is not None:
        return Path(requested)

    run_dir = infer_run_dir(model_path)
    run_id = infer_run_id(model_path)
    preferred = run_dir / f"spin_meta_{run_id}.npz"
    if preferred.exists():
        return preferred

    candidates = sorted(run_dir.glob("spin_meta_*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    return None


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


def draw_overlay(frame, data, heading_history: list[float], xy_history: list[tuple[float, float]]) -> None:
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
) -> Path:
    model, data = build_spin_scene(camera_height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    run_id = infer_run_id(model_path)
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
            draw_overlay(frame, data, heading_history, xy_history)
        video_recorder.write(frame=frame)

    video_recorder.release()
    renderer.close()
    candidates = sorted(
        set(output_dir.glob(f"{video_name}*")) - before_recording,
        key=lambda path: path.stat().st_mtime,
    )
    video_path = candidates[-1]
    console.log(f"Video saved -> {video_path}", style="green")
    console.log(f"Final reward: {orientation_reward(heading_history, xy_history):.4f}")
    return video_path


def main() -> None:
    install()
    console = Console()
    args = parse_args()

    model_path = Path(args.model) if args.model is not None else find_latest_spin_model()
    meta_path = infer_meta_path(model_path, args.meta)
    output_dir = Path(args.output_dir) if args.output_dir is not None else infer_run_dir(model_path) / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Model: {model_path}")
    console.log(f"Metadata: {meta_path if meta_path is not None else 'not found; inferred'}")
    console.log(f"Output dir: {output_dir}")

    network, meta, weight_format = load_gait_network(model_path, meta_path)
    console.log(
        f"Loaded {weight_format}: num_joints={int(meta['num_joints'])}, "
        f"dt={float(meta['dt']):.4f}"
    )
    record_spin_video(
        network=network,
        model_path=model_path,
        duration=args.dur,
        output_dir=output_dir,
        camera_height=args.camera_height,
        width=args.width,
        height=args.height,
        fps=args.fps,
        overlay=not args.no_overlay,
        console=console,
    )


if __name__ == "__main__":
    main()
