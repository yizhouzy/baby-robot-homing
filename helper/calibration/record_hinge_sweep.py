"""Record a sequential hinge sweep video for the Baby robot.

Examples:
    uv run --project ariel python helper/record_hinge_sweep.py
    uv run --project ariel python helper/record_hinge_sweep.py --sweep-time 2.0
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

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


RESULTS_DIR = Path("results/hinge_sweep")


@dataclass(frozen=True)
class SweepSegment:
    name: str
    duration: float
    start: float
    end: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record hinges 0-7 sweeping through their MuJoCo control ranges.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--camera-height", type=float, default=2.2)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ramp-time", type=float, default=0.75)
    parser.add_argument("--sweep-time", type=float, default=1.5)
    parser.add_argument("--hold-time", type=float, default=0.25)
    parser.add_argument("--settle-time", type=float, default=0.4)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def build_scene(camera_height: float):
    world = SimpleFlatWorld()
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, 0, camera_height],
        xyaxes=[-1, 0, 0, 0, -1, 0],
    )
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def command_for_segment(segment: SweepSegment, elapsed: float) -> float:
    if segment.duration == 0:
        return segment.end
    phase = smoothstep(np.clip(elapsed / segment.duration, 0.0, 1.0))
    return float(segment.start + (segment.end - segment.start) * phase)


def sweep_segments(lower: float, upper: float, args: argparse.Namespace) -> list[SweepSegment]:
    return [
        SweepSegment("to min", args.ramp_time, 0.0, lower),
        SweepSegment("hold min", args.hold_time, lower, lower),
        SweepSegment("min to max", args.sweep_time, lower, upper),
        SweepSegment("hold max", args.hold_time, upper, upper),
        SweepSegment("to zero", args.ramp_time, upper, 0.0),
        SweepSegment("settle", args.settle_time, 0.0, 0.0),
    ]


def draw_overlay(
    frame,
    model,
    data,
    actuator_idx: int,
    segment: SweepSegment,
    command: float,
) -> None:
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_idx)
    lower, upper = model.actuator_ctrlrange[actuator_idx]
    lines = [
        "Sequential hinge sweep",
        f"hinge: {actuator_idx}  actuator: {actuator_name}",
        f"phase: {segment.name}",
        f"command: {command:+.3f} rad ({np.rad2deg(command):+.1f} deg)",
        f"range: [{lower:+.3f}, {upper:+.3f}] rad",
        f"t: {data.time:5.2f}s",
    ]
    for idx, text in enumerate(lines):
        y = 28 + idx * 25
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)


def record_hinge_sweep(args: argparse.Namespace, console: Console) -> Path | None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir is not None else RESULTS_DIR / run_id
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    model, data = build_scene(args.camera_height)
    try:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    except Exception as exc:
        console.log(f"Video renderer unavailable: {exc}", style="yellow")
        return None

    video_name = f"hinge_sweep_{run_id}"
    before_recording = set(video_dir.glob(f"{video_name}*"))
    video_recorder = VideoRecorder(
        file_name=video_name,
        output_folder=video_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    viz = mujoco.MjvOption()
    viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    metadata = {
        "run_id": run_id,
        "num_actuators": int(model.nu),
        "actuators": [
            {
                "index": idx,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx),
                "ctrlrange": model.actuator_ctrlrange[idx].tolist(),
            }
            for idx in range(model.nu)
        ],
        "ramp_time": args.ramp_time,
        "sweep_time": args.sweep_time,
        "hold_time": args.hold_time,
        "settle_time": args.settle_time,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "camera_height": args.camera_height,
        "camera_xyaxes": [-1, 0, 0, 0, -1, 0],
    }
    (run_dir / f"hinge_sweep_meta_{run_id}.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    steps_per_frame = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))
    ctrl = np.zeros(model.nu, dtype=np.float32)

    try:
        for actuator_idx in range(model.nu):
            lower, upper = model.actuator_ctrlrange[actuator_idx]
            for segment in sweep_segments(float(lower), float(upper), args):
                segment_start = float(data.time)
                while data.time - segment_start < segment.duration:
                    elapsed = float(data.time - segment_start)
                    command = command_for_segment(segment, elapsed)
                    for _ in range(steps_per_frame):
                        ctrl[:] = 0.0
                        ctrl[actuator_idx] = command
                        data.ctrl[:] = ctrl
                        mujoco.mj_step(model, data)

                    renderer.update_scene(data, scene_option=viz, camera=camera_id)
                    frame = renderer.render().copy()
                    if not args.no_overlay:
                        draw_overlay(frame, model, data, actuator_idx, segment, command)
                    video_recorder.write(frame)
    finally:
        video_recorder.release()
        renderer.close()

    candidates = sorted(
        set(video_dir.glob(f"{video_name}*")) - before_recording,
        key=lambda path: path.stat().st_mtime,
    )
    video_path = candidates[-1]
    console.log(f"Video saved -> {video_path}", style="green")
    console.log(f"Metadata saved -> {run_dir / f'hinge_sweep_meta_{run_id}.json'}")
    return video_path


def main() -> None:
    install()
    console = Console()
    args = parse_args()
    record_hinge_sweep(args, console)


if __name__ == "__main__":
    main()
