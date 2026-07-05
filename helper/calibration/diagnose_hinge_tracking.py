"""Measure commanded vs actual MuJoCo hinge angles for the Baby robot.

This diagnostic tests whether the simulated hinge actuators really reach the
angles that we command with ``data.ctrl``. It commands one actuator at a time
through the MuJoCo control range and records both:

* command angle: the actuator target in ``data.ctrl``
* actual angle: the joint position in ``data.qpos``

If the actual angle range is much smaller than the command range, then the
simulation is under-moving compared with a real servo that tracks commands more
strongly.

Example:
    uv run --project ariel python helper/diagnose_hinge_tracking.py
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ariel.simulation.environments import SimpleFlatWorld
from blocks.baby_robot import baby_robot


RESULTS_DIR = Path("results/hinge_tracking")


@dataclass(frozen=True)
class Segment:
    name: str
    duration_s: float
    start_rad: float
    end_rad: float


@dataclass(frozen=True)
class SummaryRow:
    actuator_index: int
    actuator_name: str
    joint_name: str
    command_min_deg: float
    command_max_deg: float
    actual_min_deg: float
    actual_max_deg: float
    actual_span_deg: float
    command_span_deg: float
    span_ratio: float
    max_abs_error_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log commanded vs actual MuJoCo hinge angles for the Baby robot.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--actuators", type=int, nargs="+", default=None)
    parser.add_argument("--ramp-time", type=float, default=0.75)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    return parser.parse_args()


def build_model():
    world = SimpleFlatWorld()
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def command_at(segment: Segment, elapsed_s: float) -> float:
    if segment.duration_s == 0.0:
        return segment.end_rad
    phase = smoothstep(float(np.clip(elapsed_s / segment.duration_s, 0.0, 1.0)))
    return segment.start_rad + (segment.end_rad - segment.start_rad) * phase


def sweep_segments(lower: float, upper: float, args: argparse.Namespace) -> list[Segment]:
    return [
        Segment("settle zero", args.settle_time, 0.0, 0.0),
        Segment("zero to max", args.ramp_time, 0.0, upper),
        Segment("hold max", args.hold_time, upper, upper),
        Segment("max to zero", args.ramp_time, upper, 0.0),
        Segment("hold zero", args.hold_time, 0.0, 0.0),
        Segment("zero to min", args.ramp_time, 0.0, lower),
        Segment("hold min", args.hold_time, lower, lower),
        Segment("min to zero", args.ramp_time, lower, 0.0),
        Segment("settle zero", args.settle_time, 0.0, 0.0),
    ]


def actuator_joint_info(model, actuator_index: int) -> tuple[int, int, str, str]:
    joint_id = int(model.actuator_trnid[actuator_index, 0])
    qpos_addr = int(model.jnt_qposadr[joint_id])
    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_index,
    )
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    return joint_id, qpos_addr, actuator_name, joint_name


def run_actuator_sweep(
    model,
    data,
    actuator_index: int,
    args: argparse.Namespace,
) -> list[dict]:
    lower, upper = model.actuator_ctrlrange[actuator_index]
    _, qpos_addr, actuator_name, joint_name = actuator_joint_info(model, actuator_index)
    sample_period = 1.0 / args.sample_hz
    ctrl = np.zeros(model.nu, dtype=np.float32)
    rows = []

    mujoco.mj_resetData(model, data)
    next_sample_time = 0.0
    local_time = 0.0

    for segment in sweep_segments(float(lower), float(upper), args):
        segment_time = 0.0
        while segment_time < segment.duration_s:
            command = command_at(segment, segment_time)
            ctrl[:] = 0.0
            ctrl[actuator_index] = command
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            if local_time >= next_sample_time:
                actual = float(data.qpos[qpos_addr])
                rows.append({
                    "actuator_index": actuator_index,
                    "actuator_name": actuator_name,
                    "joint_name": joint_name,
                    "time_s": local_time,
                    "segment": segment.name,
                    "command_rad": float(command),
                    "actual_rad": actual,
                    "error_rad": actual - float(command),
                    "command_deg": float(np.rad2deg(command)),
                    "actual_deg": float(np.rad2deg(actual)),
                    "error_deg": float(np.rad2deg(actual - float(command))),
                })
                next_sample_time += sample_period

            local_time += model.opt.timestep
            segment_time += model.opt.timestep

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "actuator_index",
        "actuator_name",
        "joint_name",
        "time_s",
        "segment",
        "command_rad",
        "actual_rad",
        "error_rad",
        "command_deg",
        "actual_deg",
        "error_deg",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict]) -> SummaryRow:
    command = np.asarray([row["command_deg"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_deg"] for row in rows], dtype=float)
    error = np.asarray([row["error_deg"] for row in rows], dtype=float)
    command_span = float(np.max(command) - np.min(command))
    actual_span = float(np.max(actual) - np.min(actual))
    return SummaryRow(
        actuator_index=int(rows[0]["actuator_index"]),
        actuator_name=str(rows[0]["actuator_name"]),
        joint_name=str(rows[0]["joint_name"]),
        command_min_deg=float(np.min(command)),
        command_max_deg=float(np.max(command)),
        actual_min_deg=float(np.min(actual)),
        actual_max_deg=float(np.max(actual)),
        actual_span_deg=actual_span,
        command_span_deg=command_span,
        span_ratio=actual_span / command_span,
        max_abs_error_deg=float(np.max(np.abs(error))),
    )


def write_summary_csv(path: Path, summaries: list[SummaryRow]) -> None:
    fieldnames = list(SummaryRow.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row.__dict__)


def plot_actuator(out_dir: Path, rows: list[dict]) -> None:
    t = np.asarray([row["time_s"] for row in rows], dtype=float)
    command = np.asarray([row["command_deg"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_deg"] for row in rows], dtype=float)
    error = np.asarray([row["error_deg"] for row in rows], dtype=float)
    actuator_index = int(rows[0]["actuator_index"])
    actuator_name = str(rows[0]["actuator_name"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    fig.suptitle(f"Actuator {actuator_index}: {actuator_name}")
    axes[0].plot(t, command, label="command", linewidth=1.5)
    axes[0].plot(t, actual, label="actual qpos", linewidth=1.5)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axhline(90.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axhline(-90.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("angle [deg]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, error, color="tab:red", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("actual - command [deg]")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(out_dir / f"actuator_{actuator_index}_tracking.png", dpi=150)
    plt.close(fig)


def print_summary(console: Console, summaries: list[SummaryRow]) -> None:
    table = Table(title="Commanded vs Actual Simulated Hinge Motion")
    table.add_column("act")
    table.add_column("actuator")
    table.add_column("cmd span")
    table.add_column("actual span")
    table.add_column("ratio")
    table.add_column("max |err|")
    for row in summaries:
        table.add_row(
            str(row.actuator_index),
            row.actuator_name,
            f"{row.command_span_deg:.1f} deg",
            f"{row.actual_span_deg:.1f} deg",
            f"{row.span_ratio:.2f}",
            f"{row.max_abs_error_deg:.1f} deg",
        )
    console.print(table)


def main() -> None:
    install()
    console = Console()
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir if args.output_dir is not None else RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model, data = build_model()
    actuator_indices = args.actuators if args.actuators is not None else list(range(model.nu))
    all_rows = []
    summaries = []

    metadata = {
        "run_id": run_id,
        "num_actuators": int(model.nu),
        "actuators_tested": actuator_indices,
        "ramp_time": args.ramp_time,
        "hold_time": args.hold_time,
        "settle_time": args.settle_time,
        "sample_hz": args.sample_hz,
        "physics_timestep": float(model.opt.timestep),
        "actuators": [
            {
                "index": idx,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx),
                "ctrlrange_rad": model.actuator_ctrlrange[idx].tolist(),
            }
            for idx in range(model.nu)
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for actuator_index in actuator_indices:
        rows = run_actuator_sweep(model, data, actuator_index, args)
        all_rows.extend(rows)
        summaries.append(summarize(rows))
        plot_actuator(out_dir, rows)

    write_csv(out_dir / "hinge_tracking_samples.csv", all_rows)
    write_summary_csv(out_dir / "hinge_tracking_summary.csv", summaries)
    print_summary(console, summaries)
    console.print(f"Saved diagnostics -> {out_dir}")


if __name__ == "__main__":
    main()
