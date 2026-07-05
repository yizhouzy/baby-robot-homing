"""Record Robohat battery readings during fixed motion phases.

The goal is to show the limitation of Robohat's battery-capacity estimate:
voltage can drift smoothly while the reported percentage jumps up or down. The
script runs a deterministic sequence:

1. IDLE
2. APPROACHING(forward)
3. SEARCHING(Left)
4. SEARCHING(Right)
5. IDLE

Each phase defaults to 8 seconds. The output CSV records the phase label,
Robohat battery voltage, Robohat battery percentage, Robohat battery status,
and the commanded joint/servo values.
"""
# ruff: noqa: E402
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import math
import sys
import time

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


@dataclass(frozen=True)
class Phase:
    name: str
    gait: str
    duration_s: float


CSV_FIELDS = [
    "elapsed_s",
    "phase_elapsed_s",
    "loop_dt_s",
    "phase",
    "gait",
    "voltage_v",
    "percentage",
    "status",
    "drain_mv_per_min",
    *[f"joint_{i}_rad" for i in range(8)],
    *[f"joint_{i}_deg" for i in range(8)],
    *[f"servo_{i}_command_deg" for i in range(8)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record battery readings during fixed gait phases.")
    parser.add_argument("--phase-duration", type=float, default=8.0)
    parser.add_argument("--idle-start-s", type=float, default=None)
    parser.add_argument("--forward-s", type=float, default=None)
    parser.add_argument("--left-s", type=float, default=None)
    parser.add_argument("--right-s", type=float, default=None)
    parser.add_argument("--idle-end-s", type=float, default=None)
    parser.add_argument("--hz", type=float, default=25.0)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--max-joint-deg", type=float, default=90.0)
    parser.add_argument("--neutral-hold-s", type=float, default=1.0)
    parser.add_argument("--battery-warmup-timeout-s", type=float, default=8.0)
    parser.add_argument("--min-valid-battery-percent", type=int, default=1)
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--delay-mode", action="store_true")
    parser.add_argument("--forward-model", type=Path, default=None)
    parser.add_argument("--forward-meta", type=Path, default=None)
    parser.add_argument("--left-model", type=Path, default=None)
    parser.add_argument("--left-meta", type=Path, default=None)
    parser.add_argument("--right-model", type=Path, default=None)
    parser.add_argument("--right-meta", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_model_and_meta(gait: str, model_arg: Path | None, meta_arg: Path | None) -> tuple[Path, Path]:
    if model_arg is None:
        model_path, meta_path = DEFAULT_GAIT_MODELS[gait]
    else:
        model_path = model_arg
        meta_path = meta_arg if meta_arg is not None else infer_meta_path(model_path)
    return resolve_existing_path(model_path, ".npy"), resolve_existing_path(meta_path, ".npz")


def phases_from_args(args: argparse.Namespace) -> list[Phase]:
    default = args.phase_duration
    return [
        Phase("IDLE", "none", default if args.idle_start_s is None else args.idle_start_s),
        Phase("APPROACHING(forward)", "forward", default if args.forward_s is None else args.forward_s),
        Phase("SEARCHING(Left)", "left", default if args.left_s is None else args.left_s),
        Phase("SEARCHING(Right)", "right", default if args.right_s is None else args.right_s),
        Phase("IDLE", "none", default if args.idle_end_s is None else args.idle_end_s),
    ]


def gait_action(network, speed: float, args: argparse.Namespace) -> np.ndarray:
    action = network.forward(turn=0.0, speed=speed)
    action = action * float(args.amplitude_scale)
    max_joint_rad = math.radians(args.max_joint_deg)
    return action.clip(-max_joint_rad, max_joint_rad)


def drain_mv_per_min(samples: list, elapsed_s: float, voltage_v: float) -> float:
    if not samples or elapsed_s <= samples[0].elapsed_s:
        return 0.0
    elapsed_min = (elapsed_s - samples[0].elapsed_s) / 60.0
    return (samples[0].voltage_v - voltage_v) * 1000.0 / elapsed_min


def wait_for_battery_monitor(robot: BabyRobotHardware, args: argparse.Namespace) -> None:
    start = time.monotonic()
    samples = []
    while time.monotonic() - start < args.battery_warmup_timeout_s:
        sample = robot.read_battery(start, samples)
        if sample.percentage >= args.min_valid_battery_percent:
            print(
                f"Battery monitor ready: {sample.percentage}%  "
                f"{sample.voltage_v:.3f}V  {sample.status}",
            )
            return
        print(
            f"Waiting for battery monitor: {sample.percentage}%  "
            f"{sample.voltage_v:.3f}V  {sample.status}",
        )
        time.sleep(0.5)
    sample = samples[-1]
    print(
        f"Battery monitor did not become valid before recording: "
        f"{sample.percentage}%  {sample.voltage_v:.3f}V  {sample.status}",
    )


def write_row(
    path: Path,
    *,
    elapsed_s: float,
    phase_elapsed_s: float,
    loop_dt_s: float,
    phase: Phase,
    battery,
    action,
    robot: BabyRobotHardware,
    run_battery_samples: list,
) -> None:
    row = {
        "elapsed_s": f"{elapsed_s:.6f}",
        "phase_elapsed_s": f"{phase_elapsed_s:.6f}",
        "loop_dt_s": f"{loop_dt_s:.6f}",
        "phase": phase.name,
        "gait": phase.gait,
        "voltage_v": f"{battery.voltage_v:.6f}",
        "percentage": battery.percentage,
        "status": battery.status,
        "drain_mv_per_min": f"{drain_mv_per_min(run_battery_samples, elapsed_s, battery.voltage_v):.6f}",
    }
    row.update({f"joint_{i}_rad": f"{value:.6f}" for i, value in enumerate(action)})
    row.update({f"joint_{i}_deg": f"{math.degrees(float(value)):.6f}" for i, value in enumerate(action)})
    row.update(
        {
            f"servo_{i}_command_deg": f"{value:.6f}"
            for i, value in enumerate(robot.last_command_degrees)
        },
    )
    append_csv_row(path, CSV_FIELDS, row)


def main() -> None:
    args = parse_args()
    phases = phases_from_args(args)
    forward_model, forward_meta = resolve_model_and_meta("forward", args.forward_model, args.forward_meta)
    left_model, left_meta = resolve_model_and_meta("left", args.left_model, args.left_meta)
    right_model, right_meta = resolve_model_and_meta("right", args.right_model, args.right_meta)

    networks = {
        "forward": load_gait_runtime(forward_model, forward_meta)[0],
        "left": load_gait_runtime(left_model, left_meta)[0],
        "right": load_gait_runtime(right_model, right_meta)[0],
    }
    direct_mode = not args.delay_mode
    robot = BabyRobotHardware(direct_mode=direct_mode, delay_s=args.delay_s, enable_camera=False)
    run_dir = make_run_dir("record_battery_phase_sequence", args.output_dir)
    csv_path = run_dir / "battery_phase_samples.csv"

    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/record_battery_phase_sequence.py",
            "phases": [asdict(phase) for phase in phases],
            "hz": args.hz,
            "speed": args.speed,
            "amplitude_scale": args.amplitude_scale,
            "max_joint_deg": args.max_joint_deg,
            "neutral_hold_s": args.neutral_hold_s,
            "battery_warmup_timeout_s": args.battery_warmup_timeout_s,
            "min_valid_battery_percent": args.min_valid_battery_percent,
            "direct_mode": direct_mode,
            "delay_s": args.delay_s,
            "forward_model": str(forward_model),
            "forward_meta": str(forward_meta),
            "left_model": str(left_model),
            "left_meta": str(left_meta),
            "right_model": str(right_model),
            "right_meta": str(right_meta),
            "mappings": mapping_metadata(DEFAULT_SERVO_MAPPINGS),
            "output_dir": str(run_dir),
            "csv": str(csv_path),
        },
    )

    print(f"Results: {run_dir}")
    print(f"CSV:     {csv_path}")
    print("Phase sequence:")
    for phase in phases:
        print(f"  {phase.name:20s} {phase.duration_s:5.1f}s")

    period_s = 1.0 / args.hz
    run_battery_samples = []
    try:
        wait_for_battery_monitor(robot, args)
        print(f"Moving to neutral for {args.neutral_hold_s:.1f}s.")
        robot.neutral()
        time.sleep(args.neutral_hold_s)

        run_start = time.monotonic()
        previous_loop = run_start
        next_print_s = 0.0
        for phase in phases:
            phase_start = time.monotonic()
            if phase.gait != "none":
                networks[phase.gait].cpg.reset()
            while time.monotonic() - phase_start < phase.duration_s:
                loop_start = time.monotonic()
                elapsed_s = loop_start - run_start
                phase_elapsed_s = loop_start - phase_start
                loop_dt_s = loop_start - previous_loop
                previous_loop = loop_start

                if phase.gait == "none":
                    action = np.zeros(8, dtype=np.float32)
                else:
                    action = gait_action(networks[phase.gait], args.speed, args)
                robot.set_joint_angles(action)

                battery = robot.read_battery(run_start, run_battery_samples)
                write_row(
                    csv_path,
                    elapsed_s=elapsed_s,
                    phase_elapsed_s=phase_elapsed_s,
                    loop_dt_s=loop_dt_s,
                    phase=phase,
                    battery=battery,
                    action=action,
                    robot=robot,
                    run_battery_samples=run_battery_samples,
                )

                if elapsed_s >= next_print_s:
                    print(
                        f"t={elapsed_s:6.2f}s  {phase.name:20s}  "
                        f"battery={battery.percentage:3d}%  {battery.voltage_v:.3f}V  "
                        f"{battery.status}",
                    )
                    next_print_s += 1.0

                time.sleep(max(0.0, period_s - (time.monotonic() - loop_start)))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"Returning to neutral for {args.neutral_hold_s:.1f}s.")
        robot.neutral()
        time.sleep(args.neutral_hold_s)
        robot.close()

    print(f"Done. Results: {run_dir}")


if __name__ == "__main__":
    main()
