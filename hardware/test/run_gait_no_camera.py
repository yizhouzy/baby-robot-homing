"""Run a learned CPG gait on the Robohat without camera feedback."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.baby_hardware import (
    BATTERY_CSV_FIELDS,
    DEFAULT_SERVO_MAPPINGS,
    IMU_CSV_FIELDS,
    BabyRobotHardware,
    append_csv_row,
    battery_sample_row,
    imu_sample_row,
    make_run_dir,
    mapping_metadata,
    write_metadata,
)
from blocks.cpg_runtime import load_gait_runtime


DEFAULT_FORWARD_MODEL = (
    ROOT
    / "results/gait_cpg"
    / "20260627_234215_113631_seed43_DR"
    / "gait_best_20260627_234215_113631_seed43_DR.npy"
)
DEFAULT_FORWARD_META = (
    ROOT
    / "results/gait_cpg"
    / "20260627_234215_113631_seed43_DR"
    / "gait_meta_20260627_234215_113631_seed43_DR.npz"
)
DEFAULT_LEFT_MODEL = (
    ROOT
    / "results/left_cpg"
    / "20260627_seed41_left"
    / "spin_best_20260627_180547_744047_seed41_DR.npy"
)
DEFAULT_LEFT_META = (
    ROOT
    / "results/left_cpg"
    / "20260627_seed41_left"
    / "spin_meta_20260627_180547_744047_seed41_DR.npz"
)
DEFAULT_RIGHT_MODEL = (
    ROOT
    / "results/right_cpg"
    / "20260627_seed43_right"
    / "spin_best_20260627_200954_521575_seed43_DR.npy"
)
DEFAULT_RIGHT_META = (
    ROOT
    / "results/right_cpg"
    / "20260627_seed43_right"
    / "spin_meta_20260627_200954_521575_seed43_DR.npz"
)

DEFAULT_GAIT_MODELS = {
    "forward": (DEFAULT_FORWARD_MODEL, DEFAULT_FORWARD_META),
    "left": (DEFAULT_LEFT_MODEL, DEFAULT_LEFT_META),
    "right": (DEFAULT_RIGHT_MODEL, DEFAULT_RIGHT_META),
}
GAIT_TURNS = {"forward": 0.0, "left": 0.0, "right": 0.0}
DEFAULT_GAIT_SPEEDS = {"forward": 0.60, "left": 0.25, "right": 0.25}

SERVO_CSV_FIELDS = [
    "elapsed_s",
    "loop_dt_s",
    "gait",
    "turn",
    "speed",
    *[f"joint_{i}_rad" for i in range(8)],
    *[f"servo_{i}_command_deg" for i in range(8)],
    *[f"servo_{i}_readback_deg" for i in range(8)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned CPG gait on Robohat.")
    parser.add_argument("--gait", choices=["forward", "left", "right"], default="forward")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--control-hz", "--hz", dest="control_hz", type=float, default=20.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help=(
            "CPG speed. If omitted, uses behavior-tree defaults: "
            "forward=0.60, left/right=0.25."
        ),
    )
    parser.add_argument("--turn", type=float, default=None)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--joint-scales", type=float, nargs=8, default=None)
    parser.add_argument("--disable-joints", type=int, nargs="*", default=[])
    parser.add_argument("--max-joint-deg", type=float, default=90.0)
    parser.add_argument("--telemetry-hz", type=float, default=2.0)
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--delay-mode", action="store_true")
    parser.add_argument("--direct-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--readback", action="store_true")
    parser.add_argument("--neutral-hold-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def infer_meta_path(model_path: Path) -> Path:
    stem = model_path.stem
    if stem.startswith("gait_best_"):
        return model_path.with_name(f"gait_meta_{stem.removeprefix('gait_best_')}.npz")
    if stem.startswith("spin_best_"):
        return model_path.with_name(f"spin_meta_{stem.removeprefix('spin_best_')}.npz")
    raise ValueError(f"Cannot infer metadata path for {model_path}; pass --meta explicitly.")


def resolve_existing_path(path: Path, suffix: str) -> Path:
    if path.exists():
        return path
    if path.suffix == "":
        suffixed = path.with_suffix(suffix)
        if suffixed.exists():
            return suffixed
    return path


def resolve_model_and_meta(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.model is None:
        model_path, meta_path = DEFAULT_GAIT_MODELS[args.gait]
    else:
        model_path = args.model
        meta_path = args.meta if args.meta is not None else infer_meta_path(model_path)
    return resolve_existing_path(model_path, ".npy"), resolve_existing_path(meta_path, ".npz")


def joint_scales_from_args(args: argparse.Namespace) -> list[float]:
    scales = [1.0] * 8 if args.joint_scales is None else list(args.joint_scales)
    for joint_index in args.disable_joints:
        scales[joint_index] = 0.0
    return scales


def speed_from_args(args: argparse.Namespace) -> float:
    if args.speed is not None:
        return float(args.speed)
    return DEFAULT_GAIT_SPEEDS[args.gait]


def write_servo_row(
    path: Path,
    elapsed_s: float,
    loop_dt_s: float,
    args: argparse.Namespace,
    turn: float,
    speed: float,
    action,
    robot: BabyRobotHardware,
) -> None:
    readback = robot.read_all_servo_degrees() if args.readback else None
    row = {
        "elapsed_s": f"{elapsed_s:.6f}",
        "loop_dt_s": f"{loop_dt_s:.6f}",
        "gait": args.gait,
        "turn": f"{turn:.6f}",
        "speed": f"{speed:.6f}",
    }
    row.update({f"joint_{i}_rad": f"{value:.6f}" for i, value in enumerate(action)})
    row.update(
        {
            f"servo_{i}_command_deg": f"{value:.6f}"
            for i, value in enumerate(robot.last_command_degrees)
        },
    )
    if readback is not None:
        row.update(
            {
                f"servo_{i}_readback_deg": f"{value:.6f}"
                for i, value in enumerate(readback)
            },
        )
    else:
        row.update({f"servo_{i}_readback_deg": "" for i in range(8)})
    append_csv_row(path, SERVO_CSV_FIELDS, row)


def main() -> None:
    args = parse_args()
    model_path, meta_path = resolve_model_and_meta(args)
    network, meta, weight_format = load_gait_runtime(model_path, meta_path)
    num_joints = int(meta["num_joints"])
    turn = GAIT_TURNS[args.gait] if args.turn is None else float(args.turn)
    speed = speed_from_args(args)
    joint_scales = joint_scales_from_args(args)
    direct_mode = args.direct_mode or not args.delay_mode
    robot = BabyRobotHardware(direct_mode=direct_mode, delay_s=args.delay_s)
    run_dir = make_run_dir("run_gait_no_camera", args.output_dir)
    servo_csv = run_dir / "servo_samples.csv"
    battery_csv = run_dir / "battery_samples.csv"
    imu_csv = run_dir / "imu_samples.csv"
    battery_samples = []

    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/run_gait_no_camera.py",
            "gait": args.gait,
            "turn": turn,
            "speed": speed,
            "speed_source": "cli" if args.speed is not None else "behavior_tree_default",
            "amplitude_scale": args.amplitude_scale,
            "joint_scales": joint_scales,
            "disable_joints": args.disable_joints,
            "max_joint_deg": args.max_joint_deg,
            "duration": args.duration,
            "control_hz": args.control_hz,
            "model": str(model_path),
            "meta": str(meta_path),
            "weight_format": weight_format,
            "num_joints": num_joints,
            "direct_mode": direct_mode,
            "delay_mode": args.delay_mode,
            "delay_s": args.delay_s,
            "readback": args.readback,
            "neutral_hold_s": args.neutral_hold_s,
            "mappings": mapping_metadata(DEFAULT_SERVO_MAPPINGS),
            "output_dir": str(run_dir),
        },
    )

    print(f"Loaded {weight_format}: num_joints={num_joints}")
    print(f"Model: {model_path}")
    print(f"Meta:  {meta_path}")
    print(f"Running {args.gait}: turn={turn:+.2f}, speed={speed:.2f}")
    print(f"Servo update mode: {'direct' if direct_mode else 'delay'}")
    print(f"Joint scales: {joint_scales}")
    print(
        f"Command limit: +/-{args.max_joint_deg:.1f} joint deg after "
        f"speed and amplitude scaling.",
    )
    print("Use --amplitude-scale below 1.0 only for cautious reduced-motion tests.")
    print("Servo readback is off by default; add --readback only if feedback wires are connected.")

    period_s = 1.0 / args.control_hz
    telemetry_period_s = 1.0 / args.telemetry_hz
    next_telemetry_s = 0.0
    try:
        print(f"Moving to neutral for {args.neutral_hold_s:.1f}s before gait.")
        robot.neutral()
        time.sleep(args.neutral_hold_s)
        start = time.monotonic()
        previous_loop = start
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            elapsed_s = loop_start - start
            loop_dt_s = loop_start - previous_loop
            previous_loop = loop_start

            action = network.forward(turn=turn, speed=speed)
            action = action * float(args.amplitude_scale)
            action = action * joint_scales
            max_joint_rad = math.radians(args.max_joint_deg)
            action = action.clip(-max_joint_rad, max_joint_rad)
            robot.set_joint_angles(action)
            write_servo_row(servo_csv, elapsed_s, loop_dt_s, args, turn, speed, action, robot)

            if elapsed_s >= next_telemetry_s:
                battery = robot.read_battery(start, battery_samples)
                imu = robot.read_imu(start)
                append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(battery))
                append_csv_row(imu_csv, IMU_CSV_FIELDS, imu_sample_row(imu))
                next_telemetry_s += telemetry_period_s
                print(
                    f"t={elapsed_s:6.2f}s  battery={battery.voltage_v:6.3f}V  "
                    f"acc=({imu.acc_x:+.2f},{imu.acc_y:+.2f},{imu.acc_z:+.2f})  "
                    f"joint0={math.degrees(float(action[0])):+6.2f}deg  "
                    f"servo0={robot.last_command_degrees[0]:6.2f}deg"
                )

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
