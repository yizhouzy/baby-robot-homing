"""Compare current Baby robot hinges with flipped core-forelimb hinges."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import quaternion as qnp
from rich.console import Console
from rich.table import Table
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import (
    HingeModule,
    ROTOR_DIMENSIONS,
    ROTOR_MASS,
    SHRINK,
    STATOR_DIMENSIONS,
    STATOR_MASS,
)
from ariel.body_phenotypes.robogen_lite.modules.module import Module
from ariel.simulation.environments import SimpleFlatWorld
from blocks.baby_robot import baby_robot
from robot_control.artifacts import (
    find_latest_best_model as find_latest_forward_model,
    infer_meta_path as infer_forward_meta_path,
    resolve_model_path as resolve_forward_model_path,
)
from robot_control.config_gait import GaitConfig
from robot_control.controllers import load_gait_network, sanitize_action
from robot_control.evaluation import quat_to_roll_pitch, run_open_loop_episode


RESULTS_DIR = Path("results/hinge_ab")
SPIN_RESULTS_DIR = Path("results/turn_cpg")


class FlippedHingeModule(Module):
    """Hinge with rotor on the parent side and stator on the distal side."""

    index: int | None = None
    module_type: ModuleType = ModuleType.HINGE

    def __init__(self, index: int) -> None:
        self.index = index
        spec = mujoco.MjSpec()

        hinge = spec.worldbody.add_body(
            name=self.module_type.name.lower(),
            mass=STATOR_MASS + ROTOR_MASS,
        )

        rotor = hinge.add_body(
            name="rotor",
            pos=[0, ROTOR_DIMENSIONS[1], 0],
        )
        rotor.add_geom(
            name="rotor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=ROTOR_MASS,
            size=ROTOR_DIMENSIONS,
            rgba=(160 / 255, 24 / 255, 33 / 255, 1),
        )

        stator = hinge.add_body(
            name="stator",
            pos=[0, ROTOR_DIMENSIONS[1] * 2 + STATOR_DIMENSIONS[1], 0],
        )
        stator.add_geom(
            name="stator",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=STATOR_MASS,
            size=np.array(STATOR_DIMENSIONS) * SHRINK,
            rgba=(223 / 255, 41 / 255, 53 / 255, 1),
        )

        self.sites = {
            ModuleFaces.FRONT: stator.add_site(
                name=f"{self.module_type.name.lower()}-front",
                pos=[0, STATOR_DIMENSIONS[1], 0],
            )
        }

        servo_name = "servo"
        stator.add_joint(
            name=servo_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=(0, 0, 1),
            pos=[0, -STATOR_DIMENSIONS[1], 0],
        )

        dynprm = np.zeros(10)
        gainprm = np.zeros(10)
        biasprm = np.zeros(10)
        gainprm[0] = 1
        biasprm[:3] = [0, -1, -1]

        spec.add_exclude(bodyname1="stator", bodyname2="rotor")
        spec.add_actuator(
            name=servo_name,
            dyntype=mujoco.mjtDyn.mjDYN_NONE,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            dynprm=dynprm,
            gainprm=gainprm,
            biasprm=biasprm,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=servo_name,
            ctrlrange=(-np.pi / 2, np.pi / 2),
        )

        self.spec = spec
        self.body = hinge
        self.rotate(angle=0)

    def rotate(self, angle: float) -> None:
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])
        self.body.quat = np.round(np.roll(qnp.as_float_array(quat), shift=-1), decimals=3)


def baby_robot_flipped_core_forelimbs() -> CoreModule:
    core = CoreModule(index=0)

    hinge_0 = FlippedHingeModule(index=1)
    hinge_0.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(body=hinge_0.body, prefix="hinge_0")

    hinge_1 = HingeModule(index=3)
    hinge_1.rotate(90)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(body=hinge_1.body, prefix="hinge_1")

    brick_0 = BrickModule(index=4)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(body=brick_0.body, prefix="brick_0")

    hinge_2 = HingeModule(index=8)
    hinge_2.rotate(90)
    brick_0.sites[ModuleFaces.FRONT].attach_body(body=hinge_2.body, prefix="hinge_2")

    brick_1 = BrickModule(index=9)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(body=brick_1.body, prefix="brick_1")

    hinge_3 = FlippedHingeModule(index=10)
    hinge_3.rotate(90)
    core.sites[ModuleFaces.RIGHT].attach_body(body=hinge_3.body, prefix="hinge_3")

    brick_2 = BrickModule(index=11)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(body=brick_2.body, prefix="brick_2")

    hinge_4 = HingeModule(index=21)
    core.sites[ModuleFaces.FRONT].attach_body(body=hinge_4.body, prefix="hinge_4")

    brick_3 = BrickModule(index=22)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(body=brick_3.body, prefix="brick_3")

    hinge_5 = HingeModule(index=23)
    brick_3.sites[ModuleFaces.FRONT].attach_body(body=hinge_5.body, prefix="hinge_5")

    brick_4 = BrickModule(index=24)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(body=brick_4.body, prefix="brick_4")

    hinge_6 = HingeModule(index=26)
    hinge_6.rotate(90)
    brick_4.sites[ModuleFaces.LEFT].attach_body(body=hinge_6.body, prefix="hinge_6")

    brick_5 = BrickModule(index=27)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(body=brick_5.body, prefix="brick_5")

    hinge_7 = HingeModule(index=28)
    hinge_7.rotate(90)
    brick_4.sites[ModuleFaces.RIGHT].attach_body(body=hinge_7.body, prefix="hinge_7")

    brick_6 = BrickModule(index=29)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(body=brick_6.body, prefix="brick_6")

    return core


@dataclass
class VariantResult:
    name: str
    mode: str
    forward_distance: float
    lateral_drift: float
    total_rotation_deg: float
    spin_reward: float
    final_z: float
    min_z: float
    max_roll_deg: float
    max_pitch_deg: float
    path_length: float
    speed: float
    fell: bool
    positions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B test current vs flipped core-forelimb hinge orientation."
    )
    parser.add_argument("--mode", choices=["forward", "spin"], default="spin")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--dur", type=float, default=30.0)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fall-z-threshold", type=float, default=0.05)
    parser.add_argument("--fall-tilt-threshold-deg", type=float, default=75.0)
    parser.add_argument("--use-tilt-fall", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    return parser.parse_args()


def find_latest_spin_model(results_dir: Path = SPIN_RESULTS_DIR) -> Path:
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


def resolve_spin_model_path(model_path: str | Path, results_dir: Path = SPIN_RESULTS_DIR) -> Path:
    path = Path(model_path)
    if path.exists():
        return path

    under_results = results_dir / path
    if under_results.exists():
        return under_results

    matches = sorted(results_dir.glob(f"*/checkpoints/{path.name}"))
    matches.extend(sorted(results_dir.glob(f"*/{path.name}")))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"{model_path} matched multiple spin checkpoints under {results_dir}; pass the full path."
        )
    raise FileNotFoundError(f"Spin checkpoint not found: {model_path}")


def infer_spin_run_id(model_path: Path) -> str:
    stem = model_path.stem
    if stem.startswith("spin_best_"):
        return stem.removeprefix("spin_best_")
    if stem.startswith("spin_ckpt_"):
        return stem.removeprefix("spin_ckpt_").rsplit("_gen", maxsplit=1)[0]
    return model_path.parent.name


def infer_spin_run_dir(model_path: Path) -> Path:
    if model_path.parent.name == "checkpoints":
        return model_path.parent.parent
    return model_path.parent


def infer_spin_meta_path(model_path: Path, requested_meta_path: str | Path | None = None) -> Path | None:
    if requested_meta_path is not None:
        return Path(requested_meta_path)

    run_dir = infer_spin_run_dir(model_path)
    run_id = infer_spin_run_id(model_path)
    preferred = run_dir / f"spin_meta_{run_id}.npz"
    if preferred.exists():
        return preferred

    candidates = sorted(run_dir.glob("spin_meta_*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def resolve_model_and_meta(mode: str, model_arg: str | None, meta_arg: str | None) -> tuple[Path, Path | None]:
    if mode == "spin":
        model_path = resolve_spin_model_path(model_arg) if model_arg else find_latest_spin_model()
        return model_path, infer_spin_meta_path(model_path, meta_arg)

    model_path = resolve_forward_model_path(model_arg) if model_arg else find_latest_forward_model()
    return model_path, infer_forward_meta_path(model_path, meta_arg)


def build_model(robot_factory) -> tuple[mujoco.MjModel, mujoco.MjData]:
    world = SimpleFlatWorld()
    world.spawn(robot_factory().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def core_heading(data) -> float:
    xmat = data.geom("robot1_core").xmat.reshape(3, 3)
    forward_2d = (xmat @ np.array([0.0, -1.0, 0.0]))[:2]
    return float(np.arctan2(forward_2d[1], forward_2d[0]))


def run_spin_episode(model, data, network, config: GaitConfig):
    mujoco.mj_forward(model, data)
    network.reset_hidden()
    current_action = np.zeros(model.nu, dtype=np.float32)
    xy_history = [(float(data.qpos[0]), float(data.qpos[1]))]
    heading_history = [core_heading(data)]
    z_history = [float(data.geom("robot1_core").xpos[2])]
    initial_roll, initial_pitch = quat_to_roll_pitch(data.qpos[3:7].copy())
    max_abs_roll = abs(initial_roll)
    max_abs_pitch = abs(initial_pitch)
    step = 0

    while data.time < config.duration:
        if step % config.control_step_freq == 0:
            current_action = sanitize_action(
                network.forward(config.training_turn, config.training_speed),
                model,
            )
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        xy_history.append((float(data.qpos[0]), float(data.qpos[1])))
        heading_history.append(core_heading(data))
        z_history.append(float(data.geom("robot1_core").xpos[2]))
        current_roll, current_pitch = quat_to_roll_pitch(data.qpos[3:7].copy())
        max_abs_roll = max(max_abs_roll, abs(current_roll))
        max_abs_pitch = max(max_abs_pitch, abs(current_pitch))
        step += 1

    xy = np.asarray(xy_history, dtype=np.float32)
    headings = np.unwrap(np.asarray(heading_history, dtype=np.float32))
    z = np.asarray(z_history, dtype=np.float32)
    total_rotation = float(abs(headings[-1] - headings[0]))
    drift = float(np.linalg.norm(xy[-1] - xy[0]))
    reward = float(total_rotation / (1.0 + drift))
    positions = np.column_stack([xy, z])
    path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    return xy, z, total_rotation, drift, reward, path_length, max_abs_roll, max_abs_pitch


def evaluate_forward_variant(name: str, robot_factory, network, config: GaitConfig) -> VariantResult:
    model, data = build_model(robot_factory)
    trace = run_open_loop_episode(model, data, network, config)
    delta = trace.final_pos - trace.initial_pos
    forward_distance = float(-delta[1])
    return VariantResult(
        name=name,
        mode="forward",
        forward_distance=forward_distance,
        lateral_drift=float(abs(delta[0])),
        total_rotation_deg=0.0,
        spin_reward=0.0,
        final_z=trace.final_z,
        min_z=trace.min_z,
        max_roll_deg=float(np.rad2deg(trace.max_abs_roll)),
        max_pitch_deg=float(np.rad2deg(trace.max_abs_pitch)),
        path_length=trace.path_length,
        speed=forward_distance / config.duration,
        fell=trace.fell,
        positions=trace.positions,
    )


def evaluate_spin_variant(name: str, robot_factory, network, config: GaitConfig) -> VariantResult:
    model, data = build_model(robot_factory)
    xy, z, total_rotation, drift, reward, path_length, max_abs_roll, max_abs_pitch = run_spin_episode(
        model,
        data,
        network,
        config,
    )
    positions = np.column_stack([xy, z])
    fell_by_z = bool(np.min(z) < config.fall_z_threshold)
    fell_by_tilt = bool(
        config.use_tilt_fall
        and max(max_abs_roll, max_abs_pitch) > config.fall_tilt_threshold_rad
    )
    return VariantResult(
        name=name,
        mode="spin",
        forward_distance=float(-(xy[-1, 1] - xy[0, 1])),
        lateral_drift=drift,
        total_rotation_deg=float(np.rad2deg(total_rotation)),
        spin_reward=reward,
        final_z=float(z[-1]),
        min_z=float(np.min(z)),
        max_roll_deg=float(np.rad2deg(max_abs_roll)),
        max_pitch_deg=float(np.rad2deg(max_abs_pitch)),
        path_length=path_length,
        speed=0.0,
        fell=fell_by_z or fell_by_tilt,
        positions=positions,
    )


def save_csv(path: Path, rows: list[VariantResult]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "variant",
            "mode",
            "forward_distance_m",
            "forward_speed_mps",
            "lateral_drift_m",
            "total_rotation_deg",
            "spin_reward",
            "path_length_m",
            "final_z_m",
            "min_z_m",
            "max_roll_deg",
            "max_pitch_deg",
            "fell",
        ])
        for row in rows:
            writer.writerow([
                row.name,
                row.mode,
                row.forward_distance,
                row.speed,
                row.lateral_drift,
                row.total_rotation_deg,
                row.spin_reward,
                row.path_length,
                row.final_z,
                row.min_z,
                row.max_roll_deg,
                row.max_pitch_deg,
                row.fell,
            ])


def save_plot(path: Path, rows: list[VariantResult]) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for row in rows:
        xy = row.positions[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2, label=row.name)
        ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=40)
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="x", s=60)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    mode = rows[0].mode if rows else "unknown"
    ax.set_title(f"{mode.title()} Spin: Normal vs. Flipped Hinges(hinge_0and hinge_3")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_table(console: Console, rows: list[VariantResult]) -> None:
    mode = rows[0].mode if rows else "unknown"
    table = Table(title=f"{mode.title()} Core Forelimb Hinge Orientation A/B")
    table.add_column("Variant", style="bold cyan")
    if mode == "spin":
        table.add_column("Rotation deg", justify="right")
        table.add_column("Spin reward", justify="right")
    else:
        table.add_column("Forward m", justify="right")
        table.add_column("Speed m/s", justify="right")
    table.add_column("Drift m", justify="right")
    table.add_column("Path m", justify="right")
    table.add_column("Min z", justify="right")
    table.add_column("Roll/Pitch deg", justify="right")
    table.add_column("Fell", justify="right")
    for row in rows:
        suffix = [
            f"{row.lateral_drift:.4f}",
            f"{row.path_length:.4f}",
            f"{row.min_z:.4f}",
            f"{row.max_roll_deg:.1f}/{row.max_pitch_deg:.1f}",
            str(row.fell),
        ]
        if mode == "spin":
            table.add_row(row.name, f"{row.total_rotation_deg:.2f}", f"{row.spin_reward:.4f}", *suffix)
        else:
            table.add_row(row.name, f"{row.forward_distance:.4f}", f"{row.speed:.4f}", *suffix)
    console.print(table)


def main() -> None:
    install()
    console = Console()
    args = parse_args()

    model_path, meta_path = resolve_model_and_meta(args.mode, args.model, args.meta)
    network, meta, weight_format = load_gait_network(model_path, meta_path)
    config = GaitConfig(
        duration=args.dur,
        seed=args.seed,
        training_turn=args.turn,
        training_speed=args.speed,
        fall_z_threshold=args.fall_z_threshold,
        fall_tilt_threshold_deg=args.fall_tilt_threshold_deg,
        use_tilt_fall=args.use_tilt_fall,
        record_video=False,
        domain_randomization=False,
        eval_repeats=1,
    )

    console.log(f"Model: {model_path}")
    console.log(f"Metadata: {meta_path}")
    console.log(
        f"Loaded {weight_format}: num_joints={int(meta['num_joints'])}, dt={float(meta['dt']):.4f}"
    )

    evaluator = evaluate_spin_variant if args.mode == "spin" else evaluate_forward_variant
    rows = [
        evaluator("current", baby_robot, network, config),
        evaluator("flipped_core_forelimbs", baby_robot_flipped_core_forelimbs, network, config),
    ]
    print_table(console, rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"hinge_orientation_ab_{args.mode}_{stamp}.csv"
    plot_path = output_dir / f"hinge_orientation_ab_{args.mode}_{stamp}.png"
    save_csv(csv_path, rows)
    save_plot(plot_path, rows)
    console.log(f"Saved CSV -> {csv_path}", style="green")
    console.log(f"Saved plot -> {plot_path}", style="green")


if __name__ == "__main__":
    main()
