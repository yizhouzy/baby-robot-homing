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
from robot_control.artifacts import find_latest_best_model, infer_meta_path, resolve_model_path
from robot_control.config_gait import GaitConfig
from robot_control.controllers import load_gait_network
from robot_control.evaluation import run_open_loop_episode


RESULTS_DIR = Path("results/hinge_ab")


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
    forward_distance: float
    lateral_drift: float
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


def build_model(robot_factory) -> tuple[mujoco.MjModel, mujoco.MjData]:
    world = SimpleFlatWorld()
    world.spawn(robot_factory().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def evaluate_variant(name: str, robot_factory, network, config: GaitConfig) -> VariantResult:
    model, data = build_model(robot_factory)
    trace = run_open_loop_episode(model, data, network, config)
    delta = trace.final_pos - trace.initial_pos
    forward_distance = float(-delta[1])
    return VariantResult(
        name=name,
        forward_distance=forward_distance,
        lateral_drift=float(abs(delta[0])),
        final_z=trace.final_z,
        min_z=trace.min_z,
        max_roll_deg=float(np.rad2deg(trace.max_abs_roll)),
        max_pitch_deg=float(np.rad2deg(trace.max_abs_pitch)),
        path_length=trace.path_length,
        speed=forward_distance / config.duration,
        fell=trace.fell,
        positions=trace.positions,
    )


def save_csv(path: Path, rows: list[VariantResult]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "variant",
            "forward_distance_m",
            "forward_speed_mps",
            "lateral_drift_m",
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
                row.forward_distance,
                row.speed,
                row.lateral_drift,
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
    ax.set_title("Forward: Normal vs. Flipped Hinges(hinge_0and hinge_3)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_table(console: Console, rows: list[VariantResult]) -> None:
    table = Table(title="Core Forelimb Hinge Orientation A/B")
    table.add_column("Variant", style="bold cyan")
    table.add_column("Forward m", justify="right")
    table.add_column("Speed m/s", justify="right")
    table.add_column("Drift m", justify="right")
    table.add_column("Path m", justify="right")
    table.add_column("Min z", justify="right")
    table.add_column("Roll/Pitch deg", justify="right")
    table.add_column("Fell", justify="right")
    for row in rows:
        table.add_row(
            row.name,
            f"{row.forward_distance:.4f}",
            f"{row.speed:.4f}",
            f"{row.lateral_drift:.4f}",
            f"{row.path_length:.4f}",
            f"{row.min_z:.4f}",
            f"{row.max_roll_deg:.1f}/{row.max_pitch_deg:.1f}",
            str(row.fell),
        )
    console.print(table)


def main() -> None:
    install()
    console = Console()
    args = parse_args()

    model_path = resolve_model_path(args.model) if args.model else find_latest_best_model()
    meta_path = infer_meta_path(model_path, args.meta)
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

    rows = [
        evaluate_variant("current", baby_robot, network, config),
        evaluate_variant("flipped_core_forelimbs", baby_robot_flipped_core_forelimbs, network, config),
    ]
    print_table(console, rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"hinge_orientation_ab_{stamp}.csv"
    plot_path = output_dir / f"hinge_orientation_ab_{stamp}.png"
    save_csv(csv_path, rows)
    save_plot(plot_path, rows)
    console.log(f"Saved CSV -> {csv_path}", style="green")
    console.log(f"Saved plot -> {plot_path}", style="green")


if __name__ == "__main__":
    main()
