"""Evolve a domain-randomized spin CPG gait controller for the Baby robot."""
# ruff: noqa: E402
from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import ray
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.config_gait import (
    CONTROL_STEP_FREQ,
    OPTIMIZED_CPG_GROUPS,
    RAY_RUNTIME_ENV,
    make_run_id,
)
from robot_control.contact_logging import (
    SelfContactAccumulator,
    SelfContactSummary,
    max_self_contact_penetration,
    total_self_contact_frames,
    total_self_contact_pairs,
    write_self_contact_summary_csv,
)
from robot_control.controllers import (
    CPGGaitController,
    get_cpg_vector,
    load_gait_network,
    sanitize_action,
    set_cpg_vector,
)
from robot_control.domain_randomization import (
    apply_action_noise,
    apply_domain_randomization,
    capture_domain_randomization_state,
)
from robot_control.evaluation import build_training_model, quat_to_roll_pitch, seed_everything
from robot_control.optimizers import NevergradCMAESOptimizer
from robot_control.spin_recording import record_spin_video


RESULTS_DIR = Path("results/turn_cpg")
LEFT_RESULTS_DIR = Path("results/left_cpg")
RIGHT_RESULTS_DIR = Path("results/right_cpg")
TRAINING_TURN = 0.0
TRAINING_SPEED = 1.0
DEFAULT_CORE_FALL_Z_THRESHOLD = 0.065


@dataclass(frozen=True)
class SpinConfig:
    budget: int = 300
    population: int = 50
    duration: float = 10.0
    num_actors: int = 8
    seed: int = 42
    sigma: float = 0.5
    record_video: bool = True
    video_camera_height: float = 3.5
    video_width: int = 640
    video_height: int = 640
    video_fps: int = 30
    video_overlay: bool = True
    results_dir: Path = RESULTS_DIR
    run_id: str | None = None
    training_turn: float = TRAINING_TURN
    training_speed: float = TRAINING_SPEED
    spin_direction: str = "either"
    control_step_freq: int = CONTROL_STEP_FREQ
    fall_z_threshold: float = DEFAULT_CORE_FALL_Z_THRESHOLD
    fall_tilt_threshold_deg: float = 75.0
    use_tilt_fall: bool = True
    eval_repeats: int = 3
    domain_randomization: bool = True
    action_noise_std: float = 0.03
    friction_scale_min: float = 0.5
    friction_scale_max: float = 1.5
    mass_scale_min: float = 0.9
    mass_scale_max: float = 1.1
    joint_strength_scale_min: float = 0.7
    joint_strength_scale_max: float = 1.3
    collision_weight: float = 3.0
    ray_runtime_env: dict = field(default_factory=lambda: RAY_RUNTIME_ENV)

    def __post_init__(self) -> None:
        object.__setattr__(self, "eval_repeats", max(1, self.eval_repeats))
        if self.run_id is None:
            run_id = make_run_id(self.seed)
            if self.domain_randomization:
                run_id = f"{run_id}_DR"
            object.__setattr__(self, "run_id", run_id)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / str(self.run_id)

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def video_dir(self) -> Path:
        return self.run_dir / "videos"

    @property
    def fall_tilt_threshold_rad(self) -> float:
        return float(np.deg2rad(self.fall_tilt_threshold_deg))


@dataclass
class SpinSetup:
    initial_guess: np.ndarray
    num_joints: int
    dt: float
    n_params: int
    pop_size: int


@dataclass
class SpinValidation:
    reward: float
    score: float
    total_rotation: float
    drift: float
    path_length: float
    min_z: float
    max_abs_roll: float
    max_abs_pitch: float
    fell: bool
    fell_by_z: bool
    fell_by_tilt: bool
    fall_time: float | None
    final_pos: np.ndarray
    xy_history: np.ndarray
    heading_history: np.ndarray
    z_history: np.ndarray
    self_contacts: list[SelfContactSummary]


@dataclass
class SpinResult:
    best_weights: np.ndarray
    best_score: float
    best_reward: float
    history: list[float]
    global_history: list[float]
    reward_history: list[float]
    global_reward_history: list[float]
    validation: SpinValidation
    num_joints: int
    dt: float
    n_params: int
    pop_size: int


_SPIN_ENV: dict = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--dur", type=float, default=10.0)
    parser.add_argument("--num-actors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--spin-direction", choices=["right", "left", "either"], default="either")
    parser.add_argument("--fall-z-threshold", type=float, default=DEFAULT_CORE_FALL_Z_THRESHOLD)
    parser.add_argument("--fall-tilt-threshold-deg", type=float, default=75.0)
    parser.add_argument("--no-tilt-fall", action="store_true")
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--action-noise-std", type=float, default=0.03)
    parser.add_argument("--friction-scale-min", type=float, default=0.5)
    parser.add_argument("--friction-scale-max", type=float, default=1.5)
    parser.add_argument("--mass-scale-min", type=float, default=0.9)
    parser.add_argument("--mass-scale-max", type=float, default=1.1)
    parser.add_argument("--joint-strength-scale-min", type=float, default=0.7)
    parser.add_argument("--joint-strength-scale-max", type=float, default=1.3)
    parser.add_argument("--collision-weight", type=float, default=3.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-camera-height", type=float, default=3.5)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=640)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SpinConfig:
    results_dir = {
        "left": LEFT_RESULTS_DIR,
        "right": RIGHT_RESULTS_DIR,
    }.get(args.spin_direction, RESULTS_DIR)
    return SpinConfig(
        budget=args.budget,
        population=args.population,
        duration=args.dur,
        num_actors=args.num_actors,
        seed=int(args.seed),
        sigma=args.sigma,
        results_dir=results_dir,
        spin_direction=args.spin_direction,
        fall_z_threshold=args.fall_z_threshold,
        fall_tilt_threshold_deg=args.fall_tilt_threshold_deg,
        use_tilt_fall=not args.no_tilt_fall,
        eval_repeats=args.eval_repeats,
        domain_randomization=not args.no_domain_randomization,
        action_noise_std=args.action_noise_std,
        friction_scale_min=args.friction_scale_min,
        friction_scale_max=args.friction_scale_max,
        mass_scale_min=args.mass_scale_min,
        mass_scale_max=args.mass_scale_max,
        joint_strength_scale_min=args.joint_strength_scale_min,
        joint_strength_scale_max=args.joint_strength_scale_max,
        collision_weight=args.collision_weight,
        record_video=not args.no_video,
        video_camera_height=args.video_camera_height,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        video_overlay=not args.no_overlay,
    )


def ensure_run_dirs(config: SpinConfig) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=False)
    config.checkpoint_dir.mkdir(exist_ok=True)
    config.video_dir.mkdir(exist_ok=True)


def initialize_training(config: SpinConfig, console: Console) -> SpinSetup:
    model, _ = build_training_model()
    num_joints = model.nu
    dt = model.opt.timestep * config.control_step_freq
    network = CPGGaitController(num_joints, dt=dt, seed=config.seed)
    initial_guess = get_cpg_vector(network)
    n_params = initial_guess.size
    min_pop_size = 4 + int(3 * np.log(max(n_params, 2)))
    pop_size = max(config.population, min_pop_size)
    if pop_size != config.population:
        console.log(
            f"Population increased from {config.population} to {pop_size} "
            "for CMA-ES covariance adaptation.",
            style="yellow",
        )
    return SpinSetup(initial_guess, num_joints, dt, n_params, pop_size)


def show_startup_summary(config: SpinConfig, setup: SpinSetup, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Optimizer", "nevergrad.ParametrizedCMA")
    table.add_row("Objective", "maximize core heading rotation / drift")
    table.add_row("Spin direction", config.spin_direction)
    table.add_row("Budget", f"{config.budget} generations")
    table.add_row("Population", f"{setup.pop_size} (requested {config.population})")
    table.add_row("Ray actors", str(config.num_actors))
    table.add_row("Sigma", f"{config.sigma:g}")
    table.add_row("Core fall z threshold", f"{config.fall_z_threshold:g}m")
    table.add_row(
        "Tilt fall threshold",
        f"{config.fall_tilt_threshold_deg:g} deg"
        if config.use_tilt_fall
        else "disabled",
    )
    table.add_row("Evaluation repeats", str(config.eval_repeats))
    table.add_row("Domain randomization", "on" if config.domain_randomization else "off")
    table.add_row("Action noise", f"{config.action_noise_std:g} rad")
    table.add_row(
        "Friction scale",
        f"{config.friction_scale_min:g}-{config.friction_scale_max:g}",
    )
    table.add_row("Mass scale", f"{config.mass_scale_min:g}-{config.mass_scale_max:g}")
    table.add_row(
        "Servo strength scale",
        f"{config.joint_strength_scale_min:g}-{config.joint_strength_scale_max:g}",
    )
    table.add_row("Collision weight", f"{config.collision_weight:g}")
    table.add_row("Duration", f"{config.duration:g}s")
    table.add_row("Seed", str(config.seed))
    table.add_row("Spawn", "origin, no target")
    table.add_row("CPG outputs", str(setup.num_joints))
    table.add_row("CPG params", f"{setup.n_params} optimized")
    table.add_row("Optimized groups", ", ".join(OPTIMIZED_CPG_GROUPS))
    table.add_row("Run dir", str(config.run_dir))
    console.print(Panel(table, title="Spin Gait Evolution", border_style="cyan"))


def core_heading(data) -> float:
    xmat = data.geom("robot1_core").xmat.reshape(3, 3)
    forward_2d = (xmat @ np.array([0.0, -1.0, 0.0]))[:2]
    return float(np.arctan2(forward_2d[1], forward_2d[0]))


def core_height(data) -> float:
    return float(data.geom("robot1_core").xpos[2])


def signed_rotation(heading_history) -> float:
    headings = np.unwrap(heading_history)
    return float(headings[-1] - headings[0])


def directional_rotation(signed_delta: float, spin_direction: str) -> float:
    if spin_direction == "right":
        return max(0.0, -signed_delta)
    if spin_direction == "left":
        return max(0.0, signed_delta)
    return abs(signed_delta)


def rotation_reward(
    heading_history,
    xy_history,
    z_history,
    z_threshold: float,
    spin_direction: str,
) -> float:
    if min(z_history) < z_threshold:
        return 0.0

    total_rotation = directional_rotation(
        signed_rotation(heading_history),
        spin_direction,
    )
    displacement = np.linalg.norm(
        np.array(xy_history[-1]) - np.array(xy_history[0])
    )
    return float(total_rotation / (1.0 + displacement))


def fitness_rotation(
    heading_history,
    xy_history,
    z_history,
    z_threshold: float,
    spin_direction: str,
) -> float:
    # Nevergrad minimizes; a fallen run gets zero reward, while successful spins
    # become negative scores so larger rotation is still preferred.
    return -rotation_reward(
        heading_history,
        xy_history,
        z_history,
        z_threshold,
        spin_direction,
    )


def self_contact_penalty(config: SpinConfig, validation: SpinValidation) -> float:
    contact_fraction = total_self_contact_frames(validation.self_contacts) / max(1, validation.z_history.size)
    return float(config.collision_weight * contact_fraction)


def run_spin_episode(
    model,
    data,
    network: CPGGaitController,
    config: SpinConfig,
    rng: np.random.Generator | None = None,
    collect_contacts: bool = False,
) -> SpinValidation:
    mujoco.mj_forward(model, data)
    network.reset_hidden()
    current_action = np.zeros(model.nu, dtype=np.float32)

    initial_pos = np.asarray(data.qpos[:3].copy(), dtype=np.float32)
    last_pos = initial_pos.copy()
    initial_roll, initial_pitch = quat_to_roll_pitch(data.qpos[3:7].copy())
    initial_core_z = core_height(data)
    min_z = initial_core_z
    max_abs_roll = abs(initial_roll)
    max_abs_pitch = abs(initial_pitch)
    path_length = 0.0
    xy_history = [(float(initial_pos[0]), float(initial_pos[1]))]
    heading_history = [core_heading(data)]
    z_history = [initial_core_z]
    fell_by_z = min_z < config.fall_z_threshold
    fell_by_tilt = (
        config.use_tilt_fall
        and max(max_abs_roll, max_abs_pitch) > config.fall_tilt_threshold_rad
    )
    fall_time = 0.0 if fell_by_z or fell_by_tilt else None
    step = 0
    contact_accumulator = SelfContactAccumulator() if collect_contacts else None

    while data.time < config.duration:
        if step % config.control_step_freq == 0:
            current_action = sanitize_action(
                network.forward(config.training_turn, config.training_speed),
                model,
            )

        data.ctrl[:] = apply_action_noise(current_action, model, config, rng) if rng else current_action
        mujoco.mj_step(model, data)
        if contact_accumulator is not None:
            contact_accumulator.update(model, data)
        step += 1

        current_pos = np.asarray(data.qpos[:3].copy(), dtype=np.float32)
        current_roll, current_pitch = quat_to_roll_pitch(data.qpos[3:7].copy())
        current_core_z = core_height(data)
        path_length += float(np.linalg.norm(current_pos - last_pos))
        last_pos = current_pos
        min_z = min(min_z, current_core_z)
        max_abs_roll = max(max_abs_roll, abs(current_roll))
        max_abs_pitch = max(max_abs_pitch, abs(current_pitch))
        current_fell_by_z = current_core_z < config.fall_z_threshold
        current_fell_by_tilt = (
            config.use_tilt_fall
            and max(abs(current_roll), abs(current_pitch)) > config.fall_tilt_threshold_rad
        )
        if fall_time is None and (current_fell_by_z or current_fell_by_tilt):
            fall_time = float(data.time)
        fell_by_z = fell_by_z or current_fell_by_z
        fell_by_tilt = fell_by_tilt or current_fell_by_tilt
        xy_history.append((float(current_pos[0]), float(current_pos[1])))
        heading_history.append(core_heading(data))
        z_history.append(current_core_z)

    xy = np.asarray(xy_history, dtype=np.float32)
    headings = np.asarray(heading_history, dtype=np.float32)
    core_z = np.asarray(z_history, dtype=np.float32)
    if fell_by_z or fell_by_tilt:
        survival = np.clip((fall_time or 0.0) / config.duration, 0.0, 1.0)
        score = float(0.5 * (1.0 - survival))
    else:
        score = float(
            fitness_rotation(
                headings,
                xy,
                core_z,
                config.fall_z_threshold,
                config.spin_direction,
            )
        )
    rotation_delta = signed_rotation(headings)
    total_rotation = directional_rotation(rotation_delta, config.spin_direction)
    drift = float(np.linalg.norm(xy[-1] - xy[0]))
    return SpinValidation(
        reward=-score,
        score=score,
        total_rotation=total_rotation,
        drift=drift,
        path_length=path_length,
        min_z=min_z,
        max_abs_roll=max_abs_roll,
        max_abs_pitch=max_abs_pitch,
        fell=fell_by_z or fell_by_tilt,
        fell_by_z=fell_by_z,
        fell_by_tilt=fell_by_tilt,
        fall_time=fall_time,
        final_pos=np.asarray(data.qpos[:3].copy(), dtype=np.float32),
        xy_history=xy,
        heading_history=headings,
        z_history=core_z,
        self_contacts=[] if contact_accumulator is None else contact_accumulator.summaries(),
    )


def init_worker_env(config: SpinConfig) -> None:
    if _SPIN_ENV:
        return
    import torch

    torch.set_num_threads(1)
    worker_seed = (config.seed + os.getpid()) % (2**32 - 1)
    seed_everything(worker_seed)
    model, data = build_training_model()
    dr_state = capture_domain_randomization_state(model)
    rng = np.random.default_rng(worker_seed)
    network = CPGGaitController(
        model.nu,
        dt=model.opt.timestep * config.control_step_freq,
        seed=config.seed,
    )
    _SPIN_ENV.update({"model": model, "data": data, "network": network, "dr_state": dr_state, "rng": rng})


def spin_score(vector, config: SpinConfig) -> float:
    init_worker_env(config)
    model = _SPIN_ENV["model"]
    data = _SPIN_ENV["data"]
    network = _SPIN_ENV["network"]
    dr_state = _SPIN_ENV["dr_state"]
    rng = _SPIN_ENV["rng"]
    set_cpg_vector(network, vector)

    total = 0.0
    for _ in range(config.eval_repeats):
        mujoco.mj_resetData(model, data)
        apply_domain_randomization(model, dr_state, config, rng)
        mujoco.mj_setConst(model, data)
        validation = run_spin_episode(
            model,
            data,
            network,
            config,
            rng,
            collect_contacts=config.collision_weight > 0.0,
        )
        total += validation.score + self_contact_penalty(config, validation)
    return total / config.eval_repeats


@ray.remote
def _evaluate_candidate(vector, config: SpinConfig) -> float:
    return spin_score(vector, config)


def validate_spin(vector, config: SpinConfig) -> SpinValidation:
    model, data = build_training_model()
    network = CPGGaitController(
        model.nu,
        dt=model.opt.timestep * config.control_step_freq,
        seed=config.seed,
    )
    set_cpg_vector(network, vector)
    mujoco.mj_resetData(model, data)
    return run_spin_episode(model, data, network, config, collect_contacts=True)


def train_spin(config: SpinConfig, console: Console) -> SpinResult:
    seed_everything(config.seed)
    setup = initialize_training(config, console)

    if not ray.is_initialized():
        ray.init(
            num_cpus=config.num_actors,
            ignore_reinit_error=True,
            runtime_env=config.ray_runtime_env,
        )
        console.log("Ray initialized", style="green")
    console.log(f"Evaluating candidates with {config.num_actors} Ray actors.")
    show_startup_summary(config, setup, console)

    optimizer = NevergradCMAESOptimizer(
        initial_guess=setup.initial_guess,
        sigma=config.sigma,
        population=setup.pop_size,
        generations=config.budget,
    )

    history = []
    global_history = []
    reward_history = []
    global_reward_history = []
    best_score = float("inf")
    best_vector = setup.initial_guess.copy()

    console.rule("[bold cyan]Training")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("pop reward [bold]{task.fields[pop_reward]}"),
        TextColumn("global reward [bold green]{task.fields[global_reward]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Evolving spin",
            total=config.budget,
            pop_reward="n/a",
            global_reward="n/a",
        )
        for gen in range(config.budget):
            candidates = optimizer.ask()
            fitness_futures = [
                _evaluate_candidate.remote(candidate, config)
                for candidate in candidates
            ]
            scores = [float(value) for value in ray.get(fitness_futures)]
            optimizer.tell(candidates, scores)

            pop_best_idx = int(np.argmin(scores))
            pop_best_score = float(scores[pop_best_idx])
            if pop_best_score < best_score:
                best_score = pop_best_score
                best_vector = np.asarray(candidates[pop_best_idx], dtype=np.float32).copy()

            history.append(pop_best_score)
            global_history.append(best_score)
            reward_history.append(-pop_best_score)
            global_reward_history.append(-best_score)
            progress.update(
                task,
                advance=1,
                pop_reward=f"{-pop_best_score:.4f}",
                global_reward=f"{-best_score:.4f}",
            )
            if gen % 10 == 0:
                console.log(
                    f"Gen {gen + 1}/{config.budget} | "
                    f"Pop score: {pop_best_score:.4f} | "
                    f"Pop reward: {-pop_best_score:.4f} | "
                    f"Global reward: {-best_score:.4f}"
                )
            if (gen + 1) % 50 == 0:
                ckpt_gen = gen + 1
                ckpt_path = config.checkpoint_dir / f"spin_ckpt_{config.run_id}_gen{ckpt_gen}.npy"
                np.save(str(ckpt_path), best_vector)
                console.log(f"Checkpoint saved -> {ckpt_path}", style="green")

    best_weights = optimizer.recommendation()
    recommended_score = spin_score(best_weights, config)
    if recommended_score > best_score:
        best_weights = best_vector
    else:
        best_score = float(recommended_score)

    validation = validate_spin(best_weights, config)
    return SpinResult(
        best_weights=best_weights,
        best_score=best_score,
        best_reward=-best_score,
        history=history,
        global_history=global_history,
        reward_history=reward_history,
        global_reward_history=global_reward_history,
        validation=validation,
        num_joints=setup.num_joints,
        dt=setup.dt,
        n_params=setup.n_params,
        pop_size=setup.pop_size,
    )


def save_validation_metrics(path: Path, validation: SpinValidation) -> None:
    np.savez(
        str(path),
        reward=validation.reward,
        score=validation.score,
        total_rotation=validation.total_rotation,
        drift=validation.drift,
        path_length=validation.path_length,
        min_z=validation.min_z,
        max_abs_roll=validation.max_abs_roll,
        max_abs_pitch=validation.max_abs_pitch,
        fell=validation.fell,
        fell_by_z=validation.fell_by_z,
        fell_by_tilt=validation.fell_by_tilt,
        fall_time=np.nan if validation.fall_time is None else validation.fall_time,
        self_contact_pairs=total_self_contact_pairs(validation.self_contacts),
        self_contact_frames=total_self_contact_frames(validation.self_contacts),
        max_self_contact_penetration=max_self_contact_penetration(validation.self_contacts),
        final_pos=validation.final_pos,
        xy_history=validation.xy_history,
        heading_history=validation.heading_history,
        z_history=validation.z_history,
    )


def save_trajectory_plot(path: Path, validation: SpinValidation, spin_direction: str) -> None:
    xy = validation.xy_history
    direction_titles = {
        "right": "Right-Turning Spin Gait Validation Trajectory",
        "left": "Left-Turning Spin Gait Validation Trajectory",
        "either": "Either-Direction Spin Gait Validation Trajectory",
    }
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xy[:, 0], xy[:, 1], "b-", linewidth=2, label="Robot path")
    ax.plot(xy[0, 0], xy[0, 1], "go", markersize=12, label="Start")
    ax.plot(xy[-1, 0], xy[-1, 1], "ro", markersize=10, label="End")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(direction_titles[spin_direction])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_artifacts(
    config: SpinConfig,
    result: SpinResult,
    console: Console,
    training_elapsed_seconds: float,
) -> dict:
    best_path = config.run_dir / f"spin_best_{config.run_id}.npy"
    np.save(str(best_path), result.best_weights)
    console.log(f"Saved spin weights -> {best_path}", style="green")

    validation_path = config.run_dir / f"spin_validation_{config.run_id}.npz"
    save_validation_metrics(validation_path, result.validation)
    console.log(f"Saved spin validation -> {validation_path}", style="green")

    contacts_path = config.run_dir / f"spin_self_contacts_{config.run_id}.csv"
    write_self_contact_summary_csv(contacts_path, result.validation.self_contacts)
    console.log(f"Saved self-contact summary -> {contacts_path}", style="green")

    meta_path = config.run_dir / f"spin_meta_{config.run_id}.npz"
    np.savez(
        str(meta_path),
        num_joints=result.num_joints,
        dt=result.dt,
        seed=config.seed,
        objective="orientation_rotation",
        spin_direction=config.spin_direction,
        score_sign="optimizer_minimizes_negative_total_rotation_over_drift",
        population=result.pop_size,
        requested_population=config.population,
        num_actors=config.num_actors,
        used_ray=True,
        budget=config.budget,
        duration=config.duration,
        fall_z_threshold=config.fall_z_threshold,
        fall_z_threshold_basis="robot1_core settles near 0.10m; threshold is about 65%",
        use_tilt_fall=config.use_tilt_fall,
        fall_tilt_threshold_deg=config.fall_tilt_threshold_deg,
        eval_repeats=config.eval_repeats,
        domain_randomization=config.domain_randomization,
        action_noise_std=config.action_noise_std,
        friction_scale_min=config.friction_scale_min,
        friction_scale_max=config.friction_scale_max,
        mass_scale_min=config.mass_scale_min,
        mass_scale_max=config.mass_scale_max,
        joint_strength_scale_min=config.joint_strength_scale_min,
        joint_strength_scale_max=config.joint_strength_scale_max,
        collision_weight=config.collision_weight,
        collision_penalty="collision_weight * total_self_contact_frames / episode_steps",
        fall_score="0.5 * (1 - fall_time / duration)",
        training_elapsed_seconds=training_elapsed_seconds,
        training_elapsed_minutes=training_elapsed_seconds / 60.0,
        sigma=config.sigma,
        optimizer="nevergrad.ParametrizedCMA",
        num_params=result.n_params,
        optimized_cpg_groups=np.asarray(OPTIMIZED_CPG_GROUPS),
        run_id=config.run_id,
        best_score=result.best_score,
        best_reward=result.best_reward,
        validation_reward=result.validation.reward,
        validation_total_rotation=result.validation.total_rotation,
        validation_drift=result.validation.drift,
        validation_path=str(validation_path),
        self_contact_summary=str(contacts_path),
        validation_self_contact_pairs=total_self_contact_pairs(result.validation.self_contacts),
        validation_self_contact_frames=total_self_contact_frames(result.validation.self_contacts),
        validation_max_self_contact_penetration=max_self_contact_penetration(result.validation.self_contacts),
        best_weights=str(best_path),
    )

    history_path = config.run_dir / f"spin_fitness_history_{config.run_id}.npy"
    np.save(str(history_path), np.asarray(result.history, dtype=np.float32))
    global_history_path = config.run_dir / f"spin_global_best_history_{config.run_id}.npy"
    np.save(str(global_history_path), np.asarray(result.global_history, dtype=np.float32))

    generations = np.arange(1, len(result.history) + 1, dtype=np.int32)
    evals_per_generation = result.pop_size * config.eval_repeats
    fitness_evaluations = generations * evals_per_generation
    total_evaluations = int(fitness_evaluations[-1]) if fitness_evaluations.size else 0
    direction_titles = {
        "right": "Right-Turning Spin Gait Evolution",
        "left": "Left-Turning Spin Gait Evolution",
        "either": "Either-Direction Spin Gait Evolution",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(fitness_evaluations, result.history, "b-", linewidth=1.2, label="Population best fitness")
    ax.plot(fitness_evaluations, result.global_history, "r-", linewidth=1.5, label="Global best fitness")
    ax.set_xlabel("Cumulative fitness evaluations")
    ax.set_ylabel("Fitness (minimization objective)")
    ax.set_title(direction_titles[config.spin_direction])
    ax.text(
        0.98,
        0.97,
        "\n".join([
            f"Generations: {len(result.history)}",
            f"Population: {result.pop_size}",
            f"Evaluation repeats: {config.eval_repeats}",
            f"Spin direction: {config.spin_direction}",
            f"Total evaluations: {total_evaluations:,}",
        ]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fitness_plot_path = config.run_dir / f"spin_fitness_{config.run_id}.png"
    fig.savefig(str(fitness_plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    trajectory_path = config.run_dir / f"spin_trajectory_{config.run_id}.png"
    save_trajectory_plot(trajectory_path, result.validation, config.spin_direction)
    console.log(f"Saved spin trajectory -> {trajectory_path}", style="green")

    return {
        "run_dir": config.run_dir,
        "best_path": best_path,
        "meta_path": meta_path,
        "validation_path": validation_path,
        "contacts_path": contacts_path,
        "history_path": history_path,
        "global_history_path": global_history_path,
        "fitness_plot_path": fitness_plot_path,
        "trajectory_path": trajectory_path,
    }


def record_validation_video(config: SpinConfig, artifacts: dict, console: Console) -> Path | None:
    console.rule("[bold cyan]Demo")
    console.log("Recording spin validation demo video...")
    network, _, _ = load_gait_network(artifacts["best_path"], artifacts["meta_path"])
    return record_spin_video(
        network=network,
        model_path=artifacts["best_path"],
        duration=config.duration,
        output_dir=config.video_dir,
        camera_height=config.video_camera_height,
        width=config.video_width,
        height=config.video_height,
        fps=config.video_fps,
        overlay=config.video_overlay,
        console=console,
    )


def show_final_summary(
    artifacts: dict,
    result: SpinResult,
    elapsed_minutes: float,
    console: Console,
) -> None:
    validation = result.validation
    table = Table(title="Spin Gait Run Summary", show_lines=True)
    table.add_column("Item", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Best optimizer score", f"{result.best_score:.6f}")
    table.add_row("Best spin reward", f"{result.best_reward:.6f}")
    table.add_row("Validation reward", f"{validation.reward:.6f}")
    table.add_row("Validation rewarded rotation", f"{np.rad2deg(validation.total_rotation):.2f} deg")
    table.add_row("Validation drift", f"{validation.drift:.4f}m")
    table.add_row("Validation path length", f"{validation.path_length:.4f}m")
    table.add_row("Validation min_z", f"{validation.min_z:.4f}m")
    table.add_row(
        "Validation max roll/pitch",
        f"{np.rad2deg(validation.max_abs_roll):.2f} / "
        f"{np.rad2deg(validation.max_abs_pitch):.2f} deg",
    )
    table.add_row("Validation fell", str(validation.fell))
    table.add_row("Validation fall reason", f"z={validation.fell_by_z}, tilt={validation.fell_by_tilt}")
    table.add_row(
        "Validation fall time",
        "n/a" if validation.fall_time is None else f"{validation.fall_time:.3f}s",
    )
    table.add_row("Elapsed", f"{elapsed_minutes:.2f} minutes")
    table.add_row("Run dir", str(artifacts["run_dir"]))
    table.add_row("Model", str(artifacts["best_path"]))
    table.add_row("Metadata", str(artifacts["meta_path"]))
    table.add_row("Validation metrics", str(artifacts["validation_path"]))
    table.add_row("Self-contact summary", str(artifacts["contacts_path"]))
    table.add_row("Population history", str(artifacts["history_path"]))
    table.add_row("Global history", str(artifacts["global_history_path"]))
    table.add_row("Fitness plot", str(artifacts["fitness_plot_path"]))
    table.add_row("Trajectory plot", str(artifacts["trajectory_path"]))
    if "video_path" in artifacts:
        table.add_row("Video", str(artifacts["video_path"]))
    console.print(table)


def main() -> None:
    install()
    console = Console(force_terminal=True, color_system="truecolor")
    args = parse_args()
    config = build_config(args)
    seed_everything(config.seed)
    ensure_run_dirs(config)

    run_start = time.time()
    train_start = time.time()
    result = train_spin(config, console)
    training_elapsed_seconds = time.time() - train_start
    artifacts = save_training_artifacts(
        config,
        result,
        console,
        training_elapsed_seconds=training_elapsed_seconds,
    )
    if config.record_video:
        video_path = record_validation_video(config, artifacts, console)
        if video_path is not None:
            artifacts["video_path"] = video_path
    elapsed_minutes = (time.time() - run_start) / 60
    show_final_summary(artifacts, result, elapsed_minutes, console)


if __name__ == "__main__":
    main()
