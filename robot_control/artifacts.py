"""Artifact saving, discovery, and Rich summaries for gait experiments."""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from rich.panel import Panel
from rich.table import Table

from robot_control.config_gait import EXCLUDED_CPG_GROUPS, GaitConfig, OPTIMIZED_CPG_GROUPS, RESULTS_DIR
from robot_control.evaluation import TargetEvaluation
from robot_control.training import TrainingResult, TrainingSetup


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FORWARD_MODEL = (
    PROJECT_ROOT
    / "results/gait_cpg"
    / "20260627_234215_113631_seed43_DR"
    / "gait_best_20260627_234215_113631_seed43_DR.npy"
)
DEFAULT_FORWARD_META = (
    PROJECT_ROOT
    / "results/gait_cpg"
    / "20260627_234215_113631_seed43_DR"
    / "gait_meta_20260627_234215_113631_seed43_DR.npz"
)
DEFAULT_LEFT_MODEL = (
    PROJECT_ROOT
    / "results/left_cpg"
    / "20260627_seed41_left"
    / "spin_best_20260627_180547_744047_seed41_DR.npy"
)
DEFAULT_LEFT_META = (
    PROJECT_ROOT
    / "results/left_cpg"
    / "20260627_seed41_left"
    / "spin_meta_20260627_180547_744047_seed41_DR.npz"
)
DEFAULT_RIGHT_MODEL = (
    PROJECT_ROOT
    / "results/right_cpg"
    / "20260627_seed43_right"
    / "spin_best_20260627_200954_521575_seed43_DR.npy"
)
DEFAULT_RIGHT_META = (
    PROJECT_ROOT
    / "results/right_cpg"
    / "20260627_seed43_right"
    / "spin_meta_20260627_200954_521575_seed43_DR.npz"
)

DEFAULT_GAIT_MODELS = {
    "forward": (DEFAULT_FORWARD_MODEL, DEFAULT_FORWARD_META),
    "left": (DEFAULT_LEFT_MODEL, DEFAULT_LEFT_META),
    "right": (DEFAULT_RIGHT_MODEL, DEFAULT_RIGHT_META),
}


def ensure_run_dirs(config: GaitConfig) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=False)
    config.checkpoint_dir.mkdir(exist_ok=True)
    config.video_dir.mkdir(exist_ok=True)


def save_validation_metrics(path: Path, rows: list[TargetEvaluation]) -> None:
    np.savez(
        str(path),
        target_positions=np.asarray([row.target_pos for row in rows], dtype=np.float32),
        scores=np.asarray([row.score for row in rows], dtype=np.float32),
        initial_positions=np.asarray([row.initial_pos for row in rows], dtype=np.float32),
        final_positions=np.asarray([row.final_pos for row in rows], dtype=np.float32),
        final_z=np.asarray([row.final_z for row in rows], dtype=np.float32),
        path_length=np.asarray([row.metrics.path_length for row in rows], dtype=np.float32),
        min_distance_to_target=np.asarray(
            [row.metrics.min_distance_to_target for row in rows], dtype=np.float32),
        time_to_target=np.asarray(
            [
                np.nan if row.metrics.time_to_target is None else row.metrics.time_to_target
                for row in rows
            ],
            dtype=np.float32,
        ),
        min_z=np.asarray([row.metrics.min_z for row in rows], dtype=np.float32),
        max_abs_roll=np.asarray([row.metrics.max_abs_roll for row in rows], dtype=np.float32),
        max_abs_pitch=np.asarray([row.metrics.max_abs_pitch for row in rows], dtype=np.float32),
        fell=np.asarray([row.metrics.fell for row in rows], dtype=bool),
        fell_by_z=np.asarray([row.metrics.fell_by_z for row in rows], dtype=bool),
        fell_by_tilt=np.asarray([row.metrics.fell_by_tilt for row in rows], dtype=bool),
    )


def save_training_artifacts(
    config: GaitConfig,
    result: TrainingResult,
    console,
    training_elapsed_seconds: float,
) -> dict:
    best_path = config.run_dir / f"gait_best_{config.run_id}.npy"
    np.save(str(best_path), result.best_weights)
    console.log(f"Saved gait weights -> {best_path}", style="green")

    stability_path = config.run_dir / f"gait_stability_metrics_{config.run_id}.npz"
    save_validation_metrics(stability_path, result.validation_rows)
    console.log(f"Saved stability metrics -> {stability_path}", style="green")

    meta_path = config.run_dir / f"gait_meta_{config.run_id}.npz"
    np.savez(
        str(meta_path),
        num_joints=result.num_joints,
        dt=result.dt,
        seed=config.seed,
        fitness=config.fitness,
        reach_radius=config.reach_radius,
        population=result.pop_size,
        requested_population=config.population,
        num_actors=config.num_actors,
        used_ray=True,
        budget=config.budget,
        duration=config.duration,
        training_elapsed_seconds=training_elapsed_seconds,
        training_elapsed_minutes=training_elapsed_seconds / 60.0,
        sigma=config.sigma,
        fall_z_threshold=config.fall_z_threshold,
        fall_tilt_threshold_deg=config.fall_tilt_threshold_deg,
        use_tilt_fall=config.use_tilt_fall,
        eval_repeats=config.eval_repeats,
        domain_randomization=config.domain_randomization,
        action_noise_std=config.action_noise_std,
        friction_scale_min=config.friction_scale_min,
        friction_scale_max=config.friction_scale_max,
        mass_scale_min=config.mass_scale_min,
        mass_scale_max=config.mass_scale_max,
        joint_strength_scale_min=config.joint_strength_scale_min,
        joint_strength_scale_max=config.joint_strength_scale_max,
        optimizer="nevergrad.ParametrizedCMA",
        target_positions=config.target_positions_array,
        num_params=result.n_params,
        optimized_cpg_groups=np.asarray(OPTIMIZED_CPG_GROUPS),
        excluded_cpg_groups=np.asarray(EXCLUDED_CPG_GROUPS),
        run_id=config.run_id,
        stability_metrics=str(stability_path),
        validation_fall_rate=result.validation_summary["fall_rate"],
        validation_worst_min_z=result.validation_summary["worst_min_z"],
        validation_max_abs_roll=result.validation_summary["max_abs_roll"],
        validation_max_abs_pitch=result.validation_summary["max_abs_pitch"],
        best_weights=str(best_path),
    )

    history_path = config.run_dir / f"gait_fitness_history_{config.run_id}.npy"
    np.save(str(history_path), np.asarray(result.history, dtype=np.float32))
    global_history_path = config.run_dir / f"gait_global_best_history_{config.run_id}.npy"
    np.save(str(global_history_path), np.asarray(result.global_history, dtype=np.float32))

    generations = np.arange(1, len(result.history) + 1, dtype=np.int32)
    evals_per_generation = result.pop_size * config.eval_repeats * len(config.target_positions)
    fitness_evaluations = generations * evals_per_generation
    total_evaluations = int(fitness_evaluations[-1]) if fitness_evaluations.size else 0

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(fitness_evaluations, result.history, "b-", linewidth=1.2, label="Population best")
    ax.plot(fitness_evaluations, result.global_history, "r-", linewidth=1.5, label="Global best")
    ax.set_xlabel("Cumulative fitness evaluations")
    ax.set_ylabel("Fitness (minimization objective)")
    ax.set_title("Forward Gait Evolution")
    ax.text(
        0.98,
        0.97,
        "\n".join([
            f"Generations: {len(result.history)}",
            f"Population: {result.pop_size}",
            f"Evaluation repeats: {config.eval_repeats}",
            f"Targets: {len(config.target_positions)}",
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
    fitness_plot_path = config.run_dir / f"gait_fitness_{config.run_id}.png"
    fig.savefig(str(fitness_plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "run_dir": config.run_dir,
        "best_path": best_path,
        "meta_path": meta_path,
        "fitness_plot_path": fitness_plot_path,
        "history_path": history_path,
        "global_history_path": global_history_path,
        "stability_path": stability_path,
        "validation_summary": result.validation_summary,
        "best_eval": result.best_eval,
    }


def show_startup_summary(config: GaitConfig, result: TrainingSetup, console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Optimizer", "nevergrad.ParametrizedCMA")
    table.add_row("Budget", f"{config.budget} generations")
    table.add_row("Population", f"{result.pop_size} (requested {config.population})")
    table.add_row("Ray actors", str(config.num_actors))
    table.add_row("Sigma", f"{config.sigma:g}")
    table.add_row("Duration", f"{config.duration:g}s per target")
    table.add_row("Seed", str(config.seed))
    table.add_row("Fitness", config.fitness)
    table.add_row("Reach radius", f"{config.reach_radius:g}m")
    table.add_row("Fall z threshold", f"{config.fall_z_threshold:g}m")
    table.add_row("Tilt fall check", "on" if config.use_tilt_fall else "tracked only")
    table.add_row("Tilt threshold", f"{config.fall_tilt_threshold_deg:g} deg")
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
    table.add_row("Targets", f"{len(config.target_positions)} forward ({config.format_targets()})")
    table.add_row("CPG outputs", str(result.num_joints))
    table.add_row("CPG params", f"{result.n_params} optimized")
    table.add_row("Optimized groups", ", ".join(OPTIMIZED_CPG_GROUPS))
    table.add_row("Excluded groups", ", ".join(EXCLUDED_CPG_GROUPS))
    table.add_row("Run dir", str(config.run_dir))
    console.print(Panel(table, title="Stage 1 CPG Gait Evolution", border_style="cyan"))


def show_final_summary(artifacts: dict, best_eval: float, elapsed_minutes: float, console) -> None:
    table = Table(title="Stage 1 Run Summary", show_lines=True)
    table.add_column("Item", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Best fitness", f"{best_eval:.6f}")
    table.add_row("Elapsed", f"{elapsed_minutes:.2f} minutes")
    table.add_row("Run dir", str(artifacts["run_dir"]))
    table.add_row("Model", str(artifacts["best_path"]))
    table.add_row("Metadata", str(artifacts["meta_path"]))
    table.add_row("Stability metrics", str(artifacts["stability_path"]))
    summary = artifacts["validation_summary"]
    table.add_row("Validation fall rate", f"{summary['fall_rate']:.2%}")
    table.add_row("Validation worst min_z", f"{summary['worst_min_z']:.4f}m")
    table.add_row(
        "Validation max roll/pitch",
        f"{np.rad2deg(summary['max_abs_roll']):.2f} / "
        f"{np.rad2deg(summary['max_abs_pitch']):.2f} deg",
    )
    table.add_row("Population history", str(artifacts["history_path"]))
    table.add_row("Global history", str(artifacts["global_history_path"]))
    table.add_row("Fitness plot", str(artifacts["fitness_plot_path"]))
    console.print(table)


def find_latest_best_model(results_dir: Path = RESULTS_DIR) -> Path:
    run_dirs = sorted(
        [path for path in Path(results_dir).glob("*") if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    for run_dir in reversed(run_dirs):
        run_id = run_dir.name
        preferred = run_dir / f"gait_best_{run_id}.npy"
        if preferred.exists():
            return preferred
        candidates = sorted(run_dir.glob("gait_best_*.npy"))
        if candidates:
            return candidates[-1]

    legacy = Path(results_dir) / "gait_best.npy"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"No gait_best checkpoint found under {results_dir}.")


def resolve_model_path(model_path: str | Path, results_dir: Path = RESULTS_DIR) -> Path:
    path = Path(model_path)
    if path.exists():
        return path

    under_results = Path(results_dir) / path
    if under_results.exists():
        return under_results

    matches = sorted(Path(results_dir).glob(f"*/checkpoints/{path.name}"))
    matches.extend(sorted(Path(results_dir).glob(f"*/{path.name}")))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"{model_path} matched multiple checkpoints under {results_dir}; "
            "pass the full path."
        )
    raise FileNotFoundError(f"Model checkpoint not found: {model_path}")


def extract_run_id(model_path: Path) -> str:
    stem = Path(model_path).stem
    if stem.startswith("gait_best_"):
        return stem.removeprefix("gait_best_")

    match = re.match(r"gait_ckpt_(.+)_gen\d+$", stem)
    if match:
        return match.group(1)

    return Path(model_path).parent.name


def infer_run_dir(model_path: Path) -> Path:
    model_path = Path(model_path)
    if model_path.parent.name == "checkpoints":
        return model_path.parent.parent
    return model_path.parent


def infer_meta_path(
    model_path: Path,
    requested_meta_path: str | Path | None = None,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    if requested_meta_path is not None:
        return Path(requested_meta_path)

    run_dir = infer_run_dir(model_path)
    run_id = extract_run_id(model_path)
    preferred = run_dir / f"gait_meta_{run_id}.npz"
    if preferred.exists():
        return preferred

    candidates = sorted(run_dir.glob("gait_meta_*.npz"))
    if len(candidates) == 1:
        return candidates[0]

    legacy = Path(results_dir) / "gait_meta.npz"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"No matching gait metadata found for {model_path}.")


def infer_checkpoint_meta_path(model_path: Path) -> Path:
    """Infer gait or spin metadata path from a saved checkpoint filename."""
    stem = Path(model_path).stem
    if stem.startswith("gait_best_"):
        return Path(model_path).with_name(f"gait_meta_{stem.removeprefix('gait_best_')}.npz")
    if stem.startswith("spin_best_"):
        return Path(model_path).with_name(f"spin_meta_{stem.removeprefix('spin_best_')}.npz")
    return infer_meta_path(Path(model_path))


def resolve_existing_path(path: Path, suffix: str) -> Path:
    """Return the existing path, accepting an omitted suffix when present."""
    if path.exists():
        return path
    if path.suffix == "":
        suffixed = path.with_suffix(suffix)
        if suffixed.exists():
            return suffixed
    return path


def demo_output_paths(model_path: Path, output_dir: str | Path | None = None):
    run_dir = infer_run_dir(model_path)
    run_id = extract_run_id(model_path)
    video_dir = Path(output_dir) if output_dir is not None else run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_name = f"gait_demo_{run_id}"
    trajectory_path = video_dir / f"gait_trajectory_{run_id}.png"
    if output_dir is None:
        trajectory_path = run_dir / f"gait_trajectory_{run_id}.png"
    return run_id, video_dir, video_name, trajectory_path
