"""Nevergrad/Ray training loop for Stage 1 CPG gait evolution."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ray
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

from robot_control.config_gait import GaitConfig
from robot_control.controllers import CPGGaitController, get_cpg_vector
from robot_control.evaluation import (
    build_training_model,
    evaluate_targets,
    gait_fitness,
    seed_everything,
    summarize_validation,
)
from robot_control.optimizers import NevergradCMAESOptimizer


@dataclass
class TrainingSetup:
    initial_guess: np.ndarray
    num_joints: int
    dt: float
    n_params: int
    pop_size: int


@dataclass
class TrainingResult:
    best_weights: np.ndarray
    best_eval: float
    history: list[float]
    global_history: list[float]
    validation_rows: list
    validation_summary: dict[str, float]
    num_joints: int
    dt: float
    n_params: int
    pop_size: int


@ray.remote
def _evaluate_candidate(vector, config: GaitConfig) -> float:
    return gait_fitness(vector, config)


def initialize_training(config: GaitConfig, console) -> TrainingSetup:
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
    return TrainingSetup(initial_guess, num_joints, dt, n_params, pop_size)


def train_gait(config: GaitConfig, console, startup_summary=None) -> TrainingResult:
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
    if startup_summary is not None:
        startup_summary(config, setup, console)

    optimizer = NevergradCMAESOptimizer(
        initial_guess=setup.initial_guess,
        sigma=config.sigma,
        population=setup.pop_size,
        generations=config.budget,
    )

    history = []
    global_history = []
    best_eval = float("inf")
    best_vector = setup.initial_guess.copy()

    console.rule("[bold cyan]Training")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("pop [bold]{task.fields[pop_best]}"),
        TextColumn("global [bold green]{task.fields[global_best]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Evolving gait",
            total=config.budget,
            pop_best="n/a",
            global_best="n/a",
        )
        for gen in range(config.budget):
            candidates = optimizer.ask()
            fitness_futures = [
                _evaluate_candidate.remote(
                    candidate,
                    config,
                )
                for candidate in candidates
            ]
            fitnesses = [float(value) for value in ray.get(fitness_futures)]
            optimizer.tell(candidates, fitnesses)

            pop_best_idx = int(np.argmin(fitnesses))
            pop_best = float(fitnesses[pop_best_idx])
            if pop_best < best_eval:
                best_eval = pop_best
                best_vector = np.asarray(
                    candidates[pop_best_idx],
                    dtype=np.float32,
                ).copy()

            history.append(pop_best)
            global_history.append(best_eval)
            progress.update(
                task,
                advance=1,
                pop_best=f"{pop_best:.4f}",
                global_best=f"{best_eval:.4f}",
            )
            if gen % 10 == 0:
                console.log(
                    f"Gen {gen + 1}/{config.budget} | "
                    f"Pop Best: {pop_best:.4f} | "
                    f"Global Best: {best_eval:.4f}"
                )
            if (gen + 1) % 50 == 0:
                ckpt_gen = gen + 1
                ckpt_path = config.checkpoint_dir / f"gait_ckpt_{config.run_id}_gen{ckpt_gen}.npy"
                np.save(str(ckpt_path), best_vector)
                console.log(f"Checkpoint saved -> {ckpt_path}", style="green")

    best_weights = optimizer.recommendation()
    recommended_eval = gait_fitness(best_weights, config)
    if recommended_eval > best_eval:
        best_weights = best_vector
    else:
        best_eval = float(recommended_eval)

    validation_rows = evaluate_targets(best_weights, config)
    validation_summary = summarize_validation(validation_rows)
    return TrainingResult(
        best_weights=best_weights,
        best_eval=best_eval,
        history=history,
        global_history=global_history,
        validation_rows=validation_rows,
        validation_summary=validation_summary,
        num_joints=setup.num_joints,
        dt=setup.dt,
        n_params=setup.n_params,
        pop_size=setup.pop_size,
    )
