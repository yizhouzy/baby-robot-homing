"""Simulation evaluation and scoring for CPG gait candidates."""
from __future__ import annotations

from dataclasses import dataclass
import os
import random

import mujoco
import numpy as np
import torch

from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.targeted_locomotion import (
    distance_to_target,
    fitness_delta_distance,
    fitness_direct_path,
    fitness_speed_to_target,
    fitness_survival_and_locomotion,
)
from blocks.baby_robot import baby_robot

from robot_control.config_gait import GaitConfig
from robot_control.controllers import CPGGaitController, sanitize_action, set_cpg_vector
from robot_control.domain_randomization import (
    apply_action_noise,
    apply_domain_randomization,
    capture_domain_randomization_state,
)


@dataclass
class EpisodeMetrics:
    path_length: float
    min_distance_to_target: float
    time_to_target: float | None
    min_z: float
    max_abs_roll: float
    max_abs_pitch: float
    fell: bool
    fell_by_z: bool
    fell_by_tilt: bool


@dataclass
class EpisodeTrace:
    initial_pos: np.ndarray
    final_pos: np.ndarray
    final_z: float
    times: np.ndarray
    positions: np.ndarray
    path_length: float
    min_z: float
    max_abs_roll: float
    max_abs_pitch: float
    fell: bool
    fell_by_z: bool
    fell_by_tilt: bool


@dataclass
class TargetEvaluation:
    target_pos: np.ndarray
    score: float
    initial_pos: np.ndarray
    final_pos: np.ndarray
    final_z: float
    metrics: EpisodeMetrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def quat_to_roll_pitch(quat) -> tuple[float, float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return float(roll), float(pitch)


def build_training_model():
    world = SimpleFlatWorld()
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def run_open_loop_episode(
    model,
    data,
    network: CPGGaitController,
    config: GaitConfig,
    rng: np.random.Generator | None = None,
) -> EpisodeTrace:
    network.reset_hidden()
    current_action = np.zeros(model.nu)

    initial_pos = np.array(data.qpos[:3].copy())
    initial_roll, initial_pitch = quat_to_roll_pitch(data.qpos[3:7].copy())
    last_pos = initial_pos.copy()
    total_path_length = 0.0
    min_z = float(initial_pos[2])
    max_abs_roll = abs(initial_roll)
    max_abs_pitch = abs(initial_pitch)
    times = [float(data.time)]
    positions = [initial_pos.copy()]
    fell_by_z = min_z < config.fall_z_threshold
    fell_by_tilt = (
        config.use_tilt_fall
        and max(max_abs_roll, max_abs_pitch) > config.fall_tilt_threshold_rad
    )
    step = 0

    while data.time < config.duration:
        if step % config.control_step_freq == 0:
            action = network.forward(config.training_turn, config.training_speed)
            current_action = sanitize_action(action, model)

        data.ctrl[:] = apply_action_noise(current_action, model, config, rng) if rng else current_action
        mujoco.mj_step(model, data)
        step += 1

        current_pos = np.array(data.qpos[:3].copy())
        current_roll, current_pitch = quat_to_roll_pitch(data.qpos[3:7].copy())
        total_path_length += float(np.linalg.norm(current_pos - last_pos))
        last_pos = current_pos
        times.append(float(data.time))
        positions.append(current_pos.copy())

        min_z = min(min_z, float(current_pos[2]))
        max_abs_roll = max(max_abs_roll, abs(current_roll))
        max_abs_pitch = max(max_abs_pitch, abs(current_pitch))
        fell_by_z = fell_by_z or min_z < config.fall_z_threshold
        fell_by_tilt = fell_by_tilt or (
            config.use_tilt_fall
            and max(abs(current_roll), abs(current_pitch)) > config.fall_tilt_threshold_rad
        )

    final_pos = np.array(data.qpos[:3].copy())
    final_z = float(data.qpos[2])
    return EpisodeTrace(
        initial_pos=initial_pos,
        final_pos=final_pos,
        final_z=final_z,
        times=np.asarray(times, dtype=np.float32),
        positions=np.asarray(positions, dtype=np.float32),
        path_length=total_path_length,
        min_z=min_z,
        max_abs_roll=max_abs_roll,
        max_abs_pitch=max_abs_pitch,
        fell=fell_by_z or fell_by_tilt,
        fell_by_z=fell_by_z,
        fell_by_tilt=fell_by_tilt,
    )


def metrics_for_target(trace: EpisodeTrace, config: GaitConfig, target_pos) -> EpisodeMetrics:
    target_arr = np.asarray(target_pos)
    planar_distances = np.linalg.norm(trace.positions[:, :2] - target_arr[:2], axis=1)
    reached_indices = np.flatnonzero(planar_distances <= config.reach_radius)
    time_to_target = None
    if reached_indices.size:
        time_to_target = float(trace.times[int(reached_indices[0])])

    return EpisodeMetrics(
        path_length=trace.path_length,
        min_distance_to_target=float(np.min(planar_distances)),
        time_to_target=time_to_target,
        min_z=trace.min_z,
        max_abs_roll=trace.max_abs_roll,
        max_abs_pitch=trace.max_abs_pitch,
        fell=trace.fell,
        fell_by_z=trace.fell_by_z,
        fell_by_tilt=trace.fell_by_tilt,
    )


def run_episode(model, data, network: CPGGaitController, config: GaitConfig, target_pos):
    trace = run_open_loop_episode(model, data, network, config)
    metrics = metrics_for_target(trace, config, target_pos)
    return trace.initial_pos, trace.final_pos, trace.final_z, metrics


def score_episode(config: GaitConfig, initial_pos, final_pos, final_z, target_pos, metrics: EpisodeMetrics) -> float:
    if metrics.fell:
        return 10.0

    target_arr = np.asarray(target_pos)
    if config.fitness == "delta":
        return fitness_delta_distance(initial_pos, final_pos, target_arr)
    if config.fitness == "distance":
        return distance_to_target(final_pos, target_arr)
    if config.fitness == "survival":
        return fitness_survival_and_locomotion(initial_pos, final_pos, target_arr, final_z)
    if config.fitness == "direct":
        return fitness_direct_path(initial_pos, final_pos, target_arr, metrics.path_length)
    if config.fitness == "speed":
        return fitness_speed_to_target(
            time_to_target=metrics.time_to_target,
            duration=config.duration,
            min_distance_to_target=metrics.min_distance_to_target,
        )
    return fitness_delta_distance(initial_pos, final_pos, target_arr)


_GAIT_ENV: dict = {}


def init_worker_env(config: GaitConfig):
    if _GAIT_ENV:
        return
    torch.set_num_threads(1)
    worker_seed = (config.seed + os.getpid()) % (2**32 - 1)
    seed_everything(worker_seed)
    model, data = build_training_model()
    dr_state = capture_domain_randomization_state(model)
    rng = np.random.default_rng(worker_seed)
    net = CPGGaitController(
        model.nu,
        dt=model.opt.timestep * config.control_step_freq,
        seed=config.seed,
    )
    _GAIT_ENV.update({"model": model, "data": data, "network": net, "dr_state": dr_state, "rng": rng})


def gait_fitness(vector, config: GaitConfig) -> float:
    init_worker_env(config)
    model = _GAIT_ENV["model"]
    data = _GAIT_ENV["data"]
    net = _GAIT_ENV["network"]
    dr_state = _GAIT_ENV["dr_state"]
    rng = _GAIT_ENV["rng"]
    set_cpg_vector(net, vector)

    total = 0.0
    for _ in range(config.eval_repeats):
        mujoco.mj_resetData(model, data)
        apply_domain_randomization(model, dr_state, config, rng)
        mujoco.mj_setConst(model, data)
        trace = run_open_loop_episode(model, data, net, config, rng)
        for target_pos in config.target_positions:
            metrics = metrics_for_target(trace, config, target_pos)
            total += score_episode(
                config,
                trace.initial_pos,
                trace.final_pos,
                trace.final_z,
                target_pos,
                metrics,
            )
    return total / (config.eval_repeats * len(config.target_positions))


def evaluate_targets(vector, config: GaitConfig) -> list[TargetEvaluation]:
    init_worker_env(config)
    model = _GAIT_ENV["model"]
    data = _GAIT_ENV["data"]
    net = _GAIT_ENV["network"]
    set_cpg_vector(net, vector)

    mujoco.mj_resetData(model, data)
    trace = run_open_loop_episode(model, data, net, config)
    rows = []
    for target_pos in config.target_positions:
        metrics = metrics_for_target(trace, config, target_pos)
        score = score_episode(
            config,
            trace.initial_pos,
            trace.final_pos,
            trace.final_z,
            target_pos,
            metrics,
        )
        rows.append(TargetEvaluation(
            target_pos=np.asarray(target_pos, dtype=np.float32),
            score=float(score),
            initial_pos=trace.initial_pos.astype(np.float32),
            final_pos=trace.final_pos.astype(np.float32),
            final_z=float(trace.final_z),
            metrics=metrics,
        ))
    return rows


def summarize_validation(rows: list[TargetEvaluation]) -> dict[str, float]:
    min_zs = np.asarray([row.metrics.min_z for row in rows], dtype=np.float32)
    max_rolls = np.asarray([row.metrics.max_abs_roll for row in rows], dtype=np.float32)
    max_pitchs = np.asarray([row.metrics.max_abs_pitch for row in rows], dtype=np.float32)
    fell = np.asarray([row.metrics.fell for row in rows], dtype=bool)
    return {
        "fall_rate": float(np.mean(fell)),
        "worst_min_z": float(np.min(min_zs)),
        "max_abs_roll": float(np.max(max_rolls)),
        "max_abs_pitch": float(np.max(max_pitchs)),
    }
