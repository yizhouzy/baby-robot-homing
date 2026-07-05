"""Domain randomization helpers for MuJoCo gait evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np


class DomainRandomizationConfig(Protocol):
    domain_randomization: bool
    action_noise_std: float
    friction_scale_min: float
    friction_scale_max: float
    mass_scale_min: float
    mass_scale_max: float
    joint_strength_scale_min: float
    joint_strength_scale_max: float


@dataclass
class DomainRandomizationState:
    geom_friction: np.ndarray
    body_mass: np.ndarray
    body_inertia: np.ndarray
    actuator_gainprm: np.ndarray
    actuator_biasprm: np.ndarray
    floor_geom_ids: np.ndarray
    body_ids: np.ndarray


def capture_domain_randomization_state(model: mujoco.MjModel) -> DomainRandomizationState:
    floor_geom_ids = np.asarray(
        [geom_id for geom_id in range(model.ngeom) if model.geom(geom_id).name == "floor"],
        dtype=np.int32,
    )
    body_ids = np.asarray(
        [body_id for body_id in range(model.nbody) if model.body_mass[body_id] > 0.0],
        dtype=np.int32,
    )
    return DomainRandomizationState(
        geom_friction=model.geom_friction.copy(),
        body_mass=model.body_mass.copy(),
        body_inertia=model.body_inertia.copy(),
        actuator_gainprm=model.actuator_gainprm.copy(),
        actuator_biasprm=model.actuator_biasprm.copy(),
        floor_geom_ids=floor_geom_ids,
        body_ids=body_ids,
    )


def apply_domain_randomization(
    model: mujoco.MjModel,
    state: DomainRandomizationState,
    config: DomainRandomizationConfig,
    rng: np.random.Generator,
) -> None:
    model.geom_friction[:] = state.geom_friction
    model.body_mass[:] = state.body_mass
    model.body_inertia[:] = state.body_inertia
    model.actuator_gainprm[:] = state.actuator_gainprm
    model.actuator_biasprm[:] = state.actuator_biasprm

    if not config.domain_randomization:
        return

    friction_scale = rng.uniform(config.friction_scale_min, config.friction_scale_max)
    model.geom_friction[state.floor_geom_ids] = state.geom_friction[state.floor_geom_ids] * friction_scale

    body_scales = rng.uniform(
        config.mass_scale_min,
        config.mass_scale_max,
        size=state.body_ids.size,
    )
    model.body_mass[state.body_ids] = state.body_mass[state.body_ids] * body_scales
    model.body_inertia[state.body_ids] = state.body_inertia[state.body_ids] * body_scales[:, None]

    strength_scales = rng.uniform(
        config.joint_strength_scale_min,
        config.joint_strength_scale_max,
        size=model.nu,
    )
    model.actuator_gainprm[:, 0] = state.actuator_gainprm[:, 0] * strength_scales
    model.actuator_biasprm[:, 1] = state.actuator_biasprm[:, 1] * strength_scales
    model.actuator_biasprm[:, 2] = state.actuator_biasprm[:, 2] * strength_scales


def apply_action_noise(
    action: np.ndarray,
    model: mujoco.MjModel,
    config: DomainRandomizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if not config.domain_randomization or config.action_noise_std <= 0.0:
        return action

    noisy_action = action + rng.normal(0.0, config.action_noise_std, size=action.shape)
    return np.clip(
        noisy_action,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
