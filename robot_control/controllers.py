"""CPG controller and checkpoint vector helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)

from robot_control.config_gait import CONTROL_STEP_FREQ, OPTIMIZED_CPG_GROUPS


BABY_ROBOT_TURN_WEIGHTS = (-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0)
BABY_ROBOT_SEARCH_WEIGHTS = (-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0)
HA_GENE_SCALE = 0.9
B_GENE_SCALE = 1.0
LEGACY_OPTIMIZED_CPG_GROUPS = ("phase", "w", "amplitudes", "ha")


class RobotController(Protocol):
    """Common shape for controllers that can be evaluated or optimized."""

    def reset(self) -> None:
        ...

    def act(self, context) -> np.ndarray:
        ...

    def get_vector(self) -> np.ndarray:
        ...

    def set_vector(self, vector) -> None:
        ...


class CPGGaitController(nn.Module):
    """NaCPG controller with external turn/speed modulation.

    Stage 1 optimizes the autonomous gait only. The robot state is intentionally
    not an input here; later behavior layers can modulate the gait with turn and
    speed commands.
    """

    def __init__(self, num_outputs: int, dt: float = 0.02, seed: int | None = None):
        super().__init__()
        adj_dict = create_fully_connected_adjacency(num_outputs)
        self.cpg = NaCPG(adj_dict, dt=dt, seed=seed)
        turn_weights = torch.tensor(BABY_ROBOT_TURN_WEIGHTS[:num_outputs], dtype=torch.float32)
        self.register_buffer("turn_weights", turn_weights)
        for param in self.cpg.parameters():
            param.requires_grad = False

    def reset_hidden(self) -> None:
        self.cpg.reset()

    def reset(self) -> None:
        self.reset_hidden()

    def act(self, context) -> np.ndarray:
        turn = getattr(context, "turn", 0.0)
        speed = getattr(context, "speed", 1.0)
        return self.forward(turn, speed)

    def get_vector(self) -> np.ndarray:
        return get_cpg_vector(self)

    def set_vector(self, vector) -> None:
        set_cpg_vector(self, vector)

    @torch.inference_mode()
    def forward(self, turn: float, speed: float) -> np.ndarray:
        angles = self.cpg.forward(time=None)
        modulated = angles * float(speed) + self.turn_weights * float(turn) * 0.5
        modulated = torch.clamp(modulated, -torch.pi / 2, torch.pi / 2)
        return modulated.detach().numpy()


def cpg_vector_size(num_outputs: int) -> int:
    """Number of optimized CPG parameters.

    The gait search uses one value per optimized CPG group and output.
    """
    return len(OPTIMIZED_CPG_GROUPS) * num_outputs


@torch.no_grad()
def get_cpg_vector(network: CPGGaitController) -> np.ndarray:
    parts = []
    for name in OPTIMIZED_CPG_GROUPS:
        param = getattr(network.cpg, name).detach().flatten()
        if name == "ha":
            param = torch.atanh(param / HA_GENE_SCALE)
        elif name == "b":
            param = torch.atanh(param / B_GENE_SCALE)
        parts.append(param)
    return torch.cat(parts).cpu().numpy().astype(np.float32)


@torch.no_grad()
def set_cpg_vector(network: CPGGaitController, vector) -> None:
    if isinstance(vector, torch.Tensor):
        vec = vector.detach().clone().to(dtype=torch.float32)
    else:
        vec = torch.as_tensor(
            np.asarray(vector, dtype=np.float32).copy(),
            dtype=torch.float32,
        )
    expected = cpg_vector_size(network.cpg.n)
    if vec.numel() != expected:
        raise ValueError(f"CPG vector has {vec.numel()} values, expected {expected}.")

    pointer = 0
    for name in OPTIMIZED_CPG_GROUPS:
        param = getattr(network.cpg, name)
        n = param.numel()
        values = vec[pointer:pointer + n].view_as(param)
        if name == "ha":
            values = HA_GENE_SCALE * torch.tanh(values)
        elif name == "b":
            values = B_GENE_SCALE * torch.tanh(values)
        param.data[:] = values
        pointer += n


@torch.no_grad()
def set_legacy_cpg_vector(network: CPGGaitController, vector) -> None:
    """Load pre-b-offset Stage 1 checkpoints into the current CPG shape."""
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32), dtype=torch.float32)
    expected = len(LEGACY_OPTIMIZED_CPG_GROUPS) * network.cpg.n
    if vec.numel() != expected:
        raise ValueError(f"Legacy CPG vector has {vec.numel()} values, expected {expected}.")

    pointer = 0
    for name in LEGACY_OPTIMIZED_CPG_GROUPS:
        param = getattr(network.cpg, name)
        n = param.numel()
        param.data[:] = vec[pointer:pointer + n].view_as(param)
        pointer += n
    network.cpg.b.data.zero_()


@torch.no_grad()
def set_full_cpg_vector(network: CPGGaitController, vector) -> None:
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32), dtype=torch.float32)
    pointer = 0
    for param in network.cpg.parameters():
        n = param.numel()
        param.data[:] = vec[pointer:pointer + n].view_as(param)
        pointer += n
    if pointer != vec.numel():
        raise ValueError(f"Full CPG vector has {vec.numel()} values, expected {pointer}.")


def load_gait_network(
    model_path: Path,
    meta_path: Path | None,
    num_outputs: int | None = None,
    dt: float | None = None,
    seed: int | None = None,
):
    """Load a saved optimized-vector or legacy full-vector CPG checkpoint."""
    if meta_path is None:
        weights = np.load(str(model_path))
        if num_outputs is None:
            if weights.size % len(OPTIMIZED_CPG_GROUPS) == 0:
                num_outputs = weights.size // len(OPTIMIZED_CPG_GROUPS)
            elif weights.size % len(LEGACY_OPTIMIZED_CPG_GROUPS) == 0:
                num_outputs = weights.size // len(LEGACY_OPTIMIZED_CPG_GROUPS)
            else:
                raise ValueError(
                    f"{model_path} has {weights.size} values, so num_joints "
                    "cannot be inferred without metadata."
                )
        if dt is None:
            dt = 0.002 * CONTROL_STEP_FREQ
        meta = {
            "num_joints": int(num_outputs),
            "dt": float(dt),
            "seed": seed if seed is not None else 42,
        }
    else:
        meta = np.load(str(meta_path))
        num_outputs = int(meta["num_joints"])
        dt = float(meta["dt"])
        seed = int(meta["seed"]) if "seed" in meta else seed
        weights = np.load(str(model_path))

    network = CPGGaitController(num_outputs, dt=dt, seed=seed)
    optimized_count = cpg_vector_size(num_outputs)
    legacy_optimized_count = len(LEGACY_OPTIMIZED_CPG_GROUPS) * num_outputs
    full_count = sum(param.numel() for param in network.cpg.parameters())

    if weights.size == optimized_count:
        set_cpg_vector(network, weights)
        weight_format = "optimized CPG vector"
    elif weights.size == legacy_optimized_count:
        set_legacy_cpg_vector(network, weights)
        weight_format = "legacy optimized CPG vector (phase,w,amplitudes,ha; b=0)"
    elif weights.size == full_count:
        set_full_cpg_vector(network, weights)
        weight_format = "full CPG vector"
    else:
        raise ValueError(
            f"{model_path} has {weights.size} values, expected either "
            f"{optimized_count} optimized values or {full_count} full CPG values."
        )

    return network, meta, weight_format


def sanitize_action(action, model) -> np.ndarray:
    """Return a valid MuJoCo control vector, or zeros for invalid actions."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (model.nu,) or not np.all(np.isfinite(action)):
        return np.zeros(model.nu)
    return action
