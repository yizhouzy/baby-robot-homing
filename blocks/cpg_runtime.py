"""Minimal CPG runtime for the physical baby robot.

This copies only the pieces needed to run trained CPG checkpoints, so the
Raspberry Pi does not need ARIEL installed. It still uses PyTorch because the
trained controller's oscillator state and update math are PyTorch-based.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


E = 1e-9
OPTIMIZED_CPG_GROUPS = ("phase", "w", "amplitudes", "ha", "b")
LEGACY_OPTIMIZED_CPG_GROUPS = ("phase", "w", "amplitudes", "ha")
BABY_ROBOT_TURN_WEIGHTS = (-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0)
CONTROL_STEP_FREQ = 25
HA_GENE_SCALE = 0.9
B_GENE_SCALE = 1.0


def create_fully_connected_adjacency(num_nodes: int) -> dict[int, list[int]]:
    return {i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)}


class NaCPG(nn.Module):
    """Normalized asymmetric CPG runtime copied from the ARIEL controller."""

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        alpha: float = 0.1,
        dt: float = 0.01,
        hard_bounds: tuple[float, float] | None = (-torch.pi / 2, torch.pi / 2),
        h: float = 1.0,
        cf_scale: float = 10.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.adjacency_dict = adjacency_dict
        self.n = len(adjacency_dict)
        self.hard_bounds = hard_bounds
        self.clamping_error = 0.0
        if seed is not None:
            torch.manual_seed(seed)

        scale = torch.pi * 2
        self.alpha = alpha
        self.dt = dt
        self.h = h
        self.cf_scale = cf_scale
        self.phase = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        self.amplitudes = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        self.w = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        self.ha = nn.Parameter(torch.rand(self.n) * 1.0 - 0.5, requires_grad=False)
        self.b = nn.Parameter(torch.rand(self.n) * 1.0 - 0.5, requires_grad=False)
        self.register_buffer("xy", torch.zeros(self.n, 2))
        self.register_buffer("xy_dot_old", torch.zeros(self.n, 2))
        self.register_buffer("angles", torch.zeros(self.n))
        self.reset()

    def reset(self) -> None:
        self.xy.data = torch.stack(
            [torch.cos(self.phase), self.b + torch.sin(self.phase)],
            dim=1,
        )
        self.xy_dot_old.data = torch.zeros_like(self.xy_dot_old)
        self.angles.data = torch.zeros_like(self.angles)

    def forward(self, time: float | None = None) -> torch.Tensor:
        if time is not None and torch.isclose(torch.tensor(time), torch.tensor(0.0)):
            self.reset()

        with torch.inference_mode():
            r_matrix = torch.zeros(self.n, self.n, 2, 2)
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        r_matrix[i, j] = torch.eye(2)
                    else:
                        phase_diff_ij = self.phase[i] - self.phase[j]
                        cos_d_ij = torch.cos(phase_diff_ij)
                        sin_d_ij = torch.sin(phase_diff_ij)
                        r_matrix[i, j] = torch.tensor([
                            [cos_d_ij, -sin_d_ij],
                            [sin_d_ij, cos_d_ij],
                        ])

            k_matrix = torch.zeros(self.n, 2, 2)
            for i in range(self.n):
                x_dot_old, _ = self.xy_dot_old[i]
                ha_i = self.ha[i]
                w_i = self.w[i]
                xi, yi = self.xy[i]
                b_i = self.b[i]
                r2i = xi**2 + (yi - b_i) ** 2
                term_a = self.alpha * (1 - r2i)
                zeta_i = 1 - ha_i * torch.sign(x_dot_old)
                term_b = (1 / (zeta_i + E)) * w_i
                k_matrix[i] = torch.tensor([
                    [term_a, -term_b],
                    [term_b, term_a],
                ])

            angles = torch.zeros(self.n)
            xy_next = torch.zeros_like(self.xy)
            xy_dot_next = torch.zeros_like(self.xy_dot_old)
            for i, (xi, yi) in enumerate(self.xy):
                b_i = self.b[i]
                centered_state = torch.stack([xi, yi - b_i])
                term_a_vec = torch.mv(k_matrix[i], centered_state)
                term_b_vec = torch.zeros(2)
                for j in self.adjacency_dict[i]:
                    xj, yj = self.xy[j]
                    neighbor_centered = torch.stack([xj, yj - self.b[j]])
                    term_b_vec += self.h * torch.mv(r_matrix[i, j], neighbor_centered)

                xi_dot, yi_dot = term_a_vec + term_b_vec
                xi_dot_old, yi_dot_old = self.xy_dot_old[i]
                cf = self.cf_scale * torch.abs(self.w[i])
                xi_dot = torch.clamp(xi_dot, xi_dot_old - cf, xi_dot_old + cf)
                yi_dot = torch.clamp(yi_dot, yi_dot_old - cf, yi_dot_old + cf)
                xi_new = xi + (xi_dot * self.dt)
                yi_new = yi + (yi_dot * self.dt)
                xy_dot_next[i] = torch.stack([xi_dot, yi_dot])
                xy_next[i] = torch.stack([xi_new, yi_new])
                angles[i] = self.amplitudes[i] * yi_new

            self.xy_dot_old.data = xy_dot_next
            self.xy.data = xy_next
            if self.hard_bounds is not None:
                angles = torch.clamp(angles, min=self.hard_bounds[0], max=self.hard_bounds[1])
            self.angles = angles
        return self.angles.clone()


class CPGGaitRuntime(nn.Module):
    """NaCPG controller with the same turn/speed modulation used in training."""

    def __init__(self, num_outputs: int, dt: float = 0.02, seed: int | None = None):
        super().__init__()
        adj_dict = create_fully_connected_adjacency(num_outputs)
        self.cpg = NaCPG(adj_dict, dt=dt, seed=seed)
        turn_weights = torch.tensor(BABY_ROBOT_TURN_WEIGHTS[:num_outputs], dtype=torch.float32)
        self.register_buffer("turn_weights", turn_weights)

    @torch.inference_mode()
    def forward(self, turn: float, speed: float) -> np.ndarray:
        angles = self.cpg.forward(time=None)
        modulated = angles * float(speed) + self.turn_weights * float(turn) * 0.5
        modulated = torch.clamp(modulated, -torch.pi / 2, torch.pi / 2)
        return modulated.detach().numpy()


def load_gait_runtime(model_path: Path, meta_path: Path | None):
    if meta_path is None:
        weights = np.load(str(model_path))
        num_outputs = weights.size // len(OPTIMIZED_CPG_GROUPS)
        dt = 0.002 * CONTROL_STEP_FREQ
        seed = 42
        meta = {"num_joints": int(num_outputs), "dt": float(dt), "seed": seed}
    else:
        loaded_meta = np.load(str(meta_path))
        num_outputs = int(loaded_meta["num_joints"])
        dt = float(loaded_meta["dt"])
        seed = int(loaded_meta["seed"]) if "seed" in loaded_meta else 42
        weights = np.load(str(model_path))
        meta = {"num_joints": num_outputs, "dt": dt, "seed": seed}

    network = CPGGaitRuntime(num_outputs, dt=dt, seed=seed)
    optimized_count = len(OPTIMIZED_CPG_GROUPS) * num_outputs
    legacy_count = len(LEGACY_OPTIMIZED_CPG_GROUPS) * num_outputs
    full_count = sum(param.numel() for param in network.cpg.parameters())

    if weights.size == optimized_count:
        set_optimized_vector(network, weights)
        weight_format = "optimized CPG vector"
    elif weights.size == legacy_count:
        set_legacy_vector(network, weights)
        weight_format = "legacy optimized CPG vector"
    elif weights.size == full_count:
        set_full_vector(network, weights)
        weight_format = "full CPG vector"
    else:
        raise ValueError(
            f"{model_path} has {weights.size} values, expected {optimized_count} or {full_count}.",
        )

    return network, meta, weight_format


@torch.no_grad()
def set_optimized_vector(network: CPGGaitRuntime, vector) -> None:
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32).copy(), dtype=torch.float32)
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
def set_legacy_vector(network: CPGGaitRuntime, vector) -> None:
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32).copy(), dtype=torch.float32)
    pointer = 0
    for name in LEGACY_OPTIMIZED_CPG_GROUPS:
        param = getattr(network.cpg, name)
        n = param.numel()
        param.data[:] = vec[pointer:pointer + n].view_as(param)
        pointer += n
    network.cpg.b.data.zero_()


@torch.no_grad()
def set_full_vector(network: CPGGaitRuntime, vector) -> None:
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32).copy(), dtype=torch.float32)
    pointer = 0
    for param in network.cpg.parameters():
        n = param.numel()
        param.data[:] = vec[pointer:pointer + n].view_as(param)
        pointer += n
