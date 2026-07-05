"""Configuration for Stage 1 CPG gait evolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results/gait_cpg")
OPTIMIZED_CPG_GROUPS = ("phase", "w", "amplitudes", "ha", "b")
EXCLUDED_CPG_GROUPS = ()

# The corrected Baby robot morphology treats negative Y as forward.
TARGET_POSITIONS = (
    (0.0, -1.0, 0.1),
    (0.0, -1.5, 0.1),
    (0.0, -2.0, 0.1),
    (0.0, -2.5, 0.1),
    (0.0, -3.0, 0.1),
)

# Stage 1 learns an open-loop forward gait. Steering is handled later.
TRAINING_TURN = 0.0
TRAINING_SPEED = 1.0
CONTROL_STEP_FREQ = 25
DEFAULT_DEMO_TARGET = (0.0, -1.8, 0.1)

RAY_RUNTIME_ENV = {
    # Ray packages the local project for workers. Exclude generated artifacts so
    # old videos, plots, and checkpoints are not zipped and uploaded at startup.
    "excludes": [
        "results/",
        "__data__/",
        ".git/",
        ".venv/",
        "ariel/.venv/",
        "**/__pycache__/",
        "**/.DS_Store",
    ],
}


def make_run_id(seed: int) -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}"


@dataclass(frozen=True)
class GaitConfig:
    budget: int = 300
    population: int = 50
    duration: float = 30
    num_actors: int = 8
    seed: int = 42
    sigma: float = 0.075
    fitness: str = "speed"
    reach_radius: float = 0.15
    fall_z_threshold: float = 0.05
    fall_tilt_threshold_deg: float = 75.0
    use_tilt_fall: bool = False
    record_video: bool = True
    results_dir: Path = RESULTS_DIR
    run_id: str | None = None
    target_positions: tuple[tuple[float, float, float], ...] = TARGET_POSITIONS
    training_turn: float = TRAINING_TURN
    training_speed: float = TRAINING_SPEED
    control_step_freq: int = CONTROL_STEP_FREQ
    demo_target: tuple[float, float, float] = DEFAULT_DEMO_TARGET
    eval_repeats: int = 3
    domain_randomization: bool = True
    action_noise_std: float = 0.03
    friction_scale_min: float = 0.3
    friction_scale_max: float = 1.0
    mass_scale_min: float = 0.9
    mass_scale_max: float = 1.1
    joint_strength_scale_min: float = 0.7
    joint_strength_scale_max: float = 1.3
    ray_runtime_env: dict = field(default_factory=lambda: RAY_RUNTIME_ENV)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reach_radius", max(0.01, self.reach_radius))
        object.__setattr__(self, "fall_z_threshold", max(0.0, self.fall_z_threshold))
        object.__setattr__(self, "eval_repeats", max(1, self.eval_repeats))
        if self.run_id is None:
            run_id = make_run_id(self.seed)
            if self.domain_randomization:
                run_id = f"{run_id}_DR"
            object.__setattr__(self, "run_id", run_id)

    @property
    def fall_tilt_threshold_rad(self) -> float:
        return float(np.deg2rad(max(0.0, self.fall_tilt_threshold_deg)))

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
    def target_positions_array(self) -> np.ndarray:
        return np.asarray(self.target_positions, dtype=np.float32)

    def format_targets(self) -> str:
        return ", ".join(f"{target[1]:.1f}m" for target in self.target_positions)
