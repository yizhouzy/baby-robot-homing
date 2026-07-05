"""Evaluation task interfaces for robot-control experiments."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from robot_control.config_gait import GaitConfig
from robot_control.controllers import RobotController
from robot_control.evaluation import (
    build_training_model,
    evaluate_targets,
    gait_fitness,
    summarize_validation,
)


class EvaluationTask(Protocol):
    """Common task shape for optimization experiments."""

    def build_scene(self):
        ...

    def evaluate(self, controller: RobotController, vector: np.ndarray) -> float:
        ...

    def validate(self, controller: RobotController, vector: np.ndarray) -> dict:
        ...


class ForwardGaitTask:
    """Current Stage 1 task: learn open-loop forward locomotion."""

    def __init__(self, config: GaitConfig) -> None:
        self.config = config

    def build_scene(self):
        return build_training_model()

    def evaluate(self, controller: RobotController, vector: np.ndarray) -> float:
        return gait_fitness(vector, self.config)

    def validate(self, controller: RobotController, vector: np.ndarray) -> dict:
        rows = evaluate_targets(vector, self.config)
        return {
            "rows": rows,
            "summary": summarize_validation(rows),
        }
