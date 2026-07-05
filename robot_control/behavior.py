"""Reusable behavior policy and action providers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from robot_control.controllers import (
    BABY_ROBOT_SEARCH_WEIGHTS,
    CPGGaitController,
    sanitize_action,
)


BATTERY_THRESHOLD = 0.2
STATION_VISIBLE_AREA = 0.01
STATION_REACHED_AREA = 0.18


@dataclass(frozen=True)
class BehaviorDemoConfig:
    duration: float
    battery_threshold: float
    station_pos: tuple[float, float, float]
    reach_radius: float
    reach_vision_area: float
    explore_speed: float
    search_speed: float
    output_dir: Path
    timestamp: str


@dataclass(frozen=True)
class ActionContext:
    vision: list[float]
    battery: float
    mode: str = ""
    turn: float = 0.0
    speed: float = 0.0


@dataclass(frozen=True)
class BehaviorSample:
    time: float
    x: float
    y: float
    yaw: float
    battery: float
    mode: str
    distance: float
    vision_area: float
    vision_centroid: float


class ActionProvider(Protocol):
    def reset(self) -> None:
        ...

    def compute(self, model, data, context: ActionContext) -> np.ndarray:
        ...


class LearnedGaitProvider:
    def __init__(self, gait_net: CPGGaitController) -> None:
        self.gait_net = gait_net

    def reset(self) -> None:
        self.gait_net.reset_hidden()

    def compute(self, model, data, context: ActionContext) -> np.ndarray:
        action = self.gait_net.forward(context.turn, context.speed)
        return sanitize_action(action, model)


class PureSpinProvider:
    def __init__(self, search_speed: float) -> None:
        self.search_speed = search_speed

    def reset(self) -> None:
        pass

    def compute(self, model, data, context: ActionContext) -> np.ndarray:
        spin_speed = 6.5 * max(0.1, self.search_speed) / 0.4
        turn_wave = np.sin(data.time * spin_speed * np.pi) * (np.pi / 2)
        action = turn_wave * np.asarray(BABY_ROBOT_SEARCH_WEIGHTS[:model.nu])
        return sanitize_action(action, model)


class LearnedTurnSearchProvider:
    def __init__(self, gait_net: CPGGaitController, search_speed: float) -> None:
        self.gait_net = gait_net
        self.search_speed = search_speed

    def reset(self) -> None:
        self.gait_net.reset_hidden()

    def compute(self, model, data, context: ActionContext) -> np.ndarray:
        scan_speed = max(self.search_speed, 0.8)
        gait_action = self.gait_net.forward(turn=1.0, speed=scan_speed)
        spin_speed = 6.5 * max(0.1, self.search_speed) / 0.4
        turn_wave = np.sin(data.time * spin_speed * np.pi) * (np.pi / 2)
        spin_action = turn_wave * np.asarray(BABY_ROBOT_SEARCH_WEIGHTS[:model.nu])
        action = np.clip(gait_action + 0.35 * spin_action, -np.pi / 2, np.pi / 2)
        return sanitize_action(action, model)


class Stage2TurnActionProvider:
    """Bridge Stage 2 turn providers into the behavior policy provider interface."""

    def __init__(self, turn_provider) -> None:
        self.turn_provider = turn_provider

    def reset(self) -> None:
        self.turn_provider.reset()

    def compute(self, model, data, context: ActionContext) -> np.ndarray:
        return sanitize_action(self.turn_provider.compute(model, context.vision).action, model)


class GaitHomingBehaviorPolicy:
    def __init__(
        self,
        gait_provider: ActionProvider,
        search_provider: ActionProvider,
        config: BehaviorDemoConfig,
    ) -> None:
        self.gait_provider = gait_provider
        self.search_provider = search_provider
        self.config = config

    def reset(self) -> None:
        self.gait_provider.reset()
        self.search_provider.reset()

    def decide(self, vision, battery: float, model, data) -> tuple[np.ndarray, str]:
        centroid_x = vision[5]
        area = vision[6]

        if area >= self.config.reach_vision_area:
            context = ActionContext(
                vision=vision,
                battery=battery,
                mode="STOPPED",
                turn=0.0,
                speed=0.0,
            )
            return self.gait_provider.compute(model, data, context), "STOPPED"

        if battery <= self.config.battery_threshold:
            if area < STATION_VISIBLE_AREA:
                context = ActionContext(
                    vision=vision,
                    battery=battery,
                    mode="SEARCHING",
                )
                return self.search_provider.compute(model, data, context), "SEARCHING"

            context = ActionContext(
                vision=vision,
                battery=battery,
                mode="APPROACHING",
                turn=float(np.clip(-centroid_x * 0.8, -1.0, 1.0)),
                speed=float(np.clip(0.3 + area * 5.0, 0.2, 1.0)),
            )
            return self.gait_provider.compute(model, data, context), "APPROACHING"

        turn = float(0.45 * np.sin(data.time * 0.8) + 0.15 * np.sin(data.time * 1.7))
        context = ActionContext(
            vision=vision,
            battery=battery,
            mode="EXPLORE",
            turn=turn,
            speed=self.config.explore_speed,
        )
        return self.gait_provider.compute(model, data, context), "EXPLORE"


class Stage2HomingBehaviorPolicy:
    """Use the frozen gait for exploration and the shared Stage 2 turn stack for homing."""

    def __init__(
        self,
        gait_provider: ActionProvider,
        homing_provider: ActionProvider,
        config: BehaviorDemoConfig,
    ) -> None:
        self.gait_provider = gait_provider
        self.homing_provider = homing_provider
        self.config = config

    def reset(self) -> None:
        self.gait_provider.reset()
        self.homing_provider.reset()

    def decide(self, vision, battery: float, model, data) -> tuple[np.ndarray, str]:
        area = vision[6]

        if area >= self.config.reach_vision_area:
            return np.zeros(model.nu, dtype=np.float32), "STOPPED"

        if battery <= self.config.battery_threshold:
            mode = "APPROACHING" if area >= STATION_VISIBLE_AREA else "SEARCHING"
            context = ActionContext(
                vision=vision,
                battery=battery,
                mode=mode,
            )
            return self.homing_provider.compute(model, data, context), mode

        turn = float(0.45 * np.sin(data.time * 0.8) + 0.15 * np.sin(data.time * 1.7))
        context = ActionContext(
            vision=vision,
            battery=battery,
            mode="EXPLORE",
            turn=turn,
            speed=self.config.explore_speed,
        )
        return self.gait_provider.compute(model, data, context), "EXPLORE"


def distance_to_station(data, station_pos) -> float:
    return float(np.linalg.norm(np.asarray(data.qpos[:2]) - np.asarray(station_pos[:2])))
