"""Stage 2 camera-based turning helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_VISIBILITY_THRESHOLD = 0.01
DEFAULT_REACH_VISION_AREA = 0.18
DEFAULT_SEARCH_TURN = 1.0
DEFAULT_BEARING_GAIN = 0.8
DEFAULT_LOW_PASS_ALPHA = 0.3
DEFAULT_LOST_HOLD_STEPS = 20
DEFAULT_SEARCH_SPEED = 0.8
DEFAULT_MIN_VISIBLE_SPEED = 0.25
DEFAULT_SIZE_SPEED_GAIN = 0.75
DEFAULT_TURN_GAINS = (-0.5, -0.5, -0.5, 1, 0.5, -0.5, -1, 1)
TURN_INPUT_NAMES = ("bearing", "visible", "size")
TURN_GAIN_ORDER = (
    "hinge_0",
    "hinge_1",
    "hinge_2",
    "hinge_3",
    "hinge_4",
    "hinge_5",
    "hinge_6",
    "hinge_7",
)


@dataclass(frozen=True)
class TurnSignalFeatures:
    bearing: float
    visible: float
    size: float
    raw_area: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.bearing, self.visible, self.size], dtype=np.float32)


@dataclass(frozen=True)
class TurnCommand:
    action: np.ndarray
    features: TurnSignalFeatures
    raw_turn: float
    turn: float
    speed: float
    held_lost_target: bool


def vision_to_turn_features(
    vision,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    reach_vision_area: float = DEFAULT_REACH_VISION_AREA,
) -> TurnSignalFeatures:
    raw_area = float(vision[6])
    visible = raw_area > float(visibility_threshold)
    bearing = float(vision[5]) if visible else 0.0
    size = np.clip(raw_area / float(reach_vision_area), 0.0, 1.0) if visible else 0.0
    return TurnSignalFeatures(
        bearing=bearing,
        visible=float(visible),
        size=float(size),
        raw_area=raw_area,
    )


def hand_coded_speed(
    features: TurnSignalFeatures,
    search_speed: float = DEFAULT_SEARCH_SPEED,
    min_visible_speed: float = DEFAULT_MIN_VISIBLE_SPEED,
    size_speed_gain: float = DEFAULT_SIZE_SPEED_GAIN,
) -> float:
    if features.visible == 0.0:
        return float(search_speed)
    return float(np.clip(1.0 - float(size_speed_gain) * features.size, min_visible_speed, 1.0))


def hand_coded_turn(
    features: TurnSignalFeatures,
    bearing_gain: float = DEFAULT_BEARING_GAIN,
    search_turn: float = DEFAULT_SEARCH_TURN,
) -> float:
    if features.visible == 0.0:
        return float(np.clip(search_turn, -1.0, 1.0))
    return float(np.clip(-float(bearing_gain) * features.bearing, -1.0, 1.0))


class TurnBiasController:
    """Apply a scalar turn command as per-joint offsets on top of a frozen gait."""

    def __init__(self, gait_net, gains=DEFAULT_TURN_GAINS) -> None:
        self.gait_net = gait_net
        self.gains = np.asarray(gains, dtype=np.float32)
        if self.gains.shape != (len(TURN_GAIN_ORDER),):
            raise ValueError(f"Turn gains must have {len(TURN_GAIN_ORDER)} values.")

    def reset(self) -> None:
        self.gait_net.reset_hidden()

    def compute(self, turn: float, speed: float) -> np.ndarray:
        base = np.asarray(self.gait_net.forward(turn=0.0, speed=float(speed)), dtype=np.float32)
        action = base + float(turn) * self.gains
        return np.clip(action, -np.pi / 2, np.pi / 2).astype(np.float32)


class HandCodedTurnProvider:
    """Stateful hand-coded Stage 2 turn provider for visualizer and recorder scripts."""

    def __init__(
        self,
        gait_net,
        gains=DEFAULT_TURN_GAINS,
        *,
        bearing_gain: float = DEFAULT_BEARING_GAIN,
        search_turn: float = DEFAULT_SEARCH_TURN,
        low_pass_alpha: float = DEFAULT_LOW_PASS_ALPHA,
        lost_hold_steps: int = DEFAULT_LOST_HOLD_STEPS,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
        reach_vision_area: float = DEFAULT_REACH_VISION_AREA,
    ) -> None:
        self.turn_bias = TurnBiasController(gait_net, gains)
        self.bearing_gain = float(bearing_gain)
        self.search_turn = float(search_turn)
        self.low_pass_alpha = float(low_pass_alpha)
        self.lost_hold_steps = int(lost_hold_steps)
        self.visibility_threshold = float(visibility_threshold)
        self.reach_vision_area = float(reach_vision_area)
        self.previous_turn = float(search_turn)
        self.lost_steps = 0
        self.seen_target = False

    def reset(self) -> None:
        self.turn_bias.reset()
        self.previous_turn = self.search_turn
        self.lost_steps = 0
        self.seen_target = False

    def compute(self, model, vision) -> TurnCommand:
        features = vision_to_turn_features(
            vision,
            visibility_threshold=self.visibility_threshold,
            reach_vision_area=self.reach_vision_area,
        )
        held_lost_target = False
        if features.visible:
            raw_turn = hand_coded_turn(features, self.bearing_gain, self.search_turn)
            self.lost_steps = 0
            self.seen_target = True
        elif self.seen_target and self.lost_steps < self.lost_hold_steps:
            raw_turn = self.previous_turn
            self.lost_steps += 1
            held_lost_target = True
        else:
            raw_turn = hand_coded_turn(features, self.bearing_gain, self.search_turn)
            self.lost_steps += 1

        turn = self.low_pass_alpha * raw_turn + (1.0 - self.low_pass_alpha) * self.previous_turn
        turn = float(np.clip(turn, -1.0, 1.0))
        self.previous_turn = turn
        speed = hand_coded_speed(features)
        action = self.turn_bias.compute(turn, speed)
        if action.shape != (model.nu,):
            action = np.zeros(model.nu, dtype=np.float32)
        return TurnCommand(action, features, float(raw_turn), turn, speed, held_lost_target)


class TurnNetwork(nn.Module):
    """Tiny camera feature to scalar turn network."""

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, features) -> float:
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
        hidden = torch.tanh(self.fc1(x))
        turn = torch.tanh(self.fc2(hidden))[0]
        return float(turn.detach().item())


class LearnedTurnProvider:
    """Stateful learned Stage 2 turn provider with the same bias interface."""

    def __init__(
        self,
        gait_net,
        turn_net: TurnNetwork,
        gains,
        *,
        low_pass_alpha: float = DEFAULT_LOW_PASS_ALPHA,
        lost_hold_steps: int = DEFAULT_LOST_HOLD_STEPS,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
        reach_vision_area: float = DEFAULT_REACH_VISION_AREA,
    ) -> None:
        self.turn_bias = TurnBiasController(gait_net, gains)
        self.turn_net = turn_net
        self.low_pass_alpha = float(low_pass_alpha)
        self.lost_hold_steps = int(lost_hold_steps)
        self.visibility_threshold = float(visibility_threshold)
        self.reach_vision_area = float(reach_vision_area)
        self.previous_turn = 0.0
        self.lost_steps = 0
        self.seen_target = False

    def reset(self) -> None:
        self.turn_bias.reset()
        self.previous_turn = 0.0
        self.lost_steps = 0
        self.seen_target = False

    def compute(self, model, vision) -> TurnCommand:
        features = vision_to_turn_features(
            vision,
            visibility_threshold=self.visibility_threshold,
            reach_vision_area=self.reach_vision_area,
        )
        held_lost_target = False
        if features.visible:
            raw_turn = self.turn_net.forward(features.as_array())
            self.lost_steps = 0
            self.seen_target = True
        elif self.seen_target and self.lost_steps < self.lost_hold_steps:
            raw_turn = self.previous_turn
            self.lost_steps += 1
            held_lost_target = True
        else:
            raw_turn = self.turn_net.forward(features.as_array())
            self.lost_steps += 1

        turn = self.low_pass_alpha * raw_turn + (1.0 - self.low_pass_alpha) * self.previous_turn
        turn = float(np.clip(turn, -1.0, 1.0))
        self.previous_turn = turn
        speed = hand_coded_speed(features)
        action = self.turn_bias.compute(turn, speed)
        if action.shape != (model.nu,):
            action = np.zeros(model.nu, dtype=np.float32)
        return TurnCommand(action, features, float(raw_turn), turn, speed, held_lost_target)


def network_param_count(network: TurnNetwork) -> int:
    return sum(param.numel() for param in network.parameters())


def turn_vector_size(network: TurnNetwork | None = None) -> int:
    turn_net = TurnNetwork() if network is None else network
    return network_param_count(turn_net) + len(TURN_GAIN_ORDER)


@torch.no_grad()
def get_turn_vector(network: TurnNetwork, gains) -> np.ndarray:
    parts = [param.detach().flatten() for param in network.parameters()]
    gain_tensor = torch.as_tensor(np.asarray(gains, dtype=np.float32), dtype=torch.float32)
    parts.append(gain_tensor)
    return torch.cat(parts).cpu().numpy().astype(np.float32)


@torch.no_grad()
def set_turn_vector(network: TurnNetwork, vector) -> np.ndarray:
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32).copy(), dtype=torch.float32)
    expected = turn_vector_size(network)
    if vec.numel() != expected:
        raise ValueError(f"Turn vector has {vec.numel()} values, expected {expected}.")

    pointer = 0
    for param in network.parameters():
        n = param.numel()
        param.data[:] = vec[pointer:pointer + n].view_as(param)
        pointer += n
    gains = vec[pointer:pointer + len(TURN_GAIN_ORDER)].cpu().numpy().astype(np.float32)
    return gains


def save_turn_checkpoint(vector, model_path: Path, meta_path: Path, **metadata) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(model_path), np.asarray(vector, dtype=np.float32))
    np.savez(
        str(meta_path),
        vector_size=np.asarray([len(vector)], dtype=np.int32),
        input_names=np.asarray(TURN_INPUT_NAMES),
        gain_order=np.asarray(TURN_GAIN_ORDER),
        network_architecture="3-8-1",
        **metadata,
    )


def load_turn_checkpoint(model_path: Path, meta_path: Path | None = None):
    vector = np.load(str(model_path))
    network = TurnNetwork()
    gains = set_turn_vector(network, vector)
    meta = np.load(str(meta_path)) if meta_path is not None else None
    return network, gains, meta


def parse_turn_gains(value: str | None):
    if value is None:
        return np.asarray(DEFAULT_TURN_GAINS, dtype=np.float32)
    gains = np.asarray([float(part.strip()) for part in value.split(",")], dtype=np.float32)
    if gains.shape != (len(TURN_GAIN_ORDER),):
        raise ValueError(f"--gains must contain {len(TURN_GAIN_ORDER)} comma-separated values.")
    return gains
