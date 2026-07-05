"""Demo: behavior tree controller layered on a learned CPG gait."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.behavior import (
    ActionContext,
    BehaviorDemoConfig,
    LearnedGaitProvider,
)
from robot_control.config_gait import make_run_id
from robot_control.controllers import load_gait_network
from robot_control.rendering import (
    build_behavior_scene,
    plot_behavior_trajectory,
    record_behavior_demo,
)
from robot_control.artifacts import (
    DEFAULT_GAIT_MODELS,
    infer_checkpoint_meta_path,
    resolve_existing_path,
)


OUTPUT_DIR = Path("results/behavior_tree")
DEFAULT_FORWARD_MODEL, DEFAULT_FORWARD_META = DEFAULT_GAIT_MODELS["forward"]
DEFAULT_LEFT_MODEL, DEFAULT_LEFT_META = DEFAULT_GAIT_MODELS["left"]
DEFAULT_RIGHT_MODEL, DEFAULT_RIGHT_META = DEFAULT_GAIT_MODELS["right"]

class VisionSwitchBehaviorPolicy:
    """Switch between learned spin and forward CPGs using target visibility."""

    def __init__(
        self,
        providers: dict[str, LearnedGaitProvider],
        config: BehaviorDemoConfig,
        visibility_threshold: float,
        bearing_threshold: float,
        lost_frame_limit: int,
        high_battery_behavior: str,
        forward_speed: float,
        search_speed: float,
        default_search_gait: str,
        invert_bearing: bool,
        recenter_during_approach: bool,
    ) -> None:
        self.providers = providers
        self.config = config
        self.visibility_threshold = float(visibility_threshold)
        self.bearing_threshold = float(bearing_threshold)
        self.lost_frame_limit = int(lost_frame_limit)
        self.high_battery_behavior = high_battery_behavior
        self.forward_speed = float(forward_speed)
        self.search_speed = float(search_speed)
        self.default_search_gait = default_search_gait
        self.invert_bearing = bool(invert_bearing)
        self.recenter_during_approach = bool(recenter_during_approach)
        self.lost_frames = 0
        self.initial_search_complete = False
        self.last_selected_gait = "none"

    def reset(self) -> None:
        for provider in self.providers.values():
            provider.reset()
        self.lost_frames = 0
        self.initial_search_complete = False
        self.last_selected_gait = "none"

    def gait_action(self, gait: str, speed: float, vision, battery: float, model, data) -> np.ndarray:
        context = ActionContext(vision=vision, battery=battery, mode="", speed=speed)
        return self.providers[gait].compute(model, data, context)

    def search_gait_from_bearing(self, bearing: float) -> str:
        signed_bearing = -bearing if self.invert_bearing else bearing
        if abs(signed_bearing) <= self.bearing_threshold:
            return self.default_search_gait
        return "right" if signed_bearing > 0.0 else "left"

    def decide_with_gait(self, vision, battery: float, model, data) -> tuple[np.ndarray, str, str]:
        area = float(vision[6])
        bearing = float(vision[5])
        visible = area >= self.visibility_threshold

        if battery > self.config.battery_threshold:
            if self.high_battery_behavior == "idle":
                return np.zeros(model.nu, dtype=np.float32), "IDLE", "none"
            gait = self.default_search_gait
            return self.gait_action(gait, self.search_speed, vision, battery, model, data), "SEARCHING", gait

        if area >= self.config.reach_vision_area:
            self.lost_frames = 0
            return np.zeros(model.nu, dtype=np.float32), "STOPPED", "none"

        if visible and abs(bearing) <= self.bearing_threshold:
            self.initial_search_complete = True
            self.lost_frames = 0
            return (
                self.gait_action("forward", self.forward_speed, vision, battery, model, data),
                "APPROACHING",
                "forward",
            )

        if visible and (not self.initial_search_complete or self.recenter_during_approach):
            gait = self.search_gait_from_bearing(bearing)
            self.lost_frames = 0
            return self.gait_action(gait, self.search_speed, vision, battery, model, data), "SEARCHING", gait

        if self.initial_search_complete:
            self.lost_frames += 1
            if self.lost_frames < self.lost_frame_limit:
                return (
                    self.gait_action("forward", self.forward_speed, vision, battery, model, data),
                    "APPROACHING",
                    "forward",
                )

        gait = self.default_search_gait
        return self.gait_action(gait, self.search_speed, vision, battery, model, data), "SEARCHING", gait

    def decide(self, vision, battery: float, model, data) -> tuple[np.ndarray, str]:
        action, mode, selected_gait = self.decide_with_gait(vision, battery, model, data)
        self.last_selected_gait = selected_gait
        return action, mode

VISION_PIXELS = 240 * 320 
DEFAULT_VISIBILITY_THRESHOLD = 0.0015
DEFAULT_REACH_VISION_AREA = 0.14

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--battery-threshold", type=float, default=1.0)
    parser.add_argument("--model", "--weights", "--forward-model", dest="weights", type=str, default=str(DEFAULT_FORWARD_MODEL))
    parser.add_argument("--meta", "--forward-meta", dest="meta", type=str, default=None)
    parser.add_argument("--left-model", type=str, default=str(DEFAULT_LEFT_MODEL))
    parser.add_argument("--left-meta", type=str, default=None)
    parser.add_argument("--right-model", type=str, default=str(DEFAULT_RIGHT_MODEL))
    parser.add_argument("--right-meta", type=str, default=None)
    parser.add_argument("--visibility-threshold", type=float, default=DEFAULT_VISIBILITY_THRESHOLD)
    parser.add_argument("--bearing-threshold", type=float, default=0.15)
    parser.add_argument("--lost-vision-hold-s", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=None, help="Legacy lost-vision hold in control frames.")
    parser.add_argument("--default-search-gait", choices=["left", "right"], default="left")
    parser.add_argument("--invert-bearing", action="store_true")
    parser.add_argument("--recenter-during-approach", action="store_true", default=False)
    parser.add_argument("--no-recenter-during-approach", dest="recenter_during_approach", action="store_false")
    parser.add_argument("--high-battery-behavior", choices=["idle", "search"], default="idle")
    parser.add_argument("--station-x", type=float, default=0.0)
    parser.add_argument("--station-y", type=float, default=-2.0)
    parser.add_argument("--reach-radius", type=float, default=0.25)
    parser.add_argument(
        "--reach-vision-area",
        type=float,
        default=DEFAULT_REACH_VISION_AREA,
        help="Stop when the target mask fills this fraction of the robot camera.",
    )
    parser.add_argument("--explore-speed", type=float, default=1)
    parser.add_argument("--forward-speed", type=float, default=1)
    parser.add_argument("--search-speed", type=float, default=1)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BehaviorDemoConfig:
    run_id = make_run_id(int(args.seed))
    return BehaviorDemoConfig(
        duration=args.duration,
        battery_threshold=float(args.battery_threshold),
        station_pos=(args.station_x, args.station_y, 0.1),
        reach_radius=max(0.01, args.reach_radius),
        reach_vision_area=max(0.01, args.reach_vision_area),
        explore_speed=args.explore_speed,
        search_speed=args.search_speed,
        output_dir=OUTPUT_DIR / run_id,
        timestamp=run_id,
    )


def resolve_model_and_meta(model_arg: str, meta_arg: str | None) -> tuple[Path, Path]:
    model_path = resolve_existing_path(Path(model_arg), ".npy")
    meta_path = (
        resolve_existing_path(Path(meta_arg), ".npz")
        if meta_arg is not None
        else infer_checkpoint_meta_path(model_path)
    )
    return model_path, meta_path


def prefixed_npz_items(prefix: str, meta) -> dict:
    return {f"{prefix}_{key}": meta[key] for key in meta.files}


def save_demo_metadata(
    args: argparse.Namespace,
    config: BehaviorDemoConfig,
    samples,
    *,
    weights_path: Path,
    meta_path: Path | None,
    left_weights_path: Path,
    left_meta_path: Path,
    right_weights_path: Path,
    right_meta_path: Path,
    forward_meta,
    left_meta,
    right_meta,
    forward_weight_format: str,
    left_weight_format: str,
    right_weight_format: str,
    cam_name: str | None,
    control_freq: int,
    lost_frame_limit: int,
    physics_timestep: float,
    simulation_elapsed_seconds: float,
) -> Path:
    metadata_path = config.output_dir / f"demo_meta_{config.timestamp}.npz"
    modes = np.asarray([sample.mode for sample in samples])
    unique_modes, mode_counts = np.unique(modes, return_counts=True)

    payload = {
        "run_id": config.timestamp,
        "script": "experiments/demo_behavior_tree.py",
        "controller": "VisionSwitchBehaviorPolicy",
        "forward_model": str(weights_path),
        "forward_meta_path": "" if meta_path is None else str(meta_path),
        "forward_weight_format": forward_weight_format,
        "left_model": str(left_weights_path),
        "left_meta_path": str(left_meta_path),
        "left_weight_format": left_weight_format,
        "right_model": str(right_weights_path),
        "right_meta_path": str(right_meta_path),
        "right_weight_format": right_weight_format,
        "output_dir": str(config.output_dir),
        "video_dir": str(config.output_dir / "videos"),
        "plot_path": str(config.output_dir / f"demo_plots_{config.timestamp}.png"),
        "duration": float(config.duration),
        "seed": int(args.seed),
        "battery_threshold": float(config.battery_threshold),
        "battery_initial": 1.0,
        "battery_drain_per_second": 1.0 / float(config.duration),
        "station_pos": np.asarray(config.station_pos, dtype=np.float32),
        "reach_radius": float(config.reach_radius),
        "reach_vision_area": float(config.reach_vision_area),
        "visibility_threshold": float(args.visibility_threshold),
        "bearing_threshold": float(args.bearing_threshold),
        "lost_vision_hold_s": float(args.lost_vision_hold_s),
        "lost_vision_hold_frames": int(lost_frame_limit),
        "patience": -1 if args.patience is None else int(args.patience),
        "default_search_gait": args.default_search_gait,
        "invert_bearing": bool(args.invert_bearing),
        "recenter_during_approach": bool(args.recenter_during_approach),
        "high_battery_behavior": args.high_battery_behavior,
        "explore_speed": float(config.explore_speed),
        "forward_speed": float(args.forward_speed),
        "search_speed": float(config.search_speed),
        "camera_name": "" if cam_name is None else cam_name,
        "camera_resolution_hw": np.asarray([24, 32], dtype=np.int32),
        "control_freq": int(control_freq),
        "physics_timestep": float(physics_timestep),
        "simulation_elapsed_seconds": float(simulation_elapsed_seconds),
        "sample_count": len(samples),
        "mode_names": unique_modes,
        "mode_counts": mode_counts.astype(np.int32),
        "time": np.asarray([sample.time for sample in samples], dtype=np.float32),
        "x": np.asarray([sample.x for sample in samples], dtype=np.float32),
        "y": np.asarray([sample.y for sample in samples], dtype=np.float32),
        "yaw": np.asarray([sample.yaw for sample in samples], dtype=np.float32),
        "battery": np.asarray([sample.battery for sample in samples], dtype=np.float32),
        "mode": modes,
        "distance": np.asarray([sample.distance for sample in samples], dtype=np.float32),
        "vision_area": np.asarray([sample.vision_area for sample in samples], dtype=np.float32),
        "vision_centroid": np.asarray([sample.vision_centroid for sample in samples], dtype=np.float32),
    }
    payload.update(prefixed_npz_items("forward_meta", forward_meta))
    payload.update(prefixed_npz_items("left_meta", left_meta))
    payload.update(prefixed_npz_items("right_meta", right_meta))
    np.savez(str(metadata_path), **payload)
    return metadata_path


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    (config.output_dir / "videos").mkdir(exist_ok=True)

    weights_path, meta_path = resolve_model_and_meta(args.weights, args.meta)
    left_weights_path, left_meta_path = resolve_model_and_meta(args.left_model, args.left_meta)
    right_weights_path, right_meta_path = resolve_model_and_meta(args.right_model, args.right_meta)

    print(f"Loading forward gait weights from {weights_path}")
    print(f"Loading forward gait metadata from {meta_path}")
    print(f"Loading left spin weights from {left_weights_path}")
    print(f"Loading left spin metadata from {left_meta_path}")
    print(f"Loading right spin weights from {right_weights_path}")
    print(f"Loading right spin metadata from {right_meta_path}")
    print(f"Run dir: {config.output_dir}")
    print(
        f"Duration: {config.duration}s  |  Station: {list(config.station_pos[:2])}  |  "
        f"Battery threshold: {config.battery_threshold:.0%}"
    )

    model, data, cam_name, mocap_id = build_behavior_scene(config)
    forward_net, meta, weight_format = load_gait_network(weights_path, meta_path)
    left_net, left_meta, left_weight_format = load_gait_network(left_weights_path, left_meta_path)
    right_net, right_meta, right_weight_format = load_gait_network(right_weights_path, right_meta_path)
    control_freq = max(1, int(round(float(meta["dt"]) / model.opt.timestep)))
    lost_frame_limit = (
        int(args.patience)
        if args.patience is not None
        else max(1, int(round(args.lost_vision_hold_s / float(meta["dt"]))))
    )
    providers = {
        "forward": LearnedGaitProvider(forward_net),
        "left": LearnedGaitProvider(left_net),
        "right": LearnedGaitProvider(right_net),
    }
    policy = VisionSwitchBehaviorPolicy(
        providers,
        config,
        visibility_threshold=args.visibility_threshold,
        bearing_threshold=args.bearing_threshold,
        lost_frame_limit=lost_frame_limit,
        high_battery_behavior=args.high_battery_behavior,
        forward_speed=args.forward_speed,
        search_speed=args.search_speed,
        default_search_gait=args.default_search_gait,
        invert_bearing=args.invert_bearing,
        recenter_during_approach=args.recenter_during_approach,
    )

    print(
        f"Forward gait loaded ({weight_format}, "
        f"{sum(p.numel() for p in forward_net.cpg.parameters())} CPG params)"
    )
    print(
        f"Left spin loaded ({left_weight_format}, "
        f"{sum(p.numel() for p in left_net.cpg.parameters())} CPG params, "
        f"trained dt={float(left_meta['dt']):.4f}s)"
    )
    print(
        f"Right spin loaded ({right_weight_format}, "
        f"{sum(p.numel() for p in right_net.cpg.parameters())} CPG params, "
        f"trained dt={float(right_meta['dt']):.4f}s)"
    )
    print(
        f"Control update every {control_freq} physics steps "
        f"(trained dt={float(meta['dt']):.4f}s, physics dt={model.opt.timestep:.4f}s)"
    )
    print(
        "Switch logic: first search waits for centered target, then "
        "visible off-centre -> bearing-selected spin, briefly lost -> forward, "
        "genuinely lost -> default search"
    )
    print(f"Visibility threshold: {args.visibility_threshold:.5f}")
    print(f"Initial search bearing threshold: {args.bearing_threshold:.2f}")
    print(f"Lost-target patience: {lost_frame_limit} control frames")
    print(f"Default search gait: {args.default_search_gait}")
    print(f"High-battery behavior: {args.high_battery_behavior}")
    print(f"Running demo ({config.duration}s)...")

    start = time.time()
    samples = record_behavior_demo(
        model,
        data,
        cam_name,
        mocap_id,
        policy,
        config,
        control_freq,
    )
    simulation_elapsed_seconds = time.time() - start
    print(f"Simulation + recording took {simulation_elapsed_seconds:.1f}s")

    if samples:
        plot_behavior_trajectory(samples, config)
        final_sample = samples[-1]
        modes_seen = sorted({sample.mode for sample in samples})
        print(
            f"Final battery: {final_sample.battery:.1%}  |  "
            f"Final distance: {final_sample.distance:.2f}m"
        )
        print(f"Modes observed: {', '.join(modes_seen)}")

    metadata_path = save_demo_metadata(
        args,
        config,
        samples,
        weights_path=weights_path,
        meta_path=meta_path,
        left_weights_path=left_weights_path,
        left_meta_path=left_meta_path,
        right_weights_path=right_weights_path,
        right_meta_path=right_meta_path,
        forward_meta=meta,
        left_meta=left_meta,
        right_meta=right_meta,
        forward_weight_format=weight_format,
        left_weight_format=left_weight_format,
        right_weight_format=right_weight_format,
        cam_name=cam_name,
        control_freq=control_freq,
        lost_frame_limit=lost_frame_limit,
        physics_timestep=model.opt.timestep,
        simulation_elapsed_seconds=simulation_elapsed_seconds,
    )
    print(f"Metadata -> {metadata_path}")
    print(f"Video -> {config.output_dir / 'videos'}")
    print("Done.")


if __name__ == "__main__":
    main()
