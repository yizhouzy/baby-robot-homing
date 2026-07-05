"""Evolve a domain-randomized forward CPG gait controller for the Baby robot."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from rich.console import Console
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.artifacts import (
    ensure_run_dirs,
    save_training_artifacts,
    show_final_summary,
    show_startup_summary,
)
from robot_control.config_gait import GaitConfig
from robot_control.controllers import load_gait_network
from robot_control.evaluation import seed_everything
from robot_control.rendering import render_demo_from_network
from robot_control.training import train_gait


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--dur", type=int, default=30)
    parser.add_argument("--num-actors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument(
        "--fitness",
        type=str,
        default="speed",
        choices=["delta", "distance", "survival", "direct", "speed"],
    )
    parser.add_argument("--reach-radius", type=float, default=0.25)
    parser.add_argument("--fall-z-threshold", type=float, default=0.05)
    parser.add_argument("--fall-tilt-threshold-deg", type=float, default=75.0)
    parser.add_argument("--use-tilt-fall", action="store_true")
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--action-noise-std", type=float, default=0.03) # radians
    parser.add_argument("--friction-scale-min", type=float, default=0.5)
    parser.add_argument("--friction-scale-max", type=float, default=1.5)
    parser.add_argument("--mass-scale-min", type=float, default=0.9)
    parser.add_argument("--mass-scale-max", type=float, default=1.1)
    parser.add_argument("--joint-strength-scale-min", type=float, default=0.7) # servo strength
    parser.add_argument("--joint-strength-scale-max", type=float, default=1.3)
    parser.add_argument("--collision-weight", type=float, default=1.5)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GaitConfig:
    return GaitConfig(
        budget=args.budget,
        population=args.population,
        duration=args.dur,
        num_actors=args.num_actors,
        seed=int(args.seed),
        sigma=args.sigma,
        fitness=args.fitness,
        reach_radius=args.reach_radius,
        fall_z_threshold=args.fall_z_threshold,
        fall_tilt_threshold_deg=args.fall_tilt_threshold_deg,
        use_tilt_fall=bool(args.use_tilt_fall),
        eval_repeats=args.eval_repeats,
        domain_randomization=not args.no_domain_randomization,
        action_noise_std=args.action_noise_std,
        friction_scale_min=args.friction_scale_min,
        friction_scale_max=args.friction_scale_max,
        mass_scale_min=args.mass_scale_min,
        mass_scale_max=args.mass_scale_max,
        joint_strength_scale_min=args.joint_strength_scale_min,
        joint_strength_scale_max=args.joint_strength_scale_max,
        collision_weight=args.collision_weight,
        record_video=not args.no_video,
    )


def main() -> None:
    install()
    console = Console(force_terminal=True, color_system="truecolor")
    args = parse_args()
    config = build_config(args)
    seed_everything(config.seed)
    ensure_run_dirs(config)

    run_start = time.time()
    train_start = time.time()
    result = train_gait(config, console, startup_summary=show_startup_summary)
    training_elapsed_seconds = time.time() - train_start
    artifacts = save_training_artifacts(
        config,
        result,
        console,
        training_elapsed_seconds=training_elapsed_seconds,
    )
    elapsed_minutes = (time.time() - run_start) / 60
    show_final_summary(artifacts, result.best_eval, elapsed_minutes, console)

    if not config.record_video:
        return

    console.rule("[bold cyan]Demo")
    console.log("Recording forward gait demo video...")
    gait_net, _, _ = load_gait_network(artifacts["best_path"], artifacts["meta_path"])
    render_demo_from_network(
        gait_net,
        config,
        output_dir=config.video_dir,
        video_name=f"gait_demo_{config.run_id}",
        trajectory_path=config.run_dir / f"gait_trajectory_{config.run_id}.png",
        console=console,
    )


if __name__ == "__main__":
    main()
