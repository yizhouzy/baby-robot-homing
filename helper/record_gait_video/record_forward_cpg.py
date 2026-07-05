"""Record a Stage 1 CPG gait video from a saved Nevergrad run.

Examples:
    uv run helper/record_video.py
    uv run helper/record_video.py --model results/cmaes/<run_id>/gait_best_<run_id>.npy
    uv run helper/record_video.py --model results/cmaes/<run_id>/checkpoints/gait_ckpt_<run_id>_gen50.npy
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

from rich.console import Console
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.artifacts import (
    demo_output_paths,
    extract_run_id,
    find_latest_best_model,
    infer_meta_path,
    resolve_model_path,
)
from robot_control.config_gait import GaitConfig
from robot_control.controllers import load_gait_network
from robot_control.rendering import render_demo_from_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a demo video for a saved Nevergrad CPG gait model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a gait .npy file. If omitted, uses the best model from the "
            "latest timestamped run under results/cmaes."
        ),
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to matching gait_meta_<run_id>.npz. If omitted, inferred from the model.",
    )
    parser.add_argument("--dur", type=float, default=30.0)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=-1.8)
    parser.add_argument("--target-z", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. If omitted, uses <run_dir>/videos like the trainer.",
    )
    return parser.parse_args()


def main() -> None:
    install()
    console = Console()
    args = parse_args()

    model_path = resolve_model_path(args.model) if args.model is not None else find_latest_best_model()
    try:
        meta_path = infer_meta_path(model_path, args.meta)
    except FileNotFoundError as exc:
        if args.meta is not None:
            raise
        meta_path = None
        console.log(f"{exc} Loading checkpoint without metadata.", style="yellow")
    run_id, output_dir, video_name, trajectory_path = demo_output_paths(
        model_path, args.output_dir)
    target_pos = (args.target_x, args.target_y, args.target_z)

    console.log(f"Run id: {run_id}")
    console.log(f"Model: {model_path}")
    console.log(f"Metadata: {meta_path if meta_path is not None else 'not found; inferred'}")
    console.log(f"Target: {list(target_pos)}")
    console.log(f"Video folder: {output_dir}")

    network, meta, weight_format = load_gait_network(model_path, meta_path)
    console.log(
        f"Loaded {weight_format}: num_joints={int(meta['num_joints'])}, "
        f"dt={float(meta['dt']):.4f}"
    )

    config = GaitConfig(
        duration=args.dur,
        seed=int(meta["seed"]) if "seed" in meta else 42,
        run_id=extract_run_id(model_path),
        demo_target=target_pos,
    )
    render_demo_from_network(
        network,
        config,
        output_dir=output_dir,
        video_name=video_name,
        trajectory_path=trajectory_path,
        duration=args.dur,
        target_pos=target_pos,
        console=console,
    )


if __name__ == "__main__":
    main()
