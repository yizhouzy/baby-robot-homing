"""Generate all summary plots for the real-world behavior-tree experiments."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.plots.real_world_exp_data import DEFAULT_INPUT_DIR, DEFAULT_PLOT_DIR
from helper.plots.summary import (
    plot_overhead_trajectories,
    plot_state_composition,
    plot_time_to_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_time_to_target.plot(args.input_dir, args.output_dir, args.pdf)
    plot_state_composition.plot(args.input_dir, args.output_dir, args.pdf)
    plot_overhead_trajectories.plot(args.input_dir, args.output_dir, 12, args.pdf)


if __name__ == "__main__":
    main()
