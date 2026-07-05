"""Generate all diagnostic plots for the real-world behavior-tree experiments."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.plots.diagonostic import (
    plot_behavior_timeline,
    plot_distance_to_target,
    plot_vision_timeseries,
)
from helper.plots.real_world_exp_data import DEFAULT_INPUT_DIR, DEFAULT_PLOT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "diagonostic")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_distance_to_target.plot(args.input_dir, args.output_dir, args.pdf)
    plot_behavior_timeline.plot(args.input_dir, args.output_dir, args.pdf)
    plot_vision_timeseries.plot(args.input_dir, args.output_dir, args.pdf)


if __name__ == "__main__":
    main()
