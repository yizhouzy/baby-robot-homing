"""Plot OptiTrack distance-to-target over time for each trial."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.plots.real_world_exp_data import (
    AXIS_LABEL_SIZE,
    AXIS_TITLE_SIZE,
    DEFAULT_INPUT_DIR,
    DEFAULT_PLOT_DIR,
    FIGURE_TITLE_SIZE,
    LEGEND_SIZE,
    PANEL_FIGURE_HEIGHT,
    PANEL_WIDTH,
    TICK_LABEL_SIZE,
    condition_display_name,
    load_trials,
    motive_valid_mask,
    present_conditions,
    save_figure,
    trial_color,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "diagonostic")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def plot(input_dir: Path, output_dir: Path, save_pdf: bool) -> None:
    trials = load_trials(input_dir)
    conditions = present_conditions(trials)
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(PANEL_WIDTH * len(conditions), PANEL_FIGURE_HEIGHT),
        sharey=True,
        constrained_layout=True,
    )
    if len(conditions) == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        condition_trials = [trial for trial in trials if trial.condition == condition]
        missing_labels = []
        for trial in condition_trials:
            mask = motive_valid_mask(trial)
            if not any(mask):
                missing_labels.append(f"{trial.label}: no OptiTrack CSV")
                continue
            ax.plot(
                trial.motive.time_s[mask],
                trial.motive.distance_m[mask],
                color=trial_color(trial),
                linewidth=2.0,
                alpha=0.95,
                label=trial.label,
            )
        if missing_labels:
            ax.text(
                0.03,
                0.95,
                "\n".join(missing_labels),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=LEGEND_SIZE,
                bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 3},
            )

        ax.set_title(condition_display_name(condition), fontsize=AXIS_TITLE_SIZE)
        ax.set_xlabel("time [s]", fontsize=AXIS_LABEL_SIZE)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=LEGEND_SIZE, loc="best")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    axes[0].set_ylabel("OptiTrack distance to target [m]", fontsize=AXIS_LABEL_SIZE)
    fig.suptitle("Distance-to-Target During Real-World Trials", fontsize=FIGURE_TITLE_SIZE)
    save_figure(fig, output_dir, "distance_to_target_over_time.png", save_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.input_dir, args.output_dir, args.pdf)


if __name__ == "__main__":
    main()
