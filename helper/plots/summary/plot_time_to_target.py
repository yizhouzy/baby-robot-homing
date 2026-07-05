"""Plot real-world time-to-target by target placement condition."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.plots.real_world_exp_data import (
    AXIS_LABEL_SIZE,
    DEFAULT_INPUT_DIR,
    DEFAULT_PLOT_DIR,
    PANEL_FIGURE_HEIGHT,
    TICK_LABEL_SIZE,
    first_reached_time,
    behavior_duration_s,
    condition_display_name,
    load_trials,
    present_conditions,
    save_figure,
    trial_color,
    write_summary_csv,
)


COMPACT_CONDITION_LABELS = {
    "Mat front target": "Mat\nfrontal",
    "Mat 90 deg right target": "Mat\nright-side",
    "Mat back target": "Mat\nback",
    "Floor front target": "Low-friction\nlab floor",
    "Grass front target": "High-friction\ngrass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def plot(input_dir: Path, output_dir: Path, save_pdf: bool) -> None:
    trials = load_trials(input_dir)
    conditions = present_conditions(trials)
    x_spacing = 0.68
    trial_offset = 0.055
    condition_x = np.arange(len(conditions), dtype=float) * x_spacing
    fig, ax = plt.subplots(
        figsize=(8.6, PANEL_FIGURE_HEIGHT * 0.92),
        constrained_layout=True,
    )
    all_y_values = []

    for condition_index, condition in enumerate(conditions):
        condition_trials = [trial for trial in trials if trial.condition == condition]
        y_values = []
        real_time_values = []
        label_offsets = [(-9, -18), (0, 10), (9, 24)]
        for trial_index, trial in enumerate(condition_trials):
            reached_time = first_reached_time(trial)
            success = not math.isnan(reached_time)
            y = reached_time if success else behavior_duration_s(trial)
            y_values.append(y)
            if trial.behavior_available:
                real_time_values.append(y)
            x = condition_x[condition_index] + (trial_index - 1) * trial_offset
            marker = "o" if success else "x"
            facecolor = trial_color(trial)
            if not trial.behavior_available:
                marker = "s"
                facecolor = "none"
            all_y_values.append(y)
            ax.scatter(
                x,
                y,
                s=72,
                marker=marker,
                color=trial_color(trial),
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
            ax.annotate(
                f"T{trial_index + 1}",
                xy=(x, y),
                xytext=label_offsets[trial_index % len(label_offsets)],
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
            )

        mean_time = float(np.mean(real_time_values or y_values))
        ax.hlines(
            mean_time,
            condition_x[condition_index] - 0.18,
            condition_x[condition_index] + 0.18,
            color="black",
            linewidth=2.0,
        )
        ax.annotate(
            f"{mean_time:.1f} s",
            xy=(condition_x[condition_index] + 0.19, mean_time),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
        )
        if not real_time_values:
            ax.annotate(
                "no robot log",
                xy=(condition_x[condition_index] + 0.17, mean_time),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )

    ax.set_title("Time Required to Reach the Vision-Based Stopping Threshold", fontsize=15)
    ax.set_ylabel("time [s]", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(condition_x)
    ax.set_xticklabels(
        [COMPACT_CONDITION_LABELS.get(condition, condition_display_name(condition)) for condition in conditions],
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(min(all_y_values) - 3.6, max(all_y_values) + 4.0)
    ax.set_xlim(condition_x[0] - 0.32, condition_x[-1] + 0.48)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2.0, label="condition mean"),
            Line2D([0], [0], color="none", label="T1-T3 labels identify trials"),
        ],
        loc="upper left",
        fontsize=TICK_LABEL_SIZE,
        frameon=True,
    )

    save_figure(fig, output_dir, "time_to_target_by_condition.png", save_pdf)
    plt.close(fig)
    write_summary_csv(output_dir / "real_world_exp_summary.csv", trials)


def main() -> None:
    args = parse_args()
    plot(args.input_dir, args.output_dir, args.pdf)


if __name__ == "__main__":
    main()
