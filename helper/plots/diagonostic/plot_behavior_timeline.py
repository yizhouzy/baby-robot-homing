"""Plot the behavior-tree state timeline for every real-world trial."""
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
    DEFAULT_INPUT_DIR,
    DEFAULT_PLOT_DIR,
    STATE_COLORS,
    STATE_ORDER,
    behavior_duration_s,
    condition_display_name,
    load_trials,
    save_figure,
    state_spans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "diagonostic")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def plot(input_dir: Path, output_dir: Path, save_pdf: bool) -> None:
    trials = load_trials(input_dir)
    fig_height = max(5.4, 0.34 * len(trials) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
    y_positions = list(range(len(trials)))[::-1]
    max_time = max(behavior_duration_s(trial) for trial in trials)

    for y, trial in zip(y_positions, trials):
        for start_s, end_s, state in state_spans(trial):
            ax.broken_barh(
                [(start_s, end_s - start_s)],
                (y - 0.38, 0.76),
                facecolors=STATE_COLORS[state],
                edgecolors="white",
                linewidth=0.8,
            )

    for index in range(1, len(trials)):
        if trials[index].condition != trials[index - 1].condition:
            ax.axhline(y_positions[index] + 0.5, color="black", linewidth=0.7, alpha=0.35)

    group_start = 0
    for index in range(1, len(trials) + 1):
        if index == len(trials) or trials[index].condition != trials[group_start].condition:
            center_y = float((y_positions[group_start] + y_positions[index - 1]) * 0.5)
            ax.text(
                max_time * 0.985,
                center_y,
                condition_display_name(trials[group_start].condition),
                ha="right",
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
            )
            group_start = index

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[state], label=state)
        for state in STATE_ORDER
        if state in {mode for trial in trials for mode in trial.behavior.mode}
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    ax.set_title("Behavior-Tree State Timeline")
    ax.set_xlabel("time [s]")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([trial.label for trial in trials], fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    save_figure(fig, output_dir, "behavior_timeline.png", save_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.input_dir, args.output_dir, args.pdf)


if __name__ == "__main__":
    main()
