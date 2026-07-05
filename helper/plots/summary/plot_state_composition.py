"""Plot how much time each trial spent in behavior-tree states."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

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
    present_conditions,
    save_figure,
    state_durations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def state_percentages(trial) -> dict[str, float]:
    duration = behavior_duration_s(trial)
    durations = state_durations(trial)
    return {state: durations.get(state, 0.0) * 100.0 / duration for state in STATE_ORDER}


def legend_handles_for_states(states: set[str]) -> list[Patch]:
    return [
        Patch(facecolor=STATE_COLORS[state], edgecolor="white", label=state)
        for state in STATE_ORDER
        if state in states
    ]


def plot_condition_summary(trials: list, output_dir: Path, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.8, max(4.4, 0.65 * len(present_conditions(trials)) + 1.7)), constrained_layout=True)

    conditions = present_conditions(trials)
    visible_states = set()
    y_positions = list(range(len(conditions)))[::-1]
    for y, condition in zip(y_positions, conditions):
        condition_trials = [trial for trial in trials if trial.condition == condition]
        percentages = [state_percentages(trial) for trial in condition_trials]
        left = 0.0
        for state in STATE_ORDER:
            width = float(np.mean([row[state] for row in percentages]))
            if width <= 0.0:
                continue
            visible_states.add(state)
            ax.barh(
                y,
                width,
                left=left,
                color=STATE_COLORS[state],
                edgecolor="white",
                height=0.72,
            )
            if width >= 7.0:
                ax.text(
                    left + width * 0.5,
                    y,
                    f"{width:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            left += width

    ax.set_title("Mean Behavior-State Time Share by Experimental Condition")
    ax.set_xlabel("share of trial duration [%]")
    ax.set_xlim(0.0, 100.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([condition_display_name(condition) for condition in conditions])
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles_for_states(visible_states), loc="lower right", frameon=True)

    save_figure(fig, output_dir, "behavior_state_composition.png", save_pdf)
    plt.close(fig)


def plot_trial_detail(trials: list, output_dir: Path, save_pdf: bool) -> None:
    fig_height = max(5.4, 0.34 * len(trials) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_height), constrained_layout=True)
    max_time = max(behavior_duration_s(trial) for trial in trials)

    y_positions = list(range(len(trials)))[::-1]
    y_labels = []
    visible_states = set()
    for y, trial in zip(y_positions, trials):
        durations = state_durations(trial)
        left = 0.0
        for state in STATE_ORDER:
            duration = durations.get(state, 0.0)
            if duration <= 0.0:
                continue
            visible_states.add(state)
            ax.barh(
                y,
                duration,
                left=left,
                color=STATE_COLORS[state],
                edgecolor="white",
                height=0.78,
            )
            left += duration
        y_labels.append(trial.label)

    for index in range(1, len(trials)):
        if trials[index].condition != trials[index - 1].condition:
            separator_y = y_positions[index] + 0.5
            ax.axhline(separator_y, color="black", linewidth=0.7, alpha=0.35)

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

    ax.set_title("Behavior-State Composition by Trial")
    ax.set_xlabel("time [s]")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles_for_states(visible_states), loc="lower right", frameon=True)

    save_figure(fig, output_dir, "behavior_state_composition_by_trial.png", save_pdf)
    plt.close(fig)


def plot(input_dir: Path, output_dir: Path, save_pdf: bool) -> None:
    trials = load_trials(input_dir)
    plot_condition_summary(trials, output_dir, save_pdf)
    plot_trial_detail(trials, output_dir, save_pdf)


def main() -> None:
    args = parse_args()
    plot(args.input_dir, args.output_dir, args.pdf)


if __name__ == "__main__":
    main()
