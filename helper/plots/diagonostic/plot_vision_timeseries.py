"""Plot vision area and bearing for one representative trial per condition."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    STATE_COLORS,
    TICK_LABEL_SIZE,
    condition_display_name,
    load_trials,
    representative_trials,
    save_figure,
    state_spans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "diagonostic")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def add_state_background(ax, trial) -> None:
    for start_s, end_s, state in state_spans(trial):
        ax.axvspan(
            start_s,
            end_s,
            color=STATE_COLORS[state],
            alpha=0.22,
            linewidth=0,
        )


def plot(input_dir: Path, output_dir: Path, save_pdf: bool) -> None:
    trials = representative_trials(load_trials(input_dir))
    fig, axes = plt.subplots(
        len(trials),
        1,
        figsize=(10.5, max(8.4, 2.2 * len(trials))),
        sharex=False,
        constrained_layout=True,
    )
    if len(trials) == 1:
        axes = [axes]

    for ax_area, trial in zip(axes, trials):
        add_state_background(ax_area, trial)
        if trial.behavior_available:
            ax_area.plot(
                trial.behavior.time_s,
                trial.behavior.area,
                color="#1F4E79",
                linewidth=1.6,
                label="visible target area",
            )
            reach_vision_area = float(trial.metadata["reach_vision_area"])
            if not math.isnan(reach_vision_area):
                ax_area.axhline(
                    reach_vision_area,
                    color="#1F4E79",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.75,
                    label="stop area threshold",
                )
        else:
            ax_area.text(
                0.5,
                0.5,
                "camera / behavior log missing\nplaceholder panel",
                transform=ax_area.transAxes,
                ha="center",
                va="center",
                fontsize=AXIS_LABEL_SIZE,
                bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9, "pad": 4},
            )
        ax_area.set_ylabel("area fraction", fontsize=AXIS_LABEL_SIZE)
        ax_area.grid(alpha=0.25)
        ax_area.set_title(f"{condition_display_name(trial.condition)}: {trial.label}", fontsize=AXIS_TITLE_SIZE)
        ax_area.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

        if trial.behavior_available:
            ax_bearing = ax_area.twinx()
            ax_bearing.plot(
                trial.behavior.time_s,
                trial.behavior.bearing,
                color="#C00000",
                linewidth=1.2,
                alpha=0.72,
                label="target bearing",
            )
            threshold = float(trial.metadata["bearing_threshold"])
            if not math.isnan(threshold):
                ax_bearing.axhline(threshold, color="#C00000", linestyle=":", linewidth=1.0, alpha=0.55)
                ax_bearing.axhline(-threshold, color="#C00000", linestyle=":", linewidth=1.0, alpha=0.55)
            ax_bearing.set_ylabel("bearing", fontsize=AXIS_LABEL_SIZE)
            ax_bearing.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax_area.set_xlabel("time [s]", fontsize=AXIS_LABEL_SIZE)

    legend_handles = [
        Line2D([0], [0], color="#1F4E79", label="visible target area"),
        Line2D([0], [0], color="#1F4E79", linestyle="--", label="stop area threshold"),
        Line2D([0], [0], color="#C00000", label="target bearing"),
        Line2D([0], [0], color="#C00000", linestyle=":", label="bearing threshold"),
        Patch(facecolor=STATE_COLORS["SEARCHING"], alpha=0.22, label="SEARCHING"),
        Patch(facecolor=STATE_COLORS["APPROACHING"], alpha=0.22, label="APPROACHING"),
        Patch(facecolor=STATE_COLORS["STOPPED"], alpha=0.22, label="STOPPED"),
        Patch(facecolor=STATE_COLORS["NO ROBOT LOG"], alpha=0.22, label="NO ROBOT LOG"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=LEGEND_SIZE)
    fig.suptitle("Vision Signals Used by the Hardware Behavior Tree", fontsize=FIGURE_TITLE_SIZE)

    save_figure(fig, output_dir, "vision_area_bearing_timeseries.png", save_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.input_dir, args.output_dir, args.pdf)


if __name__ == "__main__":
    main()
