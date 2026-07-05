"""Plot OptiTrack overhead trajectories for each target placement condition."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

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


TARGET_BASE_SIZE_M = 0.27
TARGET_BASE_ANGLE_DEG = 45.0
CANONICAL_TARGET_TO_START_ANGLE_DEG = -32.0
SIMULATION_DIR_NAME = "simulation_trajectories"
SIMULATION_COLOR = "#6E6E6E"
SIMULATION_FILENAMES = {
    "mat_front_target.csv",
    "mat_90_deg_right_target.csv",
    "mat_back_target.csv",
    "floor_front_target.csv",
    "grass_front_target.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--downsample", type=int, default=12)
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def target_aligned_coordinates(
    x_m: np.ndarray | float,
    z_m: np.ndarray | float,
    target_x_m: float,
    target_z_m: float,
    start_x_m: float,
    start_z_m: float,
):
    """Target-centred coordinates with the initial robot pose aligned across panels."""
    dx = np.asarray(x_m) - target_x_m
    dz = np.asarray(z_m) - target_z_m
    start_dx = start_x_m - target_x_m
    start_dz = start_z_m - target_z_m
    current_angle = np.arctan2(start_dz, start_dx)
    target_angle = np.deg2rad(CANONICAL_TARGET_TO_START_ANGLE_DEG)
    angle = target_angle - current_angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return cos_a * dx - sin_a * dz, sin_a * dx + cos_a * dz


def aligned_trial_coordinates(trial, mask: np.ndarray, downsample: int):
    target_x = float(np.nanmedian(trial.motive.target_x_m[mask]))
    target_z = float(np.nanmedian(trial.motive.target_z_m[mask]))
    robot_x = trial.motive.robot_x_m[mask]
    robot_z = trial.motive.robot_z_m[mask]
    x, y = target_aligned_coordinates(
        robot_x[::downsample],
        robot_z[::downsample],
        target_x,
        target_z,
        float(robot_x[0]),
        float(robot_z[0]),
    )
    return x, y


def read_simulation_tracks(input_dir: Path) -> dict[str, list[dict[str, np.ndarray | float]]]:
    sim_dir = DEFAULT_PLOT_DIR / SIMULATION_DIR_NAME
    if not sim_dir.exists():
        sim_dir = input_dir / SIMULATION_DIR_NAME
    if not sim_dir.exists():
        return {}
    tracks: dict[str, list[dict[str, np.ndarray | float]]] = {}
    for path in sorted(sim_dir.glob("*.csv")):
        if path.name not in SIMULATION_FILENAMES:
            continue
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        condition = rows[0]["condition"]
        track = {
            "time_s": np.asarray([float(row["time_s"]) for row in rows], dtype=float),
            "x_m": np.asarray([float(row["x_m"]) for row in rows], dtype=float),
            "y_m": np.asarray([float(row["y_m"]) for row in rows], dtype=float),
            "target_x_m": float(rows[0]["target_x_m"]),
            "target_y_m": float(rows[0]["target_y_m"]),
            "friction_scale": float(rows[0]["friction_scale"]),
        }
        tracks.setdefault(condition, []).append(track)
    return tracks


def aligned_simulation_coordinates(track: dict[str, np.ndarray | float]):
    x_m = track["x_m"]
    y_m = track["y_m"]
    target_x_m = float(track["target_x_m"])
    target_y_m = float(track["target_y_m"])
    return target_aligned_coordinates(
        x_m,
        y_m,
        target_x_m,
        target_y_m,
        float(x_m[0]),
        float(y_m[0]),
    )


def rotated_square(center_x: float, center_y: float, side_m: float, angle_deg: float) -> np.ndarray:
    half = side_m * 0.5
    corners = np.asarray([
        [-half, -half],
        [half, -half],
        [half, half],
        [-half, half],
        [-half, -half],
    ])
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    return corners @ rotation.T + np.asarray([center_x, center_y])


def plot(input_dir: Path, output_dir: Path, downsample: int, save_pdf: bool) -> None:
    trials = load_trials(input_dir)
    simulation_tracks = read_simulation_tracks(input_dir)
    conditions = present_conditions(trials)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(PANEL_WIDTH * 3.0, PANEL_FIGURE_HEIGHT * 1.95),
        constrained_layout=True,
    )
    flat_axes = list(np.ravel(axes))

    for ax, condition in zip(flat_axes, conditions):
        condition_trials = [trial for trial in trials if trial.condition == condition]
        missing_labels = []
        for trial in condition_trials:
            mask = motive_valid_mask(trial)
            if not np.any(mask):
                missing_labels.append(f"{trial.label}: no OptiTrack CSV")
                continue
            x, y = aligned_trial_coordinates(trial, mask, downsample)
            color = trial_color(trial)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )
            ax.scatter(x[0], y[0], color="white", edgecolor="black", s=46, zorder=3)
            ax.scatter(x[-1], y[-1], color=color, edgecolor="black", s=46, zorder=3)

        for simulation_track in simulation_tracks.get(condition, []):
            sim_x, sim_y = aligned_simulation_coordinates(simulation_track)
            ax.plot(
                sim_x,
                sim_y,
                color=SIMULATION_COLOR,
                linewidth=1.3,
                linestyle="--",
                alpha=0.85,
                zorder=3.5,
            )

        target_border = rotated_square(0.0, 0.0, TARGET_BASE_SIZE_M, TARGET_BASE_ANGLE_DEG)
        ax.plot(
            target_border[:, 0],
            target_border[:, 1],
            color="#B00020",
            linewidth=2.0,
            linestyle="--",
            zorder=4,
        )
        ax.scatter(
            0.0,
            0.0,
            marker="*",
            color="#B00020",
            edgecolor="black",
            s=150,
            zorder=5,
        )
        if missing_labels:
            ax.text(
                0.03,
                0.03,
                "\n".join(missing_labels),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=LEGEND_SIZE,
                bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 3},
            )

        ax.set_title(condition_display_name(condition).replace(": ", ":\n"), fontsize=AXIS_TITLE_SIZE)
        ax.set_xlabel("Target-centred X [m]", fontsize=AXIS_LABEL_SIZE)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        trial_handles = [
            Line2D([0], [0], color=trial_color(trial), linewidth=2.0, label=trial.label)
            for trial in condition_trials
            if np.any(motive_valid_mask(trial))
        ]
        semantic_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="white",
                linestyle="none",
                label="robot start",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="0.35",
                linestyle="none",
                label="robot final position",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="#B00020",
                markeredgecolor="black",
                linestyle="none",
                markersize=10,
                label="estimated target centre",
            ),
            Line2D(
                [0],
                [0],
                color="#B00020",
                linewidth=2.0,
                linestyle="--",
                label="target borderline",
            ),
        ]
        if condition in simulation_tracks:
            semantic_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=SIMULATION_COLOR,
                    linewidth=1.3,
                    linestyle="--",
                    label="simulation",
                ),
            )
        ax.legend(handles=trial_handles + semantic_handles, fontsize=LEGEND_SIZE, loc="upper right")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    for ax in flat_axes[: len(conditions)]:
        ax.set_xlim(-0.25, 2.25)
        ax.set_ylim(-2.0, 0.5)

    for ax in flat_axes[len(conditions) :]:
        ax.axis("off")

    axes[0, 0].set_ylabel("Target-centred Y [m]", fontsize=AXIS_LABEL_SIZE)
    axes[1, 0].set_ylabel("Target-centred Y [m]", fontsize=AXIS_LABEL_SIZE)
    fig.suptitle("Target-Centred Robot Trajectories Across Experimental Conditions", fontsize=FIGURE_TITLE_SIZE)
    save_figure(fig, output_dir, "overhead_trajectories.png", save_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.input_dir, args.output_dir, args.downsample, args.pdf)


if __name__ == "__main__":
    main()
