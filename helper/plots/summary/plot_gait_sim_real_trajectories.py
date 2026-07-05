"""Compare simulated and real trajectories for the three deployed gaits."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.artifacts import DEFAULT_GAIT_MODELS
from robot_control.controllers import load_gait_network, sanitize_action
from robot_control.evaluation import build_training_model
from helper.plots.real_world_exp_data import DEFAULT_PLOT_DIR, save_figure


GAIT_COLORS = {
    "forward": "#B88400",
    "left": "#B14A22",
    "right": "#1F6F9F",
}
SIM_COLOR = "#4F4F4F"
DEPLOYMENT_SPEEDS = {"forward": 0.60, "left": 0.25, "right": 0.25}
REAL_FILE_PATTERNS = {
    ("forward", 1.0): "Take forward mat speed 1.0*.csv",
    ("forward", 0.60): "Take forward mat speed 0.6.csv",
    ("left", 1.0): "Take left spin mat speed 1.0.csv",
    ("left", 0.25): "Take left spin mat speed 0.25.csv",
    ("right", 1.0): "Take right spin mat speed 1.0.csv",
    ("right", 0.25): "Take right spin mat speed 0.25.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=Path("results/5_optictrack_data_backup"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def read_robot_track(path: Path, duration_s: float) -> np.ndarray:
    time_s = []
    xy = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        reading_samples = False
        for row in reader:
            if not row:
                continue
            if row[0] == "Frame":
                reading_samples = True
                continue
            if not reading_samples:
                continue
            points = []
            for index in range(2, len(row) - 2, 3):
                if row[index] == "" or row[index + 1] == "" or row[index + 2] == "":
                    continue
                x_m = float(row[index])
                y_m = float(row[index + 1])
                z_m = float(row[index + 2])
                if 0.05 <= y_m <= 0.30:
                    points.append((x_m, z_m))
            if points:
                arr = np.asarray(points, dtype=float)
                time_s.append(float(row[1]))
                xy.append(np.median(arr, axis=0))
    time = np.asarray(time_s, dtype=float)
    track = np.asarray(xy, dtype=float)
    mask = time <= time[0] + duration_s
    track = track[mask]
    return track - track[0]


def simulate_gait(gait: str, speed: float, duration_s: float) -> np.ndarray:
    model_path, meta_path = DEFAULT_GAIT_MODELS[gait]
    network, meta, _ = load_gait_network(model_path, meta_path)
    model, data = build_training_model()
    mujoco.mj_resetData(model, data)
    network.reset_hidden()
    control_freq = max(1, int(round(float(meta["dt"]) / model.opt.timestep)))
    current_action = np.zeros(model.nu, dtype=np.float32)
    xy = []
    step = 0
    while data.time < duration_s:
        if step % control_freq == 0:
            current_action = sanitize_action(network.forward(turn=0.0, speed=speed), model)
            xy.append(np.asarray(data.qpos[:2].copy(), dtype=float))
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1
    track = np.asarray(xy, dtype=float)
    return track - track[0]


def collect_tracks(args: argparse.Namespace) -> dict[str, dict[str, dict[float, list[np.ndarray]]]]:
    tracks: dict[str, dict[str, dict[float, list[np.ndarray]]]] = {}
    for gait in ["forward", "left", "right"]:
        tracks[gait] = {"simulation": {}, "hardware": {}}
        for speed in [1.0, DEPLOYMENT_SPEEDS[gait]]:
            tracks[gait]["simulation"][speed] = [simulate_gait(gait, speed, args.duration)]
            real_tracks = [
                read_robot_track(path, args.duration)
                for path in sorted(args.real_dir.glob(REAL_FILE_PATTERNS[(gait, speed)]))
            ]
            tracks[gait]["hardware"][speed] = real_tracks
    return tracks


def limits_for(track_groups: list[list[np.ndarray]]) -> tuple[tuple[float, float], tuple[float, float]]:
    arrays = [track for group in track_groups for track in group if len(track) > 0]
    points = np.vstack(arrays)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centre = (mins + maxs) * 0.5
    span = float(max(np.max(maxs - mins), 0.35))
    padding = span * 0.10
    half = span * 0.5 + padding
    return (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half)


def gait_limits(tracks: dict[str, dict[str, dict[float, list[np.ndarray]]]]) -> dict[str, tuple]:
    forward_groups = [
        tracks["forward"][source][speed]
        for source in ["simulation", "hardware"]
        for speed in tracks["forward"][source]
    ]
    spin_groups = [
        tracks[gait][source][speed]
        for gait in ["left", "right"]
        for source in ["simulation", "hardware"]
        for speed in tracks[gait][source]
    ]
    spin_limits = limits_for(spin_groups)
    return {
        "forward": limits_for(forward_groups),
        "left": spin_limits,
        "right": spin_limits,
    }


def draw_track(ax, track: np.ndarray, color: str, speed: float, alpha: float, zorder: int) -> None:
    linestyle = "-" if speed < 1.0 else (0, (4, 2))
    linewidth = 2.5 if speed < 1.0 else 1.65
    ax.plot(
        track[:, 0],
        track[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.scatter(track[0, 0], track[0, 1], color="white", edgecolor="black", s=36, zorder=6)
    ax.scatter(track[-1, 0], track[-1, 1], color=color, edgecolor="black", s=34, zorder=6, alpha=alpha)


def plot(args: argparse.Namespace) -> None:
    tracks = collect_tracks(args)
    limits = gait_limits(tracks)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2))
    for ax, gait in zip(axes, ["forward", "left", "right"]):
        for speed in [1.0, DEPLOYMENT_SPEEDS[gait]]:
            for track in tracks[gait]["simulation"][speed]:
                draw_track(ax, track, SIM_COLOR, speed, 0.72 if speed == 1.0 else 0.9, 2)
            for track in tracks[gait]["hardware"][speed]:
                draw_track(ax, track, GAIT_COLORS[gait], speed, 0.82 if speed == 1.0 else 0.98, 4)
        ax.set_xlim(*limits[gait][0])
        ax.set_ylim(*limits[gait][1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_title(f"{gait.capitalize()} gait", fontsize=13)
        ax.set_xlabel("start-aligned X [m]")
    axes[0].set_ylabel("start-aligned Y [m]")

    handles = [
        Line2D([0], [0], color=SIM_COLOR, linewidth=2.1, label="simulation"),
        Line2D([0], [0], color="#333333", linewidth=2.1, label="hardware uses gait colour"),
        Line2D([0], [0], color="#333333", linestyle=(0, (4, 2)), linewidth=1.8, label="speed 1.0"),
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.5, label="deployment speed"),
    ]
    axes[-1].legend(handles=handles, fontsize=8.4, loc="best")
    fig.suptitle(
        f"Trajectory Comparison of Deployed Gait Controllers in Simulation and Hardware ({args.duration:.0f} s)",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.80, wspace=0.24)
    save_figure(fig, args.output_dir, "gait_sim_real_trajectories.png", args.pdf)
    plt.close(fig)


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
