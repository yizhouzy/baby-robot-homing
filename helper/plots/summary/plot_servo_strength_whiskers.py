"""Plot simulated and measured servo angle ranges as neutral-centred bars."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.plots.real_world_exp_data import DEFAULT_PLOT_DIR, save_figure


DEFAULT_SIM = Path("results/2_calibration_sim/hinge_tracking/check_current_hinge/hinge_tracking_summary.csv")
DEFAULT_REAL = Path(
    "results/3_calibration_real/2_servo_channel_and_strength/"
    "measure_real_hinge_tracking_20260626_203353/real_hinge_tracking_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-summary", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--real-summary", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def read_summary(path: Path) -> dict[int, dict]:
    with path.open(newline="") as f:
        return {int(row["actuator_index"]): row for row in csv.DictReader(f)}


def simulation_servo_limits(row: dict) -> tuple[float, float]:
    """Display MuJoCo joint limits as equivalent 90-degree-neutral servo angles."""
    low = 90.0 + float(row["actual_min_deg"])
    high = 90.0 + float(row["actual_max_deg"])
    return min(low, high), max(low, high)


def hardware_servo_limits(row: dict) -> tuple[float, float]:
    """Use calibrated Robohat readback angles for the physical servo range."""
    low = float(row["readback_servo_min_deg"])
    high = float(row["readback_servo_max_deg"])
    return min(low, high), max(low, high)


def draw_range_bar(
    ax,
    x: float,
    low: float,
    high: float,
    color: str,
    width: float,
    label: str | None = None,
) -> None:
    ax.bar(
        x,
        high - low,
        bottom=low,
        width=width,
        color=color,
        edgecolor=color,
        alpha=0.34,
        linewidth=1.4,
        zorder=3,
        label=label,
    )
    ax.hlines([low, high], x - width / 2, x + width / 2, color=color, linewidth=2.0, zorder=4)


def plot(args: argparse.Namespace) -> None:
    sim = read_summary(args.sim_summary)
    real = read_summary(args.real_summary)
    actuators = sorted(set(sim) & set(real))
    x = np.arange(len(actuators), dtype=float)
    sim_color = "#5F6368"
    real_color = "#4A90E2"
    offset = 0.17
    width = 0.26

    fig, ax = plt.subplots(figsize=(10.6, 5.5), constrained_layout=True)
    for index, actuator in enumerate(actuators):
        sim_low, sim_high = simulation_servo_limits(sim[actuator])
        real_low, real_high = hardware_servo_limits(real[actuator])
        draw_range_bar(
            ax,
            x[index] - offset,
            sim_low,
            sim_high,
            sim_color,
            width,
            "simulation achieved range" if index == 0 else None,
        )
        draw_range_bar(
            ax,
            x[index] + offset,
            real_low,
            real_high,
            real_color,
            width,
            "hardware achieved range" if index == 0 else None,
        )

    ax.axhline(90.0, color="#222222", linewidth=1.3, alpha=0.82, label="neutral command: 90 deg")
    ax.set_xticks(x)
    ax.set_xticklabels([str(index) for index in actuators])
    ax.set_xlabel("servo index")
    ax.set_ylabel("achieved servo angle [deg]")
    ax.set_ylim(-8.0, 188.0)
    ax.grid(axis="y", alpha=0.24)
    ax.axhline(
        0.0,
        color="0.32",
        linewidth=1.1,
        linestyle=(0, (3, 2)),
        alpha=0.85,
        label="command limits: 0 and 180 deg",
    )
    ax.axhline(180.0, color="0.32", linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.85)
    ax.set_title("Servo Range Tracking in Simulation and Hardware")

    ax_rad = ax.secondary_yaxis("right", functions=(np.deg2rad, np.rad2deg))
    ax_rad.set_ylabel("achieved servo angle [rad]")
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    save_figure(fig, args.output_dir, "servo_strength_whiskers.png", args.pdf)
    plt.close(fig)


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
