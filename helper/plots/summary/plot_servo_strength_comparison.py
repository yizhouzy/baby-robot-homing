"""Plot simulated and measured hinge tracking strength per actuator."""
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
        rows = list(csv.DictReader(f))
    return {int(row["actuator_index"]): row for row in rows}


def plot(args: argparse.Namespace) -> None:
    sim = read_summary(args.sim_summary)
    real = read_summary(args.real_summary)
    actuators = sorted(set(sim) & set(real))
    x = np.arange(len(actuators), dtype=float)
    width = 0.34
    sim_ratio = np.asarray([float(sim[index]["span_ratio"]) for index in actuators])
    real_ratio = np.asarray([float(real[index]["span_ratio"]) for index in actuators])
    sim_span = np.asarray([float(sim[index]["actual_span_deg"]) for index in actuators])
    real_span = np.asarray([float(real[index]["actual_span_deg"]) for index in actuators])

    fig, ax = plt.subplots(figsize=(10.2, 5.4), constrained_layout=True)
    ax.bar(x - width / 2, sim_ratio, width, color="#8A8A8A", label="simulation span ratio")
    ax.bar(x + width / 2, real_ratio, width, color="#4C78A8", label="hardware span ratio")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("actual span / commanded span")
    ax.set_xlabel("actuator index")
    ax.set_xticks(x)
    ax.set_xticklabels([str(index) for index in actuators])
    ax.grid(axis="y", alpha=0.25)

    ax_span = ax.twinx()
    ax_span.plot(x - width / 2, sim_span, "o", color="#444444", markersize=5, label="simulation actual span")
    ax_span.plot(x + width / 2, real_span, "D", color="#1F4E79", markersize=5, label="hardware actual span")
    ax_span.set_ylabel("actual span [deg]")
    ax_span.set_ylim(0.0, max(float(np.max(sim_span)), float(np.max(real_span))) * 1.18)

    handles, labels = ax.get_legend_handles_labels()
    span_handles, span_labels = ax_span.get_legend_handles_labels()
    ax.legend(handles + span_handles, labels + span_labels, fontsize=9, loc="lower right")
    ax.set_title("Servo Strength Calibration: Simulated vs Hardware Hinge Tracking")
    save_figure(fig, args.output_dir, "servo_strength_comparison.png", args.pdf)
    plt.close(fig)


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
