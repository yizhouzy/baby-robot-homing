"""Plot Robohat battery-estimator behavior from a fixed phase recording."""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


PHASE_COLORS = {
    "IDLE": "#EAEAEA",
    "APPROACHING(forward)": "#D9EAD3",
    "SEARCHING(Left)": "#D9EAF7",
    "SEARCHING(Right)": "#FCE5CD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot battery phase sequence results.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--filename", default="battery_phase_limitation.png")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def numeric_column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def text_column(rows: list[dict], key: str) -> list[str]:
    return [row[key] for row in rows]


def phase_spans(time_s: np.ndarray, phases: list[str]) -> list[tuple[float, float, str]]:
    spans = []
    start_index = 0
    for index in range(1, len(phases)):
        if phases[index] != phases[index - 1]:
            spans.append((time_s[start_index], time_s[index], phases[index - 1]))
            start_index = index
    spans.append((time_s[start_index], time_s[-1], phases[-1]))
    return spans


def add_phase_background(ax, spans: list[tuple[float, float, str]]) -> None:
    for start_s, end_s, phase in spans:
        ax.axvspan(
            start_s,
            end_s,
            color=PHASE_COLORS.get(phase, "#EEEEEE"),
            alpha=0.55,
            linewidth=0,
        )
        ax.text(
            (start_s + end_s) * 0.5,
            0.97,
            phase,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            rotation=0,
        )


def padded_limits(values: np.ndarray, padding_fraction: float = 0.18) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span == 0.0:
        span = max(abs(high), 1.0)
    padding = span * padding_fraction
    return low - padding, high + padding


def summarize(time_s: np.ndarray, voltage_v: np.ndarray, percentage: np.ndarray) -> dict:
    d_voltage_mv = np.diff(voltage_v) * 1000.0
    d_percentage = np.diff(percentage)
    nonmonotonic = (d_percentage > 0.0) & (d_voltage_mv <= 0.0)
    return {
        "duration_s": float(time_s[-1] - time_s[0]),
        "voltage_start_v": float(voltage_v[0]),
        "voltage_end_v": float(voltage_v[-1]),
        "voltage_min_v": float(np.min(voltage_v)),
        "voltage_max_v": float(np.max(voltage_v)),
        "percentage_start": int(percentage[0]),
        "percentage_end": int(percentage[-1]),
        "percentage_min": int(np.min(percentage)),
        "percentage_max": int(np.max(percentage)),
        "positive_percentage_steps_while_voltage_not_increasing": int(np.sum(nonmonotonic)),
        "largest_positive_percentage_step": float(np.max(np.maximum(d_percentage, 0.0))),
    }


def plot(run_dir: Path, out_dir: Path, filename: str, save_pdf: bool) -> None:
    rows = read_rows(run_dir / "battery_phase_samples.csv")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    time_s = numeric_column(rows, "elapsed_s")
    voltage_v = numeric_column(rows, "voltage_v")
    percentage = numeric_column(rows, "percentage")
    phases = text_column(rows, "phase")
    spans = phase_spans(time_s, phases)
    summary = summarize(time_s, voltage_v, percentage)

    fig, ax_voltage = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    fig.suptitle("Robot Battery Readout During Different Behavior Phases", fontsize=13)

    add_phase_background(ax_voltage, spans)
    voltage_line = ax_voltage.plot(
        time_s,
        voltage_v,
        color="#1F4E79",
        linewidth=2.0,
        label="battery voltage [V]",
    )
    ax_voltage.set_ylabel("battery voltage [V]", color="#1F4E79")
    ax_voltage.tick_params(axis="y", labelcolor="#1F4E79")
    ax_voltage.set_ylim(*padded_limits(voltage_v))
    ax_voltage.grid(True, alpha=0.25)

    ax_percentage = ax_voltage.twinx()
    percentage_line = ax_percentage.step(
        time_s,
        percentage,
        where="post",
        color="#C00000",
        linewidth=1.8,
        label="Robohat accu capacity [%]",
    )
    ax_percentage.set_ylabel("Robohat capacity [%]", color="#C00000")
    ax_percentage.tick_params(axis="y", labelcolor="#C00000")
    ax_percentage.set_ylim(*padded_limits(percentage))
    ax_voltage.set_xlabel("time [s]")

    lines = voltage_line + percentage_line
    labels = [line.get_label() for line in lines]
    ax_voltage.legend(
        lines,
        labels,
        loc="center left",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / filename
    fig.savefig(png_path, dpi=180)
    if save_pdf:
        fig.savefig(png_path.with_suffix(".pdf"))
    plt.close(fig)

    summary_path = out_dir / "battery_phase_summary.json"
    summary_path.write_text(json.dumps({**summary, "metadata": metadata}, indent=2, sort_keys=True) + "\n")
    print(f"Saved plot: {png_path}")
    if save_pdf:
        print(f"Saved PDF:  {png_path.with_suffix('.pdf')}")
    print(f"Saved summary: {summary_path}")
    print(
        "Non-monotonic capacity evidence: "
        f"{summary['positive_percentage_steps_while_voltage_not_increasing']} upward percentage "
        "steps occurred when voltage did not increase.",
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.run_dir / "plots"
    plot(args.run_dir, out_dir, args.filename, args.pdf)


if __name__ == "__main__":
    main()
