"""Plot diagnostics from a ``run_gait_no_camera.py`` result folder."""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot physical gait run diagnostics.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    columns = rows[0].keys()
    data = {}
    for column in columns:
        values = []
        for row in rows:
            text = row[column]
            try:
                values.append(np.nan if text == "" else float(text))
            except ValueError:
                values.append(np.nan)
        data[column] = np.asarray(values, dtype=float)
    return data


def select_time_window(data: dict[str, np.ndarray], seconds: float) -> dict[str, np.ndarray]:
    keep = data["elapsed_s"] <= seconds
    return {key: value[keep] for key, value in data.items()}


def wrapped_robohat_value(values: np.ndarray) -> np.ndarray:
    """Show Robohat unsigned wrapped IMU values around zero for readability."""
    return np.where(values > 3276.8, values - 6553.6, values)


def metadata_title(metadata: dict) -> str:
    return (
        f"{metadata.get('gait', 'gait')}  speed={metadata.get('speed')}  "
        f"amp={metadata.get('amplitude_scale')}  model={Path(metadata.get('model', '')).name}"
    )


def plot_commands(run_dir: Path, out_dir: Path, seconds: float, metadata: dict) -> None:
    data = select_time_window(read_csv(run_dir / "servo_samples.csv"), seconds)
    t = data["elapsed_s"]
    joint_deg = np.vstack([
        np.degrees(data[f"joint_{i}_rad"])
        for i in range(8)
    ])
    servo_deg = np.vstack([data[f"servo_{i}_command_deg"] for i in range(8)])

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(metadata_title(metadata))

    for i in range(8):
        axes[0].plot(t, joint_deg[i], label=f"joint {i}", linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Controller Joint Commands")
    axes[0].set_ylabel("joint angle [deg]")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i in range(8):
        axes[1].plot(t, servo_deg[i], label=f"servo {i}", linewidth=1.2)
    axes[1].axhline(90.0, color="black", linestyle="--", linewidth=0.9, label="raw 90 deg")
    axes[1].set_title("Raw Robohat Servo Commands After Neutral/Sign Mapping")
    axes[1].set_ylabel("raw servo angle [deg]")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    peak_to_peak = np.nanmax(joint_deg, axis=1) - np.nanmin(joint_deg, axis=1)
    axes[2].bar(np.arange(8), peak_to_peak)
    axes[2].set_title("Per-Joint Command Range")
    axes[2].set_xlabel("joint index")
    axes[2].set_ylabel("peak-to-peak [deg]")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.savefig(out_dir / "gait_commands.png", dpi=160)
    plt.close(fig)


def plot_timing_imu(run_dir: Path, out_dir: Path, seconds: float, metadata: dict) -> None:
    servo = select_time_window(read_csv(run_dir / "servo_samples.csv"), seconds)
    imu_path = run_dir / "imu_samples.csv"
    battery_path = run_dir / "battery_samples.csv"

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(metadata_title(metadata))

    axes[0].plot(servo["elapsed_s"], servo["loop_dt_s"] * 1000.0, linewidth=1.0)
    axes[0].set_title("Control Loop Timing")
    axes[0].set_ylabel("loop dt [ms]")
    axes[0].grid(True, alpha=0.3)

    imu = select_time_window(read_csv(imu_path), seconds)
    for axis in ("x", "y", "z"):
        axes[1].plot(
            imu["elapsed_s"],
            wrapped_robohat_value(imu[f"acc_{axis}"]),
            label=f"acc {axis}",
        )
    axes[1].set_title("IMU Acceleration, Wrapped For Readability")
    axes[1].set_ylabel("driver units")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for axis in ("x", "y", "z"):
        axes[2].plot(
            imu["elapsed_s"],
            wrapped_robohat_value(imu[f"gyro_{axis}"]),
            label=f"gyro {axis}",
        )
    axes[2].set_title("IMU Gyro, Wrapped For Readability")
    axes[2].set_ylabel("driver units")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    battery = select_time_window(read_csv(battery_path), seconds)
    axes[3].plot(battery["elapsed_s"], battery["voltage_v"], marker="o", linewidth=1.0)
    axes[3].set_title("Battery Voltage")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("voltage [V]")
    axes[3].grid(True, alpha=0.3)

    fig.savefig(out_dir / "gait_timing_imu_battery.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir if args.out_dir is not None else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    plot_commands(run_dir, out_dir, args.seconds, metadata)
    plot_timing_imu(run_dir, out_dir, args.seconds, metadata)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
