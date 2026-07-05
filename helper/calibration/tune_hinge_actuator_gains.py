"""Tune MuJoCo hinge position-servo gains against real robot sweep data.

This script does not edit ``hinge.py``. It builds the Baby robot simulation,
patches the compiled MuJoCo model's position actuator gains in memory, runs the
same commanded-vs-actual sweep used by ``diagnose_hinge_tracking.py``, and ranks
``kp``/``kv`` pairs by how closely simulated joint motion matches the real
Robohat feedback summary.

Example:
    uv run --project ariel python helper/tune_hinge_actuator_gains.py
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.calibration.diagnose_hinge_tracking import build_model, run_actuator_sweep, summarize


DEFAULT_REAL_SUMMARY = (
    ROOT
    / "results/hardware_tests/measure_real_hinge_tracking_20260626_203353"
    / "real_hinge_tracking_summary.csv"
)
RESULTS_DIR = ROOT / "results/hinge_gain_tuning"


@dataclass(frozen=True)
class RealTarget:
    actuator_index: int
    actual_min_deg: float
    actual_max_deg: float
    actual_span_deg: float


@dataclass(frozen=True)
class TuningRow:
    kp: float
    kv: float
    actuator_index: int
    actuator_name: str
    real_min_deg: float
    real_max_deg: float
    real_span_deg: float
    sim_min_deg: float
    sim_max_deg: float
    sim_span_deg: float
    min_error_deg: float
    max_error_deg: float
    span_error_deg: float
    actuator_score: float


@dataclass(frozen=True)
class GainScore:
    kp: float
    kv: float
    mean_score: float
    mean_abs_min_error_deg: float
    mean_abs_max_error_deg: float
    mean_abs_span_error_deg: float
    max_actuator_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Baby robot MuJoCo hinge kp/kv gains.")
    parser.add_argument("--real-summary", type=Path, default=DEFAULT_REAL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--actuators", type=int, nargs="+", default=None)
    parser.add_argument("--kp-values", type=float, nargs="+", default=[1, 2, 5, 10, 20, 40, 80])
    parser.add_argument("--kv-values", type=float, nargs="+", default=[1, 2, 4, 8, 12, 16, 24])
    parser.add_argument("--ramp-time", type=float, default=0.75)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--span-weight", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def load_real_targets(path: Path) -> dict[int, RealTarget]:
    targets = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            actuator_index = int(row["actuator_index"])
            targets[actuator_index] = RealTarget(
                actuator_index=actuator_index,
                actual_min_deg=float(row["actual_min_deg"]),
                actual_max_deg=float(row["actual_max_deg"]),
                actual_span_deg=float(row["actual_span_deg"]),
            )
    return targets


def set_position_servo_gains(model: mujoco.MjModel, actuator_indices: list[int], kp: float, kv: float) -> None:
    for actuator_index in actuator_indices:
        model.actuator_gainprm[actuator_index, :] = 0.0
        model.actuator_biasprm[actuator_index, :] = 0.0
        model.actuator_gainprm[actuator_index, 0] = kp
        model.actuator_biasprm[actuator_index, :3] = [0.0, -kp, -kv]


def score_summary(real: RealTarget, sim_summary, span_weight: float) -> tuple[float, float, float, float]:
    min_error = sim_summary.actual_min_deg - real.actual_min_deg
    max_error = sim_summary.actual_max_deg - real.actual_max_deg
    span_error = sim_summary.actual_span_deg - real.actual_span_deg
    score = min_error**2 + max_error**2 + span_weight * span_error**2
    return min_error, max_error, span_error, score


def write_csv(path: Path, rows: list) -> None:
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_best(console: Console, scores: list[GainScore], top: int) -> None:
    table = Table(title=f"Best {min(top, len(scores))} MuJoCo Hinge Gain Candidates")
    table.add_column("rank", justify="right")
    table.add_column("kp", justify="right")
    table.add_column("kv", justify="right")
    table.add_column("mean score", justify="right")
    table.add_column("|min err|", justify="right")
    table.add_column("|max err|", justify="right")
    table.add_column("|span err|", justify="right")
    table.add_column("worst act", justify="right")
    for rank, row in enumerate(scores[:top], start=1):
        table.add_row(
            str(rank),
            f"{row.kp:g}",
            f"{row.kv:g}",
            f"{row.mean_score:.1f}",
            f"{row.mean_abs_min_error_deg:.1f}",
            f"{row.mean_abs_max_error_deg:.1f}",
            f"{row.mean_abs_span_error_deg:.1f}",
            f"{row.max_actuator_score:.1f}",
        )
    console.print(table)


def main() -> None:
    install()
    console = Console()
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir if args.output_dir is not None else RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    real_targets = load_real_targets(args.real_summary)
    model, data = build_model()
    actuator_indices = args.actuators if args.actuators is not None else sorted(real_targets)

    sweep_args = argparse.Namespace(
        ramp_time=args.ramp_time,
        hold_time=args.hold_time,
        settle_time=args.settle_time,
        sample_hz=args.sample_hz,
    )
    detail_rows = []
    gain_scores = []

    for kp in args.kp_values:
        for kv in args.kv_values:
            set_position_servo_gains(model, actuator_indices, kp, kv)
            gain_rows = []
            for actuator_index in actuator_indices:
                rows = run_actuator_sweep(model, data, actuator_index, sweep_args)
                sim_summary = summarize(rows)
                real = real_targets[actuator_index]
                min_error, max_error, span_error, score = score_summary(real, sim_summary, args.span_weight)
                row = TuningRow(
                    kp=kp,
                    kv=kv,
                    actuator_index=actuator_index,
                    actuator_name=sim_summary.actuator_name,
                    real_min_deg=real.actual_min_deg,
                    real_max_deg=real.actual_max_deg,
                    real_span_deg=real.actual_span_deg,
                    sim_min_deg=sim_summary.actual_min_deg,
                    sim_max_deg=sim_summary.actual_max_deg,
                    sim_span_deg=sim_summary.actual_span_deg,
                    min_error_deg=min_error,
                    max_error_deg=max_error,
                    span_error_deg=span_error,
                    actuator_score=score,
                )
                detail_rows.append(row)
                gain_rows.append(row)
            gain_scores.append(
                GainScore(
                    kp=kp,
                    kv=kv,
                    mean_score=float(np.mean([row.actuator_score for row in gain_rows])),
                    mean_abs_min_error_deg=float(np.mean([abs(row.min_error_deg) for row in gain_rows])),
                    mean_abs_max_error_deg=float(np.mean([abs(row.max_error_deg) for row in gain_rows])),
                    mean_abs_span_error_deg=float(np.mean([abs(row.span_error_deg) for row in gain_rows])),
                    max_actuator_score=float(np.max([row.actuator_score for row in gain_rows])),
                ),
            )
            console.print(f"tested kp={kp:g}, kv={kv:g}")

    gain_scores.sort(key=lambda row: row.mean_score)
    metadata = {
        "run_id": run_id,
        "real_summary": str(args.real_summary),
        "actuators": actuator_indices,
        "kp_values": args.kp_values,
        "kv_values": args.kv_values,
        "ramp_time": args.ramp_time,
        "hold_time": args.hold_time,
        "settle_time": args.settle_time,
        "sample_hz": args.sample_hz,
        "span_weight": args.span_weight,
        "physics_timestep": float(model.opt.timestep),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_csv(out_dir / "gain_tuning_details.csv", detail_rows)
    write_csv(out_dir / "gain_tuning_scores.csv", gain_scores)
    print_best(console, gain_scores, args.top)
    console.print(f"Saved gain tuning results -> {out_dir}")


if __name__ == "__main__":
    main()
