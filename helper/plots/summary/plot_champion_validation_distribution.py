"""Visualize champion robustness over held-out simulation configurations."""
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


DEFAULT_CAPTION = Path("final_report/assets/snippets/champion_validation_caption.tex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--caption-path", type=Path, default=DEFAULT_CAPTION)
    parser.add_argument("--winner", default=None)
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def discover_input_csv() -> Path:
    candidates = sorted(Path("results/model_selection").glob("**/*validation*.csv"))
    candidates.extend(sorted(Path("results/model_selection").glob("**/*summary*.csv")))
    if not candidates:
        raise FileNotFoundError("No model-selection validation CSV found; pass --input-csv.")
    return candidates[-1]


def read_rows(path: Path) -> tuple[list[dict], str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if "success" in rows[0]:
        return rows, "success"
    if "score" in rows[0]:
        return rows, "score"
    raise ValueError("Validation CSV must contain either a 'success' or 'score' column.")


def champion_key(row: dict) -> str:
    gait = row.get("gait", "").strip()
    champion = row.get("champion", "").strip() or row.get("seed", "").strip()
    return f"{gait}: {champion}" if gait else champion


def write_caption(path: Path, metric_name: str) -> None:
    if metric_name == "success":
        y_text = "binary success"
    else:
        y_text = "continuous performance score"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "Champion validation distribution across held-out simulation configurations. "
            f"The y-axis reports {y_text}. Success is defined as reaching within "
            "25\\,cm of the target before timeout without falling; the selected "
            "controller is the champion with consistently high and low-variance "
            "performance across configurations.\n"
        ),
    )


def plot(args: argparse.Namespace) -> None:
    input_csv = args.input_csv if args.input_csv is not None else discover_input_csv()
    rows, metric = read_rows(input_csv)
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(champion_key(row), []).append(float(row[metric]))
    labels = sorted(groups)
    values = [groups[label] for label in labels]
    means = {label: float(np.mean(groups[label])) for label in labels}
    winner = args.winner if args.winner is not None else max(labels, key=lambda label: means[label])

    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(labels)), 5.2), constrained_layout=True)
    box = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
    for label, patch in zip(labels, box["boxes"]):
        patch.set_facecolor("#F2F2F2" if label != winner else "#FFE6A6")
        patch.set_edgecolor("#444444" if label != winner else "#B8860B")
        patch.set_linewidth(1.2 if label != winner else 2.2)
    rng = np.random.default_rng(42)
    for index, label in enumerate(labels, start=1):
        jitter = rng.normal(0.0, 0.035, size=len(groups[label]))
        color = "#555555" if label != winner else "#B8860B"
        ax.scatter(np.full(len(groups[label]), index) + jitter, groups[label], color=color, s=28, alpha=0.78, zorder=3)
    ax.set_ylabel("success" if metric == "success" else "performance score")
    ax.set_xlabel("candidate champion")
    ax.set_title("Champion Robustness Across Held-Out Simulation Configurations")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=25)
    save_figure(fig, args.output_dir, "champion_validation_distribution.png", args.pdf)
    plt.close(fig)
    write_caption(args.caption_path, metric)
    print(f"Input: {input_csv}")
    print(f"Winner highlighted: {winner}")
    print(f"Caption: {args.caption_path}")


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
