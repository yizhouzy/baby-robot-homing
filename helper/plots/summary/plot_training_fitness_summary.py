"""Summarize fitness histories over seeds for the final gait controllers."""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.plots.real_world_exp_data import DEFAULT_PLOT_DIR, save_figure


GAIT_SPECS = {
    "forward": {
        "root": Path("results/gait_cpg/final_forward"),
        "history_glob": "gait_fitness_history_*.npy",
        "global_glob": "gait_global_best_history_*.npy",
        "meta_glob": "gait_meta_*.npz",
        "color": "#A87400",
        "title": "Forward gait",
    },
    "left": {
        "root": Path("results/left_cpg/final_left"),
        "history_glob": "spin_fitness_history_*.npy",
        "global_glob": "spin_global_best_history_*.npy",
        "meta_glob": "spin_meta_*.npz",
        "color": "#A3431C",
        "title": "Left spin",
    },
    "right": {
        "root": Path("results/right_cpg/final_right"),
        "history_glob": "spin_fitness_history_*.npy",
        "global_glob": "spin_global_best_history_*.npy",
        "meta_glob": "spin_meta_*.npz",
        "color": "#1F5F8B",
        "title": "Right spin",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "summary")
    parser.add_argument("--max-generations", type=int, default=300)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def seed_label(path: Path, meta) -> str:
    if "seed" in meta.files:
        return f"seed {int(meta['seed'])}"
    match = re.search(r"seed(\d+)", path.name)
    return f"seed {match.group(1)}" if match else path.name


def evals_per_generation(meta) -> int:
    population = int(meta["population"]) if "population" in meta.files else 50
    repeats = int(meta["eval_repeats"]) if "eval_repeats" in meta.files else 1
    return population * repeats


def load_runs(gait: str, include_failures: bool, max_generations: int) -> list[dict]:
    spec = GAIT_SPECS[gait]
    runs = []
    for run_dir in sorted(spec["root"].iterdir()):
        if not run_dir.is_dir():
            continue
        if "failure" in run_dir.name.lower() and not include_failures:
            continue
        history_paths = sorted(run_dir.glob(spec["history_glob"]))
        global_paths = sorted(run_dir.glob(spec["global_glob"]))
        meta_paths = sorted(run_dir.glob(spec["meta_glob"]))
        if not history_paths or not global_paths or not meta_paths:
            continue
        meta = np.load(meta_paths[-1], allow_pickle=True)
        history = np.load(history_paths[-1])[:max_generations].astype(float)
        global_history = np.load(global_paths[-1])[:max_generations].astype(float)
        generations = np.arange(1, len(global_history) + 1, dtype=float)
        runs.append({
            "label": seed_label(run_dir, meta),
            "history": history,
            "global_history": global_history,
            "generations": generations,
            "population": int(meta["population"]) if "population" in meta.files else 50,
            "eval_repeats": int(meta["eval_repeats"]) if "eval_repeats" in meta.files else 1,
            "evals_per_generation": evals_per_generation(meta),
        })
    return runs


def stacked_curves(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(run[key]) for run in runs)
    values = np.vstack([run[key][:min_len] for run in runs])
    evals_per_gen = int(np.median([run["evals_per_generation"] for run in runs]))
    x = np.arange(1, min_len + 1, dtype=float) * evals_per_gen
    return x, values


def common_or_range(values: list[int]) -> str:
    unique = sorted(set(values))
    if len(unique) == 1:
        return f"{unique[0]:,}"
    return f"{unique[0]:,}-{unique[-1]:,}"


def run_config_text(runs: list[dict]) -> str:
    generations = [len(run["global_history"]) for run in runs]
    populations = [run["population"] for run in runs]
    repeats = [run["eval_repeats"] for run in runs]
    total_evals = [
        len(run["global_history"]) * run["population"] * run["eval_repeats"]
        for run in runs
    ]
    return "\n".join([
        f"Generations: {common_or_range(generations)}",
        f"Population size: {common_or_range(populations)}",
        f"Evaluation repeats: {common_or_range(repeats)}",
        f"Total evaluations: {common_or_range(total_evals)}",
    ])


def plot(args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.0))
    spin_limits = []
    right_config_text = ""
    for ax, gait in zip(axes, GAIT_SPECS):
        spec = GAIT_SPECS[gait]
        runs = load_runs(gait, args.include_failures, args.max_generations)
        color = spec["color"]
        for run in runs:
            x = run["generations"] * run["evals_per_generation"]
            ax.step(
                x,
                run["global_history"],
                where="post",
                color="#9C9C9C",
                linewidth=1.0,
                alpha=0.42,
            )
        mean_x, global_values = stacked_curves(runs, "global_history")
        mean_global = np.mean(global_values, axis=0)
        ax.step(
            mean_x,
            mean_global,
            where="post",
            color=color,
            linewidth=2.9,
            label="mean incumbent objective",
        )
        if gait in {"left", "right"}:
            spin_limits.append((float(np.min(global_values)), float(np.max(global_values))))
        ax.text(
            0.03,
            0.05,
            f"{len(runs)} seeds",
            transform=ax.transAxes,
            fontsize=9,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.86, "pad": 2.0},
        )
        if gait == "right":
            right_config_text = run_config_text(runs)
        ax.set_title(spec["title"], fontsize=13)
        ax.set_xlabel("cumulative fitness evaluations")
        ax.grid(alpha=0.25)
    spin_low = min(limit[0] for limit in spin_limits)
    spin_high = max(limit[1] for limit in spin_limits)
    spin_padding = (spin_high - spin_low) * 0.06
    axes[1].set_ylim(spin_low - spin_padding, spin_high + spin_padding)
    axes[2].set_ylim(spin_low - spin_padding, spin_high + spin_padding)
    axes[0].set_ylabel("fitness value (lower is better)")
    handles = [
        Line2D([0], [0], color="#9C9C9C", linewidth=1.0, alpha=0.52, label="global best fitness per seed"),
        Line2D([0], [0], color="0.15", linewidth=2.9, label="mean global best fitness across seeds"),
    ]
    axes[2].legend(handles=handles, fontsize=8.0, loc="upper right", frameon=True)
    axes[2].text(
        0.98,
        0.72,
        right_config_text,
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color="0.18",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "0.72",
            "alpha": 0.90,
        },
    )
    fig.suptitle("CMA-ES Convergence of Gait Controllers", fontsize=15)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.82, wspace=0.22)
    save_figure(fig, args.output_dir, "training_fitness_summary.png", args.pdf)
    plt.close(fig)


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
