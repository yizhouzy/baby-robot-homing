"""Select deployable CPG models from saved training outputs.

This script does not generate training commands and does not re-run simulation.
It reads each candidate's saved metadata and validation metrics, then ranks the
models using deployment-oriented criteria.

Forward ranking:
1. successful run first: reached the saved target, did not fall, contact is OK;
2. lower horizontal drift;
3. lower remaining target distance;
4. faster time to target;
5. lower self-contact.

Spin ranking:
1. successful run first: enough rotation, drift OK, did not fall, contact is OK;
2. larger rotation;
3. lower drift;
4. lower self-contact.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re

import numpy as np


FORWARD = "forward"
LEFT = "left"
RIGHT = "right"
TASKS = (FORWARD, LEFT, RIGHT)
OUTPUT_ROOT = Path("results/model_selection")


@dataclass(frozen=True)
class Candidate:
    task: str
    direction: str
    model_path: Path
    meta_path: Path
    metrics_path: Path | None
    run_dir: Path
    run_id: str
    seed: int


SUMMARY_FIELDS = [
    "task",
    "direction",
    "rank",
    "run_id",
    "seed",
    "sigma",
    "budget",
    "duration_s",
    "eval_repeats",
    "target_y_m",
    "model_path",
    "meta_path",
    "metrics_path",
    "success",
    "reached",
    "not_fallen",
    "contact_ok",
    "score",
    "min_distance_to_target_m",
    "time_to_target_s",
    "forward_progress_m",
    "horizontal_drift_m",
    "path_length_m",
    "path_efficiency",
    "final_x_m",
    "final_y_m",
    "final_z_m",
    "min_z_m",
    "max_abs_roll_deg",
    "max_abs_pitch_deg",
    "self_contact_pairs",
    "self_contact_frames",
    "self_contact_frame_fraction",
    "max_self_contact_penetration_m",
    "total_rotation_deg",
    "spin_drift_m",
    "reward",
    "fitness",
    "best_score",
    "best_reward",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank saved forward, left-spin, and right-spin CPG checkpoints.",
    )
    parser.add_argument("--tasks", nargs="+", choices=[*TASKS, "all"], default=["all"])
    parser.add_argument("--forward-dir", type=Path, default=Path("results/gait_cpg"))
    parser.add_argument("--left-dir", type=Path, default=Path("results/left_cpg"))
    parser.add_argument("--right-dir", type=Path, default=Path("results/right_cpg"))
    parser.add_argument("--spin-dir", type=Path, default=Path("results/turn_cpg"))
    parser.add_argument("--forward-models", nargs="*", type=Path, default=[])
    parser.add_argument("--left-models", nargs="*", type=Path, default=[])
    parser.add_argument("--right-models", nargs="*", type=Path, default=[])
    parser.add_argument("--max-candidates-per-task", type=int, default=0)
    parser.add_argument("--require-dr", action="store_true")
    parser.add_argument("--reach-radius", type=float, default=0.25)
    parser.add_argument("--spin-success-deg", type=float, default=180.0)
    parser.add_argument("--max-spin-drift", type=float, default=0.25)
    parser.add_argument("--max-self-contact-penetration", type=float, default=0.002)
    parser.add_argument("--max-self-contact-frame-fraction", type=float, default=0.05)
    parser.add_argument("--ignore-self-contact-success", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def requested_tasks(args: argparse.Namespace) -> tuple[str, ...]:
    if "all" in args.tasks:
        return TASKS
    return tuple(args.tasks)


def create_run_dir(root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"select_best_models_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_npz(path: Path | None) -> dict:
    if path is None:
        return {}
    data = np.load(str(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def scalar(values: dict, key: str, default=""):
    if key not in values:
        return default
    value = values[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def first_float(values: dict, key: str, default=math.nan) -> float:
    if key not in values:
        return float(default)
    value = values[key]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float(default)
        return float(value.reshape(-1)[0])
    return float(value)


def first_bool(values: dict, key: str, default=False) -> bool:
    if key not in values:
        return bool(default)
    value = values[key]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return bool(default)
        return bool(value.reshape(-1)[0])
    return bool(value)


def first_vector(values: dict, key: str, length: int) -> np.ndarray:
    value = values[key]
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:length]
    return array.reshape(-1, length)[0]


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def finite_or_large(value: float) -> float:
    if finite(value):
        return float(value)
    return 1e9


def finite_or_negative(value: float) -> float:
    if finite(value):
        return float(value)
    return -1e9


def meta_value(meta: dict, key: str, default=""):
    value = scalar(meta, key, default)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def find_meta_path(model_path: Path, prefix: str) -> Path:
    run_dir = model_path.parent
    run_id = run_dir.name
    preferred = run_dir / f"{prefix}_meta_{run_id}.npz"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob(f"{prefix}_meta*.npz"))
    return candidates[-1]


def find_metrics_path(model_path: Path, prefix: str) -> Path | None:
    run_dir = model_path.parent
    run_id = run_dir.name
    names = [
        f"{prefix}_stability_metrics_{run_id}.npz",
        f"{prefix}_validation_{run_id}.npz",
    ]
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    candidates = sorted(run_dir.glob(f"{prefix}_stability_metrics*.npz"))
    if candidates:
        return candidates[-1]
    candidates = sorted(run_dir.glob(f"{prefix}_validation*.npz"))
    if candidates:
        return candidates[-1]
    return None


def infer_seed(model_path: Path, meta: dict) -> int:
    seed = meta_value(meta, "seed", None)
    if seed is not None:
        return int(seed)
    match = re.search(r"_seed(\d+)", model_path.name)
    if match:
        return int(match.group(1))
    return 0


def candidate_from_model(task: str, model_path: Path) -> Candidate:
    prefix = "gait" if task == FORWARD else "spin"
    meta_path = find_meta_path(model_path, prefix)
    meta = load_npz(meta_path)
    run_id = str(meta_value(meta, "run_id", model_path.parent.name))
    direction = task
    if task != FORWARD:
        direction = str(meta_value(meta, "spin_direction", task))
    return Candidate(
        task=task,
        direction=direction,
        model_path=model_path,
        meta_path=meta_path,
        metrics_path=find_metrics_path(model_path, prefix),
        run_dir=model_path.parent,
        run_id=run_id,
        seed=infer_seed(model_path, meta),
    )


def filter_candidates(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    filtered = candidates
    if args.require_dr:
        filtered = [candidate for candidate in filtered if "_DR" in candidate.run_id]
    if args.max_candidates_per_task > 0:
        filtered = filtered[: args.max_candidates_per_task]
    return filtered


def discover_forward_candidates(args: argparse.Namespace) -> list[Candidate]:
    model_paths = args.forward_models or sorted(
        args.forward_dir.rglob("gait_best_*.npy"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return filter_candidates([candidate_from_model(FORWARD, path) for path in model_paths], args)


def discover_spin_candidates(task: str, args: argparse.Namespace) -> list[Candidate]:
    explicit = args.left_models if task == LEFT else args.right_models
    if explicit:
        return filter_candidates([candidate_from_model(task, path) for path in explicit], args)

    search_dirs = [args.left_dir if task == LEFT else args.right_dir, args.spin_dir]
    model_paths = sorted(
        {
            path
            for search_dir in search_dirs
            for path in search_dir.rglob("spin_best_*.npy")
        },
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    candidates = [candidate_from_model(task, path) for path in model_paths]
    candidates = [candidate for candidate in candidates if candidate.direction == task]
    return filter_candidates(candidates, args)


def contact_ok(row: dict, args: argparse.Namespace) -> bool:
    if args.ignore_self_contact_success:
        return True
    return (
        row["max_self_contact_penetration_m"] <= args.max_self_contact_penetration
        and row["self_contact_frame_fraction"] <= args.max_self_contact_frame_fraction
    )


def contact_sample_count(meta: dict, metrics: dict, default_duration: float) -> int:
    for key in ("times", "z_history", "xy_history", "heading_history"):
        if key in metrics:
            return int(np.asarray(metrics[key]).shape[0])
    duration = float(meta_value(meta, "duration", default_duration))
    return max(1, int(round(duration / 0.002)))


def base_row(candidate: Candidate, meta: dict, metrics: dict) -> dict:
    return {
        "task": candidate.task,
        "direction": "" if candidate.task == FORWARD else candidate.direction,
        "rank": "",
        "run_id": candidate.run_id,
        "seed": candidate.seed,
        "sigma": meta_value(meta, "sigma", ""),
        "budget": meta_value(meta, "budget", ""),
        "duration_s": meta_value(meta, "duration", ""),
        "eval_repeats": meta_value(meta, "eval_repeats", ""),
        "target_y_m": "",
        "model_path": str(candidate.model_path),
        "meta_path": str(candidate.meta_path),
        "metrics_path": "" if candidate.metrics_path is None else str(candidate.metrics_path),
        "success": False,
        "reached": "",
        "not_fallen": "",
        "contact_ok": "",
        "score": "",
        "min_distance_to_target_m": "",
        "time_to_target_s": "",
        "forward_progress_m": "",
        "horizontal_drift_m": "",
        "path_length_m": "",
        "path_efficiency": "",
        "final_x_m": "",
        "final_y_m": "",
        "final_z_m": "",
        "min_z_m": "",
        "max_abs_roll_deg": "",
        "max_abs_pitch_deg": "",
        "self_contact_pairs": "",
        "self_contact_frames": "",
        "self_contact_frame_fraction": "",
        "max_self_contact_penetration_m": "",
        "total_rotation_deg": "",
        "spin_drift_m": "",
        "reward": "",
        "fitness": meta_value(meta, "fitness", meta_value(meta, "objective", "")),
        "best_score": meta_value(meta, "best_score", ""),
        "best_reward": meta_value(meta, "best_reward", ""),
    }


def evaluate_forward_candidate(candidate: Candidate, args: argparse.Namespace) -> dict:
    meta = load_npz(candidate.meta_path)
    metrics = load_npz(candidate.metrics_path)
    row = base_row(candidate, meta, metrics)

    initial_pos = first_vector(metrics, "initial_positions", 3)
    final_pos = first_vector(metrics, "final_positions", 3)
    target_pos = first_vector(metrics, "target_positions", 3)
    min_distance = first_float(metrics, "min_distance_to_target")
    time_to_target = first_float(metrics, "time_to_target")
    fell = first_bool(metrics, "fell")
    frames = int(first_float(metrics, "self_contact_frames", 0.0))
    frame_fraction = frames / contact_sample_count(meta, metrics, 30.0)
    reached = min_distance <= args.reach_radius or finite(time_to_target)
    forward_progress = abs(float(final_pos[1] - initial_pos[1]))
    horizontal_drift = abs(float(final_pos[0] - initial_pos[0]))
    path_length = first_float(metrics, "path_length")
    path_efficiency = forward_progress / path_length if path_length > 0.0 else math.nan

    row.update({
        "target_y_m": float(target_pos[1]),
        "reached": reached,
        "not_fallen": not fell,
        "score": first_float(metrics, "scores"),
        "min_distance_to_target_m": min_distance,
        "time_to_target_s": time_to_target,
        "forward_progress_m": forward_progress,
        "horizontal_drift_m": horizontal_drift,
        "path_length_m": path_length,
        "path_efficiency": path_efficiency,
        "final_x_m": float(final_pos[0]),
        "final_y_m": float(final_pos[1]),
        "final_z_m": float(final_pos[2]),
        "min_z_m": first_float(metrics, "min_z"),
        "max_abs_roll_deg": math.degrees(first_float(metrics, "max_abs_roll")),
        "max_abs_pitch_deg": math.degrees(first_float(metrics, "max_abs_pitch")),
        "self_contact_pairs": int(first_float(metrics, "self_contact_pairs", 0.0)),
        "self_contact_frames": frames,
        "self_contact_frame_fraction": frame_fraction,
        "max_self_contact_penetration_m": first_float(metrics, "max_self_contact_penetration", 0.0),
    })
    row["contact_ok"] = contact_ok(row, args)
    row["success"] = bool(row["reached"] and row["not_fallen"] and row["contact_ok"])
    return row


def evaluate_spin_candidate(candidate: Candidate, args: argparse.Namespace) -> dict:
    meta = load_npz(candidate.meta_path)
    metrics = load_npz(candidate.metrics_path)
    row = base_row(candidate, meta, metrics)

    duration = float(meta_value(meta, "duration", 10.0))
    total_rotation = first_float(metrics, "total_rotation", first_float(meta, "validation_total_rotation"))
    drift = first_float(metrics, "drift", first_float(meta, "validation_drift"))
    fell = first_bool(metrics, "fell", False)
    frames = int(first_float(metrics, "self_contact_frames", first_float(meta, "validation_self_contact_frames", 0.0)))
    row.update({
        "not_fallen": not fell,
        "total_rotation_deg": math.degrees(total_rotation),
        "spin_drift_m": drift,
        "reward": first_float(metrics, "reward", first_float(meta, "validation_reward")),
        "path_length_m": first_float(metrics, "path_length", math.nan),
        "min_z_m": first_float(metrics, "min_z", math.nan),
        "max_abs_roll_deg": math.degrees(first_float(metrics, "max_abs_roll", math.nan)),
        "max_abs_pitch_deg": math.degrees(first_float(metrics, "max_abs_pitch", math.nan)),
        "self_contact_pairs": int(first_float(metrics, "self_contact_pairs", first_float(meta, "validation_self_contact_pairs", 0.0))),
        "self_contact_frames": frames,
        "self_contact_frame_fraction": frames / contact_sample_count(meta, metrics, duration),
        "max_self_contact_penetration_m": first_float(
            metrics,
            "max_self_contact_penetration",
            first_float(meta, "validation_max_self_contact_penetration", 0.0),
        ),
    })
    if "final_pos" in metrics:
        final_pos = first_vector(metrics, "final_pos", 3)
        row.update({
            "final_x_m": float(final_pos[0]),
            "final_y_m": float(final_pos[1]),
            "final_z_m": float(final_pos[2]),
        })
    rotation_success = row["total_rotation_deg"] >= args.spin_success_deg
    drift_ok = row["spin_drift_m"] <= args.max_spin_drift
    row["reached"] = rotation_success
    row["contact_ok"] = contact_ok(row, args)
    row["success"] = bool(rotation_success and drift_ok and row["not_fallen"] and row["contact_ok"])
    return row


def sort_rows(task: str, rows: list[dict]) -> list[dict]:
    if task == FORWARD:

        def key(row: dict) -> tuple:
            return (
                not row["success"],
                finite_or_large(row["horizontal_drift_m"]),
                finite_or_large(row["min_distance_to_target_m"]),
                finite_or_large(row["time_to_target_s"]),
                not row["contact_ok"],
                finite_or_large(row["self_contact_frame_fraction"]),
                finite_or_large(row["max_self_contact_penetration_m"]),
                -finite_or_negative(row["forward_progress_m"]),
            )
    else:

        def key(row: dict) -> tuple:
            return (
                not row["success"],
                -finite_or_negative(row["total_rotation_deg"]),
                finite_or_large(row["spin_drift_m"]),
                not row["contact_ok"],
                finite_or_large(row["self_contact_frame_fraction"]),
                finite_or_large(row["max_self_contact_penetration_m"]),
            )

    ranked = sorted(rows, key=key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def csv_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.8g}"
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in SUMMARY_FIELDS})


def json_safe(row: dict) -> dict:
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        cleaned[key] = value
    return cleaned


def print_selection(selected: dict[str, dict]) -> None:
    print("\nSelected models")
    print("---------------")
    for task in TASKS:
        row = selected.get(task)
        if row is None:
            print(f"{task}: no candidate")
            continue
        if task == FORWARD:
            details = (
                f"success={int(row['success'])}, "
                f"drift={float(row['horizontal_drift_m']):.3f}m, "
                f"min_dist={float(row['min_distance_to_target_m']):.3f}m, "
                f"progress={float(row['forward_progress_m']):.3f}m"
            )
        else:
            details = (
                f"success={int(row['success'])}, "
                f"rotation={float(row['total_rotation_deg']):.1f}deg, "
                f"drift={float(row['spin_drift_m']):.3f}m"
            )
        print(f"{task}: {row['model_path']} ({details})")


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.output_dir)
    tasks = requested_tasks(args)

    candidates_by_task = {}
    if FORWARD in tasks:
        candidates_by_task[FORWARD] = discover_forward_candidates(args)
    if LEFT in tasks:
        candidates_by_task[LEFT] = discover_spin_candidates(LEFT, args)
    if RIGHT in tasks:
        candidates_by_task[RIGHT] = discover_spin_candidates(RIGHT, args)

    print(f"Output directory: {run_dir}")
    for task, candidates in candidates_by_task.items():
        print(f"{task}: {len(candidates)} candidates")

    all_rows = []
    selected = {}
    for task in tasks:
        rows = []
        for candidate in candidates_by_task.get(task, []):
            if task == FORWARD:
                rows.append(evaluate_forward_candidate(candidate, args))
            else:
                rows.append(evaluate_spin_candidate(candidate, args))
        ranked = sort_rows(task, rows)
        if ranked:
            selected[task] = ranked[0]
        all_rows.extend(ranked)

    write_csv(run_dir / "model_selection_summary.csv", all_rows)
    payload = {
        "criteria": {
            "mode": "offline_saved_metrics_only",
            "forward": (
                "success first, then minimize horizontal_drift_m, then target distance, "
                "time to target, and self-contact"
            ),
            "spin": (
                "success first, then maximize rotation, minimize drift, and minimize self-contact"
            ),
            "reach_radius_m": args.reach_radius,
            "spin_success_deg": args.spin_success_deg,
            "max_spin_drift_m": args.max_spin_drift,
            "max_self_contact_penetration_m": args.max_self_contact_penetration,
            "max_self_contact_frame_fraction": args.max_self_contact_frame_fraction,
        },
        "selected": {task: json_safe(row) for task, row in selected.items()},
    }
    (run_dir / "selected_models.json").write_text(json.dumps(payload, indent=2) + "\n")
    print_selection(selected)
    print(f"\nWrote {run_dir / 'model_selection_summary.csv'}")
    print(f"Wrote {run_dir / 'selected_models.json'}")


if __name__ == "__main__":
    main()
