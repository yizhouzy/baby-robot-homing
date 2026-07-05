"""Shared loaders for real-world behavior-tree experiment plots.

The behavior CSVs are produced by ``hardware/test/run_behavior_tree_hardware.py``.
The Motive CSVs in ``results/4_real_world_exp`` currently contain unlabeled
marker positions, so the robot and target centers are estimated from marker
height:

* robot markers: low markers attached to the robot head/body
* target marker: higher marker attached to the target

This matches the current experiment files and keeps the plotting scripts small.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import re

import numpy as np


DEFAULT_INPUT_DIR = Path("results/4_real_world_exp")
DEFAULT_PLOT_DIR = Path("results/7_plots")
PANEL_FIGURE_HEIGHT = 4.8
PANEL_WIDTH = 4.8
SUMMARY_FIGURE_SIZE = (14.4, 4.8)
FIGURE_TITLE_SIZE = 18
AXIS_TITLE_SIZE = 13
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 8

CONDITION_ORDER = [
    "Mat front target",
    "Mat 90 deg right target",
    "Mat back target",
    "Floor front target",
    "Grass front target",
    "Front target",
    "90 deg right target",
    "Back target",
]

CONDITION_COLORS = {
    "Mat front target": "#B8892E",
    "Mat 90 deg right target": "#3A9D91",
    "Mat back target": "#6F5AA7",
    "Floor front target": "#A65A7A",
    "Grass front target": "#9E4B2B",
    "Front target": "#B8892E",
    "90 deg right target": "#3A9D91",
    "Back target": "#6F5AA7",
}

PLACEMENT_TRIAL_COLORS = {
    "mat_front": {
        1: "#D8B365",
        2: "#B8892E",
        3: "#7F5B14",
    },
    "mat_right": {
        1: "#7DCBBF",
        2: "#3A9D91",
        3: "#1F6F67",
    },
    "mat_back": {
        1: "#9C89C9",
        2: "#6F5AA7",
        3: "#47377E",
    },
    "floor_front": {
        1: "#C994A6",
        2: "#A65A7A",
        3: "#733955",
    },
    "grass_front": {
        1: "#C77C54",
        2: "#9E4B2B",
        3: "#6F321F",
    },
}

CONDITION_DISPLAY_NAMES = {
    "Mat front target": "Mat: frontal target",
    "Mat 90 deg right target": "Mat: right-side target",
    "Mat back target": "Mat: back target",
    "Floor front target": "Low-friction lab floor: frontal target",
    "Grass front target": "High-friction grass: frontal target",
    "Front target": "Frontal target",
    "90 deg right target": "Right-side target",
    "Back target": "Rear target",
}

TARGET_DISTANCE_PRIOR_M = 2.2
HIGH_MARKER_CLUSTER_DISTANCE_M = 0.55
TARGET_CLUSTER_MATCH_DISTANCE_M = 0.70
RUN_START_SEARCH_STEP_S = 0.5

STATE_ORDER = [
    "SEARCHING",
    "APPROACHING",
    "STOPPED",
    "IDLE",
    "NO ROBOT LOG",
]

STATE_COLORS = {
    "SEARCHING": "#8EC7E8",
    "APPROACHING": "#A8D08D",
    "STOPPED": "#BFBFBF",
    "IDLE": "#E6E6E6",
    "NO ROBOT LOG": "#F2D7D5",
}


@dataclass(frozen=True)
class BehaviorData:
    time_s: np.ndarray
    dt_s: np.ndarray
    mode: list[str]
    gait: list[str]
    visible: np.ndarray
    reached: np.ndarray
    bearing: np.ndarray
    area: np.ndarray
    battery_v: np.ndarray
    battery_percentage: np.ndarray


@dataclass(frozen=True)
class MotiveTrack:
    time_s: np.ndarray
    robot_x_m: np.ndarray
    robot_z_m: np.ndarray
    target_x_m: np.ndarray
    target_z_m: np.ndarray
    distance_m: np.ndarray
    robot_marker_count: np.ndarray
    target_marker_count: np.ndarray


@dataclass(frozen=True)
class TrialData:
    trial_dir: Path
    motive_csv: Path | None
    experiment_number: int
    trial_number: int
    condition: str
    label: str
    behavior: BehaviorData
    motive: MotiveTrack
    metadata: dict
    behavior_available: bool
    motive_available: bool
    motive_start_s: float


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def float_column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def bool_column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([int(float(row[key])) != 0 for row in rows], dtype=bool)


def condition_from_dir_name(name: str) -> str:
    lower = name.lower()
    terrain_prefix = ""
    if "floor" in lower:
        terrain_prefix = "Floor "
    elif "grass" in lower:
        terrain_prefix = "Grass "
    elif "mat" in lower:
        terrain_prefix = "Mat "

    if "back" in lower:
        return f"{terrain_prefix}back target".strip().capitalize()
    if "right" in lower:
        return f"{terrain_prefix}90 deg right target".strip().capitalize()
    return f"{terrain_prefix}front target".strip().capitalize()


def number_from_name(name: str, prefix: str, default: int) -> int:
    match = re.search(rf"{prefix}_(\d+)", name)
    if match:
        return int(match.group(1))
    return default


def trial_number_from_name(name: str, default: int) -> int:
    match = re.search(r"(?:trial|trail)_(\d+)", name)
    if match:
        return int(match.group(1))
    return default


def condition_sort_index(condition: str) -> int:
    if condition in CONDITION_ORDER:
        return CONDITION_ORDER.index(condition)
    return len(CONDITION_ORDER)


def trial_sort_key(trial: TrialData) -> tuple[int, int, int]:
    return condition_sort_index(trial.condition), trial.trial_number, trial.experiment_number


def trial_color(trial: TrialData) -> str:
    lower = trial.condition.lower()
    if "floor" in lower:
        palette = PLACEMENT_TRIAL_COLORS["floor_front"]
    elif "grass" in lower:
        palette = PLACEMENT_TRIAL_COLORS["grass_front"]
    elif "right" in lower:
        palette = PLACEMENT_TRIAL_COLORS["mat_right"]
    elif "back" in lower:
        palette = PLACEMENT_TRIAL_COLORS["mat_back"]
    else:
        palette = PLACEMENT_TRIAL_COLORS["mat_front"]
    return palette.get(trial.trial_number, palette[3])


def condition_display_name(condition: str) -> str:
    return CONDITION_DISPLAY_NAMES.get(condition, condition)


def present_conditions(trials: list[TrialData]) -> list[str]:
    return [
        condition
        for condition in CONDITION_ORDER
        if any(trial.condition == condition for trial in trials)
    ] + sorted({
        trial.condition
        for trial in trials
        if trial.condition not in CONDITION_ORDER
    })


def discover_trial_dirs(input_dir: Path) -> list[Path]:
    trial_dirs = [
        path
        for path in input_dir.iterdir()
        if path.is_dir() and (path / "behavior_samples.csv").exists()
    ]
    return sorted(trial_dirs, key=lambda path: number_from_name(path.name, "exp", 999))


def discover_top_level_motive_csvs(input_dir: Path) -> list[Path]:
    return sorted(
        input_dir.glob("Take Exp*.csv"),
        key=lambda path: number_from_take_name(path.name, "exp", 999),
    )


def number_from_take_name(name: str, key: str, default: int) -> int:
    match = re.search(rf"{key}\s*(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default


def trial_number_from_take_name(name: str, default: int) -> int:
    match = re.search(r"(?:trial|trail)\s*(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:grass|floor|mat|front|back|right)\s*(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default


def read_behavior(path: Path) -> BehaviorData:
    rows = read_csv_dicts(path)
    time_s = float_column(rows, "elapsed_s")
    if "loop_dt_s" in rows[0]:
        dt_s = float_column(rows, "loop_dt_s")
    else:
        sample_period = float(np.median(np.diff(time_s)))
        dt_s = np.diff(time_s, append=time_s[-1] + sample_period)
    return BehaviorData(
        time_s=time_s,
        dt_s=dt_s,
        mode=[row["mode"] for row in rows],
        gait=[row["selected_gait"] for row in rows],
        visible=bool_column(rows, "visible"),
        reached=bool_column(rows, "reached"),
        bearing=float_column(rows, "bearing"),
        area=float_column(rows, "area"),
        battery_v=float_column(rows, "battery_v"),
        battery_percentage=float_column(rows, "battery_percentage"),
    )


def discover_motive_csv(trial_dir: Path) -> Path:
    return sorted(trial_dir.glob("Take*.csv"))[0]


def empty_motive(duration_s: float) -> MotiveTrack:
    time_s = np.asarray([0.0, max(float(duration_s), 0.0)], dtype=float)
    missing_float = np.full_like(time_s, math.nan)
    missing_int = np.zeros_like(time_s, dtype=int)
    return MotiveTrack(
        time_s=time_s,
        robot_x_m=missing_float,
        robot_z_m=missing_float,
        target_x_m=missing_float,
        target_z_m=missing_float,
        distance_m=missing_float,
        robot_marker_count=missing_int,
        target_marker_count=missing_int,
    )


def missing_behavior(duration_s: float) -> BehaviorData:
    time_s = np.asarray([0.0, max(float(duration_s), 0.0)], dtype=float)
    dt_s = np.asarray([time_s[-1], 0.0], dtype=float)
    missing_float = np.full_like(time_s, math.nan)
    missing_bool = np.zeros_like(time_s, dtype=bool)
    return BehaviorData(
        time_s=time_s,
        dt_s=dt_s,
        mode=["NO ROBOT LOG", "NO ROBOT LOG"],
        gait=["missing", "missing"],
        visible=missing_bool,
        reached=missing_bool,
        bearing=missing_float,
        area=missing_float,
        battery_v=missing_float,
        battery_percentage=missing_float,
    )


def high_marker_clusters(points: list[tuple[float, float]]) -> list[tuple[int, float, float]]:
    unused = set(range(len(points)))
    clusters = []
    while unused:
        seed = unused.pop()
        component = {seed}
        changed = True
        while changed:
            changed = False
            for index in list(unused):
                if any(
                    math.hypot(points[index][0] - points[item][0], points[index][1] - points[item][1])
                    <= HIGH_MARKER_CLUSTER_DISTANCE_M
                    for item in component
                ):
                    unused.remove(index)
                    component.add(index)
                    changed = True
        cluster_points = np.asarray([points[index] for index in component], dtype=float)
        clusters.append((
            len(component),
            float(np.mean(cluster_points[:, 0])),
            float(np.mean(cluster_points[:, 1])),
        ))
    return clusters


def choose_target_cluster(
    clusters: list[tuple[int, float, float]],
    robot_x_m: float,
    robot_z_m: float,
) -> tuple[int, float, float] | None:
    if not clusters or not math.isfinite(robot_x_m) or not math.isfinite(robot_z_m):
        return None

    def score(cluster: tuple[int, float, float]) -> float:
        marker_count, x_m, z_m = cluster
        distance = math.hypot(x_m - robot_x_m, z_m - robot_z_m)
        return abs(distance - TARGET_DISTANCE_PRIOR_M) - 0.08 * min(marker_count, 3)

    return min(clusters, key=score)


def target_center_from_initial_frames(
    time_s: np.ndarray,
    robot_x_m: np.ndarray,
    robot_z_m: np.ndarray,
    clusters_by_frame: list[list[tuple[int, float, float]]],
) -> tuple[float, float]:
    initial_mask = np.isfinite(robot_x_m) & np.isfinite(robot_z_m)
    if np.any(time_s <= 2.0):
        initial_mask &= time_s <= 2.0
    robot_x = float(np.nanmedian(robot_x_m[initial_mask]))
    robot_z = float(np.nanmedian(robot_z_m[initial_mask]))
    candidates = []
    for time_value, clusters in zip(time_s, clusters_by_frame):
        if time_value > 2.0 and candidates:
            break
        cluster = choose_target_cluster(clusters, robot_x, robot_z)
        if cluster is not None:
            candidates.append((cluster[1], cluster[2]))
    candidate_array = np.asarray(candidates, dtype=float)
    return float(np.median(candidate_array[:, 0])), float(np.median(candidate_array[:, 1]))


def target_marker_count_for_frame(
    clusters: list[tuple[int, float, float]],
    target_x_m: float,
    target_z_m: float,
) -> int:
    if not clusters:
        return 0
    distances = [
        math.hypot(cluster[1] - target_x_m, cluster[2] - target_z_m)
        for cluster in clusters
    ]
    best_index = int(np.argmin(distances))
    if distances[best_index] > TARGET_CLUSTER_MATCH_DISTANCE_M:
        return 0
    return clusters[best_index][0]


def read_motive_track(
    path: Path,
    robot_y_min: float = 0.05,
    robot_y_max: float = 0.30,
    target_y_min: float = 0.35,
) -> MotiveTrack:
    time_s = []
    robot_x_m = []
    robot_z_m = []
    robot_marker_count = []
    clusters_by_frame = []

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

            time_s.append(float(row[1]))
            robot_points = []
            high_points = []
            for index in range(2, len(row) - 2, 3):
                if row[index] == "" or row[index + 1] == "" or row[index + 2] == "":
                    continue
                x_m = float(row[index])
                y_m = float(row[index + 1])
                z_m = float(row[index + 2])
                if robot_y_min <= y_m <= robot_y_max:
                    robot_points.append((x_m, z_m))
                elif y_m >= target_y_min:
                    high_points.append((x_m, z_m))

            robot_marker_count.append(len(robot_points))
            clusters_by_frame.append(high_marker_clusters(high_points))
            if robot_points:
                robot_array = np.asarray(robot_points, dtype=float)
                robot_x_m.append(float(np.mean(robot_array[:, 0])))
                robot_z_m.append(float(np.mean(robot_array[:, 1])))
            else:
                robot_x_m.append(math.nan)
                robot_z_m.append(math.nan)

    time = np.asarray(time_s, dtype=float)
    robot_x = np.asarray(robot_x_m, dtype=float)
    robot_z = np.asarray(robot_z_m, dtype=float)
    target_x_value, target_z_value = target_center_from_initial_frames(
        time,
        robot_x,
        robot_z,
        clusters_by_frame,
    )
    target_x = np.full_like(robot_x, target_x_value, dtype=float)
    target_z = np.full_like(robot_z, target_z_value, dtype=float)
    target_marker_count = np.asarray([
        target_marker_count_for_frame(clusters, target_x_value, target_z_value)
        for clusters in clusters_by_frame
    ], dtype=int)
    distance = np.hypot(robot_x - target_x, robot_z - target_z)
    return MotiveTrack(
        time_s=time,
        robot_x_m=robot_x,
        robot_z_m=robot_z,
        target_x_m=target_x,
        target_z_m=target_z,
        distance_m=distance,
        robot_marker_count=np.asarray(robot_marker_count, dtype=int),
        target_marker_count=np.asarray(target_marker_count, dtype=int),
    )


def crop_motive_track(track: MotiveTrack, start_s: float, duration_s: float) -> MotiveTrack:
    end_s = start_s + duration_s
    mask = (track.time_s >= start_s) & (track.time_s <= end_s)
    return MotiveTrack(
        time_s=track.time_s[mask] - start_s,
        robot_x_m=track.robot_x_m[mask],
        robot_z_m=track.robot_z_m[mask],
        target_x_m=track.target_x_m[mask],
        target_z_m=track.target_z_m[mask],
        distance_m=track.distance_m[mask],
        robot_marker_count=track.robot_marker_count[mask],
        target_marker_count=track.target_marker_count[mask],
    )


def estimate_run_start_s(track: MotiveTrack, duration_s: float) -> float:
    valid = np.isfinite(track.distance_m)
    time_s = track.time_s[valid]
    distance_m = track.distance_m[valid]
    if len(time_s) == 0 or time_s[-1] <= duration_s:
        return 0.0

    best_start = 0.0
    best_score = math.inf
    candidate_count = int((time_s[-1] - duration_s) / RUN_START_SEARCH_STEP_S) + 1
    for candidate_index in range(candidate_count):
        start_s = candidate_index * RUN_START_SEARCH_STEP_S
        end_s = start_s + duration_s
        start_distance = float(np.interp(start_s, time_s, distance_m))
        end_distance = float(np.interp(end_s, time_s, distance_m))
        progress = start_distance - end_distance
        score = end_distance - 0.25 * progress
        if start_distance >= 1.0 and score < best_score:
            best_score = score
            best_start = start_s
    return best_start


def load_trial(trial_dir: Path) -> TrialData:
    experiment_number = number_from_name(trial_dir.name, "exp", 999)
    condition = condition_from_dir_name(trial_dir.name)
    trial_number = trial_number_from_name(trial_dir.name, experiment_number)
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    behavior = read_behavior(trial_dir / "behavior_samples.csv")
    motive_csvs = sorted(trial_dir.glob("Take*.csv"))
    if motive_csvs:
        motive_csv = motive_csvs[0]
        motive = read_motive_track(motive_csv)
        start_s = estimate_run_start_s(motive, float(behavior.time_s[-1]))
        motive = crop_motive_track(motive, start_s, float(behavior.time_s[-1]))
        motive_available = True
    else:
        motive_csv = None
        motive = empty_motive(float(behavior.time_s[-1]))
        motive_available = False
    return TrialData(
        trial_dir=trial_dir,
        motive_csv=motive_csv,
        experiment_number=experiment_number,
        trial_number=trial_number,
        condition=condition,
        label=f"Trial {trial_number}",
        behavior=behavior,
        motive=motive,
        metadata=metadata,
        behavior_available=True,
        motive_available=motive_available,
        motive_start_s=start_s if motive_available else 0.0,
    )


def load_motive_only_trial(path: Path) -> TrialData:
    experiment_number = number_from_take_name(path.name, "exp", 999)
    trial_number = trial_number_from_take_name(path.name, experiment_number)
    condition = condition_from_dir_name(path.stem)
    motive = read_motive_track(path)
    duration_s = float(motive.time_s[-1]) if len(motive.time_s) else 0.0
    return TrialData(
        trial_dir=path.parent,
        motive_csv=path,
        experiment_number=experiment_number,
        trial_number=trial_number,
        condition=condition,
        label=f"Trial {trial_number}",
        behavior=missing_behavior(duration_s),
        motive=motive,
        metadata={
            "missing_robot_log": True,
            "source_motive_csv": path.name,
            "reach_vision_area": math.nan,
            "bearing_threshold": math.nan,
        },
        behavior_available=False,
        motive_available=True,
        motive_start_s=0.0,
    )


def load_trials(input_dir: Path = DEFAULT_INPUT_DIR) -> list[TrialData]:
    trials = [load_trial(path) for path in discover_trial_dirs(input_dir)]
    trials.extend(load_motive_only_trial(path) for path in discover_top_level_motive_csvs(input_dir))
    return sorted(trials, key=trial_sort_key)


def first_reached_time(trial: TrialData) -> float:
    if not trial.behavior_available:
        return math.nan
    reached_indices = np.flatnonzero(trial.behavior.reached)
    if len(reached_indices) == 0:
        return math.nan
    return float(trial.behavior.time_s[int(reached_indices[0])])


def behavior_duration_s(trial: TrialData) -> float:
    return float(trial.behavior.time_s[-1])


def state_durations(trial: TrialData) -> dict[str, float]:
    durations = {state: 0.0 for state in STATE_ORDER}
    for mode, dt_s in zip(trial.behavior.mode, trial.behavior.dt_s):
        durations[mode] = durations.get(mode, 0.0) + float(dt_s)
    return durations


def state_spans(trial: TrialData) -> list[tuple[float, float, str]]:
    time_s = trial.behavior.time_s
    modes = trial.behavior.mode
    sample_period = float(np.median(np.diff(time_s)))
    spans = []
    start_index = 0
    for index in range(1, len(modes)):
        if modes[index] != modes[index - 1]:
            spans.append((float(time_s[start_index]), float(time_s[index]), modes[index - 1]))
            start_index = index
    spans.append((float(time_s[start_index]), float(time_s[-1] + sample_period), modes[-1]))
    return spans


def motive_valid_mask(trial: TrialData) -> np.ndarray:
    return (
        np.isfinite(trial.motive.distance_m)
        & np.isfinite(trial.motive.robot_x_m)
        & np.isfinite(trial.motive.target_x_m)
    )


def path_length_m(trial: TrialData) -> float:
    mask = motive_valid_mask(trial)
    x = trial.motive.robot_x_m[mask]
    z = trial.motive.robot_z_m[mask]
    if len(x) < 2:
        return math.nan
    return float(np.sum(np.hypot(np.diff(x), np.diff(z))))


def trial_summary_row(trial: TrialData) -> dict:
    mask = motive_valid_mask(trial)
    distances = trial.motive.distance_m[mask]
    durations = state_durations(trial)
    reached_time = first_reached_time(trial)
    if len(distances):
        initial_distance_m = float(distances[0])
        min_distance_m = float(np.min(distances))
        final_distance_m = float(distances[-1])
        median_robot_markers = float(np.median(trial.motive.robot_marker_count[mask]))
        median_target_markers = float(np.median(trial.motive.target_marker_count[mask]))
    else:
        initial_distance_m = math.nan
        min_distance_m = math.nan
        final_distance_m = math.nan
        median_robot_markers = math.nan
        median_target_markers = math.nan
    return {
        "experiment": trial.experiment_number,
        "trial_dir": trial.trial_dir.name,
        "motive_csv": "" if trial.motive_csv is None else trial.motive_csv.name,
        "condition": trial.condition,
        "behavior_log_available": int(trial.behavior_available),
        "motive_log_available": int(trial.motive_available),
        "motive_start_s": trial.motive_start_s,
        "success": int(not math.isnan(reached_time)),
        "reached_time_s": reached_time,
        "behavior_duration_s": behavior_duration_s(trial),
        "initial_distance_m": initial_distance_m,
        "min_distance_m": min_distance_m,
        "final_distance_m": final_distance_m,
        "path_length_m": path_length_m(trial),
        "search_time_s": durations.get("SEARCHING", 0.0),
        "approach_time_s": durations.get("APPROACHING", 0.0),
        "stopped_time_s": durations.get("STOPPED", 0.0),
        "missing_robot_log_time_s": durations.get("NO ROBOT LOG", 0.0),
        "visible_time_s": float(np.sum(trial.behavior.dt_s[trial.behavior.visible])) if trial.behavior_available else math.nan,
        "final_area": float(trial.behavior.area[-1]) if trial.behavior_available else math.nan,
        "final_bearing": float(trial.behavior.bearing[-1]) if trial.behavior_available else math.nan,
        "median_robot_markers": median_robot_markers,
        "median_target_markers": median_target_markers,
    }


def write_summary_csv(path: Path, trials: list[TrialData]) -> None:
    rows = [trial_summary_row(trial) for trial in trials]
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def representative_trials(trials: list[TrialData]) -> list[TrialData]:
    selected = []
    for condition in present_conditions(trials):
        condition_trials = [trial for trial in trials if trial.condition == condition]
        ranked = sorted(condition_trials, key=first_reached_time)
        selected.append(ranked[len(ranked) // 2])
    return selected


def save_figure(fig, output_dir: Path, filename: str, save_pdf: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / filename
    fig.savefig(png_path, dpi=180)
    if save_pdf:
        fig.savefig(png_path.with_suffix(".pdf"))
    print(f"Saved {png_path}")
