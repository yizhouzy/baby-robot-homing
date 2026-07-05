"""Demo scene, video recording, and trajectory plotting for learned gaits."""
from __future__ import annotations

from pathlib import Path
import math

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import VideoRecorder
from blocks.baby_robot import baby_robot

from robot_control.behavior import (
    BehaviorDemoConfig,
    BehaviorSample,
    GaitHomingBehaviorPolicy,
    distance_to_station,
)
from robot_control.config_gait import GaitConfig
from robot_control.controllers import CPGGaitController, sanitize_action
from robot_control.vision import find_robot_camera, render_vision_pip, sample_target_vision


def add_traffic_cone_target(world, name: str, pos):
    """Build the realistic traffic cone as one mocap body.

    Each geom below has a local position inside the same body; moving the body
    with mocap_pos therefore moves the full cone as one target object.
    """
    shell_material = f"{name}_hsv_target_shell"
    world.spec.add_material(
        name=shell_material,
        rgba=[1.0, 0.0, 0.25, 1.0],
        emission=1.0,
        specular=0.25,
        shininess=0.25,
        reflectance=0.25,
    )
    target_body = world.spec.worldbody.add_body(name=name, mocap=True, pos=pos)
    base_half_width = 0.125
    base_half_height = 0.0125
    lower_collar_radius = 0.105
    upper_collar_radius = 0.085
    cone_bottom_z = 0.07
    cone_height = 0.40
    bottom_radius = 0.095
    top_radius = 0.025

    # Base plate: low, dark square foot that makes the target visible from above.
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[base_half_width, base_half_width, base_half_height],
        pos=[0, 0, base_half_height],
        rgba=[0.02, 0.025, 0.02, 1.0],
    )

    # Lower collar: dark circular ring sitting on the base plate.
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[lower_collar_radius, 0.015],
        pos=[0, 0, 0.04],
        rgba=[0.035, 0.04, 0.035, 1.0],
    )

    # Upper collar: transition ring before the orange cone shell begins.
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[upper_collar_radius, 0.008],
        pos=[0, 0, 0.062],
        rgba=[0.015, 0.018, 0.015, 1.0],
    )

    # Tapered shell: stacked cylinders approximate a cone while staying one body.
    segments = 24
    segment_height = cone_height / segments
    for i in range(segments):
        z = cone_bottom_z + (i + 0.5) * segment_height
        t = (z - cone_bottom_z) / cone_height
        radius = bottom_radius + (top_radius - bottom_radius) * t
        is_white_band = 0.17 <= t <= 0.32 or 0.52 <= t <= 0.66
        rgba = (
            [0.94, 0.94, 0.88, 1.0]
            if is_white_band
            else [1.0, 0.0, 0.25, 1.0]
        )
        geom_kwargs = {}
        if not is_white_band:
            geom_kwargs["material"] = shell_material
        target_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, segment_height / 2],
            pos=[0, 0, z],
            rgba=rgba,
            **geom_kwargs,
        )

    # Front label patch: small pale marker copied from the realistic cone design.
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.028, 0.003, 0.012],
        pos=[0, -0.086, 0.17],
        rgba=[0.95, 0.92, 0.86, 1.0],
    )

    # Front label stripe: adds a visible facing direction.
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.018, 0.004, 0.006],
        pos=[0, -0.089, 0.17],
        rgba=[0.45, 0.45, 0.42, 1.0],
    )
    return target_body


def build_demo_scene(target_pos, camera_height: float = 6.0):
    world = SimpleFlatWorld()
    add_traffic_cone_target(world, "target_marker", target_pos)
    world.spec.worldbody.add_camera(
        name="video_cam",
        # pos: camera's x, y, z position in world coordinates
        pos=[0, -1.0, camera_height],

        # xyaxes: camera's local X and Y axes in world coordinates
        xyaxes=[-1, 0, 0, 0, -1, 0],
    )
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def build_behavior_scene(config: BehaviorDemoConfig):
    world = SimpleFlatWorld()
    add_traffic_cone_target(world, "charging_station", config.station_pos)
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[-1, -4, 1.8],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.4561, 0.8899],
    )
    world.spawn(baby_robot().spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    cam_name = find_robot_camera(model)
    mocap_id = model.body("charging_station").mocapid[0]
    return model, data, cam_name, mocap_id


def mode_color(mode: str) -> tuple[int, int, int]:
    colors = {
        "EXPLORE": (0, 255, 0),
        "IDLE": (180, 180, 180),
        "WANDER": (0, 255, 0),
        "SEARCHING": (255, 165, 0),
        "ALIGNING": (255, 220, 80),
        "APPROACHING": (0, 255, 255),
        "STOPPED": (80, 180, 255),
    }
    return colors.get(mode, (255, 255, 255))


def mode_plot_color(mode: str) -> str:
    colors = {
        "EXPLORE": "tab:green",
        "IDLE": "tab:gray",
        "WANDER": "tab:green",
        "SEARCHING": "tab:orange",
        "ALIGNING": "gold",
        "APPROACHING": "tab:cyan",
        "STOPPED": "tab:blue",
    }
    return colors.get(mode, "tab:gray")


def yaw_from_qpos(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def draw_behavior_overlays(
    frame,
    battery,
    mode,
    vision,
    camera_frame,
    mask,
    data,
    config: BehaviorDemoConfig,
) -> None:
    distance = distance_to_station(data, config.station_pos)
    cv2.putText(frame, f"Battery: {battery:.0%}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {mode}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color(mode), 2)
    cv2.putText(
        frame,
        f"Station: ({config.station_pos[0]:.2f}, {config.station_pos[1]:.2f})  "
        f"Distance: {distance:.2f}m",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Vision area: {vision[6]:.3f}  centroid: {vision[5]:+.2f}",
        (10, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    bar_w = int(200 * battery)
    cv2.rectangle(frame, (10, 450), (210, 470), (80, 80, 80), -1)
    bar_color = (0, 255, 0) if battery > config.battery_threshold else (255, 165, 0)
    cv2.rectangle(frame, (10, 450), (10 + bar_w, 470), bar_color, -1)
    cv2.rectangle(frame, (10, 450), (210, 470), (200, 200, 200), 1)
    thresh_x = int(10 + 200 * config.battery_threshold)
    cv2.line(frame, (thresh_x, 448), (thresh_x, 472), (255, 255, 255), 1)
    render_vision_pip(frame, camera_frame, mask)


def record_behavior_demo(
    model,
    data,
    cam_name,
    mocap_id,
    policy: GaitHomingBehaviorPolicy,
    config: BehaviorDemoConfig,
    control_freq: int,
) -> list[BehaviorSample]:
    mujoco.mj_resetData(model, data)
    policy.reset()
    data.mocap_pos[mocap_id] = config.station_pos

    dt = model.opt.timestep
    battery = 1.0
    drain = dt / config.duration
    current_action = np.zeros(model.nu)
    current_mode = "EXPLORE"
    current_camera_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    current_mask = np.zeros((24, 32), dtype=np.uint8)
    current_vision = [0.0] * 7

    try:
        ctrl_renderer = mujoco.Renderer(model, height=24, width=32)
        vid_renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception:
        print("Renderer unavailable; skipping behavior-tree video.")
        return []

    video_recorder = VideoRecorder(
        file_name=f"demo_behavior_tree_{config.timestamp}",
        output_folder=str(config.output_dir / "videos"),
    )

    fps = 30
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    viz = mujoco.MjvOption()
    viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    samples = []

    while data.time < config.duration:
        for _ in range(steps_per_frame):
            data.mocap_pos[mocap_id] = config.station_pos
            step = int(np.ceil(data.time / dt))
            if step % control_freq == 0:
                current_camera_frame, current_mask, current_vision = sample_target_vision(
                    ctrl_renderer, data, cam_name)
                current_action, current_mode = policy.decide(
                    current_vision, battery, model, data)

            data.ctrl[:] = current_action
            mujoco.mj_step(model, data)
            battery = max(0.0, battery - drain)

        vid_renderer.update_scene(data, scene_option=viz, camera=camera_id)
        frame = vid_renderer.render().copy()
        draw_behavior_overlays(
            frame,
            battery,
            current_mode,
            current_vision,
            current_camera_frame,
            current_mask,
            data,
            config,
        )
        video_recorder.write(frame=frame)
        samples.append(BehaviorSample(
            time=float(data.time),
            x=float(data.qpos[0]),
            y=float(data.qpos[1]),
            yaw=yaw_from_qpos(data),
            battery=float(battery),
            mode=current_mode,
            distance=distance_to_station(data, config.station_pos),
            vision_area=float(current_vision[6]),
            vision_centroid=float(current_vision[5]),
        ))

    video_recorder.release()
    ctrl_renderer.close()
    vid_renderer.close()
    return samples


def plot_behavior_trajectory(samples: list[BehaviorSample], config: BehaviorDemoConfig) -> bool:
    if len(samples) < 2:
        print("Not enough trajectory points for plots.")
        return False

    x = np.array([sample.x for sample in samples])
    y = np.array([sample.y for sample in samples])
    battery = np.array([sample.battery for sample in samples])
    times = np.array([sample.time for sample in samples])
    modes = np.array([sample.mode for sample in samples])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list("battery", ["red", "orange", "green"])
    lc = LineCollection(segments, cmap=cmap, linewidth=2.5)
    lc.set_array(battery[:-1])
    ax1.add_collection(lc)
    for mode in sorted(set(modes)):
        mode_mask = modes == mode
        ax1.scatter(
            x[mode_mask],
            y[mode_mask],
            s=8,
            color=mode_plot_color(mode),
            alpha=0.35,
            label=mode,
            zorder=4,
        )
    ax1.plot(x[0], y[0], "o", color="green", markersize=12, label="Start", zorder=5)
    ax1.plot(config.station_pos[0], config.station_pos[1], "*", color="red",
             markersize=18, label="Charging Station", zorder=5)
    ax1.autoscale()
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Trajectory (coloured by battery)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.colorbar(lc, ax=ax1, label="Battery")

    distance = np.array([sample.distance for sample in samples])
    ax2.plot(times, battery, "g-", lw=2, label="Battery")
    span_start = 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or modes[i] != modes[span_start]:
            ax2.axvspan(
                times[span_start],
                times[i - 1],
                color=mode_plot_color(modes[span_start]),
                alpha=0.08,
                lw=0,
            )
            span_start = i
    ax2.axhline(y=config.battery_threshold, color="orange", ls="--", alpha=0.7,
                label="Threshold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Battery", color="green")
    ax2r = ax2.twinx()
    ax2r.plot(times, distance, "r-", lw=2, label="Distance")
    ax2r.axhline(y=config.reach_radius, color="red", ls="--", alpha=0.5,
                 label="Reach radius")
    ax2r.set_ylim(bottom=0.0)
    ax2r.set_ylabel("Distance (m)", color="red")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2)
    ax2.set_title("Battery & Distance Over Time")
    ax2.grid(True, alpha=0.3)

    plot_path = config.output_dir / f"demo_plots_{config.timestamp}.png"
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved -> {plot_path}")
    return True


def _step_demo_controller(
    model,
    data,
    network: CPGGaitController,
    config: GaitConfig,
    current_ctrl: np.ndarray,
    step: int,
) -> np.ndarray:
    if step % config.control_step_freq == 0:
        action = network.forward(config.training_turn, config.training_speed)
        return sanitize_action(action, model)
    return current_ctrl


def record_demo_video(
    model,
    data,
    network: CPGGaitController,
    config: GaitConfig,
    target_pos,
    output_dir: Path,
    video_name: str,
    duration: float | None = None,
    console=None,
) -> bool:
    if console is not None:
        console.rule("[bold cyan]Recording Demo Video[/bold cyan]")

    try:
        vid_renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception:
        if console is not None:
            console.log("Video renderer unavailable, skipping.", style="yellow")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    video_recorder = VideoRecorder(file_name=video_name, output_folder=str(output_dir))

    mujoco.mj_resetData(model, data)
    network.reset_hidden()
    mocap_id = model.body("target_marker").mocapid[0]
    data.mocap_pos[mocap_id] = target_pos

    physics_dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (30 * physics_dt))))
    current_ctrl = np.zeros(model.nu)
    step = 0
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    viz = mujoco.MjvOption()
    viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    end_time = config.duration if duration is None else duration

    while data.time < end_time:
        for _ in range(steps_per_frame):
            data.mocap_pos[mocap_id] = target_pos
            current_ctrl = _step_demo_controller(model, data, network, config, current_ctrl, step)
            data.ctrl[:] = current_ctrl
            mujoco.mj_step(model, data)
            step += 1

        vid_renderer.update_scene(data, scene_option=viz, camera=camera_id)
        video_recorder.write(frame=vid_renderer.render())

    video_recorder.release()
    vid_renderer.close()
    if console is not None:
        console.log(f"Video saved -> {output_dir}", style="green")
    return True


def plot_demo_trajectory(
    model,
    data,
    network: CPGGaitController,
    config: GaitConfig,
    target_pos,
    trajectory_path: Path,
    duration: float | None = None,
    console=None,
) -> bool:
    if console is not None:
        console.rule("[bold cyan]Plotting Trajectory[/bold cyan]")

    mujoco.mj_resetData(model, data)
    network.reset_hidden()
    mocap_id = model.body("target_marker").mocapid[0]
    data.mocap_pos[mocap_id] = target_pos

    traj = []
    current_ctrl = np.zeros(model.nu)
    step = 0
    end_time = config.duration if duration is None else duration

    while data.time < end_time:
        data.mocap_pos[mocap_id] = target_pos
        next_ctrl = _step_demo_controller(model, data, network, config, current_ctrl, step)
        if step % config.control_step_freq == 0:
            traj.append((data.qpos[0], data.qpos[1]))
        current_ctrl = next_ctrl
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)
        step += 1

    if not traj:
        if console is not None:
            console.log("No trajectory samples were recorded.", style="yellow")
        return False

    xs = [point[0] for point in traj]
    ys = [point[1] for point in traj]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, "b-", linewidth=2, label="Robot path")
    ax.plot(xs[0], ys[0], "go", markersize=12, label="Start")
    ax.plot(target_pos[0], target_pos[1], "r*", markersize=18, label="Target")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Forward Gait Demonstration Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(trajectory_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    if console is not None:
        console.log(f"Trajectory plot saved -> {trajectory_path}", style="green")
    return True


def render_demo_from_network(
    network: CPGGaitController,
    config: GaitConfig,
    output_dir: Path,
    video_name: str,
    trajectory_path: Path,
    duration: float | None = None,
    target_pos=None,
    console=None,
) -> None:
    target = list(config.demo_target if target_pos is None else target_pos)
    model, data = build_demo_scene(target)
    record_demo_video(model, data, network, config, target, output_dir, video_name, duration, console)
    plot_demo_trajectory(model, data, network, config, target, trajectory_path, duration, console)
