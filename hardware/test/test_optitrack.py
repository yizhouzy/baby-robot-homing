"""Test and record OptiTrack/NatNet overhead tracking data.

Put one tracked object on the robot head center and one tracked object on the
target center. The script records raw 3D positions and also projects them onto
a configurable horizontal plane for robot/target XY logging.

Examples:
    python3 hardware/test/test_optitrack.py --server-ip 192.168.1.132
    python3 hardware/test/test_optitrack.py --robot-id 1 --target-id 2 --duration 30
    python3 hardware/test/test_optitrack.py --source marker --robot-id 101 --target-id 102
    python3 hardware/test/test_optitrack.py --plane-axes x z
"""
# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import logging
import math
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.baby_hardware import make_run_dir, write_metadata


SELECTED_FIELDS = [
    "elapsed_s",
    "natnet_timestamp_s",
    "robot_seen",
    "target_seen",
    "robot_source",
    "robot_id",
    "robot_name",
    "robot_x_m",
    "robot_y_m",
    "robot_z_m",
    "robot_plane_u_m",
    "robot_plane_v_m",
    "target_source",
    "target_id",
    "target_name",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "target_plane_u_m",
    "target_plane_v_m",
    "relative_u_m",
    "relative_v_m",
    "distance_m",
    "bearing_rad",
    "bearing_deg",
    "num_rigid_bodies",
    "num_markers",
]

OBJECT_FIELDS = [
    "elapsed_s",
    "natnet_timestamp_s",
    "source",
    "object_id",
    "object_name",
    "x_m",
    "y_m",
    "z_m",
    "plane_u_m",
    "plane_v_m",
]


@dataclass(frozen=True)
class TrackedObject:
    source: str
    object_id: str
    name: str
    position: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record OptiTrack robot/target XY data.")
    parser.add_argument("--server-ip", default="192.168.1.132")
    parser.add_argument(
        "--local-ip",
        default=None,
        help="Pi IP address to bind for NatNet. Use the Pi wlan0 IP when streaming over WiFi.",
    )
    parser.add_argument("--unicast", action="store_true", help="Use unicast instead of multicast.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--source", choices=["auto", "rigid", "marker"], default="auto")
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--robot-name", default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument(
        "--plane-axes",
        nargs=2,
        choices=["x", "y", "z"],
        default=["x", "y"],
        help="Raw NatNet axes to treat as horizontal plot/logging axes.",
    )
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--log-all-objects", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--debug-natnet", action="store_true")
    return parser.parse_args()


def configure_logging(debug_natnet: bool) -> None:
    level = logging.DEBUG if debug_natnet else logging.INFO
    logging.basicConfig(level=level)
    logging.getLogger("natnet").setLevel(level)


def attr(obj, names: tuple[str, ...]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def object_id(obj) -> str:
    value = attr(obj, ("id", "ID", "model_id", "marker_id", "tracking_id", "rigid_body_id"))
    return "" if value is None else str(value)


def object_name(obj) -> str:
    value = attr(obj, ("name", "model_name", "label", "marker_name"))
    return "" if value is None else str(value)


def object_position(obj) -> tuple[float, float, float]:
    value = attr(obj, ("position", "pos", "translation"))
    if value is not None:
        return float(value[0]), float(value[1]), float(value[2])
    return float(obj.x), float(obj.y), float(obj.z)


def as_tracked_objects(source: str, objects) -> list[TrackedObject]:
    return [
        TrackedObject(
            source=source,
            object_id=object_id(obj),
            name=object_name(obj),
            position=object_position(obj),
        )
        for obj in objects
    ]


def timestamp_value(timing) -> float:
    value = attr(timing, ("timestamp", "timecode", "system_timestamp"))
    return float(value) if value is not None else math.nan


def frame_rigid_bodies(frame):
    return attr(
        frame,
        (
            "rigid_bodies",
            "rigid_body_data",
            "rigidBodies",
            "rigid_body_list",
        ),
    ) or []


def frame_markers(frame):
    return attr(
        frame,
        (
            "labeled_markers",
            "labelled_markers",
            "markers",
            "marker_data",
            "labeled_marker_data",
        ),
    ) or []


def create_natnet_client(natnet_module, args: argparse.Namespace):
    if hasattr(natnet_module, "Client"):
        client = natnet_module.Client.connect(args.server_ip, timeout=args.timeout)
        return client, "callback"

    kwargs = {
        "server_ip_address": args.server_ip,
        "use_multicast": not args.unicast,
    }
    if args.local_ip is not None:
        kwargs["local_ip_address"] = args.local_ip
    client = natnet_module.NatNetClient(**kwargs)
    client.connect(timeout=args.timeout)
    return client, "sync"


def stop_natnet_client(client) -> None:
    if hasattr(client, "stop"):
        client.stop()
    elif hasattr(client, "shutdown"):
        client.shutdown()
    elif hasattr(client, "stop_async"):
        client.stop_async()


def axis_value(position: tuple[float, float, float], axis: str) -> float:
    return position[{"x": 0, "y": 1, "z": 2}[axis]]


def plane_position(obj: TrackedObject, plane_axes: list[str]) -> tuple[float, float]:
    return axis_value(obj.position, plane_axes[0]), axis_value(obj.position, plane_axes[1])


def matches(obj: TrackedObject, wanted_id: str | None, wanted_name: str | None) -> bool:
    if wanted_id is not None and obj.object_id == str(wanted_id):
        return True
    if wanted_name is not None and obj.name == wanted_name:
        return True
    return wanted_id is None and wanted_name is None


def select_objects(
    rigid_bodies: list[TrackedObject],
    markers: list[TrackedObject],
    args: argparse.Namespace,
) -> tuple[TrackedObject | None, TrackedObject | None]:
    if args.source == "rigid":
        objects = rigid_bodies
    elif args.source == "marker":
        objects = markers
    else:
        objects = rigid_bodies if rigid_bodies else markers

    robot_candidates = [
        obj for obj in objects if matches(obj, args.robot_id, args.robot_name)
    ]
    target_candidates = [
        obj for obj in objects if matches(obj, args.target_id, args.target_name)
    ]

    robot = robot_candidates[0] if robot_candidates else None
    target = target_candidates[0] if target_candidates else None
    if robot is None and len(objects) >= 1:
        robot = objects[0]
    if target is None and len(objects) >= 2:
        target = objects[1]
    return robot, target


def object_row(
    elapsed_s: float,
    natnet_timestamp_s: float,
    obj: TrackedObject,
    plane_axes: list[str],
) -> dict:
    plane_u, plane_v = plane_position(obj, plane_axes)
    return {
        "elapsed_s": f"{elapsed_s:.6f}",
        "natnet_timestamp_s": f"{natnet_timestamp_s:.6f}",
        "source": obj.source,
        "object_id": obj.object_id,
        "object_name": obj.name,
        "x_m": f"{obj.position[0]:.6f}",
        "y_m": f"{obj.position[1]:.6f}",
        "z_m": f"{obj.position[2]:.6f}",
        "plane_u_m": f"{plane_u:.6f}",
        "plane_v_m": f"{plane_v:.6f}",
    }


def selected_row(
    elapsed_s: float,
    natnet_timestamp_s: float,
    robot: TrackedObject | None,
    target: TrackedObject | None,
    rigid_bodies: list[TrackedObject],
    markers: list[TrackedObject],
    plane_axes: list[str],
) -> dict:
    row = {
        "elapsed_s": f"{elapsed_s:.6f}",
        "natnet_timestamp_s": f"{natnet_timestamp_s:.6f}",
        "robot_seen": int(robot is not None),
        "target_seen": int(target is not None),
        "num_rigid_bodies": len(rigid_bodies),
        "num_markers": len(markers),
    }
    for prefix, obj in (("robot", robot), ("target", target)):
        if obj is None:
            row.update({
                f"{prefix}_source": "",
                f"{prefix}_id": "",
                f"{prefix}_name": "",
                f"{prefix}_x_m": "",
                f"{prefix}_y_m": "",
                f"{prefix}_z_m": "",
                f"{prefix}_plane_u_m": "",
                f"{prefix}_plane_v_m": "",
            })
        else:
            plane_u, plane_v = plane_position(obj, plane_axes)
            row.update({
                f"{prefix}_source": obj.source,
                f"{prefix}_id": obj.object_id,
                f"{prefix}_name": obj.name,
                f"{prefix}_x_m": f"{obj.position[0]:.6f}",
                f"{prefix}_y_m": f"{obj.position[1]:.6f}",
                f"{prefix}_z_m": f"{obj.position[2]:.6f}",
                f"{prefix}_plane_u_m": f"{plane_u:.6f}",
                f"{prefix}_plane_v_m": f"{plane_v:.6f}",
            })

    if robot is not None and target is not None:
        robot_u, robot_v = plane_position(robot, plane_axes)
        target_u, target_v = plane_position(target, plane_axes)
        relative_u = target_u - robot_u
        relative_v = target_v - robot_v
        distance = math.hypot(relative_u, relative_v)
        bearing = math.atan2(relative_u, relative_v)
        row.update({
            "relative_u_m": f"{relative_u:.6f}",
            "relative_v_m": f"{relative_v:.6f}",
            "distance_m": f"{distance:.6f}",
            "bearing_rad": f"{bearing:.6f}",
            "bearing_deg": f"{math.degrees(bearing):.6f}",
        })
    else:
        row.update({
            "relative_u_m": "",
            "relative_v_m": "",
            "distance_m": "",
            "bearing_rad": "",
            "bearing_deg": "",
        })
    return row


def print_live(row: dict, plane_axes: list[str]) -> None:
    if row["robot_seen"] and row["target_seen"]:
        print(
            f"t={float(row['elapsed_s']):6.2f}s "
            f"robot=({row['robot_plane_u_m']},{row['robot_plane_v_m']}) "
            f"target=({row['target_plane_u_m']},{row['target_plane_v_m']}) "
            f"d={row['distance_m']}m bearing={row['bearing_deg']}deg "
            f"plane={plane_axes[0]}{plane_axes[1]}",
        )
    else:
        print(
            f"t={float(row['elapsed_s']):6.2f}s "
            f"robot_seen={row['robot_seen']} target_seen={row['target_seen']} "
            f"rigid={row['num_rigid_bodies']} markers={row['num_markers']}",
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.debug_natnet)

    import natnet

    run_dir = make_run_dir("test_optitrack", args.output_dir)
    selected_csv = run_dir / "optitrack_selected_xy.csv"
    objects_csv = run_dir / "optitrack_all_objects.csv"
    write_metadata(
        run_dir / "metadata.json",
        {
            "script": "hardware/test/test_optitrack.py",
            "server_ip": args.server_ip,
            "local_ip": args.local_ip,
            "unicast": args.unicast,
            "duration": args.duration,
            "source": args.source,
            "robot_id": args.robot_id,
            "target_id": args.target_id,
            "robot_name": args.robot_name,
            "target_name": args.target_name,
            "plane_axes": args.plane_axes,
            "log_all_objects": args.log_all_objects,
            "output_dir": str(run_dir),
        },
    )

    print(f"Connecting to NatNet server {args.server_ip}...")
    client, client_mode = create_natnet_client(natnet, args)
    print(f"Connected using {client_mode} NatNet client.")
    print(f"Results: {run_dir}")
    print(f"Selected XY CSV: {selected_csv}")
    if args.log_all_objects:
        print(f"All objects CSV: {objects_csv}")

    start = time.monotonic()
    next_print_s = 0.0
    selected_file = selected_csv.open("w", newline="")
    selected_writer = csv.DictWriter(selected_file, fieldnames=SELECTED_FIELDS)
    selected_writer.writeheader()
    objects_file = None
    objects_writer = None
    if args.log_all_objects:
        objects_file = objects_csv.open("w", newline="")
        objects_writer = csv.DictWriter(objects_file, fieldnames=OBJECT_FIELDS)
        objects_writer.writeheader()

    def process_frame(rigid_bodies, markers, timing) -> None:
        nonlocal next_print_s
        elapsed_s = time.monotonic() - start
        natnet_timestamp_s = timestamp_value(timing)
        tracked_rigid_bodies = as_tracked_objects("rigid", rigid_bodies)
        tracked_markers = as_tracked_objects("marker", markers)
        robot, target = select_objects(tracked_rigid_bodies, tracked_markers, args)
        row = selected_row(
            elapsed_s,
            natnet_timestamp_s,
            robot,
            target,
            tracked_rigid_bodies,
            tracked_markers,
            args.plane_axes,
        )
        selected_writer.writerow(row)
        if objects_writer is not None:
            for obj in [*tracked_rigid_bodies, *tracked_markers]:
                objects_writer.writerow(object_row(elapsed_s, natnet_timestamp_s, obj, args.plane_axes))

        if elapsed_s >= next_print_s:
            print_live(row, args.plane_axes)
            next_print_s = elapsed_s + 1.0 / args.print_hz

    try:
        if client_mode == "callback":
            client.set_callback(process_frame)
            print("Callback set. Waiting for frames...")
            while time.monotonic() - start < args.duration:
                client.spin()
        else:
            print("Polling update_sync(). Waiting for frames...")
            while time.monotonic() - start < args.duration:
                frame = client.update_sync()
                if frame is not None:
                    process_frame(frame_rigid_bodies(frame), frame_markers(frame), frame)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        stop_natnet_client(client)
        selected_file.close()
        if objects_file is not None:
            objects_file.close()

    print("OptiTrack test finished.")


if __name__ == "__main__":
    main()
