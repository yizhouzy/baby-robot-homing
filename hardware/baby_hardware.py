"""Small VU Robohat helper layer for the physical baby robot.

The real hardware API works in servo degrees from 0 to 180, with 90 as the
neutral position. Robohatlib handles low-level servo calibration through its
``ServoData`` objects and can read servo position back with
``get_servo_single_angle()``. This file only maps baby-robot actuator order to
Robohat servo channels and provides small logging helpers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import importlib
import json
import math
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "hardware_tests"
ROBOHAT_ROOT = PROJECT_ROOT / "robohat"


@dataclass(frozen=True)
class ServoMapping:
    """Mapping from one MuJoCo actuator index to one Robohat servo channel."""

    actuator_index: int
    actuator_name: str
    channel: int
    readback_channel: int | None = None
    neutral_deg: float = 90.0
    sign: float = 1.0
    min_deg: float = 10.0
    max_deg: float = 170.0


@dataclass(frozen=True)
class BatterySample:
    elapsed_s: float
    voltage_v: float
    percentage: int
    status: str
    drain_mv_per_min: float


@dataclass(frozen=True)
class ImuSample:
    elapsed_s: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float
    mag_y: float
    mag_z: float


DEFAULT_SERVO_MAPPINGS = [
    ServoMapping(0, "robot1_hinge_0servo", 31, readback_channel=15, neutral_deg=93, sign=-1.0),
    ServoMapping(1, "robot1_hinge_1servo", 30, readback_channel=14, neutral_deg=80),
    ServoMapping(2, "robot1_hinge_2servo", 29, readback_channel=13, neutral_deg=85),
    ServoMapping(3, "robot1_hinge_3servo", 28, readback_channel=12, neutral_deg=90),
    ServoMapping(4, "robot1_hinge_4servo", 0, readback_channel=16, neutral_deg=89, sign=-1.0),
    ServoMapping(5, "robot1_hinge_5servo", 1, readback_channel=17, neutral_deg=86, sign=-1.0),
    ServoMapping(6, "robot1_hinge_6servo", 2, readback_channel=18, neutral_deg=87),
    ServoMapping(7, "robot1_hinge_7servo", 3, readback_channel=19, neutral_deg=96, sign=-1.0),
]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir(script_name: str, output_dir: Path | str | None = None) -> Path:
    root = DEFAULT_OUTPUT_ROOT if output_dir is None else Path(output_dir)
    run_dir = root / f"{script_name}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def mapping_metadata(mappings: list[ServoMapping]) -> list[dict]:
    return [asdict(mapping) for mapping in mappings]


def servo_readback_channel(mapping: ServoMapping) -> int:
    """Use a separate feedback channel when the wiring does not match PWM."""
    return mapping.channel if mapping.readback_channel is None else mapping.readback_channel


def clamp_servo_degrees(angle_deg: float, mapping: ServoMapping) -> float:
    """Keep commands inside the joint's conservative mechanical range."""
    return min(mapping.max_deg, max(mapping.min_deg, float(angle_deg)))


def sim_rad_to_servo_degrees(sim_rad: float, mapping: ServoMapping) -> float:
    """Map a controller output in radians to the Robohat 0-180 degree API."""
    return clamp_servo_degrees(
        mapping.neutral_deg + mapping.sign * math.degrees(float(sim_rad)),
        mapping,
    )


def joint_deg_to_servo_degrees(joint_deg: float, mapping: ServoMapping) -> float:
    """Map a robot-joint offset in degrees to the Robohat 0-180 degree API."""
    return clamp_servo_degrees(
        mapping.neutral_deg + mapping.sign * float(joint_deg),
        mapping,
    )


class DisabledRobohatCamera:
    """No-op replacement for Robohat's built-in camera wrapper."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def init_camera(self) -> None:
        pass

    def is_cam_available(self) -> bool:
        return False

    def get_capture_array(self):
        return None

    def test_camera(self) -> None:
        print("Robohat camera disabled for this script.")


def create_robohat(enable_camera: bool = True):
    """Construct and initialize the VU Robohat using ``testlib.TestConfig``.

    Keep the Robohat import here so scripts fail only when they actually try to
    run on the Pi without the VU library on ``PYTHONPATH``.
    """
    if str(ROBOHAT_ROOT) not in sys.path:
        sys.path.insert(0, str(ROBOHAT_ROOT))

    robohat_module = importlib.import_module("robohatlib.Robohat")
    if not enable_camera:
        robohat_module.Camera = DisabledRobohatCamera
    Robohat = robohat_module.Robohat
    from testlib.TestConfig import (
        SERVOASSEMBLY_1_CONFIG,
        SERVOASSEMBLY_2_CONFIG,
        SERVOBOARD_1_DATAS_LIST,
        SERVOBOARD_2_DATAS_LIST,
        TOPBOARD_ID_SWITCH,
    )

    robohat = Robohat(
        SERVOASSEMBLY_1_CONFIG,
        SERVOASSEMBLY_2_CONFIG,
        TOPBOARD_ID_SWITCH,
    )
    robohat.init(
        SERVOBOARD_1_DATAS_LIST,
        SERVOBOARD_2_DATAS_LIST,
    )
    return robohat


class BabyRobotHardware:
    """Tiny wrapper around VU Robohat for the baby robot's eight servos."""

    def __init__(
        self,
        mappings: list[ServoMapping] | None = None,
        *,
        enable_servos: bool = True,
        direct_mode: bool = False,
        delay_s: float = 0.02,
        beep: bool = True,
        enable_camera: bool = True,
    ) -> None:
        self.mappings = DEFAULT_SERVO_MAPPINGS if mappings is None else mappings
        self.enable_servos = enable_servos
        self.robohat = create_robohat(enable_camera=enable_camera)
        if beep:
            self.robohat.do_buzzer_beep()
        if self.enable_servos:
            self.robohat.start_servo_drivers()
            self.robohat.wakeup_servo()
            self.robohat.set_servo_direct_mode(direct_mode, delay_s)
        self.last_command_degrees = [mapping.neutral_deg for mapping in self.mappings]
        self._all_servo_degrees = [90.0] * 32
        for mapping in self.mappings:
            self._all_servo_degrees[mapping.channel] = mapping.neutral_deg

    def _send_all_servo_degrees(self) -> None:
        self.robohat.set_servo_multiple_angles(self._all_servo_degrees)

    def set_servo_degrees(self, index: int, angle_deg: float) -> float:
        """Command a raw Robohat servo angle; this does not apply ``sign``."""
        mapping = self.mappings[index]
        command_deg = clamp_servo_degrees(angle_deg, mapping)
        self._all_servo_degrees[mapping.channel] = command_deg
        self._send_all_servo_degrees()
        self.last_command_degrees[index] = command_deg
        return command_deg

    def set_joint_degrees(self, index: int, joint_deg: float) -> float:
        """Command a signed joint offset from the calibrated neutral position."""
        return self.set_servo_degrees(
            index,
            joint_deg_to_servo_degrees(joint_deg, self.mappings[index]),
        )

    def set_joint_angles(self, sim_rads) -> list[float]:
        commands = []
        for index, sim_rad in enumerate(sim_rads):
            command_deg = sim_rad_to_servo_degrees(sim_rad, self.mappings[index])
            mapping = self.mappings[index]
            self._all_servo_degrees[mapping.channel] = command_deg
            self.last_command_degrees[index] = command_deg
            commands.append(command_deg)
        self._send_all_servo_degrees()
        return commands

    def read_servo_degrees(self, index: int) -> float:
        return float(self.robohat.get_servo_single_angle(servo_readback_channel(self.mappings[index])))

    def read_all_servo_degrees(self) -> list[float]:
        return [self.read_servo_degrees(index) for index in range(len(self.mappings))]

    def neutral(self) -> None:
        for index, mapping in enumerate(self.mappings):
            self.set_servo_degrees(index, mapping.neutral_deg)

    def read_battery(self, start_time: float, previous_samples: list[BatterySample]) -> BatterySample:
        elapsed_s = time.monotonic() - start_time
        voltage_v = float(self.robohat.get_battery_voltage())
        sample = BatterySample(
            elapsed_s=elapsed_s,
            voltage_v=voltage_v,
            percentage=int(self.robohat.get_battery_percentage_capacity()),
            status=str(self.robohat.get_battery_status()),
            drain_mv_per_min=drain_mv_per_min(previous_samples, elapsed_s, voltage_v),
        )
        previous_samples.append(sample)
        return sample

    def read_imu(self, start_time: float) -> ImuSample:
        elapsed_s = time.monotonic() - start_time
        acc = self.robohat.get_imu_acceleration()
        gyro = self.robohat.get_imu_gyro()
        mag = self.robohat.get_imu_magnetic_fields()
        return ImuSample(
            elapsed_s=elapsed_s,
            acc_x=float(acc[0]),
            acc_y=float(acc[1]),
            acc_z=float(acc[2]),
            gyro_x=float(gyro[0]),
            gyro_y=float(gyro[1]),
            gyro_z=float(gyro[2]),
            mag_x=float(mag[0]),
            mag_y=float(mag[1]),
            mag_z=float(mag[2]),
        )

    def calibrate_servo_readout(self, min_deg: float, max_deg: float) -> None:
        self.robohat.do_servo_fit_formula_readout_vs_angle_multiple_servos(
            min_deg,
            max_deg,
        )

    def close(self) -> None:
        if self.enable_servos:
            self.neutral()
            time.sleep(0.5)
            self.robohat.stop_servo_drivers()
        self.robohat.exit_program()


def drain_mv_per_min(
    samples: list[BatterySample],
    elapsed_s: float,
    voltage_v: float,
) -> float:
    if not samples or elapsed_s <= samples[0].elapsed_s:
        return 0.0
    first = samples[0]
    elapsed_min = (elapsed_s - first.elapsed_s) / 60.0
    return (first.voltage_v - voltage_v) * 1000.0 / elapsed_min


def battery_sample_row(sample: BatterySample) -> dict:
    return {
        "elapsed_s": f"{sample.elapsed_s:.6f}",
        "voltage_v": f"{sample.voltage_v:.6f}",
        "percentage": sample.percentage,
        "status": sample.status,
        "drain_mv_per_min": f"{sample.drain_mv_per_min:.6f}",
    }


def imu_sample_row(sample: ImuSample) -> dict:
    return {
        "elapsed_s": f"{sample.elapsed_s:.6f}",
        "acc_x": f"{sample.acc_x:.6f}",
        "acc_y": f"{sample.acc_y:.6f}",
        "acc_z": f"{sample.acc_z:.6f}",
        "gyro_x": f"{sample.gyro_x:.6f}",
        "gyro_y": f"{sample.gyro_y:.6f}",
        "gyro_z": f"{sample.gyro_z:.6f}",
        "mag_x": f"{sample.mag_x:.6f}",
        "mag_y": f"{sample.mag_y:.6f}",
        "mag_z": f"{sample.mag_z:.6f}",
    }


BATTERY_CSV_FIELDS = [
    "elapsed_s",
    "voltage_v",
    "percentage",
    "status",
    "drain_mv_per_min",
]

IMU_CSV_FIELDS = [
    "elapsed_s",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
]
