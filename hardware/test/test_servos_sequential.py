"""Sequentially test the baby robot's real servos.

By default this script uses ``BabyRobotHardware`` and maps requested servo
angles through each actuator's neutral position and sign. For example,
``--servo-angles 10 170`` means "negative side, then positive side"; a servo
with ``sign=-1`` receives the opposite raw Robohat command order.

Use ``--raw-robohat`` only when you want the old diagnostic behavior that
bypasses ``BabyRobotHardware`` and commands raw PWM channels through
``SerTestClass``.
"""
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time


ROBOHAT_ROOT = Path(__file__).resolve().parents[2] / "robohat"
if str(ROBOHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOHAT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SerTest import SerTestClass
from hardware.baby_hardware import DEFAULT_SERVO_MAPPINGS, BabyRobotHardware


ROBOT_CHANNELS = [16, 17, 18, 19, 20, 21, 22, 23]
ROBOT_ACTUATOR_NAMES = [
    "robot1_hinge_0servo",
    "robot1_hinge_1servo",
    "robot1_hinge_2servo",
    "robot1_hinge_3servo",
    "robot1_hinge_4servo",
    "robot1_hinge_5servo",
    "robot1_hinge_6servo",
    "robot1_hinge_7servo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move baby robot servos sequentially.")
    parser.add_argument("--servos", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--servo-angles", type=float, nargs="+", default=[10.0, 90.0, 170.0, 90.0])
    parser.add_argument("--raw-servo-angles", action="store_true")
    parser.add_argument("--offsets", type=float, nargs="+", default=None)
    parser.add_argument("--raw-robohat", action="store_true")
    parser.add_argument("--channels", type=int, nargs="+", default=ROBOT_CHANNELS)
    parser.add_argument("--angles", type=float, nargs="+", default=[10.0, 90.0, 170.0, 90.0])
    parser.add_argument("--neutral", type=float, default=90.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--between-servos-s", type=float, default=1.0)
    parser.add_argument("--delay-mode", action="store_true")
    return parser.parse_args()


def run_baby_hardware_test(args: argparse.Namespace) -> None:
    robot = BabyRobotHardware(direct_mode=not args.delay_mode)
    try:
        print("Baby robot sequential servo test:")
        for mapping in DEFAULT_SERVO_MAPPINGS:
            print(
                f"  servo {mapping.actuator_index}: {mapping.actuator_name} "
                f"-> PWM channel {mapping.channel}, neutral={mapping.neutral_deg:.1f}, "
                f"sign={mapping.sign:+.0f}",
            )
        print(f"Test servos: {args.servos}")
        if args.offsets is None:
            print(f"Requested servo angles: {args.servo_angles}")
            if args.raw_servo_angles:
                print("Raw servo angle mode: mapping.sign is not applied.")
            else:
                print("Mapped servo angle mode: mapping.sign is applied around neutral.")
        else:
            print(f"Joint offsets: {args.offsets}")
            print("mapping.sign is applied in joint offset mode.")
        robot.neutral()
        time.sleep(args.between_servos_s)

        for servo_index in args.servos:
            mapping = DEFAULT_SERVO_MAPPINGS[servo_index]
            print("\n" + "=" * 50)
            print(
                f"Watch servo {servo_index}: {mapping.actuator_name} "
                f"(PWM channel {mapping.channel}, sign={mapping.sign:+.0f})",
            )
            print("=" * 50)
            if args.offsets is None:
                for angle in args.servo_angles:
                    if args.raw_servo_angles:
                        command_deg = robot.set_servo_degrees(servo_index, angle)
                    else:
                        command_deg = robot.set_joint_degrees(servo_index, angle - 90.0)
                    print(
                        f"servo {servo_index} requested {angle:.1f} deg "
                        f"-> raw command {command_deg:.1f} deg",
                    )
                    time.sleep(args.hold_s)
            else:
                for offset in args.offsets:
                    command_deg = robot.set_joint_degrees(servo_index, offset)
                    print(
                        f"servo {servo_index} joint offset {offset:+.1f} deg "
                        f"-> raw command {command_deg:.1f} deg",
                    )
                    time.sleep(args.hold_s)
            robot.set_servo_degrees(servo_index, mapping.neutral_deg)
            time.sleep(args.between_servos_s)
    finally:
        robot.close()


def run_raw_robohat_test(args: argparse.Namespace) -> None:
    ser_test = SerTestClass()
    robohat = ser_test._SerTestClass__robohat

    try:
        print("Robot servo command map:")
        for actuator_index, channel in enumerate(ROBOT_CHANNELS):
            print(
                f"  actuator {actuator_index}: "
                f"{ROBOT_ACTUATOR_NAMES[actuator_index]} -> PWM channel {channel}",
            )
        print(f"Test channels: {args.channels}")
        print(f"Angles: {args.angles}")
        print("First proving the exact SerTest all-servo command path:")
        for angle in [args.neutral, args.angles[0], args.neutral]:
            print(f"all channels -> {angle:.1f} deg")
            robohat.set_servo_multiple_angles([angle] * 32)
            time.sleep(args.hold_s)

        for channel in args.channels:
            actuator_label = "not in robot map"
            if channel in ROBOT_CHANNELS:
                actuator_index = ROBOT_CHANNELS.index(channel)
                actuator_label = (
                    f"actuator {actuator_index}: "
                    f"{ROBOT_ACTUATOR_NAMES[actuator_index]}"
                )
            print("\n" + "=" * 50)
            print(f"Watch PWM channel {channel} ({actuator_label})")
            print("=" * 50)
            for angle in args.angles:
                commands = [args.neutral] * 32
                commands[channel] = angle
                print(f"channel {channel} -> {angle:.1f} deg, all others -> {args.neutral:.1f} deg")
                robohat.set_servo_multiple_angles(commands)
                time.sleep(args.hold_s)
    finally:
        print("\nReturning all servo slots to neutral")
        robohat.set_servo_multiple_angles([args.neutral] * 32)
        time.sleep(1.0)
        robohat.exit_program()


def main() -> None:
    args = parse_args()
    if args.raw_robohat:
        run_raw_robohat_test(args)
    else:
        run_baby_hardware_test(args)


if __name__ == "__main__":
    main()
