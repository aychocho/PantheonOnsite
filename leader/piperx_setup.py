#!/usr/bin/env python3
"""PiperX bring-up test using the pyAgxArm SDK.

Connects over CAN, enables the arm, and prints firmware/state to verify the
stack end to end. Read-only by default; --move adds a small joint sweep.

    python piperx_setup.py                 # connect + enable + read state
    python piperx_setup.py --move          # also sweep joints and return to zero
    python piperx_setup.py --can can1      # non-default CAN channel

Prerequisite: CAN must be up, e.g.
    sudo ip link set can0 up type can bitrate 1000000
"""

import argparse
import sys
import time
from pathlib import Path

# The repo's `pyAgxArm/` checkout (setup.py dir) shadows the installed package
# when running from this directory; point imports at the real package inside it.
_sdk_repo = Path(__file__).resolve().parent / "pyAgxArm"
if (_sdk_repo / "pyAgxArm").is_dir():
    sys.path.insert(0, str(_sdk_repo))

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

FW_CHOICES = {"default": PiperFW.DEFAULT, "v183": PiperFW.V183, "v188": PiperFW.V188}


def check_can_up(channel):
    state_file = Path(f"/sys/class/net/{channel}/operstate")
    if not state_file.exists():
        return f"CAN interface '{channel}' does not exist."
    if state_file.read_text().strip() == "down":
        return (f"CAN interface '{channel}' is DOWN. Bring it up with:\n"
                f"  sudo ip link set {channel} up type can bitrate 1000000")
    return None


def wait_motion_done(robot, timeout=10.0):
    time.sleep(0.5)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            return True
        time.sleep(0.1)
    print(f"  WARNING: motion not confirmed done within {timeout:.0f}s")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="CAN channel (default: can0)")
    ap.add_argument("--interface", default="socketcan",
                    help="python-can interface: socketcan, virtual, slcan (default: socketcan)")
    ap.add_argument("--fw", choices=FW_CHOICES, default="default",
                    help="firmware family: default (<=S-V1.8-2), v183, v188 (>=S-V1.8-8)")
    ap.add_argument("--move", action="store_true", help="run a small joint sweep (arm will move!)")
    ap.add_argument("--speed", type=int, default=20, help="speed percent for --move (default: 20)")
    ap.add_argument("--timeout", type=float, default=10.0, help="enable timeout in seconds (default: 10)")
    args = ap.parse_args()

    if args.interface == "socketcan":
        err = check_can_up(args.can)
        if err:
            print(err)
            return 1

    print(f"[1/4] Connecting to PiperX on {args.can} (fw={args.fw}) ...")
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=FW_CHOICES[args.fw],
        interface=args.interface,
        channel=args.can,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    print(f"[2/4] Enabling arm (timeout {args.timeout:.0f}s) ...")
    deadline = time.monotonic() + args.timeout
    while not robot.enable():
        if time.monotonic() > deadline:
            print(f"FAIL: arm did not enable within {args.timeout:.0f}s. "
                  "Is it powered and connected to the CAN adapter?")
            return 1
        time.sleep(0.01)
    print("  arm enabled")

    print("[3/4] Reading state ...")
    time.sleep(0.5)  # let feedback streams populate
    fw = robot.get_firmware()
    print(f"  firmware:     {fw.msg if fw else 'n/a'}")
    status = robot.get_arm_status()
    print(f"  arm status:   {status.msg if status else 'n/a'}")
    ja = robot.get_joint_angles()
    print(f"  joints (rad): {ja.msg if ja else 'n/a'}  ({ja.hz:.0f}Hz)" if ja else "  joints: n/a")
    pose = robot.get_flange_pose()
    print(f"  flange pose:  {pose.msg if pose else 'n/a'}")
    print(f"  feedback ok:  {robot.is_ok()}")

    if args.move:
        print(f"[4/4] Motion test at {args.speed}% speed ...")
        robot.set_speed_percent(args.speed)
        sweep = [0.0, 0.2, -0.2, 0.3, -0.2, 0.5][: robot.joint_nums]
        print(f"  move_j {sweep}")
        robot.move_j(sweep)
        wait_motion_done(robot)
        print("  move_j back to zero")
        robot.move_j([0.0] * robot.joint_nums)
        wait_motion_done(robot)
    else:
        print("[4/4] Skipping motion test (pass --move to enable)")

    print("PASS: PiperX setup OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
