#!/usr/bin/env python3
"""PiperX teleop link over UDP (pyAgxArm SDK).

Leader rig:   streams the local arm's joint angles as JSON datagrams.
Follower rig: listens on --port and applies the freshest sample to the arm.

    python piperx_setup.py --leader                            # sends to 127.0.0.1:8080
    python piperx_setup.py --leader --net enp6s0               # broadcast on that interface's subnet
    python piperx_setup.py --leader --addr 192.168.50.2        # unicast to a specific follower
    python piperx_setup.py --follower                          # listens on :8080 (local or cross-rig)
    python piperx_setup.py --home --can can0                   # return arm to zero pose

--leader polls the arm's joint feedback at --rate (default 100Hz) and publishes
it; it does not change the arm's control mode. Use the teach button to put the
arm in drag mode for hand-guiding.
CAN must be up first: sudo ip link set can0 up type can bitrate 1000000
"""

import argparse
import fcntl
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

# The repo's `pyAgxArm/` checkout (setup.py dir) shadows the installed package
# when running from this directory; point imports at the real package inside it.
_sdk_repo = Path(__file__).resolve().parent / "pyAgxArm"
if (_sdk_repo / "pyAgxArm").is_dir():
    sys.path.insert(0, str(_sdk_repo))

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

SPEED_PERCENT = 50  # follower tracking speed


def _iface_addr(iface, ioctl_code):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(s.fileno(), ioctl_code,
                             struct.pack("256s", iface.encode()[:15]))
        return socket.inet_ntoa(packed[20:24])
    except OSError:
        sys.exit(f"No IPv4 address on interface '{iface}'. Available: "
                 f"{', '.join(sorted(os.listdir('/sys/class/net')))}")
    finally:
        s.close()


def iface_ipv4(iface):
    """IPv4 address of a network interface (Linux)."""
    return _iface_addr(iface, 0x8915)  # SIOCGIFADDR


def iface_bcast(iface):
    """IPv4 broadcast address of a network interface (Linux)."""
    return _iface_addr(iface, 0x8919)  # SIOCGIFBRDADDR


def connect_arm(channel):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=PiperFW.DEFAULT,
        interface=os.environ.get("PIPERX_INTERFACE", "socketcan"),
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    # Link check: a powered arm streams feedback unsolicited at 200Hz.
    deadline = time.monotonic() + 2.0
    while robot.get_joint_angles() is None:
        if time.monotonic() > deadline:
            sys.exit(f"No data from arm on '{channel}'. Check arm power and CAN cable;\n"
                     f"if needed: sudo ip link set {channel} up type can bitrate 1000000")
        time.sleep(0.05)
    return robot


def enter_can_ctrl(robot):
    """Out of standby/teach into CAN control: reset -> joint mode -> enable
    (per the AgileX double_piper guide). The arm goes limp briefly."""
    robot.reset()
    time.sleep(1.0)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    time.sleep(0.2)
    while not robot.enable():
        time.sleep(0.01)
    robot.set_speed_percent(SPEED_PERCENT)


def run_leader(robot, gripper, sock, target, rate):
    period = 1.0 / rate
    while True:
        ja = robot.get_joint_angles()
        if ja is not None:
            msg = {"joints": list(ja.msg), "t": time.time()}
            gs = gripper.get_gripper_status() if gripper else None
            if gs is not None and gs.msg.mode == "width":
                msg["gripper"] = gs.msg.value  # meters
            sock.sendto(json.dumps(msg).encode(), target)
            print(f"tx joints: {[round(j, 4) for j in msg['joints']]}"
                  f" gripper: {msg.get('gripper')} (feedback {ja.hz:.0f}Hz)", flush=True)
        time.sleep(period)


def run_follower(robot, gripper, sock):
    while True:
        data = sock.recv(2048)
        try:
            while True:  # drain anything queued: act on the freshest sample only
                data = sock.recv(2048, socket.MSG_DONTWAIT)
        except BlockingIOError:
            pass
        msg = json.loads(data)
        joints = msg["joints"]
        print(f"rx joints: {[round(j, 4) for j in joints]} gripper: {msg.get('gripper')}", flush=True)
        robot.move_j(joints)
        if gripper and msg.get("gripper") is not None:
            gripper.move_gripper_m(msg["gripper"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--leader", action="store_true", help="stream joint angles out")
    mode.add_argument("--follower", action="store_true", help="stream joint angles in")
    mode.add_argument("--home", action="store_true", help="move the arm to the zero pose and exit")
    ap.add_argument("--addr", default=None, metavar="IP[:PORT]",
                    help="leader: send unicast to this follower (default: 127.0.0.1, or broadcast with --net)")
    ap.add_argument("--net", default=None, metavar="IFACE",
                    help="leader: broadcast on this ethernet interface's subnet (e.g. enp6s0)")
    ap.add_argument("--port", type=int, default=8080, help="port (default: 8080)")
    ap.add_argument("--can", default="can0", help="CAN channel (default: can0)")
    ap.add_argument("--rate", type=float, default=100.0, help="leader publish rate in Hz (default: 100)")
    args = ap.parse_args()

    # Resolve/validate networking before touching the arm.
    bcast = iface_bcast(args.net) if args.net else None

    robot = connect_arm(args.can)

    if args.home:
        print("Homing: entering CAN control (arm may sag briefly), then moving to zero pose.", flush=True)
        enter_can_ctrl(robot)
        robot.move_j([0.0] * robot.joint_nums)
        time.sleep(0.5)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            st = robot.get_arm_status()
            if st is not None and getattr(st.msg, "motion_status", None) == 0:
                print("Home reached:", [round(j, 4) for j in robot.get_joint_angles().msg])
                return
            time.sleep(0.1)
        print("Timed out waiting for home (15s).")
        return

    try:
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    except Exception:
        gripper = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if args.leader:
            if args.addr:
                host, _, p = args.addr.partition(":")
                target = (host, int(p) if p else args.port)
            elif bcast:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                target = (bcast, args.port)
            else:
                target = ("127.0.0.1", args.port)
            st = robot.get_arm_status()
            mode = getattr(st.msg, "ctrl_mode", "unknown") if st else "unknown"
            print(f"Leader: arm on {args.can} (ctrl mode: {mode}), "
                  f"streaming udp to {target[0]}:{target[1]} @ {args.rate:.0f}Hz", flush=True)
            run_leader(robot, gripper, sock, target, args.rate)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", args.port))
            print("Resetting arm into CAN control mode — it may sag briefly.", flush=True)
            enter_can_ctrl(robot)
            print(f"Follower: arm on {args.can} enabled, listening on udp :{args.port}", flush=True)
            run_follower(robot, gripper, sock)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
