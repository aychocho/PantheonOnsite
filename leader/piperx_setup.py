#!/usr/bin/env python3
"""PiperX teleop link over ZMQ (pyAgxArm SDK).

Leader rig:   streams the local arm's joint angles out on a PUB socket.
Follower rig: applies joint angles from a SUB socket to the local arm.

    python piperx_setup.py --leader                            # binds tcp://*:8080
    python piperx_setup.py --follower                          # connects to localhost:8080
    python piperx_setup.py --follower --addr tcp://10.103.0.45:8080   # cross-rig later
    python piperx_setup.py --home --can can0                   # return arm to zero pose

--leader polls the arm's joint feedback at --rate (default 100Hz) and publishes
it; it does not change the arm's control mode. Use the teach button to put the
arm in drag mode for hand-guiding.
CAN must be up first: sudo ip link set can0 up type can bitrate 1000000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The repo's `pyAgxArm/` checkout (setup.py dir) shadows the installed package
# when running from this directory; point imports at the real package inside it.
_sdk_repo = Path(__file__).resolve().parent / "pyAgxArm"
if (_sdk_repo / "pyAgxArm").is_dir():
    sys.path.insert(0, str(_sdk_repo))

import zmq
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

SPEED_PERCENT = 50  # follower tracking speed, matches joint_controller.py default


def connect_arm(channel, require_feedback=True):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=PiperFW.DEFAULT,
        interface=os.environ.get("PIPERX_INTERFACE", "socketcan"),
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    # Link check: a normal arm streams feedback unsolicited; a master/leader
    # arm sends control frames instead, and may be quiet until it is moved.
    deadline = time.monotonic() + 2.0
    while robot.get_joint_angles() is None and robot.get_leader_joint_angles() is None:
        if time.monotonic() > deadline:
            if require_feedback:
                sys.exit(f"No data from arm on '{channel}'. Check arm power and CAN cable;\n"
                         f"if needed: sudo ip link set {channel} up type can bitrate 1000000")
            print(f"No data from arm on '{channel}' yet — master arms can be quiet at rest; "
                  "continuing, drag the arm to start the stream.", flush=True)
            break
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


def run_leader(robot, gripper, sock, rate):
    period = 1.0 / rate
    while True:
        # Poll the standard feedback; fall back to master-arm frames in case
        # the arm is still configured as a leader/master.
        ja = robot.get_joint_angles() or robot.get_leader_joint_angles()
        if ja is not None:
            msg = {"joints": list(ja.msg), "t": time.time()}
            gs = gripper.get_gripper_status() if gripper else None
            if gs is not None and gs.msg.mode == "width":
                msg["gripper"] = gs.msg.value  # meters
            sock.send_string(json.dumps(msg))
            print(f"tx joints: {[round(j, 4) for j in msg['joints']]}"
                  f" gripper: {msg.get('gripper')} (feedback {ja.hz:.0f}Hz)", flush=True)
        time.sleep(period)


def run_follower(robot, gripper, sock):
    while True:
        msg = json.loads(sock.recv())  # CONFLATE: always the freshest sample
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
    ap.add_argument("--addr", default=None,
                    help="override: leader bind addr / follower connect addr (default: localhost, port 8080)")
    ap.add_argument("--port", type=int, default=8080, help="port (default: 8080)")
    ap.add_argument("--can", default="can0", help="CAN channel (default: can0)")
    ap.add_argument("--rate", type=float, default=100.0, help="leader publish rate in Hz (default: 100)")
    args = ap.parse_args()

    robot = connect_arm(args.can, require_feedback=args.follower or args.home)

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
    ctx = zmq.Context()
    try:
        if args.leader:
            addr = args.addr or f"tcp://*:{args.port}"
            sock = ctx.socket(zmq.PUB)
            sock.bind(addr)
            st = robot.get_arm_status()
            mode = getattr(st.msg, "ctrl_mode", "unknown") if st else "unknown"
            print(f"Leader: arm on {args.can} (ctrl mode: {mode}), publishing {addr} @ {args.rate:.0f}Hz",
                  flush=True)
            run_leader(robot, gripper, sock, args.rate)
        else:
            addr = args.addr or f"tcp://localhost:{args.port}"
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.CONFLATE, 1)  # must be set before connect
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.connect(addr)
            print("Resetting arm into CAN control mode — it may sag briefly.", flush=True)
            enter_can_ctrl(robot)
            print(f"Follower: arm on {args.can} enabled, listening on {addr}", flush=True)
            run_follower(robot, gripper, sock)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
