#!/usr/bin/env python3
"""ZMQ joint-angle controller for the AgileX PiPER arm.

Binds a ZMQ SUB socket (CONFLATE=1, so only the freshest command is kept)
and drives the arm over CAN at a fixed rate, decoupled from the network rate.

Message format (JSON, single-part):
    {"joints": [j1..j6] (radians), "gripper": meters (optional),
     "seq": int (optional), "t": unix-float (optional)}

Until the first valid command arrives, nothing is sent to the arm.
If commands stop arriving (watchdog timeout), the last target is held.
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import zmq

# The repo's `piper_sdk/` checkout (setup.py dir) shadows the installed package
# when running from this directory; point imports at the real package inside it.
_sdk_repo = Path(__file__).resolve().parent / "piper_sdk"
if (_sdk_repo / "piper_sdk").is_dir():
    sys.path.insert(0, str(_sdk_repo))

from piper_sdk import C_PiperInterface_V2

RAD_TO_MILLIDEG = 180000.0 / 3.14159265358979  # rad -> 0.001 degree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("joint_controller")


def parse_command(raw):
    """Parse and validate a command message. Returns dict or raises ValueError."""
    msg = json.loads(raw)
    joints = msg.get("joints")
    if not isinstance(joints, list) or len(joints) != 6:
        raise ValueError(f"'joints' must be a list of 6 floats, got: {joints!r}")
    joints = [float(j) for j in joints]
    gripper = msg.get("gripper")
    if gripper is not None:
        gripper = float(gripper)
    return {"joints": joints, "gripper": gripper, "seq": msg.get("seq"), "t": msg.get("t")}


def connect_arm(can_name, enable_timeout=10.0):
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    deadline = time.time() + enable_timeout
    while not piper.EnablePiper():
        if time.time() > deadline:
            raise RuntimeError(
                f"Could not enable arm on {can_name} within {enable_timeout}s. "
                "Is the CAN interface up? (see piper_sdk/piper_sdk/can_activate.sh)"
            )
        time.sleep(0.01)
    return piper


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="CAN interface name (default: can0)")
    ap.add_argument("--bind", default="tcp://*:5555", help="ZMQ bind address (default: tcp://*:5555)")
    ap.add_argument("--rate", type=float, default=200.0, help="control loop rate in Hz (default: 200)")
    ap.add_argument("--speed", type=int, default=50, help="arm speed percentage 0-100 (default: 50)")
    ap.add_argument("--timeout", type=float, default=0.25, help="watchdog timeout in seconds (default: 0.25)")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)  # must be set before bind
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.bind(args.bind)
    log.info("Listening on %s", args.bind)

    log.info("Connecting to arm on %s ...", args.can)
    piper = connect_arm(args.can)
    log.info("Arm enabled.")

    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    target = None
    last_rx = None
    stale = False
    period = 1.0 / args.rate
    next_t = time.monotonic()
    loop_count = 0
    last_status = time.monotonic()

    while running:
        # Drain the freshest command (CONFLATE keeps at most one).
        try:
            raw = sock.recv(zmq.NOBLOCK)
            try:
                cmd = parse_command(raw)
                if target is None:
                    log.info("First command received: %s", cmd["joints"])
                target = cmd
                last_rx = time.monotonic()
                if stale:
                    log.info("Commands resumed.")
                    stale = False
            except (ValueError, json.JSONDecodeError) as e:
                log.warning("Dropping malformed message: %s", e)
        except zmq.Again:
            pass

        if target is not None:
            if not stale and time.monotonic() - last_rx > args.timeout:
                stale = True
                log.warning("No command for %.2fs - holding last position.", args.timeout)

            piper.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)
            j = [round(r * RAD_TO_MILLIDEG) for r in target["joints"]]
            piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
            if target["gripper"] is not None:
                piper.GripperCtrl(abs(round(target["gripper"] * 1e6)), 1000, 0x01, 0)

        loop_count += 1
        now = time.monotonic()
        if now - last_status >= 1.0:
            age = (now - last_rx) if last_rx is not None else None
            log.info(
                "rate=%.0fHz target=%s cmd_age=%s",
                loop_count / (now - last_status),
                [f"{r:.3f}" for r in target["joints"]] if target else None,
                f"{age:.3f}s" if age is not None else "n/a",
            )
            loop_count = 0
            last_status = now

        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()  # fell behind; reset deadline

    log.info("Shutting down (arm holds via firmware).")
    sock.close(0)
    ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
