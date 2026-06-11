#!/usr/bin/env python3
"""Test publisher for joint_controller.py.

Sends joint-angle commands (JSON) on a ZMQ PUB socket. Either a fixed pose:
    python send_joints.py --pose 0 0.2 -0.2 0.3 -0.2 0.5
or a small sine sweep around zero:
    python send_joints.py --sine
"""

import argparse
import json
import math
import sys
import time

import zmq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default="tcp://localhost:5555", help="controller address (default: tcp://localhost:5555)")
    ap.add_argument("--rate", type=float, default=100.0, help="send rate in Hz (default: 100)")
    ap.add_argument("--gripper", type=float, default=None, help="gripper opening in meters (optional)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pose", type=float, nargs=6, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                      help="fixed joint angles in radians")
    mode.add_argument("--sine", action="store_true", help="sine sweep on all joints (0.2 rad, 0.1 Hz)")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.connect(args.connect)
    time.sleep(0.2)  # let the connection establish

    print(f"Publishing to {args.connect} at {args.rate}Hz (Ctrl-C to stop)")
    period = 1.0 / args.rate
    seq = 0
    t0 = time.time()
    try:
        while True:
            if args.sine:
                joints = [0.2 * math.sin(2 * math.pi * 0.1 * (time.time() - t0))] * 6
            else:
                joints = args.pose
            msg = {"joints": joints, "seq": seq, "t": time.time()}
            if args.gripper is not None:
                msg["gripper"] = args.gripper
            sock.send_string(json.dumps(msg))
            seq += 1
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
