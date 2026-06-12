#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pin", "numpy"]
# ///
"""6-DOF delta-clutched VR teleop for the AgileX PiPER arm.

Listens for Meta Quest controller poses (UDP JSON from quest_server.py),
solves IK with Pinocchio against the PiPER URDF, and streams joint commands
as 40-byte WIRE datagrams to piperx_setup.py --follower (it impersonates a
--leader, so the follower is unchanged).

Hold the GRIP button to clutch in: the end-effector tracks your hand's
position+orientation deltas (latched at grip press, so re-gripping never
jumps). The analog TRIGGER sets gripper width (pulled = closed).

Run on the rig (quest_server.py must target this machine: --udp <rig-ip>:5557):
    uv run ik_teleop.py                 # sends to follower on 127.0.0.1:8080
    uv run ik_teleop.py --dry-run       # print solutions, send nothing
"""

import argparse
import json
import math
import socket
import struct
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE.parent / "piper_ros" / "src" / "piper_description" / "urdf" / "piper_description.urdf"

# Wire format shared with piperx_setup.py:
# little-endian | t: float64 | seq: uint32 | j1..j6: float32 | gripper: float32 (NaN = none)
WIRE = struct.Struct("<dI6ff")

# WebXR (x right, y up, z toward user) -> robot base (x forward, z up).
# Applied to *deltas only*, so absolute headset calibration never matters.
AXIS_MAP = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# Nominal ready pose (rad): elbow bent, gripper reaching forward. The follower
# aligns here with a planned move_j on its first datagram.
HOME = np.array([0.0, 0.9, -0.8, 0.0, 0.6, 0.0])


def quat_to_mat(x, y, z, w):
    """Rotation matrix from an xyzw quaternion (normalizes input)."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
