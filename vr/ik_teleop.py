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


class Clutch:
    """Delta clutch: while grip is held, the EE target follows the
    controller's pose *delta* from the grip-press moment, applied to the EE
    pose latched at that moment (anchor). Hysteresis avoids chatter."""

    ENGAGE, RELEASE = 0.8, 0.5

    def __init__(self, scale=1.0):
        self.scale = scale
        self.engaged = False
        self._p0 = self._R0 = self._T0 = None

    def update(self, grip, p_xr, R_xr, anchor_T):
        """One controller sample. anchor_T: current EE pose (latched on
        engage). Returns the target SE3 while engaged, else None."""
        if not self.engaged:
            if grip > self.ENGAGE:
                self.engaged = True
                self._p0, self._R0 = p_xr.copy(), R_xr.copy()
                self._T0 = pin.SE3(anchor_T.rotation.copy(), anchor_T.translation.copy())
            else:
                return None
        elif grip < self.RELEASE:
            self.engaged = False
            return None
        dp = self.scale * (AXIS_MAP @ (p_xr - self._p0))
        dR = AXIS_MAP @ (R_xr @ self._R0.T) @ AXIS_MAP.T
        return pin.SE3(dR @ self._T0.rotation, self._T0.translation + dp)
