"""Unit tests for ik_teleop. Run from vr/:
    uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v
"""
import math
from pathlib import Path

import numpy as np

import ik_teleop
from ik_teleop import AXIS_MAP, WIRE, quat_to_mat

ROOT = Path(__file__).resolve().parent.parent


def test_wire_matches_follower():
    # Must stay byte-identical to the follower's struct in piperx_setup.py.
    src = (ROOT / "leader" / "piperx_setup.py").read_text()
    assert '"<dI6ff"' in src
    assert WIRE.format == "<dI6ff"
    assert WIRE.size == 40


def test_axis_map_is_rotation():
    assert np.allclose(AXIS_MAP @ AXIS_MAP.T, np.eye(3))
    assert math.isclose(np.linalg.det(AXIS_MAP), 1.0)


def test_axis_map_directions():
    up_xr = np.array([0.0, 1.0, 0.0])          # XR +y (up)
    toward_user_xr = np.array([0.0, 0.0, 1.0])  # XR +z (toward user)
    assert np.allclose(AXIS_MAP @ up_xr, [0, 0, 1])           # robot +z (up)
    assert np.allclose(AXIS_MAP @ toward_user_xr, [-1, 0, 0])  # robot -x (backward)


def test_quat_to_mat_identity_and_known():
    assert np.allclose(quat_to_mat(0, 0, 0, 1), np.eye(3))
    # 90 deg about z: x-axis maps to y-axis
    s = math.sin(math.pi / 4)
    R = quat_to_mat(0, 0, s, math.cos(math.pi / 4))
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)
    # Unnormalized input is normalized internally
    R2 = quat_to_mat(0, 0, 2 * s, 2 * math.cos(math.pi / 4))
    assert np.allclose(R, R2)
