"""Unit tests for ik_teleop. Run from vr/:
    uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v
"""
import math
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from ik_teleop import AXIS_MAP, DEFAULT_URDF, HOME, WIRE, Clutch, IkSolver, quat_to_mat

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
    right_xr = np.array([1.0, 0.0, 0.0])        # XR +x (right)
    assert np.allclose(AXIS_MAP @ up_xr, [0, 0, 1])           # robot +z (up)
    assert np.allclose(AXIS_MAP @ toward_user_xr, [-1, 0, 0])  # robot -x (backward)
    assert np.allclose(AXIS_MAP @ right_xr, [0, -1, 0])        # robot -y (left)


def test_quat_to_mat_identity_and_known():
    assert np.allclose(quat_to_mat(0, 0, 0, 1), np.eye(3))
    # 90 deg about z: x-axis maps to y-axis
    s = math.sin(math.pi / 4)
    R = quat_to_mat(0, 0, s, math.cos(math.pi / 4))
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)
    # Unnormalized input is normalized internally
    R2 = quat_to_mat(0, 0, 2 * s, 2 * math.cos(math.pi / 4))
    assert np.allclose(R, R2)


def _se3(R=None, p=(0, 0, 0)):
    return pin.SE3(np.eye(3) if R is None else R, np.array(p, dtype=float))


def test_clutch_hysteresis():
    c = Clutch()
    anchor = _se3(p=(0.3, 0.0, 0.2))
    p, R = np.zeros(3), np.eye(3)
    assert c.update(0.0, p, R, anchor) is None and not c.engaged
    assert c.update(0.7, p, R, anchor) is None and not c.engaged   # below engage
    assert c.update(0.9, p, R, anchor) is not None and c.engaged   # engages
    assert c.update(0.6, p, R, anchor) is not None and c.engaged   # held (above release)
    assert c.update(0.4, p, R, anchor) is None and not c.engaged   # releases


def test_clutch_position_delta_axis_mapped():
    c = Clutch(scale=2.0)
    anchor = _se3(p=(0.3, 0.0, 0.2))
    c.update(1.0, np.zeros(3), np.eye(3), anchor)
    # Move controller 0.1m along XR -z (away from user) => robot +x, scaled 2x
    T = c.update(1.0, np.array([0.0, 0.0, -0.1]), np.eye(3), anchor)
    assert np.allclose(T.translation, [0.5, 0.0, 0.2])
    assert np.allclose(T.rotation, np.eye(3))


def test_clutch_orientation_delta():
    c = Clutch()
    anchor = _se3(p=(0.3, 0.0, 0.2))
    c.update(1.0, np.zeros(3), np.eye(3), anchor)
    # Rotate controller 90deg about XR +y (up) => robot +z (up)
    s, co = math.sin(math.pi / 4), math.cos(math.pi / 4)
    T = c.update(1.0, np.zeros(3), quat_to_mat(0, s, 0, co), anchor)
    expected = quat_to_mat(0, 0, s, co)  # 90deg about robot z
    assert np.allclose(T.rotation, expected, atol=1e-12)
    assert np.allclose(T.translation, anchor.translation)


def test_clutch_regrip_does_not_jump():
    c = Clutch()
    a1 = _se3(p=(0.3, 0.0, 0.2))
    c.update(1.0, np.zeros(3), np.eye(3), a1)
    c.update(1.0, np.array([0.0, 0.1, 0.0]), np.eye(3), a1)  # moved up 0.1
    c.update(0.0, np.zeros(3), np.eye(3), a1)                 # release
    # Re-grip with the controller somewhere totally different: first update
    # latches the NEW anchor and returns it unchanged - no jump.
    a2 = _se3(p=(0.3, 0.0, 0.3))
    T = c.update(1.0, np.array([5.0, 5.0, 5.0]), np.eye(3), a2)
    assert np.allclose(T.translation, a2.translation)


@pytest.fixture(scope="module")
def solver():
    return IkSolver(DEFAULT_URDF)


def test_model_reduced_to_six_joints(solver):
    assert solver.model.nq == 6
    assert solver.model.nv == 6


def test_home_pose_is_sane(solver):
    T = solver.fk(HOME)
    # Ready pose must put the gripper in front of the base, above the table.
    assert T.translation[0] > 0.1, f"home EE x={T.translation[0]:.3f}, want forward"
    assert T.translation[2] > 0.05, f"home EE z={T.translation[2]:.3f}, want above base"


def test_fk_ik_roundtrip(solver):
    rng = np.random.default_rng(42)
    lo, hi = solver.model.lowerPositionLimit, solver.model.upperPositionLimit
    margin = 0.1 * (hi - lo)
    for _ in range(20):
        q_true = rng.uniform(lo + margin, hi - margin)
        T_goal = solver.fk(q_true)
        q0 = np.clip(q_true + rng.uniform(-0.1, 0.1, 6), lo, hi)  # warm start nearby
        q_sol = solver.solve(T_goal, q0, iters=100)
        T_sol = solver.fk(q_sol)
        assert np.linalg.norm(T_sol.translation - T_goal.translation) < 1e-3
        rot_err = pin.log3(T_sol.rotation.T @ T_goal.rotation)
        assert np.linalg.norm(rot_err) < 0.01


def test_solve_respects_joint_limits(solver):
    # An unreachable target (1m up) must still yield an in-limits solution.
    T_goal = pin.SE3(np.eye(3), np.array([0.0, 0.0, 1.5]))
    q = solver.solve(T_goal, HOME.copy(), iters=100)
    assert np.all(q >= solver.model.lowerPositionLimit - 1e-9)
    assert np.all(q <= solver.model.upperPositionLimit + 1e-9)


def test_solve_step_cap(solver):
    T_goal = pin.SE3(np.eye(3), np.array([0.4, 0.3, 0.4]))  # far from HOME's EE
    q = solver.solve(T_goal, HOME.copy(), step_cap=0.02)
    assert np.max(np.abs(q - HOME)) <= 0.02 + 1e-9
    assert np.all(q >= solver.model.lowerPositionLimit - 1e-9)
    assert np.all(q <= solver.model.upperPositionLimit + 1e-9)
