# VR IK Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vr/ik_teleop.py` — 6-DOF delta-clutched VR teleop: Quest controller pose (UDP JSON in) → Pinocchio IK → `piperx_setup.py --follower` (UDP 40-byte WIRE out), right arm only.

**Architecture:** Single PEP-723 script (matches `quest_server.py` pattern) plus a test file. Pure-logic core (`quat_to_mat`, `Clutch`, `IkSolver`, `Teleop.tick`) is socket-free and unit-tested; `main()` is only socket plumbing. The node impersonates a `piperx_setup.py --leader`, so neither existing script changes.

**Tech Stack:** Python ≥3.9, `pin` (Pinocchio, PyPI name is `pin`, imported as `pinocchio`), `numpy`, `pytest` (test-time only), run via `uv`.

**Spec:** `docs/superpowers/specs/2026-06-11-vr-ik-teleop-design.md`

**Test command (used by every task):**
```bash
cd /home/aycho/personal/interviews/pantheon/onsite/vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v
```
First run downloads Pinocchio wheels; subsequent runs are cached.

**Reference facts (verified against the repo):**
- Quest UDP JSON (from `quest_server.py`, port 5557): `{"t": float, "seq": int, "t_recv": float, "controllers": [{"hand": "right"|"left", "pos": [x,y,z], "quat": [x,y,z,w], "trigger": 0..1, "grip": 0..1, ...}]}`
- Follower wire format (`leader/piperx_setup.py`): `struct.Struct("<dI6ff")` = 40 bytes: t float64, seq uint32, j1..j6 float32 (radians), gripper float32 (meters, NaN = none). Follower listens on UDP :8080, aligns to the **first** datagram with a planned `move_j`, then high-follows.
- URDF `piper_ros/src/piper_description/urdf/piper_description.urdf`: revolute `joint1..joint6`, prismatic gripper fingers `joint7`/`joint8` (lock these), EE frame = link `gripper_base`. Joint limits: j1 ±2.618, j2 [0, 3.14], j3 [−2.967, 0], j4 ±1.745, j5 ±1.22, j6 ±2.0944.
- WebXR axes: x right, y up, z toward user. Robot base: x forward, z up. Axis map (deltas only): robot x = −z_xr, robot y = −x_xr, robot z = y_xr.

---

### Task 1: Scaffold — constants, axis map, quaternion → rotation matrix

**Files:**
- Create: `vr/ik_teleop.py`
- Create: `vr/test_ik_teleop.py`

- [ ] **Step 1: Write the failing tests**

Create `vr/test_ik_teleop.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ik_teleop'`

- [ ] **Step 3: Write the scaffold**

Create `vr/ik_teleop.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add vr/ik_teleop.py vr/test_ik_teleop.py
git commit -m "feat(vr): ik_teleop scaffold - wire format, axis map, quat math"
```

---

### Task 2: Clutch — grip hysteresis + 6-DOF delta latching

**Files:**
- Modify: `vr/ik_teleop.py` (append after `quat_to_mat`)
- Modify: `vr/test_ik_teleop.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `vr/test_ik_teleop.py`:

```python
import pinocchio as pin

from ik_teleop import Clutch


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 4 new FAIL — `ImportError: cannot import name 'Clutch'`

- [ ] **Step 3: Implement Clutch**

Append to `vr/ik_teleop.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add vr/ik_teleop.py vr/test_ik_teleop.py
git commit -m "feat(vr): delta clutch with grip hysteresis"
```

---

### Task 3: IkSolver — reduced model, FK, damped-least-squares IK

**Files:**
- Modify: `vr/ik_teleop.py` (append after `Clutch`)
- Modify: `vr/test_ik_teleop.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `vr/test_ik_teleop.py`:

```python
import pytest

from ik_teleop import DEFAULT_URDF, HOME, IkSolver


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 5 new FAIL — `ImportError: cannot import name 'IkSolver'`

- [ ] **Step 3: Implement IkSolver**

Append to `vr/ik_teleop.py`:

```python
class IkSolver:
    """Damped-least-squares IK on the PiPER URDF (gripper fingers locked)."""

    def __init__(self, urdf_path=DEFAULT_URDF, ee_frame="gripper_base"):
        full = pin.buildModelFromUrdf(str(urdf_path))
        lock = [full.getJointId(n) for n in ("joint7", "joint8")]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(ee_frame)

    def fk(self, q):
        """End-effector pose (SE3 copy) for configuration q."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.fid]
        return pin.SE3(T.rotation.copy(), T.translation.copy())

    def solve(self, T_target, q0, iters=10, damping=1e-3, tol=1e-4, step_cap=None):
        """Iterate DLS from warm start q0. Always returns an in-limits q;
        if step_cap is set, |q - q0| <= step_cap per joint (bounds EE speed
        even if the target jumps)."""
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        q = q0.copy()
        for _ in range(iters):
            err = pin.log(self.fk(q).actInv(T_target)).vector
            if np.linalg.norm(err) < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid,
                                         pin.ReferenceFrame.LOCAL)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + dq, lo, hi)
        if step_cap is not None:
            q = q0 + np.clip(q - q0, -step_cap, step_cap)
        return q
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 13 PASS. If `test_home_pose_is_sane` fails, adjust `HOME` (keep j2 in [0, 3.14], j3 in [−2.967, 0]) until the FK prints a forward/raised pose — e.g. try j2=1.2, j3=−1.0, j5=0.4 — and re-run; do not weaken the assertions.

- [ ] **Step 5: Commit**

```bash
git add vr/ik_teleop.py vr/test_ik_teleop.py
git commit -m "feat(vr): pinocchio DLS ik solver with limits and step cap"
```

---

### Task 4: Teleop — per-tick logic (parse sample, clutch, solve, gripper)

**Files:**
- Modify: `vr/ik_teleop.py` (append after `IkSolver`)
- Modify: `vr/test_ik_teleop.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `vr/test_ik_teleop.py`:

```python
from ik_teleop import Teleop


def _quest_msg(hand="right", pos=(0, 0, 0), quat=(0, 0, 0, 1), trigger=0.0, grip=0.0):
    return {"t": 1.0, "controllers": [
        {"hand": hand, "pos": list(pos), "quat": list(quat),
         "trigger": trigger, "grip": grip}]}


@pytest.fixture()
def teleop(solver):
    return Teleop(solver)


def test_tick_holds_home_without_grip(teleop):
    q, grip_m = teleop.tick(_quest_msg(grip=0.0))
    assert np.allclose(q, HOME)
    q, _ = teleop.tick(None)  # no sample this tick
    assert np.allclose(q, HOME)


def test_tick_ignores_other_hand(teleop):
    q, _ = teleop.tick(_quest_msg(hand="left", grip=1.0, pos=(0, 0.2, 0)))
    assert np.allclose(q, HOME)
    assert not teleop.clutch.engaged


def test_tick_tracks_while_gripped(teleop):
    teleop.tick(_quest_msg(grip=1.0))                       # engage at origin
    target_before = teleop.solver.fk(teleop.q).translation.copy()
    for _ in range(50):                                      # move up 0.1m in XR => +z robot
        q, _ = teleop.tick(_quest_msg(grip=1.0, pos=(0, 0.1, 0)))
    moved = teleop.solver.fk(q).translation
    assert moved[2] - target_before[2] > 0.08               # converged most of the way
    assert not np.allclose(q, HOME)


def test_tick_freezes_on_release(teleop):
    teleop.tick(_quest_msg(grip=1.0))
    for _ in range(50):
        teleop.tick(_quest_msg(grip=1.0, pos=(0, 0.1, 0)))
    q_held, _ = teleop.tick(_quest_msg(grip=0.0, pos=(0.5, 0.5, 0.5)))
    q_after, _ = teleop.tick(_quest_msg(grip=0.0, pos=(-0.5, 0.1, 0.9)))
    assert np.allclose(q_held, q_after)                      # frozen


def test_trigger_maps_to_gripper_width(teleop):
    _, w_open = teleop.tick(_quest_msg(trigger=0.0))
    _, w_closed = teleop.tick(_quest_msg(trigger=1.0))
    _, w_half = teleop.tick(_quest_msg(trigger=0.5))
    assert math.isclose(w_open, 0.07)
    assert math.isclose(w_closed, 0.0)
    assert math.isclose(w_half, 0.035)


def test_gripper_nan_until_first_sample(solver):
    t = Teleop(solver)
    _, w = t.tick(None)
    assert math.isnan(w)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 6 new FAIL — `ImportError: cannot import name 'Teleop'`

- [ ] **Step 3: Implement Teleop**

Append to `vr/ik_teleop.py`:

```python
class Teleop:
    """Socket-free teleop core: feed Quest JSON dicts into tick(), get back
    (q, gripper_width) to put on the wire."""

    def __init__(self, solver, hand="right", scale=1.0, max_grip=0.07,
                 step_cap=0.03):
        self.solver = solver
        self.hand = hand
        self.clutch = Clutch(scale)
        self.q = HOME.copy()
        self.gripper = math.nan  # NaN on the wire = "no gripper command"
        self.max_grip = max_grip
        self.step_cap = step_cap
        self.T_target = None
        self.ik_err = 0.0

    def tick(self, msg):
        """Consume one Quest message (or None). Returns (q, gripper_width)."""
        c = None
        if msg:
            c = next((c for c in msg.get("controllers", [])
                      if c.get("hand") == self.hand), None)
        if c is not None:
            p = np.asarray(c["pos"], dtype=float)
            R = quat_to_mat(*c["quat"])
            anchor = self.solver.fk(self.q)
            self.T_target = self.clutch.update(c.get("grip", 0.0), p, R, anchor)
            self.gripper = (1.0 - float(c.get("trigger", 0.0))) * self.max_grip
        if self.clutch.engaged and self.T_target is not None:
            self.q = self.solver.solve(self.T_target, self.q,
                                       step_cap=self.step_cap)
            err = pin.log(self.solver.fk(self.q).actInv(self.T_target)).vector
            self.ik_err = float(np.linalg.norm(err))
        return self.q, self.gripper
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 19 PASS

Note on `test_tick_tracks_while_gripped`: while engaged, the anchor passed to `Clutch.update` is recomputed but ignored (the clutch already latched `_T0`), so the target stays pinned to the press moment — the repeated ticks just let the step-capped solver converge.

- [ ] **Step 5: Commit**

```bash
git add vr/ik_teleop.py vr/test_ik_teleop.py
git commit -m "feat(vr): teleop tick - clutch + ik + trigger gripper"
```

---

### Task 5: main() — sockets, fixed-rate publish, status line, --dry-run

**Files:**
- Modify: `vr/ik_teleop.py` (append after `Teleop`)

- [ ] **Step 1: Implement main()**

Append to `vr/ik_teleop.py`:

```python
def parse_hostport(s, default_host="127.0.0.1"):
    host, _, p = s.rpartition(":")
    return (host or default_host, int(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="0.0.0.0:5557",
                    help="UDP host:port for Quest JSON in (default: 0.0.0.0:5557)")
    ap.add_argument("--target", default="127.0.0.1:8080",
                    help="follower UDP host:port (default: 127.0.0.1:8080)")
    ap.add_argument("--hand", default="right", choices=("right", "left"))
    ap.add_argument("--rate", type=float, default=100.0,
                    help="publish rate Hz (default: 100)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="hand-to-EE motion scale (default: 1.0)")
    ap.add_argument("--max-grip", type=float, default=0.07,
                    help="gripper width at trigger=0, meters (default: 0.07)")
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--dry-run", action="store_true",
                    help="print instead of sending UDP to the follower")
    args = ap.parse_args()

    teleop = Teleop(IkSolver(args.urdf), hand=args.hand, scale=args.scale,
                    max_grip=args.max_grip)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(parse_hostport(args.listen, default_host="0.0.0.0"))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = parse_hostport(args.target)

    print(f"Quest in: udp :{args.listen}  ->  follower out: udp {target[0]}:{target[1]}"
          f"{' (DRY RUN)' if args.dry_run else ''}\n"
          f"hand={args.hand} scale={args.scale} rate={args.rate:.0f}Hz\n"
          f"Publishing HOME until first grip; follower will move_j-align to it.",
          flush=True)

    period = 1.0 / args.rate
    seq = rx_count = 0
    last_report = time.monotonic()
    next_t = time.monotonic()
    while True:
        # Drain everything queued; act on the freshest sample only.
        msg = None
        while True:
            try:
                data = rx.recv(65536)
            except BlockingIOError:
                break
            rx_count += 1
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                pass  # malformed datagram: keep whatever we had

        try:
            q, grip_m = teleop.tick(msg)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"Bad sample, skipping: {e}", flush=True)
            q, grip_m = teleop.q, teleop.gripper

        pkt = WIRE.pack(time.time(), seq, *q, grip_m)
        if not args.dry_run:
            tx.sendto(pkt, target)
        seq += 1

        now = time.monotonic()
        if now - last_report >= 1.0:
            print(f"rx {rx_count / (now - last_report):.0f}Hz tx {args.rate:.0f}Hz "
                  f"clutch={'ON' if teleop.clutch.engaged else 'off'} "
                  f"ik_err={teleop.ik_err:.4f} grip={grip_m:.3f} "
                  f"q={[round(v, 3) for v in q]}", flush=True)
            rx_count = 0
            last_report = now

        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()  # fell behind; reset deadline


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite (no regressions)**

Run: `cd vr && uv run --with pytest --with pin --with numpy python -m pytest test_ik_teleop.py -v`
Expected: 19 PASS

- [ ] **Step 3: Smoke-test dry-run with a fake Quest sample**

Start the node: `cd vr && timeout 6 uv run ik_teleop.py --dry-run &`
Then inject two samples (engage grip, then move hand up 0.1 m):

```bash
sleep 2 && python3 - <<'EOF'
import json, socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send(pos, grip):
    s.sendto(json.dumps({"t": time.time(), "controllers": [
        {"hand": "right", "pos": pos, "quat": [0, 0, 0, 1],
         "trigger": 0.3, "grip": grip}]}).encode(), ("127.0.0.1", 5557))
send([0, 0, 0], 1.0)            # engage
for _ in range(100):
    send([0, 0.1, 0], 1.0)      # hold: hand 0.1m up
    time.sleep(0.02)
EOF
wait
```

Expected output: startup banner; status lines showing `clutch=ON`, `grip=0.049` (0.7 × 0.07), `ik_err` shrinking toward ~0, and `q` changed from HOME. No traceback.

- [ ] **Step 4: Commit and update the plan checkboxes**

```bash
git add vr/ik_teleop.py docs/superpowers/plans/2026-06-11-vr-ik-teleop.md
git commit -m "feat(vr): ik_teleop main loop - udp io, fixed-rate publish, dry-run"
```

---

### On-hardware checklist (manual, after the plan — not automated)

1. Rig: `python piperx_setup.py --follower --can can0` (arm resets into CAN mode).
2. Rig: `uv run ik_teleop.py` — follower should move_j-align to HOME.
3. Laptop: `uv run quest_server.py --udp <rig-ip>:5557`; Quest opens the page, Enter VR.
4. Keep the e-stop in reach. Squeeze grip gently and make a small slow motion first; verify direction mapping matches intuition (hand forward = arm forward). If a direction feels mirrored, fix `AXIS_MAP` (one place) and re-run tests.
5. Trigger → gripper close; release grip → arm freezes.
