# VR IK Teleop — Design

2026-06-11

## Goal

Drive the PiPER arm's end-effector from a Meta Quest controller: 6-DOF
(position + orientation) delta-clutched teleop, active only while the **grip**
button is held, with the analog **trigger** controlling the gripper. Single arm
for now (right controller → right arm); structured so a second arm is a CLI
flag later.

## Architecture

```
Quest browser (WebXR)
   │ WSS (poses, JSON)
quest_server.py            ← unchanged; run with --udp <rig-ip>:5557
   │ UDP JSON :5557
vr/ik_teleop.py  (NEW)     ← runs on the rig, Pinocchio IK
   │ UDP 40-byte WIRE datagrams (<dI6ff: t, seq, j1..j6 rad, gripper m)
piperx_setup.py --follower ← unchanged; listens :8080, move_j align then
   │ CAN                      high-follow move_js
PiPER arm
```

The IK node impersonates a `piperx_setup.py --leader`: same wire format, same
port. No changes to either existing script.

## Components (all in `vr/ik_teleop.py`, PEP-723 script: `pin`, `numpy`)

**Model.** Pinocchio loads
`piper_ros/src/piper_description/urdf/piper_description.urdf`, gripper joints
locked via `buildReducedModel` (6 revolute joints remain). End-effector frame:
the gripper base link frame.

**Clutch (per hand).** Grip analog with hysteresis: engage > 0.8, release
< 0.5. On engage, latch controller pose (p₀, R₀) and current EE target pose
(P₀, Q₀) from FK of the current solution. While held:

- target position  P = P₀ + s · A(p − p₀)   (s = scale, default 1.0)
- target rotation  Q = (A R R₀⁻¹ A⁻¹) Q₀     (relative rotation, world axes)

where A is the fixed WebXR→robot axis map applied to *deltas only* (so headset
calibration/facing direction never matters): robot x = −z_xr, y = −x_xr,
z = y_xr; adjustable constant in one place.

On release the target freezes; the node keeps publishing it (follower holds).

**IK.** Per incoming sample: damped least squares on
`log6(T_current⁻¹ · T_target)`, ≤ ~10 iterations, warm-started from the last
solution. Joint limits clamped each iteration. Per-sample joint step capped
(safety: bounds EE speed even if the target jumps). If the error doesn't
converge (unreachable target), publish the best-effort clamped solution — the
clutch math keeps the target continuous so this degrades gracefully at the
workspace boundary.

**Gripper.** width = (1 − trigger) · max_width (pulled = closed), default
max_width 0.07 m, sent in every datagram.

**Startup.** Node starts with q = nominal home config and publishes it
immediately at a fixed rate (~100 Hz, decoupled from input). The follower's
existing align-with-move_j logic walks the arm there safely; high-follow takes
over after.

**Output.** UDP socket → `--target 127.0.0.1:8080` (default). Monotonic `seq`,
`t = time.time()`. Continuous publish at fixed rate, independent of Quest
sample arrival (hold-last semantics, keeps follower stream alive).

## Error handling

- Malformed/missing UDP JSON: skip sample, keep publishing last solution.
- Quest stream stops: keep publishing last solution (arm holds still).
- IK divergence: step cap + limit clamps bound the damage; warn-log once/sec.
- Status line once per second (rx Hz, tx Hz, clutch state, IK error, q).

## Testing

- Unit tests (`vr/test_ik_teleop.py`): FK→IK round-trip on random reachable
  poses (solve to < 1 mm / < 0.01 rad); clutch latch/release math; axis-map
  sanity; wire-format pack matches `piperx_setup.WIRE`.
- `--dry-run` flag: full pipeline, prints solutions, no UDP send — for desk
  testing with a live Quest before touching the arm.
