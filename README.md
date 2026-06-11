Technical Task: Cross-Rig Teleoperation and VR Integration
The goal of this task is to establish a robust, low-latency cross-rig teleoperation link and integrate Virtual Reality (VR) for advanced control and immersive visualization using two AgileX PiPER 6-DOF arms.
Setup and Environment
Rigs: Two separate PC nodes, each connected to an AgileX PiPER 6-DOF arm
Networking: Rigs are on the same local network (LAN)
SDK: Access to the piper sdk monorepo is provided (feel free to modify or build as needed)
VR/Haptics: Meta Quest 3 Headset
SSH Details:
Right Rig: pantheon@10.103.0.45, pwd pantheon
Left Rig: pantheon@10.103.0.74, pwd ubuntu123

Download piper_sdk here if not already on rig: https://github.com/agilexrobotics/piper_sdk
Background
Previous proof of concepts for remote teleoperation demonstrated that while joint data streaming is possible, the system suffers from significant latency issues (approx. 30Hz), leading to inconsistent or jerky arm movements. Current rig architecture uses localized state storage which can lead to control overrides; this exercise explores the necessary steps toward a more robust, low-latency coordination layer and an enhanced operator experience.
The Task: 5 Phases
Phase 1: Single-Rig Baseline (Local Teleoperation).
Establish a standard leader-follower connection on a single PC to verify piper sdk integration and local hardware state.
Phase 2: Cross-Rig Implementation (Joint Streaming).
Configure one PC as the leader and the second PC as the follower.
Implement a networking streaming pipeline (e.g., using UDP or a similar low-latency protocol) to reliably relay joint positions across the network.
Phase 3: Latency Optimization (Low-Latency Control Loop).
Optimize the data frequency and transmission logic (e.g., using a push-based system or delta encoding) to achieve a smoother control loop, targeting a sensible latency for production readiness.
Phase 4: VR End Effector Control (Meta Quest 3).
Once rig-to-rig communication is stable, integrate the Meta Quest 3.
Implement a control loop that maps the VR controller's pose (position and orientation) to the leader arm's end effector commands. These commands must be relayed via the cross-rig link to control the follower arm's end effector pose (Cartesian control).
Phase 5: Low-Latency VR Camera Stream (Immersive Visualization).
Stream the follower rig's camera feed (e.g., wrist-cam or exo-cam) into the Meta Quest 3 headset.
The primary goal is minimizing end-to-end latency to provide the operator with a real-time, immersive view crucial for dexterous manipulation tasks.

Inspiration for VR portion: https://x.com/FrostierFridge/status/2044122345217085797?s=20
Discussion Points
How would you scale this coordination layer from 1 to N robots while maintaining strong consistency?
What are the reliability constraints when moving from a LAN-based intervention to a fully remote, push-based system over a global service?
How should the system handle hardware state resets or "flushing" when the leader arm switches control between multiple followers?
VR Integration: How does the drift or jitter in Meta Quest 3 tracking affect the precision required for long-duration teleoperation tasks?
Latency Budget: What is the acceptable latency budget for an immersive VR camera stream compared to the control loop, and how would you prioritize bandwidth allocation?


