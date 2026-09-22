# The released policy dresses in PyBullet; in ours the frame was wrong and the opening starts too high

Date: 2026-09-23. Revises [2026-09-22-wang-checkpoint-transfer.md](2026-09-22-wang-checkpoint-transfer.md),
whose conclusion ("reads the scene and still does not dress; closed as a route") rested on a
frame that was 63 degrees off and on a placement nobody had compared against a running reference.

## Which checkpoint this is

The three files on the owner's desktop are FMVP's (Hao, Wang et al., CoRL 2025), not Wang RSS 2023's:

| file | keys | step | what it is |
|---|---:|---:|---|
| `vision_based_policy.pt` | 44 | 750,137 | FMVP's vision policy pretrained in FleX (SAC, `log_alpha` optimizer present) |
| `fmvp_sim.pt` | 52 | 20,000 | its IQL fine-tune in PyBullet; best 0.99791 is the name `eval_ckpts.py` loads |
| `fmvp_real.pt` | 52 | 1,200 | fine-tuned with real data |

The eight extra tensors are FiLM layers on a 3-vector, the summed cloth-on-human contact force in
the model frame (`dressing.py:96-102`). The FMVP paper says the vision policy was "pre-trained in
NVIDIA FleX"; its architecture is Wang's, which is why the bridge's strict load succeeded.

## A running reference exists after all

FMVP's released code (`/home/ge47gax/kun/fmvp_pb`) is a sim-to-sim transfer of exactly this
checkpoint into PyBullet. It needs Zackory's `bullet3@pybullet_3_0_9` fork (the stock wheel rejects
`linkLowerLimits`); it builds with
`pip install --no-deps --target <dir> "git+https://github.com/Zackory/bullet3.git@pybullet_3_0_9#egg=pybullet"`
under the `dressing` conda env and runs from `fmvp_pb/dressing_pb` with that dir first on
`PYTHONPATH`. Its deployment code is the interface specification the previous record said was
missing:

* observation: PyBullet z-up points permuted `[:, [1, 2, 0]]` (model y is up), camera-visible arm and
  garment voxelised at 6.25 cm, gripper-centred, gripper last;
* action: model `(x, y, z) * 0.025` m sent as PyBullet `(z, x, y)`; rotation keeps only the model-y
  (vertical) component, clamped to a few degrees per step; x and z rotations are dropped.

**`vision_based_policy.pt` dresses zero-shot there**: garment `tshirt_26`, static arm, 223
decisions, upper-arm ratio **0.991**, whole arm 0.996. So nothing about particle-versus-mesh
cloth prevents this network from dressing a mesh simulator.

## The frame was 63 degrees off

PyBullet's first observation fixes the model frame: gripper-relative fingertip (0.033, -0.106,
0.003), forearm direction +x, upper arm (0, 0.21, 0.98). Mapping our cell (tshirt_26 on body 14045)
through `up_axis_rotation(yaw)` puts the forearm on +x at yaw 267, and the upper arm then lands on
(-0.15, 0.39, 0.99): both bones agree. The previous yaw, 330, maximised alignment with the
gripper-to-shoulder chord, but the policy heads along the forearm first. The handedness check (the
vertical component of forearm x upper arm) is negative in both, so there is no mirror.

At yaw 267 the two simulators' gripper paths agree for the first 20 decisions, in model
coordinates relative to the fingertip:

| decision | PyBullet | ours |
|---:|---|---|
| 0 | (-0.03, 0.11, 0.00) | (-0.03, 0.11, -0.01) |
| 20 | (0.13, 0.17, 0.03) | (0.15, 0.17, 0.05) |
| 30 | (0.19, 0.18, 0.04) | (0.22, 0.20, 0.08) |

The policy is doing the same thing in both. The garment is not.

## Where they part: the opening starts at the hand, not below it

| t = 0, model frame | PyBullet | ours (socket placement) |
|---|---|---|
| fingertip relative to gripper | 10.6 cm below | 11.3 cm below |
| opening centre relative to fingertip | 13.5 cm below, 8.5 cm out | level, 16.6 cm out |
| opening relative to grasp | straight below, 19 cm | 40 degrees below horizontal, 18.5 cm |

The grasp (Wang's 121 `grasping_particle_indices`) and the opening polygon are the same vertices of
the same 3,889-vertex mesh in both; PyBullet lets the garment hang from the grasp, our socket
placement holds the opening coaxial with the forearm. The policy lifts about 11 cm in both. In
PyBullet that brings the opening from 13.5 cm below the fingertip to 2.4 cm below it at t = 30,
the hand enters and the lift command falls to zero. In ours the same lift carries the opening to
14 cm *above* the fingertip, it passes over the hand, and the policy keeps lifting and stalls.

Three tests, rotation off throughout (`act[3:] = 0`):

* **Action interface does not matter.** Rotation off, FMVP's vertical-only rotation, softgym's
  `to_yz` in the model frame, one or three env steps per decision: all stall at 0.6 to 0.7 of the
  fingertip-to-shoulder chord with forearm 0.00.
* **Clamping the lift threads the sleeve.** Model-frame y action clamped to at most 0: threaded 1
  by decision 50, forearm 0.52 (0.49 with the garment cloud cropped at 0.38 m), upper arm 0.
* **Lowering the opening threads it without any clamp.** Swinging the placed garment 15 degrees
  about the picker toward straight down, 9 cm further out along the forearm and 6 cm lower (the
  closest legal start: larger swings put the garment's body through the arm):

| body, seed | threaded | forearm (best) | upper arm | expert upper arm |
|---|---:|---:|---:|---:|
| 14045, 1000 | 1 | 0.415 | 0 | 0.985 (default placement) |
| 14045, 2000 | 1 | 0.433 | 0 | – |
| 14048, 1000 | 1 | 0.61 | 0 | 0.196 |
| 14046, 1000 | 0 | 0 | 0 | 0.958 |

Three of four thread. Before this, the policy's forearm ratio in our simulator had never left zero.

## What is still open

No run reaches the upper arm. After threading, the gripper hovers about 0.22 to 0.27 m past the
fingertip and 0.2 m above it; in PyBullet it hovered in the same place for 70 decisions while the
whole-arm ratio crept from 0.10 to 0.24, then turned toward the shoulder (+z) and finished. Two
differences that could explain the missing turn were not separated:

* **Garment compliance.** PyBullet's cloth is a mass-spring at `springElasticStiffness` 0.5 with
  41 anchors; ours is strain-limited IPC holding 48 vertices. The creep that preceded the turn in
  PyBullet may not happen in ours.
* **Arm proportion.** PyBullet's "finger" is the wrist less a hand radius and its forearm 28.4 cm;
  ours is the fingertip and 38 cm on body 14045. The policy turned near PyBullet's elbow, which in
  ours is 60 to 70 per cent of the way along the forearm.

`fmvp_sim.pt`, the PyBullet-adapted weights, was not tried: the bridge's encoder has no FiLM path.
FMVP needed 20,000 IQL steps to adapt this policy to PyBullet; a comparable fine-tune in our
simulator, from these weights, is the route their own work took.

## Reproduction

Scripts are under [scripts/wang_transfer/](../../scripts/wang_transfer/); they import the bridge
from the `residual-rl` worktree (branch `sac-stability`, `python/uipc_manip/wang_bridge.py`).

* `fmvp_pb_run.py`: the checkpoint in FMVP's PyBullet env with FMVP's action transform; logs the
  gripper, fingertip, opening and grasp in the model frame.
* `wang_trace.py`: our env, policy beside the expert, per-decision model-frame trace.
* `wang_tilt.py`: the swung placement (`--tilt 15 --out-shift 0.09 --drop 0.06`), with a legality
  check against `garment_arm_gap`.
