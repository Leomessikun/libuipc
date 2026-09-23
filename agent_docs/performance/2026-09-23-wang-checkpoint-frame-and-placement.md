# The released policy dresses in PyBullet; in ours the frame was wrong and the opening starts too high

> **Update, same day: `fmvp_sim.pt` dresses in our simulator from a gravity-hung start** (upper arm
> at least 0.7 on 15 of 18 runs; see the last two sections). The earlier sections are the path there.

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
placement holds the opening coaxial with the forearm. In PyBullet the gripper rises 7 cm over
the first 30 decisions, which brings the opening from 13.5 cm below the fingertip to 2.4 cm below
it; the hand enters and the lift command falls to zero. In ours the same early commands carry the
opening to 8.5 cm above the fingertip by decision 30, it passes over the hand, the hand never
enters, and the policy keeps lifting: 13 cm by decision 40, the opening 14 cm above the fingertip,
then it stalls. The asymmetry is the finding: the lift stops when the hand is inside the opening,
and in ours it never is.

Three tests, rotation off throughout (`act[3:] = 0`):

* **Action interface does not matter.** Rotation off, FMVP's vertical-only rotation, softgym's
  `to_yz` in the model frame, one or three env steps per decision: all stall at 0.6 to 0.7 of the
  fingertip-to-shoulder chord with forearm 0.00.
* **Clamping the lift threads the sleeve.** Model-frame y action clamped to at most 0: threaded 1
  by decision 50, forearm 0.52 (0.49 with the garment cloud cropped at 0.38 m), upper arm 0.
* **Lowering the opening threads it without any clamp.** Swinging the placed garment 15 degrees
  about the picker toward straight down, 9 cm further out along the forearm and 6 cm lower (the
  closest legal start: larger swings put the garment's body through the arm):

| body, seed | threaded | forearm (best) | upper arm | expert upper arm, same start |
|---|---:|---:|---:|---:|
| 14045, 1000 | 1 | 0.415 | 0 | not run (0.985 from the default placement) |
| 14045, 2000 | 1 | 0.433 | 0 | not run |
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

The swung start is also not PyBullet's start, so three of four threading is a proxy for a
matched start. The opening ends 20 to 26 cm outside the fingertip against PyBullet's 8.5, and the
fixed 6 cm drop leaves the gripper only 2.3 to 4 cm above the fingertip where PyBullet's is 10.6;
on 14046, the body that failed, it is 2.3 cm. The principled test is the one both references use:
pin the grasp patch at PyBullet's offset from the fingertip (3.3 cm out, 10.6 cm up), let the
garment hang and settle under gravity with the arm present, compare the opening with PyBullet's
(8.5 cm out, 13.5 cm down), then run the policy. Our socket placement imposes an orientation that
the 48-vertex hold keeps through the settle; `gravity_aligned_socket` only re-rolls it.

The camera is untested as well. `wang_static_arm` uses the port's camera, about 19 degrees below
horizontal, against Wang's 35, and cropping the garment cloud at 0.30 m sent the gripper straight
along the arm axis, so the low-hanging garment points steer the policy strongly.

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

## Update: the gravity-hung start, and `fmvp_sim.pt` dresses

**The start.** The garment is hung from Wang's two picker vertices (968, 2896) in libuipc, far from
the arm, with zero actions; it is still after 100 decisions (`hang_bake.py`). Hanging freely, the
opening sits 21.6 cm from the picker and 55 degrees below horizontal, against the socket
placement's 18.5 cm and 45 degrees and PyBullet's 24.6 cm and 78 degrees; the rest is stretch our
strain-limited cloth does not have. That shape is placed with the picker at PyBullet's offset from
the fingertip (3.3 cm out, 10.6 cm up, model frame) and turned about the vertical to the legal
turn nearest PyBullet's opening, then held by the usual 48 vertices (`wang_hang_run.py`). The
opening starts 7 cm below the fingertip and 16 cm out on every body tried; rigid re-posing could
not get there, because every turn that points the opening straight down puts the garment's body
through the hand.

**The bridge** now loads FMVP's FiLM fine-tunes (commit fa6c0949 on `sac-stability`) and takes an
optional force. The force fed to FiLM is minus the summed IPC contact force on the arm (the force
on the garment, as FMVP sums it), rotated into the model frame, times a scale.

**Results**, tshirt_26, yaw 267, rotation off, success = upper-arm ratio at least 0.7 (our env's
`success_upperarm_ratio`), two identical `fmvp_sim` slots per world where listed:

| body | vision | fmvp_sim, force 0 | fmvp_sim, force x0.003 / 0.01 / 0.03 | expert, same start |
|---|---|---|---|---|
| 14045 | 0 of 2 | 3 of 3 | 0 / 0 / 0 | 2 of 2 |
| 14046 | 0 of 1 | 2 of 2 | 1 / - / 0 | 1 of 1 |
| 14048 | 0 of 2 | 3 of 3 | 0 / 0 / 1 | 2 of 2 |
| 14047 | - | 1 of 2 | - | 1 of 1 |
| 14049 | - | 2 of 2 | - | 1 of 1 |
| 14010 (training pose) | - | 2 of 2 | - | 1 of 1 |
| 5045 (region 4) | - | 2 of 2 | - | 1 of 1 |
| 23045 (region 22) | - | 0 of 2 | - | 0 of 1 (peak 0.16) |

`fmvp_sim` with zero force: **15 of 18**, crossing 0.7 at decisions 182 to 262, when the expert
crosses at 147 to 238; on the seven bodies the expert can dress, 15 of 16. The two identical slots
of a world do not follow identical trajectories (14045: peaks 0.979 and 0.999; 14047: 0.635 and
0.983), so single runs are samples, not measurements. The FleX vision policy threads the sleeve on
this start but never reaches the upper arm (forearm 0.59 to 0.71).

**What is wrong with these successes.**

* *It pulls too far and too hard.* At the crossing the gripper is 1.5 to 2.1 fingertip-to-shoulder
  chords from the fingertip, where the expert is at 1.15 to 1.45, and the contact force on the arm
  is 37 to 837 N, where the expert's is 1 to 233 N and mostly under 35. A policy trained on cloth
  that stretches learned to over-travel; ours transmits it as force.
* *It does not stop.* PyBullet ends the episode at 0.99. Held still after crossing 0.7, 5 of 7
  successes on the second body set did not stay: the taut sleeve snaps past the shoulder in one or
  two decisions (5045: 0.33, 0.81, 0.995, then 0), after which the progress metric reads zero. The
  expert's held ratio stays where it was.

**Force as the FiLM input hurt**: 2 of 10 with any non-zero scale against 8 of 8 on the same
bodies at zero. FMVP's force is PyBullet's soft-body contact force times 10; over the successful
PyBullet episode it is zero on 73 per cent of decisions and at most 0.077 in norm. It is not in
newtons in any calibrated sense, and our readings run from tens to hundreds of newtons, so no
single scale maps one distribution onto the other; per-decision IPC forces also reproduce poorly
(2026-09-12 contact-force records). The IPC force is useful here as a measurement: it is what
exposes the over-pull above.

**Next.** Stopping on success is not enough while the snap happens within a decision of the
crossing. The route FMVP itself took is to fine-tune these weights in the target simulator; ours
would add the arm force as a cost, which is what the 37 to 837 N says it needs. A cheaper probe
first: slow the policy near the shoulder (scale the translation once the forearm ratio is 1) and
see whether the held ratio survives.

## The gripper load is the force channel to use, and it is reliable

FMVP's real system conditions on the **end-effector force** of its Sawyer (from joint torques,
smoothed by an exponential moving average), not on force on the person; FleX gave no usable force,
so force entered only in the real-world IQL fine-tune (reward: a learned preference model plus a
force penalty normalised by 8 N; safety stop at 18 N). The PyBullet release substitutes summed cloth
contact force.

Our analogue is the hold itself: each held vertex is a spring of stiffness
`constraint_strength * m_i / dt^2`, so `F = -sum k_i (target_i - x_i)` is the force the garment puts on
the gripper, what a wrist sensor would read (`scripts/wang_transfer/wang_force_probe.py`; vertex mass
from `2 * cloth_thickness * cloth_density` per area, since `thickness` is a half-thickness). At rest it
reads 9 to 10 N against the garment's weight of 8.8 N (0.896 kg).

Same commands in two slots of one world, 260 decisions, gravity-hung start:

| run | gripper load within 20 % | arm contact within 20 % | gripper p90 / max (N) |
|---|---:|---:|---|
| fmvp_sim, 14046 | 0.98 | 0.85 | 206 / 474 |
| fmvp_sim, 14045 | 1.00 | 0.85 | 1218 / 1987 |
| expert, 14046 | 0.78 | 0.71 | 335 / 878 |
| expert, 14045 | 0.72 | 0.59 | 387 / 648 |

The gripper load reproduces per decision where the arm contact force does not, and it is the
quantity a real robot measures. Its magnitude is the problem: the expert itself runs at a p90 of
335 to 387 N against FMVP's 18 N stop. The garment weighs 0.9 kg and the IPC cloth barely stretches,
so before any absolute force threshold is meaningful the garment's mass and stretch must be checked
against a real T-shirt; until then a force term can only be relative (for example, against the
expert's load at the same progress).

## With the complete body: every controller jams at the shoulder, unless the cloth may stretch

The arm-only successes above do not survive a torso. The Codex session's full-body collider
(complete SMPL-X surface, `collision_geometry="full_body"`, uncommitted in both checkouts at the time;
record `2026-09-23-fmvp-full-body-rollout.md`) turns them into forearm prefixes: the arm-only wins
over-pulled 1.5 to 2.1 chords, through where the chest is. The runs below use its collector with
three added options (rotation handling, `cloth_strain_rate`, `cloth_youngs`), gravity-hung start,
5 mm up offset, success at upper arm 0.7 held for 5 decisions, 1,000 N gripper-load abort.

**Rotation.** The PyBullet success turns the gripper about the vertical by up to 119 degrees and back
(t = 100 to 222, mostly in the upper-arm phase); every IPC run so far had dropped rotation. Restoring
FMVP's vertical-only, clamped rotation: 2 of 9 against 1 of 9 without (14046 at 90 N against 301 N;
14047 rescued; 14045, 14048, 14049, 14050 still fail). It helps, it is not the cause.

**The scene is jammed, not the policy.** The scripted expert with the full body: 0 of 6 (peak upper
arm 0.08 to 0.57). Without the abort, expert and policy both plateau at upper arm 0.5 to 0.7 while
the gripper load climbs to 2 to 8.7 kN and tears the held patch away.

**The strain limit decides it.** Same bodies that jammed (14045, 14049), rotation on:

| `cloth_strain_rate` | fmvp_sim 14045 | fmvp_sim 14049 | expert 14045 | expert 14049 |
|---|---|---|---|---|
| 100 (default) | fail x2 | fail x2 | fail | fail |
| 10 | 0.79, peak 451 N | 0.70, peak 621 N | fail | fail |
| 1 | 0.74, peak 510 N | 0.73, peak 550 N | 0.72 | fail |

At 10 the garment's median edge length stays at 1.00 to 1.06 of the start, as at 100; at 1 it
shrinks to 0.85 to 0.89, which is not physical, so 1 is rejected. All three settings end with
p99 edge stretch 2.2 to 2.6 and single edges at 3 to 12 times, near the hold; that local distortion
is its own open problem.

Reading: the checkpoint was trained on cloth that stretches (FleX, then PyBullet's mass-spring), and
a knit T-shirt stretches too; the default strain limit makes the armhole unable to pass the shoulder
with a torso present, for the expert as much as for the policy. One run per cell; before `10` becomes
a default it needs a force-extension calibration against real jersey and replication over more bodies.
