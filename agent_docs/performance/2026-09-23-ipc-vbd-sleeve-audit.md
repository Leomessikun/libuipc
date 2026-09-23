# Full-body FMVP rollouts: IPC versus Newton VBD sleeve audit

The recorded upper-arm travel ratio is **not** sufficient to label a rollout
as a complete dressing demonstration. A valid endpoint must retain several
successive semantic sleeve rings around a nonzero span of the arm. This audit
uses Newton's own pure `multi_section_topology` and
`evaluate_sleeve_retention` functions on saved cloth meshes; no simulator
replay or Matplotlib is involved. The IPC recordings use the fingertip as the
available wrist-side landmark, while Newton uses its wrist landmark. Both
checks use the same seven `tshirt_26` sleeve rings, whose cuff indices match
the IPC mesh exactly.

| Saved rollout group | Scalar or original success | Multi-ring retained at endpoint | Material and start |
| --- | ---: | ---: | --- |
| Genesis+IPC, FMVP actor, body 14045/14049, two replicas each | 4/4 held upper-arm ratio >=0.7 | **0/4**; none retained at any recorded state | Full SMPL-X body, gravity-hung start, `cloth_strain_rate=10`, 0.896 kg shirt |
| Genesis+IPC, earlier body 14046 baseline and half-speed actor | 2/2 held upper-arm ratio >=0.7 | **0/2**; none retained at any recorded state | Full body, default material and no actor rotation |
| Newton+VBD, evolved heuristic, body 24 | 19/30 accepted by its axial-occupancy success gate | **0/19** saved accepted endpoints | Full-body baked state; all 30 attempts started arm-axis-threaded |

At the IPC upper-arm ratio crossing, all seven ring centers have projected
arm-chain coordinate `s=1` at the shoulder. At most the cuff and the next
ring wind around the arm; the sleeve is bunched or draped near the shoulder.
The native Genesis front and rear views agree with this failure mode. IPC's
99th-percentile cloth-edge length at the crossing is 2.0–2.6 times the first
recorded state; peak simulated gripper loads are 442–554 N. These loads are
not calibrated real-world force readings.

Newton's 19 accepted endpoints are less deformed by its own rest-edge metric
(`max_edge_strain` 0.258–0.316), and the recorded sampled body penetration is
zero. But all 19 have only the cuff and second ring wrapped, with retained
ring span at most 0.024 along the arm chain; the strict gate requires at least
three contiguous rings and span 0.15. Newton's existing success gate checks
axial sleeve-vertex occupancy, which can pass when the semantic rings bunch
near the shoulder. The Newton data therefore does not establish that VBD
produces complete sleeve trajectories. Its baked, prethreaded start and
heuristic controller also differ from the hanging-start FMVP actor, so the
success fractions are not a matched solver benchmark.

Reducing the IPC shirt density from 3333 to 750 kg/m³ (roughly 0.896 to
0.202 kg) at `cloth_strain_rate=10` reduced peak gripper load to 210–258 N,
but neither of two body rollouts reached the upper-arm threshold. Lower mass
alone did not solve the geometry or provide a calibrated jersey material.
Both did hold the forearm-ratio milestone. Through that earlier milestone,
their peak loads were only 23–26 N, versus 186–307 N for the heavy-shirt
prefixes; 99th-percentile edge-length ratios stayed around 1.30–1.32 in
both cases. The lighter-shirt clips therefore look like the better *partial*
motion source, pending material calibration. The [native Genesis forearm
view](../../output/uipc_manip/fmvp_ipc_rot_strain10_mass020_pilot_20260923/genesis_body14049_forearm_ring_centers.png)
still shows much of the garment hanging behind the body, so it is not a clean
full-sleeve demonstration.

The four heavy-shirt IPC episodes contain **384 actions** through the first
held forearm-ratio milestone. At that point the cuff and next ring encircle
the arm and the cuff has reached `s≈0.29`, so these are partial checkpoint
motion probes. They are labeled `partial_cuff_progress`, **not complete
expert demonstrations**. No strict full-sleeve checkpoint trajectory has
been collected in the audited pilot.
The lighter-shirt pair adds **two** separately labeled partial clips with
**193 actions**. Do not merge the two material conditions into one dataset.

The next useful change is to correct the dressing geometry and endpoint
criterion before collecting a large dataset or fine-tuning on purported
successes. Genesis+IPC is an existing FMVP bridge and a useful physical
validator, **not a proven sole source of full-sleeve demonstrations**: neither
its scalar crossings nor Newton's saved successes passed the same topology
gate. Newton+VBD needs a matched hanging start and FMVP observation/action
adapter before it can be compared as a rollout source. In every solver require
multi-ring retention, acceptable deformation, grasp integrity, and a held
endpoint. Calibrate fabric mass and force-extension response against the
intended garment before using simulated force as a safety threshold.

Reproduce the saved-state audit (Newton's Python environment has Torch):

```bash
PY=/home/ge47gax/kun/newton/.venv/bin/python
IPC=output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923
"$PY" scripts/wang_transfer/audit_sleeve_topology.py \
  "$IPC"/body_14045_seed_1004/baseline_rep{0,1}.npz \
  "$IPC"/body_14049_seed_1004/baseline_rep{0,1}.npz \
  --newton-raw /home/ge47gax/kun/newton/runs/topo_ab_fc35b699/evolved_gen7/tshirt_26@h24/raw_states.pt \
  --out "$IPC"/topology_audit.json

python3 scripts/wang_transfer/extract_checkpoint_prefixes.py \
  --sources "$IPC" --topology-audit "$IPC"/topology_audit.json \
  --out output/uipc_manip/fmvp_ipc_partial_audited_example \
  --hold 5 --max-gripper-peak-N 350 --max-edge-p99-ratio 1.4
```

Raw audit: `output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/topology_audit.json`.
Earlier 14046 audit: `output/uipc_manip/fmvp_full_body_14046_topology_audit_20260923.json`.
The [front](../../output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/genesis_body14045_upperarm.png),
[rear](../../output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/genesis_body14045_upperarm_back.png),
and [ring-center overlay](../../output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/genesis_body14045_upperarm_ring_centers.png)
are native Genesis captures; the interactive viewer is
`scripts/wang_transfer/view_rollouts_genesis.py`.

## Lighter-shirt three-controller collection and data gate

A follow-up Genesis+IPC collection used the full body, the released
`fmvp_sim.pt` actor with FMVP yaw, `cloth_strain_rate=10`, and density 750
kg/m³ (about 0.20 kg for this mesh). It compared the actor, an actor-to-scripted
handoff after forearm ratio 0.5, and the scripted controller on bodies 14046,
14047, 14048, and 14050. Of **12 attempted episodes**, two actor episodes
held the old scalar upper-arm threshold; **zero** held the multi-ring retention
gate. No candidate is a complete dressing demonstration. The handoff and
scripted variants did not improve this result; several runs exceeded the
1,000 N simulated-load collection cutoff. These numbers are a controller
comparison under one uncalibrated garment setting, not a population success
rate or a human-force safety claim.

In the actor's body-14046 scalar success, the gripper crossed the anatomical
shoulder projection around state 125 and stopped at 1.70 times the
fingertip-to-shoulder span. The cuff center ended at 1.14 times that span,
about 7 cm past the shoulder, while only the first two sleeve rings remained
wrapped. The [native Genesis frame](../../output/uipc_manip/fmvp_ipc_mass020_three_controllers_20260923/genesis_body14046_frame212.png)
shows the shirt draped across the shoulder toward the neck, leaving the
forearm exposed. The scalar upper-arm metric therefore rewards pulling the
opening beyond the target rather than retaining the sleeve on the arm.

The collection's `accepted.json` reflects the **old scalar rule** and should
not be fed to imitation training. The independent
`output/uipc_manip/fmvp_ipc_mass020_three_controllers_20260923/quality_manifest.json`
records all 12 attempts and has an empty `strict_episode_paths` list. Its
source audits are in each `body_*/topology_audit.json`. The next controller
should track cuff position and ring engagement, with a shoulder stop/turn
condition, before more data collection.

The local FMVP PyBullet environment instantiates a Sawyer URDF and maps
end-effector commands through IK, but disables robot–cloth collision. Its
required custom build is the `pybullet_3_0_9` branch of `Zackory/bullet3` at
commit `a3804ac`. That branch was built for the isolated `dressing` Python
3.7 environment, without replacing its installed stock PyBullet. The wheel
and SHA256 are under
`output/uipc_manip/fmvp_pb_custom_bullet_20260923/`. The first replay adapter
called FMVP's `trunk()` directly and applied one `tanh`; a direct parity
check against `Actor.forward()` found zero action difference on a saved
observation. The maintained runner now calls FMVP's `Actor.forward()` itself.
A briefly launched, *unsquashed* adapter run was interrupted
before trajectory export and marked invalid in its output directory. The
completed first static replay under the correct transform failed the ring
retention audit. Four matched static/moving-arm replays are now under
`output/uipc_manip/fmvp_pb_robot_traj_verified_20260923/`. All four held the
old upper-arm ratio threshold, but **none held multi-ring retention for five
decisions**. The static-arm run retained six contiguous rings over a 0.28–0.29
arm-span only at states 133–135; actor continuation lost that geometry. A
controlled replay holds its Cartesian command at state 133 to test whether
this moment can be salvaged. The aggregate audit is
`output/uipc_manip/fmvp_pb_robot_traj_verified_20260923/topology_audit.json`.
Sawyer joints are source metadata, not Franka joint targets. The first
completed moving-arm replay (motion 1) scored 0.990 upper arm and ended early
at state 149, yet **zero of its 150 states retained the sleeve**.
At the endpoint all seven ring centers projected to the shoulder (`s=1`),
with no span down the arm. A [native Genesis
replay](../../output/uipc_manip/fmvp_pb_robot_traj_verified_20260923/genesis_m1_final.png)
shows the cloth bunched at the shoulder and the forearm exposed. That view
draws only arm-landmark proxy cylinders because this PyBullet recording did
not save the human's full collision mesh. Its saved Cartesian path passed a
provisional *position-only* Panda IK and speed screen, so robot reachability
did not fix the false sleeve outcome.

Because the real embodiment is a Franka, every *accepted* Cartesian gripper
path also needs a robot gate. The optional
`scripts/wang_transfer/check_franka_reachability.py` loads a Franka URDF,
requires an explicit base pose, and checks IK error, joint limits, and joint
speed at the saved 20 ms decision rate. For a real acceptance gate use the
exact Panda versus FR3 and gripper description, measured base-to-person
transform, and calibrated grasp-frame orientation. A position-only run using
the local Panda URDF and the source Sawyer base as a provisional pose screened
the failed-sleeve static PyBullet path with zero IK/limit/speed violations, but
cannot certify a useful trajectory or robot/person clearance. The final
transfer gate must replay a physically modeled Franka gripper together with
the full body and garment, including gripper–cloth and robot–person collision;
the FMVP PyBullet environment itself omits robot–cloth collision. The current
IPC dressing scene uses a virtual held-vertex gripper, so it has **no robot
URDF to swap**. Adding Franka there means driving the exact robot geometry
and joints alongside the cloth, not merely changing a file path. The separate
`uipc_manip.genesis_env` cloth-manipulation scene already uses a Panda MJCF and
IK, which is a starting point for that integration, not a dressing result.
For a diffusion/flow policy that predicts Cartesian grasp-frame action chunks,
the proposal dataset can retain gripper poses without Franka joint commands,
then reject trajectories that fail Franka reachability and full-scene replay.
If the policy instead predicts joint commands, the source Sawyer `robot_q`
must never be used as labels; only joint trajectories generated and validated
with the actual Franka model are appropriate.

## Dataset scale for expressive-policy post-training

For an initial diffusion or flow-matching **imitation** baseline, use 100
strictly valid episodes as a feasibility pilot and target roughly 500 diverse
training episodes plus 100 held-out evaluation episodes. Study data scaling
at 50, 100, 250, and 500 training episodes. This is a planning target rather
than a universal sample requirement: the [Diffusion Policy
study](https://journals.sagepub.com/doi/full/10.1177/02783649241273668)
used 136 Push-T, 162 bimanual mat-unrolling, 210 bimanual egg-beater, and 284
bimanual shirt-folding demonstrations. Dressing has different contact and
safety demands. Vary body geometry, garment/start state, and successful
control style; correlated replicas or forearm-only prefixes do not count as
distinct full-sleeve demonstrations.

The intended later contribution is **online post-training of the expressive
policy**, not merely fitting a flow model to FMVP actions. Following the
[EXPO recipe](https://arxiv.org/abs/2507.07986), a flow/diffusion base
proposes action chunks; a small bounded policy edits them; a long-horizon
value estimate selects candidates; validated online outcomes train the edit
and are distilled into the base. IPC one-step lookahead or gradients could
provide a physics-aware local editing signal, but cannot replace credit
assignment for held sleeve retention. The [post-training
article](https://pd-perry.github.io/posts/post-training.html) also emphasizes
the success test, resets, and intervention protocol. Here those are not
administrative details: the old scalar reward has already produced a concrete
false success. Failed episodes remain useful for value learning and diagnosis,
but are not positive imitation data.

## Checkpoint rollout decision, 2026-09-24

The answer to whether `fmvp_sim.pt` can **currently supply enough complete
single-sleeve trajectories** is **no**. In the lighter, full-body Genesis+IPC
pilot, the independent topology/force/deformation screen extracted seven
checkpoint-only **forearm** clips (731 actions, four body IDs) at
`output/uipc_manip/fmvp_ipc_posttraining_seed_20260924/manifest.json`.
These are stage-specific motion seeds, not full-sleeve positives and not a
diffusion/flow post-training dataset. The 12 complete attempts in that run
produced zero retained upper-sleeve successes. An additional 10-start geometry
preflight was legal, but the expanded rollout sweep was stopped after two
completed cases because the same failure repeated: body 14051 never reached
the upper arm and lost grasp; body 14052 crossed the old scalar threshold,
lost grasp, and had zero retained-sleeve states. The remaining eight starts
were **not** completed and must not enter the denominator.

The original FMVP PyBullet pilot is not a clean fallback. In a deterministic
replay of its best transient geometry, stopping the Cartesian command at
state 133 retained the sleeve for only states 133–134, short of the five-step
hold. Its `tshirt_26` cloth already had a 99th-percentile edge length of
3.12 times the scaled OBJ rest mesh at the **first recorded state**; final
rest-edge p99 values were 3.04–3.57 times rest over four matched replays.
See `output/uipc_manip/fmvp_pb_robot_traj_verified_20260923/deformation_audit.json`.
The source FleX `pyflex` module could be imported only with old CUDA and math
compatibility libraries, but `pyflex.init` aborted with stack-smashing on this
workstation; there is no verified FleX rollout source here.

A native [Genesis view at state 130](../../output/uipc_manip/fmvp_ipc_mass020_three_controllers_20260923/genesis_body14046_state130.png)
shows genuine partial elbow/upper-arm insertion before the garment body moves
away from the torso. The Newton ring diagnostic has one discontinuity there:
one ring moved only about 2 mm from states 126 to 127, yet its nearest arm-axis
choice switched from forearm to upper arm and its measured winding flipped.
An elbow-aware audit checks both nearby axes and requires the cuff to reach
`s>=0.7` with three consecutive rings wrapping a span of at least 0.15.
It still found no held upper-sleeve state. In the strongest original IPC run,
the cuff got only to `s=0.523` while three rings remained wrapped. This is a
real shoulder-phase deficit, not just the elbow-frame bookkeeping issue.

A precise stop at state 126 kept grasp and low simulated loads but left the
cuff near the elbow. A matched checkpoint-speed sweep on body 14046 yielded
one old scalar crossing for normal speed; half and quarter speed did not cross
within 260 decisions. None held elbow-aware upper-sleeve retention. The data
are at `output/uipc_manip/fmvp_ipc_targeted_hold126_20260924/` and
`output/uipc_manip/fmvp_ipc_targeted_speed085_20260924/`. Replayed prefixes
are numerically variable across IPC worlds, so a fixed decision number is not
a reliable event trigger. More bulk rollouts of this controller are unlikely
to create complete positive examples. The next experiment must change the
shoulder-phase control or grasp mechanics and verify at least one complete,
held, low-deformation sleeve in the native Genesis view before scaling data
collection or starting post-training.
