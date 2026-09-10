# uipc_manip — robot cloth and cable manipulation pretraining on the IPC solver

`uipc_manip` trains a point-cloud Soft Actor-Critic policy to manipulate a
deformable object with a Franka Panda, where every deformable contact is
resolved by libuipc's Incremental Potential Contact solver. The scene, robot,
and rendering come from Genesis; the cloth or cable is a native libuipc
geometry inserted through the Genesis IPC coupler.

Environments are batched the way the Newton teacher batches its simulation:
one Genesis scene holds `N` robots and `N` deformable copies, and one libuipc
world solves all of them together with contact isolated per environment by
IPC subscenes. Thirty-two copies step about ten times as many environment
steps per second as one, which is what makes a real transition budget
reachable on this solver.

The algorithm, hyperparameters, and transition semantics are ported from the
Newton cloth-dressing teacher on the `leomessikun/fmvp-sac-retrain` branch,
which follows Wang RSS 2023 as used for FMVP simulation pretraining. The task
suite is deliberately smaller: three goal-reaching tasks rather than a
multi-garment dressing curriculum.

## The dressing task

`--task dressing` is the Newton cloth-dressing teacher's environment rebuilt
on the IPC solver: thread the sleeve opening of a pre-worn garment along a
human's right arm to the shoulder. Every slot holds one pre-worn
(garment, human) cell from the Newton bake cache; the human's right-arm
collision mesh is a fixed libuipc affine body, the garment a strain-limiting
Baraff-Witkin shell in its own IPC subscene, and twelve cuff vertices near the
picker follow the 6-D gripper action (translation and rotation) through a
soft position constraint. All slots share one libuipc world.

The MDP is the Wang RSS 2023 `pointcloud_3` preset the Newton
`--fmvp-pretrain-defaults` launcher reproduces: 900 decisions at 60 Hz, a
0.15 m/s end-effector speed cap split per axis, 5 degrees of rotation per
step with the x-rotation zeroed, a 12 mm no-move collision shell around the
arm, the line-triangle progress reward with the upper arm worth five times
the forearm, a dual-camera visible point cloud with voxel downsampling,
camera jitter and dropout, an explicit tool point, and a 768-point budget.
Success is an upper-arm dressed ratio of at least 0.7 at the time limit.

The cache is not in this repository. It is read from
`UIPC_MANIP_DRESSING_CACHE` (default: the `hand_cached_states.pkl` of the
`ppf-contact-solver` bake on this machine) and the garment semantics from
`UIPC_MANIP_DRESSING_SEMANTICS` (default: the Newton branch's
`canonical_drape` directory). Seventeen of the 23 cached cells build under
libuipc's intersection and distance checks; the six that do not carry
residual self-intersections from the bake. A world holds one human, so a run
is a regional teacher in the Newton sense; garments cycle across slots.

```bash
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task dressing --cell-source cache --human 0 \
    --garments tshirt_26 tshirt_392 --policy heuristic --eval-only --num-envs 4 --num-eval-episodes 4
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task dressing --cell-source cache --human 0 \
    --garments tshirt_26 tshirt_392 --num-envs 16 --encoder transformer \
    --total-transitions 100000 --eval-freq 1800 --num-eval-episodes 16 --checkpoint-interval 900
```

Where the IPC port departs from the Newton teacher, and why:

* The cuff is held by a soft position constraint on twelve vertices rather
  than a hard kinematic pin; the picker patch is the same. libuipc's strength
  is a rate multiplied by the vertex mass, so the default of 1e4 is what makes
  the hold behave like a pin: at the library default of 100 the cuff lags the
  tool by centimetres in free air and, once the sleeve touches the hand,
  stays behind while the tool moves on, ending more than a metre away.
* The arm collider is the cached `right_arm_faces` mesh eroded 6 mm along its
  normals. The cache was accepted with centimetre-scale interpenetration,
  which libuipc refuses; erosion is how the states become legal.
* The garment is a strain-limiting Baraff-Witkin shell at 0.5 kg/m^2 with a
  0.15 mm collision radius. A thicker radius is impossible because cached
  layers already sit closer than that, and a Neo-Hookean shell at that
  radius stretched by tens of centimetres under its own weight.
* The libuipc FEM preconditioner is the multilevel additive Schwarz one. With
  block-Jacobi the conjugate-gradient solve took seconds per Newton
  iteration on this cloth.
* Wang's garment curriculum is available but off by default, there is no
  held-out human (a single-pose regional teacher reserves none, as in the
  Newton port), and the hospital gown is opt-in. FMVP's early-turn detector is reported per episode
  (`early_turn_rate`, `paper_filter_rate`) but, as in the paper, it filters
  trajectories rather than defining success.
  Its elbow region is a pair of projected slabs with no bound on distance
  from the arm, so `early_turn_rate` alone says little about episodes that
  never dressed the arm; the paper applies it after the 0.7 ratio gate, as
  the collector does.

`--policy heuristic` runs a port of the Newton seven-stage dressing expert
(approach, finger, middle, align-yaw, align-pitch, elbow-hook, last) as the
reachability baseline. With the cuff held at strength 1e4 it dresses the
whole forearm (ratio 1.0) and 0.28 of the upper arm on tshirt_26/human_0
within 900 decisions, which is below the success threshold of 0.7 but is
sustained upper-arm progress of the kind the reference says appears only
after hundreds of steps. At the old strength of 100 the same expert never
passed a forearm ratio of 0.08 because the garment was left behind by the
tool, so every reachability number recorded before that fix measured a
detached anchor. The same expert reaches an upper-arm ratio of 0.7 in 20 of
35 cells under Newton's VBD cloth; see the evidence record.

## Requirements

The package needs Genesis, PyTorch, and `pyuipc` in one interpreter. On this
machine that is the Genesis 1.1.2 checkout's environment:

```bash
GENESIS_PY=/home/ge47gax/kun/genesis-world/.venv/bin/python
```

Genesis is pinned to 1.1.2 and `pyuipc` to 0.0.28. The environment reaches the
coupler through Genesis private attributes, so a Genesis upgrade needs a
recheck. The libuipc checkout's own `.venv` carries `pyuipc` only and cannot
run training.

## Tasks

| Task | Deformable | Held | Marker whose centroid must reach the goal |
|---|---|---|---|
| `cloth_drag` | 25 cm sheet, 400 vertices | one corner | the whole sheet |
| `cloth_fold` | the same sheet | one corner | the held corner, target is the opposite corner |
| `cable_drag` | 30 cm rod, 26 points | one end | the rod midpoint |

Every episode runs to its time limit. Success is reported as a metric, never
used as a terminal condition, following the reference MDP.

## Commands

Scripted reachability check. Run this before training a task; if the scripted
policy cannot do it, SAC will not either.

```bash
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac \
    --task cloth_drag --policy heuristic --eval-only \
    --num-envs 2 --num-eval-episodes 6 \
    --work-dir output/uipc_manip --run-name heuristic_cloth_drag --save-trajectories
```

SAC training. `--num-envs` is the number of copies in the single IPC world;
32 is the default and the measured sweet spot on the RTX PRO 6000.

```bash
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac \
    --task cloth_drag --num-envs 32 --total-transitions 60000 \
    --eval-freq 250 --num-eval-episodes 32 \
    --work-dir output/uipc_manip --run-name cloth_drag_seed1 --seed 1
```

Policy and critic variants. The defaults reproduce the reference; the others
are there to be compared against it, not assumed better.

| Flag | Choices | Default | Meaning |
|---|---|---|---|
| `--actor` | `wang-flow`, `flat` | `wang-flow` | tool-point readout of a segmentation encoder (reference) or a globally pooled encoder |
| `--algo` | `sac`, `flashsac` | `sac` | scalar twin critic (reference) or the bounded categorical critic from the Newton `flashsac` path |
| `--obs-mode` | `visible_dual`, `visible_single`, `xray`, `wang_static_arm`, `stretch3_head`, `stretch3_head_wrist` | `visible_dual` | dressing: the cameras behind the point cloud (`dressing_obs.py`). `visible_dual` is the FMVP preset: two cameras derived from the arm, the sleeve hiding the arm as it goes. `wang_static_arm` is Wang RSS 2023's observation: one camera, the arm captured before dressing so the sleeve never hides it, the garment hidden by the body. `stretch3_head` puts a Hello Robot Stretch 3 pan-tilt head camera (D435if, portrait) in front of and to the right of the person; `stretch3_head_wrist` adds its gripper camera (D405, 7 to 50 cm) on the tool. The rigs render the whole body as an occluder. A resume keeps the checkpoint's mode; an explicit one only replays under another rig. See `agent_docs/performance/2026-09-10-camera-rigs.md` |
| `--critic-input` | `points`, `privileged` | `points` | dressing: the critic encodes the point cloud as the actor does (reference), or reads the simulator's 35-float state in a frame fixed to the arm (`dressing_privileged.py`): an asymmetric critic, used only in training, that takes the critic's point encoder out of every update. Replay then stores that state beside each observation, so the two forms do not resume from each other's checkpoints or snapshots |
| `--teacher-checkpoints` | paths | - | dressing: Wang's distillation. Regional teacher checkpoints, each trained on cells of one arm-pose region; every replay row is labelled with its slot's region, and the actor loss adds that region teacher's loss on the same observation: the squared distance of the squashed means plus that of the square-rooted standard deviations, summed over the batch as `SAC_AWAC.py` sums it. Every training slot's region needs a teacher |
| `--distill-weight` | float | `0.01` | weight of that teacher loss; the paper states 0.01 and the reference launcher 0.002 |
| `--encoder-precision` | `fp32`, `bf16` | `fp32` | run the point encoders under bfloat16 autocast and hand their features to fp32 heads; targets and losses stay fp32 because Q values near 90 resolve only to about 0.5 in bfloat16. Weights are the same, so a checkpoint evaluates under either, but a resume keeps the saved choice |
| `--encoder` | `pointnet2`, `transformer` | `pointnet2` | dense masked PointNet++ (reference) or a set transformer with a learned global token |
| `--action-repeat` | int | `1` dressing, `5` otherwise | simulation steps per decision. The tool speed cap covers the whole decision, as Newton's `decimation` does, so `--horizon 150 --action-repeat 6` is the reference's 900 simulation steps with six times fewer decisions: the configuration Newton's own sweep found best for upper-arm progress |
| `--dt` | float | `1/60` | dressing: simulation step [s]; `--action-repeat` x `--dt` is the decision period, 0.1 s in the configuration every ceiling was measured at. The held cuff's physical stiffness is `--cuff-strength` x vertex mass / dt^2 and the settle lasts `--settle-steps` steps, so a run at 1/30 s that keeps both passes `--action-repeat 3 --cuff-strength 4e4 --settle-steps 15`. The trainer scales none of them; `pretrain_wang` writes the matched values for its own `--dt`. Horizon and discount count decisions and do not change |
| `--settle-steps` | int | `30` dressing, `40` otherwise | simulation steps the world settles before the snapshot every reset restores; a dressing run now passes it to the environment, and a resume keeps the saved value |
| `--garment-curriculum-interval` | int | `0` | dressing: Wang's `curriculum_update_freq`. Every this many vector steps one more garment's slots are admitted to replay, easiest first; all slots keep stepping. Use a multiple of the horizon so a garment joins at an episode boundary. `0` trains on every garment from the start. Wang's value is in neither the original nor the Newton checkout, so any interval used in a run is a choice of this port. Evaluation plays every garment at every stage, unlike Wang's `evaluate`, which scores only the admitted ones |
| `--garment-curriculum-order` | names | Wang's five | dressing: preference order, easiest first; absent garments are skipped and unnamed ones appended |
| `--cell-source` | `live`, `cache` | `live` | dressing: `live` drapes each garment online in libuipc and places it on a generated SMPL-X body along the forearm, tshirt_4 and tshirt_392 as Wang places them (sleeve pointing away from the hand, torso over the forearm, the opening being the armhole seam), at a per-cell clearance checked against the whole arm mesh, so any (garment, body) pair is a cell; the hospital gown and tshirt_68 keep their baked hang (`LiveCellConfig.hang_as_baked_garments`), since the measured socket would start the gown upside down and tilts tshirt_68 64 to 72 degrees; `cache` reads the Newton bake's pre-worn states |
| `--body-seeds` | ints | - | dressing, live: the SMPL-X body seeds; every listed garment is placed on every body. An id of `1000 (r + 1) + k` samples its arm pose inside Wang's arm-pose region `r` (0 to 26); smaller ids sample the whole range. `--human N` remains the single-body shorthand and cannot be combined with it |
| `--heldout-bodies` | int | 2 live, 1 cache, 0 single body | dressing: whole bodies reserved from training, the greatest ones that carry every garment. Their slots step and are evaluated but never write to replay, and checkpoint selection ranks their scores first |
| `--heldout-body-seeds` | ints | - | dressing: explicit held-out bodies instead of the default choice |
| `--anchor-count` | int | `48` | dressing: cuff vertices held by the picker; Newton's twelve let the cuff slip off the hand on live cells |
| `--allow-partial-cell-coverage` | flag | off | dressing: accept a cell library that does not cover every requested garment and body, or fewer evaluation episodes than cells; for labelled smoke tests only |
| `--init-temperature` | float | `0.1` | initial SAC temperature; the reference value at 150 steps. The horizon-equivalent helpers rescale the reward and the temperature learning rate for a 900-step horizon but not this, so the critic target carries a six-times larger entropy term; `0.0167` is the variant under test |

For dressing prefer `--encoder transformer`: the 768-point observation makes the dense ball query several times more expensive per update than attention.

Replay a checkpoint without modifying it, and resume training from one.

```bash
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task cloth_drag \
    --resume output/uipc_manip/<run>/checkpoints/best.pt --eval-only --num-eval-episodes 8

PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task cloth_drag --num-envs 4 \
    --resume output/uipc_manip/<run>/checkpoints/checkpoint_0000500.pt \
    --resume-replay output/uipc_manip/<run>/checkpoints/replay_0000500
```

Watch one episode in the Genesis viewer, or render a saved trajectory offline.

```bash
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task cloth_drag --policy heuristic --eval-only --vis
PYTHONPATH=python $GENESIS_PY -m uipc_manip.preview output/uipc_manip/<run>/trajectories/episode_000.npz
```

### Wang's pretraining protocol

`pretrain_wang` runs Wang RSS 2023's pipeline the way its runnable reference,
`curl/train.py` under `launch_train_curl.py`, does: one SAC teacher per arm-pose
region, trained on that region's 45 training poses crossed with the five garments
and scored on its five held-out poses, then one student that learns every chosen
region with Wang's teacher loss. A slot's cell is fixed for the life of a world, so
the world is torn down and rebuilt on a fresh draw of training configurations every
`--rotate-every` episodes (one by default, Wang's per-episode draw). Replay,
optimizers and the held-out evaluation worlds persist across rebuilds.

```bash
# A regional teacher (region 13 is the middle interval of all three arm angles).
PYTHONPATH=python $GENESIS_PY -m uipc_manip.pretrain_wang teacher --region 13

# The student over those teachers' regions.
PYTHONPATH=python $GENESIS_PY -m uipc_manip.pretrain_wang student --regions 4 13 22 \
    --teacher-checkpoints output/uipc_manip/wang_teacher_r4_s1/checkpoints/best.pt \
    output/uipc_manip/wang_teacher_r13_s1/checkpoints/best.pt output/uipc_manip/wang_teacher_r22_s1/checkpoints/best.pt

# Continue a run from its latest checkpoint, optionally to a larger total budget.
PYTHONPATH=python $GENESIS_PY -m uipc_manip.pretrain_wang resume output/uipc_manip/wang_teacher_r13_s1 --transitions 900000
```

| Flag | Default | Meaning |
|---|---|---|
| `--region` / `--regions` | - | teacher: its one region (0-26); student: its regions, each with a checkpoint in `--teacher-checkpoints` |
| `--garments` | Wang's five | garments of the distribution |
| `--train-poses`, `--eval-poses` | 0-44, 45-49 | poses per region; body `1000 (r + 1) + k` is pose `k` of region `r` |
| `--num-envs` | 24 | episodes a training world holds at once |
| `--transitions` | 600,000 | total replay transitions of the run, counted across resumes; the reference launcher runs 5,000,000 |
| `--replay-capacity`, `--batch-size` | 400,000, 64 | Wang's values; the capacity is split evenly over the buffers |
| `--replay-split` | teacher `none`, student `region` | one buffer, one per garment, or one per region; each update draws its whole batch from one buffer, chosen uniformly among those holding more than a batch |
| `--temperatures` | `shared` | one entropy temperature, which is what the reference trains (its student update is handed a one-buffer list, so `alpha_idx` stays 0), or `per-buffer`, one per replay buffer as `SAC_AWAC.py` allocates them and its PCGrad baseline uses them |
| `--rotate-every` | 1 | episodes each world plays before it is rebuilt on a fresh draw |
| `--eval-every` | 10,000 | transitions between evaluations, taken at the next episode boundary; each round plays one deterministic episode per held-out configuration, and `best.pt` ranks their mean final upper-arm ratio first |
| `--eval-slots` | 32 | largest evaluation world; more held-out configurations use several worlds, all kept for the run |
| `--checkpoint-every` | 50,000 | transitions between checkpoints. Each writes the agent, the replay snapshot (only the latest is kept; 400,000 transitions are 17 GB in memory, about 2.9 GB compressed and 90 s to write) and `state.json`, which carries the rotation RNG and the counters a resume needs |
| `--garment-curriculum-interval` | 0 | transitions between admitting one more garment to the draw, easiest first; the reference launcher has none |
| `--dt` | 1/60 | simulation step; the action repeat (0.1 s decisions), cuff strength (the same physical hold) and settle follow it unless passed explicitly |

Every other `train_sac` flag (`--critic-input`, `--encoder-precision`, `--hidden-dim`,
`--updates-per-step`, `--anchor-count`, ...) passes through, and the flags the protocol
sets itself (`--num-eval-episodes`, `--body-seeds`, `--total-transitions`, ...) are
refused. A (garment, body) whose placement cannot clear the arm is dropped from its
pool the first time it is drawn and listed in the checkpoint metadata; nothing is
filtered on whether the scripted expert dresses it. Section 6 of
`agent_docs/performance/2026-09-10-one-policy-protocol.md` maps each choice to the
reference and states the deviations.

### The FMVP simulation pipeline

Wang RSS 2023 trains one SAC teacher per arm-pose region; FMVP rolls those
teachers out, keeps the trajectories that end dressed without an early turn
at the elbow, and clones them into one visual policy. The three stages map
onto three launchers here, with the set-transformer encoder and the libuipc
solver in place of PointNet++ and FleX.

```bash
# Stage I-A: SAC teacher with Wang's garment curriculum (interval is this port's choice).
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task dressing --cell-source cache --human 0 \
    --garments tshirt_26 tshirt_392 --num-envs 16 --encoder transformer \
    --total-transitions 100000 --garment-curriculum-interval 1800 \
    --eval-freq 1800 --num-eval-episodes 16 --checkpoint-interval 900

# Stage I-B: roll the frozen teacher out and keep the paper-filtered episodes.
PYTHONPATH=python $GENESIS_PY -m uipc_manip.collect_rollouts --task dressing --cell-source cache --human 0 \
    --garments tshirt_26 tshirt_392 --num-envs 16 --encoder transformer \
    --checkpoint output/uipc_manip/<run>/checkpoints/best.pt \
    --target-kept-episodes 2514 --max-episodes 8000 --run-name <run>

# Stage I-C: behaviour-clone the kept episodes into a student (NLL, Adam 1e-4, batch 128, 40k updates).
PYTHONPATH=python $GENESIS_PY -m uipc_manip.distill \
    --source-dirs output/uipc_manip/<run>/rollouts --run-name <run>_student

# Evaluate the student like any checkpoint.
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task dressing --cell-source cache --human 0 \
    --garments tshirt_26 tshirt_392 --encoder transformer --eval-only \
    --resume output/uipc_manip/<run>_student/checkpoints/actor_best.pt
```

The collector stores one compressed `episode_*.npz` per kept episode (about
3 MB at 900 steps, the padded rows compress away) and every attempt in
`episode_metrics.json`; the
distiller splits by episode, holds out ten per cent, and caps what it loads
with `--max-train-transitions`. The paper's 0.7 filter keeps nothing until a
teacher ends episodes dressed, which none does here yet: `--policy heuristic
--min-upperarm-ratio 0` with `--loss mse` (the scripted expert's bang-bang
actions have no finite log-likelihood) exercises the stages mechanically and
is labelled a smoke setting, not a result.

## Reward

The reward is progress toward the goal measured in units of `max_translation`,
so a unit action straight at the goal earns about +1 per decision, plus +1 on
every decision whose marker centroid is within the task tolerance. Per-step
rewards therefore stay on the reference's `[-1, 1]` scale. A ten-times smaller
progress reward, tried first, left the SAC entropy term dominating the policy
objective for tens of thousands of transitions.

## Observation and action

The action is a three-dimensional tool displacement in `[-1, 1]`, scaled by
`max_translation` per decision and applied over `action_repeat` simulation
steps. The tool orientation is held pointing down, so the arm never has to
trade rotation tracking against position tracking.

The observation is one flat float32 vector: a segmented point cloud expressed
relative to the tool centre point, with the goal and the tool inserted as
explicit points, followed by seven scalars carrying the absolute tool position,
the tool-relative goal offset, and an attachment flag. Padded rows carry no
segmentation flag, which is how the encoder derives its validity mask. The
layout lives in `obs.py` and is shared by the environment, the replay buffer,
and the networks.

## Ported from the Newton teacher

* The `wang-flow` actor: a segmentation PointNet++ gives every point a feature
  and the policy trunk reads the explicit tool point's row, which localises the
  policy at the gripper. This is the actor the reference
  `--fmvp-pretrain-defaults` preset trains.
* Scalar SAC with twin critics, a learned entropy temperature targeting
  `-action_dim`, and the reference form `Q(encode(s), a)` where the action
  joins after the encoder. The `flashsac` categorical critic with its
  C51-on-mean target projection is ported as an option.
* Batched collection: all environments share the horizon, reset together from
  one dumped IPC snapshot, and feed one replay buffer with one gradient update
  per collected transition.
* The Wang `pointcloud_3` defaults: actor and critic learning rate `1e-4`,
  batch 64, actor update every fourth optimizer step, Q-head Polyak `0.01`,
  encoder Polyak `0.05` every second update, no gradient clipping, no random
  prefill, PointNet++ set-abstraction radii `[0.05, 0.1]` and ratios
  `[1.0, 1.0]`, segmentation head widths `[128, 128]`.
* `wang_equivalent_discount`, `wang_equivalent_alpha_lr`, and
  `wang_equivalent_reward_scale`, which rescale the discount, the temperature
  learning rate, and the replay reward scale when the horizon is not 150 steps.
  At the default 150-step horizon all three return the reference values.
* `gradient_update_budget`, the replay-prefill rule that keeps one gradient
  update per collected transition without over-training the first minibatch.
* The time-limit bootstrap convention: a horizon end is stored with
  `not_done = 1` and the observation (and, for the privileged critic, the
  state) captured before the reset, so the critic
  never bootstraps across an episode boundary.
* A checkpoint protocol covering the point budget, observation and action
  dimensions, trunk width, and full encoder configuration. Loading refuses a
  mismatch, and evaluation refuses a checkpoint trained on another task.

## Rewritten rather than ported

The Newton PointNet++ is built on PyTorch Geometric, which is not installed in
the Genesis environment. `models.py` implements the same architecture on dense
`[B, N, C]` tensors with a validity mask: ball queries are a masked top-k over
pairwise distances, aggregation is a masked max, and padded points never
contribute. Neighbour counts per level are the cost knob, since the dense query
is linear in them. With the reference ratios of `1.0` the segmentation
encoder's feature propagation is the identity and is implemented as direct
concatenation. Tests cover padding invariance, permutation invariance, and the
tool-point readout.

`--encoder transformer` is not a port. It is a small self-attention encoder
with a learned global token, offered because attention over a few hundred
points is dense, cheap, and has no radius to tune against the scene scale.

## Limitations

* **Synchronised episodes.** All copies share one IPC world, so they reset
  together at the fixed horizon; per-environment early termination is not
  supported. This matches the reference's fixed-horizon slots.
* **Private Genesis API.** `_ipc_objects`, `_ipc_animator`,
  `_ipc_contact_tabular`, and `_ipc_world` are Genesis 1.1.2 internals, the
  same access pattern as the upstream IPC examples.
* **Picker attachment, not a friction grasp.** The held vertices follow the
  tool through a soft position constraint, as in SoftGym, Wang RSS 2023, and
  the Newton teacher. The fingers stay open and still collide with the rest of
  the deformable through IPC. A physical closed-finger grasp is what
  `ipc_robot_deformables.py` in the Genesis checkout demonstrates. The hold is
  soft: in contact it still yields by up to 5 cm at strength 1e4, so the
  reference's kinematic pin is approximated, not reproduced. The `tracking_error`
  info key and the evaluation's `max_tracking_error` report the gap.
* **State-based point cloud.** Points are sampled from simulator state, not
  rendered from cameras. There is no visibility filtering, camera jitter, or
  dropout, so this does not reproduce the reference visual observation.
* **No force in the observation or reward,** matching the reference, which
  keeps force out of simulation pretraining.
* **This is pretraining only.** There is no rollout filter, behaviour-cloning
  distillation, or sim-to-real stage.
