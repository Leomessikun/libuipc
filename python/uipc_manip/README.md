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
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task dressing --human 0 \
    --garments tshirt_26 tshirt_392 --policy heuristic --eval-only --num-envs 4 --num-eval-episodes 4
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac --task dressing --human 0 \
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
* No garment curriculum and no held-out human, and the hospital gown is
  opt-in. FMVP's early-turn detector is reported per episode
  (`early_turn_rate`, `paper_filter_rate`) but, as in the paper, it filters
  trajectories rather than defining success.

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
| `--encoder` | `pointnet2`, `transformer` | `pointnet2` | dense masked PointNet++ (reference) or a set transformer with a learned global token |
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
  `not_done = 1` and the observation captured before the reset, so the critic
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
