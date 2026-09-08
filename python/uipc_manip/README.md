# uipc_manip — robot cloth and cable manipulation pretraining on the IPC solver

`uipc_manip` trains a point-cloud Soft Actor-Critic policy to manipulate a
deformable object with a Franka Panda, where every deformable contact is
resolved by libuipc's Incremental Potential Contact solver. The scene, robot,
and rendering come from Genesis; the cloth or cable is a native libuipc
geometry inserted through the Genesis IPC coupler.

The algorithm, hyperparameters, and transition semantics are ported from the
Newton cloth-dressing teacher on the `leomessikun/fmvp-sac-retrain` branch,
which follows Wang RSS 2023 as used for FMVP simulation pretraining. The task
suite is deliberately smaller: three goal-reaching tasks rather than a
multi-garment dressing curriculum.

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

SAC training.

```bash
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac \
    --task cloth_drag --num-envs 4 --total-transitions 20000 \
    --eval-freq 250 --num-eval-episodes 4 \
    --work-dir output/uipc_manip --run-name cloth_drag_seed1 --seed 1
```

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

* Scalar SAC with twin critics, a learned entropy temperature targeting
  `-action_dim`, and the reference form `Q(encode(s), a)` where the action
  joins after the encoder.
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
is linear in them. Tests cover padding invariance and permutation invariance.

## Limitations

* **One environment per process.** Native IPC geometry reaches Genesis through
  coupler internals that support a single scene, so parallelism uses subprocess
  workers, each initialising Genesis itself.
* **Private Genesis API.** `_ipc_objects`, `_ipc_animator`,
  `_ipc_contact_tabular`, and `_ipc_world` are Genesis 1.1.2 internals, the
  same access pattern as the upstream IPC examples.
* **Picker attachment, not a friction grasp.** The held vertices follow the
  tool through a soft position constraint, as in SoftGym, Wang RSS 2023, and
  the Newton teacher. The fingers stay open and still collide with the rest of
  the deformable through IPC. A physical closed-finger grasp is what
  `ipc_robot_deformables.py` in the Genesis checkout demonstrates.
* **State-based point cloud.** Points are sampled from simulator state, not
  rendered from cameras. There is no visibility filtering, camera jitter, or
  dropout, so this does not reproduce the reference visual observation.
* **No force in the observation or reward,** matching the reference, which
  keeps force out of simulation pretraining.
* **This is pretraining only.** There is no rollout filter, behaviour-cloning
  distillation, or sim-to-real stage.
