# 2026-09-08 — IPC manipulation pretraining throughput and task validation

- Status: Accepted (baseline record; no libuipc solver change)
- Before commit: n/a, new package
- After commit: `python/uipc_manip/` on `cloth-cable-manip-rl`
- Benchmark manifest name and samples commit: n/a; this uses the package's own
  scripted sweep, not `scripts/run_benchmark.py`

## Question

Can a robot manipulation reinforcement-learning environment run on this
project's IPC solver at a throughput that makes policy pretraining practical,
and do the three goal-reaching tasks admit a solution at all?

Throughput is the deciding quantity. The Newton cloth-dressing teacher this
work ports its algorithm from trains against a Vertex Block Descent cloth
solver. IPC resolves contact to a penetration-free guarantee instead, which
costs more per step, so the reference transition budgets do not transfer
without measurement.

## Environment

| Field | Value |
|---|---|
| GPU / driver | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.84 |
| CUDA toolkit | PyTorch 2.12.0+cu130 |
| OS / compiler | Ubuntu 24.04.4 LTS |
| Build type and important flags | released `pyuipc` 0.0.28 wheel, not a local build of this checkout |
| Genesis | checkout `genesis-world` at `aedae39b`, version 1.1.2 |
| Worktree state | dirty; the package under test is the change |

The GPU was shared throughout with an unrelated IsaacLab training process
holding 13.7 GiB at full utilisation. Every number below is therefore a
pessimistic reading of an uncontended machine, and the absolute values should
not be compared against measurements taken on an idle GPU.

## Workload and method

The scene is a Franka Panda, a table plane, and one deformable: a 400-vertex
25 cm cloth sheet or a 26-point 30 cm elastic cable. Genesis drives the robot
and the plane; the deformable is a native libuipc geometry in the Genesis IPC
coupler. One decision advances five simulation steps at `dt = 0.01`.

Timings discard the first warm-up decisions of each run and report medians over
the remainder. Coupler phase costs were measured by timing the individual
phases of `IPCCoupler.couple` around ten steps. The scripted sweep plays six
deterministic episodes per task through the package launcher.

## Correctness and safety

- The IPC world is asserted valid immediately after `scene.build()` and after
  every decision; deformable positions are checked finite. No run reported a
  simulation error.
- Reset recovers a dumped IPC snapshot taken after the initial settle. Restored
  positions matched the snapshot exactly, to `0.0` maximum absolute deviation.
- End-effector tracking error stayed below 3.5 cm in every episode across all
  tasks, so the reported motion is the arm's, not a constraint dragging the
  deformable through the robot.
- Package tests: 9 passed on CPU, 1 deselected as the `cuda`-marked environment
  test.

## Results

Per-step cost decomposition, cloth task, medians:

| Scope | Cost | Notes |
|---|---:|---|
| `world.advance` inside the coupler | 68 ms | the IPC solve itself |
| Full Genesis `scene.step` | 120 ms | includes the Genesis rigid solver and coupling |
| Inverse kinematics per simulation step | 5 ms | one seeded sample, no restarts |
| Observation construction | 0.4 ms | |
| One decision, five simulation steps | 585 ms | |

Vector-environment scaling, subprocess workers, cloth task:

| Workers | Vector step | Transitions per second |
|---:|---:|---:|
| 1 | 439 ms | 2.3 |
| 2 | 608 ms | 3.3 |
| 4 | 906 ms | 4.4 |

SAC optimizer update cost, batch 64:

| Point budget | Ball-query neighbours | Update | Peak memory |
|---:|---:|---:|---:|
| 256 | 8 | 47 ms | 1.2 GiB |
| 256 | 16 | 72 ms | 1.9 GiB |
| 512 | 8 | 161 ms | 2.2 GiB |
| 512 | 16 | 196 ms | 3.7 GiB |

Scripted reachability sweep, six deterministic episodes per task, five
simulation steps per decision:

| Task | Success | Mean final error | Mean return | Max tracking error |
|---|---:|---:|---:|---:|
| `cloth_drag` | 6/6 | 3.3 mm | 14.19 | 22.2 mm |
| `cloth_fold` | 6/6 | under 0.01 mm | 14.76 | 14.7 mm |
| `cable_drag` | 6/6 | 14.4 mm | 11.08 | 20.6 mm |

## Interpretation

Directly measured: the IPC solve is 68 ms of a 120 ms scene step, so slightly
over half the wall time is the contact solver and the rest is the Genesis rigid
solver and coupling bookkeeping. Reducing decision cost by cutting the number
of simulation steps per decision is therefore bounded by the Genesis half.

Directly measured: subprocess scaling is strongly sublinear, 1.9 times
throughput for four times the processes. The workers contend for one GPU that
is already shared, so this is a contention ceiling, not a defect in the vector
environment. More workers past four buy little.

Inferred: at 4.4 environment transitions per second before optimizer time, the
Newton teacher's five-million-transition budget is not reachable here. This
package targets the thousands-to-tens-of-thousands range, and a full-scale
teacher would need either a dedicated GPU, a coarser deformable, or fewer
simulation steps per decision.

Two measured regressions shaped the defaults:

- **Three simulation steps per decision is not enough.** At three the scripted
  policy stalled about 4 cm from the goal and scored 0/6 on both drag tasks; at
  five it scores 6/6. The commanded tool displacement per decision is identical
  either way, so the difference is the deformable's settling time under the
  moved constraint, not reach. Five is the validated setting.
- **Truncating the marker set by vertex index biases the observation.** With a
  marker set larger than the point budget, keeping the first vertices showed the
  policy one region of the sheet while the reward measured the centroid of all
  of it. The scripted policy drove that region to the goal and stalled at 4.3 cm
  in 6 of 6 episodes. Sampling markers at random instead restores 6/6.

The second finding generalises beyond this package: whenever an observation
subsamples a mesh whose reward is computed over the whole mesh, the subsample
must be spatially unbiased or the two disagree about what is being optimised.

## Decision

Accept as the baseline record. Defaults set from it: five simulation steps per
decision, a 256-point budget, and ball-query neighbour counts of 8 and 16 per
set-abstraction level. All three tasks pass the scripted gate, so all three are
worth training.

## Reproduction and artifacts

```bash
GENESIS_PY=/home/ge47gax/kun/genesis-world/.venv/bin/python
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac \
    --task cloth_drag --policy heuristic --eval-only \
    --num-envs 2 --num-eval-episodes 6 \
    --work-dir output/uipc_manip --run-name heuristic_cloth_drag
```

Metrics, trajectories, and checkpoints land under `output/`, which is ignored.
