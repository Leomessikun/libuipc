# 2026-09-08 — IPC manipulation pretraining throughput and task validation

- Status: Accepted (baseline record, revised the same day for the batched world; no libuipc solver change)
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

### Second pass: one IPC world for all environments

The first pass ran one IPC world per subprocess. The Newton teacher gets its
transition budget from batched simulation instead, and the Genesis coupler
already supports the IPC counterpart: `N` deformable copies in one world,
each in its own subscene so contact never crosses environments, with the
Newton solve covering the whole batch. The environment was rewritten that way.
Measured on the idle GPU (the unrelated training job had finished), cloth task,
`N` Franka robots and `N` cloth copies in one Genesis scene:

| Copies | Step | Environment steps per second |
|---:|---:|---:|
| 1 | 45 ms | 22 |
| 8 | 78 ms | 102 |
| 16 | 97 ms | 164 |
| 32 | 144 ms | 222 |
| 64 | 242 ms | 264 |

The single-environment step is 45 ms here against 120 ms in the first pass.
The difference is GPU contention with the job that was running then, not a
solver change; the first-pass numbers were pessimistic by that factor.

Batching preserves the physics. With identical per-environment goal seeds,
episode zero of the scripted cloth drag ends 3.3 mm from its goal alone and
3.4 mm inside a batch of 32; the six goals that succeed at two environments
succeed at the same 3.3 mm inside batches of 8 and 32. Wall time for 32
scripted episodes was 206 s in one batch against 415 s for four episodes run
one at a time, a sixteen-fold gain per episode.

Two goal seeds fail at every batch size, ending about 33 mm away with a
return near 1.0. Those goals lie in the direction that folds the held corner
back over the sheet, where dragging the corner moves the centroid little. The
scripted policy cannot solve them and the learned policy will have to; they
were invisible at two environments because two environments sample too few
goal directions.

Optimizer cost per update at batch 64 and a 256-point budget, with the
32-environment action pass:

| Actor | Critic | Encoder | Update | 32-env action | Peak memory |
|---|---|---|---:|---:|---:|
| flat | sac | pointnet2 | 68 ms | 7.5 ms | 1.8 GiB |
| wang-flow | sac | pointnet2 | 122 ms | 12.7 ms | 2.0 GiB |
| wang-flow | flashsac | pointnet2 | 121 ms | 12.7 ms | 2.0 GiB |
| wang-flow | sac | transformer | 19 ms | 1.7 ms | 0.6 GiB |
| flat | sac | transformer | 18 ms | 1.8 ms | 0.6 GiB |

With one gradient update per collected transition and 32 environments, the
update pass is 32 updates per vector step. The reference segmentation
PointNet++ therefore costs about 3.9 s of optimizer time per 0.72 s of
simulation, roughly 7 transitions per second end to end; the transformer
encoder brings the same loop to roughly 24 transitions per second.

### Third pass: reward scale

With the batched world and the reference actor in place, the transformer
encoder run still showed almost nothing after 16,800 transitions: mean final
distance 97.8 mm at 8,000 and 91.8 mm at 16,000, 0 of 32 successes at both.
The cause was reward scale. The progress reward was ten times the metres of
marker motion toward the goal, so one decision could earn at most 0.06, while
the initial entropy term of the policy objective contributed about 0.43 per
decision. The reference calibrates its temperature and learning rates for
per-step rewards in `[-1, 1]`, and its own notes warn that every reference
hyperparameter silently miscalibrates when the value scale changes.

The reward was rescaled to progress in units of `max_translation`, so a unit
action straight at the goal earns about +1, with +1 more on every decision
inside the tolerance. Same network, hyperparameters, environment, and seed:

| Transitions | Success (32 episodes) | Mean final distance | Mean return |
|---:|---:|---:|---:|
| 8,000, old reward | 0/32 | 97.8 mm | 0.04 |
| 16,000, old reward | 0/32 | 91.8 mm | 0.10 |
| 8,000, rescaled | 27/32 | 14.6 mm | 121.3 |
| 16,000, rescaled | 27/32 | 13.2 mm | 124.3 |

The scripted policy scores 27/32 on the same goal seeds, so 8,000 transitions
reach the scripted baseline. The five remaining goals lie in the direction
that folds the held corner over the sheet; whether the learned policy passes
them is what the longer run measures. The run collected 8,000 transitions in
480 s with 32 environments, about 17 per second, while an unrelated training
job shared the GPU.

The resume paths were exercised on a real checkpoint from this run:
evaluation-only loading, training continuation with the replay snapshot
(transition and update counts continue from the checkpoint), and a deliberate
encoder mismatch, which is refused with both protocols printed.

### Fourth pass: the Newton dressing task on IPC

The owner's target is the Newton cloth-dressing teacher, so the dressing MDP
was rebuilt on the batched IPC world from the Newton bake cache (23 pre-worn
garment/human cells). Findings, each measured on a single tshirt_26/human_0
cell unless stated:

| Finding | Evidence |
|---|---|
| 17 of 23 cells build under libuipc's checks | pure-pyuipc init sweep; the six failures are residual self-intersections from the bake, not arm contact |
| Cloth collision radius must be the thin 0.15 mm the sim2sim path used | with a 1 mm radius libuipc reports cached layers closer than the summed radii; with the checks off the solver asserts "thickness violated" on those pairs |
| Block-Jacobi PCG dominates: 83 of 84.5 solver seconds over 25 Newton iterations | libuipc timer; step 7.3 s at 200 kg/m^3, 3.9 s at 3333 kg/m^3 |
| Multilevel additive Schwarz preconditioner: 3.9 s to 0.67 s per step | same probe; PCG time per Newton iteration 1.5 s to 0.1 s; once the garment settles a step costs 12-55 ms |
| Contact stiffness is not a lever | libuipc clamps the requested kappa into a range set by the geometry, so 1e5 and 1e7 behave alike |
| Genesis re-tessellates an imported arm mesh, 1307 vertices to 4885 with duplicates, and the IPC trajectory filter then asserted on NaN distances | device assert `D=-nan` in the simplex trajectory filter, twice, both times on a cloth face against a duplicated arm vertex; a native fixed affine body from the cached mesh runs 800 steps cleanly and builds in 6 s against 116 s |
| A Neo-Hookean shell at the thin radius stretches by tens of centimetres under its own weight | membrane stiffness scales with the radius, 6e4 Pa times 0.3 mm is 18 N/m; the strain-limiting Baraff-Witkin shell holds shape and is what the reference-validated Genesis path used |
| The cached hand-to-shoulder pull threads the sleeve over the fingertips only | forearm ratio 0.01-0.04 with the anchor at the shoulder, unchanged by erosion 15-25 mm, 40 anchored vertices, or a 1000 hold; the opening (radius 8-10 cm) is wider than the hand, so this is the motion, not the geometry, and the learned policy has to do better |
| Batched throughput, mixed tshirt_26 / tshirt_392 / hospital_gown slots | 8 slots 480 ms per step, 16 slots 929 ms per step: near-linear, the garment vertex count saturates the GPU, so roughly 17 environment steps per second at 16 slots |
| The cuff hold at libuipc's default strength of 100 does not carry this garment | `SoftPositionConstraint` energy is `0.5 * strength * vertex_mass * |x - aim|^2`, so 100 is a spring with a 0.6 s period; moving the anchor 20 cm through free air at the reference's 0.15 m/s the cuff sags 1.4 cm at rest, lags up to 6.5 cm in motion, and 3.1 cm after a 30-step hold; at 1e4: 0.6 mm, 1.5 cm, 2 mm; at 1e5: 0.1 mm, 5 mm, 0.4 mm; step cost 138, 202, 211 ms for two slots |
| Once the sleeve touches the hand, a strength-100 hold lets the tool leave the garment behind | expert run, tshirt_26/human_0: the largest held-vertex error grows 2 cm at step 100, 11 cm at 300, 43 cm at 500, 107 cm at 800, while the forearm ratio stays at 0.07; the previous reachability sweep measured a detached anchor |

The libuipc default gravity is along -y; a pure-pyuipc probe that omitted the
Genesis gravity setting spent an hour blaming contact for a garment that was
simply falling sideways. Set `config["gravity"]` in any standalone probe.

Reachability with the ported Newton seven-stage expert, tshirt_26/human_0,
900 decisions, maximum forearm dressed ratio reached (upper arm never
reached, success requires an upper-arm ratio of 0.7):

| Variant | Forearm ratio | Note |
|---|---:|---|
| reference: friction 0.3, erosion 6 mm, 12 anchors at strength 100 | 0.08 | opening passes the fingertips, threaded by the winding test |
| friction 0.1 | 0.17 | |
| friction 0.0 | 0.18 | friction is a factor, not the blocker |
| friction 0.1, erosion 20 mm (fingers removed) | 0.22 | finger snagging is not the blocker |
| cached waypoint pull instead of the expert, any erosion, 40 anchors, or a 1000 hold | 0.01 to 0.04 | |
| hold strength 1e4, first attempt | 0.00 | inconclusive at the time: the anchor sat 2.6 cm from the elbow inside the 12 mm no-move shell and steps cost up to 3 s; superseded below |

Every row above was measured with a hold that had let go of the garment (see
the two findings on the strength-100 hold). With the held-vertex error
logged, the same expert, cell, and horizon give:

| Hold strength | Forearm ratio | Upper-arm ratio | Held-vertex error in contact | Median step, 1 slot |
|---:|---:|---:|---:|---:|
| 100 (old default) | 0.07 | 0.00 | 0.43 m by step 500, 1.07 m by 800 | 245 ms |
| 1e4 (new default) | 1.00 by step 800 | 0.28 at 900 | 3-5 cm | 177 ms, spikes to 2.2 s while hooking the elbow |
| 1e5 | 0.72 by step 400 | not reached, run stopped | 2-5 cm | 115-644 ms until the elbow hook, then over 3.6 s per step; unusable for training |

The sleeve opening is nearly twice as wide as the hand (radius 9.9 cm against
a hand radius of 6 cm), so the geometry admits threading, and with a hold
that carries the garment the expert threads it. The hold was a blocker; the
friction, erosion, and cloth-model rows above were all measured with a
detached anchor and say nothing either way, so they have not been ruled out
as what separates 0.28 from the 0.7 threshold, and none has been retested
with the working hold yet. The reference's kinematic pin is still only
approximated: in contact the soft hold yields by 3 to 5 cm at either
strength. The no-move shell also stalls the expert for about 200 decisions
at the elbow. Under Newton's VBD cloth, where the cloth stretches and
penetrates slightly, the same expert reaches 0.7 in 20 of 35 cells.

First SAC evaluation on the dressing task, 16 slots, tshirt_26 and
tshirt_392, transformer encoder, 28,800 transitions in 44 minutes at 1.4 s
per vector step: 0 of 16 successes, forearm and upper-arm ratios 0.0, task
reward -0.099, which is the pre-insertion finger-distance term. Consistent
with the reference's own note that upper-arm progress begins only after
hundreds of sustained steps. The second evaluation at 57,600 transitions was
the same: 0 of 16, ratios 0.0, return -92.5. The run used the strength-100
hold and was stopped at 3,700 vector steps once the hold was shown to detach
the garment. Its critic had drifted from a Q mean of 1.4 to 86 against
returns of -95. The checkpoint's temperature explains the drift: alpha was
0.079 (from 0.1, at the horizon-equivalent alpha learning rate of 1.67e-5),
so with the policy near its target entropy of 6 nats the soft value carries
about 0.5 per step of entropy bonus against a scaled reward of -0.017 per
step, summed over an effective horizon of 600 steps instead of the
reference's 100. The reward was rescaled by 150/900 to keep its discounted
sum at the reference's scale; the temperature was not, so the critic's
target is dominated by an entropy term six times larger relative to reward
than in Wang's setting, while the actor's per-step trade-off `alpha log pi -
Q` is unchanged. The critic will keep chasing that growing baseline for
hundreds of thousands of updates. This is left as measured; the rerun keeps
the reference temperature so its Q mean is comparable, and dividing the
initial temperature by the same 150/900 is the candidate change if the
drift recurs. Wall-clock figures from the stopped run are contaminated by
the probes that shared the GPU.

Rerun with the strength-1e4 hold, same 16 slots and settings: 1.87 s per
vector step over the first 1,800 steps with no other GPU load, a 16-episode
evaluation in 7 minutes. First evaluation at 28,800 transitions: 0 of 16,
forearm and upper-arm ratios 0.0 including their episode maxima, return
-84.5, task reward -0.087, and the held cuff now stays within 2.8 cm of the
tool. The critic drift recurs: Q mean 1.5 to 52 over the first 28,800
updates against training returns near -139.

## Newton reference compared, same machine

The Newton teacher's own runs live in `runs/dressing_pointcloud_sac` of the
`leomessikun/fmvp-sac-retrain` worktree, 33 of them with evaluations, on this
same RTX PRO 6000 Blackwell. Their TensorBoard scalars carry
`perf/env_steps_per_s`, so speed is measured on both sides rather than
inferred. Newton runs 40 environments per process against this port's 16.

| | Newton, VBD cloth | This port, libuipc IPC |
|---|---:|---:|
| Vector steps per second | 0.97 to 1.00 | 0.53 |
| Environment transitions per second | 38.9 to 42.7 | 8.9 |
| Milliseconds per environment transition | 23 to 26 | 112 |

Newton is 4.4 times faster per transition. Its older runs at 5 to 23
transitions per second are not slower physics: fidelity is identical across
them (`cloth_substeps` 10, `vbd_iterations` 15, decimation 1), and the
Aug 25 sweep shows eight concurrent MPS jobs at 6 to 8 transitions per second
each against 42 solo, so the low numbers are GPU contention. The gap is what
the two solvers do per step: a fixed-cost local VBD sweep against a full
Newton solve with continuous collision detection, line search, and a global
linear system that guarantees no penetration.

Neither has solved the task. Success is the same definition on both sides, a
final upper-arm ratio of at least 0.7.

| | Newton | This port |
|---|---|---|
| Runs with evaluations | 33 | 2 |
| Best success rate | 0.025, one episode of 40, in four runs; the other 29 runs scored 0.000 | 0 of 16 |
| Best rate of touching 0.7 during an episode | 0.35 | 0.0 |
| Best mean of the per-episode maximum upper-arm ratio | 0.381 at horizon 150, 0.21 to 0.27 at horizon 900 | policy 0.0, scripted expert 0.28 |
| Sleeve latched over the hand | 0.75 to 1.0 | the expert threads it |
| Largest budget spent | 74,400 vector steps at 40 environments, 2.98M transitions, 72 hours | 50k transitions, 1.8 hours |

So the reference reaches the same plateau this port reaches: the sleeve
latches over the hand and then stalls a fifth to a third of the way up the
arm. The scripted expert here, at 0.28, is inside the band of Newton's
trained policies at horizon 900. That plateau is a property of the task as
posed, not of the solver.

Newton's best upper-arm result came from its shortest-decision configuration,
horizon 150 with decimation 6: the same 900 simulation steps per episode, but
six simulation steps per decision instead of one. This port runs the opposite
extreme, horizon 900 with `action_repeat` 1, which is the most expensive
setting and the one Newton's own sweep found worst.

## The Wang / FMVP pipeline, ported

The owner asked for Yufei Wang's method (RSS 2023, original code in the
local `dressing/` checkout) as FMVP extends it (Appendix A.1) and as the
Newton branch reproduces it (`dressing/docs/PIPELINE.md`, the authoritative
local spec), with this port's encoder and solver. What was missing and is
now in place:

| Stage | Reference | Here |
|---|---|---|
| I-A garment curriculum | `train_multi_garments.py`: `curriculum_step = step // curriculum_update_freq + 1`, garments sampled among the first `curriculum_step` of `hospital_gown, tshirt_26, tshirt_68, tshirt_4, tshirt_392` | slots are bound to garments, so the schedule gates which slots write to replay while all keep stepping (the Newton form); `--garment-curriculum-interval`, off by default because Wang's frequency is in neither checkout |
| I-A decision rate | Newton's best upper-arm result used horizon 150 with decimation 6, the tool speed cap spanning the whole decision | `--action-repeat` now reaches the dressing environment, whose cap already spans the decision |
| I-B rollout filter | more than 8,000 episodes; keep final upper-arm ratio at least 0.7 and no early turn; 2,514 kept | `collect_rollouts`: same two filters, one compressed file per kept episode, every attempt recorded |
| early-turn test | gripper in the projected quarter-segments around the elbow and on the inner side by two signed XZ cross products (y-up FleX) | same region; the cross products are taken in the arm's own bend plane with the sign resolved against the shoulder and finger, because a literal XZ copy is a vertical plane in this z-up scene |
| I-C distillation | NLL behaviour cloning, Adam 1e-4, batch 128, 40,000 updates, PointNet++ actor | `distill`: same, on the teacher's actor and encoder (set transformer); the student saves as a full SAC checkpoint so the evaluator is unchanged |
| checkpoint selection | FMVP's final-ratio criterion first | success (final ratio at least 0.7), final ratio, maximum ratio, return |

Not ported, deliberately: per-pose replay sampling (the `yufei_r111_s100`
Newton variant, not the reference launcher, which uses one buffer); a
held-out human (a single-pose regional teacher reserves none, as in Newton);
the 5M-transition budget per regional teacher, which is about 156 hours at
this port's 8.9 transitions per second.

Mechanical check on the GPU, two slots, the scripted expert at strength 1e4,
900 decisions, filter threshold lowered to 0 so something is kept:

| | tshirt_26 | tshirt_392 |
|---|---:|---:|
| Final upper-arm ratio | 0.272 | 0.184 |
| Forearm ratio | 1.00 | 1.00 |
| Early turn flagged | no | no |
| Largest held-vertex error | 6.5 cm | 32 cm |
| Return | 139 | 34 |

The early-turn detector does not flag the expert, which hooks the elbow from
the outside by construction, and the unit test flags the mirrored path. The
32 cm hold error on tshirt_392 says the 1e4 hold still loses that garment
for part of the episode; it is the next thing to look at for the second
garment. A 300-update MSE distillation of those two episodes brings the
action error from 0.34 to 0.02, and the student loads and plays through the
unchanged evaluator. None of this is a result: the paper filter keeps
nothing at 0.7 because no policy here ends an episode dressed to 0.7.

## Machine utilisation, measured

At 16 environments one training process holds 7.8 GB of the 96 GB of GPU
memory and 5.2 GB of host RAM, and drives one CPU core of 32 at 100 per cent.
GPU utilisation reads 93 to 100 per cent at 356 to 370 W of a 600 W limit.
Neither memory nor the CPU is the limit.

Running a second training process was measured rather than assumed:

| | Solo | Two processes |
|---|---:|---:|
| Seconds per vector step, first run | 1.79 | 2.93 |
| Seconds per vector step, second run | - | 3.28 |
| Aggregate environment transitions per second | 8.9 | 10.3 |
| GPU memory | 7.8 GB | 17.9 GB |

A second process buys 1.16 times the aggregate throughput while making each
run 64 per cent slower. Newton's eight-way MPS sweep measured the same shape,
1.24 times aggregate. The GPU is saturated by small serial solver kernels, so
neither more processes nor more environments per process helps: the earlier
scaling measurement, 8 slots at 480 ms and 16 slots at 929 ms per step, is
linear, which means transitions per second is flat in the environment count.

What would raise throughput is fewer decisions per episode, which is the
`action_repeat` and horizon trade Newton's own sweep found best, or fewer
garment vertices. Adding hardware pressure will not.

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

Accept as the baseline record. Defaults set from it: one batched IPC world
with 32 copies, five simulation steps per decision, a 256-point budget, and
ball-query neighbour counts of 8 and 16 per set-abstraction level. All three
tasks pass the scripted gate, so all three are worth training. The subprocess
vector environment of the first pass was removed; it was the wrong
architecture for this solver.

## Reproduction and artifacts

```bash
GENESIS_PY=/home/ge47gax/kun/genesis-world/.venv/bin/python
PYTHONPATH=python $GENESIS_PY -m uipc_manip.train_sac \
    --task cloth_drag --policy heuristic --eval-only \
    --num-envs 2 --num-eval-episodes 6 \
    --work-dir output/uipc_manip --run-name heuristic_cloth_drag
```

Metrics, trajectories, and checkpoints land under `output/`, which is ignored.
