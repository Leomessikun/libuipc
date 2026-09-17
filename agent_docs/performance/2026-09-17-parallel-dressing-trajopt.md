# Parallel IPC dressing trajectory optimization

Status: implementation, 12 focused tests, native pilot, throughput comparison and
independent five-control validation complete. All jobs finished.
This is a derivative-free shooting optimizer, not a new RL algorithm or a complete
contact adjoint. No policy training is launched by this module.

## Implementation

`python -m uipc_manip.parallel_trajopt` searches short gripper action sequences with
the cross-entropy method (CEM), using the existing native CUDA IPC dressing world.
All contact transitions are evaluated by the forward simulator; there is no
fixed contact schedule, learned dynamics, SciPy factorization or Hessian export.

CUDA Torch tensors perform candidate generation, interpolation of temporal control
knots, action projection, risk-adjusted ranking and elite distribution updates.
The IPC world places each dressing copy in its own collision subscene and solves
all copies together on the GPU. It still shares global convergence/line-search
decisions. These are physically separated copies, not independent CUDA solvers.
Time steps within each candidate remain sequential.

This is **not entirely GPU-resident**: the existing environment uses NumPy for
controller acceptance, target updates, metrics and parts of observations; native
retrieval and snapshot restoration also cross the host boundary. Plan banks cross
to the host once per evaluator call, not once per action. Improving the remaining
boundary requires end-to-end measurement, not declaring that a CUDA simulator
makes the complete Python environment GPU-only.

## Comparison and objective contracts

- The source checkpoint's architecture and physics settings are retained. The
  runner currently requires unaugmented observations, a single-frame policy and
  the existing controller that ignores x rotation.
- Every copy executes the same saved action prefix. Positions can still differ
  because of numerical/forward variability; the starting spread is reported.
- The entire native batch is snapshotted. Every candidate is evaluated on **every
  saved slot**, using a cyclic assignment without padding. With K candidates,
  B slots, H decisions and R repeats, work is K*B*H*R physical decisions. Compare
  B=1,R=4 with B=4,R=1 for equal numbers of candidate evaluations.
- Every restored slot retains its own state; a low-coverage slot cannot be omitted
  from a candidate's score. This controls assignment bias, not all numerical
  sensitivity or global convergence coupling.
- Each candidate is clipped to the native action box and its translation and
  effective rotation norms are capped by the reference at **each decision**.
  An optimized route cannot gain by increasing command magnitude. This also means
  the search cannot introduce rotation at a reference step with zero rotation.
- Search score is minimum upper-arm coverage over the final four decisions plus
  0.1 times final sleeve-opening projection on the elbow-to-shoulder axis, normalized
  by upper-arm length. Across saved slots/repeats, subtract one population standard
  deviation from the mean. Reject a candidate if any decision has invalid grasp or
  nonfinite geometry. This is a search surrogate, not completed-dressing success.
- CEM keeps the original plan and its incumbent in the population. The first bank
  therefore includes duplicate reference plans, useful for observing variability.
  Candidate 2 of that first bank is fixed before scoring as the random-route control.
- After selection, evaluate reference, CEM, magnitude-capped scaling, fixed random
  and **closed-loop SAC from the branch state**. The first four execute their plan
  and then SAC; the fifth uses SAC throughout. The capped scaling control can
  change direction when individual action coordinates clip.
- Validation reports every final coverage, grasp history and last-12-decision
  sustained success. It does not select another plan after observing validation.
  A finite tail is not a full-episode or hardware robustness result.

The complete temporal contact derivative remains unimplemented. This path tests
the value of forward IPC trajectory search without depending on that derivative.
It does not demonstrate that CEM is better than equal-budget random search, that
IPC is better than another simulator, or that a learned policy benefits.

## Reproduction

Use the existing Release build and Genesis environment. Choose fresh output paths.

```bash
env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.parallel_trajopt \
  --checkpoint output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt \
  --out output/uipc_manip/parallel_trajopt_20260917/run \
  --batch-size 4 --population 8 --iterations 2 --approach 60 --horizon 12 --tail-steps 60
```

`benchmark_bank.npz` contains the first candidate bank and exact approach commands.
`plans.npz` contains selected/control plans and approach commands. `result.json`
contains the config, initial state spread, per-candidate/per-slot decision traces,
restore checks, physical work counts, wall times and validation outcomes.

For a matched bank throughput comparison, use the same command with
`--benchmark-bank <run>/benchmark_bank.npz`, fresh outputs, and respectively
`--batch-size 1 --repeats 4` and `--batch-size 4 --repeats 1`. These replay the
same commands but do not guarantee identical physical trajectories across batch
sizes, because of the known forward/numerical variability. Run sequentially to
avoid GPU contention. Native timers include environment work and restoration;
total command time also includes world construction and the approach.

`--validate-bank <run>/plans.npz` replays a saved selection with all five controls
without repeating optimization. It does not retune or select using validation.
`--tail-steps` must leave the rollout strictly before the environment's auto-reset.

## Verification

```bash
env PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m pytest \
  python/uipc_manip/tests/test_parallel_trajopt.py \
  python/uipc_manip/tests/test_physics_gradient_trajopt.py -q
```

12 tests pass, including actual CUDA/CPU agreement, action magnitude constraints,
balanced candidate accounting, rejection of invalid/nonfinite outcomes, closed-loop
control/tail execution and learning a temporally varying synthetic target. The
synthetic learning test establishes optimizer mechanics, not physical success.

## Native evidence

Artifacts: `output/uipc_manip/parallel_trajopt_20260917/`.
Checkpoint: `dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt`.
Cell: tshirt_26/body 14046, 60-decision approach, 12-decision plans, seed 1097.
These are development states, not held-out generalization.

### Pilot

`batch4/`: eight candidates, four copies, two CEM generations, then four controls
with 60 additional SAC decisions. 2,208 physical decisions, 397.50 s including
setup. Search banks took 84.48 and 101.44 s for 384 decisions each. The best
second-generation search score was .20715 versus reference .17199.

| Pilot control | Mean coverage after plan + SAC continuation |
|---|---:|
| Reference | .57833 |
| CEM selected plan | .57308 |
| Capped scaling | .57618 |
| Fixed random | .56736 |

All four replicas of every control retained valid grasp; all scored 0/4 sustained
successes. The search-score advantage did **not** become a continuation advantage.
This pilot preceded the final cyclic scheduler and explicit closed-loop SAC control;
the committed path is checked by the bank benchmarks and independent validation.

### Matched saved-bank throughput

Both runs replay the pilot's exact saved prefix and first candidate bank. Run
sequentially, with no overlapping native experiment. Each evaluates 8 candidates
4 times for 12 decisions: **384 branch decisions in either case**.

| Run | Slots | Repeats per slot | Branch time | Transitions/s | Total command time |
|---|---:|---:|---:|---:|---:|
| `serial_bench/` | 1 | 4 | 163.19 s | 2.353 | 196.62 s |
| `parallel_bench/` | 4 | 1 | 113.37 s | 3.387 | 182.24 s |

Observed branch throughput improves **1.439x**. Setup and approach reduce the
whole-command speedup to **1.079x**; the parallel command performs more approach
work (240 versus 60 slot-decisions) and builds more cloth copies. Neither value
is an established policy-training speedup. Retaining a world can amortize setup,
but that benefit is not measured by this standalone runner.

The serial branch starts at coverage .01790; parallel copies start at
.00799/.01114/.01014/.04634. Thus identical commands do not guarantee identical
physical paths across batch sizes. This is an end-to-end observed workload
comparison, not an isolated CUDA kernel scaling claim. During native execution,
GPU utilization samples were 88–99%; those samples are not an average, SM occupancy
measurement or evidence that every GPU resource is saturated.

### Independent five-control validation

`validation/`: fresh four-copy world, the saved selected/control trajectories, and
an explicit closed-loop SAC baseline. No additional optimization or selection
uses these outcomes. 1,680 physical decisions including approach, 249.18 s.

| Control | Mean final coverage | Range over four saved states | Sustained valid successes |
|---|---:|---:|---:|
| Reference plan + SAC | .58330 | .57248–.61057 | 0/4 |
| CEM plan + SAC | .57704 | .57063–.58450 | 0/4 |
| Capped scaling + SAC | .57887 | .56626–.60341 | 0/4 |
| Fixed random + SAC | .55891 | .55377–.56541 | 0/4 |
| Closed-loop SAC throughout | .57259 | .56243–.59438 | 0/4 |

All 20 continuations maintain valid grasp and have zero controller collision or
tether rejections; maximum tracking error is .00795 m. The CEM-minus-SAC coverage
differences are -1.657, +.571, +2.207 and +.660 percentage points. Their mean
+.445 percentage points is not a robust established improvement. CEM also loses
to the fixed reference plan's mean. These 72-decision continuations stop at
episode decision 132; they are not complete 300-decision episodes.

The four native commands total **4,956 physical decisions and 1,025.53 s** including
construction/approach. All completed without simulation errors. `summary.json`
and `comparison.png` in the artifact root retain the aggregation and plot.

The implementation demonstrates parallel search and an observed throughput gain.
It does not yet justify training on the optimized corrections: search-score
improvement does not reliably survive resimulation and policy continuation.
Prioritize the forward/recovery variability and the search objective/horizon
before spending a larger policy-training budget. No production SAC weights or
gradient path were changed.
