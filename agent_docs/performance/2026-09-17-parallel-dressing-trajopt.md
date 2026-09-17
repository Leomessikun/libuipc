# Parallel IPC dressing trajectory optimization

Status: implementation and 12 focused tests complete; native measurements pending.
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

Pending. A four-copy pilot is running with finite optimization and validation
budgets. Do not infer a policy gain or parallel speedup from GPU utilization.
