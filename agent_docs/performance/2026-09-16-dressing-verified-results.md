# Dressing correction test and training throughput

Date: 2026-09-16. Scope: actual assistive dressing, using existing data and the
existing dense/plain checkpoint. This supersedes the unimplemented status of the
diagnostic in [the redesign proposal](2026-09-16-dressing-rl-redesign.md), not its
requirement for controlled full-episode learning results. No cloth-drag job was
restarted.

## Implemented experiment

`physics_gradient_finetune --verified` now tests the proposed interface before
spending a large online training budget:

1. Restore an elbow state and collect four consecutive policy anchors. Verify
   native position recovery within 10 micrometres and restore observation RNG
   state for the anchor branches.
2. Query the last-frame IPC value gradient at the current policy action; compare
   SAC's action gradient and a random direction. The IPC signal is the historical
   restricted local surrogate, not a complete differentiable dressing decision.
3. Bound translation and rotation corrections separately to .25 normalized
   action units. Change the first action only, then follow the same frozen
   policy for a total of 12 decisions. Two reference branches estimate repeat
   variability. Accept only if two candidate repeats both beat their references
   by `max(.01, 2 * abs(reference_return_difference))` and preserve terminal
   upper-arm coverage within 1e-4. Returns here use raw task rewards.
4. Fit SAC-only, IPC, SAC-direction and random-direction actor copies for 100
   update attempts, with the critic frozen. This diagnostic uses binary accepted
   weights and correction-loss coefficient 1. A cumulative .25 action-norm guard
   on the four anchors rolls back both parameters and Adam state after an
   oversized update. This is a sampled guard, not a global policy guarantee.
5. Evaluate every actor twice from the available elbow/stall/passed snapshots.
   Save targets and explicitly labelled actor-only checkpoints.

This is a limited implementation of the proposal. It does not integrate accepted
corrections into online replay, store complete branch transitions, fit a recent
correction buffer, enforce a guard on broad replay states, or implement a new
Bellman update. The deployment observation interface remains unchanged.

## Result: no accepted IPC targets

Both cases use
`output/uipc_manip/abl_dense_s1/checkpoints/checkpoint_00125016.pt`, seed 0,
the original dressing constraints, and the settings above.

| Garment/body | IPC accepted | SAC accepted | Random accepted | Measured query/verification/evaluation transitions |
| --- | ---: | ---: | ---: | ---: |
| tshirt_26 / 14049 | 0/4 | 0/4 | 0/4 | 628 |
| tshirt_392 / 14046 | 0/4 | 0/4 | 0/4 | 748 |

The gate abstained on all eight anchors. Do not lower the gate after seeing this
result or present the resulting actor fits as IPC learning. With all weights
zero, every trained arm reduces to the same SAC-only objective. Small differences
between these copies include numerical training/rollout variation. On tshirt_26,
the local SAC fit raised elbow coverage .07561 to .09377; on tshirt_392 it lowered
elbow coverage .01250 to zero and stall coverage .14772 to .11955. These are short
development-state probes, not dressing success rates.

Logged verification/evaluation rollout costs were 129.09 s and 199.67 s; gradient
queries including their one forward decision cost 1.17 s and 1.20 s. The counters
exclude world setup, expert approach, and anchor collection. Each 100-attempt
actor fit took roughly one second. Repeated full simulator branches dominate the
proposed verifier's cost. The current code explicitly records that cost scope.

This rejects spending a large training budget on this particular small,
first-action correction scheme now. It does not show that all IPC gradients are
useless: the action radius, first-action intervention, short horizon, critic,
and two garment/body development cases limit the conclusion. Neither historical
elbow gradient walks nor this experiment prove a novel general RL algorithm.

Artifacts under `output/uipc_manip/dressing_redesign_20260916/`:
`verified_26/verified.json`, `verified_392/verified.json`, their
`verified_targets.npz`, actor-only checkpoints, and corresponding `.log` files.
The report is written incrementally so incomplete runs remain identifiable.

## Why dressing training is slow

On the otherwise idle RTX PRO 6000 Blackwell, the same tshirt_26/body14049 expert
approach was measured for 100 decisions at each batch size. Each decision retains
six IPC substeps. These are homogeneous clones, not mixed garment training.

| Environments | Environment wall time | Simulator transitions/s | Time inside `_sim_step` |
| --- | ---: | ---: | ---: |
| 1 | 14.249 s | 7.02 | 95.94% |
| 8 | 43.729 s | 18.29 | 97.27% |
| 24 | 135.547 s | 17.71 | 98.18% |

Eight environments amortize overhead relative to one; increasing to 24 does not
improve this measurement. These single measurements do not establish a universal
optimal batch size. Existing production training already used 24 environments,
so the 2.61x ratio between eight and one is not a newly delivered training speedup.
Eight is used for the bounded continuation tests. GPU samples during actual
dressing runs were around 91–99% busy; that is not an SM occupancy measurement.
Unused VRAM is not sufficient evidence that a larger shared IPC system runs faster.

Artifacts: `batch_profile.json` and `batch_profile.log`. The reusable
`scripts/profile_dressing.py` measures the same method boundaries. Its optional
`--timer-steps` instruments a separate trailing window because native timers add
GPU synchronization overhead. Timings of nested methods are inclusive.

### Native bottleneck and concurrent worlds

The native instrumented window at decisions 101–103 comprised 18 IPC frames and
41 Newton iterations. Pipeline time was 1.568 s: disjoint DCD/trajectory candidate
detection scopes totalled .8765 s (55.9%), global linear-system build/solve .4046 s
(25.8%), and FusedPCG within that scope .1677 s (10.7%). This small expert window
points to collision search as the larger target; it does not profile every late
training stall. The configuration already uses MAS and conditional CUDA-graph
PCG. We did not reduce simulation tolerances, collision checks, or substeps.

A single eight-slot process connected to the existing MPS server measured 18.045
transitions/s including action selection. Two concurrent eight-slot processes
measured **29.112 transitions/s combined**, a **1.61x aggregate throughput ratio**.
The pair completed 1,600 transitions over the union of their measured windows,
54.960 s; initialization is excluded. Their individual simulation rates were
14.962 and 14.567 transitions/s. Each individual process slowed down, but the
GPU completed more total work per second. No native error was reported.

This was one homogeneous expert benchmark, not learning or a shared-policy
asynchronous collector. Use it as evidence for a small number of concurrent
independent experiments, not a promise that one learner now trains 61% faster.
The MPS server already existed with a default active-thread percentage of 100;
the test only connected two clients and did not change global server settings.
Artifacts: `mps_single.json` (including native timers), `mps_pair_0.json`,
`mps_pair_1.json`, and `mps_comparison.json`. Start two profiler processes with
distinct output paths and `CUDA_MPS_PIPE_DIRECTORY` pointing to the existing
server; do not run unrelated workloads during this comparison.

## Continuation repair and comparison

The historical positive physics/control pair discarded Adam state and the old
replay. `pretrain_wang` now offers `--init-optimizers` and `--init-replay` alongside
`--init-from`, so branches can preserve those sources of learning stability.
Replay dimensions, checkpoint/replay transition counts, and reward scaling are
checked. Other SAC configuration changes remain rejected when restoring Adam,
except the existing allowlist of runtime physics-update settings.

The total transition target includes historical replay. Evaluation/checkpoint
schedules start above that count. Ordinary `resume` remains the way to continue
the same experiment and its run bookkeeping; initialization starts a new world
draw and run directory. Weights-only initialization keeps its previous behavior.
The production physics log also now separates cumulative `physics_s` from the
last query's `physics_query_s`; previously the latter overwrote the former.

Input-gradient queries in `physics_gradient_actor` now use `autograd.grad` rather
than computing and accumulating all actor/critic parameter gradients. Tests
verify the analytical input derivatives and preservation of existing parameter
gradients. No end-to-end speedup is attributed to this small change without a
matched measurement.

The bounded SAC and IPC-actor continuation comparison uses the same source
checkpoint/replay, seed 1, dense/plain architecture, eight training poses drawn
from region 13, tshirt_26, no observation augmentation, and 2,400 new transitions
(125,016 -> 127,416). Full 300-decision evaluations before/after use bodies 14046
and 14049. These two repeatedly inspected cells are development evaluation,
not a final held-out test. The physics arm changes only the actor weight to .5.
Both runs completed all 2,400 new transitions and updates without a simulator or
build error. Their final evaluations used the same seed block (1097, 1098).

| Arm | Initial coverage | Final coverage | Final success | Training-loop seconds | Simulation seconds | Update seconds | Physics-query seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAC | .15164 | .51295 | 0/2 | 390.8 | 312.5 | 77.6 | 0 |
| SAC + historical IPC actor, weight .5 | .15018 | .36100 | 0/2 | 592.2 | 475.6 | 78.7 | 37.2 |

The table excludes setup, evaluation and checkpoint saving. SAC's two evaluation
rounds took 152.39 s; the IPC arm's took about 140 s. Each final replay/checkpoint
save took 22 s. SAC's final per-body coverage was .60520 and .42070. Neither arm
reached the .7 success threshold. Average stochastic training-episode returns
were 197.44 for SAC and 69.25 for IPC. Final deterministic evaluation returns
were 236.25 and 184.53. About 163 of the IPC arm's extra 201 seconds occurred in
forward simulation, beyond the direct gradient-query cost.

All 2,400 new physics labels were marked valid by the historical production gate;
they comprise only 1.8836% of the 127,416-row replay. The initial replay has no
physics labels. This is an important limit of a short warm-start comparison.
The fixed physics coefficient was 3.10191 after gradient-norm calibration. The
last nonempty physics-batch statistics may persist in the merged training log;
use the saved replay's validity array to count label coverage.

In this run SAC improved more and trained faster than the existing IPC term.
One short continuation with two development cells does not establish a general
ranking or prove that retained learning state caused the improvement. The same
source checkpoint's initial returns also varied substantially (52.80 vs 121.93),
despite similar final coverage. Additional fixed-seed checkpoint evaluations
are recorded separately to check endpoint repeatability.

### Fixed-seed endpoint rechecks

Two further full 300-decision evaluations per checkpoint used the same two bodies
and seed block (1097, 1098), on an otherwise idle GPU, with no learning. Including
the original final evaluation gives three rounds, six episodes per arm:

| Checkpoint | Coverage by round | Mean coverage | Mean episode return | Success |
| --- | --- | ---: | ---: | ---: |
| SAC | .51295, .46957, .43789 | .47347 | 232.25 | 0/6 |
| IPC actor | .36100, .37391, .38411 | .37301 | 195.03 | 0/6 |

All repeats completed without simulator errors. These repeated evaluations show
the same ordering and substantial numerical/path variation. They are **not six
independent configurations or three training seeds**, and do not establish a
general statistically significant advantage. Both policies remain below complete
dressing success on these cases. Artifacts: `policy_repeats.json`, its `.log`, and
`recheck_policies.py` under the experiment output directory. The trained full SAC
checkpoints are `warm_sac/checkpoints/checkpoint_00127416.pt` and
`warm_physics/checkpoints/checkpoint_00127416.pt`; their source 125k training cost
must not be omitted from any claim of training from scratch.

## Decision after the tests

Keep the replay/optimizer-preserving SAC continuation as the measured baseline.
Do not extend the failed small first-action correction prototype or launch a large
IPC-actor sweep based on this pilot. The useful engineering changes are learning-
state reuse, falsifiable finite-correction tests, accurate cost logging and a
reproducible GPU scheduling measurement. A new RL algorithm with a demonstrated
advantage is not finished.

Before another online IPC variant, require proposals that produce repeatable
finite elbow improvements on development states. Existing successful trajectory
segments can supply candidate multi-action behaviors, but a sequence-based actor
target and its continuation must be evaluated consistently; the failed one-action
test is not evidence that simply increasing the horizon will work. A shared-policy
multi-process collector is a separate implementation task, justified for further
measurement by the 1.61x aggregate simulation result. All bounded training,
profiling, and endpoint-recheck jobs launched in this pass completed; none was
left running.

Validation of this implementation: 28 focused CPU tests passed, including replay
pairing/reward rejection, remaining-budget continuation, bounded proposals,
independent confirmation, and input-gradient correctness/isolation. Both native
verified dressing experiments and the full SAC continuation completed. Broader
policy generalization and a final held-out test remain unmeasured. The matched
IPC continuation and native single/concurrent-world profiler runs also completed.

## Reproduction

Use the native build and Genesis virtual environment, for example:

```bash
export PYTHONPATH=build_raw/python/src:python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib
GENESIS_PY=/home/ge47gax/kun/genesis-world/.venv/bin/python
$GENESIS_PY -m uipc_manip.physics_gradient_finetune --verified \
  --garment tshirt_26 --body 14049 \
  --checkpoint output/uipc_manip/abl_dense_s1/checkpoints/checkpoint_00125016.pt \
  --out output/uipc_manip/dressing_redesign_20260916/verified_26 \
  --verified-anchors 4 --updates 100 --horizon 12 --eval-repeats 2 \
  --eval-states elbow stall passed
$GENESIS_PY scripts/profile_dressing.py --envs 1 8 24 --steps 100 --out profile.json
```

For the second cell substitute tshirt_392/body14046. New continuation run commands
are retained verbatim in their run `config.json` and checkpoint `state.json`.
The profile uses no policy by default; `--checkpoint` switches to deterministic
policy actions. It does not measure learner updates or evaluation overhead.
