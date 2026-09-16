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
Results and validation are recorded below after both bounded runs complete.

The SAC control completed all 2,400 new transitions and updates without a
simulator/build error. Mean final coverage rose .15164 -> .51295; per-body final
coverage was .60520 and .42070. Success remained 0/2 at the .7 threshold. Simulation
took 312.51 s, updates 77.61 s, and the two evaluation rounds 152.39 s. The final
replay/checkpoint save took another 22 s. This is one short continuation with two
development cells; retaining the old learning state did not itself prove the
cause of improvement. The matched IPC-actor run is pending at this stage.

Validation of this implementation: 28 focused CPU tests passed, including replay
pairing/reward rejection, remaining-budget continuation, bounded proposals,
independent confirmation, and input-gradient correctness/isolation. Both native
verified dressing experiments and the full SAC continuation completed. Broader
policy generalization and a final held-out test remain unmeasured.

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
