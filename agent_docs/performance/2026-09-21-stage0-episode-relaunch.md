# Stage 0: corrected finite-episode objective and relaunch

Date: 2026-09-21. The owner authorized stopping the defective runs, resolving the
audit findings and relaunching. Original PIDs 1360583 and 1361270 were terminated
and verified absent; their outputs under `stage0_20260921` remain intact.

## Why retain this experiment

Stage 0 is an objective-alignment baseline, not the proposed novel algorithm.
The motivating warm-policy branches showed coverage gained while violating the
evaluation constraint, but did not establish that an ordinary learner given
the constraint could not do better. A matched control tests that missing premise.
It does not require IPC gradients, a new representation, or a query allocator.
If both arms fail to acquire progress, that remains an acquisition bottleneck;
it is not evidence that pricing feasibility caused immobility or that Stage 2 is
necessary. If the treatment improves valid completion, a benchmark/objective
repair must precede any algorithmic novelty claim.

## Objective, units, and finite horizon

The opt-in `--constraint-episode` mode adds elapsed episode fraction to both
arms' observations (5,385 floats: the previous cloud/scalars, violation flag,
and clock). A clock is needed because the remaining horizon changes future
value. The pre-existing observation protocols retain their dimensions.

Let zero-based decision `t` run from 0 through `T-1`. The environment emits
`c_t = 1` only on the first decision exceeding 2 cm tracking, including a
transient substep violation. At update time, the critic receives

```text
training_reward_t = scaled_task_reward_t - lambda * gamma**(-t) * c_t
sum_t gamma**t * (lambda * gamma**(-t) * c_t)
    = lambda * indicator(any episode violation)
```

Thus the task reward retains its original discount (.995), while a first
violation has the same episode-start penalty at decision 1 or 300. The largest
correction is about 4.48, not an unbounded long-rollout importance weight. The
clock's stored fraction reconstructs the integer timestep by rounding. True
episode endings now store `done=True` and stop Bellman bootstrapping. Both arms
use the same clock, horizon, termination and task reward.

Lambda updates only from each fresh batch of completed **training** episodes:

```text
lambda <- clip(lambda + dual_lr * (mean(episode_violation) - budget), 0, 50)
```

Replay sampling never updates it in this mode. Held-out slots never enter this
estimate. The collector checks that every stored flag edge equals the info cost
and that the sum equals the final absorbing indicator. Partial episodes do not
enter the dual. Evaluation/checkpoint intervals and transition budgets must
align to episode boundaries; garment admission cannot change mid-episode.
Checkpoints preserve lambda and completed-episode count. Branch snapshots also
preserve violation history; a constrained restore with missing history fails.

This fixes the **specified objective and estimator units**, not the general
convergence problem of off-policy neural constrained RL. SAC still learns
approximate critics from replay, the behavior policy changes while collecting
episodes, and a finite capped multiplier does not guarantee zero violations.
The task objective still rewards dressing progress; completion and physical
grasp certification are not implied by that surrogate or by the tracking proxy.

## Fixed comparison before corrected training

- Five garments by five bodies; bodies 14048/14049 held out of training.
- Horizon 300, action repeat 6, timestep 1/60 s: 30 physical seconds and an
  8.660 mm per-axis decision limit.
- Both arms start fresh, seed 1, visible-dual observations without augmentation,
  existing scalar SAC/encoder and .995 task discount.
- Both start lambda at zero. Control dual rate 0; treatment dual rate **1 per
  completed 15-episode batch**, budget zero, cap 50. This is not the old
  replay-update learning rate: 270,000 admitted transitions contain 60 full
  batches, so the multiplier can reach the cap if violations persist. No rate
  search is being hidden as the experiment.
- 270,000 admitted transitions per arm, 450,000 simulated training decisions,
  2.7 million physics substeps before evaluation. Evaluation every 600 vector
  decisions, checkpoints every 1,800 (evaluation also saves a checkpoint).
- Valid completion requires final-12-decision minimum coverage at least .7 and
  no tracking violation anywhere in the episode. Report validity-weighted
  coverage and completion separately, including per-cell and held-out summaries.
  Both arms select checkpoints by the same constrained criteria.
- Output root: `output/uipc_manip/stage0_20260921_episode`. The old runs/replay
  are not used as corrected initialization. Launcher: `scripts/run_stage0_constrained.sh`.

The old 4.6-hour estimate is not a promise for this two-arm, six-substep protocol.
Actual wall time must include both training jobs and the preflight/evaluations.

## Validation and launch record

608 CPU tests passed, 15 GPU tests deselected. The added tests cover equality of
early/late discounted penalties, episode-only dual updates, held-out exclusion,
terminal observations and bootstrap masks, episode-boundary configuration,
checkpoint protocol restoration, and violation history during branch restore.
A pre-existing missing-distance warning in the simulator-error test remains.

The physical preflight completed the existing geometric controller on all 25
cells with the corrected launcher, writing `preflight_heuristic/eval.json`:

| Metric | Result |
| --- | ---: |
| Simulator errors | 0 |
| Cells with positive peak upper-arm coverage | 24/25 |
| Mean final upper-arm coverage | .511388 |
| Geometric final success | 11/25 |
| Whole-episode tracking validity | 11/25 |
| Valid sustained completion | 6/25 |
| Mean validity-weighted sustained coverage | .234110 |

Every reported episode validity flag agreed with its recorded whole-episode
maximum tracking error. These results establish attainable progress and some
valid completions in the corrected environment; they are controller validation,
not results of the newly trained policies. The 21 constraint contract tests also
passed after adding an analysis-loader round-trip for the new clock protocol.

The two corrected jobs were launched at **2026-09-21 18:57:44 UTC**, from commit
`57dfa87d`, fresh and without old replay/checkpoint initialization:

- Control PID **2316255**, log `stage0_20260921_episode/control.log`.
- Treatment PID **2316256**, log `stage0_20260921_episode/treatment.log`.

Paths above are relative to `output/uipc_manip`. `launch.json` in that root records
the exact command, commit, PIDs, timestamp and budget; each arm writes its own
resolved configuration and CSVs. Both run detached and have a finite 270,000
admitted-transition budget. Their first learning updates are checked before the
handoff; episode rates and lambda changes become available after decision 300.

Startup verification: both reached decision 20, 300 admitted transitions and
235 learner updates with finite critic losses, Q values and temperatures. Lambda
is still zero in both, as expected before the first completed episode. Resolved
environment configurations match exactly, and learner configurations differ only
in dual rate. Elapsed training time was 197.1/195.5 seconds, of which 176.6/175.2
seconds were physics and 20.2/20.1 seconds learning. Extrapolating this very early
concurrent throughput gives roughly 49 hours before evaluation overhead for the
18,000-decision budget; this is not a stable ETA, since physics cost changes with
policy/contact state, but it makes clear that the earlier 4.6-hour estimate does
not apply. No simulator failures or nonfinite learning statistics were observed
at this startup check.
