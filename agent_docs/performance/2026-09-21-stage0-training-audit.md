# Stage 0 audit: repair the experiment before spending the remaining budget

Date: 2026-09-21. Audited the two running arms under
`output/uipc_manip/stage0_20260921/{control,treatment}`. No new physics training
was launched, and the running processes were not stopped or restarted by this
audit. Editing Python source does not repair code already imported by those
processes.

## Measured state at vector step 2,000

| Quantity | Control | Treatment |
| --- | ---: | ---: |
| Replay transitions | 30,000 | 30,000 |
| Simulated training decisions | 50,000 | 50,000 |
| Learner updates | 29,935 | 29,935 |
| Logged elapsed seconds before this evaluation | 7,536.6 | 7,553.0 |
| Learned multiplier | disabled | .24219 |
| First-violation edges in saved replay | 2 | 2 |

Read the arrays in `checkpoints/replay_0002000/replay.npz`, not just sampled
training logs: both buffers contain 108 already-violated observations and only
two rising flag edges. The mean stored cost is .00006667. These edges inherit
the timing bug below; they are not an independently certified episode failure
count.

At evaluations 500, 1,000 and 1,500, both arms had zero final and peak forearm
and upper-arm coverage, zero success and reported grasp-valid rate 1.0. The
largest tracking errors across those evaluations were 9.870 mm (control) and
9.970 mm (treatment), both below 20 mm. Thus the terminal-only validity bug did
not change these particular all-valid evaluations. At step 1,500 the returns
were -50.647 and -51.209. Improvement in negative reward has not produced
measurable arm coverage. Only 11.1% of the planned replay budget had completed;
zero success alone is not proof of a failed learner.

The step-2,000 evaluation completed during the audit: returns improved further
to -36.549 and -36.012, still with zero peak/final forearm and upper-arm coverage,
zero success and all tracking maxima below tolerance. This does not change the
interpretation above.

## Confirmed defects and changes

1. **The launcher changed the physical task.** It omitted `--action-repeat 6`.
   Both saved run configurations use repeat 1, whereas the warm-SAC checkpoint
   and completed feasible-segment experiment use repeat 6. With the same .15 m/s
   speed cap and 1/60 s timestep, the per-axis decision limit is 1.443 mm versus
   8.660 mm. Horizon 300 is five versus thirty simulated seconds. This is not
   evidence that dressing is impossible in five seconds; it means the experiment
   does not test the setting that motivated it. The launcher now pins repeat 6
   and uses `stage0_20260921_corrected` to preserve original artifacts.
2. **The cost was attached to the following action.** `dressing_env.step`
   encoded `obs` before updating `_violated`. Replay derives cost from the
   observation flag's rising edge, so the breaking action was unpenalized and
   the next action charged. A first violation at the terminal decision was lost
   altogether. Encoding now happens after updating all flags and before saving
   terminal observations/resetting. CPU tests exercise the real step method with
   a stub solver, including a transient first-substep violation that recovers by
   the final substep, terminal violations, and both reset modes.
3. **Evaluation and selection did not implement the stated endpoint.** Evaluation
   read the final `grasp_valid` only; recovery could erase an earlier failure.
   It now remembers all decisions. The legacy `valid_grasp_success_rate` remains
   final geometry plus whole-episode validity; the new
   `valid_sustained_success_rate` requires the last 12 upper-arm coverage values
   all at least .7 and whole-episode validity. Short or interrupted episodes
   cannot certify it. `mean_validity_weighted_coverage` and constrained success
   are also split into training/held-out summaries. With `--constraint-objective`,
   both arms now select checkpoints by these constrained metrics before the old
   geometric tie-breakers. Earlier unconstrained selection stays unchanged.
4. **Checkpoints omitted the learned multiplier.** Saving/loading now preserves
   `constraint_lambda`. Resuming constrained training from legacy checkpoints
   without it is rejected; actor-only evaluation remains possible. This omission
   did not cause the current uninterrupted curves, but would invalidate a resume.
5. **A fixed positive penalty was silently disabled.** The real update path used
   the penalty only when its learning rate was positive, even if its initial
   multiplier was positive. It now applies fixed penalties too. This was not
   active in the current runs, whose initial multipliers are both zero.

The added regressions reproduced defects 2, 3, 4 and 5 before their fixes.
Original replay is not a clean corrected-run initialization: its observations
and cost timing reflect the old task and instrumentation. Preserve it as an
audit artifact rather than silently treating it as corrected data.

## Remaining objective limitation

The undiscounted episode sum of a correctly aligned first-violation cost is an
indicator; its expectation is the episode violation probability. The learner
instead updates lambda from **mean replay-transition cost**, then fits a single
discounted reward critic with gamma .995. Neither quantity is that probability.
For complete equal-length episodes, the first is roughly the episode failure
rate divided by the horizon; this collector also resets partial episodes for
evaluation, so multiplying every minibatch by 300 is not an exact repair.
Discounting makes a violation at decision 300 worth about .223 of one at decision
1. Finite lambda and its cap add another approximation. Zero budget avoids a
positive-budget unit mismatch but does not remove replay staleness or discounted
credit assignment.

This audit corrects the CLI/config descriptions; it does not silently replace the
learning algorithm. A rerun must explicitly choose either a discounted penalty
baseline or a properly specified finite-episode constraint learner. An exact
chance-constraint claim needs an episode-level estimator and compatible cost
value/termination semantics. The original protocol's .998333 discount calculation
was also not the actual run setting (.995).

## Recommendation and validation gate

Do not spend the remaining budget on these unchanged processes. Their mismatch
and instrumentation defects prevent the intended Stage 0 interpretation. A
fresh, separately named corrected run first needs a bounded validation: confirm
the physical configuration, demonstrate progress with the known controller, and
check a breaking transition's info cost against its stored flag edge. Then use a
matched short training pilot before authorizing the remaining long budget.

Both arms presently fail before producing the progress-versus-validity conflict
that motivated Stage 0. These data therefore establish neither that constraints
solve the problem nor that representation/allocator changes are necessary. They
also do not establish that any one bug caused all of the zero-coverage behavior.
The old 4.6-hour estimate cannot be reused: 270,000 admitted transitions require
450,000 training decisions per arm because ten of the 25 slots are held out;
with repeat 6 that is 2.7 million physics steps before evaluation overhead.

Validation: 52 CPU tests passed across `test_constraint_objective.py`,
`test_train_sac.py` and `test_cellplan.py`; 78 related tests passed across
`test_sac_agent.py`, `test_pretrain_offline.py`, `test_pretrain_wang.py` and
`test_train_sac_cells.py`. The existing simulator-error evaluation test emits a
mean-of-empty-slice warning. `bash -n scripts/run_stage0_constrained.sh` and
`git diff --check` passed. No new GPU rollout was used to claim that the corrected
learner now succeeds.
