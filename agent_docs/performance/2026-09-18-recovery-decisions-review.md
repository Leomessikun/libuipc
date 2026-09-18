# Review of the recovery-decision result at 58ed3416

Reviewed 2026-09-18 in response to the owner pointing to
[the recovery-decision record](2026-09-18-recovery-decisions.md).
Scope: source inspection and reanalysis of the two completed saved collections.
No simulator, learner, controller or reward was changed or run. The other two
collections' artifacts were incomplete when inspected and are excluded.

## Research decision

Do not build or scale the proposed state-dependent recovery learner on this
evidence. In these two garment/body cells, one fixed macro captures most of the
measured benefit. The marginal payoff for adapting the macro to each sampled
state is small even before paying for learning and collecting its supervision.
This weakens the direction proposed in the
[CS 285 research assessment](2026-09-18-cs285-research-direction.md).

It does not refute simulator supervision, general recovery learning, or the
broader dressing project. The comparison is restricted to seven macro choices,
one policy, one body, three snapshot times and short continuations. There is
still no end-to-end policy improvement or novel algorithm result.

## Recomputed results

Inputs: `output/uipc_manip/decision_branches_20260918/{t26_14046,t392_14046}/`
`result.json` and `states.npz`, plus the saved `identifiability.json`.
Audit output, including source-result SHA-256 hashes and per-repeat values:
`output/research/recovery_decisions_review_20260918/summary.json`.

To reduce selection optimism, hold out one repeat at a time. Use the other two
repeats to choose (a) the best macro averaged across the cell and (b) the best
macro separately at each state. Score both selections on the excluded repeat.
Average over all three choices of held-out repeat. Both selectors use the same
seven candidates, including the policy reference.

| Completed cell | Policy mean | Selected fixed macro | Fixed macro mean | State-selected mean | Additional state-selection benefit |
|---|---:|---|---:|---:|---:|
| tshirt_26 / 14046 | 0.566515 | lift | 0.578923 | 0.581189 | 0.002266 |
| tshirt_392 / 14046 | 0.602111 | forward | 0.631710 | 0.632699 | 0.000988 |

The selected fixed macro is unchanged across the three folds in each cell.
The original best-fixed versus empirical-oracle comparison was approximately
0.003 and 0.002; excluding the repeat used for scoring makes the estimated
increment smaller. This is a repeated-outcome diagnostic at known states,
**not** evaluation of a learned selector on new states, garments or bodies.
The folds overlap and do not supply independent samples for a naive confidence
interval.

All 1,008 completed branches satisfy the saved whole-branch grasp limit. None
reaches sustained coverage >=0.7. Here "sustained" is the minimum over the last
12 decisions of a 40-decision branch. This does not establish episode success
or failure under a different, repeated-intervention controller. Grasp invalidity
does not explain the low coverage of these particular branches.

## Corrections to the protocol interpretation

### 1. The recovery collector repeats a policy, not an identical command tape

`collect_decision_branches.py::collect` recomputes `agent.act(branch_obs,
deterministic=True)` at every decision of every branch. Fixed directional macros
use equal commands for their first eight decisions, but their subsequent
32-decision policy continuations can differ. The `policy` and `policy_scaled`
arms also react to observations during the first eight decisions.

The saved slot-zero command arrays verify this directly: all 21 macro/snapshot
combinations in each completed cell differ across repeats over their complete
40-decision sequences. Thus the measured spread includes closed-loop policy
feedback as well as simulator/observation variation. This remains an appropriate
comparison of fixed macro interventions followed by a shared policy; calling it
identical-command repeatability is incorrect. The separate predictability script
does explicitly replay a fixed command tape and is a different experiment.

### 2. Direction was not isolated from rotation

`decision_branches.py::macro_action` preserves the policy's rotational command
for `policy_scaled`, whereas fixed directional macros initialize a zero action
and write only its translation. They therefore remove rotation as well as
change translation direction and its eight-decision structure.

The retained Y/Z rotation is nonzero. For slot zero at decision 100, its mean
normalized norm over the first eight commands is 0.228 for tshirt_26's policy
and 0.376 for tshirt_392's policy; the corresponding scaled arms are 0.240 and
0.413. The directional macros have zero rotation. These components are applied
by the controller, even though X rotation is disabled.

Supported conclusion: scaling translation magnitude alone yields less benefit
than the tested directional, zero-rotation interventions. Unsupported conclusion:
"the policy's fault is direction, not step size." Direction, rotation and command
structure require matched controls before assigning the cause. The table does
not establish rotation is the culprit either.

### 3. Repeat range is not a statistical detection threshold

`consequence` defines spread as the mean of the seven within-macro ranges from
three repeats. "Decisive" means the selected best mean exceeds the policy mean
by more than that range. This is a descriptive screening rule, not a calibrated
significance test. The best macro is also selected using those same outcomes.

A mean improvement smaller than individual-run spread can be measurable with
adequate independent repetitions. Conversely, exceeding this range does not
establish statistical significance after selecting among seven candidates.
The repeated-outcome check above supports a small marginal benefit; it does not
prove that benefit is exactly zero or fundamentally unresolvable. Report effect
size and collection cost without converting this heuristic into a noise barrier.

### 4. The predictor screen does not establish generalization or an information limit

`identifiability_probe.evaluate` leaves out one state, not an episode or cell.
Other snapshot times from the same seed/approach remain in training, and both
garment/body cells remain represented. The original report acknowledges the
garment-recognition possibility; the 0.67 observation accuracy should therefore
be treated as a within-collection screen. Compare against a cell-conditioned
constant and hold out whole seed/approach groups before attributing a gain to
understanding contact state. New-body/garment claims need those held out too.

The privileged model's 0.75 top-1 versus the observation model's 0.67 does not
prove that the remaining decision information is invisible to the robot. These
are finite-sample linear probes over different feature sets, not Bayes-optimal
predictors. Their mean regrets are 0.002318 and 0.002400, respectively: the
privileged probe's advantage in the chosen outcome is only about 0.000082.
No causal sensing bottleneck is established by that difference.

The shuffled-label check also uses a fixed middle-grid ridge penalty, whereas
the reported real-label model selects its penalty inside each fold. Permuting
individual states breaks episode/cell structure. It is an informal reference,
not a matched permutation significance test. A formal test must preserve the
appropriate groups and repeat the same model-selection procedure.

### 5. The history response feature is expressed in a moving coordinate frame

`decision_features.history_features` differences the cloth centroid obtained
from tool-relative points. It does not add the change in the tool's world
position from the observation extras. Consequently this feature measures
visible cloth motion relative to the tool, including changes in sampling and
visibility; it is not directly the physical motion of material cloth points per
commanded metre. A stationary world-space cloth centroid changes in this frame
when the tool moves.

Such a relative feature may still be useful. Correct its interpretation, and
include world-frame compensation if the intended test concerns actual garment
motion. The current result cannot reject history-based learning. Five previous
observations plus the current one enter the feature stack; these are not a
matched-capacity recurrent policy experiment.

### 6. Zero restore error validates positions, not every hidden solver quantity

`physics_gradient_probe.restore` calls world recovery and restores environment
bookkeeping, then returns the maximum cloth-position discrepancy. Its zero
result is useful evidence of restored geometry. That scalar does not itself
verify all velocities, friction history, solver caches or the full Markov state.
This audit found no missing-state defect; equally, the reported scalar alone
cannot rule one out or prove that physical chaos caused the observed variation.

## What to do next

First incorporate the two additional cell results when they are complete, using
the same distinctions above. Do not launch more branch collection merely to
train a selector whose observed advantage is tiny. Preserve all original data.

If a further native experiment is justified, make it a bounded **complete-episode
comparison of simple controls**: unchanged policy, policy with matched treatment
of rotation/magnitude, and a fixed recovery intervention under a common trigger
and action budget. Select the macro/trigger on development cases, then freeze
them for held-out evaluation. A fixed macro chosen separately using each test
cell's outcomes is an oracle control, not a deployed solution. Inspect the saved
geometry and report retained completion and grasp validity.

Repeated interventions might compound or might undo their earlier gains;
the current short branches do not answer that. If simple interventions improve
complete dressing, they provide a useful control and potentially better training
experience, not automatic algorithmic novelty. If they do not, stop scaling this
macro-based recovery recipe. Revisit the broader objective only from a measured
bottleneck rather than declaring another new learner from prediction scores.

No running collection was interrupted, no production analysis or feature code
was changed, and no new policy-performance claim follows from this audit.
