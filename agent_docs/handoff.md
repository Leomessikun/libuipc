# Handoff — Current State of the Repo

## 2026-09-20 — Isolate transfer of the already-successful recovery

The owner again asked to reuse completed experiments. The September 17 teacher
already completed closed-loop recovery with .976--.984 sustained coverage, while
its trained BC student failed the target. Do not frame successful closed-loop
recovery as untested. The SAC-plus-recovery pilot was stopped before treatment;
it is not a completed negative result.

The [transfer audit](performance/2026-09-20-recovery-transfer-audit.md) implements
the missing comparison: original/student prefixes crossed with frozen original,
student and existing teacher continuations, all from matched decision-120 states
through decision 300. **Completed: 4,800 decisions / 682.80 s (11.38 min)**.
On each prefix group: teacher 4/4, original BC 0/4, recovery BC 0/4. Both prefixes
had valid grasp; teacher sustained coverage .97473--.99243. All 24 continuations
fail the separate historical early-turn filter. Two slots and two repeats are
not independent task draws. Approach drift is not the sole transfer explanation:
student fails from the original prefix, and teacher recovers the student's prefix.

Artifacts: `output/uipc_manip/recovery_transfer_audit_20260920/`; all workers
exited. No training or CEM. Read-only analysis reproduces the student's .012806
old-label MSE but finds .057100/.046372 on new teacher trajectories. Moving-command
translation RMS error rises from 2.286 mm on old rows to 5.216/5.353 mm on new
ones, against an 8 mm command cap. Stopped-command errors remain much smaller.
This is a measurable fitting/generalization gap, not proof of observation
aliasing or that lowering MSE alone solves rollout. Command routing checks agree
within 1.15e-6; 10 focused tests and all saved-trajectory integrity checks pass.
Collector implementation: `af120bc6`. The completion stage adds analysis/plot
and final results. This BC transfer study must not substitute for the completed
ordinary SAC diagnosis. DAgger, DART and GPS/MDGPS are relevant existing controls;
generic search plus supervised projection is not novel.

## 2026-09-20 — Fixed-critic comparison and complete-continuation check finished

The owner asked to inspect already-completed ideas and directly evaluate real
gaps. [Evidence map](performance/2026-09-20-action-selection-evidence.md) links
the completed actor/FD audit, CEM, macro recovery, full episodes, duration and
tolerance tests. Do not propose those as unrun. Reanalysis of the existing 960
RAL branches excludes each scoring repeat from selection: joint duration/action
choice gains only .005493 sustained coverage; holding the unmodified action
four steps gains -.000934. No grasp fields exist in that source.

Reopening existing full controls locates tshirt_26's first grasp violations at
decisions 32--50, before upper-arm coverage begins at 61--65; all eight controls
have this ordering. Later instantaneous grasp validity cannot repair the
whole-episode criterion. tshirt_392's eight controls keep grasp.

Completed native evaluation: `uipc_manip.action_selector_audit`, output
`output/uipc_manip/action_selector_audit_20260920/`. It freezes warm-SAC Q and
compares projected-Q-gradient selection with sampled-Q selection in the same
action trust region; all candidates receive repeated physical evaluation.
Eight states, 320 branches, 8,240 decisions / 1,215.83 s. Gradient extraction
increases raw sustained coverage by .000812 on average; sampled-Q selection
changes it by -.012696, strongly influenced by one zero-ending execution.
After zeroing grasp-invalid outcomes the gains are .002207 and .000302.
One gradient state improves >.01 in every repeat; no sampled-Q state does.
All short branches have zero dressing success. Four tshirt_26 snapshots have
already-invalid prefixes, which the scorer retains rather than forgetting.
Initial/terminal observations and commands are saved for future reanalysis.

Independent validation uses new seeds 9301--9304, a fresh world, the same
gradient rule at decision 140, and continuation through decision 300. Artifact:
`output/uipc_manip/action_selector_validation_20260920/`. SAC and gradient+SAC
each score 0/8 (four common prefixes, two repeats); all tshirt_392 executions
preserve grasp but finish at zero coverage. Local improvements do not secure
terminal success under the old continuation. Validation: 3,120 decisions /
250.13 s. Combined: 11,360 decisions / 24.43 min. All workers exited.

Three focused checks and artifact integrity checks pass. Commit `7796cfba`
contains the first collector/reanalysis stage; the completion stage adds the
two-arm full-horizon validation and final evidence. No CEM was run this session;
its 12-action trajectory experiment is historical. No policy was trained.
The evidence does not select gradient removal as the solution. A future
comparison must improve the closed-loop continuation before the relevant
failure, rather than assuming a late single-action gain solves full dressing.

## 2026-09-20 — Owner prioritizes a replacement RL algorithm

The owner clarified that the goal is a new RL structure, not another change to
the existing pretraining pipeline, and asked whether gradients are unsuitable.
[Research assessment](performance/2026-09-20-rl-algorithm-research-assessment.md)
distinguishes simulator derivatives, SAC's learned action-value gradient,
likelihood-ratio policy gradients and supervised network fitting. The evidence
does not identify removing action gradients as the primary solution. The owner
explicitly asked for independent assessment rather than agreement with that
hypothesis. Useful experience, temporal credit, value accuracy and observation
aliasing must compete as explanations; a derivative-free loop is one candidate.

Closest prior art: PI2-GPS (Levine et al.), MPO, RSPI, robust MPO, iCEM and
multi-fidelity RL. A forward-search / supervised-fit loop is established, and
the repository's earlier CEM already failed to establish robust continuation
gains. Wang's original dressing paper uses SAC, while RFCL supplies an
experience-distribution counter-hypothesis. One conditional candidate is a
replacement policy-iteration loop using finite
behaviors, with joint allocation of duration, solver accuracy and repeats;
sampling uncertainty, numerical discrepancy and policy-projection failures must
be distinguished. This joint operator is an unvalidated research hypothesis,
not a certified novel algorithm. Retain multimodal alternatives and verify the
fitted deployable policy, not only the privileged search result.

The proposed discrimination experiments separate local versus sampled action
selection from learned-Q versus rollout scoring, and test the same gradient
learner with valid near-progress restarts and physical-duration controls. Require
independent full-continuation improvement before training on branch labels.
The [previous audit](performance/2026-09-20-training-issue-review.md) supplies
experimental controls, not the research contribution. No simulator or training
was launched; a scalar gradient counterexample was checked on CPU. Owner's
updated scope and request for independent judgment are recorded in `rule.md`.

## 2026-09-20 — Training issue inventory reviewed; causal claims corrected

[Review](performance/2026-09-20-training-issue-review.md) checks the inventory
against code, saved configs/CSVs/branches and primary literature. No simulator,
training or GPU job was launched. CPU-only reproduction:
`scripts/audit_dressing_training_inventory.py`; output and source hashes:
`output/uipc_manip/training_inventory_review_20260920/audit.json`.

The clean state run is genuinely 0/25 but is not a controlled upper bound:
decision rate changed 10 to 60 Hz, episode duration 30 to 15 s, angular-rate
limit 50 to 300 degrees/s per enabled axis, and discount time about 20 to 10 s.
Its 35D geometric summary is not full Markov state. The 12,500-transition
threading figure is a trained actor's peak-event rate, not random exploration.
An actual `wang_progress` counterexample gives task slopes 1/1/5 across the
two joins; continuity does not establish a reward plateau.

Four-step calibration has a real local positive signal: at sigma 1, 16/24 states
improve by >3 pooled SD and two deteriorate by >3 SD. All start on the upper arm;
this is not a diagnosis of the state actor's earlier failure. Top-two gaps do
not measure incumbent-policy advantage. Tight-tolerance repeatability is a
promising local measurement, not a demonstrated learning speedup. Force remains
uncalibrated, not proven intrinsically unusable. Existing evaluator definitions
still disagree. Historical teacher categories are 6 valid-final, 5 invalid-grasp
final, 4 covered-then-lost and 10 never-covered; 15 is not an upper bound.

ADR 0009 requires revision: confidence-gated rollout supervision and allocation
are established prior art (RSPI 2008 / rollout allocation 2010). Do not implement
it on the basis of the previous novelty/flatness claims. Next proposed order:
freeze scoring and physical-time controls; bounded matched-state accuracy test;
completion/grasp-aligned data and objective; matched single-action/held-action/
chunked RL pilot, judged by full-episode improvement per total workstation time.
No claim of repaired training or a new algorithm is made by this review.

## 2026-09-19 — The feasibility question, and the one experiment that answers it

The direction was set by the owner: stop researching the simulator, and decide whether
"train dressing fast on one workstation" has any hope, at minimum cost. Two candidate
gates were considered and one was closed by evidence already in this repository.

**Closed: the metric is not corrupting the signal.** The isolated one-decision zero
rate over 2,424 branch traces and 96,960 decisions is 0.024 %, and the six captures in
`2026-09-19-intervention-and-agent-baseline.md` report zero isolated zeros each. The
4-9 % figure in that record is contiguous marginal states, not spikes, and the record
already says reading them as a measurement artifact was too strong. Nothing to fix.

**Open, and never tested: what the control problem costs when perception is free.**
The 35-float privileged state is documented as critic-only by design
(`dressing_privileged.py`), a `StateActor` exists in `sac.py` but `--actor` did not
accept `state`, and the one asymmetric-critic dressing run ever started died at 13,920
transitions with no checkpoint. No dressing replay on disk carries privileged state.
So no one has measured whether this control problem is learnable at all given perfect
information — every negative result is confounded with a 5,383-float point cloud.

`--actor state` is now reachable: the replay holds whatever the actor reads, the loop
and the evaluator feed the privileged vector, and the terminal row pairs with the
terminal privileged state. Verified end to end on dressing at four slots.

**Pre-registered reading of the run.** Train on exactly the 25 cells it is evaluated
on, perfect state, nothing held out — the most generous setting there is — for the
budget of the longest SAC run on record (270,216 transitions, 26.2 h, peak 3/25, final
0/25). Scored afterwards by the project's coverage-and-grasp rule against the scripted
teacher's 9/25 with the 5 cm shoulder ray and 6/25 without it:

* **13/25 or better** — the control problem is learnable and the remaining problem is
  perception, which is a standard and much cheaper one. The road has hope.
* **9-12/25** — learning with perfect information merely matches a 733-second scripted
  controller. Weak.
* **8/25 or worse** — with free perception and a full budget, learning does not beat
  the script. The road as posed is closed, and that is the verdict to report.

Throughput on this workstation, measured: 40-50 decisions per second, flat in the slot
count, so about 3.5-4.2 million environment steps per 24 hours.

## 2026-09-19 — The diagnostic has a control, and it moves the question

[Record](performance/2026-09-19-decision-signal-to-noise.md). Every negative result on
dressing so far was measured by a diagnostic that had only ever been run on dressing.
It has now been run unchanged on the cloth-drag benchmark of the closed IAQL study —
same backend, same snapshot and restore, a SAC policy that goes from -4 to +13 return
over 20,000 transitions — and it reports 14 of 24 states decisive, rank agreement 1.00
and five different macros winning at different states. The diagnostic does not always
say no.

The comparison says more than the control was built for, and one sentence of it was
wrong when first written. Repeated macros with bitwise identical commands from a
restored state do not reproduce on either task, so the source is the solver's and is
shared — but the two environments do not run the solver alike. Dressing loosens the
Newton velocity tolerance to 0.1 and the conjugate-gradient tolerance to 1e-2 against
library defaults of 0.05 and 1e-3; the cloth-drag environment tightens Newton to 0.001
and keeps the library's linear default. The control task was running a hundred times
tighter in one and ten times in the other. Amplification is therefore not established
as the difference, and the tolerance is now the leading explanation. With one outcome
statistic on both tasks, dressing's decision *consequence* is the larger of the two and
its eta-squared is 0.86-0.96, so telling a bad intervention from a good one is easy
there. Telling the best from the next best is not: that gap is 193.6 repeat standard
deviations on cloth drag against 1.31 and 1.24 on dressing, and it falls below the
noise of a single repeat at 42 % of dressing states against 8 % of the control's. That
is the comparison a learner must win to improve a policy that is already competent.

`scripts/consequence_to_noise.py` is the reusable comparison; it reports eta-squared, F
and the top-two margin together, because the naive consequence-over-noise ratio depends
on where an outcome measure saturates and does not separate the two tasks.

[Record](performance/2026-09-19-tolerance-sets-the-noise-floor.md). The tolerance is
measurable on its own and it moves almost everything. Replaying bitwise identical
commands from an exactly restored state separates by 2.57 mm after twenty decisions and
spreads final coverage by 0.00583 at dressing's settings; at 0.01 and 1e-4 the same
replay separates by 0.019 mm and spreads coverage by 0.00001 — 139 and 583 times lower
— for about 2.5 times the wall time, with no solver code changed and no iteration-cap
warning at either setting. Dressing's best-to-next-best margin is 1.31 repeat standard
deviations at the loose setting against 193.6 on the control; a floor 583 times lower
would put it on the far side of that comparison.

A branch run at the tight tolerance, every other setting matched to
`decision_branches_20260918/t26_14046` (seed 3301, window 8, follow 32, three repeats,
eight slots), is measuring whether the margin actually rises — and whether the raw
consequence survives, since a more accurate solve could instead smooth the contact
until no macro matters. `scripts/feedback_value_profile.py` is written and smoke-tested
(the kick does not break the grasp: 0.38-0.51 cm against the 2 cm limit) but its
default-tolerance run was stopped: it would measure feedback value against a noise
floor now known to be an artifact of that setting.

**Every earlier dressing measurement in this repository was taken at the loose
tolerance** — the predictability horizon, the plateau, the prior-support audit, the
teacher supervision. They are not wrong, but the task they measured includes this
setting, and their negatives should be re-read at the control's numerics before being
cited as properties of dressing.

## 2026-09-19 — Teacher repair closed by measurement; labelling the policy's own states instead

[Record](performance/2026-09-19-teacher-supervision.md). The scripted teacher scores
9 of 25 cells under coverage-and-grasp, and its 16 failures are six complete dressings
disqualified by a grip that slips at decision 111-130 while coverage is still 0.03-0.18,
six that never reach the arm (four of them `tshirt_392`), and four partial.

Two supervisors were built from those failures and both fail. Acting on stalls,
containment and grasp scores 0 of 25 and breaks nine cells while raising grasp validity
from 10 to 21. Measuring the base rates explains it: over the teacher's own episodes,
a tracking error above 1.5 cm fires on 0.0 % of successful decisions and 40.2 % of
failed ones, containment past 0.6 radii on 5.7 % against 15.0 %, and an arc-progress
stall on 35.7 % against 33.8 %, which is no separation at all. A second supervisor
acting only on the grasp, keeping the axial command and dropping the transverse pull,
scores 8 of 25 and fixes no cell: where it fires it fires 150-209 times in 300
decisions, because the tracking error is a state the episode enters and stays in.
So the teacher cannot be repaired by command-level overrides on the signals it exposes.
`scripts/separability_of_failure_signals.py` is the reusable check that should precede
any future rule.

What the prior audit implies instead is now running:
`scripts/collect_policy_state_labels.py` drives the trained policy and records every
decision with the branch study's best macro for that garment, so the states are the
policy's and the actions are the ones measured to help. The manifest marks the dataset
as fit for a behavior model only, since its successors belong to the executed action.
Refitting the behavior flow on it and re-running the reversal audit is the direct test
of whether the useful directions can be brought inside the prior.

## 2026-09-19 — Force ruled out as a signal, the stale IAQL assertion fixed, the teacher repair under test

Three results on `research/flow-latent-steering`, continuing the owner's instruction to
keep solving rather than stop.

**The contact force does not reproduce** ([record](performance/2026-09-19-force-reproducibility.md)).
Identical commands replayed from an exactly restored state disagree by 50-126 % on
summed and peak normal force, about 190 % on friction and 24 % on a contact count,
while the same runs' coverage spreads by 0.007-0.037. Twelve channels are identically
zero at that state. So a per-decision contact force cannot be a policy input, a reward
term, a critic feature or a safety threshold here, and the "settled" channels, which
exist to report only persistent contacts, are worse than the raw ones. Three uses
survive: a contact count, an average over repeats for offline comparison, and a
converged-state readout that this measurement does not test.

**A repaired teacher is under test.** `uipc_manip.dressing_supervisor` overrides the
scripted expert while it stalls with the sleeve on the arm, while the opening's centre
drifts past 0.6 of its own radius from the arm's centreline, or while the tracking
error exceeds 1.2 cm, using the directions the branch study measured to help. A first
smoke run exposed the containment rule firing through the whole approach, where the
opening is legitimately off the axis; both overrides now require the sleeve to be on
the arm. `scripts/evaluate_supervised_teacher.py` runs the expert twice over the same
25 cells and seeds under the corrected five-centimetre shoulder ray;
`expert_baseline --supervised` collects with it, and `scripts/widen_flow_prior.sh`
chains collection, replay, refit and re-audit for when it passes.

**The suite is green again.** `test_iaql`'s assertion that a gated physics batch omits
its keys predated another session's deliberate change to report explicit zeros
(236b21e8); it now follows the new contract, and 560 tests pass.

## 2026-09-19 — New branch `research/flow-latent-steering`: the helpful action is outside the prior

The owner asked for a branch for the flow-reversal-steering ideas and for the
prior-support audit before any latent-space learner.
[Record](performance/2026-09-19-flow-prior-support.md). `uipc_manip.flow_reversal`
carries an action back through the behavior flow's own Euler discretization and reports
the noise it corresponds to, where that noise sits in the prior's standard normal, and
whether a forward pass reproduces the action. Six focused tests.

Measured on the FQL behavior flow (ten steps, six-dimensional action, 7,500 transitions
from 25 scripted-expert episodes, bodies 14045-14047 fitted and 14048-14049 withheld):
in-sample expert actions reverse to noise at percentile 0.025 with reconstruction error
0.034; withheld-body actions to percentile 0.602 with error 0.172 and 4.5 times the
inversion drift, so the field is rougher out of sample. The recovery macros that beat
the policy at 43 of 48 states reverse to noise of length 5.2-5.5 at percentile 1.000,
98-100 % beyond the 99th, with reconstruction error 0.23-0.29. That is not a magnitude
effect: each macro commands 0.924 in normalized units and the expert data's own action
norm has median 0.924.

So the behaviour that improves dressing is not a mode this prior contains, and latent
steering, noise-space cloning and latent-space reinforcement learning would all search
a space without the answer. Widen the prior first, and reuse this audit to check it.

## 2026-09-19 — The policy does not stall: recovery closed, grasp and containment named, agent harness added

[Record](performance/2026-09-19-intervention-and-agent-baseline.md). Three arms from
the same eight seeds over full 300-decision episodes, two cells: the policy alone, the
policy with a deployment-side stall detector handing eight decisions to the garment's
best macro, and the same detector handing them to a random macro. The detector fired
twice in 24 episodes on tshirt_26 and never on tshirt_392, so two arms are the control
repeated; sustained coverage 0.538 / 0.550 / 0.583 on tshirt_26 is run-to-run spread,
and every arm is 0/8 successes on both cells. The policy commands a median 2.72 mm per
decision and the garment follows, so there is no stall to detect.

What the episodes contain instead: tshirt_26 loses the grasp in 24 of 24 episodes,
reaching 30 mm against the 2 cm limit, while coverage plateaus at 0.54-0.58;
tshirt_392 keeps the grasp and loses the sleeve sideways at decision 227-250, with the
opening ring's centre 8.6-9.1 cm from the arm's centreline against its own 9.0 cm
radius. Recomputing the metric from saved geometry (`scripts/capture_policy_geometry.py`,
`scripts/audit_progress_metric.py`) shows the one-decision jumps are that marginal
configuration rather than a bug, with 4-9 % of decisions reading zero while the ring
still encircles the arm, and the long zero stretches genuine (centre 20-23 cm away).

`uipc_manip.agent_harness` gives a program the tool surface the agentic-robotics
demonstrations use — segmented cloud in the tool frame, tool pose, goal direction,
garment motion per commanded metre, rendered views, commands in metres under the
controller's own limits, optional hand-back to the trained policy — with task metrics
behind report() and every call logged. Two programs written in this session reach
0.000 peak coverage on tshirt_26/14046 where the trained policy reaches 0.598 from the
same seed; the first dragged the cuff to the shoulder outside the arm, the second kept
the grasp but never enveloped the fingertip. Ten focused tests.

Next measurements follow from the two named failures: what the grip does in the
decisions before the tracking error crosses 2 cm, and what the opening's lateral offset
does before it leaves the arm, both from quantities a robot can see.

## 2026-09-18 — Recovery-decision result audited; do not scale the selector

[Review of 58ed3416](performance/2026-09-18-recovery-decisions-review.md).
Saved-data reanalysis selects macros using two repeats and scores the held-out
third. Per-state selection adds only 0.002266 / 0.000988 sustained coverage over
fixed lift / forward in the completed tshirt_26 / tshirt_392 cells. All 1,008
branches are grasp-valid; none reaches sustained coverage >=0.7. These are
short-branch outcomes, not a learned policy or full-episode evaluation.

Corrections to the historical entry below: commands are recomputed in closed
loop, not replayed identically; fixed directional macros zero rotation while
policy_scaled preserves it, so direction was not isolated; repeat range is a
screening heuristic, not statistical significance. Leave-one-state-out probes
retain other times from the same approach and do not establish generalization.
The privileged probe's top-1 advantage does not prove missing deployment
information. History centroid differences are tool-relative. Zero restore
position error alone does not validate every hidden solver variable.

This weakens the proposed state-dependent recovery-learning direction. Analyse
the remaining cells when complete before expanding collection. Any next native
gate should compare simple controls over complete episodes with rotation,
trigger, budget and held-out evaluation specified. No physics/training run was
launched or interrupted; collector and learner code are unchanged.

## 2026-09-18 — Counterfactual recovery decisions measured: the action exists, the decision does not

[Record](performance/2026-09-18-recovery-decisions.md). The saved SAC policy is
driven to decisions 100, 140 and 180; the world is snapshotted (restore error
0.00e+00 m throughout) and seven macros each replace the policy for eight decisions
before the same policy resumes for 32, three identical-command repeats per macro,
eight seeds per snapshot. Two cells done: 24 states and 504 branches each, 21,600
decisions in 40.8 and 43.5 minutes, no simulator error; `tshirt_68/14046` and
`hospital_gown/14046` are running.

Premise one holds: at 43 of 48 states some macro beats the policy's own continuation
by more than the repeat spread (median +0.013 on tshirt_26, +0.027 on tshirt_392;
10 % and 28 % of the gap to 0.7 coverage), and the ranking agrees across repeats
(top-1 0.75 and 0.90). The fault is direction, not magnitude: the policy's own
direction at the macros' 8 mm gains 0.001 and 0.006 against 0.012 and 0.030 for the
best macro. The gain halves with every 40 decisions of delay.

Premise two fails: the winner is nearly constant per garment (`lift` at 17 of 24
states, `forward` at 19 of 24), and a single fixed macro is worth 0.002-0.003 less
than a per-state oracle, inside the noise. The choice is identifiable from the
deployment observation (leave-one-state-out top-1 0.67 against a 0.44 constant
baseline and a 0.42 shuffled control, privileged 0.75, history 0.62), but acting on
it saves 0.005 of coverage. So a method that spends simulation deciding which
recovery to take at which state would solve a problem that is not there at these
states; the open question is whether repeated interventions compound over a whole
episode, which `uipc_manip.recovery_intervention` (stall detector reading the
garment's motion per commanded metre, with control and random-macro arms) is built
to answer and has not yet run.

Pre-existing failure on HEAD, from another session's 236b21e8, not from this work:
`test_iaql.py::test_state_batch_feeds_the_actor_term_from_the_same_label`.

## 2026-09-18 — Predictability of garment-arm contact measured; decision-branch collector added

The owner asked for the open scientific questions of this area rather than another
learner, then approved attacking them. First answer recorded:
[how predictable garment-arm contact is](performance/2026-09-18-predictability-horizon.md).
`scripts/measure_predictability_horizon.py` replays one command sequence from an
exactly restored state (restore error 0.00e+00 m over 147 restores), five times
identically and five times each with 0.87 / 8.7 / 87 micrometre first-command
perturbations, 40 decisions, at 3-4 snapshots in two cells; 6,220 decisions /
38.7 min, no simulator error. Identical commands separate by 0.16 mm after one
decision, millimetres by ten and centimetres by forty (growth 0.004-1.258 per
decision, state dependent); the perturbation's size changes nothing, so consequences
of commands differing by under about one percent of a decision's translation are
inseparable. But the scales differ: 98 % of vertices move apart by more than a
millimetre while the coverage ratio spreads by at most 0.013 and the opening centroid
by 4.4 mm. Contact with the arm suppresses divergence; free-hanging fabric amplifies
it. This is the mechanism behind the one-decision gradient horizon and the failed
IPC-labelled updates, and it sets the repeat count for any counterfactual label.

New `uipc_manip.decision_branches` and `uipc_manip.collect_decision_branches` collect
counterfactual recovery branches at states the saved SAC policy visits: snapshot,
run one of seven macros (policy, policy_scaled, forward, retreat, outward, lift,
retreat_outward) for a short window, hand back to the same policy, repeat with
identical commands. `scripts/analyze_decision_branches.py` reports consequence
against the policy's own continuation, repeat spread, ranking agreement, and whether
one fixed macro would do as well as choosing per state. Eleven focused tests pass.
Two collections are running; no result is claimed yet.

## 2026-09-18 — Full current-course text review and narrowed research question

[CS 185/285 research assessment](performance/2026-09-18-cs285-research-direction.md)
and [reading record](performance/2026-09-18-cs285-reading-review.md): reviewed
all extracted page text from 44 current-course PDFs (1,065 pages), all five
public navigation pages, and visually checked 63 selected lecture pages.
Videos, historical offerings and every visual element were not exhaustively
reviewed. A committed URL/page/hash manifest records the source snapshot.

Recommendation: study whether finite simulator interventions can teach better
recovery decisions from deployment-compatible observation/action histories at
lower total training cost. First validate outcomes and repeated action comparisons
at existing ordinary-SAC failures, then test current-frame/history/privileged
predictors before another policy experiment. History and counterfactual response
models already exist here; do not rebrand them as a new algorithm.

Novelty correction: adaptive allocation of rollout comparisons has direct prior
art from 2008–2010; informed asymmetric RL also already selects privileged signals
by value informativeness. This supersedes the narrower "not found" search claim
in the earlier research-plan review below. A return-gap auxiliary loss alone is
a baseline, not a contribution. The proposed learning/selection mechanism remains
unimplemented and its novelty unproven. No training or physics run was launched
by this review; separate predictability work was left untouched.

## 2026-09-18 — Continue/stop research assessment after the motion pilot

[Decision and primary-source review](performance/2026-09-18-research-direction-decision.md):
stop scaling motion-encoder initialization; retain the broader data-efficient
dressing objective under one bounded diagnostic gate. First validate the task
contract, then test whether finite alternative actions provide repeatable useful
information at observed failures. Do not claim another recovered trajectory,
lower prediction error, or generic reset curriculum is a novel RL result.
Online FQL continuation remains unimplemented and untested.

Code review confirms the reward has no direct grasp-tracking term, although
evaluation requires whole-episode tracking <=2 cm. Teacher difficulty confounds
the held-out gap but does not prove its cause; missing alternative action labels
do not mathematically rule out critic generalization. This qualifies stronger
causal wording in the earlier review, which is retained below as history.
No training or simulator run was launched by this assessment. Separate ongoing
predictability diagnostics were left untouched; no incomplete result is claimed.

## 2026-09-18 — Motion-pretraining pilot completed: no policy improvement

[Implementation, protocol and results](performance/2026-09-18-motion-pretraining.md).
Reused six saved SAC geometry trajectories: five training episodes / 1,480
windows, one withheld episode / 296 windows. Only actor.encoder transfers into
otherwise unchanged FQL; this is not a full PointZero reproduction or a new RL
algorithm. Fourteen focused tests and CUDA smoke checks pass.

All variants completed 3,000 FQL updates and eight full native evaluation
rollouts. Coverage plus whole-episode grasp passes: **random 2/8, geometry 1/8,
motion 0/8**. Withheld-body passes: 1/4, 1/4, 0/4. No simulator errors.
The motion actor never reaches peak coverage .7 in any episode. Both pretraining
runs took about 37 s; the whole sequential experiment took about 28.7 min.

Motion prediction improves globally (11.74 mm versus zero-motion 15.95 mm and
constant-velocity 12.39 mm), but near the held-out elbow it is worse than zero
motion (5.20 versus 2.45 mm). Exact sleeve-opening query coverage is sparse.
This pilot does not justify scaling the actor-initialization recipe. It is one
training seed on narrow development data; the fresh baseline also differs from
the historical 4/8 pilot due to non-bitwise GPU training. No general negative
claim about PointZero or all dynamics pretraining is supported.

Metric caveat: both withheld configurations satisfy the early-turn heuristic
already at reset. All variants have zero paper-filter AND valid-grasp passes;
that heuristic alone cannot diagnose a policy-caused bad cloth route. The reward's
bounded shoulder extension also remains a limitation. No criterion was changed.

Root: `output/uipc_manip/motion_pretrain_20260918/`; see `summary.json`,
`pipeline.json`, and prediction/metric audit artifacts. **No new training running.**
Next gate, if continued: motion supervision that covers opening/elbow behavior
and predicts stationary contact states, plus validated task metrics, before
another matched policy test. Do not restart abandoned IPC-gradient corrections.

## 2026-09-18 — Review of the data-efficient dressing RL plan

Offline measurements on the completed FQL pilot plus three literature surveys;
nothing trained or simulated. [Record](performance/2026-09-18-research-plan-review.md).
Measured: the teacher itself fails one of the two withheld evaluation cells
(tshirt_26/14048, 0.35 upper arm, 3.3 cm tracking), and passes 7/15 training
against 2/10 withheld cells, so "4/4 versus 1/4" is mostly the teacher's gap;
stratify evaluation cells by teacher outcome. Eight failed teacher episodes
stall for about 130 decisions with a saturated command: 1,340 of 7,500 rows
(18 %) carry one action label at a stall, so the critic has no alternative to
rank there; the 13 "regained" progress losses are ray-metric discontinuities.
`early_turn` is inherited from the teacher (24/25). Identical commands from a
restored snapshot give final coverages 0.230/0.246/0.197, so ranking actions
at a state needs several continuations each.

Surveys (primary sources): every part of the "selective recovery experience"
mechanism is published (Tavakoli 2018 restart distributions with TD-error
prioritization at equal cost; RFCL ICLR 2024, the strongest reset baseline;
SCOUT July 2026 names "scaffold access" and "allocation"; VinePPO branches
from every state in language), all rigid-body; not found: visited-state
selection by critic unreliability, branched Monte-Carlo critic targets in
robotics, a budget that charges the restore, or any deformable reset
curriculum. Generalization: Lin et al. ICLR 2025 and Mediratta et al. ICLR
2024 put the lever on distinct configurations (8 → 0.8, 32 → 0.9), we have
three bodies on the withheld axis; the reference pipeline used 6,750
configurations and states no GPU hours. Learner: FQL is behind QC-FQL/AQC/RQL/
ReBRAC-v2 on OGBench (1–100 M transitions), and flow policies lose to
Gaussian ones on narrow 25-demo data; run a tuned cloning arm, RLPD, WSRL,
DSRL and QC-FQL beside it. Recommended order stays: fix the success rule,
repair the teacher at the elbow, scale bodies with the scripted generator,
then compare learners at equal total simulator time; the first mechanism
experiment is restore-and-branch from the eight stalled states with scripted
alternatives, measuring outcome spread and restore cost.


## 2026-09-17 — Owner authorizes FQL implementation and training

[Implementation, method choice and bounded protocol](performance/2026-09-17-fql-pretraining.md).
FQL is selected for this first offline-to-online-oriented dressing experiment;
newer Levine-group RQL reports stronger offline benchmark averages, but neither
is established as best for this task. The point-cloud FQL learner and explicit
checkpoint/transition contracts are implemented. Nineteen focused tests pass.
The learner has no SAC entropy term and uses no IPC derivative. An additional
39 existing tests pass (58 total across the selected files).

Reconstruction replays the existing 25 expert action sequences with observations
and actual successors, including failures and a declared opt-in 5 cm shoulder-
ray reward correction. Old unrelabelable replay rewards are excluded. Native
preparation finished: 7,500 transitions / 662.56 s, with 4,500 training and 3,000
body-disjoint validation rows. The first 3,000 CUDA updates finished in 334.00 s.
Repeated full-episode evaluation: FQL actor 4/8 coverage-plus-whole-grasp successes,
flow prior 0/8; all four FQL passes retain coverage for the final 20 decisions.
Two successes are repeats of a withheld configuration. **Both score 0/8 under
the historical early-turn paper filter.** No robust-policy or Q-term-only causal
claim: one-step distillation and iterative flow generation also differ.

The 30,000-update continuation and evaluation **finished** (2026-09-18).
FQL / flow prior: 5/8 versus 1/8 coverage-plus-whole-grasp successes. FQL training-
body results rise 2/4 to 4/4 from the pilot, but withheld-body results fall 2/4
to 1/4. These tiny repeated-cell samples do not establish overfitting or a robust
improvement. Every FQL rollout triggers early_turn. The prior's single historical
paper-filter pass fails grasp validity; both have zero combined strict passes.
Total training time to 30k: 54.55 min; completed preparation, both training and
evaluation phases: 79.61 min (excluding aborted setup and engineering).
Root: `output/uipc_manip/fql_pretrain_20260917/`; see `summary_30k.json` and
`eval_a100_s17_30k/result.json`. No FQL training process remains.

Next: audit the early-turn flag and persistent withheld failures, then a matched
one-step distillation-only control to isolate Q's contribution. Connect the
collector only under a declared bounded offline-to-online protocol and compare
against from-scratch FQL and SAC/prior-data SAC at common total cost. See the
[research plan](performance/2026-09-17-fql-pretraining.md#research-plan-after-the-completed-pilot).
Full online FQL collector integration and matched controls remain unimplemented.
The old correction experiments remain stopped. No new training was launched for
this research-plan update; prior authorization for bounded experiments persists.

## 2026-09-17 — Prior-data inventory for the FQL design

Offline only; nothing running. The [inventory](performance/2026-09-17-prior-data-inventory.md)
tests the data premise of the [shared FQL design](performance/2026-09-17-rl-pretraining-contract.md)
(other session, `4dbd2464`, `f23627f6`). The eight compatible replays hold
945,864 transitions and 3,072 slot-episodes; 105 episodes (3.4 %) and 0.49 % of
decisions ever come near success, and four runs have none. All have `priv_dim`
0, so the shoulder-overshoot reward correction cannot be computed for them:
corrected-reward Bellman data is the ~11,000 scripted-expert rows, or the
replays keep the cliff reward (a geometry-free mask of the 2,000 on-arm →
off-arm drop rows is a partial way out). A behavior prior fit to this corpus
learns the elbow stall. On the 25 evaluation cells the scripted expert scores
11 final / 15 at the peak in 733 s; the best RL evaluation ever logged is 7,
one snapshot. FQL against RLPD and imitation at matched wall time remains the
right comparison once a repaired teacher has generated data with success in
it; choose the learner last. The record also lists sourced open problems;
the one that matches IPC's guarantee is cloth against moving arms, which
FleX-based work states it cannot simulate and our fixed-arm scene does not
yet do either.

## 2026-09-17 — Concrete shared-policy pretraining design, not yet implemented

The owner's follow-up asks how to train dressing and for algorithm/pretraining
ideas. The [concrete design](performance/2026-09-17-rl-pretraining-contract.md#concrete-first-training-design-shared-fql-pretraining-and-continuation)
selects established FQL for the first explicit pretrain-to-online experiment:
behavior prior, return critics and one-step actor trained from compatible
existing data, then the same objectives continued with fresh IPC experience.
Preserve the observation/action/controller interface; SAC updates are a separate
baseline, not a silent continuation of the FQL critic. No IPC derivative is
required. This is a proposed pretraining structure, not a novel algorithm or a
demonstrated dressing improvement. RLPD is the prior-data comparison.

Resolve the audited reward/success discontinuity consistently across arms.
Do not mix old and corrected rewards without exact relabeling. Some existing
tapes lack a final successor; missing bootstrap rows cannot become fabricated
terminals. Data incompatible with Q updates may still have a declared prior-
training role. Count data preparation, learning, IPC and evaluation time.
No new training, native rollout or algorithm implementation. All abandoned
correction experiments remain stopped; older teacher/collection proposals below
are not the selected next experiment.

## 2026-09-17 — Owner requests RL/pretraining reassessment using CS 285

The owner challenged the sleeve-transfer recommendation and referred to Sergey
Levine's course. [Course-grounded correction and source audit](performance/2026-09-17-rl-pretraining-contract.md)
withdraws that recommendation as the default next experiment. Preserve fast,
reward-based dressing policy learning on one workstation as the objective.
Pretraining can learn representations, behavior, values or dynamics; it is not
synonymous with offline RL, and standalone BC is not its downstream evaluation.

The existing components have distinct contracts: representation transfer copies
actor encoder/history only; BC trains actions; the bounded IQL test used SAC's
own replay with no expert data or online continuation. Component-only pretraining
is valid; no requirement says every network must be pretrained. What remains
unproven is the downstream gain of a declared data-reuse/pretraining recipe.

RLPD/FQL are established references, not new contributions or selected dressing
solutions. Establish the compatible data, initialization, online-improvement and
total-cost evaluation contract. Include the newer shoulder/reward audit below;
do not use peak coverage alone as proof of retained valid dressing. Existing
observed SAC transitions remain useful to model-free RL despite absent geometry.
No training or native rollout in this review; no new algorithm implementation.
Earlier suggested teacher repairs/new collection are not automatically launched.

## 2026-09-17 — IPC-label SAC closed on five seeds; sleeve-path and reward audit of the existing data

Two offline results, no simulation or training launched, nothing running.

**The five-seed cloth-drag study closes the IPC-label SAC updates.** The owner
stopped it at 35k of 40k transitions; the pre-registered rule is applied to the
last common evaluation ([record](performance/2026-09-14-iaql-benchmark.md),
last section). Fresh-batch label against fresh-batch SAC: 1 of 4 seeds,
−0.11 ± 0.35 return; replay label against replay SAC: 2 of 4, +0.09 ± 1.33;
both label arms trail SAC at 10k; SAC alone reaches 59.5 of 64 successes. The
label beats a random direction of equal norm in 3 of 4 seeds, which is not the
bar. `B_fresh_ipc_s0` died on a MAGMA allocation assertion with 25 processes on
one GPU. The export modes, the friction coupling, the lockstep multi-slot
environment and the batched GPU tangent are validated and stay.

**Owner's goal for the new direction (recorded in `rule.md`)**: train a
dressing policy fast and easily on one workstation; the reference pipeline's
per-region RL teachers are hard to scale and need a cluster.

**[Sleeve-path and reward audit](performance/2026-09-17-sleeve-path-audit.md)**
of the 25 scripted-expert episodes and the ordinary SAC replay, bearing on the
[transfer proposal](performance/2026-09-17-dressing-transfer-direction.md):
the opening is 19 cm from the tool and moves 11 cm relative to it within an
episode, and the tool path realising one sleeve path differs by garment by
14–30 cm, but the expert already servos the opening centroid, so the tight
sleeve path (1.2 cm lateral spread against 8.5 cm for the tool) is partly a
property of the data generator. All ten real expert failures are one executor
failure: the opening stalls just past the elbow (arc 0.61–0.79) while the tool
runs 0.14–0.27 ahead in nine, tshirt_392 on all five bodies. Four more
"failures" are complete dressings: the upper-arm ratio falls from 1.00 to 0.00
in one decision with under 1 mm of opening motion once the opening passes the
shoulder (`wang_progress` ray origin), so the expert dresses 15 of 25 at the
peak, 11 by the final reading, 6 with the grasp limit. In the SAC replay,
branch switches of the reward carry 91 % of the summed squared one-decision
reward change (128 on-arm → off-arm drops, median 0.76, against a median change
of 0.0032). The SAC replay has no privileged geometry (`priv_dim` 0).

Before the proposal's first comparison: fix the success rule (past the shoulder
is complete; stop rule or peak-with-valid-grasp), truncate the four overshoot
episodes at their peak, collect a few hundred geometry-logged expert episodes
(25 cost 733 s), and test a teacher with stall detection, back-off and lateral
centring on the same 25 cells (about 12 minutes). That last test decides
whether sleeve feedback repairs the dominant failure; nothing here was run.
No cross-simulator transfer result can come from existing data.

**[Prior-art check](performance/2026-09-17-sleeve-goal-prior-art.md)** of the
sleeve-goal proposal (two web surveys; the key paper verified directly): sleeve
state in arm coordinates as the control target is already published for
dressing (Wearing A Coat, arXiv 2607.10999, July 2026: hand-set multi-phase
goals, gradient MPC through a non-IPC differentiable cloth simulator taking
seconds, friction with the human given as input, no learned controller).
Kotsovolis and Demiris learn forward models of the opening for MPC. Learned
goal geometry with a separate executor exists outside dressing (TAX3D,
DefGoalNet). Not found: a learned sleeve-goal policy with a closed-loop
executor for dressing, any dressing work on IPC contact, and any controlled
test of transfer across garment-to-arm friction, stiffness or simulator. That
last axis is where a contribution could be; the first cheap measurement is
whether successful sleeve paths shift with friction under one servo teacher.

## 2026-09-17 — Research pivot after completed SAC audit

The owner renewed the instruction to abandon the previous idea and research a
new direction. The [new research assessment](performance/2026-09-17-dressing-transfer-direction.md)
uses the completed ordinary SAC audit and supersedes the next-step instruction
in the older section below. Do not restart IPC/SAC corrections. No training or
native evaluation was launched during this research; no new policy result exists.

Recommended hypothesis: learn desired sleeve-geometry sequences from existing
trajectories and execute them with domain-specific feedback control, investigating
transfer across contact dynamics. This is a task/transfer research proposal, not
an established novel RL algorithm. Dressing diffusion/MPC, TAX3D, ArticuBot,
keypoint actions and latent transfer substantially constrain novelty. Existing
arm-relative privileged features are infrastructure, not a new contribution.

First proposed screen is a matched native action-sequence versus sleeve-goal
policy comparison; only a positive full-episode result justifies the subsequent
source-to-target transfer study. These experiments are specified, not run.
The inspected Newton buffer has x-ray observations and kinematic cuff grasp but
no explicit full geometry, so geometry-rich source records must be identified
before pooling goals. Reusing actions or padding observations across domains is
not valid. Real sensor-to-command deployment and reduced adaptation cost remain
unproven. Further ordinary SAC checkpoint evaluation is useful as a baseline,
not a reason to withhold the requested research assessment.

## 2026-09-17 — Owner abandons IPC correction; ordinary SAC diagnosis first

The owner abandoned the proposed research direction and then clarified twice:
analyse **normal SAC policy rollouts** before proposing a replacement. The
[SAC-only audit](performance/2026-09-17-normal-sac-rollout-audit.md) takes precedence
over the older proposals below. No training is running. The gradient-correction
prototype remains historical and must not be resumed as the current plan.

Saved ordinary SAC traces show tshirt_26 stalling at .51–.67 upper-arm progress,
while tshirt_68 peaks at .21–.27 and loses the arm intersection; all eight episodes
have zero controller rejection. New unchanged-policy geometry captures confirm
the opening moves away from the upper arm in tshirt_68, including one episode
with valid grasp throughout. Two sampled training poses finish at .720 and .150;
the first exceeds the grasp limit earlier, so neither is whole-episode valid
success. The problem is not solely unseen-body generalization. Completed SAC
diagnostics: 1,800 decisions / 240.69 s including setup/serialization; metrics
recomputed from geometry agree to 1.12e-16. All jobs finished.

The source ordinary SAC did not use the existing expert trajectories. Its 125k
transitions include 408 complete episodes across 225 planned configurations, with
logged rolling training success at most .025. Historical ordinary dense/residual
SAC reached 7/25 reported geometric successes before returning to 0/25; the peak
needs repeated, grasp-aware validation. No specific critic/perception/exploration
cause is yet isolated. First compare saved best/final ordinary SAC on a fixed
matrix, then test one cause at a documented SAC failure window. BC/recovery results
are not explanations of SAC's failures; the initially launched BC diagnostic was
stopped after clarification and excluded. No replacement algorithm is selected.

## 2026-09-17 — Novelty objection: gradient-correction research, no training running

The owner explicitly rejected SAC plus recovery imitation as insufficiently
novel and asked for deep research on a new RL algorithm. Read the
[prior-art comparison and precise candidate](performance/2026-09-17-ipc-policy-gradient-research.md)
before implementation. Q-Prop/Stein, TPX, AHAC, adaptive value expansion, MFPG
and recent targeted-gradient/compute-allocation work limit generic novelty claims.
Candidate: allocate expensive IPC root queries, continuation depth and repetition
to improve the accuracy of a return-based SAC actor correction per second.
The baseline control-variate and inverse-probability identities are established;
the joint allocator, native collector and actual benefit are still unimplemented
or unproven. A standalone `policy_gradient_correction` mathematical primitive has
seven passing tests and is not wired into training. Respect policy freshness,
selection probabilities, full root state and the existing time-limit bootstrap;
teacher actions/returns cannot be treated as current-policy branch samples.
The newly launched guided-SAC comparison was stopped during its control run;
there is no completed training comparison. All experiment processes are stopped.

## 2026-09-17 — IPC recovery guidance integrated into SAC pretraining

The owner clarified that the desired contribution belongs inside RL pretraining,
with the existing downstream policy interface. [Implementation and fixed pilot](performance/2026-09-17-recovery-sac-pretraining.md).
`pretrain_wang --recovery-source-dirs` now adds verified active-action MSE to the
same scheduled SAC actor loss/optimizer step; the critic still learns ordinary
soft Bellman targets. Recovery arrays stay on GPU and an independent batch is
sampled per actor update. Guidance is explicit per update and is absent from
plain SAC continuation/inference. Source contracts, completed verification and
held-out body exclusion are checked. No new force input, architecture, controller
cap, reward, solver or runtime teacher is introduced. The old BC-only test did
not test this integration. The fixed live comparison was stopped after the
owner's novelty objection; the treatment never started and no final continuation
checkpoint exists. The source's initial evaluation is not a new training result.
Seventy-four focused tests pass, and all 360 recovery rows load on CUDA under
the saved SAC contract. Keep this option as an established baseline, disabled by
default. See the report for cost and inference boundaries.

## 2026-09-17 — Recovery teacher and BC transfer experiment complete

The owner asked whether recovery teaching is novel and to try a promising
contribution. [Prior-art audit and fixed experiment](performance/2026-09-17-recovery-teacher.md)
find DAgger/GPS, selective intervention and MPC-guided SAC close precedents;
the broad recipe is not new. `recovery_teacher` now tests four existing geometric
recovery routes and policy/scaled-policy controls from the BC actor's own failed
tshirt_68/14046 state, through the full episode. A fresh-world verification must
beat both controls before any labels are admitted. `step(reset_on_done=False)`
retains terminal state without changing default reset or horizon. Seventeen focused
tests and two native terminal/reset tests pass. Search found outward recovery
above .99 sustained coverage; independent verification reaches .98444/.97604,
admitting 360 rows. Original/continued/recovery BC full-episode successes are
2/8, 2/8, 3/8; all fail target tshirt_68/14046 twice. Recovery BC fits teacher
actions about ten times better but does not transfer that recovery. A separate
command-cap check gives 1/4, 0/4, 1/4 and still fails the target. Total teacher
and evaluation cost is 14,520 native decisions / 25.46 min, plus ~58.2 s fitting.
All jobs in that experiment finished. IPC supplies a useful teacher here;
robust policy improvement and algorithmic novelty remain unproven. The owner's
subsequent SAC integration is separate; do not merge its results with BC.

## 2026-09-17 — Existing-expert actor pretraining and evaluation complete

The owner approved connecting existing expert data to fast policy pretraining.
[Corpus audit, training and protocol](performance/2026-09-17-expert-policy-pretraining.md).
Larger Newton collections exist but have incompatible observation/physics
contracts. Six compatible IPC expert sequences lacked observations; replaying
each twice reconstructed 3,600 transitions in 318.63 s. All 12 pass the explicit
coverage/grasp rule (mean .96398 coverage), but all fail historical early-turn
admission, recorded separately. BC trains on body 14046 and withholds 14047/14048
with all repeats grouped. A fresh actor with the saved SAC architecture completed
3,000 CUDA MSE updates in 86.8 s through final scheduled validation; no simulation
occurs during BC. Critics remain untrained. Seventeen focused tests pass. Full
300-decision evaluation on four development configurations, two rounds, is done:
BC/SAC coverage-and-grasp successes 2/8 versus 0/8, mean coverage .25968/.29953.
Both BC successes are tshirt_26/14046; withheld-body BC is 0/4 with zero upper-arm
coverage. BC has no invalid-grasp decisions or controller rejections; SAC has
109/2,400 invalid-grasp decisions. Both policies pass historical early-turn
admission 0/8. One BC training-case trace reaches ~.99 then collapses to zero
without grasp failure; inspect geometry/metric before labeling recoveries.
Evaluation cost 4,800 decisions / 454.46 s. All jobs finished. The fast BC path
works, but robustness, useful IPC recoveries and a new RL method remain unproven.
No online training started. Optional coverage filtering now only tightens the
explicit kept label; this unused option does not change the measured run.

## 2026-09-17 — Parallel IPC trajectory optimizer implemented and tested

The owner requested parallel GPU implementation and testing. New
`uipc_manip.parallel_trajopt` uses CUDA CEM candidate generation/ranking and native
batched CUDA IPC rollouts, with balanced candidate-to-state assignment and equal
per-decision command norm caps. It uses no CPU Hessian factorization. The existing
environment still uses CPU control/metrics/transfers, so this is not GPU-only.
Twelve focused tests pass, including CUDA execution and closed-loop SAC/tail
accounting. Four completed native runs total 4,956 decisions / 1,025.53 s. Matched
384-decision bank evaluation takes 163.19 s serial versus 113.37 s with four slots
(1.439x observed speedup); construction/approach reduce whole-command benefit to
1.079x. Physical paths differ across batch sizes, so this is not isolated kernel
scaling. Fresh five-control validation gives CEM/reference/closed-loop SAC mean
coverage .57704/.58330/.57259, each 0/4 sustained successes after 72 decisions.
All 20 continuations have valid grasp and zero controller rejections. The small
CEM-SAC difference does not establish a policy gain. All jobs finished; no policy
training launched. See [results and boundaries](performance/2026-09-17-parallel-dressing-trajopt.md).

## 2026-09-17 — Deep dressing research and repeatability decomposition

Read [the evidence and staged redesign](performance/2026-09-17-dressing-research-redesign.md).
New `scripts/diagnose_dressing_repeatability.py` completed 132 native decisions
in 61.41 s at tshirt_26/14046 decision 60. Cached-observation deterministic actor
outputs are identical, but three restored fixed-action 12-decision runs finish
at .22954/.24619/.19703 coverage: a .04916 range without policy feedback. All
grasp checks pass, commands are exactly identical, and no controller rejection
occurs. Position restore error is zero; source code does restore velocities and
previous positions. The cause within forward/recovery/numerical execution is
not isolated. Next compare cold-prefix replay with recovery and inspect complete
state before trusting small local improvement labels. Stochasticity alone does
not prove RL cannot learn.

One existing expert diagnostic subset has 11/25 final geometric successes,
6/25 also within the current 2 cm maximum tracking limit, and 0/25 passing the
early-turn admission filter. Four reach near-full coverage then end at zero;
inspect geometry/metric before naming the cause. This subset stores actions and
35D geometry summaries, not policy observation sequences. Locate compatible
records in the existing corpus; the previous offline controls used policy replay,
not all expert trajectories. Twelve-action optimization was already tried:
most gain came from larger movements. The proposed next mechanism tests distinct
recovery routes with equal movement/compute controls, then supervises SAC at
learner-visited states only if the teacher wins. MPC+SAC, chunking and distillation
already have close prior art; no new successful RL algorithm is claimed. No
policy training was launched; all new diagnostic work is complete.

## 2026-09-17 — Bounded SAC actor-action audit

The owner approved the next actor test, not another long training run. Read
[the protocol and derivative limits](performance/2026-09-17-dressing-actor-audit.md).
The diagnostic six-frame chain now includes tool-relative arm/goal/extras and
recorded controller acceptance, clipping and finite rotations. Observation RNG
is restored with snapshots. Mesh-contact friction history is still missing;
the batched training signal has not been replaced with this unvalidated chain.
`physics_actor_audit` compares full executed finite differences and six action
corrections against the stronger existing SAC at four later development states
plus two earlier elbow states, with repeated 12-decision continuations and a
grasp/coverage gate. **All six proposal types pass 0/6 state gates**. Native cost:
1,794 decisions / 404.92 s, all jobs finished; 38 focused CPU tests pass. No policy
training ran. Geometric task gradients often agree well, but value directions
do not pass accuracy at both finite-difference scales. Earlier body 14046's
repeated SAC coverage differs by .06709 despite restored positions/RNG, so
complete trajectory repeatability must be checked before interpreting that case.
Small positive coverage changes elsewhere are below the preset .01 threshold.
Do not claim IPC cannot help or the critic alone explains the failures.

## 2026-09-17 — First-principles audit and bounded offline controls

Read [the audit and research plan](performance/2026-09-17-dressing-first-principles.md).
The point-cloud dressing replay path omitted the original action when applying
IPC labels, bypassing its locality gate. Fixed and regression-tested. Evaluation
now retains grasp-valid success separately from geometric success. Dressing info
reports collision/tether rejection counts and accepted anchor translation without
changing control. A matched evaluation script records these per decision.

`offline_rl` provides explicit IQL, BC and replay-only SAC controls on existing
data, not a claimed new algorithm or a solved offline-to-online method. The
125,016-row dense/plain source yields 124,584 exact successor links in 432
segments. The new offline updates use a common segment split and fresh optimizers;
the source checkpoint had already seen that corpus. IQL/BC/replay-only SAC
completed 2,000 updates in 220/67/137 seconds. Full two-round, two-cell development
evaluation gave source/IQL/BC/replay-SAC mean coverage .161/.135/.184/.223,
**0/4 successes each**, substantial variation, and 0/23/497/582 invalid-grasp
decisions out of 1,200 per policy. No robust advantage is established. Evaluation
cost another 4,800 transitions and 637 s. All jobs finished. 88 focused CPU tests
pass; an unrelated existing ADR 0008 heading check fails. Do not infer policy
quality from the loss curves or silently resume IQL's non-soft critic as SAC.

At the previous IPC checkpoint, only 31/2,400 labelled actions fall within the
.5 radius of the current actor: .0243% of all replay, or about 1.55% chance of
a nonempty physics batch of 64. This endpoint audit is not a training-time
reconstruction. Correct locality mostly abstains on that replay. Empty physics
batches now log explicit zero rows/loss, eliminating stale nonempty metrics.
Next IPC actor work must address fresh same-action labels and the complete
six-substep/controller/tool-relative observation derivative, or use task-based
sequence supervision if critic guidance remains unreliable. The original policy
stalls on one development arm despite accepted commands and valid grasp; BC
also creates a separate controller-rejection failure. Neither justifies claiming
all failures have one cause.

The owner explicitly challenged treating offline-to-online RL as solved or IPC
integration as impossible. Neither claim is supported. The bounded offline
comparison is diagnostic; a large offline study is not the agreed next step.
The next IPC mechanism should follow measured controller/cloth/policy failure
classification. Gradient-free options include privileged value learning,
execution prediction and simulated recovery supervision; none is validated here.

## 2026-09-16 — Dressing implementation, negative correction gate, and preserved-state continuation

Read [the measured experiment record](performance/2026-09-16-dressing-verified-results.md)
before launching more training. The owner requested implementation/testing and
speed diagnosis after both critic/actor routes had already been tried.
`physics_gradient_finetune --verified` now tests bounded finite IPC/SAC/random
corrections and fits guarded actor copies; all three proposal types had 0/8
accepted targets over tshirt_26/14049 and tshirt_392/14046. No IPC policy benefit
was established, so do not extend this particular correction run blindly.

`pretrain_wang --init-optimizers --init-replay ... --init-from ...` preserves
existing Adam/replay when branching a controlled experiment. Targets include
historical transitions. The 125,016 -> 127,416 SAC continuation completed with
zero simulator errors: two-cell development coverage .15164 -> .51295, success
still 0/2. The matched existing IPC-actor arm finished at .36100 coverage, also
0/2 successes, and took 592 s of training-loop time against SAC's 391 s. This is
a short development comparison with only 1.88% physics-labelled replay. Three
fixed-seed evaluation rounds per final checkpoint give mean coverage .47347 (SAC)
versus .37301 (IPC), with 0/6 successes each on the two repeated development
configurations. No general advantage or complete dressing policy is established.
All jobs launched in this pass finished. Input
derivative queries avoid parameter-gradient accumulation; physics timing logs
now preserve the cumulative timer. 28 focused CPU tests pass.

Dressing-only homogeneous batch profiling measured 7.02/18.29/17.71 simulator
transitions/s at 1/8/24 environments; 96–98% of environment time was in simulation.
Two eight-slot processes on the existing MPS server then delivered 29.11 versus
18.05 aggregate transitions/s for one process (1.61x). This is not a shared-policy
training speedup claim. `scripts/profile_dressing.py` provides a reusable profile
and separate native timer window. Collision-candidate search was 56% of that
instrumented window; PCG was 11%. The
unrelated cloth-drag study remains stopped.

## 2026-09-16 — Dressing RL research resumed; cloth-drag study stopped

The owner stopped the unrelated-to-dressing 40k diagnostic study: 24 surviving
workers and its launcher were terminated, logs retained, no final checkpoints.
Do not present its cloth-marker success rates as dressing results or restart it
as a prerequisite for dressing research. The trajectory-generation pilot below
is deprioritized; the existing trajectory corpus is the starting point.

[Redesign research](performance/2026-09-16-dressing-rl-redesign.md) distinguishes
the historical dressing last-frame value-gradient surrogate from the full-state
benchmark's Bellman-gradient update. It proposes finite-rollout verification of
occasional IPC action proposals, bounded policy fitting, and a controlled dressing
comparison. This is a proposal, not an implemented or validated new algorithm.
Primary literature, novelty limits, data/reset requirements, and stop rules are
recorded. No new training or native experiment was launched in this research pass.

## 2026-09-16 — Preparing a dressing trajectory-generation pilot

Added an opt-in outward elbow route, explicit pose subsets / parameter files for
expert collection, per-decision metric traces, and a bounded sequential comparison
runner. [Protocol and limits](performance/2026-09-16-dressing-route-pilot.md).
32 CPU checks pass; native execution is pending GPU capacity. The current study
was left running. No demonstration-quality or speed improvement has been measured.

> **Routing note (2026-08-30)**: this file is the chronological audit trail and
> may retain detailed commands/incidents. New durable architecture rationale
> belongs in `agent_docs/adr/`; reusable performance conclusions and rejected
> experiments belong in `agent_docs/performance/`. Add a short handoff pointer
> instead of growing this file as the only source of truth.

> **Embedded C++ METIS migration (2026-09-03, `refactor-main`)**: geometry now
> links the private `uipc_metis` target from `src/geometry/metis/`; the separate
> `external/METIS` and `external/GKlib` source trees and build targets were
> removed. CMake and XMake keep METIS independently compilable and exclude its
> sources from direct `uipc_geometry` compilation. The C++ port was made
> const-correct for diagnostic strings and self-contained for modern MSVC.
> Unused glibc getopt/regex/qsort and optional MT19937-64 sources were removed;
> required sorting now uses a clean in-tree deterministic C++
> partition/insertion implementation that preserves the former equal-key
> ordering, so only the METIS/GKlib Apache notices remain. A deterministic
> public `mesh_partition` regression covers the linked
> API; zero-sized partitions, 32-bit capacity overflow, and invalid returned
> partition IDs are rejected before division or indexing. Portable CPU/wall
> timers and out-of-core temporary-file removal replace incomplete port stubs.
> Against preserved pre-removal C binaries, three synthetic graph families,
> four full `fluffy_ball.msh` configurations, and two `animal_well.msh`
> configurations on both Windows/MSVC and Linux/GCC produced identical return
> codes, edge cuts, and every per-vertex partition ID on the same platform. See
> [`ADR 0007`](adr/0007-embedded-cpp-metis.md) for exact hashes and boundaries.
> PR #492's first Linux XMake run caught a compiler-filtered `-fPIC` flag being
> silently omitted from `uipc_metis`; the target now applies the C++ flag
> without a tool-name filter while retaining XMake's support probe, matching
> CMake's PIC property and allowing its
> thread-local GKlib state to link into `libuipc_geometry.so`. The adjacent
> GNU/POSIX `strerror_r` portability warning was fixed at the same time so Linux
> error paths return the correct diagnostic string.

> **AL-IPC sample 88 trajectory correction (2026-09-03, `refactor-main`)**:
> the severe early trajectory split was traced to AL ignoring the configured
> `K_min=6`, an EE friction derivative assembled with the PT Jacobian, and the
> uniform `diag_norm` penalty being unsuitable as the default for mixed
> cloth/volumetric masses. AL now delays cumulative safe-path attenuation until
> `K_min`, uses the correct EE Jacobian, and defaults to mass-based `per_vertex`
> scaling while keeping `diag_norm` experimental. Exhausted line search restores its recorded start
> point rather than accepting the last energy-increasing trial. PT/EE/plane
> finite-difference tests pass, as does the AL `K_min` simulation assertion.
> In matched 60-frame sample 88 runs, upper/lower centroid error versus IPC fell
> from 0.2963/0.0807 to 0.0115/0.0045. A full corrected 250-frame run completed
> with all 250 frames converged and no line-search limit, Newton limit, or
> runtime error. See the durable evidence and
> regression boundary in
> [`2026-09-03-al-ipc-case88-correction.md`](performance/2026-09-03-al-ipc-case88-correction.md).
> **Parallel-EE policy (2026-09-03, `refactor-main`)**: standard IPC retains
> normal `need_mollify()` detection with coefficient `1e-3` and its complete
> mollified normal energy/gradient/Hessian; standard IPC friction skips the
> detected parallel pairs. AL-IPC uses the shared negative disabled threshold
> for both normal and frictional contact, so all AL pairs follow ordinary EE
> paths. The CUDA regression checks both threshold behaviors with exactly
> parallel edges.
> Final validation passed CMake/Core 36 cases / 1040 assertions, CUDA backend
> 22 / 335, the single-process simulation suite 95 / 14213, Python portable
> tests 80 passed / 1 skipped, repository contracts 43/43, fast CTest 3/3,
> the complete Doxygen/MkDoxy/MkDocs build, and the XMake production CUDA
> target. XMake's aggregate CUDA test target still encounters the already
> documented CUDA 13.2/fmt 12 character-literal incompatibility; the CMake
> build compiled and executed every new CUDA test.

> **AL-IPC `AL-release` integration audit (2026-09-02, `refactor-main`)**:
> the six fork-only commits were reviewed against current libuipc and the
> AL-IPC paper rather than cherry-picked across 188 intervening local commits.
> The useful core is reimplemented in current `cuda_tool`: per-vertex
> earliest-TOI candidate filtering (excluding existing pairs), decay-derived
> active-pair lifetime, stable preservation of old pair state, configurable
> Hessian-diagonal penalty initialization, AL-specific CCD safety margin, and
> the full-step boundary between inner Newton work and outer multiplier/CCD
> updates. Scratch is persistent and amortized. Important fork defects were
> not copied: its 32-bit `__float_as_int` atomic corrupts this project's
> double-precision `Float`; its fixed 25-update lifetime disagrees with the
> documented `gamma < 0.01` rule; it includes existing pairs in the candidate
> minima; and its raw displacement convergence test discards the current
> `NewtonToleranceManager`. The fork's global CCD-margin edit was scoped to
> AL-IPC so normal IPC retains its established margin. Large benchmark assets,
> screenshots, and unrelated example/README edits were intentionally omitted.
> At this audit revision AL kept its pre-existing `K_min = 1`
> cumulative-safe-path termination; the 2026-09-03 correction above supersedes
> that limitation with the configured multi-state `K_min > 1` rule. On
> CUDA 13.2 / RTX 5090, CMake and XMake production builds passed, as did the
> focused AL math test (17 assertions after final review), Core (36 cases /
> 1040 assertions), CUDA backend (17 / 291), all 19 AL sections, and the full
> single-process simulation suite (95 / 14212). Repository contracts passed
> 43/43, the current native Python schema reports 48 keys, and the full
> Doxygen/MkDoxy/MkDocs site built. In mixed ABD/FEM case 18, maximum active
> pairs fell from 81 to 42 and total PCG iterations from 1125 to 700; outer
> solves rose from 114 to 135, so future performance claims must still use the
> canonical large scenes.

> **cuBLAS-free CUDA runtime boundary (2026-09-02)**: published wheels through
> 0.0.27 directly import CUDA 12 cuBLAS, but current source no longer uses or
> links cuBLAS. `LinearSystemContext` dot/norm now use named block-partial
> kernels plus CUB final reduction; partial/result storage is persistent and
> geometrically grown, CUB scratch remains per-stream and persistent, and norm
> uses a scaled-square state to avoid overflow/underflow. The heavy reduction
> header is included only by the three calling TUs and focused CUDA test, so the
> lightweight `linear_system.h`/`cuda_tool.h` path does not transitively pull in
> all CUB algorithms. CMake and XMake links are synchronized. A wheel CI audit
> rejects dynamic Toolkit libraries via dumpbin/readelf; compatibility policy
> requires driver >=525.60.13 (Linux) or >=528.33 (Windows) when a packaged
> SASS image applies, and >=570.124.06 / >=572.61 when the CUDA 12.8 Update 1
> PTX image must be JIT compiled. Final
> CUDA 13.2/RTX 5090 validation passed the CMake and XMake backend builds, the
> 12-assertion focused reduction test, all 274 CUDA-backend assertions, all
> 14,212 assertions in the 95-case simulation suite, four representative
> benchmark scenes, 43 repository-script tests, 79 portable Python tests, the
> real CUDA doctor probe, and a full Doxygen/MkDoxy/MkDocs build. Both local
> build systems produced a backend DLL with no CUDA Toolkit imports.

> **Windows wheel RDC fix (2026-09-01)**: `main@b6f2b006` had five real
> Windows wheel failures; CMake, XMake, repository contracts, and all five Linux
> wheels passed. scikit-build-core's Visual Studio generator compiled all 198
> domain OBJECT sources with RDC but omitted `nvcc -dlink`, ending with 198
> unresolved `__cudaRegisterLinkedBinary_*` symbols. A minimal cross-TU CUDA
> probe reproduced the failure for both OBJECT attachment idioms. Giving the
> final shared target one directly owned, generated comment-only `.cu` source
> made Visual Studio emit the correct device-link and the probe passed. CMake
> now carries that language anchor; domain ownership and XMake are unchanged,
> and a repository contract prevents its removal. Validation then configured
> the real project with the Visual Studio 2022 generator, compiled all 198 CUDA
> sources, executed `nvcc -dlink` over every domain object, produced and loaded
> the 17,622,528-byte `uipc_backend_cuda.dll`, and passed `0_abd_gravity`
> (IPC + AL-IPC, 178 assertions). The existing Ninja build also regenerated,
> device-linked, and rebuilt `sim_case` successfully; repository contracts pass
> 36/36. GitHub then passed CMake run `33490698492`, XMake run `33490698388`,
> and Repository Contracts run `33490698389`. A manual non-publishing wheel run
> `33490746393` passed all ten CPython 3.10–3.14 Windows/Linux jobs; the five
> Windows jobs that had failed on `main` now build, install-test, and upload
> successfully.

> **Canonical benchmark suite expansion (2026-09-01)**: root benchmark
> ownership now covers sample 6 (`rigid-wrecking-balls`), 88
> (`stiff-gipc-case2`), 89 (`mas-bunny`), and 93 (`cube-wall-cloth`). The sample
> implementations share machine-readable headless reporting without changing
> scene parameters. Normal runs keep synchronized Timer scopes off and archive
> full-precision frame times, backend Newton/line-search/linear-solver counts,
> final-state observables, raw logs, revisions/runtime facts, and an approximate
> WDDM total-memory peak; separate Timer-enabled runs provide stage diagnostics.
> All four entries completed real three-frame smoke runs and then three full
> interleaved throughput runs. Median run means on RTX 5090/CUDA 13.2 are
> 129.5 ms/frame (rigid, 120 frames), 201.1 (case2, 250), 60.2 (MAS bunny,
> 100), and 125.6 (wall/cloth, 100); all frames completed/converged with no
> iteration-limit hits. Collision-rich trajectories and WDDM memory/timing
> have measured envelopes rather than exact goldens. Durable method, raw run
> IDs, stage diagnostics, and interpretation are in
> `agent_docs/performance/2026-09-01-cross-domain-baseline.md`; historical
> 73/156/301 ms notes below are not the current baseline.

> **Post-merge clang-format race fix (2026-09-01)**: PR #486 merged while its
> format job was running. The job then fetched moving `origin/main` with
> `--depth=1`, replacing the original PR base and producing `no merge base`;
> no source formatting violation occurred. The workflow now diffs the immutable
> pull-request `base.sha...head.sha`, and a repository contract prevents the
> moving shallow-fetch pattern from returning.

> **QR-SVD float sign hardening (2026-09-01)**: the Wilkinson shift in
> libuipc, GPU_IPC, and Stiff-GIPC now computes its magnitude in the template
> scalar type and applies the sign with `d < 0 ? -shift : shift`, explicitly
> taking `sign(0)=+1`. This removes the unsafe standard-library sign-copy call
> from CUDA float paths. The sibling commits are GPU_IPC `4fb6019` and
> Stiff-GIPC `bb2849a`. Libuipc has a named-kernel regression covering float
> positive, negative, and negative-zero inputs on the GPU; the repository
> contract also forbids reintroducing the call. A standalone CUDA 13.2 / sm_120
> execution returned `[-1, -0.414213538, 2.41421366]`; libuipc's complete CUDA
> backend built with device-link in 541 seconds and sim_case passed 95 cases /
> 14212 assertions. GPU_IPC and Stiff-GIPC both completed full Release builds;
> GPU_IPC also completed one frame. Stiff cases 1 and 2 remain blocked before
> QR-SVD by their pre-existing `DeviceBuffer allocation size overflow` during
> initial triplet allocation. The local aggregate backend test target remains
> blocked by the unrelated CUDA 13.2/fmt 12 character-literal issue, although
> the new test translation unit itself compiles.

> **Samples cloth calibration (2026-09-01)**: at that checkpoint,
> `libuipc-samples/main` ended at `4e83b83` and the parent gitlink followed it.
> The later benchmark contract advances the gitlink to `4fb26b7` on the
> samples `benchmark-baseline` branch; measurements used its scene-identical
> instrumentation parent `8701983`, and the child only corrects benchmark
> prose. All
> eight simulated-cloth
> examples use Baraff-Witkin membrane plus formula-based Discrete Shell
> bending with one-sided thickness `r=1e-3` and density 200. Example 88 remains
> the common baseline (`stretch E=5e4`, `shear E=1e1`, `nu=0.49`,
> `strain_rate=100`, `bending E=3e4`); subsequent owner tuning sets example 11
> to stretch/bending `E=1e4`, example 34 to `E=1e4`, `nu=0.40`, and example 93
> to `strain_rate=10000`. Python syntax checks passed for every changed scene.

> **Semi-implicit Newton default (2026-09-01, `refactor-main`)**:
> `newton/semi_implicit/enable` now defaults to `1` and
> `newton/semi_implicit/K_min` defaults to `6`. The beta tolerance remains
> `1e-3`, and `K_min` remains an accumulation start rather than a hard Newton
> iteration floor; `newton/min_iter` is still the separate floor and defaults
> to zero. The scene-config schema test locks both new defaults. Validation
> passed Core 36 cases / 1005 assertions and the complete CUDA sim suite 95
> cases / 14212 assertions.

> **Thin-shell reference-weight correction (2026-09-01, `refactor-main`)**:
> the elastic, strain-plastic, and stress-plastic Discrete Shells paths had
> multiplied their already integrated `L0/h_bar = 3L0^2/A` metric by `A` a
> second time. Their generic outer weight is now neutral, preserving the
> paper's inverse-area normalization and uniform-scale invariance. The stored
> vertex `thickness=r` is explicitly one-sided: formula-based bending uses
> `E*(2r)^3/(12*(1-nu^2))`, and Baraff-Witkin stretch uses
> `(lambda+2mu)*(2r)`. Its separately calibrated shear coefficient remains
> thickness-independent. Focused regression coverage checks all three bending
> variants at two uniformly scaled rest hinges. Validation passed repository
> contracts 30/30, Core 36 cases / 1001 assertions, the full CUDA backend build
> including device-link, sim_case 95 cases / 14212 assertions, and the focused
> MAS stitch regression 1 case / 4 assertions. Rebuilding the complete local
> CUDA test executable remains blocked by the existing CUDA 13.2/fmt 12
> character-literal incompatibility in unrelated `.cu` tests; the new host-side
> reference-weight test itself compiles.

> **GitHub check portability fixes (2026-08-31, `refactor-main`)**: successive
> full XMake runs exposed that CUDA OBJECT dependencies first lacked the project
> `src/` include, then backend definitions, then Linux PIC, and finally did not
> enter XMake's CUDA device-link at all. The last issue produced 198 unresolved
> `__cudaRegisterLinkedBinary_*` symbols on Windows; Linux had allowed the same
> unresolved references in a shared object. The final design keeps CMake OBJECT
> targets, but XMake attaches its matching logical component manifest directly
> to the one shared target. That gives both platforms one complete RDC
> device-link and inherits the final target's includes, definitions, and PIC.
> A repository-contract test prevents reintroducing XMake CUDA OBJECT
> dependencies. Repository Contracts deliberately checks out no submodules, so
> its benchmark-manifest test now
> validates the declaration without requiring
> samples assets; runtime asset validation has an isolated partial-checkout unit
> test. Rapid-push cancellations were concurrency behavior, not additional
> source failures. Local repository contracts passed 29/29. A clean Windows
> `xmake build --jobs=8 cuda` emitted the expected
> `devlinking.release uipc_backend_cuda_gpucode.cu.obj`, linked the DLL, and
> completed in 574.109 seconds; a legacy-off configuration omitted exactly the
> three legacy filter sources. The local XMake configuration was then restored
> to legacy-on and native architecture selection.

> **Aggregate architecture validation (2026-08-30, through `67ff50c3`)**:
> the default Release build completed for Core, none/CUDA backends, and all
> native test targets. CTest passed 7/7 aggregates with CPU work concurrent and
> all four GPU entries serialized by `uipc_gpu`; this includes sim_case 95 /
> 14212. Python passed 79 portable tests (1 skipped, 54 deselected) and 48 CUDA
> non-example tests (1 skipped, 85 deselected). Repository/script contracts
> passed 27/27; default, legacy-off, and CUDA-off XMake configurations parsed;
> full MkDocs+Doxygen output built; clang-format-18 passed for every C++ change
> versus `origin/main`. Both tracked submodules were clean and `refactor-main`
> matched `origin/refactor-main` before this documentation checkpoint.

> **ADR and performance evidence archive (2026-08-30, `refactor-main`)**:
> five accepted ADRs now cover the backend handshake, CUDA component boundary,
> scene-config contract, deterministic SimSystem topology, and test/benchmark
> entry points. `agent_docs/performance/` defines evidence/interpretation rules,
> provides a reusable template, and consolidates the four 2026-08-30 case2
> assembly stages plus the rejected stencil split. Existing handoff history is
> preserved rather than rewritten. Archive numbering/sections/local links are
> enforced by the scripts contract suite (27/27 passed).

> **Test sharding and benchmark registry (2026-08-30, `refactor-main`)**:
> the 95-case single-process `uipc.sim_case` remains the authoritative
> global-state-pollution regression. `run_sim_case_isolated.py` now also emits
> JSON manifests and stable sorted round-robin shards for parallel diagnosis;
> a real 4-way discovery produced 24 cases in shard 0. CTest serializes its four
> aggregate GPU executables through the `uipc_gpu` resource lock while allowing
> CPU concurrency. Root `benchmarks/manifest.json` promotes samples example 88
> as `stiff-gipc-case2`; `run_benchmark.py` validates assets/canonical env and
> records both git revisions plus run status. Validation: runner unit tests 8
> passed (complete scripts contract suite 25/25), dry-run/list, and an actual
> one-frame case2 run (return 0, metadata written with hardware/runtime facts
> and parsed frame timing).

> **Optional legacy collision component (2026-08-30, `refactor-main`)**:
> the V0, stackless, and linear-BVH simplex trajectory filters now belong to
> the logical `collision_legacy` component (CMake target
> `cuda_collision_legacy_objects`), separate from the default collision path.
> Compatibility builds retain all 198 CUDA sources; setting
> `UIPC_WITH_CUDA_LEGACY_COLLISION=OFF` or
> `cuda_legacy_collision=false` omits those three registrations and links 195
> sources. The scene schema carries the same build capability and exposes only
> selectors present in the DLL. A CUDA integration test constructs an engine
> for every advertised selector. Validation covered CMake/XMake ON and OFF,
> Python schema tests 5 passed in both configurations, default IPC simulation
> in both configurations, Core 36 cases / 1001 assertions, CUDA backend 13 cases
> / 250 assertions, and the full simulation suite 95 cases / 14212 assertions
> after restoring the default compatibility build. CUDA-only test targets are
> now conditionally included as well, so no-CUDA CMake/XMake configuration no
> longer retains dangling simulation/regression/example dependencies; an
> isolated no-CUDA build passed the none-only sanity suite (3 cases / 50
> assertions).

> **Deterministic SimSystem topology (2026-08-30, `refactor-main`)**:
> backend creators are sorted by complete demangled type name before system
> construction; exact lookup uses `std::type_index` rather than a potentially
> colliding raw hash; compatible derived lookup follows the same order and
> skips invalid variants. Build, invalidation, formatting, and `systems.json`
> now share that order. Active strong-dependency cycles abort initialization
> with the complete cycle path. Validation: new dependency-graph unit tests
> (including disabled nodes and self-cycles), Core/Common CTest, CUDA backend
> 12 cases / 238 assertions, ordinal-sorted IPC and AL-IPC system manifests,
> and full simulation suite 95 cases / 14212 assertions.

> **Single-source scene configuration contract (2026-08-30,
> `refactor-main`)**: `scene_default_config.cpp` now declares each key once,
> including its typed default and schema metadata. `Scene::default_config()`
> and `Scene::config_schema()` are derived from that same contract, eliminating
> the previous parallel default/metadata lists. The normalized public schema is
> byte-for-byte equivalent to the pre-refactor schema. Validation: focused C++
> schema case 668 assertions, Core 36 cases / 988 assertions, and Python schema
> tests 5 passed.

> **CUDA internal component build (2026-08-30, `refactor-main`)**: the CUDA
> backend's 198 compiled sources are partitioned into seven primary domain
> components plus optional legacy collision. CMake uses internal OBJECT targets;
> XMake's matching manifest attaches sources directly to `uipc_backend_cuda` so
> its built-in device-link sees every RDC object. Runtime registration and ABI
> behavior remain one DLL. Configuration rejects missing or duplicate source
> ownership. Validation: clean Release build, CUDA backend 12 cases / 238
> assertions, and full simulation suite 95 cases / 14212 assertions. The XMake
> implementation detail was corrected after the later CI audit above.

> **Backend ABI handshake and artifact parity (2026-08-30,
> `refactor-main`)**: every backend now exports `uipc_query_module` in
> addition to init/create/destroy. Before initialization, Core validates the
> size-versioned ABI record, exact backend identity, and libuipc major/minor
> version, turning stale/mixed DLLs into an immediate diagnostic. CMake no
> longer changes a backend from MODULE to SHARED when tests are enabled;
> CMake and XMake both always produce the same runtime-loadable shared-library
> form. Both none/CUDA DLL export tables contain all four symbols. Validation:
> Core 36 cases / 988 assertions through the new loader path, plus CUDA backend
> 12 cases / 238 assertions.

> **Profile-guided contact/FEM assembly (2026-08-30, `refactor-main`)**:
> Nsight Systems identified the two fused simplex-contact assembly kernels,
> StableNeoHookean3D gradient/Hessian, and shell bending as the dominant raw
> assembly kernels. SNH no longer materializes dense `9x12 dF/dx` and `12x12`
> Hessian matrices: reusable FEM helpers project the energy gradient/Hessian
> directly through tetrahedron shape gradients into the four gradient vectors
> and ten upper-triangular `3x3` blocks. The SNH kernel's per-thread stack fell
> from 6440 to 1320 bytes and its profiled average from 1.795 to 1.047 ms
> (-41.7%). Case-88 `Assemble Subsystems` fell from 3.60 to 2.97 ms/Newton
> (-17.6%) and `Build Linear System` from 7.93 to 7.32 ms/Newton (-7.8%). Two
> clean 60-frame runs measured 156.0-157.0 ms mean and 173.1-173.9 ms median,
> versus 158.1/178.4 ms before this stage; iteration-count variation limits
> the wall-time gain, so the scoped/kernel measurements are primary.
>
> The simplex contact kernels retain one PT/EE/PE/PP launch but compile
> separate gradient-only and Hessian variants. Gradient-only resources are
> normal 106 registers/272-byte stack and friction 112/144, while full
> Hessian performance stays flat (combined profiler average about 4.12 to
> 4.08 ms). A tested per-contact-type split reduced static stack usage but
> serialized rare, individually expensive PT/EE Hessian threads; it regressed
> `Assemble Dytopo Effect` from 4.52 to 7.67 ms/Newton and was rejected.
> Validation: Release CUDA build; CUDA backend 12 cases / 238 assertions;
> focused contact/FEM/MAS/bending 10 cases / 1426 assertions; full simulation
> suite 95 cases / 14212 assertions; targeted compute-sanitizer memcheck 0
> errors and 0 leaked bytes (2 cases / 504 assertions).

> **Device-side line-search energy aggregation (2026-08-30,
> `refactor-main`)**: top-level ABD, FEM, and DyTopo energy reporters now write
> their totals into contiguous device slots. `LineSearcher` performs one final
> CUB reduction into a separate output slot, then downloads all reporter totals
> and the aggregate with one contiguous D2H copy/synchronization. Per-reporter
> finite-value diagnostics and detailed reporting remain intact. ABD and FEM
> retain their existing component reductions but combine those device results
> with named one-thread kernels; DyTopo reduces directly into its assigned
> slot. On case 88 over 60 frames, initial-energy evaluation fell from 0.624 to
> 0.529 ms/call (-15.2%), trial-energy evaluation from 0.542 to 0.448 ms/call
> (-17.3%), and aggregate line search from 7.38 to 7.01 ms/Newton (-5.0%). Wall
> mean/median moved from 162.7/182.3 to 158.1/178.4 ms/frame. Validation: CUDA
> backend build, 11 CUDA test cases / 217 assertions, and the full simulation
> suite (95 cases / 14212 assertions).

> **Batched collision-count readback (2026-08-30, `refactor-main`)**:
> the default `InfoStacklessBVH` exposes launch-only detect/query operations
> plus an explicit result-finalization step. The simplex trajectory filter now
> launches all active PP/PE/PT/EE broad-phase queries, gathers their four device
> counters with one tiny kernel, and performs one contiguous D2H copy/sync.
> Overflow queues retain required-based growth and are the only queries rerun.
> The four CUB selection counts are likewise stored contiguously and downloaded
> once. Thus a fully populated detect/filter cycle uses two count readbacks
> instead of eight; synchronous BVH callers keep their original API. On case 88
> after the discard-growth change, clean-run trajectory detection moved from
> 5.07 to 5.01 ms/Newton and aggregate DCD from 4.67 to 4.61 ms/detect; wall
> mean/median moved 163.7/183.2 to 162.7/182.3 ms but remains within normal
> contact-stage variance. Validation: CUDA backend build, 11 CUDA test cases /
> 213 assertions, and the full simulation suite (95 cases / 14212 assertions).

> **Required-based CUDA output growth (2026-08-30, `refactor-main`)**:
> `cuda_tool::DeviceVector` now distinguishes value-initialized `resize()`
> from `resize_discard()` / `resize_preserve()` and exact or amortized reserve
> operations. Discard growth allocates 150% of the latest requirement, does
> not copy stale contents, and does not initialize ranges that a following
> kernel or CUB primitive completely regenerates. Existing subsystem-specific
> 1.1x/1.5x policies remain in force through exact `reserve_discard()` calls.
> The migration covers matrix-converter scratch, global/DyTopo triplets,
> line-search energy arrays, active-set scratch, and collision candidate/TOI
> buffers; state vectors and buffers with an initialization contract retain
> normal `resize()`. Case 88, 60 frames on RTX 5090, reduced `Scan and
> Allocate` from 80.8 ms total to 38.0-65.5 ms and the two `Compute Energy`
> scopes from 667.1 ms to 390.2-415.8 ms across clean runs. Wall time remains
> contact-sensitive (163.7-171.0 ms mean versus a 169.8 ms baseline), so the
> scoped timers are the reliable result. Unchanged runs diverge at atomic
> roundoff scale and reach about 0.59 mm by frame 60, matching the observed
> baseline-to-change envelope. Validation: CUDA backend build, 11 CUDA test
> cases / 213 assertions, full simulation suite (95 cases / 14212 assertions),
> and compute-sanitizer memcheck (0 errors, 0 leaked bytes).

> **CUB completion and active sparse-format clarification (2026-08-25,
> `refactor-main`)**: the legacy `stackless_bvh` and
> `info_stackless_bvh_v0` builders now use CUB radix sort and exclusive scan,
> with named initialization kernels and separate persistent sort outputs. MAS
> hierarchy prefix scans also moved from Thrust to `cuda_tool::DeviceScan`.
> All of these calls reuse the existing persistent per-stream CUB scratch cache;
> no Newton-iteration path allocates temporary CUB storage per call. Unused
> Thrust compatibility iterators/includes were removed from the CUDA backend.
> The active linear solve remains symmetric 3x3 **BCOO**, not BSR: triplets are
> canonicalized/reduced into `bcoo_A`, and both PCG SpMV paths consume it. BSR
> currently exists only as a container/converter option. Validation on RTX 5090:
> CUDA backend build/link; legacy/default/V0 BVH tests (38 assertions); all ten MAS
> simulation cases (275 assertions); full simulation suite (95 cases / 14212
> assertions); and MAS soft-stitch regression (4 assertions). The monolithic
> build reached the already-linked Python extension, then its local post-build
> package-uninstall step was denied access to the user-level `uipc.exe`; this is
> outside the compiled targets and is not a source/link failure.

> **Wheel CI path filtering (2026-08-25, `refactor-main`)**: prose/docs-only
> pull requests, including `agent_docs/`-only maintenance, no longer start the
> native builds or ten-wheel Python matrix. Release, code-bearing push/PR, and
> manual publication triggers remain unchanged.

> **Single-receiver DyTopo assembly fast path (2026-08-25,
> `refactor-main`)**: pure FEM or pure ABD scenes whose only diagonal
> `DyTopoEffectReceiver` owns the dynamic vertex prefix now forward the raw
> contact/inter-primitive doublets and triplets directly to that receiver.
> The final `GlobalLinearSystem` matrix conversion already canonicalizes and
> reduces those entries, so the former intermediate sort/reduce plus
> classify/copy pass was redundant. Trailing non-DOF global vertices such as
> half planes are allowed; multi-receiver and ABD/FEM coupling scenes retain
> the original conversion/distribution path. On RTX 5090, the clean case-88
> 60-frame run reduced `Compute DyTopo Effect` from 2014.6 ms / 330 Newton
> iterations (6.11 ms/iteration) to 1576.6 ms / 331 (4.76 ms/iteration,
> -22.0%). `Convert To BCOO` stayed effectively flat at 1.85 versus 1.86
> ms/iteration, while wall mean/median moved 175.6/196.4 to 165.5/184.2
> ms/frame. Maximum trajectory displacement versus the CUB baseline was
> 0.55 mm, below the 1.35-1.84 mm unchanged-run variability already measured.
> Pure ABD, pure FEM, and mixed ABD/FEM focused tests pass (736 assertions),
> as do the full simulation suite (95 cases / 14212 assertions) and the MAS
> soft-stitch regression (4 assertions).

> **CUB BVH build hot-path optimization (2026-08-25, `refactor-main`)**:
> the default `info_stackless_bvh` broad phase no longer uses Thrust. Morton
> pair sorting uses `cuda_tool::DeviceRadixSort`, internal-node offsets use
> `DeviceScan`, and identity generation plus all required build-state resets
> are fused into one named initialization kernel. The CUB wrappers reuse their
> persistent per-stream scratch workspace and grow it only when capacity is
> insufficient. Scene-AABB, leaf-LCA, and depth initialization now precede
> their parallel reductions/builds, removing the former cross-block reset
> races. On RTX 5090, case 88 over the same 60-frame phase reduced trajectory
> detection from 7.98 to 4.98-5.00 ms/Newton (-37.5%) and aggregate DCD from
> 7.35 to 4.59-4.60 ms/detect (-37.5%); two runs measured 175.6-177.1
> ms/frame versus 208.1 ms before (-14.9% to -15.6%). The full simulation
> suite passes (95 cases / 14212 assertions), as does the dedicated MAS
> soft-stitch regression (4 assertions).

> **MAS assembly hot-path optimization (2026-08-25, `refactor-main`)**:
> the mesh partition and multi-level MAS hierarchy are now built once during
> engine initialization instead of being restored and rebuilt on every Newton
> iteration. Hessian scatter traverses the BCOO arrays directly, removing the
> per-iteration identity-index allocation/fill/read, and the 48x48 Gauss-Jordan
> kernel no longer executes a redundant block-wide barrier between independent
> row updates. On RTX 5090, sample 88 over the same 60-frame phase and 333
> Newton iterations reduced `Assemble Preconditioner` from 1149.9 ms to
> 834.4 ms (-27.4%, 3.45 to 2.51 ms/Newton); wall mean moved from 213.3 to
> 208.1 ms/frame, with the remaining wall variance dominated by contact stages.
> The full simulation suite passes (95 cases / 14212 assertions); this includes
> all ten MAS sim cases (275 assertions). The dedicated MAS soft-stitch
> regression also passes (4 assertions), as do the runtime-check CUDA build and
> clang-format-18 gate.

> **Repository contracts and source hygiene (2026-08-25,
> `refactor-main`)**: all external GitHub Actions now use reviewed full commit
> SHAs, and the vcpkg action/container revision matches the project's registry
> baseline. A dedicated repository-contracts workflow rejects mutable action
> refs, zero-byte source files, and drift between exported constitution classes,
> pybind classes, and binding initializer registration. Thirteen empty CUDA/C++
> scaffold translation units were removed; the two public zero-byte headers are
> now documented compatibility includes. The full docs helper finds a standard
> Windows Doxygen install even when it is absent from `PATH`, and the local
> preview guide distinguishes deployable API builds from prose-only previews.
> Local validation: full C++/CUDA pybind build; 36 core cases / 988 assertions;
> 79 portable Python tests; 48 non-interactive CUDA tests; 5 repository-contract
> tests; clang-format-18; release-policy/parity/pin/zero-byte checks; and a full
> Doxygen + MkDoxy site containing the `Engine::frame_stats()` API page.

> **Structured solver observability (2026-08-25, `refactor-main`)**:
> the backend-neutral `Engine::frame_stats()` API (also Python) returns `{}` by
> default; CUDA schema v1 reports latest-frame pipeline, completion/convergence,
> Newton and cumulative line-search/linear-solver counts, iteration-limit hits,
> and final line-search/CCD/CFL factors. Python also now exposes the existing
> C++ `Engine::to_json()`, `Engine::status()`, and status `clear()`. Profile runs
> persist one `frame_stats.json` entry per measured frame, and performance
> baselines consume the structured counters as diagnostics instead of scraping
> log messages. Focused none/IPC/AL tests cover optional-backend behavior,
> status access, schema values, and profile persistence.

> **Reproducible performance gates (2026-08-25, `refactor-main`)**:
> `uipc.profile` now excludes warmup/recovery from reported wall time and drains
> warmup Timer data before the first measured frame. Saved artifacts use a
> versioned schema and include the exact phase plan plus runner/build
> compatibility facts. `python -m uipc benchmark baseline/check` (also Python
> APIs) creates deterministic JSON baselines and returns a structured,
> CI-friendly nonzero regression result for wall average and Timer median/p95;
> Newton counts are diagnostic. Checks reject missing scenes, frame/phase drift,
> and environment mismatch unless explicitly relaxed. Focused validation covers
> pass/fail thresholds, overrides, deterministic output, schema rejection,
> environment matching, CLI output, and CUDA/none session integration.

> **IPC/AL frame-lifecycle parity (2026-08-25, `refactor-main`)**: both CUDA
> advance paths now share the ordered external-force lifecycle (clear old
> device buffers, run animation, consume current forces) before DOF prediction.
> AL-IPC no longer enables the process-global Timer or prints a merged report on
> every frame, and its adaptive-mu/CFL stages plus Newton/line-search indices are
> timed/tracked consistently. A parameterized FEM regression exercises the same
> two-frame apply/clear sequence under `ipc` and `al-ipc` and rejects implicit
> Timer output. Local validation: 2 focused cases, 48 non-interactive CUDA
> tests, and 64 portable Python tests pass.

> **Portable local docs builder (2026-08-25, `refactor-main`)**:
> `scripts/build_docs.py` now runs MkDocs through the active Python interpreter
> (`sys.executable -m mkdocs`) instead of assuming a `mkdocs` launcher is on
> `PATH`. This fixes full local builds from ordinary Windows virtual
> environments while preserving the production `mkdocs-with-api.yaml` path.

> **Validated scene-configuration contract (2026-08-25,
> `refactor-main`)**: `Scene::config_schema()` / `Scene.config_schema()` exposes
> all 46 registered keys with defaults, types, units, hard constraints,
> lifecycle/status, descriptions, and source consumers; `python -m uipc
> config-schema [key]` makes it available to tools and agents. Construction and
> `World::init(scene)` validate the contract, including mutable edits and
> cross-field kappa/Newton ordering. The unimplemented
> `newton/use_adaptive_tol` is constrained to `0`, `sanity_check/mode=quiet` is
> now an explicit choice, and non-empty Contact/Subscene extension configs are
> rejected instead of silently ignored. `SceneGUI` consumes the same schema and
> renders reserved entries read-only.

> **Python/CUDA compatibility policy and doctor (2026-08-25,
> `refactor-main`)**: the next wheel matrix covers CPython 3.10–3.14; the
> immutable 0.0.26 release still stops at 3.13. Published builds now target
> 75/80/86/89 SASS plus compute-89 PTX rather than only architecture 89, and
> embed ABI/toolkit/architecture metadata in `build_info()`. The packaged
> `compatibility.json` is checked against both pyprojects and CI. `python -m
> uipc doctor` diagnoses Python ABI, the self-contained CUDA runtime boundary,
> backend dynamic loading, NVIDIA driver/GPU architecture, and optionally a real
> CUDA engine construction via `--probe-cuda`.
> Pytest now defaults to the portable `not example and not cuda` suite; GPU and
> interactive cases have explicit markers, module-stubbing tests restore global
> import state, and every cibuildwheel job executes the portable suite against
> the installed artifact. Local validation covered 59 portable tests and 47
> non-interactive CUDA tests on CPython 3.14.

> **Executable CI and release gates (2026-08-25, `refactor-main`)**: CMake now
> registers all C++ tests with CTest and labels the `common`/`core`/`geometry`
> CPU suite for fast CI execution; the CMake and XMake workflows execute those
> binaries on pushes, pull requests, and manual runs instead of stopping after a
> successful compile. Every built wheel is installed and smoke-tested, TestPyPI
> must expose the complete interpreter/platform matrix and pass an exact-version
> install before formal publication, and PyPI is verified the same way afterward.
> The hosted smoke test uses the no-GPU backend; CUDA-runtime validation remains a
> distinct compatibility gate rather than an inferred result of `import uipc`.

> **Documentation navigation hygiene (2026-08-25, `refactor-main`)**: XMake and
> deterministic-mode guides are now reachable from the site navigation, directory
> links point at explicit index pages, and the RMR/SpreadSheetIO links use valid
> MkDocs source paths. Prose-only builds now leave only expected generated-API
> warnings when Doxygen output is absent. Workflow-dispatch run `32758550867`
> completed the UID, Doxygen, MkDoxy, and MkDocs stages successfully without
> deploying; the docs workflow then moved checkout/setup-python to their current
> Node 24-based v7 majors.

> **XMake parity and deterministic packaging (2026-08-25, `refactor-main`)**:
> stale GUI/torch/RPC configuration was removed, ccache is explicitly disabled,
> and optional OpenUSD/OpenVDB targets now mirror CMake. The pybind target enables
> USD consistently and performs one synchronous source copy plus explicit
> extension/runtime-library copies, eliminating duplicate detached copy races.
> The XMake user guide and build-agent notes describe the current switches.

> **Python packaging and helper parity (2026-08-25, `refactor-main`)**: release
> and development metadata now both include matplotlib, require pytest 9.0.3+ for
> development, and state the prebuilt-wheel CUDA 12.8 runtime requirement. The
> Warp empty-strides fallback calls the real element-size helper. Python now
> exposes the C++ `Scene.Objects.created_count()` ID upper bound, and
> `assets.strip_constitutions` uses it to process sparse-ID scenes. Focused Python
> tests cover the Warp fallback and sparse create/delete lifecycle.

> **Complete UID documentation generation (2026-08-25, `refactor-main`)**:
> `scripts/gen_uid_doc.py` now parses designated initializers and statement-based
> `UIDInfo` assignments, restoring constitution UIDs 15, 17, 31, and 32 to the
> generated specification. A dependency-free unit test covers both forms and the
> formerly missing real registrations. Documentation CI now runs the test plus
> the generator's `--check`, and UID source changes trigger that workflow. The
> docs build wrapper also reports the requested output path and propagates
> MkDocs/Doxygen failures instead of returning success after a failed build.

> **Runtime lifecycle parity (2026-08-25, `refactor-main`)**: an implicit
> synchronization performed by `World::retrieve()` now marks the engine
> synchronized, so repeated retrieves do not issue redundant backend syncs until
> another `advance()`. The `none` backend now enters the Scene pending phase at
> initialization and settles pending geometry creation/destruction on each
> advance, matching the frontend lifecycle contract without pretending to run a
> physical simulation. Core tests cover both behaviors.

> **Evolving-only atlas projection (2026-08-25, `refactor-main`)**:
> `GeometryAtlas::create(..., true)` now filters named collections and every
> collection inside a geometry to slots marked `is_evolving`, preserving
> collection topology and row counts for baseline-dependent streaming. Evolving
> markers now survive attribute clones and atlas JSON round trips (legacy JSON
> defaults to `false`), and Python exposes the same optional argument. Core tests
> cover strict filtering, clone behavior, dimension preservation, and serialized
> round trips; the modified pybind translation unit also compiles independently.

> **Scene lifecycle hardening (2026-08-25, `refactor-main`)**: snapshot commits
> now replicate current/rest geometry independently, explicit slot removals,
> sparse IDs and next-ID state, contact/subscene topology, and the contact default
> model's user-set state. Attribute commits and full attribute serialization also
> preserve row counts when a collection has no columns. Decoders validate
> duplicate IDs, current/rest topology, commit/removal overlap, and invalid atlas
> entries while retaining legacy-field fallbacks. Focused core tests cover
> full-snapshot and commit-JSON round trips, different current/rest mutations,
> deletion plus exact-ID insertion, allocation gaps, table state, and empty
> nonzero attribute collections.

> **Post-merge update (2026-08-23)**: everything below (the whole
> refactor-main line: muda→cuda_tool, raw kernels, Stiff-GIPC alignment,
> cloth stiffness model, kappa policy, hygiene batch) was merged to `main`
> via **PR #468** (merge commit `9bf45950`, 2026-08-22). CI on the PR was
> green at merge time. `refactor-main` is done; new work branches off
> `main`. Sections below are pre-merge history — still accurate as records
> of *why* things are the way they are. For open issues and plans see
> **`09-known-issues-and-roadmap.md`**; for the collected pitfalls see
> **`08-pitfalls-and-debugging.md`**.
>
> Post-merge events:
> - CI repair round on the PR (all pushed, all green): mass clang-format
>   (327 files, `72cf876b`); tinygltf v2.9.6 stale hash → overlay port
>   `ports/tinygltf` wired via `overlay-ports` in the generated
>   `vcpkg-configuration.json` (`bcd04cc2`; env-var wiring does NOT reach
>   cibuildwheel's inner vcpkg install — see doc 07/08); xmake pins
>   `octree v2.5` (`f4230c31`) and `tinygltf <3` (`557f7017`) against
>   upstream v3 layout drift.
> - PR #469 merged: pytest >=9.0.3 (dependabot tmpdir CVE), `uv.lock`
>   regenerated with uv 0.12.5 (large diff — it backfilled missing entries).
> - Samples repo (separate, `spiriMirror/libuipc-samples`): added case
>   **87_robot_hand** (URDF hand + ABD cube, manual GUI posing; ported from
>   references/Robotics-Libuipc and rewritten; scripted auto-grasp removed
>   at user request) and case **88_stiff_gipc_benchmark** (the old
>   `Stiff-GIPC-benchmark.py` moved there with a GUI; `--headless [N]`
>   keeps the benchmark loop; parameters untouched).
> - External PR #461 (EmbeddedCollisionMesh) reviewed — verdict and the
>   three must-fix bugs recorded in doc 09.
> - **FusedPCG CUDA-graph block replay (2026-08-23)**: `check_interval`-sized
>   iteration blocks are captured once per (buffer-set, N, triplet-count) and
>   replayed as single graph launches; case2 250-frame benchmark 301 →
>   233 ms/frame (~1.29×). Config `linear_system/use_cuda_graph` (default 1);
>   `linear_system/check_interval` is now a registered key (default 5 — it was
>   unregistered and silently dropped before). Details and traps in doc 05/08.
>   Notable bugs found during the work: rz_tol async-upload vs graph-launch
>   stream race (sync upload now), triplet_count missing from the graph key,
>   and Timer objects created during stream capture deterministically crashing
>   the single-process suite binary (0xC0000409) — capture path creates no
>   Timer objects anymore (plain path keeps them for case 59's SpMV counts).
>   The al-ipc pipeline is gated off graph replay for now (crash observed only
>   in the C++ suite binary's al-ipc section; python repro passes — root cause
>   open, see doc 09).
> - **MAS preconditioner line (2026-08-23, PRs #473/#474)**: ported from
>   Stiff-GIPC and then hardened. The apply path is stream-plumbed so MAS
>   scenes join the PCG CUDA graph. Activation is now all-or-nothing via
>   scene config `linear_system/fem_preconditioner = "mas"` (default
>   "diag"): every non-Empty FEM geometry is auto-partitioned internally
>   (fixed cluster size 16 = BANKSIZE) on a private clone; the python
>   `mesh_partition` export was REMOVED (partial coverage measured
>   net-negative — the coverage-rule table is in doc 09). Sim cases 53-61/81
>   migrated to the switch. Escape hatch if MAS+graph ever misbehaves:
>   `linear_system/use_cuda_graph = 0`.
> - **Newton exit semantics split (2026-08-24)**: `newton/min_iter` is a pure
>   hard floor (default 0 = off); the semi-implicit beta start moved to
>   `newton/semi_implicit/K_min` (default 1 at that revision; superseded by
>   the 2026-09-01 default of 6 with semi-implicit termination enabled). Found
>   via the case-89 parity run: Stiff averages 2.55 Newton/frame while we forced
>   >=6. Case 88
>   429 -> 320 ms/frame from this alone.
> - **Perf rounds on case 88 (2026-08-24, now ~266 ms mean / ~299 ms
>   median)**: two-level warp->block reduction in `Spmv_rbk_sym_spmv_dot` /
>   `fused_dot` (same-address atomic storms eliminated); exact-distance DCD
>   leaf predicates (kills the 450k-candidate retry double-traversal).
>   Negative result: make_spd Cholesky early-out (register blow-up) — see
>   doc 08. Frame budget and next levers in doc 09.
> - **Samples repo**: 87 robot_hand, 88 case2 benchmark, 89 MAS parity
>   bunny (NO_MAS/NO_GRAPH env A/B), 90-93 = the remaining Stiff set_cases
>   1/4/5/6 (see doc 09 samples section for the mapping + asset notes).
> - **Documentation authentication-popup fix (2026-08-24)**: replacing
>   Bilibili iframes with links removed one popup source, but the deployed
>   site still loaded `polyfill.io` globally from both MkDocs configs. The
>   service now returns HTTP 401, causing a browser authentication dialog on
>   every page. The obsolete ES6 polyfill was removed; MathJax 3 remains
>   loaded from jsDelivr. The docs workflow's path filters were also fixed
>   to watch the repository's actual `mkdocs*.yaml` files (it previously
>   watched only the unused `.yml` suffix, so config-only fixes did not
>   deploy).
> - **Documentation demo separation (2026-08-24)**: the standalone docs home
>   Demos section, Gallery navigation entry, and `docs/gallery.md` were
>   removed so the project-wide showcase lives on the homepage. Focused clips
>   and well-framed poster screenshots remain allowed when they explain an API
>   result. Both MkDocs configurations expose `spiriMirror/libuipc` as the
>   global project repository link in the site header.
> - **Configuration documentation audit (2026-08-24)**: the scene-config
>   reference now enumerates all 46 unique keys registered by
>   `scene_default_config.cpp`, including defaults, SI units, selector values,
>   operational domains, relative-value precedence, lifecycle rules, and
>   the reserved `newton/use_adaptive_tol` key, whose nonzero values are now
>   rejected instead of silently ignored. Dedicated
>   Newton/linear-solver and contact/collision pages trace behavior through the
>   CUDA consumers and explain MAS, graph modes, adaptive kappa, pairwise
>   contact elements, and sanity checks. Navigation exposes all three pages.
> - **Source-backed scenario tutorials (2026-08-24)**: the tutorial landing
>   page now routes to scene assembly, pure ABD, pure volumetric FEM, cloth,
>   and rigid-soft/contact guides. Each physical guide contains complete
>   asset-free C++ and Python programs, parameter/state explanations, failure
>   checks, and links to the matching `libuipc-samples`, C++ regression cases,
>   public headers, and backend attribute consumers. The four Python programs
>   run against the current source build; sample `91_pinned_cloth --headless
>   1` also passes. MkDocs builds successfully and expands every code snippet;
>   strict mode is still blocked only by the pre-existing generated-API nav
>   entries when Doxygen output is absent.
> - **PyPI 0.0.26 CUDA-major runtime finding (2026-08-24)**: installation and
>   `import uipc` succeed, but the Windows wheel's CUDA backend directly
>   imports `cublas64_12.dll`. A machine with only CUDA 13.2 therefore fails at
>   `Engine("cuda", ...)` even though the driver is new enough. The source
>   build on that machine imports `cublas64_13.dll` and works. User-facing
>   install docs and package metadata now say the prebuilt wheel requires the
>   CUDA 12.8 runtime; CUDA 13 users must install 12.8 side-by-side or build
>   from source. A plain import is not an adequate release smoke test.
>
> Older header note (2026-08-20, pre-merge): the muda→cuda_tool migration
> is complete AND fully verified: all apps/tests pass, including the
> 95-case sim suite (2/2 runs, 14214 assertions — same count as the
> pre-migration baseline).
> Verify against the working tree before assuming anything beyond this file.

## TL;DR

- **All of this is merged to `main`** (PR #468, `9bf45950`). The CUDA
  backend no longer depends on muda in any form (no submodule, no vendored
  copy, no xmake package).
- All 273 lambda kernel launch sites were rewritten as named `__global__`
  functions with raw `<<<>>>` launches.
- **All tests green**: 6 fast binaries (common/core/geometry/sanity_check/
  backend_cuda/regression) + `uipc_test_sim_case.exe` full suite 95/95 cases,
  14214 assertions, run twice deterministically.
- (The "uncommitted fix batch" mentioned in older revisions was committed
  and merged as part of PR #468.)

## Commits (oldest → newest, on top of `74a5df62`)

```
ef87325c docs(agent_docs): record muda vendoring completion and fix stale references
b2aec545 feat(cuda_tool): complete primitives for muda replacement
8e3299af refactor(cuda): migrate backend from muda to cuda_tool
423be546 refactor(cuda): rewrite lambda kernels as named __global__ functions
cb9341c1 build: drop the vendored muda from cuda_tool and sync xmake
f6fd6bb3 refactor(cuda_tool): trim unused primitives and refresh agent docs
ee4bea1e refactor(cuda): convert the last lambda kernel and remove ParallelFor
2a8f78d7 refactor(cuda_tool): second trim of zero-reference helpers
```

## Fix commits on top of `2a8f78d7` (root-cause fix for the suite failure)

1. `fix(cuda_tool)` — `launch.h`: `best_block_dim` occupancy cache keyed by
   kernel function address (`std::unordered_map<const void*, int>`) instead
   of a `static thread_local int` per template instantiation (**ROOT-CAUSE
   FIX**, see below); `buffer.h`: `DeviceVector::resize(n)` value-initializes
   the grown tail (memset 0 for trivial types, `T{}` fill otherwise),
   matching thrust/muda resize semantics.
2. `test/build sync` — `apps/tests/backends/cuda/CMakeLists.txt`:
   `/Zc:preprocessor` (CUDA>=13 CCCL requires it),
   `--extended-lambda --expt-relaxed-constexpr` (test .cu use cuda_tool
   launch/dense math), nvcc diag-suppress list; 5 test .cu files gain
   global-scope `namespace cuda_tool = uipc::backend::cuda_tool;` alias
   fixes (`lbvh.cu` uses `copy_from` instead of rvalue copy-init); xmake
   parity (static check only, no local xmake):
   `apps/tests/backends/cuda/xmake.lua` gains the same three flags,
   `src/backends/cuda/xmake.lua` gains `-Xcompiler=/Zc:preprocessor`
   (public, windows block); agent_docs refreshed.

## The full-suite failure and its root cause (RESOLVED)

Symptom: after the migration, `uipc_test_sim_case.exe` (95 cases, one
process) failed deterministically 3/3 at case `36_no_surf_but_contact_on`
frame 17: `Line Search Exits with Max Iteration: 8`. Case 36 run in an
isolated process passed 6/6. Pre-migration baseline (`ef87325c`) full suite
passed 3/3 (14214 assertions).

Root cause: `best_block_dim(Kernel kernel)` cached the occupancy result in a
`static thread_local int` **per template instantiation**, i.e. per function
*pointer type*. Distinct kernels with identical signatures share one
pointer type, so whichever same-signature kernel was queried first set the
block size for all of them. muda's equivalent cache was keyed per unique
lambda type (= per call site), so no cross-kernel pollution existed. In the
full-suite process, 35+ engines ran before case 36 and poisoned shared cache
entries; wrong block sizes on atomic-accumulation assembly kernels perturbed
float atomic-reduction order, shifting the FP trajectory enough to push the
frame-17 line search over the iteration limit. In isolation fewer collisions
occurred, so the perturbation stayed below the threshold.

Fix: cache keyed by kernel address. Verified: full suite 2/2 green with the
same assertion count as baseline; case-36 isolation residual series matches
baseline except residual ULP-level noise (expected: two different binaries
have different address layouts → different atomic arrival order).

Ruled out during the hunt (do not re-open): compile-flag drift (none —
`git diff` of CMakeLists), wrapper-vs-raw occupancy difference (probe with
the verbatim `abd_linear_subsystem_assemble_reporters_k2` body: 256 == 256,
see `output/probe_occupancy2.cu`), eigen port drift (normalized diff vs muda
ext/eigen: macro/namespace renames only, math bodies identical), thrust
calls in bvh (verbatim from baseline), buffer fill/copy block sizes
(per-element ops, no FP effect), stream defaults, cub/cublas call shapes.

## What was done (migration recap)

1. **cuda_tool primitive completion** (`b2aec545`) — stream/view/view_nd/
   launch/buffer/cub/linear_system(+views)/debug/logger/atomic + the eigen
   subdirectory (ported verbatim from muda ext/eigen, numerically
   bit-identical). The `UIPC_KERNEL_*` macro family is enabled together
   with `uipc::RUNTIME_CHECK`.
2. **Mechanical migration** (`8e3299af`) — 280 files `muda::`→`cuda_tool::`,
   umbrella header changed to `cuda_tool/cuda_tool.h`, macros renamed,
   `.name("...")` labels deleted.
3. **Kernel rewrite** (`423be546` + `ee4bea1e`) — 273 lambda kernels →
   named `__global__` (anonymous namespace, bodies verbatim, captures →
   parameters). Launches use `cuda_tool::best_grid_dim/best_block_dim` to
   keep the same occupancy choice; the `ParallelFor` mechanism was
   subsequently removed from cuda_tool (business lambda kernels = 0).
4. **Delete vendored muda + build-system sync** (`cb9341c1`) — deleted
   `cuda_tool/muda/` (288 files) and `muda_compat.h`; CMake dropped the
   MUDA_* macros; xmake dropped `add_requires/add_packages("muda")`.
5. **Two rounds of cuda_tool trimming** (`f6fd6bb3`, `2a8f78d7`) — deleted
   zero-reference primitives and helpers.

- **Performance-investigation lessons (6_wrecking_balls, 619a5412 baseline
  A/B)**:
  1. The real regression root cause: the first-version cuda_tool cub
     wrappers did a per-call `cudaMalloc/cudaFree` of temporary storage
     (~10-100µs each plus an implicit device sync; dozens of cub calls per
     frame) → fixed as a stream-level workspace cache (`cub.h` details);
     frame time 73.9→66.7ms, and the median of the first 8 frames
     (pre-contact) is level with the baseline.
  2. Investigation pitfall a: **build contention thoroughly pollutes
     timing** (sanity_check was once misjudged as the ~112ms/frame culprit;
     on an idle machine it is only ~2-5ms/frame) — timing experiments must
     run on an idle machine.
  3. Investigation pitfall b: after contact activates, frame-to-frame
     phase comparison is meaningless — the two binaries have different
     block sizes → ULP-level differences → trajectory divergence, so later
     frames are no longer in the same physical state.
  4. The ~65ms/frame floor of this scene consists of: dump ~5-10ms +
     per-frame Timer.report/log printing ~10-30ms + sanity_check ~2-5ms +
     the real pipeline 20-45ms (the baseline is the same) — the "feels
     slow" is mostly inherent structure, not a regression.

## Scene-diagonal adaptive parameters (Stiff-GIPC alignment, after `7cf19f21`)

- At `GlobalVertexManager` init the rest bounding-box diagonal
  `scene_diagonal()` is computed (printed to the log; measured 28.93 in the
  wrecking-ball scene, exactly matching Stiff-GIPC's
  √bboxDiagSize2=√834.9).
- New config (default 0 = off, backward compatible):
  - `contact/d_hat_relative`: when >0, d_hat = relative value × diagonal
    (Stiff-GIPC's relative_dhat convention; its dHat stores the square, so
    the original text is rel²·diag²).
  - `newton/velocity_tol_relative`: when >0, the Newton exit threshold =
    relative value × diagonal × dt (MaxTranslationChecker, Stiff-GIPC's
    threshold×diag×dt convention).
- 6_wrecking_balls is now fully parameter-aligned with set_case3 (mu 0.2,
  per-object densities 1000/7680, tol_rate 1e-4, relative d_hat/tol).
  **Note: the previously recorded "advance 26-28ms/frame ≈ Stiff 1.1×" was
  a mismeasurement taken pre-contact / with a polluted dll; the true
  contact-phase behavior once collapsed to 3-18s/frame — root cause and fix
  in the next section.**
- Performance-investigation record: the cub workspace fix + the two
  measurement pitfalls (build contention, trajectory divergence) are in doc
  05 and the performance section above.

## Stiff-GIPC barrier alignment: the log² hard barrier (after `4294c1d5`)

- **Root-cause chain (wrecking ball once at 3-18s/frame, Newton hitting the
  1024 cap)**:
  1. In Stiff-GIPC's incremental potential the barrier term is **not
     multiplied by dt²** (GIPC.cu `computeEnergy`); libuipc's barrier
     coefficient is `kt2 = κ·dt²` → at the same nominal κ the barrier
     strength differs by 1/dt² (10⁴× at dt=0.01). The "set κ=1e4 on both
     sides" alignment was therefore a fake alignment.
  2. Stiff-GIPC's barrier is a RANK=2 log² hard barrier
     `κ(D-d̂²)²ln²(D/d̂²)` (GIPC.cu:28 `#define RANK 2`; `_d_EE` returns
     squared distance); in the deep-penetration regime its force is
     ~2|ln(D/d̂²)|× stronger than the classic log barrier, so under the
     same load the equilibrium gap is ~10× wider, the equilibrium curvature
     ~10× lower, and Newton converges in 3-5 iterations/frame. libuipc's
     classic log barrier (with κ_eff 10⁴× weaker) sinks deep into the
     ill-conditioned D→0 region, CCD crushes α to ~1e-3, and Newton crawls
     linearly for hundreds of iterations.
- **Changes**:
  - `sym/codim_ipc_contact.inl` regenerated (generator notebook
    `scripts/symbol_calculation/codim_ipc_contact.ipynb` updated in sync):
    ξ==0 (volume/ground contact) takes the newly generated log² barrier
    `KappaBarrierLog2` family; ξ>0 (codim shells) keeps the classic
    thickness barrier. The public entry points `KappaBarrier`/
    `dKappaBarrierdD`/`ddKappaBarrierddD` became runtime dispatchers —
    PT/EE/PE/PP, the ground half-plane, and friction normal_force all
    follow automatically, with no call-site changes.
  - samples 6_wrecking_balls: κ=1e8 (equivalent conversion: libuipc κ =
    Stiff Kappa / dt² = 1e4/1e-4).
- **Verification**: the probe converged in all 120 frames; Newton mean
  4.54/max 22 (Stiff 3.44/7); min_alpha 0.15-0.72 (previously crawling at
  ~1e-3); frame average 132ms (was 3-18s); the free-swing segment (f0-80)
  trajectory, after translation-aligning with Stiff, differs by <3mm; the
  full suite is green: 95 cases / 14214 assertions.
- **This section's "remaining gap" was superseded by a later fix**: the PCG
  spikes / pressing-frame crawl recorded at the time were rooted in
  d_hat_relative not being propagated (see "biggest hidden bug" in the next
  section) and disappeared after the fix. For the final comparison basis
  see the **second correction** in the "⚠ comparison-basis correction"
  section (clean-run data).
- **Measurement-pitfall memo**:
  1. post-build syncs the dlls only when pyuipc is relinked — if you only
     change backend .cu files, the `uipc_backend_cuda.dll` in site-packages
     is not updated; you must sync manually:
     `cp build/Release/bin/uipc_*.dll build/python/src/uipc/_native/` and
     `.../site-packages/uipc/_native/`.
  2. pyuipc's `Scene(config)` copies the config dict by value; modifying
     the python-side dict after construction silently has no effect — all
     config keys must be set before `Scene(config)`.
  3. The Stiff-GIPC reference copy block-buffers printf when stdout is
     redirected, so a timeout kill loses logs — an instrumented printf must
     be followed by `fflush(stdout)`.

## Second alignment round (friction smoothing + CFL semantics, after `81e52fce`)

- **`contact/eps_velocity_relative`** (new config, default 0=off): when >0,
  the friction C1 smoothing threshold eps_velocity = relative value ×
  scene_diagonal (Stiff-GIPC convention: its per-step slip threshold is
  sqrt(fDhat)·dt = 1e-2·diag·dt). libuipc's original default was an
  absolute 0.01 m/s, which in this scene makes the friction Hessian
  curvature ~840× stiffer than Stiff's. Implemented in
  `global_contact_manager.cu` Impl::init (same pattern as d_hat_relative),
  default key registered in `scene_default_config.cpp`. 6_wrecking_balls
  and the probe are set to 1e-2. **Real effect (measured after the
  registration fix): Newton mean 4.46→3.70 (Stiff 3.44 — essentially
  aligned), sum_pcg 349→250, frame average 129.9→114.5ms.**
- **Config-key lesson**: scene config only honors keys registered in
  `scene_default_config.cpp`; an unregistered key pushed from python is
  **silently dropped** (`find` returns nullptr and the default branch
  runs). The first version of eps_velocity_relative was never registered,
  wasting a whole alignment-experiment run — a new key must have its
  default registered at the same time, and its taking effect must be
  confirmed via the log line ("Contact eps_velocity (relative): ...").
  **→ Now fixed at the root (see below)**: `from_config_json` now
  recursively checks the user json first and directly throws a guided error
  for unregistered keys ("typo, or a missing default registration"). That
  check immediately unearthed two historical typos/dead keys: ①
  `sanity_check/method` in `apps/tests/core/engine.cpp` (the schema has
  `mode`; the real intent was `enable=0`); ② `contact/al-ipc/mu_scale` in
  `apps/tests/sim_case/11_abd_ramp_sliding.cpp` (that key was split into
  `mu_scale_fem`/`mu_scale_abd`; the old key had been silently dropped all
  along). Note: python already raises KeyError for unknown top-level keys;
  the new check covers nested typos under a valid prefix (such as
  `contact/dhat_typo`).
- **d_hat_relative propagation fix (biggest hidden bug)**: the first
  version only changed `GlobalContactManager`'s scalar d_hat (used by
  CFL/logging); **the per-vertex `d_hats` buffer (what the filter/contact
  kernels actually read) was still filled with the absolute default
  `contact/d_hat`=0.01**. wrecking ball therefore ran at d_hat=0.01 (it
  should be 0.0289) — a too-small d_hat leaves contact inactive at shallow
  gaps, vertices plunge into deep gaps before the barrier stops them, and
  the system becomes ill-conditioned (this was the true source of the
  earlier PCG spikes of 150-565). Fix: at the end of
  `GlobalVertexManager::Impl::init`, propagate `d_hat_relative ×
  scene_diagonal` into the per-vertex buffer (compare-and-set only
  overwrites entries holding the absolute default; per-geometry meta d_hat
  is preserved; `global_vertex_manager.{h,cu}`). Post-fix wrecking ball:
  **Newton mean 1.88/max 6 (Stiff 3.44/7), sum_pcg 56 (Stiff 53), min_alpha
  constant 1.0, frame average 73.0ms (Stiff clean run 42.8ms/frame
  GPU-timed, ~1.7×)**; the set_case2 ported scene's self-contact pair count
  collapsed from 450k (bogus) to ~3.7k (same magnitude as Stiff's 3.5k).
- **CFL semantics correction**: the original implementation only counted
  displacements of "activated contact vertices" (the
  `vert_is_active_contact` mask) — measured zero triggers over 120
  wrecking-ball frames, because vertices hurtling toward contact at high
  speed happen to be outside the mask and can plunge into deep gaps in one
  step. Changed to the Stiff-GIPC design: max|dx| covers **all surface
  vertices** (`GlobalSimplicialSurfaceManager::surf_vertices`, falling back
  to all vertices when there is no surface manager); and in the
  `advance_ipc.cu` line search it is applied only when CCD hits
  (ccd_alpha<1) — avoiding needlessly capping free-flight frames. In this
  scene it still never triggers in practice (vertex steps within contact
  iterations are mostly under 5cm) — this is a semantic alignment whose
  value lies in high-speed-impact scenes.
- **Current state of the remaining frame-time gap**: after the d_hat fix,
  wrecking ball averages 73.0ms/frame, about **1.7×** Stiff's clean-run
  42.8ms/frame (GPU event timing, frames 2-120). The early "PCG spikes"
  (150-565 iterations/solve) were in fact caused by small-d_hat deep gaps
  and vanished with the d_hat fix (sum_pcg now 56 ≈ Stiff 53).
  Verification means on record: linear-system dump (config
  `extras/debug/dump_linear_system=1`) + scipy recomputation.

## Stiff-GIPC set_case2 ported benchmark (now samples case `88_stiff_gipc_benchmark`; formerly `examples/Stiff-GIPC-benchmark.py`)

- Scene: ABD bunny (scale 0.2, y+0.5, ρ1000, ABD κ=1e8) + FEM bunny (same
  mesh, y-0.65, SNH E=1e4/ν0.49/ρ1000) + cloth (cloth_high.obj 4225
  vertices, E=5e4/ν0.49/ρ200/t=1e-3/strain_rate=100, bending value-matched
  to E=5e7→κ_b=5.48e-3) + ground y=-1; μ=0.2, κ=1e8 (=Stiff 1e4/dt²),
  dt=0.01, g=-9.8, d_hat_relative=1e-3, velocity_tol_relative=1e-2,
  eps_velocity_relative=1e-2, tol_rate=1e-4, **semi_implicit enabled
  (Kmin=6, beta_tol=1e-2 — Stiff's beta early-exit design: hard stacking
  frames exit capped at ~6 Newton iterations by design)**.
- Mesh assets: bunny2.msh was converted to standard Gmsh 2.2 (the original
  file is Stiff MeshProcess's nonstandard 7-field variant, unreadable by
  libigl; the tetrahedron content is bit-identical) and copied into samples
  assets (tetmesh/bunny2.msh + trimesh/cloth_high.obj).
- **Results (250 frames)**: libuipc averages 301ms/frame (stacking phase
  steady at ~310ms, no spikes), Newton capped at ~6 (=Kmin); Stiff clean
  run on the same scene: **142.8ms/frame** (GPU event timing, 447-frame
  average; frames 2-250 is 171.8ms), Newton mean ~2.4 (under that counting
  basis). So the case2 gap is about **1.8-2.1×** (not the previously
  miscomputed 5× — that came from misaligned frame grouping when
  recomputing marginal quantities on an instrumented run). The remaining
  gap is per-iteration throughput: this scene's dense bunny surface (6mm
  spacing after 0.2 scaling) makes each detect's AABB candidates ~450k
  (only ~3-8k active after distance filtering) — the separation of
  candidate generation from activation filtering is structural overhead,
  and fused/distance-aware queries are the follow-up optimization lead;
  there is also a uniform gap in PCG iteration count and kernel throughput
  (dissected in the next section).
- Note: cloth contact in libuipc takes the thickness-offset barrier (the
  ξ=1e-3>0 branch); since 238df28e the barrier shape is unified to log²
  (the thickness branch uses the log² form of the shifted distance (D-ξ²));
  Stiff has no thickness concept and is uniformly log² — this is the
  closest semantic equivalent.

## Line-search pre-cap alignment (feasible-step pre-cap, after `3982c6bb`)

- **Stiff-GIPC design** (GIPC.cu:10941-10973): the line search first
  computes `alpha = min(1, ground_feasible(0.8), self_feasible(0.8, MCP))`
  — an exact CCD cap over the currently active contact set (each pair keeps
  ≥20% of its current gap) — and **then** generates the full CCD trajectory
  candidates on the capped step (smaller swept boxes → fewer candidates);
  CFL intervenes only when CCD pairs exist (`h_ccd_cpNum>0`), with the
  floor semantics `alpha = max(alpha, alpha_CFL)` (prevents
  over-crushing).
- **libuipc implementation** (`global_contact_manager.{h,cu}` +
  `engine/advance_ipc.cu`): `GlobalContactManager::compute_feasible_step()`
  reuses the project's own
  `distance::{point_triangle,edge_edge,point_edge,point_point}_ccd`
  (`utils/distance/ccd.h`) to compute the CCD-TOI pair by pair over the
  active PT/EE/PE/PP pairs exposed by `SimplexTrajectoryFilter`'s public
  accessors (eta=1-slackness=0.2), with a single DeviceReduce().Min
  reduction; alpha is capped after the line search's record_start_point and
  before `detect_trajectory_candidates(alpha)` — trajectory candidates are
  generated on the capped step and shrink accordingly. The compound
  semantics of the `filter_toi` lambda was also fixed: the filter returns a
  fraction of the swept step; the absolute step = alpha·toi.
- **Verification**: in a weak-κ synthetic test (where the barrier cannot
  hold), the pre-cap triggers correctly (alpha=0.047/0.011/0.012); the
  wrecking-ball probe shows zero regression (71.7ms/frame, Newton 1.84,
  min_alpha constant 1); Stiff-side reference (instrumented): its feasible
  step triggers 273 times / 40s on case2 (alpha 0.2-0.4). Rare triggering
  in a well-conditioned aligned scene is correct behavior (under a strong
  barrier the Newton direction does not overshoot the active pairs); its
  value lies in under-converged / high-speed-impact scenes.
- **CFL floor not aligned (deliberately deferred)**: Stiff's
  `alpha = max(alpha, alpha_CFL)` floor semantics can push the step past
  the CCD hit point and requires an isIntersected-style crossing test as
  backstop (ground signed distance + edge-face crossing, D=0 contact
  legal). libuipc's current filter D>0 assertion would kill the process
  outright, so before adding the floor, penetration detection must first be
  changed to crossing semantics — follow-up standalone work.

## Kernel-level dissection of case2's remaining ~2× gap (nsys evidence)

Frame time ~245ms/frame (stacking phase), of which GPU kernels are busy
~115ms — **host-side API overhead is about half**: per frame 7429 kernel
launches + 522 memcpys + 1787 memsets + 563 stream syncs. Breakdown
(/frame):
- FusedPCG 68.6ms: ~83 iterations/solve × 118µs/iteration; of that, the
  kernels actually compute only ~36µs — the rest is launch gaps (7
  kernels/iteration in serial dependency + a convergence D2H sync every 5
  iterations draining the pipeline). → Lever: cooperative-groups
  persistent-kernel fusion (1 launch/iteration) or CUDA graph capture of
  the PCG inner loop; estimated 2-3× reduction.
- BVH self-queries (stacklessSelf 1.34ms + stacklessOther 0.89ms ×
  ~14/frame ≈ 31.5ms): the dense bunny surface yields ~450k candidates per
  detect. → Lever: fuse exact distance tests into the query predicate and
  materialize only active pairs.
- Contact assembly do_assemble ×2 ≈ 26ms; SNH G/H 2.12ms×7 ≈ 14.9ms
  (Stiff's equivalent FEM assembly kernel is 0.77ms/call — a 2.75× gap; the
  make_spd 9×9 EVD is a suspect, but removing it makes convergence worse —
  do not simply delete it); cloth DSB 8.8ms.
- Stiff same-scene nsys: ~900 launches/frame (libuipc is 8× that), FEM
  assembly 0.77ms/call.
- Negative result: `linear_system/check_interval` 5→25 has no effect on
  frame time (206 vs 205.7ms) — PCG's pipeline-drain stalls are not the
  main cost, don't spend time here; PCG's cost is mainly the iteration
  count itself (83/solve vs Stiff 27/solve, determined by the system
  condition number).
Conclusion: case2's ~2× is the product of structural host overhead +
multi-kernel throughput, and needs a dedicated round of kernel
fusion/graph-capture engineering (every change must pass the full-suite
regression).
- **⚠ Comparison-basis correction (second, final)**: the Stiff log's
  "average time cost" = totalTime/totalNT (GIPC.cu:11262) — it is the GPU
  time **per Newton iteration** (totalNT/totalTime/total_Frames are
  file-level globals accumulated across frames), NOT per-frame time! Also,
  the PAIRCOUNT printf/fflush I added to the reference copy inflates its
  GPU event timing. **Clean reference numbers (uninstrumented runs):
  wrecking ball 42.8ms/frame (frames 2-120, GPU event timing; full-run
  1654-frame average 50.4ms), case2 142.8ms/frame (447-frame average).**
  Corresponding final comparisons: wrecking ball libuipc 73.0ms vs
  42.8-50.4ms ≈ **1.5-1.7×**; case2 libuipc 301ms vs 142.8ms ≈ **2.1×**
  (same 250-frame window: 171.8ms → 1.75×).

## Cloth stiffness model update + strain_rate exposure (after `b7056879`)

> **Partly superseded on 2026-09-01:** the historical area multiplier and
> one-sided stretch/bending formulas below were corrected as recorded at the
> top of this handoff. They remain here only as the chronological explanation
> of the regression.

- Cloth stiffness formulas aligned with mas-pncg; membrane-element weights
  use the triangle **area** (not volume, avoiding incorrect volume-measure
  weighting of the thickness-independent shear):
  - stretch: `StrainLimitingBaraffWitkinShell`'s triangle attribute
    `"lambda"` is written as `(λ+2μ)·t` (identically `E·t/(1-ν²)`); in the
    backend `strain_limiting_baraff_witkin_shell_2d.cu`, the measure in
    both energy kernels was changed from `area·2t` to pure `rest_area`.
  - shear: attribute `"mu"` = `E/(2(1+ν))`, thickness-independent (no t
    under an area measure).
  - bend: the `DiscreteShellBending` family (including strain/stress
    plastic variants) unified its measure to area `V_bar = A` (3 function
    headers); the κ semantics of raw `apply_to(sc, κ)` became "stiffness
    per unit area" — **all raw call sites were migrated to κ×t to preserve
    physical consistency** (libuipc tests 33/82-87 and regression all
    t=0.001; samples 11/24/26/33ext/34 at their respective thicknesses).
    The formula overload `apply_to(sc, E, ν)` and the static helper
    `bending_stiffness(E,ν,t)` write the literal value `E·t³/(12(1-ν²))`.
  - strainRate: no longer hardcoded to 100 — `apply_to(...,
    strain_rate=100)` writes the triangle attribute `"strain_rate"`, and
    the backend reads the attribute (old scenes missing it get it
    auto-created and backfilled with 100); pybind exposure synced.
  - **stretch/shear material-parameter separation**:
    `StrainLimitingBaraffWitkinShell::apply_to` dual-modulus overload
    `apply_to(sc, stretch_moduli, shear_moduli, ρ, t, strain_rate)`
    (stretch uses `(λ_s+2μ_s)·t`, shear uses `μ_sh`, each with independent
    (E,ν), aligned with mas-pncg ClothMaterialConfig); the old
    single-modulus overload is kept for compatibility. samples
    11_bunny_cloth/34_cloth_stack now use the separated parameters (shear
    softened 100:1), and 11's bend additionally demonstrates the formula
    overload; the new
    `apps/tests/core/strain_limiting_baraff_witkin_constitution.cpp` covers
    the dual-modulus attribute layout.
  - **Single source of truth for thickness**: the DSB formula overload is
    `apply_to(sc, E, ν)` — bending is optional, stretch is required, so
    thickness is set only by the membrane constitution (vertex
    `"thickness"` attribute); DSB reads it averaged over edge endpoints
    (naturally supporting non-uniform-thickness shells); a clear error is
    raised when it is missing.
- Verification: Python smoke numerics all correct; DSB-related sim_case 33,
  82-87 all pass; the 6 fast binaries all pass; the full suite passes;
  `11_bunny_cloth` headless OK.
- Note: `ElasticModuli2D`/`EP_to_lame_2D` and other constitutions such as
  NeoHookeanShell are untouched (scope is only the cloth BW strain-limiting
  shell + DSB); the BW shell is unused by any suite/sample, so the measure
  change has zero regression surface.

## Hygiene & test-robustness batch (after `d2f48087`)

- **Duplicate-include sweep**: 17 files had identical `#include` lines
  (mostly migration-cruft `cuda_tool/cuda_tool.h`); deduped. The
  `geometry_export_types.inl` double-include in `geometry_factory.cpp` is an
  intentional X-macro pattern — do NOT dedupe it.
- **Catch2 v3.8 filter syntax (measured)**: multiple specs as separate argv
  are AND-intersected ("No tests ran"); OR requires comma-separated specs in
  ONE argv: `uipc_test_sim_case "0_abd_gravity,13_fem_3d_gravity"`.
  `--list-tests --verbosity quiet` prints one case name per line.
- **`file(GLOB ... CONFIGURE_DEPENDS)`** added to all 44 project CMake files
  (external/ untouched) — adding/removing sources no longer needs a manual
  re-configure.
- **Line-search diagnostics**: the "Line Search Exits with Max Iteration"
  warning and the strict-mode exception now carry
  `alpha_last / E0 / E_last / rel_E_increase / ccd_alpha / cfl_alpha`
  (`engine/advance_ipc.cu`, `engine/advance_al.cu`) so a threshold-crossing
  can be judged as real regression vs ULP jitter from the log alone.
- **Isolated suite runner**: `scripts/run_sim_case_isolated.py` runs each
  sim case in its own process (`--filter/--start-from/--timeout`) —
  complements the single-process suite to separate cross-case global-state
  pollution from case-local failures.
- The CMake-side ccache integration from this session was implemented and then
  reverted at user request. The later XMake audit also removed its stale
  `dev=true` ccache policy; both build paths now keep compiler caches disabled.

## Build-time optimization (after `88965feb`)

- **Umbrella split**: `cuda_tool/cuda_tool.h` no longer includes `cub.h`
  (CCCL device-algorithm headers add ~165K preprocessed lines per TU); the
  ~23 files using `Device*` wrappers include `<cuda_tool/cub.h>` explicitly
  (found via API grep over `DeviceReduce(/DeviceScan(/...` + `cub::`).
  `linear_system.h` also dropped its (unused) cub.h include.
- **RDC correction roller-coaster**: d2f48087 once turned RDC off claiming
  "no cross-TU device symbols" — **that conclusion was wrong**. The
  `UIPC_GENERIC` free functions in `affine_body/utils.cu` (`q_to_transform`
  etc.) are called by kernels in other TUs, so disabling RDC guarantees
  `ptxas fatal: Unresolved extern`. It went undetected then because
  CMake+ninja does not track flag changes, so stale RDC-on objects slipped
  through the link; the pyuipc build triggered a broad recompile and
  exposed it. `CUDA_SEPARABLE_COMPILATION/RESOLVE_DEVICE_SYMBOLS ON` has
  been restored and xmake gained `-rdc=true`. The "full rebuild ~4.7 min"
  figure in the next entry is also affected (some TUs were not recompiled)
  — for reference only.
- ~~**RDC off**~~ (corrected, see above)
- Measured (32-core, CUDA 13.2): full rebuild wall **~4.7 min** (was ~10 min
  perceived), CUDA TU CPU 8.3K→6.5K s, per-TU avg 38→35 s. Line-count
  attribution of a non-cub TU (~1.63M lines after -E): CUDA toolkit headers
  ~860K, WinSDK ~310K, MSVC STL ~150K, Eigen ~150K, project <20K — the
  remaining cost is toolchain headers, not project includes. This session once
  listed ccache (`CMAKE_CUDA_COMPILER_LAUNCHER`) as a possible next lever; the
  owner subsequently rejected compiler caches, so rule 8 supersedes that idea.
- Verified after both changes: 6 fast binaries pass; full sim suite passes
  (14214 assertions / 95 cases).

## Environment notes (unchanged)

- Build: `output/build.bat` (vcvars64 + `cmake --build build --config
  Release --target sim_case -j8`); for full error collection use
  `output/build_keepgoing.bat` (`ninja -k 0`); all test targets via
  `output/build_all_tests.bat`.
- nvcc needs the MSVC environment; a bare shell reports "Cannot find
  compiler 'cl.exe'".
- Configure: `cmake -S . -B build --preset ci-release
  -DUIPC_BUILD_BENCHMARKS=OFF -DUIPC_BUILD_EXAMPLES=OFF` (you must pass `-B
  build`; after removing/adding source files you must re-configure, because
  file(GLOB) is expanded at configure time).
- Filtering Catch2 by multiple case names does not work on this machine —
  run them one by one; single-name filtering works (e.g.
  `./uipc_test_sim_case.exe "36_no_surf_but_contact_on"`).
- `output/test_compile.cu` + `output/compile_smoke.bat` are the standalone
  smoke compile/run entry points for cuda_tool
  (`src/.../test_compile.cu.txt` is the in-repo archive).
- Occupancy probe: `output/probe_occupancy2.cu` + `.bat` (a
  `cudaOccupancyMaxPotentialBlockSize` comparison template of bare kernel
  vs muda-wrapped kernel).
- Full-suite logs: `output/test_sim_case_fix1.log` / `_run2.log`
  (post-migration, all passing); baseline reference
  `output/test_sim_case_baseline.log`.

## Default kappa policy (after `3982c6bb`)

- Rule (user requirement): if the user never calls `default_model(...)`, the
  effective default contact stiffness is `contact/adaptive/min_kappa`
  (default 1e8); if the user set it, the value is clamped into
  [min_kappa, max_kappa] (defaults [1e8, 1e11]) with a warning that prints the
  valid range; negative kappa (adaptive-kappa opt-in) is never clamped and
  takes precedence (the GIPCAdaptiveParameterStrategy path).
- Implementation: `ContactTabular` tracks `default_model_is_user_set()` (new
  public getter, additive); the policy is applied at
  `GlobalContactManager::Impl::_build_contact_tabular` when the host-side
  coeff table is filled, so the device table always carries the resolved
  values while the adaptive strategy keeps reading the core attribute for its
  negative-marker detection (unclamped by design).
- Verified: smoke test of all five cases (unset / below-min / above-max /
  in-range / negative marker) behaves exactly per spec; full sim suite
  95/14214 + 6 fast binaries green. Note the built-in default stiffness
  effectively changes 1e9 -> 1e8 for scenes that never set the default model.

## Wheel CUDA architecture list never reached nvcc (2026-09-03)

- Symptom: `pyuipc` 0.0.27 wheels (cp312 and cp313 manylinux both checked)
  contain `sm_75` SASS only and zero PTX, although `pyproject.toml` and
  `compatibility.json` declared `75/80/86/89-real` plus `89-virtual`. On an
  RTX 5090 (`sm_120`) `world.init(scene)` throws CUDA error 500
  `named symbol not found` from `cuda_tool/launch.h`.
- Root cause: `set_target_properties(... PROPERTIES ... CUDA_ARCHITECTURES
  ${UIPC_CUDA_ARCHITECTURES} ...)` did not quote the variable. An unquoted
  multi-element list expands into separate arguments and destroys the
  key/value pairing, so the property kept `75-real` and the remaining entries
  became bogus property names (`80-real` ended up as a property whose value
  was `86-real`). The `89-virtual` PTX entry was lost the same way.
- Fixed by quoting three sites: `src/backends/cuda/CMakeLists.txt`,
  `src/backends/cuda/components.cmake`,
  `apps/tests/backends/cuda/CMakeLists.txt`.
- Release matrix now also carries `120-real` for consumer Blackwell, mirrored
  in `python/src/uipc/compatibility.json` so
  `scripts/check_release_policy.py` stays green. Note this grows wheel size
  and CI compile time by one full architecture.
- Verification: configured with the full list and confirmed all 199 CUDA
  translation units in `compile_commands.json` now carry six
  `arch=compute_*,code=*` entries including `code=[compute_89]`; before the
  fix only `sm_75` was emitted. Reference build with `native` on the 5090
  (`sm_120`) runs `hello_affine_body` and the Python `0_check_libuipc` sample
  to completion.
- XMake was deliberately left alone (rule 7 reviewed): its CUDA arch surface
  is `add_cugencodes("sm_89")` under `github_actions` and
  `add_cugencodes("native")` otherwise, passed as single values with no
  multi-arch list, so it has neither the quoting defect nor a release matrix
  to mirror. XMake does not build the published wheel; the PyPI path is
  scikit-build-core plus CMake.
- Follow-up in the same area: the option's documented comma form
  (`-DUIPC_CUDA_ARCHITECTURES=75,89`) was normalized into
  `CMAKE_CUDA_ARCHITECTURES` only, while the backend targets read the option
  itself, so the comma reached nvcc and the first `.cu` failed with
  `'89' is not in 'keyword=value' format`. The root `CMakeLists.txt` now
  normalizes the cache entry in place. A `get_target_property` fast fail on the
  `cuda` target guards the truncation case; configure was checked with the full
  six-entry list, the comma form, and `native`.
- Post-merge hardening on `refactor-main` replaces all three raw property sites
  with `uipc_set_target_cuda_architectures`, which sets and reads back the
  property for the final library, every component OBJECT target, and the CUDA
  test target. A repository contract prevents any target kind from bypassing
  the helper. Wheel compatibility now distinguishes the CUDA 12.x SASS driver
  floor from the CUDA 12.8 PTX-JIT floor; `uipc doctor` reports the selected
  code path and no longer labels an old-driver PTX-only GPU compatible.
  Validation configured the comma-form release matrix and found all 214 CUDA
  translation units in the test-enabled build carrying all six codegen flags,
  then restored `native`. CMake and XMake production builds, fast CTest 3/3,
  repository contracts 48/48, portable Python tests 80 passed / 1 skipped, the
  real Python 3.14 CUDA doctor probe, and the complete documentation build all
  passed.

## 2026-09-08: local cloth/cable setup

Created a Python 3.11 environment using the pyuipc 0.0.28 wheel and added
`python/examples/cloth_cable_manipulation.py`. A 180-step CUDA run validated
cloth and rod soft-grip movement, finite state, and target tracking. Setup,
commands, results, and the neighboring Genesis integration are documented in
`agent_docs/cloth_cable_setup.md`. No native solver build was performed.

## 2026-09-08: IPC robot manipulation pretraining package

Added `python/uipc_manip/`, a point-cloud SAC pretraining stack in which a
Franka Panda manipulates deformables whose contact is solved by this project's
IPC backend. Genesis 1.1.2 supplies the robot, plane, and viewer; the cloth
sheet and elastic cable are native libuipc geometries in the Genesis IPC
coupler, reached through the same private attributes the upstream Genesis IPC
examples use. Parallelism is one IPC scene per subprocess.

The agent, hyperparameters, and transition semantics are ported from the Newton
cloth-dressing teacher on `leomessikun/fmvp-sac-retrain` (Wang RSS 2023 as used
for FMVP simulation pretraining): scalar SAC with twin critics and the
reference `Q(encode(s), a)` form, a learned entropy temperature, the
horizon-equivalent discount, temperature learning rate and replay reward-scale
helpers, the replay-prefill gradient budget, time-limit bootstrapping from the
pre-reset observation, and a checkpoint protocol that refuses a mismatched
observation layout or another task. The reference PointNet++ requires PyTorch
Geometric, absent from the Genesis environment, so it was reimplemented on
dense masked tensors with padding and permutation invariance tests.

Two measured findings set the defaults. Three simulation steps per decision
leaves the deformable lagging the tool, and the scripted policy then scores 0/6
on both drag tasks; five restores 6/6. Truncating an oversized marker set by
vertex index biased the observation to one region of the sheet while the reward
measured the whole of it, also 0/6; random subsampling restores 6/6 at a 3.3 mm
mean final error. All three tasks pass the scripted gate.

Throughput on a shared RTX PRO 6000 Blackwell: the IPC solve is 68 ms of a
120 ms Genesis scene step, four subprocess workers reach 4.4 environment
transitions per second, and a SAC update costs 47 ms at the 256-point budget.
Evidence is in `agent_docs/performance/2026-09-08-uipc-manip-pretraining.md`.
No libuipc solver code was changed and no native build was performed; the
package runs against the released `pyuipc` 0.0.28 wheel.

Second pass the same day, after the owner questioned the throughput and the
fidelity of the port. Two findings were confirmed against the code. The Newton
teacher's transition budget comes from batched simulation, and the Genesis
IPC coupler already supports its counterpart: `N` deformable copies in one
libuipc world, isolated by subscenes, with batched robots. The environment was
rewritten that way and the subprocess wrapper removed; 32 copies in one world
step 222 environment steps per second against 22 for one, and per-episode
results are unchanged under matched seeds. The active reference actor is
`WangFlowActor`, a segmentation PointNet++ read at the tool point, not the
flat global actor first ported; it is now the default, with the categorical
`flashsac` critic and a set-transformer encoder as measured options (19 ms per
update against 122 ms for the reference encoder). Evidence is in the same
performance record.

Third pass: with batching and the reference actor in place the policy still
did not learn, and the cause was reward scale. Rewards of at most 0.06 per
decision were dominated by an entropy term near 0.43, where the reference
calibrates for per-step rewards in `[-1, 1]`. Progress is now measured in
units of `max_translation`. At 8,000 transitions the same run went from 0/32
successes and 97.8 mm to 27/32 and 14.6 mm, matching the scripted baseline.
The viewer path (`--vis`) holds the window open after evaluation and exits
cleanly when it is closed.

Fourth pass: the owner clarified that the target is the Newton dressing task
itself, not simpler cloth tasks. `python/uipc_manip/dressing_env.py` rebuilds
that MDP on the batched IPC world from the Newton bake cache: fixed affine-body
arm collider (eroded 6 mm), strain-limiting Baraff-Witkin garment, twelve
anchored cuff vertices following the 6-D action, the Wang line-triangle reward
with the 5x upper-arm term, the dual-camera visible cloud with jitter and
dropout, and the 900-step horizon. The multilevel additive Schwarz
preconditioner is required (block-Jacobi PCG cost seconds per Newton
iteration), the arm must be a native body (Genesis re-tessellation produced
NaN distances in the trajectory filter), and 17 of 23 cells build. The cached
scripted pull threads the sleeve only over the fingertips (forearm ratio 0.04),
so the reachability baseline is weak; the trainer runs end to end with
per-garment dressing metrics. Details are in the performance record.

Fifth pass: the Newton seven-stage dressing expert is ported as the
`--policy heuristic` reachability baseline for the dressing task. On IPC it
threads the opening over the fingertips but stalls at a forearm ratio of
0.08 (reference friction), 0.17 (friction 0.1), 0.22 (2 cm erosion); the
upper arm is never reached. Geometry admits threading, friction and fingers
are not the blocker, the hard-pin hypothesis is under test, and the first SAC
evaluation at 28,800 transitions shows no insertion. The performance record
has the table. Trajectory files now carry the static arm mesh and the preview
renders it.

Sixth pass: the cuff hold was a blocker. libuipc's soft position constraint
strength is a rate times the vertex mass, and at the library default of 100
the held cuff lags the tool by centimetres in free air and is left more than
a metre behind once the sleeve touches the hand, so every earlier
reachability number measured a detached anchor. At the new default of 1e4
the expert dresses the whole forearm and 0.28 of the upper arm. The
environment now reports the real held-vertex error as `tracking_error`, a
GPU test asserts the hold tracks through free air, and the strength-100 SAC
run was stopped as invalid (two evaluations, 0 of 16 each). Training is
relaunched with the fixed hold. Friction, erosion, and cloth-model sweeps
were all taken with the detached anchor and are void, not ruled out.

Ninth pass: the dressing task is solved by a learned policy for the first
time. With the cloth corrected to the drape bake's measured stiffness (6e3
stretch, a hundredth of that in shear, bending 0.1) the curriculum teacher at
horizon 150 with six simulation steps per decision reaches a 0.50 success
rate at 24,008 replay transitions. That number needs its caveat: with one
cell per garment, a fixed snapshot reset and a fixed evaluation seed block the
effective sample size is two, and the peak is one garment finishing at 0.7528
against a 0.70 threshold while the second garment never exceeded 0.065. The
following evaluation's zero is the same garment at 0.4541, not a collapse:
training success rose across that window. The same run at the baseline
stiffness had 0.00 and less than half the forearm coverage, so the cloth
finding stands. Selection now ranks the continuous ratio, evaluation reseeds
per round, and the temperature is logged. Details, the speed work, and the
open problems are in the September 9 correctness record; what training one
policy over many cells still requires is in the September 10 protocol record.

Tenth pass: one policy over many cells is wired end to end. The environment
takes a (garment, body) per slot from the Newton cache or live, where live
drapes the garment in libuipc and places it on an SMPL-X body generated from a
seed. The trainer plans slots, reserves whole bodies from training (their slots
step and are evaluated but never reach replay), summarises evaluation per cell
with held-out and training blocks, ranks checkpoints on the held-out continuous
ratio first, and refuses a resume onto a different plan. `--cell-source live` is
the new default, so single-body commands need `--cell-source cache`. Two live
blockers were found and fixed: a ground plane the task never used, which the
hanging garment hit, and a Genesis initialisation order, where starting cuBLAS
before gs.init (generating a GPU body first) makes Quadrants assert. A sixteen-
cell world of tshirt_26 and tshirt_4 on bodies 0-7 steps at 1.32 s per decision
with a 3.0 mm grasp error. tshirt_68 and tshirt_392 do not yet build live, and
the settle is a free fall because the episode pins twelve anchors where the
bake pinned the grasp patch and the opening.

Thirteenth pass: Wang's pretraining protocol is in the codebase as
`python -m uipc_manip.pretrain_wang teacher | student | resume`. A slot's cell is
fixed by its reset snapshot, so Wang's per-episode draw is a world rebuild; a gate
showed the process returns to the same GPU footprint after every teardown, and
`close()` now destroys the scene and deletes its libuipc workspace. Held-out poses
are scored in evaluation worlds kept for the run. Re-reading the reference: the
launcher runs `curl/train.py`, whose teacher keeps one 400,000-transition buffer
without a curriculum and whose student trains one shared temperature, since
`train_multi_garments.py` cannot run against the checked-in agent. Those are the
defaults; per-garment buffers, per-buffer temperatures and the curriculum are
flags. `--dt` and `--settle-steps` reach the dressing config. Section 6 of the
protocol record has the mapping, the gate and the costs.

Twelfth pass: the SAC update's ball query no longer goes through `cdist`,
which cut an update from 310 to 86 ms in the running job and a vector step from
9.5 to 4.15 s. `--critic-input privileged` adds an asymmetric critic on a
35-float arm-frame simulator state in place of point clouds; it runs as a
controlled variant beside the unchanged baseline, as recorded in the correctness
record. `--encoder-precision bf16` lowers only the point encoders; the two
options together make an update 2.85 times faster. A profile at the run's
configuration puts 98% of `env.step` in the libuipc solve and finds 62% of each
point cloud is padding. Cutting a batch to its valid prefix is exact and makes
the network passes about three times faster.

Eleventh pass: live cells are dressable. A sixteen-cell expert check showed the
tenth pass's placement was not: the opening sat 20 cm out on the fingertip-to-
shoulder chord, one opening radius off the forearm, and the expert got the sleeve
over the hand on one cell of sixteen. The live factory now puts the socket on the
forearm axis at a per-cell clearance from an exact whole-arm check, all four
garments build on bodies 0-7, and the anchor default is 48. Over 300 decisions
the expert reaches the upper-arm threshold on five of eight tshirt_26 bodies and
tshirt_4 reaches nothing; the other garments' ceilings follow in the correctness
record, which also records why a stiffer grip and differentiable simulation were
rejected.

Thirteenth pass: tshirt_4 and tshirt_392 are placed the way Wang places them.
The tshirts' opening
polygon is the armhole seam, not the sleeve's cuff, and the live factory had
turned every garment so the opening faced the fingertips, which laid the sleeve
along the forearm with its cuff toward the hand: the expert's forearm ceilings
followed the cuff radius (9.3, 6.6, 6.0 and 3.8 cm on tshirt_26, tshirt_68,
tshirt_4 and tshirt_392), not the opening's. `dressing_live.SLEEVE_OUTWARD_GARMENTS` now
places tshirt_4 and tshirt_392 with the insertion axis along the forearm, the
torso over it and the sleeve pointing away from the hand. They hang as baked and
use the online drape, as Wang's reset and Newton's cached cells do. The expert now
reaches the forearm on all eight bodies of each, where before it reached none,
and passes the upper-arm threshold on three tshirt_4 bodies. tshirt_26 and
tshirt_68 keep the flip. Placed sleeve outward they also reach the forearm on
every body, but they lose their upper-arm ceilings: the opening stops at the
elbow, where the environment's no-move rule drops nearly every `elbow_hook`
step. The hospital gown keeps its placement, although its opening shows the same
signature (a 5.5 cm cuff 12.7 cm out on the sleeve side), and it hangs as baked
through `hang_as_baked_garments`, with which the table composes. The flip had also
started tshirt_392 upside down (156 to 180 degrees). tshirt_68 now hangs as
baked too. On bodies 0 to 7 the expert reaches its forearm on 8 and passes the
upper-arm threshold on 5, against 5 and 3 on the measured socket. The ceilings are
in the correctness record.

Seventh pass: the Wang RSS 2023 / FMVP simulation pipeline is ported around
this encoder and solver: the garment curriculum gating replay writes, the
decision-rate flag, FMVP's early-turn detector (bend-plane form), the
rollout collector with the paper's two filters, and NLL distillation into a
student that the unchanged evaluator plays. All three stages ran end to end
on the GPU with the scripted expert and a lowered filter; the paper's 0.7
filter keeps nothing yet because no policy ends dressed. The Newton
reference itself was compared on this machine: 4.4 times faster per
transition and no better than one success in 40 after 2.98M transitions.

Eighth pass (a Codex session working in the same tree, integrated here): six
correctness fixes to the dressing port, recorded in
`agent_docs/performance/2026-09-09-dressing-correctness.md`: overflowing
observations no longer drop the garment, resume restores the saved
experiment and refuses silent protocol changes, CSV logs keep their history
and late columns, a failed simulator step aborts before polluting replay,
evaluation covers every slot equally, and the curriculum budget counts only
admitted transitions. `diagnose_dressing.py` records reproducible expert
traces. Two of the earlier record's interpretations were withdrawn: the
"solver-independent plateau" and the spring-period reading of the hold
strength. GPU grasp measurements follow in the correctness record.

Gown pass (branch `agent/garment-gown`): the hospital gown's live cells started
upside down. The measured socket's roll turns its drape 114 to 123 degrees from its
baked hang on SMPL-X bodies 0 to 7, so the 0.64 kg gown fell through the settle, was
still swinging when the expert approached, and met the hand with its opening 4 to 5 cm
off the forearm and squeezed to 6 cm. The expert reached the forearm on none of eight
bodies, and the held-cuff drift of up to 148 mm was the caught rim, not the grip.
`LiveCellConfig.hang_as_baked_garments` now re-rolls the gown alone through
`gravity_aligned_socket`; every shirt keeps the measured socket, which a CPU test pins
exactly. Over 300 decisions on bodies 0 to 7 the expert now reaches the forearm on 7
of 8 gown bodies and passes 0.7 on the upper arm on 5, with a worst held-cuff error of
56 mm against tshirt_68's 52 mm; tshirt_26 (8 and 5 of 8) and tshirt_68 (6 and 4 of 8)
did not drop. `DEFAULT_GARMENTS` in `dressing_env.py` still calls the gown unusable
after Newton's notes. Evidence and the variants tried are in the correctness record's
"The hospital gown started upside down".

Integration pass: the four parallel branches landed on `cloth-cable-manip-rl`.
Each was fast-forwarded or cherry-picked, then pushed after the CPU suite passed:
- The Wang pretraining infrastructure (`fb38b326`, `a1da4af3`, `9f8591bb`):
  `pretrain_wang.py`, a rotating world, per-buffer replay and temperatures, a
  held-out eval world, and `--dt`.
- The gown's baked hang (`353e92e6`).
- The camera-rig observation modes (`62233509` to `541498b0`). No rig decodes
  the task state better than `visible_dual`; see `2026-09-10-camera-rigs.md`.
- The tshirt placement fix (`30441b28`): tshirt_4 and tshirt_392 sleeve outward
  as Wang places them, and tshirt_68 in its baked hang.

The merged tree passes the four CUDA smoke tests. The gown note under
`DEFAULT_GARMENTS` is corrected (`a9bf76c6`), and the time-step probes are in
`2026-09-10-dressing-timestep.md`.

A resumed live-cell run now also keeps its garment placement.
`reconcile_resume_placement` compares `pre_insertion`, `hang_as_baked` and the
per-garment lists against the checkpoint. It refuses a training resume onto a
changed placement and prints the difference in playback. Checkpoints written
before the lists existed read them as empty.

Still open:
- Orientation differs across garments. tshirt_26, tshirt_68 and the gown keep
  the cuff-first flip; tshirt_4 and tshirt_392 are sleeve outward, as in Wang.
- tshirt_392 reaches 0.7 of the upper arm on none of eight bodies. Its 3.8 cm
  cuff is narrower than a hand.
- Under the sleeve-outward placement the expert stalls at the elbow: the 12 mm
  no-move rule drops almost every `elbow_hook` step. A proximity push fixed 3 of
  4 cells in the garment agent's probe, but it is not merged.

Orientation pass:
- tshirt_26 now starts sleeve outward, as Wang places it.
- The scripted expert applies its `last`-stage lift during `elbow_hook` too, where the
  12 mm no-move rule otherwise drops almost every step.
- tshirt_68 and the gown keep the cuff-first flip in their baked hang, a recorded
  deviation, because sleeve outward costs them three and two elbows.

Over bodies 0 to 7 at dt 1/60 the expert passes 0.7 on the upper arm on 23 of 40 cells:
tshirt_26 6, tshirt_68 5, the gown 7, tshirt_4 5 and tshirt_392 0. Before this pass it
was 18. The tables are in the correctness record's "tshirt_26 goes sleeve outward, and
the expert lifts at the elbow". The dt probes in `2026-09-10-dressing-timestep.md` keep
the simulation step at 1/60 s.

The committed configuration (`638e63f6`) then measured 21 of 40 on the same bodies,
not 23: tshirt_26 5, tshirt_68 5, the gown 6, tshirt_4 5 and tshirt_392 0, with the
forearm reached on 39 of 40. The correctness record has the table.

Wang pretraining launch, 2026-09-11 01:45 CEST:
- `output/uipc_manip/launch_wang_chain.sh` trains the teachers for regions 13, 4 and
  22 one after another, then the student on all three. A crashed stage resumes from
  its latest checkpoint, up to four attempts. A second process on this GPU buys 1.16
  times the aggregate throughput while slowing each run by 64 per cent
  (`2026-09-08-uipc-manip-pretraining.md`), so the stages run back to back and
  region 13 shows whether the task is learnable before the other two start.
- Every stage passes `--obs-mode wang_static_arm --no-obs-augment` and otherwise runs
  Wang's defaults: 24 environments, 600k transitions, one 400k buffer for a teacher
  and one per region for the student, a shared temperature, horizon 300 at six 1/60 s
  steps, the 25 held-out configurations evaluated every 10k transitions and a
  checkpoint every 50k.
- The chain imports from the detached worktree `.claude/worktrees/pretrain-638e63f6`.
  Runs land in `output/uipc_manip/wang_teacher_r{13,4,22}_s1` and
  `wang_student_r4-13-22_s1`, logs in `output/uipc_manip/logs/`, and a finished stage
  gets `CHAIN_DONE`. `watch_wang_chain.sh` next to the launcher prints checkpoint
  saves, failures and a 45-minute stale-log alarm.
- The exact command passed a smoke first: 4 environments, horizon 20, 320
  transitions, four rotations, three evaluations, two checkpoints, exit 0. The first
  world took 528 s because it baked the garments online; later worlds rebuilt in 21 to
  30 s.

The launch was stopped by hand at 05:03. Region 13's teacher had slowed from 3.6 s to 130 s
per vector step late in episode 2, and then its log went silent. The run is kept as
`output/uipc_manip/wang_teacher_r13_s1_stalled_20260911`.
- **The hold is not the cause in these probes.** With the expert, random or persistent actions the
  held patch stays within millimetres while the cost climbs. The learned policy later proved
  otherwise: it drags a caught sleeve and the hold error reaches 179 mm (see the tether entry below).
- **The cost is the linear solve.** PCG iterations per Newton step rise with contact and with large
  actions, and a 24-cell world pays for its hardest cell.
- **Contact settings and step size do not change it.** d_hat 3 mm, kappa 1e6 and half the
  translation per decision each leave the PCG work per decision within 10 per cent.
- **The fix.** `DressingConfig` now caps Newton at 128 iterations per step. It also has a decision
  watchdog, at 8 times the recent median and at least 30 s. A trip is an ordinary simulator error, so
  `pretrain_wang` rebuilds the world.
- **The expert check holds.** With the cap it passes 22 of 40, against 21 without.
- **`anchor_tether_m`.** It exists but stays off. From relaunch 3 on it is on; see the tether entry below.

The record is `2026-09-11-dressing-solver-stall.md`.

Relaunch 1 started at 06:44 from `953b0cff`.
- **Trips.** It tripped the watchdog in both episodes where the policy acted, at steps 215 and 124.
- **No evaluation or checkpoint.** The loop ran its schedule only when an episode reached its horizon,
  so the run never evaluated or saved. `300d743c` runs the schedule on the trip path too.
- **PCG tolerance.** In an expert replay, 5e-2 was 14 per cent slower than 1e-2.

Relaunch 2 started at 08:43 from `.claude/worktrees/pretrain-300d743c`, with `--checkpoint-every 10000`.
The stall record has the tables.

Once the policy acts, the teacher runs at about 2 transitions per second. On this GPU, with an
evaluation every 10k, a 600k teacher takes about five days.

Relaunch 2's checkpoint at 14,400 transitions found the slow tail, and the chain was stopped at about
12:00 with 19.6k transitions. The run is kept, resumable, as
`output/uipc_manip/wang_teacher_r13_s1_relaunch2_notether_20260911`.
- **The tail is the hold.** Replayed by `stall_probe.py --policy checkpoint`, the actor leads a caught
  sleeve and the held vertices trail their targets by up to 179 mm. Decisions 150 to 182 average 25 s,
  128 s at worst, against 4 s before.
- **A tether on the patch centre leaks.** It checks the translation only, so the centre gap sat at
  its 5 cm limit while the grasp rotation moved the far vertices another 35 mm (median).
- **The fix.** `anchor_tether_m` now bounds every held vertex and drops the whole move, rotation
  included. It defaults to 0.06 m, just above the scripted expert's 59 mm ceiling. On the same
  replay the hold stops at 60 mm, the tail averages 5.8 s and the run takes 12.9 minutes, against
  21.1 with the centre tether.
- **The expert check holds.** With the tether it passes 22 of 40, as without it.
- **It needs a fresh run.** Resume restores the saved `DressingConfig`, so an older run keeps its
  unbounded picker.
- **The reproducer.** The checkpoint and the probe scripts are in `output/uipc_manip/reproducers/`.

Relaunch 3 started at 12:53 from `.claude/worktrees/pretrain-d19489c8`, fresh, with the tether on. At
13,920 transitions it had used 4,539 s against relaunch 2's 6,452 s. Its first evaluation, at 14,400,
scored a held-out forearm ratio of 0.48 and upper-arm ratio of 0.054 with no simulator error. The leftover
agent process that had shared the GPU since the night before was stopped at 14:37 at the user's request.

The barrier-free AL-IPC pipeline (Zheng et al. 2026) is already in libuipc as
`contact/constitution = "al-ipc"`. On the dressing scene it is 2.2 times slower than IPC over 137
decisions of the learned policy, 191 s at worst in contact, and its `diag_norm` mode aborts on a negative
time of impact. The stall record has the table. The teacher keeps IPC.

The pretraining infrastructure was reviewed from first principles in
`performance/2026-09-11-pretraining-infrastructure-review.md`.
- **The largest levers** are parallel teachers and evaluation off the critical path.
- **Newton pretraining is not worth it.** At matched decimation Newton is only 1.1 to 1.7 times
  faster, and its VBD cloth differs at the elbow.
- **Contact forces** are available through `ContactSystemFeature`, but not yet calibrated or wired
  into the reward.

At the user's request, several AL-release artifacts were deleted: its build, its worktree, the CUDA
12.8 conda env, vcpkg with its cache, and the `wiso-enoji` refs.

Five research tracks (dressing literature, sample-efficient learning, fast cloth simulators, an
audit of Wang's repository, the expert as a data source) fed
`performance/2026-09-11-training-infrastructure-proposal.md`.
- **Keep IPC.** No simulator documents a tenfold speedup on cloth-on-body contact together with
  transfer evidence.
- **The teacher changes what it sees and where it starts:** a privileged 35-float MLP actor and
  critic, replay seeded with expert episodes, a bounded residual on the expert as the first probe.
- **The student** is Wang's recipe (SAC loss plus the teacher term on its own rollouts) on Stretch 3
  clouds.
- **Step 0 first.** End the expert's last stage on progress. Its target 10 cm past the shoulder can
  push the opening out of the metric's view, and the reward then falls off a cliff (one of four
  tshirt_26 successes, for 75 decisions). Then measure the expert on region 13 under the final-ratio
  metric, which nobody has done.
- **Wang's final horizon is 150** (`curl/launch_train_curl.py:241`), against our recorded 300, so the
  current path costs 7 to 14 GPU-days per region-13 teacher.
- **The GPU switch** is folded into relaunch 3's 300k decision (around 2026-09-12 15:00).

Step 0 of that proposal was built on 2026-09-12 while the region-13 teacher kept running.
- **The expert stops on the reading.** Its last stage ended on a target 10 cm past the shoulder,
  where Wang's upper-arm ray no longer hits and the reward falls to minus the fingertip's distance
  from the opening; `dressing_heuristic` now ends it on `GenesisIPCDressingEnv.progress`.
- **Evaluation worlds can run without the watchdog** (`DressingConfig.decision_watchdog`); it voided
  two of the teacher's first twelve rounds, the second with no other job on the GPU.
- **`expert_baseline.py`** measures the expert on a region under the evaluation metric and records
  every decision's privileged state, action and reward for a demonstration replay.
- Nothing reaches the running teacher until it restarts, which is the 300k decision.

Relaunch 3 was stopped on 2026-09-12 at 15:13 at 265,584 transitions, 26.4 hours in, with the user's
agreement. Why, in the order that decided it:
- **The evaluation stopped measuring.** 14 of 27 rounds hit the decision watchdog, which raises for a
  whole world, so one slow configuration ended all 25 episodes and they were scored at their
  last-seen mid-pull readings.
- **Those readings then took `best.pt` twice.** `checkpoint_score` never read `sim_errors`, so a
  voided round at 0.302 displaced the best honest round at 0.283, and a later one at 0.367 displaced
  that. Both carry `sim_errors` 25 and zero successes.
- **Most episodes no longer finished.** In the last 39,672 transitions, worth 5.5 episodes, 3
  finished against 7 trips. The horizon is where the reward is earned and where the time limit
  bootstraps.
- **It was self-reinforcing.** A deeper sleeve is a harder contact, so the better the policy got, the
  more often the watchdog fired.
- Over 13 valid rounds the teacher scored 5 successes in 325 episodes, 1.5 per cent, with no trend.
  That is 265,584 transitions, about 880 episodes, against Wang's roughly 13,000 for this region: 7
  per cent of his budget, and a dirty 7 per cent.

Wang has none of these: FleX costs the same per step, so no watchdog is needed; his `simulator_error`
means an explosion, not a timeout; and one environment per process means an error ends one episode.
His teachers also read the same point cloud as the student (`SAC_AWAC.py:1011`,
`teacher_obs = non_randomized_obs`), only without the randomisation, and his `asymmetric_ac` flag has
no implementation in the training code.

Both GPU smoke tests pass on the fixed tree. Step 0's held-out expert baseline is running.


## 2026-09-12 — Contact-force takeover and training audit

Recovered the current Claude session and public subagent outputs. Added a reproducible
loaded-particle calibration command and eight CUDA regressions: the installed backend
exports zero friction after one-iteration termination, but tighter convergence recovers
the known tangential load at dt 1/60 and 1/120. Hardened exporter availability, numeric
validation and single-instance attribution. CPU suite: 184 passed; new CUDA matrix: 8 passed.
See [audit](performance/2026-09-12-force-learning-audit.md) for evidence, corrected paper
claims, IsaacIPC ideas and unfinished task groups. No training or native solver change
was made; dressing-force validity and zero-success training remain open.


## 2026-09-12 — Deep research: force learning and elbow failure

Compared primary dressing, tactile, asymmetric-learning and constrained-control papers; audited
code at `4bfe88c2` and raw evaluation artifacts. Archived hashes and all27 teacher evaluation rows:
only13 have no simulator errors; best valid mean is at125016, final265584 has25 errors. Expert
final success11/25 differs from ever-success15/25 and paper-filter0/25. Corrected misleading
September11 endpoint and garment-mean interpretations. Actual gamma is0.995.

The [research report](performance/2026-09-12-force-training-research.md) distinguishes reusable
force-training baselines from a falsifiable spatial-information contribution. The experiment
specification uses matched state/action budgets and strong wrench/shuffled-map controls; no
policy or branching experiment was launched. No code changed or new tests were warranted.

Status on archival, 2026-09-13: `024c5716` subsequently closed per-decision force training after
failed repeatability measurements. All five companion reports carry that superseding notice.
The dense critic fix (`ef1b3c81`) and optional residual trunks (`01bf913e`) postdate this audit.

## 2026-09-13 — Recurrent pretraining branch and design

Created `pretrain/recurrent-force-memory` from `01bf913e`. Two parallel audits reviewed the
RLT primary report at upstream `1bee93a9` and the current teacher/student training integration.
The upstream repository supplies a technical specification, no implementation or measured gain.
The [proposal](performance/2026-09-13-recurrent-pretraining-proposal.md) specifies episode-linked
replay, external rollout states, causal sequence distillation, and a history-aware dense Q function.
Compare the corrected feedforward baseline, H4/H8, GRU, then a small RLT-inspired alternative.
Per-decision force labels remain excluded after the failed gate. No runtime changes or GPU jobs;
validation is document/link/JSON consistency, not a new training result.

## 2026-09-13 — Episode-aware replay (recurrent pretraining, stage 1)

Implemented the proposal's data contract without touching the actor, critic or SAC loss.
`FlatReplayBuffer`/`ReplaySet` accept `sequence=True`; `ReplayStreams` allocates episode
identities per vector slot and closes them on reset, world rotation, evaluation and simulator
error. `sample_sequences(length)` returns exact same-episode windows with the true pre-reset
successor. Snapshots carry schema version 1 and rebuild links on load, including into a smaller
capacity; flat and sequence snapshots stay deliberately incompatible. Both trainers expose
`--sequence-replay`, default disabled.

The [record](performance/2026-09-13-sequence-replay.md) states the collection boundaries,
window semantics and persistence rules. Validation is 64 CPU tests including a stub simulation
through failure, rotation and resume; no GPU job ran and no learning improvement is claimed.
Upstream RLT was re-checked on 2026-09-13: still `1bee93a9`, still no implementation.

## 2026-09-13 — Padded windows and streaming rollout state (stage 2, part 1)

`sample_sequences(..., pad=True)` left-pads windows whose episode began fewer than `L` steps
earlier, so every recorded transition becomes a learning step. Without it the first `L-1` decisions
of every episode would never be trained on, while a deployed policy has to act through exactly
those steps with an empty history. `SequenceBatch` now always carries a `valid` mask; padded
positions are zero and carry identity `-1`. Padding never reaches into an earlier episode and does
not change windows that already existed.

`history.RolloutHistory` is the collection-side counterpart, holding the raw prefix rather than
encoded features so that a weight update cannot leave it stale. `length=1` is the single-frame
policy and the parity setting for the history-aware model still to be written. The parity gate
passes for `L` in 1, 2, 4 and 7: streaming reproduces the sampler's padded window at every
decision, mask included. 269 CPU tests pass; no GPU job ran, and no learning change was made.

## 2026-09-13 — Ordered feature history in the policy heads (stage 2, part 2)

`FrameHistory` in `models.py` is the proposal's H4/H8 memory: each cloud is encoded spatially on
its own, and the resulting frame vectors are concatenated in decision order with their validity
flags and the commands recorded between them. Padded frames are zeroed and flagged, so an episode
opening stays distinguishable from a frame that encodes to zero. `Actor`, `WangFlowActor` and the
dense `Critic` take `history_length`; `1` is the default and is the single-frame network exactly —
same parameter names, same outputs — which is the parity the H4 comparison against `abl_dense_s1`
depends on.

In the dense critic every earlier frame is encoded with the command actually recorded after it and
only the current frame sees the candidate action, so scoring a candidate never rewrites the observed
past and `dQ/da` flows as before. A history on the rejected `latent` critic is refused. The module
holds no parameters, so the target critic has nothing new to track under Polyak. `SACConfig`, the
sequence update and the trainers do not expose the head yet; the default runtime is unchanged and
no GPU job ran.

## 2026-09-13 — History-aware SAC update and rollout interface (stage 2, part 3)

`SACConfig.history_length` (default 1) now reaches the agent. Above one, `SACAgent` draws padded
windows from sequence replay and trains one learning step per window: its last transition, seen
through H frames. The successor window advances the history with the command actually recorded in
replay before the current policy proposes its next candidate, so a Bellman target never rewrites
the observed past. Rollout state lives outside the network — `make_history(num_streams)` hands each
collector, evaluation world and teacher its own `RolloutHistory`, and `act(obs, deterministic,
history)` reads it and records the decision.

Unsupported combinations fail at construction, before a world is allocated: flashsac, the
privileged and latent critics, teacher distillation onto a history-aware student, and stochastic
observation augmentation, which is not temporally consistent across a window. `protocol()` carries
`history_length` only when it is not 1, so earlier checkpoints still load, and a consumer that
cannot carry rollout state can refuse the key.

289 CPU tests pass, including H4 end to end on both actor types over a toy vector rollout, and the
H1 parity gate: the default agent acts, updates, saves and loads exactly as before. The trainers do
not expose the flag yet and no GPU job ran.

## 2026-09-13 — Frame history reaches the trainers (stage 2 complete, unmeasured)

`--history-length H` on both trainers (the Wang launcher forwards it) builds the H-frame policy
and requires `--sequence-replay`. `history.rollout_state`/`act_with` keep the single-frame path
byte-for-byte: no state is created and `act` is called as before. Above 1, the training world,
each evaluation world, `expert_baseline` and `collect_rollouts` own a `RolloutHistory`, emptied on
episode end, rotation, evaluation, simulator error and resume; warm-up commands enter it like policy
commands; `evaluate` resets only the finished slots. `distill` refuses a history teacher because
causal sequence BC is not written. The [record](performance/2026-09-13-sequence-replay.md) carries
the exact H4 command against the running dense/plain ablation; it was not launched, the GPU being
held by that ablation, which the proposal's stage 0 gate requires to finish first. The stage 3 GRU
pilot and the RLT-inspired variant remain proposals.

## 2026-09-13 — Review follow-up on the frame history

Three corrections from review, no behaviour change at H=1. Padded sequence sampling in `ReplaySet`
now follows the flat `sample` rule exactly — uniform over the buffers holding more than a batch,
the whole batch from one — so an H4 run and the dense ablation draw garments and temperatures
alike, and the comparison is not confounded by a second sampling rule. `FrameHistory` zeroes
padded frames with `torch.where` rather than multiplication, because a padded frame is an empty
cloud and an encoder may return NaN for it; the H4 agent test now also runs the transformer
encoder and checks every actor parameter stays finite. `--history-length` without
`--sequence-replay` is refused right after argument parsing in both trainers, before Genesis or
any world is built; the stub tests assert no world was constructed. 293 CPU tests pass.

## 2026-09-13 — RLT fork audit

The owner asked whether RLT itself was used: it was not. This branch implements the proposal's
stages 1–2, episode replay and a finite ordered frame history, which the proposal places before
any RLT model. The fork `awdemos/recurrent-looped-transformer`, pushed today, holds a demo-scale
Rust/candle language-model implementation of the paper; the proposal now records what it is, what
it is good for (an executable specification of state, eviction and exact replay) and what a stage 4
head in our `history_input` slot would consist of. No code from it was taken and no GPU job ran.

## 2026-09-13 — Should the SAC structure change? The REAL lab's line, read at the source

The owner asked whether Shuran Song's group's work means our SAC infrastructure should change. The
[record](performance/2026-09-13-prior-and-residual-rl.md) reads DICE-RL (ICML 2026), Latent Policy
Barrier, Compliant Residual DAgger, Gated Memory Policy and the force-adaptation paper at the source
and puts the answer in one number: the reference solved this task with about five million
transitions and we collect about fourteen thousand an hour, while the lab's line spends 100k–600k
online steps refining a *usable prior*. We have no such prior — the scripted expert is deterministic,
11/25, 0/25 filter passes — so the first change is a prior gate (bounded random residuals on the
expert; needs a small `--residual-scale` flag) and, if it fails, a per-cell privileged parameter
search producing a one-off dataset; only then residual RL at a matched budget against the running
dense ablation. Two conflicts are recorded as decisions: the dense critic's 36 ms update against
DICE-RL's UTD 10–20 on a frozen encoder, and open-loop action chunks against the 6 cm tether. No
code changed, no GPU job ran; DF-ExpEnse could not be read.

## 2026-09-13 — G14 closed: the expert fails the reference filter in its own `middle` stage

A CPU probe rebuilt the 25 held-out expert bodies from their ids (arm block matches to 2 mm),
mapped the recorded tool positions to the world and re-scored the early-turn flag under our
bend-plane criterion, FMVP's horizontal-plane intent and the Newton port's literal XZ. Every
dressed episode is flagged under FMVP's own plane, so 0/25 filter passes is not a criterion
artifact; the flag first fires in the scripted `middle` stage (or `approach`) on the arm's inner
side, while `elbow_hook` and `last` alone pass 8/11 (ours) and 11/11 (FMVP). The
[record](performance/2026-09-13-early-turn-filter-audit.md) names the cheapest fix: an outward
offset of the `middle`-stage path, rerun on the same cells. Probe at
`output/uipc_manip/early_turn_probe.py`; no GPU job ran.

## 2026-09-13 — RLT in the pretraining infrastructure (stage 4 built, unmeasured on GPU)

The owner rejected the deferral: "how could we use RLT in our pretrain infra". The
[record](performance/2026-09-13-rlt-in-pretrain.md) answers with code on the branch: `rlt.py`
implements the report's equations 2.1–2.16 (causal encoder with memory groups, gated merge,
decoder blocks of sliding-window attention → memory cross-attention → FFN, optional tying) with
three execution schedules — window `run`, per-position `branch`, streaming `step` — held equal by
tests; `--history-kind rlt` puts it in the history slot of the actor and the dense critic, and
`SACAgent._update_rlt` learns at every recorded position of a padded window under the report's
replay contract (recorded pass rebuilt under current parameters, candidates branched from the
recorded state, successor advanced with the recorded command, nothing detached inside the window);
`TrajectoryPretrainingHead` is the 5.1 objective with continuous targets over all recorded
trajectories. CPU cost probe: per learning position the RLT-H8 window costs 1.6× a single-frame
transition and 2.8× less than the H4 frame history. Not built, named in the record: the pretraining
trainer command (the objective is a tested function only), recorded sequence data, a frozen
spatial encoder arm and a cached feature column for whole-episode replay, and unconditional
`priv` recording in sequence mode for the pretraining target. Commits `30bc39b1..` on
`pretrain/recurrent-force-memory`; no GPU job ran (`abl_dense_s1` at 115k/125k).

## 2026-09-13 — Independent review of RLT pretraining at dc953a34

Three bounded audits reviewed model equations, replay/SAC semantics and primary recurrent-RL
literature. Fixed new-run RLT learning to endpoint sliding-window semantics; preserved legacy
prefix checkpoint/launcher behavior explicitly. Strict context rejects evicted or gapped prefixes
instead of pretending they are episode openings. Endpoint target history receives Polyak updates.
Action-conditioned prediction heads now separate optional behavior regression (zero default),
use component element means and robust padding; streaming KV follows model dtype. The cost probe
measures full optimizer cadence and actual spatial forwards/valid positions. A bounded 40-point CPU
probe is archived; it is not a GPU speed prediction. Added optional privileged-target recording
independent of the point critic. No GPU training launched or existing job interrupted.

See [review](performance/2026-09-13-pretraining-infrastructure-review.md) for retained limitations,
source papers and the recommendation to prioritize a reusable trajectory corpus and elbow roll-in
curriculum before larger memory. No offline trainer, usable prior or learned elbow gain is claimed.
Final full CPU validation: 344 passed, 14 CUDA-marked tests deselected; one existing invalid-evaluation
fixture warning. New launcher arguments pin endpoint mode, while old saved RLT commands retain prefix.

## 2026-09-13 — Offline representation pretraining (review item 2, taken over)

The review's author had left `SACAgent.initialize_representation` uncommitted at 19:39; this
entry commits it with the sending half. `python -m uipc_manip.pretrain_offline` reads sequence
replay snapshots (one buffer or a garment-split set), splits whole episodes into training and
validation, fits target statistics on the training rows, and trains the actor's spatial encoder —
and at `--history-kind rlt` its recurrent history, on its own learning rate — through the
action-conditioned trajectory head; validation reports each target against the constant
predictor's loss on the same windows, which is the number the "does history improve prediction"
question reads. `--init-representation` on both trainers adopts the encoder (and history) into a
fresh run, leaving trunk, critic, temperature and optimisers new; refused with `--resume`, skipped
on a Wang resume, recorded in checkpoint metadata as `representation_init`. The protocol check
ignores `rlt_learning_mode`, an online setting. Tested on a toy corpus only: no run has recorded
sequences with `priv`, so no real corpus exists and no online run has started from a pretrained
representation. The [record](performance/2026-09-13-offline-pretraining.md) has the diagnostic
commands. Full CPU suite: 351 passed, 14 CUDA-marked deselected. No GPU job launched; the critic
ablation chain moved on to `abl_residual_s1` on its own.

## 2026-09-13 — Physics gradients as control signals: Level 1 probe (new line, branch `research/physics-gradients`)

The owner rejected further recombination of known recipes and asked for a novel contribution to
policy training. The line opened here uses the simulator's own optimizer: IPC solves
`x⁺ = argmin E(x; u)`, so `∂x⁺/∂u = −H⁻¹ ∂²E/∂x∂u` with the Newton Hessian the solver already
assembled and the gripper entering through `SoftPositionConstraint`. Novelty is stated narrowly
(DiffCloth did differentiable assisted dressing, DiffIPC the adjoint through IPC, SHAC the
actor-critic split): IPC barrier contact + sleeve threading + closed-loop policy learning.
`python -m uipc_manip.physics_gradient_probe` (Level 1, black box) drives the expert to its
elbow / passed / stall states on one cell, dumps and restores them (`World.dump`/`recover`), and
measures repeatability, locality (central differences at 1, 2, 5 mm, three repeats) and
usefulness (twelve 4 mm decisions along each gradient, the expert, random, hold), recording the
translation the environment actually executed. Three cells, seven states, ~45 shared-GPU
minutes. Findings: decision noise is state-dependent (five states ≤ 0.2 mm, two elbows 0.8 and
23 mm); after the elbow the coverage gradient is a genuine derivative (repeat cosine 1.0, stable
across step sizes, SNR 10²–10³) and walking it beat the expert and the best random direction at
all three executable post-elbow states with half the expert's travel (e.g. +24 mm at 16 N against
the expert's +20 mm at 368 N); at the elbow the coverage reward is flat in every cell (Wang's
ratio is zero before the elbow), so a one-step reward gradient cannot be the elbow's signal — the
state sensitivity exists there, the reward's does not, which is SHAC's split (physics `∂x/∂u`,
learned `∂V/∂x`); cell 1's stall is a lock of the environment's no-move collision rule, not the
tether (held-vertex gap 19.6 mm < 60 mm), a non-physical command→executed map no adjoint sees.
The probe now also walks the normalised sum of the coverage and minus-force gradients and a
release-then-advance sequence (not yet run), repeats walks at noisy states, and reports
differences per executed metre; `tests/test_physics_gradient_probe.py` (3) covers that
bookkeeping on a fake environment. [Record](performance/2026-09-13-physics-gradients.md) has the
tables, the Level 2 backend design (solve `Hλ = g` on the converged frame's system, chain six
frames through the inertia term, validate against these differences to cosine ≥ 0.95) and the
Level 3 plan. No learner changed, no training launched.

## 2026-09-13 — Physics gradients: the locked stall re-probed with the combined walks

Owner's go for one more few-minute run: cell 1's stall state only, with the two walks added
after the queue. The anchor's clearance to the arm shell at the stall is 12.6 mm against the
no-move rule's 12 mm, which confirms the lock is that rule (forward walks execute 0.7 mm then
nothing, the expert 0 of 141 mm; the ± probes' forward sides were partly or wholly refused, so
the earlier stall differences were one-sided). Both combinations of the two one-step gradients
escape: their normalised sum +16.5 mm at 42 N, and four decisions down the force gradient then
eight up the coverage gradient +17.5 mm with +0.053 of coverage — more than the earlier fixed
forward-up direction and more than any single gradient, the expert (0 mm executed) or three
per-state random directions. Reading: the jam needs a policy that sequences release and advance,
not a longer sensitivity; a short-horizon physics gradient with h ≥ 4 decisions carries that.
[Record](performance/2026-09-13-physics-gradients.md), ~4 shared-GPU minutes, nothing else launched.

## 2026-09-14 — Physics gradients: rotation at the elbows, the adjoint validated, the export inside the solver

Owner's go for the whole line ("do them all"). (1) The two quiet elbows re-probed with the two
rotation axes the environment executes (0.5°, 1°, 2.5°) and every walk three times: coverage is
flat in all five dimensions at both, decisions scatter 1–1.5 mm but walk outcomes do not; the 5-D
axis-proxy gradient (translation + rotation) walks cell 3's elbow over with +0.077 coverage (3× the
expert at half its travel) and stays below the expert at cell 1 (+0.015 vs +0.027), so the elbow
needs the critic for generality and the 5-D walk is Level 3's baseline. (2) Level 2a, no backend
change: `python -m uipc_manip.physics_gradient_adjoint` reads the engine's `dump_linear_system`
files (or the Level 2b export), chains a decision's six Hessians through the BDF1 inertia term and
the soft position constraint's cross term, and compares the reverse pass with 1–2 mm central
differences of the objective and the tangent pass with differences of every vertex: smooth
objective cosine 0.989–0.995 at four states (both stalls, the jam, a passed state), free-cloth field
within 25 % in magnitude where contact is light, not at the jam (cosines 0.14–0.23 on two axes).
The first pass had halved the cloth's mass — `thickness` is a half-thickness in libuipc
(`compute_vertex_volume`: `h = 2 r`) — and had blamed a doubled constraint stiffness; the script now
uses the backend's `volume` attribute, `uipc_test_diff_sim` measures the constraint's block as
exactly `s·m·I`, and the environment's cloth is twice as heavy as `cloth_thickness` reads. (3)
Level 2b: `LinearSystemAdjointFeature` (`diff_sim/linear_system_adjoint`, core + CUDA + pybind)
exports the frame's assembled `bcoo_A` and `b` after every advance and solves `H x = rhs` by
iterative refinement over the frame's own PCG (the env's `tol_rate` 1e-2 gives residual 3.6 unrefined,
2e-7 refined, 0.15 s); one-process check on the dressing scene: exported and dumped systems are
bit-identical, a solve between decisions leaves the next one within the run-to-run scatter.
`apps/tests/diff_sim` is a new module-loading test target (`backend_cuda` cannot host a `World`:
duplicate kernels). Tests: `test_physics_gradient_probe.py` (4), `test_physics_gradient_adjoint.py`
(3, readers agree, reverse = transpose of tangent), `test_physics_gradient_trajopt.py` (4, the
multi-decision chain's last decision equals the one-decision adjoint, command bookkeeping against
differences of the linearised aim model, refused substeps contribute nothing, coverage gradient on
the hit triangle). `python -m uipc_manip.physics_gradient_trajopt` (Level 3's first experiment:
open-loop trajectory optimisation of 12 decisions at an elbow with the 72-frame chain, rotation
included, executed-command derivatives, checks against differences at chosen lags) ran on both
quiet elbows: coverage after twelve decisions 0.073 → 0.167 ± 0.001 at cell 3 and 0.018 → 0.106 ±
0.002 at cell 1 against the expert's 0.028, force not higher than the expert's. Two findings
bound it: the chain's twelve rows are identical — the BDF1 inertia coupling decays within a frame
or two, so beyond one decision the gradient is the static sensitivity of the end state to a shift
of the gripper path (right in direction at most lags, cosine 0.91–0.99 in translation where the
differences repeat, wrong for the first decision of a fresh approach at cell 3, cosine −0.30); and
the control — the one-step 5-D proxy direction driven at the action box with no gradient —
already gets 0.163 / 0.107 from the same restored state (three rollouts each), so the optimiser's
own contribution is +0.007 coverage and −12 N at cell 3 and nothing at cell 1 (−0.002, +10 N);
the +0.012 a first control from a fresh re-drive had suggested was re-drive spread (0.013). Reading recorded: the useful physics horizon is one decision; the actor that fits
is SVG(1)-like (one-decision solver Jacobian × TD critic), sized as minutes on the elbows first. Build: this tree's CUDA backend and pyuipc built in `build/`
with the dedicated toolchain (memory `libuipc-build-toolchain`); the shared training venv keeps
the 0.0.28 wheel. [Record](performance/2026-09-13-physics-gradients.md); ~3 shared-GPU hours over the
day, nothing else launched, the other sessions' runs untouched.

## 2026-09-14 — Physics gradients: the critic through the adjoint, and the same-state control

Owner's "ok" to the plan. (1) The trajectory optimiser's own contribution, measured from the
same restored state (`--init actions-json --baselines expert scaled_proxy`): +0.007 coverage and
−12 N at cell 3, nothing at cell 1 (−0.002, +10 N); the +0.012 a fresh-re-drive control had
suggested was re-drive spread. (2) `python -m uipc_manip.physics_gradient_actor`: a trained
checkpoint's critic (`abl_dense_s1` at 125k updates, dense critic, the other session's ablation,
read only) as `V(x') = min Q(s', μ(s'))`, differentiated exactly through the observation function
(visibility, 6.25 cm voxel centroids, tool-relative packing rebuilt in torch from the environment's
own discrete choices) and pushed through the one-decision adjoint. Against 2 mm differences of `V`
the analytic direction agrees loosely at hold (cosine 0.6–0.7, magnitude 0.05–0.19 of the measured
change: the critic is not smooth at that scale). Followed greedily for twelve decisions at the
action box, it beats SAC's own `∂Q/∂a`, the checkpoint's policy and the expert at all four states
(cell 3 elbow 0.168 vs 0.114 / 0.076 / 0.046; cell 1 elbow 0.135 vs 0.000 / 0.016 / 0.028; cell 3
stall 0.172, cell 1 passed 0.151), beats the hand proxy at cell 1's elbow (0.076) and matches it at
cell 3's (0.163) at half the force; after the elbow the proxy is better (0.214, 0.220). The
like-for-like reading, `∂Q/∂a` at the hold action (`--walks sac_hold`), retreats at the three
elbow and stall states (0.000, 0.018 → 0.000, 0.000) and wins after the elbow (0.184 vs 0.151).
Differences of `V` with the observation's visibility and voxel membership frozen close a factor
2–7 of the 20–30× gap between `V`'s differences and its derivative; the rest is the network.
The 14k-update latent teacher checkpoint inverts the picture (its own value falls along the
physics walk), a critic-quality contrast. The checkpoint loader accepts pre-dense-critic
checkpoints as latent. A backend
assertion (`simplex_normal_contact.cu:439`, EE/energy size mismatch) aborted the first walk run;
reproduced in one minute (`output/uipc_manip/physics_gradient_actor/dump_after_recover.log`):
restore → decision → dump → restore → decision is fine, restore → dump straight away → restore that
dump → decision aborts. A dump taken without an advance since the recover writes a state the
contact system cannot recover from; the walk now dumps only after a decision. Worth an upstream
issue on `World.dump()`/`recover()`.
Tests: `test_physics_gradient_actor.py` (3). Record, README row and memory updated. ~1.3
shared-GPU hours; the other sessions' processes untouched, their checkpoint read only.

## 2026-09-14 — Physics gradients: the direction as a training signal, critic frozen

Owner's "ok" for the step past the probes, kept to ~25 shared-GPU minutes.
`python -m uipc_manip.physics_gradient_finetune` collects 288 transitions from one elbow state
(stochastic policy and noisy proxy episodes) with the physics direction `∂V(x')/∂u` stored per
transition (dense critic through the observation, last-frame device solve, 0.13–0.21 s each),
then fine-tunes three actor copies on the same batches with the critic frozen — SAC's own actor
loss, the physics direction on the mean action (`β` matched to the SAC gradient's norm once), and
their sum — and evaluates each deterministically for twelve decisions at the tuned elbow and at
a state of the cell that was not a tuning start state. The tuned actors sit on the action box
(the loss is linear in μ; 174–179 mm and 85° executed everywhere), so what was learned is a sign
pattern. Physics: passes both elbows (0.158, 0.128) and is the gentler and better one at both
non-start states (0.222 at 18 N vs 0.168 at 318 N; 0.189 vs 0.116). SAC's loss: stronger at cell
3's elbow (0.196 at 53 N), a retreat at cell 1's (0.000), a jam at cell 3's stall. The sum, at the
β used, never best. `tests/test_physics_gradient_finetune.py` (2). Record section, README
row, memory updated. Not a training run: critic frozen, one cell at a time, two draws per
evaluation. The other sessions' processes untouched.

## 2026-09-14 — Physics gradients: the direction inside the SAC update, seed-matched pair launched

Owner's go for the training run. The physics direction is now an opt-in term of the actor
update, default off: `--physics-actor-weight w` (relative to the SAC term's first gradient
norm, matched once and kept, saved in the checkpoint as `physics_beta`) and
`--physics-actor-gate` (least executed fraction). `physics_actor_signal.PhysicsActorSignal`
computes, after every vector step, every slot's `∂V(x')/∂u` at once: the critic's value of the
returned observation differentiated through the observation function (visibility and voxel
membership captured from the environment's own call inside `env.step`), one refined device
solve against the world's global system (block-diagonal across slots), and each slot's held
vertices; `FlatReplayBuffer`/`ReplaySet` store `(physics, physics_valid)` per row (returned last
by `sample`, kept in snapshots, older snapshots load as rows without a direction);
`SACAgent._update_actor_and_alpha` adds `−β · unit(g) · μ` on the rows that count.
`--init-from` starts a fresh run from a checkpoint's weights and temperature. `pretrain_wang`
binds the signal to every rotated world and aborts it on a simulator error. Smoke on 8 envs ×
240 transitions from `abl_dense_s1` at 125k: 0.28 s of physics per vector step (solve 0.18 s,
residual 4e-5, 87.5 % of rows counted, β 0.14), checkpoint saved and loads. Tests:
`test_physics_actor_signal.py` (4); the agent, replay, trainer and pretrain suites still pass
(124). Launched 14:55 (`scratchpad/run_pg_pair.sh`, logs `output/uipc_manip/logs/pg_{phys,ctrl}_s1.log`):
`pg_phys_s1` (weight 0.5) then `pg_ctrl_s1` (weight 0), each `teacher --region 13 --num-envs 24
--transitions 24000 --eval-every 0 --init-from abl_dense_s1/checkpoint_00125016.pt`, held-out
evaluation before and after, then the elbow yardstick (`physics_gradient_actor --walks policy`
at both cells) on each final checkpoint. Expected ~2.5 h per arm. Not yet evaluated.

## 2026-09-14 — IPC Adjoint Q-Learning design

On `research/ipc-adjoint-q-learning`, the owner requested an algorithm design
with simpler benchmarks before dressing. [ADR 0008](adr/0008-ipc-adjoint-q-learning.md)
records the proposed target/loss contract, complete-substep adjoint, replay
freshness, contact-conditioned directional trust, observation limitations, and
`cloth_drag`-first experiment gates. Literature verification identified MAGE
(2020) and First-order Sobolev RL (2025) as direct precedents, beyond JAVE.
The existing physics actor signal remains a distinct experiment. A standalone
NumPy check (`python3 scripts/verify_iaql_design.py`) passes 24 finite-difference
cases and adjoint/tangent/error-bound identities; it exercises synthetic
equations, not the production backend or learner. No IAQL training launched.

## 2026-09-14 — IAQL benchmark prototype: gate, fixed-teacher fit, refit control, tolerance and friction probes; pair launched

Takeover of the Codex session's uncommitted implementation of ADR 0008. Its
work: `iaql_env` (a direct-picker `cloth_drag` variant with x/v state read
through `FiniteElementStateAccessorFeature`), `iaql.soft_targets` /
`derivative_loss`, `SACAgent.update_state_batch` with the `state` actor and
matched SiLU networks, `tangent_pass(return_frames=True)`, `iaql_benchmark`
and its tests; it had diagnosed a stale-velocity adapter and a ReLU kink in
the gate, switched both arms to SiLU and launched `silu_friction0_s0`, which
finished at 16:05: gate 4/4 at friction 0 (Bellman cosine ≥ 0.9987, relative
error 0.011–0.19 rising with the number of guided drag steps), fixed-teacher
fit held-out slope cosine 0.38 → 0.84 at β = 0.1 with unchanged value error,
256-step online smoke inconclusive (no success in either arm). Added here:
`--phase refit` (CPU, on the saved dataset) — shuffled labels never help and
at β = 1 destroy slopes and values, the train-mean slope has zero cosine,
paired labels reach 0.84 at β ≥ 0.1, so the gain is the physics; strict Newton
tolerance (1e-5) changes nothing at the three dragged states (errors identical
to three digits, forward 1.6–2.2× slower), so the residual 5–21 % tangent
error is the SPD-projected retained matrix, not convergence; friction 0.6
fails the gate (1/4: Bellman error 0.11 / 0.37 / 0.77 / 0.26, reward-gradient
error up to 0.68, hence the mechanics — the inertia-only chain lacks the
lagged friction terms), so no frictional learning run without those terms or
a trust weight. The online loop now keeps every TD transition with a bounded
mechanics sidecar (`--tangent-rows`), evaluates periodically from a snapshot
and runs one arm per process. Full CPU suite 376 passed, 14 CUDA-marked
deselected. Launched 16:24 (`scratchpad/run_iaql_pair_now.sh`, logs
`output/iaql/online5k_s0_{sac,iaql}.log`): the matched pair at friction 0,
5,000 steps per arm in parallel, two updates per step, β 0.1, evaluation of
3 × 50 steps every 500; about 1.7 s per step on the shared GPU, so 2.5–3 h per
arm. Not yet evaluated. The other sessions' jobs untouched.
[Record](performance/2026-09-14-iaql-benchmark.md).

## 2026-09-14 — Adjoint export modes: the raw Hessian at the accepted state

Owner's reading of the tolerance probe: the SPD projection is the likely
cause of the tangent error but not proven until the raw Hessian is compared;
the experiment to run is `D_PSD` vs `D_raw` vs `D_FD`. Implemented in the CUDA
backend without touching the forward solve: a device-side projection switch
read by every `make_spd` site and the friction helper's 2x2 projection
(`utils/make_spd.h`, defined in `linear_system_adjoint.cu`; StableNeoHookean-3D's
analytic projection is not covered), `GlobalLinearSystem::Impl::build_linear_system(bool
with_preconditioner)`, and after the Newton loop of the IPC pipeline a
re-detection and re-assembly at the accepted state when
`LinearSystemAdjointFeature.set_export_mode` is `converged` or `converged_raw`
(RAII guard restores the switch; `solve()` refuses the raw mode; AL pipeline
untouched). `iaql_env` takes `export_mode`; the probe captures each decision
once per `--export-modes` entry against one set of differences and
`--guided-steps-min 1` keeps every snapshot one advance past its recover.
Built in `build_raw/` (the in-place post-build copy would have replaced the
`.so` under the running 5k pair). `uipc_test_diff_sim` gained the
export-modes case (3 cases, 563 assertions pass). Smoke on two states: the
re-assembly at the accepted state changes nothing to three digits and the raw
matrix brings the position tangent error from 2–8 % to 0.03–0.2 % and the
Bellman-gradient error from 0.04 to 0.0003–0.006, so the projection was the
whole frictionless error. Also added: the ADR's total-derivative statement for
`dY/du`, and the online shuffled-label control `--shuffle-labels` (batch-wise
permutation among valid rows, everything else matched; test). The 100-state
three-mode probe is running (about 1 min per state with the pair sharing the
GPU); the 5k pair is at 1,500 steps, both arms still at zero success.
[Record](performance/2026-09-14-iaql-benchmark.md).

## 2026-09-14 — Friction's lagged coupling exported for the adjoint chain

The owner's P2 diagnosis (the inertia-only chain lacks friction's dependence on
the previous substep) implemented as an export, not a solve change: the
re-assembly at the accepted state also computes `dG_f/dx_prev` for every
half-plane friction pair (`IPCVertexHalfPlaneFrictionalContact::do_compute_prev_coupling`,
central differences of the friction gradient in the previous position with a
step of 1e-4 of `eps_v*dt`; the lagged normal force and the relative
displacement both enter through it, the tangent basis of a fixed plane is
constant, one lag per frame), reachable as
`LinearSystemAdjointFeature.export_prev_coupling()` with global vertex ids.
`tangent_pass(prev_coupling=...)` adds `-B_fric X_{k-1}` beside the inertia
term (unit test against the explicit recurrence); `iaql_env(friction_chain=True)`
and the probe's `--friction-chain` use it in the converged export modes.
Simplex (cloth-body) friction coupling is not exported yet. Backend and test
targets rebuilt in `build_raw/` without the Python copy step (the running
100-state probe maps that copy); `uipc_test_diff_sim` passes. Queued
(`scratchpad/run_friction_probes.sh`): after the 100-state probe exits, the
Python copy is refreshed and the friction-0.6 gate runs on four states in
all three export modes, without and with the chain
(`output/iaql/fric06_modes{,_chain}`). Not yet measured.
[Record](performance/2026-09-14-iaql-benchmark.md).

## 2026-09-14 — Export modes on 100 states: the projection was the whole frictionless error

`modes100_s0` (101 shared-GPU minutes): 100 snapshots with 1/5/9/13 guided
drag steps, every centre decision captured in all three export modes against
one set of differences. Re-assembling at the accepted state with projection
reproduces the last-iterate numbers to three digits; the raw re-assembly
divides the median position tangent error by 22 (4.3 % → 0.20 %), the p90 by
17 (10.9 % → 0.65 %) and the Bellman-gradient error by 19 (3.3 % → 0.17 %),
and the growth with drag (1.6 → 8.1 %) disappears (0.16–0.25 % at every
depth). 99/100 pass the gate in every mode; the one rejected state is the
critic's (raw reward-gradient error 0.7 %, Bellman 14.5 %). Cost 0.2 s per
decision. Record, performance index and roadmap updated. The 5k pair is at
3,000 steps with both arms at zero success; the friction gate with and
without the lagged coupling is running.

## 2026-09-14 — Friction 0.6 passes the gate with the raw Hessian and the lagged coupling

`fric06_modes` and `fric06_modes_chain` (four states, ~4 shared-GPU minutes
each): without the lagged block every export mode fails (0/4, position
tangent error 24–26 % median, Bellman error 0.25–0.74); the chain with the
projected Hessian gets 8.5 % and 3/4; the chain with the raw Hessian gets
0.14 % median position error, reward-gradient error ≤ 0.18 % on all four
states and 3/4, the rejected state being the critic's (Bellman 19 % against
0.1 % mechanics). The state that had looked like a stick-slip switch was the
missing lagged block. So P1 and P2 are closed for the half-plane contact:
`converged_raw` plus `export_prev_coupling` is the complete frictional
tangent of this decision map. Open: simplex friction coupling, a
stick-slip-switching state, and the learning benchmark itself (the 5k pair
is past 3,000 steps with both arms at zero success). Record, index and
roadmap updated.

## 2026-09-14 — The 5,000-step SAC/IAQL pair: inconclusive

Both arms finished (13,135 and 13,288 s of training, 2.6 s per step on the
shared GPU). Neither learns the task in 5,000 steps: returns within ±5 per
episode until 3,500 steps, large actions from 4,000, one SAC success at
4,000, final mean distance 0.078 (SAC) and 0.077 (IAQL) with no success;
the arms are indistinguishable against a three-episode evaluation. Not
evidence either way. The next learning experiment needs a learnable benchmark
first: a vectorised world (several cloths per solve) or a smaller cloth to
bring the decision far below 2.6 s, and a budget at which vanilla SAC shows a
curve; then paired seeds and the shuffled-label online control. Record,
index and roadmap updated. All of today's runs are done; nothing is running
from this session.

## 2026-09-15 — One branch for both gradient consumers

`research/physics-gradients` (the actor-side line: greedy walks, frozen-critic
fine-tune, the SAC-loop `--physics-actor-weight` term) had no commit that
`research/ipc-adjoint-q-learning` lacked, so it was fast-forwarded to this
branch's tip; the two lines consume the same one-decision label
`g = R_u + γ m Dᵀ∇V̄(s')`, the actor directly at the replay action and the
critic through the Sobolev slope loss, and will be developed as one learner
here. The actor-side arm `pg_phys_s1` finished at 21:35 on 2026-09-14:
held-out success 0.000 → 0.240 and upper-arm ratio 0.146 → 0.309 after 24k
transitions from the `abl_dense_s1` 125k checkpoint, whose own 14 held-out
evaluations over 125k transitions never left 0.000 (one 0.040); its control
arm `pg_ctrl_s1` (weight 0, same seed and start) is running and decides the
attribution. Its Jacobian is still the last frame's projected last-iterate
matrix without the friction coupling, i.e. the three errors this branch
removed today; unifying it with `converged_raw` + `export_prev_coupling` +
the full decision chain is the first step of the combined learner.

## 2026-09-15 — Physics gradients: the seed-matched pair reports

`pg_phys_s1` (physics term, weight 0.5, β 0.155) and `pg_ctrl_s1` (plain SAC), both from
`abl_dense_s1` at 125k with seed 1 and 24,000 new transitions: held-out 0.146 → **0.309 with 6/25
successes** against 0.160 → **0.043 with 0/25**; the control's evaluation took 3.5 h because the
solver stalls under its policy. Both arms tripped the watchdog 4 times and finished one complete
episode; 85 % of the physics arm's transitions carried a direction. Yardstick (12 deterministic
decisions from the probe's states): physics 0.036 / 0.135 / 0.000 / 0.159, control 0 / 0 / 0 /
0.046 (retreating). Diagnostics: both drifted far from the start (cosine 0.30–0.47); the physics
arm sits closer to the action box (x saturated 39–49 %) and mildly aligned with its direction
(+0.06 vs −0.11). Reading recorded: a usable training signal inside an off-policy update, the best
held-out policy so far, but one seed and a collapsed plain arm, so not yet a beaten healthy
baseline. The control's 6 h limit cut it 72 transitions short; `pretrain_wang resume <run>
--transitions N` finished it (and is the way to add a final evaluation to any run). ~13
shared-GPU hours. Next: two more seeds, a control that keeps the optimizer state or halves the
actor learning rate, weights 0.25 and 1.0. Record section "fourth experiment"; README row updated.

## 2026-09-15 — The combined learner and the actor-gradient fidelity probe

In the state benchmark's `update_state_batch` the refreshed label
`g = dR/du + γ m Dᵀ∇V̄(s')` now feeds two consumers: the critic's slope loss
(`adjoint_weight`) and the actor's direction term (`physics_actor_weight`,
the physics-gradient line's `−β·unit(g)·μ(s)` through the existing
`_update_actor_and_alpha` path, counted only on sidecar rows whose replayed
action is within `physics_actor_action_distance` of the policy's mean;
`physics_actor_fraction` reports the share). `iaql_benchmark --actor-weight`
sets it, so one process runs any of the five arms (SAC, critic, actor, both,
shuffled). `--phase fidelity` is the ADR's counterfactual-action check: at
states of a frozen policy it compares each loaded critic's `dQ/da`, the exact
one-decision label with that critic's continuation, and the reward gradient
against the finite-difference gradient of the frozen policy's `H`-decision
return, and rolls every candidate out along `±η·unit(g)` for its own
improvement. Launched 00:10 on the two 5k checkpoints (10 states, H = 8,
η = 0.1, `output/iaql/fidelity_s0`), about an hour on the shared GPU. Tests:
`test_iaql` (9), agent and physics-signal suites pass (51).

## 2026-09-15 — Fidelity probe: direct label > Sobolev critic > SAC critic; the GPU was the bottleneck

With the other sessions' jobs finished the direct-picker decision costs
0.12 s, not the 2.6 s of yesterday's pair, so the 5k-step budget verdict was
contention, not the benchmark. The fidelity probe (ten states, 8-decision
return, 2 GPU minutes): the exact one-decision label improves the frozen
policy's return at every state (mean cosine 0.84–0.86 with the return
gradient, +0.149) against the SAC critic's own action gradient (0.33, +0.038,
wrong sign twice) and the Sobolev critic's (0.60, +0.087, wrong sign once);
the reward gradient alone is as good as the label, so the continuation term
adds nothing with 5k-step critics. Launched 12:10, five matched 30k-step
arms in parallel with the raw-Hessian label (`scratchpad/run_five_arms.sh`,
`output/iaql/arms30k_s0_{sac,critic,actor,both,shuffled}`; critic slope 0.1,
actor direction 0.5, both, both with shuffled labels; evaluation every
2,000 steps). Not yet read. Record updated.

## 2026-09-15 — Estimator replacement, continuation trust, terminal reward; the arms rerun one at a time

Owner's reading of the fidelity probe: the direct label is the actor's best
gradient and the continuation term adds nothing yet, so (1) the actor term
should replace a fraction of the critic's gradient rather than add a
fixed-norm direction: `physics_actor_mode="mix"` applies
`−ρ·sg[c·(g − ∇_a Q(s, a_θ))]·a_θ` at the policy's sampled action, whose action
gradient is `(1−ρc)∇_a Q + ρc·g` (ρ = 0 is SAC, ρ = 1 the label alone); the
locality `c` is the hard action-distance gate or, with `physics_actor_sigma`,
a Gaussian of it; (2) the label splits into reward and continuation parts and
`continuation_trust_kappa` discounts the continuation by `exp(−κ d²)` with `d`
the twin critics' disagreement (`iaql.soft_targets_detailed`, one reverse
pass per head); (3) `iaql_env(reward_mode="terminal")` pays the distance only
at the last decision, the benchmark on which the reward gradient alone
cannot work; (4) every update logs the continuation-to-reward ratio, the
trust, the disagreement and the critic's cosine with the label at the
replayed action, and the fidelity probe reports each continuation's cosine
with the return gradient's residual beyond the reward part. Tests: 49 pass.
Five 30k arms launched in parallel at 12:08 contended to 1.6 s per step (five
processes on one GPU through MPS) and were stopped at 600 steps; six 20k
arms now run one after another (`scratchpad/run_arms_seq.sh`,
`output/iaql/arms20k_s0_{sac,actor_mix,actor_dir,critic,both,shuffled}`,
about an hour each): SAC; mix ρ 0.5; the direction term 0.5; critic slope
0.1; mix + critic; mix + critic with shuffled labels. Not yet read.

## 2026-09-15 — Lockstep slots: several cloths per World

Another session's `pretrain_wang resume` job took the GPU back at 12:10, the
single-slot SAC arm fell to 1 s per step, and six sequential 20k arms would
have taken 30 hours. `IAQLEnvConfig.num_slots` now builds N identical cloths
one metre apart in one World (never touching), stepping in lockstep with
per-slot anchors, goals, rewards and tangents; one exported system and one
factorisation per substep serve every slot, each slot's tangent being its own
block of the solution (`tangent_pass` with the slot's dof offset), the friction
coupling rows filtered per slot, velocities sliced by `backend_fem_vertex_offset`.
The single-slot interface is unchanged (scalars), the online loop counts
transitions per slot and makes `updates_per_step` updates per transition,
evaluation runs N episodes per seed. Four-slot smoke with the mix actor term:
0.44 s per transition against about 1 s single-slot under the same
contention. The single-slot sequential arms were stopped at 2,000 steps and
relaunched with eight slots (`scratchpad/run_arms_vec.sh`,
`output/iaql/vec20k_s0_{sac,actor_mix,critic,both,shuffled,actor_dir}`, 20k
transitions each, evaluation of 8 episodes every 2,000). Not yet read.

## 2026-09-15 — Where the time goes, and per-slot factorisations

Owner asked why the GPU looks cold. Profile of one eight-slot online step
under the other session's job (GPU shared, 30 GB): 2.52 s, of which the
backend's five substeps 2.1 s (0.42 s per substep for 3,200 vertices,
against 24 ms per substep for 400 vertices on the idle GPU yesterday: the
contention, plus hundreds of small launches per substep that cannot fill a
Blackwell), export and factorisation 0.16 s, tangent passes 0.25 s, sixteen
learner updates 0.17 s on the CPU. The process is one Python thread blocked
in synchronous backend calls; it cannot use more of the GPU on its own.
Scaling under the same contention: 1 slot 1.37 s per step, 8 slots 2.40 s
(0.30 s per transition), 32 slots 4.41 s forward (0.14 s per transition) but
a 9.9 s captured step because the tangent solved one 38k-dof factorisation
480 times. Slots never touch, so `slot_factorizations` now factorises each
slot's own 1,200-dof block (test against the global solve); the per-slot LU
and tangent are then linear in the slot count. Parallel arms are the other
lever: five single-slot arms in parallel gave 3.1 transitions per second
aggregate against 1 for one arm under the same contention.

## 2026-09-15 — Throughput: 64 slots, per-slot factorisations from triplets, learner on the GPU

The owner stopped the other job and asked for the training pipeline to use
the GPU and CPU properly now. Measured on the idle GPU: the forward step
costs 0.43 s for 8 slots, 1.0 s for 32, 1.46 s for 64 (54 → 31 → 23 ms per
transition; the backend's launches amortise), while the capture path grew to
1.3 s per step at 64 slots: assembling the 77k-dof global matrix (48 ms) and
64 block factorisations (4.4 ms each) per substep. `slot_factorizations_from_triplets`
now assigns every exported block triplet to its slot and factorises the slot
blocks on a thread pool (SuperLU releases the GIL: 8 threads give 4×), never
assembling the whole system; the learner moves to CUDA with `--device cuda`
(inputs and sidecar tensors follow). A 64-slot online step with the mix
actor term, the continuation trust and 128 GPU updates: 3.0 s, i.e. 47 ms
per transition against 221 ms for yesterday's 8-slot SAC arm. Launched at
15:40, five 64-slot arms in parallel (`scratchpad/run_arms_vec4.sh`,
`output/iaql/vec20k_s0_{sac64,actor_trust,critic_trust,both_trust,shuffled_trust}`,
20k transitions each, κ = 2, σ = 0.5, evaluation of 64 episodes every 2,000);
the 8-slot κ = 0 hard-gate mix arm finishes alongside as an ablation. Tests:
per-slot factorisations against the global solve, the cross-slot refusal.

## 2026-09-15 — The decision tangent on the GPU as batched dense block algebra

Under the five parallel 64-slot arms the GPU reads 99 % busy but 240–280 W
of 600 and 12–20 % memory bus: kernels on 25k-vertex scenes fill a fraction
of the SMs and every arm process sits at 100 % CPU between them. Aggregate
throughput of the five arms: 27.6 ms per transition (87 ms for the SAC arm,
162 ms for each mechanics arm), eight times one 8-slot arm.
`iaql_tangent.BatchedTangent` takes the whole capture path off the host: the
exported triplets are scattered into dense `(N, 3n, 3n)` slot blocks, one
batched `lu_factor` per substep (single precision, two rounds of
double-precision residual refinement), the inertia chain, the aim term and
friction's lagged blocks as batched tensor ops, three axes at once; the host
keeps no factorisation. Reproduces `tangent_pass` with the coupling to 1e-10
(double) and 1e-6 (refined single) in the test. `IAQLEnvConfig.tangent_device`
/ `--tangent-device cuda` select it; `--batch-size` lets the learner make
fewer, larger updates at a matched sample rate. Measured batched LU of 256
dense 1200-dof blocks in double: 0.41 s; timings of the whole step follow.

## 2026-09-15 — Host memory: the thread-pooled factorisations grew to 40 GB per arm

At 15:36 the kernel OOM-killed one of the five parallel 64-slot arms (host
RAM 99 of 123 GB): the two arms with host-side per-slot factorisations had
reached 40 GB RSS each while the SAC arm sat at 1.3 GB. Per step they
allocate 320 SuperLU factorisations on eight threads and glibc's arenas do
not return that memory, so RSS grows without bound; the desktop may have
stalled at that moment. The two survivors were stopped, the smoke that timed
the 64-slot step gave 134 ms per transition on the host path against 78 ms
on the GPU tangent under the same five-way contention, and a 256-slot world
did not finish its settle and first steps within four minutes, so the four
mechanics arms were relaunched at 16:00 with 64 slots on the GPU tangent
(`scratchpad/run_arms_vec5.sh`, same names), no host factorisation at all.
`--updates-per-step` is a float now, so a larger batch can keep the sample
rate. The 8-slot κ = 0 hard-gate mix arm finished: see the record.

## 2026-09-15 — First round on the throughput rebuild: the actor label doubles SAC's return

`vec20k_s0_*`, 64 slots, 20k transitions, κ = 2, σ = 0.5, four mechanics
arms in parallel on the GPU tangent (109–116 ms per transition) beside the
SAC arm (68 ms): SAC 6.78 return / 0.062 m / 9 of 64 successes; critic slope
alone 6.58 / 0.063 / 5; actor direction (mix ρ 0.5) 13.31 / 0.022 / 29;
actor + critic 12.33 / 0.028 / 18; both with permuted labels 9.05 / 0.048 /
9. Direct label > Sobolev critic ≈ SAC; the permuted control keeps a third
of the gain. Record section written. Launched 16:40 the second round, six
64-slot arms in parallel (`scratchpad/run_arms_vec6.sh`,
`output/iaql/vec20k_s{1,2}_{sac64,actor_trust}`, `vec20k_s0_actor_shuffled`,
`vec20k_s0_actor_rho1`): two more seeds of SAC and the actor arm, an
actor-only permuted-label control, and ρ = 1. The 15:48 relaunch is the one
the previous entry calls 16:00.


## 2026-09-15 — Fresh IPC actor and dense/residual defaults

Implemented same-noise, same-action fresh actor updates independent of TD
replay, post-tanh locality, semantic controls, terminal horizon correctness,
and checkpoint-aware dense/residual dressing defaults. See
[protocol and evidence](performance/2026-09-15-fresh-ipc-actor.md).
Native CUDA smoke passes at rho=.5 and rho=1; no new learning claim yet.
Next: solver-derived counterfactual response experiments requested by owner.

## 2026-09-15 — Counterfactual response pilots

Implemented `counterfactual.response_objective` and `response_experiment` native
collection/offline comparisons. See [results and limitations](performance/2026-09-15-counterfactual-response.md).
Collected 640 transitions each at friction 0 and .6, checked finite-radius true
rollouts and nominal repeatability, and ran three-seed objective/history/Q
probes and cross-friction evaluation. Literal CF barely differs from response
training; radius-normalized differences improve decoder derivatives. Frozen
representation/value advantages remain unproven. A constant tangent is a strong
baseline (.988 no-friction cosine); do not read high cosine as encoder success.
Diagnostic checkpoints are fixed-mesh models, not production dressing weights.
Also found old replay actor seed 2 loses to SAC: seed-level instability remains.

## 2026-09-16 — Single-policy teacher-free entry and live-run audit

Added `pretrain_wang joint --regions ...`: one dense/residual policy, no teacher
loading/distillation, existing rotation/replay/evaluation/resume. All 17 launcher
tests pass including joint resume. The owner stopped the old-worktree regional
training at 134,832 logged transitions; its last evaluation had degraded from the
best CSV checkpoint. Artifacts are preserved. No long fresh-hybrid result yet. See
[status and algorithm corrections](performance/2026-09-16-single-policy-plan.md).

## 2026-09-16 — Native actor-interface and optimizer-history probe

Owner resumed bounded experiments after stopping the old dressing run. Added
`actor_interface_probe.py`, reusing SAC and the native full-state IPC diagnostic.
Three saved actor checkpoints, two eight-slot anchor batches each, paired eight-
step reward rollouts; no terminal critic in evaluation. Target fitting and its
normalized linear surrogate produce identical actual anchor actions. Target
radius .03 nevertheless permits .16142 actual movement with inherited Adam.
Even a zero-gradient target moves .14292 because of stored Adam moments.
Clearing moments plus an actual-movement guard changes mean reward improvement
from -.03109 to +.01366; bounded random gives -.00763. The bounded IPC arm improves
all six batch means, but this is not a learning result or a diagnosis of prior
seed collapse. Do not reset shared training Adam on the strength of this probe.
Full [results and scope](performance/2026-09-16-actor-interface-results.md),
including raw artifact paths and tracked per-batch numbers. All 17 fresh/probe
tests pass, plus 17 launcher tests. No long training restarted; visual dressing
IPC integration and optimizer-history-safe matched learning remain open.

## 2026-09-16 — Transactional fresh IPC action trust radius

Commit `53c89d77` adds `physics_actor_max_action_step` and retry count to the
fresh actor path. An over-radius optimizer step restores actor and Adam state,
halves the actor learning rate, and retries; after exhaustion it restores the
original state. `iaql_benchmark` exposes this as `--actor-step-radius` and
`--actor-step-retries`. Native eight-slot/256-transition smoke completed with
24 fresh actor and 200 critic updates; max measured action step was .00212 under
the .03 radius. The long matched SAC/IPC runs were then launched from this
implementation; their outputs are separate from the smoke artifact.

## 2026-09-16 — Seed audit, two harness fixes, five-seed 40k study launched

Owner asked whether seed 2's loss came from incomplete training and whether
the IPC actor idea should be abandoned. Audit of `output/iaql/vec20k_s{0,1,2}_*`:
every run trained 20,032 transitions with nine evaluations, the same flags and
no traceback, so seed 2 was not cut short. Every run did see only two
150-decision episode rounds (about 128 goals per run). The replay-batch actor
term's coverage collapsed after about 5k transitions in all three seeds
(1–9 of 32 rows, mean weight .01–.05), and its label drew `torch.randn_like`
from the global RNG, so labelled and unlabelled arms did not share policy
sampling noise. Result on record: two wins, one loss, not significant, and the
"doubles SAC" reading in the performance index and roadmap overstates it.

Fixes: `update_state_batch(label_noise=...)` lets the replay label take its
next-action noise from the driver's `label_rng` (test: a labelled update leaves
the global RNG exactly where an unlabelled one does); `replay_physics=False`
keeps fresh-protocol replay actor steps plain SAC, which removes the
"IAQL batch needs mechanics and validity" crash of the 14:47 matched fresh_ipc
run (test). The fresh actor's step statistics use `.get`, fixing three
pre-existing `test_fresh_iaql` failures under a mocked actor step. 82 related
CPU tests pass; native 16-slot smokes of both protocols run.

Launched 15:55, all 25 runs in parallel (`scripts/launch_seed_study.sh`,
`output/iaql/study40k/`): group A replicates the 20k round's replay-batch
configuration against SAC, seeds 0–4, 40k transitions, horizon 150; group B is
the fresh-batch protocol (fresh SAC, fresh IPC ρ 1, norm-matched random), seeds
0–4, 40k transitions, horizon 50. Decision rule agreed in advance: the IPC arm
counts as better only if it beats its matched SAC arm in at least four of five
seeds and on the mean of the final evaluation, and the random control does not.
The SAC continuation arms' final-evaluation collapse (periodic ≈ 8, final −16 /
−11) is not yet explained.
