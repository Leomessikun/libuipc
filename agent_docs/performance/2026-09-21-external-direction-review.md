# Review of an external three-direction research proposal — 2026-09-21

Status: assessment of a proposal the owner brought in from outside this project,
against the measurements already on record here. No new physics run, no training,
no algorithm claim. Two read-only checks were performed: the completed
`state_ub_clean_s1` evaluation log, and an arXiv API lookup of the three dressing
citations the proposal's argument depends on.

The proposal ranks three candidate research lines:

1. **Event-probability gradients** — estimate `grad_theta Pr(success)` by locating,
   with IPC queries, the action-space boundaries at which a task event flips
   (threaded / grasp kept / stuck), and adding the boundary term
   `p(b) [G(b+) - G(b-)]` to the interior pathwise term.
2. **Decision-relevant active contact identification** — under occlusion, choose
   actions that make hidden contact states distinguishable, and separate the
   informational from the physical effect of a probe.
3. **Decision-relevant simulator calibration** — calibrate the ranking of action
   consequences rather than state prediction error.

## Where the proposal and this project's independent reads agree

- Its scalar counterexample (`R(a)=1[a>b]`, zero pathwise derivative, positive
  `dJ/dmu`) is the same one already recorded in
  [the algorithm assessment](2026-09-20-rl-algorithm-research-assessment.md) §1.
- Its prior-art screens match reads made here independently: AHAC/AGPO and
  gradient-variance work for "when to trust which gradient", multi-fidelity
  control variates for "few expensive, many cheap" samples, value-aware model
  learning for "train the model on the decision loss", DiffCloth and Deep Haptic
  MPC as dressing precedents. The conclusion it draws — none of these is a
  contribution by itself — is the conclusion reached in
  [the clean-slate pass](2026-09-20-clean-slate-rl-research.md).
- Its refusal to accept "diffusion/RL/teacher-student recombination" as novelty
  matches the boundary this project already set.

Citations checked against the arXiv API (`id_list` query, 2026-09-21):
`2607.10999` *Wearing A Coat* (2026-07-13), `2509.12741` *Force-Modulated Visual
Policy* (2025-09-16, 12 participants, 264 trials), `2609.04759` *Dressing in
Motion: A Human Motion-Aware Diffusion Policy* (2026-09-04, 9 participants, 3
garments, 6 arm-motion patterns). All three exist as described. The proposal's
reading of the dressing field is therefore sound; the disagreement below is about
which direction this project's own data supports, not about the literature.

## The measurement that decides the ranking

The relevant record is [the privileged upper bound and RAL
calibration](2026-09-20-upper-bound-and-ral-calibration.md), whose run
`output/uipc_manip/state_ub_clean_s1` I re-read directly from `eval_log.csv` and
the run log: 270,000 transitions, 10,800 vector steps, 16,469.5 s (4.58 h), one
seed, actor reading the 35-float privileged vector, privileged critic, all 25
cells both trained and evaluated.

| step | success | max upper-arm ratio | final forearm ratio | max threaded rate | grasp valid |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0/25 | .0000 | .0000 | .00 | 1.00 |
| 4,500 | 0/25 | .0000 | .0479 | .72 | 1.00 |
| 7,000 | 0/25 | .0000 | .1110 | 1.00 | 1.00 |
| 8,000 | 0/25 | .0000 | .2953 | 1.00 | 1.00 |
| 10,800 | 0/25 | .0000 | .3048 | .80 | 1.00 |

Upper-arm ratio is exactly .0000 in all 22 evaluations, including each episode's
maximum, while the return rises from -272.6 to +67.8, threading becomes reliable,
and the sleeve advances to 42 % of the forearm at peak. The objective has **two
flat junctions**, fingertip and elbow; 175,000 transitions bought the first and
the remaining 95,000 did not reach the second. Per the
[evidence review](2026-09-20-training-issue-review.md) this run is **not** a
matched upper bound — its control period, episode length, angular speed limit and
discount time differ from the SAC runs, and the junction continuity is not proof
of zero slope. It is one seed with a 35-float summary rather than full state.
What it does establish is where the wall sits: the elbow, the same place the
scripted expert's ten real failures stall (arc .61-.79).

The same document's calibration adds the resolution facts that any query-based
estimator must live with: at these states a full-magnitude displacement of **one**
command separates at a median 1.45 pooled standard deviations (nothing resolves at
single-decision granularity), a four-decision burst reaches 5.29 and clears three
standard deviations at 75 % of states, and threading probability is **degenerate**
- zero within-action variance - wherever the milestone is already decided.

## Direction 1: event-probability gradients

This is **not** the closed IAQL line. IAQL back-propagated IPC pathwise
derivatives into the actor; this proposes a sampled boundary term with IPC
tangents used only to propose query directions. The objections are different.

1. **The events it targets split into one already crossed and one never
   reached.** A boundary estimator adds `p(b) [G(b+) - G(b-)]` only for events the
   sampled behavior actually straddles. The proposal scopes itself to early events
   — threading and grasp retention — and those *are* straddled: in the privileged
   run the maximum threaded rate moves .00 -> .52 -> .72 -> 1.00 over the first
   175,000 transitions and grasp validity moves .64/.76 -> 1.00. But ordinary SAC
   crossed that plateau on its own, without any boundary machinery, so the best
   available claim for direction 1 there is an acceleration of something already
   achieved in 175,000 transitions (4.58 h of workstation time buys the whole run).
   The plateau that actually stops the task, the elbow, has never been reached by
   any learner here: the upper-arm event has `p(b)` measured at zero in all 22
   evaluations, so its boundary term is unmeasured rather than refuted, and the
   clean-slate pass already recorded that branching methods "cannot create success
   when no sampled behavior reaches it". Where the calibration did measure a
   threading probability — at 24 states already on the upper arm — its
   within-action variance was zero, i.e. the milestone was decided; that says
   nothing about threading at approach states.
2. **The measured obstacle is a plateau, not a jump** — which is the
   proposal's own exit criterion ("contact-set changes need not make the return
   discontinuous; if the objective is smooth, do not add a jump term"), so this is
   agreement with its exit rule rather than an outside objection. `wang_progress` joins are
   continuous in value (the [inventory review](2026-09-20-training-issue-review.md)
   corrected the earlier zero-slope argument); what reproduces is a region with no
   usable slope plus a success rule that reads 1.00 -> 0.00 once the opening passes
   the shoulder. Boundary-gradient machinery is the right tool for a delta at a
   discontinuity, not for a flat region with zero event probability.
3. **Boundary localization costs more than the decision is worth here.**
   Identical commands from an exactly restored state give coverage 0.230/0.246/0.197
   and 50-126 % relative spread on normal force
   ([force reproducibility](2026-09-19-force-reproducibility.md)), so each boundary
   query needs repeats to resolve a flip, and a single displaced command does not
   separate at all (median 1.45 pooled SD). The affordable unit is a four-decision
   segment with five repeats: about 480 decisions, roughly 45 s per state
   improvement on this workstation under the review's corrected throughput.
   Against that, the measured value of
   choosing per state is .002-.003 ([recovery decisions](2026-09-18-recovery-decisions.md))
   and held-out duration/perturbation selection buys .005
   ([action selection](2026-09-20-action-selection-evidence.md)). The clean-slate
   pass already rejected thousands of simulated decisions per state target on this
   workstation budget.
4. **Its core family was probed here on 2026-09-20 and produced no novelty.** The
   terminal-event-weighted branching estimator in
   [clean-slate](2026-09-20-clean-slate-rl-research.md) is the nearest in-repo
   mechanism: corrected branching cut gradient variance 18x over independent
   likelihood scores, the proposed score-aware priority did not improve on it
   (1.05632e-5 versus 1.04587e-5), and P3O, SPO, weighted ensemble and particle
   value functions cover the core.
5. **Its premise is no longer a project constraint.** The owner removed the
   IPC-specific-contribution requirement on 2026-09-20 (`rule.md`). "We already pay
   for IPC, so exploit it" is a design option, not a reason to rank it first.

Condition under which it becomes live: a learner whose distribution straddles the
elbow event with repeat-separable probability, measured over four-decision
segments rather than single commands. Producing that distribution is the
transfer/experience problem below. Once it exists, the estimator would still have
to beat plain likelihood-ratio branching at equal wall time, which the in-repo
probe has not shown, and beat plain SAC, which crossed the first plateau unaided.

## Direction 2: active contact identification

Not refuted, but mis-sited by the proposal. Two records apply.

- At 48 late stalled states, the per-state decision is worth .002-.003, inside the
  repeat spread, while a fixed macro captures nearly all the gain. Identifiability
  from deployment observations is real but small (top-1 .67 against .44 constant,
  privileged .75) — so at those states, the information the proposal wants to
  acquire is not what is missing.
- The proposal's own scope (early states with genuine action divergence) is
  untested here, and the cheapest existing instance of it is the
  [transfer audit](2026-09-20-recovery-transfer-audit.md): the teacher reads named
  opening/alignment vertices and stage history and succeeds 4/4 from both prefixes
  (sustained coverage .979/.988); the student reads a single-frame point cloud and
  succeeds 0/4 from the same restored states, with translation command RMS error
  rising from 2.29 mm on its training rows to 5.22/5.35 mm on freshly executed
  successful teacher trajectories, against an 8 mm cap. Whether that geometry is
  inferable from the deployed observation is exactly the direction-2 question, and
  it is answerable with existing artifacts and a matched history/privileged
  ablation rather than a new probing algorithm.

## Direction 3: decision-relevant calibration

Correctly deferred by the proposal for lack of real contact data (no robot here;
a Stretch 3 is a later step). A second block applies that the proposal could not
know: force does not reproduce **inside our own simulator** across identical
commands from an exactly restored state, so a sim-versus-real force-ranking
calibration is blocked at both ends.

## What the proposal does not cover

It argues correctly for whole-task success rules, wall-time accounting and early
grasp retention. It has no access to four measurements that currently dominate
this project:

1. **Controller-to-policy transfer failure** — a verified controller exists
   (4/4, .98 coverage) and its improvement does not survive execution by the
   fitted policy (0/4). No direction in the proposal addresses the projection step.
2. **The success rule is broken past the shoulder** — `wang_progress` reads
   1.00 -> 0.00 in one decision, and 4 of 25 scripted-expert "failures" are
   complete dressings.
3. **Grasp is lost before the interesting phase begins** — all eight tshirt_26
   controls first violate the 2 cm whole-episode criterion at decisions 32-50,
   before their first positive upper-arm coverage at decisions 61-65. Any method
   that intervenes late cannot rescue that cell.
4. **The behavior prior does not contain the helpful actions** — the recovery
   macros that beat the policy sit at flow-prior percentile 1.000, |z| 5.2-5.5.

## Position

Do not adopt direction 1 as the main line on the strength of this proposal. Its
mathematics is correct and its prior-art screening is honest, but the quantity it
estimates is either already handled by ordinary SAC (threading, grasp) or
measured at zero (the elbow) in this system, its query cost collides with the
single-workstation budget, and its closest in-repo relative produced no defensible
novelty. Direction 3 stays blocked. Direction 2 is worth keeping only in the form
the transfer audit already predeclared: which deployment-available information and
which corrective labels let a verified improvement survive execution by the learned
policy, measured as complete grasp-valid episodes per workstation hour.

This is a ranking of research risk against evidence, **not** a validated novelty
claim for any direction, and not a demonstration that a boundary estimator would
fail in a system where the decisive event is actually reached.
