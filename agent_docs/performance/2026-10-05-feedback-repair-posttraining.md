# Candidate: post-training from complete observable feedback repairs

Status: a concrete, unvalidated method hypothesis. The compiler, bounded repair
execution, source-only router fitting and guarded physical pipeline are now
implemented. The source pilot is running; no teacher gain, actor gain or real-robot
transfer is established.
The [completed robustness study](2026-10-05-sim2real-posttraining-study.md) motivates
checking transfer, but does not establish that observation aliasing or loss of
plan consistency caused our failures. Those mechanisms must be tested first.

## Proposed contribution and its limits

Change the object used to post-train the existing policy. Instead of one
privileged action label at each state, use a distribution over **complete,
executable feedback repairs**. Each repair has a short common prefix, a branch
selected from observed garment response, a corrective continuation, and a return
to the frozen policy. Keep the chosen repair identity throughout its execution.
Select and absorb only repairs whose complete task outcomes support improvement
across source models, including the cost of mistakes by the observable router.

Example: lightly move the cuff; if it advances relative to the hand, continue;
if it moves with the hand or folds back, use a different correction. This is a
proposed behavior, not one that our data has already demonstrated. It must use
point-cloud history and executed tool motion, not hidden contact states, material
IDs, exact body velocities, force labels, or future observations.

The narrow prospective contribution is a practical supervision operator that
preserves the **joint distribution of successful feedback repairs** during
generative-policy post-training. It must reduce harmful updates and preserve
improvement under a held-out physical domain at a fixed query budget. New action
proposals are essential: recalibrating the current three actors cannot create
improvement on cases where all three fail.

Trees, robust policy mixtures, latent modes, memory, flow matching, base-policy
anchoring, and training a deployable student are individually established. This
proposal is insufficient for a method paper unless its particular supervision
and absorption mechanism outperforms the closest simpler methods. A successful
IPC planner alone would not establish that contribution.

## Finite supervision operator

Let `h` be causal camera/tool/action history; `pi_0` the frozen FMVP fallback;
`m` a source model including cloth physics, sensing and independent reset noise.
A repair `T` consists of a prefix `u_T`, an observation-only response router
`g_T(h)`, and one closed-loop suffix `v_T,b` for each branch `b`. After a bounded
repair window, every alternative uses the same `pi_0` continuation. Full dressing
success and grasp retention are the outcomes; short progress is not a terminal
value proxy.

For each prefix, measure:

`q_m,b = P_m(g_T(H_after_prefix) = b | u_T)`

`V_m,b,a = E_m[Y | u_T, observed branch b, execute suffix a, then pi_0]`.

The conditioning matters: do not assign the same hidden material state to every
branch, and do not fit `q` from material IDs. Wrong branches must be included in
`V`; branch observations and outcomes come from physical continuations. Candidate
suffixes may be state-feedback controllers; they need not run open-loop.

Enumerate a small bank of branch assignments, yielding executable repairs. For
each complete repair:

`Delta_m(T) = sum_b q_m,b V_m,b,T(b) - J_m(pi_0)`.

Include the original policy as a zero-gain alternative. Compute a distribution
over whole repairs by the linear program

`maximize_(lambda,t) t`

`subject to sum_T lambda_T Delta_m(T) >= t for every m,`

`lambda_T >= 0, sum_T lambda_T = 1`.

This linear program is ordinary robust mixture optimization. Its expression is
**not claimed as new**. The proposed post-training mechanism uses its joint
solution as supervision and accounts for observable execution before selecting
labels. Replacing it with per-model privileged argmax actions would answer a
different, easier problem.

The implemented `compile_feedback_repairs.py` enumerates `A^B` assignments for
`A` suffixes and `B` response branches, then solves this LP. With two branches
and two suffixes the bank has four repairs plus fallback. It consumes point
estimates; it does not implement uncertainty calibration or certification.
Unobserved branches have unknown values: use fallback or pessimistic intervals,
never invent favorable conditional outcomes to complete the table. Independent
physical replicas and held-out router predictions are required
before using estimates to update a dressing actor. A finite-source minimum
does not cover a real system outside the source models.

## Actor update and deployment

Post-train a camera-only residual or flow actor initialized from the existing
policy, with the same translation/action conventions. A small internal repair
mode `z` is sampled once at a repair's start and retained for the repair window.
Its distribution is learned from the compiler's `lambda`; it contains no
privileged environment information. Branch choices use new observations.

Use the following losses:

`L_mode = - E_h sum_T lambda_T(h) log p_theta(z=T | h)`

`L_route = E_i sum_b g_theta(b | h_i,z_i) C_i,z_i,b`

`L_action = E_(T~lambda, physical branch histories) [w(h_0) L_FM(u_T,b | h_i,z=T,b)]`

`L_total = L_mode + L_route + L_action + beta L_keep(pi_theta,pi_0)`.

`C_i,z,b` is the measured loss in full outcome from routing a physical sample to
branch `b`, relative to the best admissible branch of that same repair. It can
supervise equivalent good branches without requiring material classification.
Use out-of-fold estimates when fitting the router; no future outcomes are inputs.
`L_FM` is the existing conditional flow-matching loss. `w` is zero on unsupported
root-level improvement; small-sample uncertainty must be disclosed. `L_keep` is
the existing reference-output retention control on fallback examples.

The whole repair must stay associated with its mode in the training data. Do not
average incompatible suffix actions, resample modes at every decision, or train
each branch from unrelated marginal winners. An ordinary multimodal flow policy
with sufficient history might already preserve the relevant associations; this
is a required baseline, not a weakness to assume.

Teacher advantage, router predictability and actor absorption are three separate
checks. Evaluate the actual observation-only actor. If the compiler improves but
the actor does not, the method has not delivered transferable post-training.

## Prior-art objections

| Primary source | Overlap and required distinction |
|---|---|
| [A2D, ICML 2021](https://proceedings.mlr.press/v139/warrington21a.html) | Already changes the privileged expert to maximize the imitating student's return. “Make a teacher aware of what the student can see” is not novel. Our finite repair-bank construction must offer a query/absorption advantage over this principle. |
| [BIG, 2024](https://arxiv.org/html/2407.00495) | Already addresses missing information-gathering behavior when learning from privileged demonstrations. A probing move is not new; its dressing benefit remains a hypothesis. |
| [Privileged POMDP learning, 2024/2025](https://arxiv.org/html/2412.00985), [To Distill or Decide, 2025](https://arxiv.org/html/2510.03207) | Already analyze privileged distillation failures and belief-based alternatives. Our Q4 data does not independently establish those failure mechanisms. |
| [LCEOPT, AAAI 2024](https://arxiv.org/abs/2305.08049), [Pessimistic Iterative Planning](https://openreview.net/pdf?id=tQMBxQZblv) | Policy-tree search, Monte Carlo outcomes, recurrent controllers and robust partial-observation planning are established. IPC trees alone would be an implementation. |
| [CoPlanner, 2025](https://arxiv.org/html/2509.17080) | Shared prefixes, later branches and contingency-aware diffusion already exist. Moving those components from driving to dressing does not supply algorithmic novelty. |
| [Play-LMP, CoRL 2019](https://proceedings.mlr.press/v100/lynch20a/lynch20a.pdf) | Trajectory-level latent plans and plan-conditioned feedback policies already exist. Persistent identity and its action loss are not independently new. |
| [Why Does Action Chunking Improve BC?, 2026](https://arxiv.org/html/2608.02547) | Delayed prediction and randomized delay ensembles explain action-chunking gains in the evaluated tasks without needing joint action consistency. A same-student randomized-delay control is needed if actor absorption proceeds; suffix resampling alone does not implement that baseline. |
| [Simulation Distillation, 2026](https://arxiv.org/html/2603.15759) | Sim-pretrained task structure plus real-data dynamics adaptation and online planning already improves hardware behavior. Whole-repair post-training needs a demonstrated cost/absorption advantage over this alternative. |
| [Set-Supervised Diffusion Policy, RSS 2026](https://arxiv.org/html/2606.01865) | Learns distributions of desired action chunks using positive/negative corrections. Our claim must concern transfer-aware, observation-executable **joint repair supervision**, not “use sets instead of one action.” |
| [ACPPO-Corr, 2026](https://arxiv.org/html/2609.36250) | Already adds stepwise feedback to chunked actions. Merely reacting during a chunk is not a contribution. |
| [Robust baseline regret](https://arxiv.org/abs/1607.03842), [Sim2Act](https://arxiv.org/html/2603.09053v1) | Base-relative robust improvement and action-ranking-aware calibration already exist. The LP and calibration do not independently establish novelty. |
| [World/behavior grounding, 2026](https://arxiv.org/html/2610.00821) | Real co-training gains depend on grounded simulated behavior outside real-data coverage. An observation-based branch cannot repair an incorrectly modeled suffix by itself. Real paired outcomes remain necessary. |

**Strongest reviewer objection:** this may be robust POMDP planning plus
set-supervised/latent-mode imitation, with no new learning principle. The only
plausible narrow method claim is a demonstrated improvement in how cheaply a
robust, observation-executable correction distribution is transferred into a
pretrained actor. If matched robust recurrent training or ordinary history-flow
distillation is equally effective, use those methods and drop the novelty claim.

## Falsifiable prediction

At matched candidate banks, physical queries and actor capacity, retaining
whole-repair associations should preserve more of the observation-only teacher's
gain on an unseen physics model than best-action labels or independently mixed
branch labels. Predeclare a meaningful pilot effect as **at least 10 percentage
points more complete-task success**, with positive gain over the frozen policy
and no extra target demonstrations. Report uncertainty; the threshold is not a
significance test.

The distinguishing controls are: same causal history for every student; flat
ensemble-best chunks; a history-flow student trained on single winners; the same
repair data with mode labels independently shuffled between prefix/suffix; and
a response router restricted to the repair-start frame. These isolate additional
data, memory, feedback, and preservation of plan associations. Include a
privileged router only as an upper bound, never as the deployed result.

## Smallest decisive experiment and cost

**A mechanism pilot, not another old-policy benchmark.** Freeze two development
cases (the first two development bodies, `tshirt_26`), three source models
(nominal, bending ×2, density ×1.5), three candidate short prefixes, and two suffix
controllers generated from the existing action proposals. The full baseline plus
six prefix/suffix combinations give seven candidate continuations per case/model.
Run two independent replicas: **2×3×7×2 = 84 source attempts**. Then freeze the
observation-only router and repair mixture. Use a new joint bending+density shift
for **2×1×7×2 = 28 target attempts**. Here the seven target controllers are the
frozen policy, best flat correction, compiled observation-only repair, privileged
router upper bound, initial-frame-only router, shuffled repair associations, and
per-decision mode resampling. **Execute these actual closed-loop controllers**;
do not infer their returns by splicing outcomes from separately settled runs.
Total: **112 complete attempts**.

Use actual simulated histories from each material's replayed prefix. Never paste
a nominal deformed cloth snapshot into another material. Save numerical state
differences and use matched reset blocks; do not assert bitwise identical
counterfactuals without verification. Candidate proposal seeds, prefix lengths,
suffix horizon, camera noise used for router fitting, and branch features must
be frozen before these outcomes. Prefix queries/setup/retries count in the budget.

This pilot first asks whether an *observable* repair bank has any gain over the
best flat correction. If it does not, stop before actor training. If only a
privileged router improves, stop this proposal. Two replicas and two bodies are
only a rejection screen, not enough to validate confidence or a publication.

The completed Q4 cost was 3.44 worker-hours for 30 seven-slot batches, or about
6.9 minutes per batch. Sixteen analogous seven-candidate batches suggest
**roughly 2–4 single-worker hours** for this pilot, including startup headroom;
material/contact trajectories may be slower. Ceiling: four worker-hours and
112 attempts, all retries charged. The compiler alone launched no GPU campaign;
the subsequently authorized physical pilot is now running, as recorded below.
The one-worker / below-90-GiB constraint
continues to apply.

If the observable repair bank survives, use a separate development set for
three-seed matched student training, then held-out garment/body/motion evaluation.
Those costs are additional and cannot be inferred from the tiny pilot. The first
hardware comparison can reuse the earlier proposal's 12 matched pairs with two
repetitions: **48 real attempts**, approximately 1.6–3.2 robot-hours at 2–4 minutes
including reset, before separate actor evaluation. Response and outcome calibration
must be assessed on different reset blocks.

Stop if there are no successful alternative repairs; visible histories do not
support useful branching; camera perturbations destroy routing; the gain vanishes
under independent replicas; mode shuffling does not change deployed performance;
the simpler history-flow control is equally good; or the real comparisons reverse
the gain. Any of those results prevents the claimed method contribution.

## What exists now

`compile_feedback_repairs.py` implements finite repair-mixture supervision.
Three analytic checks pass: complementary whole policies can improve robustly;
observable response enables more gain than an aliased response; and uniformly
harmful repairs return to the frozen policy. These are numerical checks of known
optimization semantics, **not dressing experiments or novelty evidence**.
Bounded candidate execution and source-only router fitting are now implemented.
Actor absorption and hardware verification remain unimplemented.

## October 5 execution: frozen rejection pilot

The owner asked where the actual processing was and why work had stopped. The
earlier turn had finished only analysis and the CPU compiler. The continuation
now runs `run_feedback_repair_pilot.py`, using a new optional hook in the existing
`collect_garment.py`; it does not restart Q4, M4 or previous collectors.

Supervisor **659533** and first worker **659650** started source collection.
Live records, source snapshot and exact commands are under
`output/uipc_manip/feedback_repairs_20261005/`. Read `status.json` for the current
worker, `events.jsonl` for all admissions and costs, `supervisor.log` for batch
completion, and `runs/*.log` for actual IPC steps. Historical PIDs in this section
are launch records; the status file is authoritative.

Frozen settings, selected before this pilot's outcomes:

- Bodies 1032 and 1041, tshirt_26, source physics nominal / bending x2 /
  density x1.5; held target is the new joint bending+density shift.
- Two reset seeds, 2026100510 and 2026100511. All seven controllers in a block
  share the nominal placement and seed. Slot ordering rotates between batches.
  Physics is set before settling; no deformed snapshot is reinterpreted under
  another material. Full trajectories retain numerical initial states.
- FMVP runs through decision 59. At decision 60, execute one of three
  12-decision feedback prefixes: half nominal action, nominal translation with
  rotation removed, or nominal plus 0.25 normalized world-z translation.
  These reuse `IPCActionFilter` proposals; they are not new skills.
- Then execute an 80-decision suffix: either half nominal action or nominal
  translation with rotation removed. Query the base policy every decision.
  At decision 152 all alternatives return to frozen FMVP. Early success uses
  the unchanged physical-sleeve / valid-grasp 20-step hold.
- The router receives 18 segmented visible-cloud statistics at repair start,
  18 changes after the prefix, and three executed virtual-tool displacements.
  It ignores packed goal/attachment scalars, forces, true material IDs and
  hidden cloth geometry. No extra point noise or dropout is used in this
  mechanism pilot; camera perturbation robustness remains untested.
- Fit a cost-sensitive decision stump with fixed quartile threshold proposals
  on one replica and apply it to the other to estimate source values. Refit
  both replicas for the frozen target router. Empty conditional cells have
  pessimistic value zero and explicit support counts. Termination before
  routing is an absorbing outcome, including any early success.
- The mixture LP uses three physics models, averaging the two development
  bodies. Maximize worst-model gain first, then mean gain among ties. This
  avoids arbitrary fallback when an easy source is at ceiling. Pure zero-gain
  ties retain FMVP. This is ordinary optimization, not a novelty claim.
- Strengthen the flat control to a **robust mixture of complete flat
  corrections**, using directly measured full outcomes. A feedback win cannot
  be attributed merely to permitting a mixture for only one method.
- The privileged-state router is an information-rich control trained on the
  same source outcomes, **not a guaranteed oracle upper bound**. Target mode
  draws are coupled by common seeds across controls; resampling keeps the
  executed prefix fixed and changes suffix identity at every decision.

The frozen source gate requires new repairs of base failures in both replicas,
nonnegative worst-model gain, positive mean gain, and at least 0.10 mean gain
over the robust flat mixture using cross-fitted source estimates. If it fails,
save the result and do not spend the 28 target attempts or launch actor training
for the unchanged bank. This is a rejection rule, not a confidence certificate
or evidence that every feedback method is impossible. If it passes, execute
actual closed-loop target controllers; do not assemble target successes by
selecting already measured outcomes.

Every worker admission charges seven attempts, even on setup failure. Setup,
saving and aborted worker wall time count toward the four-worker-hour cap.
Maximum 112 attempts, one worker, admission below 84 GiB, terminate only that
worker's process group at 89.75 GiB. No unrelated processes are modified.

Nine focused CPU tests pass: the three original LP cases plus camera-only input
integrity, tool-frame compensation, common-prefix and fallback behavior,
privileged-input independence, the LP ceiling tie, and rejecting a no-repair
source bank. These verify implementation contracts, not scientific efficacy.

### First completed batch and interpretation

Nominal physics / body 1032 / repeat 0 has finished: base and all six repair
combinations fail (0/7). Base loses the grasp at decision 141; corrections end
between decisions 125 and 300. Longer grasp retention is not dressing success.
All seven records have a visible response feature. The read-only analyzer
confirms exact static body/garment/grasp arrays and initial tool positions, zero
force input, and interventions confined to the declared decision window.

Settled cloth vertices differ by up to **3.924 mm**, and by decision 60 the
largest vertex difference is **0.361 m**, before a repair has executed. This is
important outcome noise and trajectory divergence, not identical-state
counterfactual evidence. The experiment estimates controller behavior over
matched reset blocks; two replicas will remain too few for strong causal or
statistical claims. Preserve these diagnostics in `analysis.json` and do not
replace a failed comparison with a tolerance selected after seeing outcomes.

The frozen collector also preserves the historical FMVP yaw adapter. Its actions
are not a strict three-translation-only hardware policy. The brief's translation
interface must be matched for both the frozen base and any candidate before
training a deployment actor or claiming hardware gain. This mechanism screen
does not certify that action-interface transfer.
