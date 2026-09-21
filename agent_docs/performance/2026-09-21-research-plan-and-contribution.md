# The research plan, and what the RL contribution would have to be

Date: 2026-09-21. Branch `research/joint-policy-query`. **This is a proposal with
preregistered gates, not a result.** No novelty is established here; every stage
below can end the line. It is written against the measurements in
[event boundaries](2026-09-21-event-boundary-analysis.md), the
[completed continuation](2026-09-21-feasible-segment-result.md), the
[transfer audit](2026-09-20-recovery-transfer-audit.md) and the
[literature audit](2026-09-21-research-literature-audit.md).

## 1. The problem instance, stated in the terms the algorithm must address

Everything here is measured on this project, not assumed.

- **The objective omits the criterion that decides success.** `dressing_reward.py`
  is Wang arm progress, a collision penalty and centre alignment. Success also
  requires an **absorbing trajectory constraint**: every per-decision anchor
  tracking maximum <= .02 m. At tshirt_26 decision 40 the highest-coverage macro
  violates it in 24/24 branches.
- **The constraint is controllable only at segment granularity.** No single action
  inside the policy's trust region changed the feasibility label at any of eight
  audited states; eight-decision segments flip it deterministically (0/24 against
  24/24 by direction).
- **A short verification does not certify.** `lift` passes an eight-decision check
  16/16 and then violates at relative decisions 17-21 in 11/16 branches. The
  certifying horizon is therefore itself a variable, not a constant to tune once.
- **The two quantities have opposite measurement profiles.** Validity is nearly
  deterministic given the segment (eta^2 .92-1.00; zero disagreement among
  identical repeats on tshirt_392) but needs a *long* horizon. Return is noisy
  (mixed-outcome cells .18-.27; a naive two-probability separation needs tens of
  repeats) but its ranking stabilises from about twenty decisions
  (rho .65-.98) — except at the earliest states, where a short reading is
  *anti-correlated* with the outcome (rho -.10 to -.24).
- **A verified controller exists and its improvement does not survive fitting.**
  The geometric teacher reaches .979/.988 sustained coverage with valid grasp,
  4/4 from both prefixes; every fitted student is 0/4 from the same restored
  states, with motion-command RMS rising from 2.29 mm on its training rows to
  5.22/5.35 mm on freshly executed successful teacher trajectories against an
  8 mm cap.
- **The simulator restores exactly** (zero position error across all audits),
  which is the asset multi-fidelity policy-gradient work explicitly cannot use
  under contact, and it costs .0610 s per training transition and .0875 s per
  decision of a long restore-based continuation.

Read together: the deployable policy does not fail because it cannot imitate a
motion. It fails because **it cannot tell, from its own observation, which
achievable change of the garment-body relation preserves an absorbing constraint
under its current grasp** — and nothing in its training signal ever told it.

## 2. What the contribution would have to be, stated so it can fail

**Claim.** Under an absorbing trajectory constraint and an expensive restorable
simulator, the budget that decides policy quality is not how many transitions are
collected but **which verification continuations are bought**. The operator
therefore allocates over a four-tuple

```text
(state to verify, candidate goal, quantity in {validity, return}, horizon)
```

to maximise the **retained** improvement of the *refitted deployable policy* per
workstation second — retained meaning measured after fitting and execution, not at
the verified segment.

The structural reason such an operator can exist is the measured asymmetry above:
validity is cheap in repeats and expensive in horizon; return is the reverse. A
single allocation rule over candidates at one fixed horizon — which is what this
project's own [joint operator](2026-09-20-clean-slate-rl-research.md) and the
standard designs do — cannot express that, and gets both quantities wrong: it
over-repeats the deterministic one and under-extends the one whose horizon
decides the answer.

**What must be shown, at equal total workstation seconds including labelling,
fitting and evaluation**, on held-out garment/body cells: higher validity-weighted
sustained coverage, and more valid completions, of the executed refitted policy
than each of

1. uniform full-length verification of every candidate;
2. DAgger with uniform teacher queries on learner states;
3. first-violation early stopping (which would have cut this project's completed
   run from 12,384 to 5,707 branch decisions);
4. a constrained-BAI-style allocator at a fixed horizon
   ([Cai and Kandasamy, AAAI-26](https://ojs.aaai.org/index.php/AAAI/article/view/39063));
5. the same learner with the constrained objective and no allocation at all.

**Prior art this must be argued against, explicitly.** Cai and Kandasamy already
choose an arm *and* whether to test performance or feasibility; the separation of
tests is theirs, not ours. Multi-fidelity BAI already optimises cost across
fidelities. Recovery RL and safety critics already separate progress from
constraint satisfaction. DAgger already collects corrective labels on learner
states. The only defensible remainder is the **sequential, execution-aware** part:
the candidate set is generated by a policy that changes, verified validity is a
property of the *continuation policy* and therefore expires when the policy is
updated, the tests share a prefix and are correlated, and the payoff is measured
after policy fitting rather than at the arm. If the comparison in row 4 matches
the operator, the claim is dead and we say so.

## 3. Plan, with the gate that can end each stage

Budgets use this project's own measured throughput: .0610 s per training
transition, .0875 s per continuation decision, 733 s for a 25-cell scripted-expert
collection, about 458 s for a 25-cell evaluation.

### Stage 0 — repair the objective and the instrumentation (hours)

Fix the success rule past the shoulder, log the approach prefix's tracking maximum
(a defect found in the collector: positive counts currently certify the branch,
not the episode), then train the same learner with the first-violation constrained
objective

```text
c_t = 1[e_t > .02] * prod_{j<t} 1[e_j <= .02],   E[sum_t c_t] = P(constraint failed)
```

against the original objective, same data and budget, history augmented so the
absorbing state is representable.

**Gate.** If this alone produces valid completions, the finding is an
objective/benchmark repair and must be reported as such. Re-scope before claiming
any estimator or allocator contribution. This is the most likely outcome and the
cheapest to obtain.

#### Stage 0 protocol, fixed before the runs

Implemented in `obs.py`, `dressing_env.py`, `sac.py`, `train_sac.py`; tests in
`python/uipc_manip/tests/test_constraint_objective.py`.

- **What changes.** The environment reports a per-decision cost that is 1 on the
  decision where the anchor tracking maximum first exceeds 2 cm and 0 for ever
  after, so the expected cost of an episode is exactly the probability that it
  violates. The observation gains one float, the flag saying the episode has
  already failed the criterion, which is what makes the absorbing state
  representable at all. The critic learns the value of `r - lambda * c`, and
  `lambda` is a dual variable raised while the sampled violation rate exceeds the
  budget. The reward itself is untouched.
- **Why the penalty is applied at update time, not at collection.** A multiplier
  folded into the stored reward is frozen at the moment a transition was recorded,
  so a buffer spanning a rising `lambda` would hold inconsistent targets. The cost
  is instead reconstructed from the stored transition as the rise of its own flag,
  which is exact because the flag is absorbing, and priced with the multiplier in
  force for that update.
- **Two arms, differing only in the objective.** Both carry the extra observation
  slot (constant zero in the control), both start from scratch, same seed, same
  25 cells, same 270,000 transitions, same everything else. Control:
  `--constraint-objective` with `--constraint-lambda-lr 0`. Treatment:
  `--constraint-objective` with lambda lr **.1**, budget 0, cap 50.

  The rate was raised from .02 before launching, for a scale reason rather than a
  result: at discount .998333 a return is worth about 600 rewards, so the critic's
  values run to the hundreds, and a one-off penalty only competes with them at
  lambda in the tens. An episode of 300 decisions contributes at most one unit of
  cost, so a batch's mean cost is about .003 even when every episode violates; at
  .02 the multiplier would need some 170,000 updates to reach 10, which is most of
  the run. At .1 it reaches about 20 in 67,000. With budget 0 the multiplier never
  decays, which is the intended behaviour and the reason the cap matters.
- **Metrics reported per evaluation:** valid grasp success, grasp valid rate
  (its complement is the violation rate), final and peak upper-arm ratio,
  and the `lambda` trajectory.
- **The three readings, fixed in advance.**
  1. Treatment produces valid completions where the control produces none: the
     finding is an objective/benchmark repair. Report it as that, and re-scope the
     rest of this plan before claiming any estimator or allocator contribution.
  2. Treatment stops moving — validity high, coverage collapsing toward the
     `retreat` behaviour already measured at .000 coverage with 24/24 validity.
     That is the known failure mode of pricing the constraint alone, and it is
     reported as a negative result, not hidden.
  3. Neither: the constraint alone is not the binding obstacle, and Stage 1's
     representation question carries the weight.
- **What makes the run uninformative:** if `lambda` never leaves its initial value
  the penalty never bit and nothing was tested. It is logged every update and
  checked at the first evaluation rather than at the end.

### Stage 1 — move the decision to the relation, keep the executor we have (about a day)

Replace `point cloud -> 6-DoF command` with `observation -> desired garment-body
relation`, executed by the existing geometric feedback controller. Labels come
from the verified teacher across the 25 cells. Baselines: the current
action-space BC student, and DAgger on actions.

**Gate.** Does the relation interface close any of the measured teacher-student
execution gap (4/4 against 0/4)? If it does not, the representation story ends
here and the work returns to Stage 0's objective.

### Stage 2 — measure the profile, then build the allocator (the contribution)

First measure, per state and per quantity, the joint (repeats x horizon) profile:
how many repeats and how long a continuation are needed before the answer stops
changing. Half of this exists already. Then implement the allocator and run the
equal-budget comparison in Section 2.

**Gate.** Beat rows 3 and 4 at equal seconds, or report that the known baselines
match it and stop.

### Stage 3 — generalisation and cost, reported honestly

Held-out garments **and** bodies from the 5x5 cell grid, per-cell and aggregate.
Report validity-weighted coverage, valid completions, and total workstation hours
including label collection, fitting and evaluation — not update counts.

## 4. What would make this not a contribution

Stated in advance so it cannot be argued away later: if Stage 0 captures the gain;
if a single fixed direction (`outward` already preserves validity 16/16) matches
the learned relation policy; if the constrained-BAI allocator at a fixed horizon
matches the operator; if the gain appears at the verified segment but not in the
refitted policy's executed episodes; or if the total workstation time exceeds the
baselines for the same executed success. Any of these ends the line, and the
result is reported as a negative one.
