# Are task-event boundaries locatable near the policy? — 2026-09-21

Status: analysis of 2,424 previously collected branches. **Interpretation corrected
on continuation of this research:** event-conditioned coverage gaps are not measured
action-space discontinuities; mixed-repeat fractions are not flip probabilities;
short survival is not an episode feasibility certificate. See the audit below.
The predeclared long-horizon experiment is now [complete](2026-09-21-feasible-segment-result.md).

Original data scope: read-only analysis of 2,424 branches that already exist on disk
(the earlier narrative reported 2,400; the six input collections sum to 2,424). No new
simulation, no training, no algorithm claim. Script `scripts/analyse_event_boundaries.py`,
artifact `output/uipc_manip/event_boundary_analysis_20260921/summary.json`.

## Why this was measured

An external proposal ranks "event-probability gradients" first: estimate
`grad_theta Pr(success)` by adding a boundary term `p(b) [G(b+) - G(b-)]` over the
action-space boundaries at which a task event flips. Its own exit criterion is that
a stable, important outcome change must exist in the early action neighbourhood.
Three things must hold before that term can be estimated at all:

1. the event is **undecided** at the state — both outcomes occur nearby;
2. the indicator is **attributable to the action**, not to solver noise;
3. the jump `G(b+) - G(b-)` is **large enough to matter**.

All three are measurable from the decision-branch collections already collected:
`decision_branches_early_20260919` (steps 20/40/60/80, two cells, 672 branches each)
and `decision_branches_20260918` (steps 100/140/180). Each collection ran 7 macros
x 8 seeded slot-states x 3 identical-command repeats from exactly restored states,
an 8-decision macro window followed by 32 decisions of the same policy.

A (state, macro) cell whose three repeats disagree exhibits outcome variability
under the same intervention. Its event-conditioned coverage gap controls macro
identity, but does not locate an action boundary, identify one-sided limits, or
remove outcome-conditioning bias. The closed-loop continuation can issue different
actions once the simulated observations diverge.

## 1. The finding that reframes the question: the conflict is a constraint, not a choice

tshirt_26/14046, snapshot step 40, all 168 branches (24 per macro):

| macro | grasp kept | coverage > 0 | both | mean sustained coverage |
|---|---:|---:|---:|---:|
| **policy (incumbent)** | **0/24** | 20/24 | **0/24** | **.207** |
| policy_scaled | 0/24 | 12/24 | 0/24 | .109 |
| forward | 0/24 | 7/24 | 0/24 | .011 |
| retreat | 24/24 | 0/24 | 0/24 | .000 |
| outward | 24/24 | 10/24 | 10/24 | .003 |
| lift | 17/24 | 17/24 | 11/24 | .024 |
| retreat_outward | 24/24 | 4/24 | 4/24 | .001 |

The incumbent buys the highest in-window coverage by losing the grasp in **24 of 24**
branches. Every macro that respects the grasp criterion gives up 88-100 % of that
coverage; `lift`, the only macro that achieves both in 11/24 branches, covers .024
against the incumbent's .207. The same shape appears at step 60, where the incumbent
keeps grasp 24/24 at coverage .346 while `forward` covers .460 and loses grasp 17/24.

So at the states where the two conflict, they conflict **by macro and almost
deterministically**. The question at decision 40 is not "which action is better" —
it is which side of a hard whole-episode constraint to be on, and the in-window
score cannot express that. This is the measured form of two standing results:
[teacher supervision](2026-09-19-teacher-supervision.md) found that fixing grasp
validity alone removes motion the task needs (0/25), and the
[action-selection audit](2026-09-20-action-selection-evidence.md) found all eight
tshirt_26 controls violating the 2 cm criterion at decisions 32-50, before their
first upper-arm coverage at 61-65.

tshirt_392/14046 shows no such conflict at steps 40-80: nearly every macro keeps
grasp and covers, and the incumbent is already the best macro (.235, .490). The
conflict is cell-specific, not a property of dressing.

## 2. The grasp event is action-determined and decided within 4-9 decisions

Permutation test of macro identity against the event indicator, per state
(2,000 permutations), plus the decision index at which grasp is first lost:

| cell | step | event rate | identical-repeat split | eta^2 (macro) | median perm. p | first violation, decisions | 8-decision label = whole-branch label |
|---|---:|---:|---:|---:|---:|---:|---:|
| t26 | 20 | .29 | .21 | .76 | .030 | 7 / **27** / 39 | 52/168 |
| t26 | 40 | .53 | .09 | .92 | .002 | 0 / **9** / 21 | 124/168 |
| t26 | 60 | .90 | .02 | .94 | .006 | 6 / **7** / 7 | 168/168 |
| t392 | 20 | .95 | .00 | 1.00 | .005 | 4 / **5** / 5 | 168/168 |
| t392 | 40 | .86 | .00 | 1.00 | .004 | 6 / **6** / 6 | 168/168 |

At steps 40-80 the grasp outcome is essentially a deterministic function of the
commanded macro (eta^2 .92-1.00, zero disagreement among identical repeats on
tshirt_392) and it is decided **near the 8-decision command window**, so an
8-decision query reproduces the 40-decision label 168/168 times at three of the five
rows above. Zero mixed cells with three repeats does not justify certainty from one future repeat.

Step 20 is different and must not be read as a grasp-determinism number: the median
first violation is decision 27, i.e. after the macro ends, so the outcome there is
mostly produced by the policy continuation, not by the command under test. The
8-decision label agrees with the whole-branch label only 52/168 times.

## 2b. Feasibility is controlled at segment granularity, and the reward never sees it

Per-macro grasp-kept rate across the eight seeded states of tshirt_26/14046:

| macro | step 40, eight states | step 60, eight states |
|---|---|---|
| policy | .00 .00 .00 .00 .00 .00 .00 .00 | 1.00 x8 |
| policy_scaled | .00 x8 | 1.00 x8 |
| forward | .00 x8 | .00 .00 .00 .00 .33 .00 1.00 1.00 |
| retreat / outward / retreat_outward | 1.00 x8 | 1.00 x8 |
| lift | .67 .67 .33 1.00 .33 .67 1.00 1.00 | 1.00 x8 |

The outcome is a function of the commanded direction and is nearly constant across
states. These eight "states" are eight seeded slots of the *same* garment, body and
decision index, so low state diversity is expected and this is not evidence that
feasibility is state-independent in general; what it does show is that seeds and
solver noise do not decide it, so direction is informative in this sample. Three repeats do not establish
a high-confidence label for future execution.

The granularity matters. In the [selector audit](2026-09-20-action-selection-evidence.md)
none of the ten single-action candidates (projected-Q gradient, eight random
directions at normalised radius .5, and the base) changed the feasibility label at
any of its eight states. Only the two tshirt_392 states are informative there — the
tshirt_26 prefixes are already violated, so their label cannot move — but taken with
the macro table above, the constraint is controllable by an eight-decision segment
and not by one command inside the trust region. The same granularity conclusion the
[RAL calibration](2026-09-20-upper-bound-and-ral-calibration.md) reached for the
continuous score holds for the binary event, in a sharper form.

Finally, `python/uipc_manip/dressing_reward.py` states in its own docstring that the
reward is Wang's arm-progress term, a collision penalty and a centre-alignment
term, and that "no force, topology, coverage, or strain terms enter the reward";
the shoulder-extension option explicitly "does not ... certify grasp validity". The
2 cm whole-episode criterion that decides success lives only in evaluation
(`decision_branches.py:127` applies the same `max tracking error <= 0.02 m`). So at
decision 40 on tshirt_26 the coverage-maximising macro is the one that violates the
success criterion in 24 of 24 branches, and the training reward has no explicit term for that violation.

Three other cells at step 100 show the same failure mode in milder form: the
incumbent loses grasp in 4/8 branches on both tshirt_68 and the hospital gown,
where `forward` keeps it 8/8 and also covers more (.143 against .050 on tshirt_68).
There the alternative dominates outright rather than trading off.

## 3. The progress event has a large conditional coverage gap and mixed repeats

| cell | step | event | rate | identical-repeat split | eta^2 | median perm. p | within-cell conditional gap (n cells) |
|---|---:|---|---:|---:|---:|---:|---:|
| t26 | 40 | opening on arm at end | .72 | .18 | .65 | .098 | +.135 +- .035 (10) |
| t26 | 60 | opening on arm at end | .62 | .27 | .70 | .054 | +.201 +- .052 (15) |
| t26 | 60 | sustained coverage > 0 | .51 | .21 | .79 | .013 | +.416 +- .019 (12) |
| t26 | 80 | sustained coverage > 0 | .83 | .05 | .93 | .006 | +.532 +- .003 (3) |

**Circularity check.** The `sustained > 0` indicator thresholds the same quantity
whose conditional gap is reported. At step 60, 83 of 168 branches have zero
coverage and every other branch is >= .298; at step 80, 28 are zero and the rest
are >= .474. At step 40, nonzero branches range from near zero to .255, median
.067. These describe sampled coverage distributions. Neither a conditional gap
nor an empty interval in a finite sample establishes an action-space jump.

The mixed-repeat fractions of .18–.27 indicate outcome variability within some
state/macro cells. An independent Bernoulli comparison designed for a .25 mean
difference at three standard errors uses up to ~72 repeats per candidate = 2,880
decisions. At .362 s/decision this is ~1,000 s per candidate per state. This is a
specified naive design's cost, not the necessary cost of estimating a gradient.
The observed macro-selection coverage gains are +.005 to +.095 in these windows.

## 4. Cheap queries cannot substitute at the earliest states

Spearman correlation between a short reading and the 40-decision outcome, over all
168 branches at each step:

| cell | step 20 | step 40 | step 60 | step 80 | step 100 | step 140 | step 180 |
|---|---:|---:|---:|---:|---:|---:|---:|
| t26, 8 decisions | **-.18** | +.22 | +.32 | +.32 | -.07 | +.20 | +.36 |
| t26, 20 decisions | **-.24** | +.37 | +.65 | +.90 | +.84 | +.82 | +.89 |
| t392, 8 decisions | **-.10** | +.70 | +.95 | +.73 | +.85 | +.82 | +.63 |
| t392, 20 decisions | **-.19** | +.69 | +.98 | +.90 | +.89 | +.77 | +.70 |

At step 20 a cheap surrogate is not merely uninformative, it is **anti-correlated**:
early progress predicts a worse outcome. A short-horizon screen would actively
misrank candidates there. From step 60 a 20-decision reading is a good proxy
(.65-.98), so the expensive horizon is only needed at the earliest states.

## 5. Corrected interpretation of the estimator evidence

The branch data establish macro-dependent threshold outcomes and noisy progress,
not a located action-space boundary. The reported .135–.201 values are descriptive
conditional coverage gaps, not estimates of `G(b+) - G(b-)`. Sustained coverage is
also not the accumulated training reward, which the collector does not store.

The ~72-repeat calculation is `2 p(1-p) (3/.25)^2`, using independent Bernoulli
sample means, worst case p=.5, and a three-standard-error separation heuristic.
It is not a power calculation or a lower bound for likelihood scores, coupled
finite differences, adaptive sampling, pooled learning or boundary estimators.
The resulting ~1,000 s versus ~3 s comparison measures different tasks and
confidence levels. The previously asserted ~350x universal cost disadvantage is
withdrawn; it remains a warning about this particular naive estimation design.

The .18–.27 figures are the fraction of three-repeat cells containing both
outcomes, not per-execution flip probabilities. For three independent Bernoulli
repeats with common p, that fraction has expectation `1-p^3-(1-p)^3=3p(1-p)`.
Zero failures in three IID repeats still permits a one-sided 95% failure-rate
upper bound of `1-.05^(1/3)=.632`; pooling 24 IID trials would give .117, but slots
and repeats here do not establish the needed independent common task distribution.

At t26 step 40 the eight-step label agrees with the 40-step label only 124/168
times. The new long-horizon [result](2026-09-21-feasible-segment-result.md) makes
this limitation concrete: lift passes 16/16 short screens and fails 11/16 later.
An observed violation certifies failure of this absorbing branch criterion; a
short surviving prefix does not certify future validity.

The absence of an explicit grasp-tracking term in the training reward is verified.
It is an objective mismatch to test, not proof that no indirect learning signal
exists, nor proof that the highest-coverage macro maximizes accumulated reward.
The finite single-action audit supports limited controllability at that tested
radius, not an impossibility theorem about all single-action controls.

## 6. Predeclared experiment: does respecting the constraint pay at episode scale?

**Completed:** see [results and protocol limitations](2026-09-21-feasible-segment-result.md).
The text below is preserved as the original preregistration. Its alternatives are
not exclusive, and zero success does not isolate the pre-snapshot decisions.

Everything above is a 40-decision window. The open question it creates is whether a
constraint-respecting segment at the binding state produces a better **episode**, or
merely a feasible one that never dresses. Fixed before running:

- Cell tshirt_26/14046, the same warm-SAC checkpoint `checkpoint_00127416.pt`, eight
  seeded slots, snapshot **decision 40**, where the instantaneous tracking error is
  .0045-.0167 m (valid) and forearm progress is already .641-.671.
- Three arms: `policy` (the incumbent, grasp-invalid in 24/24 in-window branches),
  `lift` (17/24 valid, the only macro achieving both in-window) and `outward`
  (24/24 valid, near-zero in-window coverage). Eight decisions of macro, then the
  **same policy to decision 298**, two repeats, horizon 300.
- Reported per arm: whole-branch grasp validity, sustained coverage over the last
  twelve decisions, valid dressing success (sustained >= .7 with grasp valid) and
  final upper-arm ratio. Budget 12,384 physical decisions, external cap 3,600 s.
- Reading fixed in advance. If a feasible arm ends with higher *valid* sustained
  coverage than the incumbent, the constraint framing is supported and the binding
  state is repairable by direction choice. If every arm ends at zero valid success,
  direction choice at decision 40 is not sufficient and the decisions before it are
  implicated. If the incumbent's grasp survives to the end here, the in-window
  criterion and the episode criterion disagree and the in-window result above must be
  restated as a transient violation.
- This is one cell, one checkpoint, one snapshot index and two repeats: a development
  comparison, not independent task draws, and not a dressing success claim either way.

`collect_decision_branches.py` gains a `--macros` subset flag for this (the subset
must keep `policy` as the no-intervention reference); nothing else changes.

## What this does not establish

Two garment/body cells with one body seed each; the eight states per step are
seeded slots of the same cell, not independent tasks; three repeats per cell make
every within-cell probability coarse. The macro library is seven hand-written
directions, not the policy's action space, so "action-determined" means
"determined at this displacement scale by this library". The 40-decision window is
not an episode, and no row here is a dressing success. Step-20 rows measure the
continuation, not the command. The permutation test treats repeats as exchangeable
within a cell, which the exact-restore protocol supports but does not prove.

## Reproduce

```bash
/home/ge47gax/kun/genesis-world/.venv/bin/python scripts/analyse_event_boundaries.py \
  --root output/uipc_manip \
  --out output/uipc_manip/event_boundary_analysis_20260921/summary.json
```

Inputs are unchanged on disk; the analysis adds no physical decisions. The branch
collections it reads cost 18,724 s of simulation when they were collected in
September 2026 and are reused here at zero additional cost.
