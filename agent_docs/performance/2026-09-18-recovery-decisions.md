# Is there a recovery decision to learn at a stalled dressing policy? — 2026-09-18

**Subsequent audit:** see [source and held-out-repeat review](2026-09-18-recovery-decisions-review.md).
The measured small marginal value of state-dependent selection remains, but
several interpretations below require correction: recovery branches recompute
closed-loop commands; directional macros also remove rotation; repeat range is
not a significance threshold; state-wise predictor validation does not establish
held-out-configuration generalization or an irreducible sensing limit. The
original measurements and narrative are preserved below as the historical record.

Status: two collections complete, two more running. No policy was trained and no
solver, reward or controller setting was changed. This tests, before any learner is
built, the three premises of a recovery-decision method: that a better action exists
at the states the policy visits, that the simulator can say which one reliably, and
that *which* one depends on the state at all.

Code: `uipc_manip.decision_branches`, `uipc_manip.collect_decision_branches`,
`uipc_manip.decision_features`, `scripts/analyze_decision_branches.py`,
`scripts/identifiability_probe.py`. Artifacts under
`output/uipc_manip/decision_branches_20260918/`.

## Protocol

The saved 127,416-transition SAC policy is driven to decision 100, 140 and 180 of an
episode. At each, the world is snapshotted (largest restore error over the whole
study: 0.00e+00 m) and each of seven macros replaces the policy for eight decisions
before the *same* policy resumes for another 32, so branches differ only in the
intervention. Each macro is repeated three times with identical commands, because
continuations from one restored state do not reproduce
([predictability record](2026-09-18-predictability-horizon.md)).

Macros, all at the same 8 mm per-decision translation: `policy` (no intervention),
`policy_scaled` (the policy's direction at that magnitude, so a macro that wins only
by moving further is visible), `forward` (along the upper arm toward the shoulder),
`retreat`, `outward` (away from the arm's centreline through the tool), `lift`, and
`retreat_outward` (half the window each). The return is sustained coverage, the
minimum upper-arm ratio over the branch's last twelve decisions.

Eight slots of one garment/body cell start from different seeds, so each snapshot
yields eight decision states. Two cells complete: `tshirt_26/14046` and
`tshirt_392/14046`, 24 states and 504 branches each, 21,600 decisions in 40.8 and
43.5 minutes, no simulator error.

## Result 1: a better action exists, and the simulator can name it

| Cell | States | Policy return | Best macro | Consequence, median | Share of the gap to 0.7 | Decisive against the repeat spread | Repeat spread |
|---|---:|---:|---|---:|---:|---:|---:|
| tshirt_26/14046 | 24 | 0.567 | `lift` 0.579 | +0.013 | 10.1 % | 19/24 | 0.007 |
| tshirt_392/14046 | 24 | 0.602 | `forward` 0.632 | +0.027 | 28.2 % | 24/24 | 0.006 |

Ranking agreement across the three identical-command repeats is 0.75 and 0.90 for
the best macro and 0.74 and 0.83 over all macro pairs. So the label is real: at 43 of
48 states some macro beats the policy's own continuation by more than the noise of a
command sequence with itself.

**The policy's fault is direction, not step size.** `policy_scaled` commands the
same 8 mm as every macro in the policy's own direction and gains 0.001 and 0.006 over
the policy, against 0.012 and 0.030 for the best macro.

**The gain shrinks the later the intervention.** Consequence by snapshot decision:

| Cell | 100 | 140 | 180 |
|---|---:|---:|---:|
| tshirt_26 | +0.022 (15 % of the gap) | +0.011 (8 %) | +0.006 (6 %) |
| tshirt_392 | +0.055 (55 %) | +0.027 (30 %) | +0.012 (12 %) |

## Result 2: which macro is best barely depends on the state

| Cell | Best single fixed macro | Per-state oracle | Value of choosing per state |
|---|---:|---:|---:|
| tshirt_26/14046 | `lift` 0.579 | 0.582 | **0.003** |
| tshirt_392/14046 | `forward` 0.632 | 0.633 | **0.002** |

Both are below the 0.006–0.007 repeat spread. Within a cell the winner is nearly
constant (`lift` at 17 of 24 states, `forward` at 19 of 24); between the two cells it
differs. A per-garment constant would capture essentially all of the available gain.

## Result 3: the choice is identifiable, and the prize is inside the noise

One model class (ridge, leave-one-state-out, penalty chosen inside each fold) over
the 48 states, predicting every macro's return from each information set:

| Information set | Features | Top-1 | Pairwise | Mean regret | Top-1 on shuffled labels |
|---|---:|---:|---:|---:|---:|
| constant (always the globally best macro) | 0 | 0.44 | 0.67 | 0.007 | – |
| deployment observation | 23 | 0.67 | 0.84 | 0.002 | 0.42 (p95 0.50) |
| observation plus five-frame history | 183 | 0.62 | 0.83 | 0.003 | 0.40 (p95 0.50) |
| simulator's privileged state | 35 | 0.75 | 0.87 | 0.002 | 0.41 (p95 0.54) |

The better macro *is* identifiable from what the robot can see: 0.67 against a 0.44
constant baseline and a 0.42 shuffled-label control whose 95th percentile is 0.50.
The privileged state does better still (0.75), so some of the decision is not visible
at deployment. History does not beat the single observation here; with 183 features
and 48 states that comparison is capacity-limited, not an argument against history.

But the regret says what it is worth: choosing perfectly instead of always using the
single best macro saves 0.005 of coverage, inside the 0.006–0.007 repeat spread and
about 3 % of the median gap to a dressed arm.

## Reading

The three premises behave differently. A better action exists and the simulator can
name it reliably — those hold. *Which* action is best is nearly a constant per
garment — that does not. So a method that spends expensive simulation to decide which
recovery to take at which state would, at these states, be solving a problem that is
not there; a per-garment constant recovery captures the same gain.

The binding constraint is elsewhere and this measurement makes it quantitative: the
policy sits at 0.57–0.60 coverage, the best eight-decision intervention moves it by
0.013–0.027, and a dressed arm is 0.7. One local recovery closes a tenth to a quarter
of that gap at the early states and less later. Whether repeated interventions
compound over a whole episode is a separate question, which the full-episode arms of
`uipc_manip.recovery_intervention` are built to answer.

## The early phase, where the episode is actually decided

The gain from intervening shrank the later it fired, so the same study was run at
decisions 20, 40, 60 and 80 on both cells: 32 states each, 672 branches each, 27,520
decisions per cell in 146-166 minutes.

| | early (20-80) | late (100-180) |
|---|---|---|
| tshirt_26: best single fixed macro | **the policy itself**, 0.267 | `lift`, 0.579 |
| tshirt_26: value of choosing per state | **0.046** | 0.003 |
| tshirt_26: decisive states | 9 of 32 | 19 of 24 |
| tshirt_392: value of choosing per state | 0.000 | 0.002 |
| tshirt_392: decisive states | 0 of 32 | 24 of 24 |
| repeat spread | 0.010 | 0.006-0.007 |

Early, no fixed direction beats the policy's own action: the best single macro *is*
`policy`. But on tshirt_26 the per-state oracle is worth 0.046, fifteen times the late
figure and four times the repeat spread — the first place in this project where
choosing per state has measurable value. On tshirt_392 there is still nothing at all.

Part of 0.046 is selection noise: taking the best of seven macros at each state
captures some of the 0.010 spread. The test that separates a real state dependence
from that is whether the choice can be predicted, and over all 64 early states:

| Information set | Top-1 | Mean regret | Top-1 on shuffled labels |
|---|---:|---:|---:|
| constant | 0.42 | 0.023 | – |
| deployment observation | 0.44 | 0.009 | 0.37 (p95 0.42) |
| observation plus five-frame history | **0.47** | **0.007** | 0.30 (p95 0.38) |
| privileged state | 0.42 | 0.005 | 0.38 (p95 0.44) |

Only the history beats its own shuffled control on top-1; the single observation and
the privileged state do not. Top-1 barely moves while the regret falls by a factor of
three, so the models capture the value structure without picking the argmax. That is
the first evidence here for the partial-observability reading: telling these states
apart needs the history, not the frame.

The size, though, is what it is. Early the gap to a dressed arm is 0.443 and a perfect
per-state choice among these seven macros is worth 0.023 of it, about five percent.
Early or late, on either cell, the action at a state is worth a few percent of what
separates the policy from a dressed arm. Whatever dominates that gap is not the choice
of action at a state.

## Limits

Two garment/body cells so far, one policy, one seed block per cell, three repeats,
an eight-decision macro window and a 40-decision branch. The eight states of a
snapshot share a cell and differ only in their drape and observation seeds, so
within-cell state diversity is low by construction, and with two cells much of the
identifiability above may be garment recognition rather than state understanding.
Collections for `tshirt_68/14046` and `hospital_gown/14046` are running to widen
that. The macro library is seven fixed primitives; a better recovery outside it would
not appear. The return is sustained coverage over a 40-decision window, not episode
success.
