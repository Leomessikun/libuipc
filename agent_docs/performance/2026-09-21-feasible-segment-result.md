# Completed early-segment continuation experiment

Date: 2026-09-21. This closes the run predeclared in
[event-boundary analysis, Section 6](2026-09-21-event-boundary-analysis.md).
The design and success threshold were not changed after observing the results.
No additional simulation was launched by this continuation of the research.

## Design and completion

One tshirt_26/14046 cell; warm-SAC checkpoint `checkpoint_00127416.pt`; snapshot
at decision 40; eight seeded slots; two repeated continuations per intervention.
Three arms: policy, lift and outward. An eight-decision intervention is followed
by 250 decisions of the same policy, ending at decision 298.

`output/uipc_manip/feasible_segment_episode_20260921/result.json` has
`completed=true`, all 48 records and all 258 decisions per record. All seven
recorded restore position errors are zero. Total reported runtime is
**1,111.773 s (18.53 minutes)**, below the 3,600 s cap. The 12,384 planned branch
decisions were executed; the logged total is **12,704**, including 320 approach
decisions. Restore-position equality does not establish deterministic future
solver execution or coupling of the random disturbances across arms.

## Results

| Intervention | Valid in first 8 decisions | Valid over all 258 decisions | Mean sustained coverage | Mean validity-weighted sustained coverage | Valid success, coverage >= .7 |
|---|---:|---:|---:|---:|---:|
| policy | 2/16 | 0/16 | .556605 | .000000 | 0/16 |
| lift | 16/16 | 5/16 | .531905 | .163085 | 0/16 |
| outward | 16/16 | 16/16 | .468254 | .468254 | 0/16 |

Sustained coverage is the **minimum** upper-arm ratio over the last 12 decisions.
Validity means every recorded per-decision maximum anchor tracking error is <=
0.02 m. Validity-weighted coverage is `mean(valid * sustained)`, with invalid
branches assigned zero; it is not the average conditional on validity. These are
existing evaluation metrics, not accumulated training reward.

Final upper-arm coverage means are .558316, .538193 and .469195, respectively.
Sustained ranges are .483794–.644515, .442426–.596906 and .423274–.509628.
Thus even the unconstrained final coverage threshold is unmet in every arm.
There is no selected success hidden behind a mean.

Both repeats agree on the main ordering. Policy validity is 0/8 and 0/8;
lift is 3/8 and 2/8; outward is 8/8 and 8/8. Outward's sustained coverage is
.464102 and .472406 across the two repeats. Its validity-weighted coverage exceeds
the matched policy in all 16 state/repeat pairs. These are descriptive matched
contrasts, not 16 independent garment/body trials or a population confidence claim.

## What the predeclared alternatives actually imply

The result meets **both** of two predeclared conditions, which were not mutually
exclusive:

1. A valid arm obtains greater validity-weighted coverage: outward preserves the
   tracked constraint and achieves substantial late progress. Early preservation
   is not inevitably permanent stalling; the early .003 coverage does not predict
   the final .468 coverage.
2. All arms have zero valid success: the tested one-off interventions plus this
   continuation do not complete the task. This does **not** identify earlier
   decisions as the cause; later control, intervention duration, the restricted
   action library and measurement semantics remain possible explanations.

The policy violates the criterion in every branch (first violation at relative
steps 4–10, median 5). Lift passes all eight-step screens but subsequently violates
it in 11/16 branches, first at relative steps 17–21, median 18 among violations.
Therefore an eight-step successful check is not a long-horizon feasibility
certificate. Outward has no observed violation through the recorded horizon.

The experiment supports a **constraint/progress trade-off with useful long-horizon
partial progress**. It does not demonstrate a new policy, a policy-learning
improvement, a feasible completion, or the general superiority of outward.

## Two limitations found in the collector audit

**Prefix validity is missing.** The collector stores tracking maxima only after
restoring the decision-40 snapshot. `dressing_env.step` initializes the tracking
maximum anew each decision. A valid instantaneous snapshot cannot rule out a
violation during decisions 1–39. Consequently, the positive counts above certify
only the recorded **branch**, not the whole episode. Zero branch-valid successes
still implies zero successes under the stricter whole-episode conjunction.
Future full-episode certification must log and carry the approach-prefix maximum.

**The new approach is not the previous collection's exact state.** It uses the
same cell/checkpoint/seed design but reruns the approach in a nondeterministic
simulation. Within-run arms share the snapshot; between-run comparisons need not.
Do not treat changes from 17/24 to 5/16 as a clean horizon-only causal effect. The
11/16 delayed lift violations are established from this run's own traces.

The proxy tests anchor tracking, not physical finger/object separation or human
contact safety. Neither a 2 cm excursion nor its absence alone establishes either
real grasp loss or real safety. Coverage geometry and the known shoulder metric
issue also need independent validation before any full-task success claim.

## A useful budget baseline, not a new algorithm

For the absorbing evaluation objective `valid * terminal_coverage`, the first
violation makes the eventual score exactly zero. Stopping each recorded branch at
its first violation would require **5,707 instead of 12,384 branch decisions**, a
53.9% reduction in decision count on these traces. This is retrospective arithmetic,
not measured wall-clock saving: eight-slot batching, inactive-slot cost, snapshot
scheduling and learning data needs can prevent proportional speedup. It does not
apply to unmasked reward or a task that permits recovery after a violation.
The failure itself must remain in the data; dropping failed samples changes the
sampling distribution. This known absorbing-state baseline should precede an
elaborate query allocator.

## Reproduce and integrity checks

```bash
python3 scripts/analyse_feasible_episode.py \
  output/uipc_manip/feasible_segment_episode_20260921/result.json \
  --out output/uipc_manip/feasible_segment_episode_20260921/audit.json
```

The CPU-only script rejects incomplete runs, duplicate/missing design cells,
truncated/nonfinite traces, and discrepancies between trace-recomputed and stored
summaries. It reports per-repeat outcomes and descriptive paired contrasts.
Source SHA-256:
`2b8e573a71123743731f334b797eecfb8dcbf021559d335563c2b5197a3b8c1a`.
