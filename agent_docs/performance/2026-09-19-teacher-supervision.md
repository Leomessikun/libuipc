# Repairing the scripted teacher, and which of its failures is worth acting on — 2026-09-19

Status: the baseline and the first supervisor are measured; the second is running.
Nothing is trained and no solver, reward or controller setting is changed. The question
is whether the teacher's measured failures can be repaired by overriding it, because
every imitation route in this project is capped by what the teacher does.

Code: `uipc_manip.dressing_supervisor`, `scripts/evaluate_supervised_teacher.py`,
`scripts/compare_teacher_arms.py`, `scripts/separability_of_failure_signals.py`,
`expert_baseline --supervised`. Artifacts under
`output/uipc_manip/supervised_teacher_20260919/`.

## The baseline, and what its failures are

Both arms run the scripted expert over the same 25 cells (five garments × bodies
14045–14049) from the same seeds for 300 decisions, under the corrected
five-centimetre shoulder ray. The unsupervised arm reproduces the figure the expert
dataset recorded: **9 successes of 25** under coverage-and-grasp, 15 cells dressed to
0.7 coverage, 10 cells with a whole-episode valid grasp, mean sustained coverage
0.6985, mean peak tracking error 2.24 cm.

Its 16 failures split three ways:

| Failure | Cells | What happens |
|---|---:|---|
| Dressed but the grip slipped | 6 | sustained coverage 0.96–1.00, peak tracking 2.26–3.66 cm |
| Never reaches the arm | 6 | peak coverage below 0.3; four of the five `tshirt_392` bodies |
| Partial | 4 | peak coverage 0.3–0.7 |

The six grip failures are the largest single lever: those episodes dress the arm
completely and are disqualified by the tracking error alone. And the grip goes early —
the error crosses 2 cm at decisions 111–130, when coverage is still 0.03–0.18, long
before the sleeve is on the upper arm, and never recovers.

## The first supervisor made it worse, and why

A supervisor with three rules — override while the sleeve stalls on the arm, while the
opening's centre is past 0.6 of its own radius from the centreline, or while the
tracking error exceeds 1.2 cm — scored **0 of 25**, broke nine cells and fixed none:
mean sustained coverage 0.0785 against 0.6985. Its override counts explain it: the
centring rule fired 13–17 times an episode and each firing held eight decisions, so
roughly a third of every episode was the supervisor's; the grasp rule fired on up to
195 of 300 decisions.

It did do what it was built for on its own terms: whole-episode grasp validity rose
from 10 of 25 to 21 of 25, and mean peak tracking fell from 2.24 cm to 1.46 cm. It kept
the grip and stopped the dressing.

## Which signals actually separate success from failure

The error was choosing thresholds from failed episodes without checking their base rate
in successful ones. Measured over the teacher's own 25 episodes, by the fraction of
decisions each condition fires on:

| Signal | In the 9 successful episodes | In the 16 failed ones |
|---|---:|---:|
| tracking error > 1.0 cm | 3.4 % | 47.0 % |
| tracking error > 1.2 cm | 0.5 % | 45.4 % |
| tracking error > 1.5 cm | **0.0 %** | **40.2 %** |
| containment > 0.4 radii | 13.0 % | 46.5 % |
| containment > 0.6 radii | 5.7 % | 15.0 % |
| containment > 0.8 radii | 0.0 % | 0.6 % |
| arc progress < 0.005 per 20 decisions | 35.7 % | 33.8 % |
| arc progress < 0.02 per 20 decisions | 42.1 % | 49.2 % |

Medians tell the same story: tracking 0.246 cm against 0.824 cm, containment 0.185
against 0.354, arc progress 0.039 against 0.021.

**Only the grasp separates.** Above 1.5 cm the successful episodes never go, and the
failed ones spend two fifths of their decisions there. Containment separates weakly and
only in a band that is common in both. **A stall does not separate at all**: by any
threshold, a sleeve that is not advancing is as ordinary in a successful dressing as in
a failed one. That also explains the earlier finding that the counterfactual branch
study's "stalled" states were not special, and it retires the stall rule.

## What the second supervisor changes

Only the grasp rule is enabled, and its response is no longer to scale the whole
command down. It projects the teacher's translation onto the arm's own axis, keeping
what advances the sleeve and dropping what pulls the cloth across the arm, on the
reasoning that the strain comes from the transverse pull while the dressing comes from
the axial one. The run is in `matrix25_v2`; its result belongs in this record.

## Limits

One teacher, 25 cells, one seed per cell, one horizon. The separability table is
measured on the same 25 episodes whose failures suggested the rules, so it is a
consistency check rather than an independent test; a threshold chosen on it should be
validated on other cells. The grasp criterion itself is the project's 2 cm
whole-episode rule, which disqualifies an otherwise complete dressing for a single
early slip.
