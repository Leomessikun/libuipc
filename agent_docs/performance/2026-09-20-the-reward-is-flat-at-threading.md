# The reward is flat exactly where the sleeve has to go on — 2026-09-20

Correction: the [inventory evidence review](2026-09-20-training-issue-review.md)
provides an actual-function counterexample to this document's flatness argument.
Equal reward values at a junction do not imply zero slope. This contaminated run
also cannot establish a converged critic, perfect state information or a causal
learning bottleneck. The historical interpretation below is not accepted evidence.

Status: measurement from an interrupted run, with a caveat recorded below. The run it
comes from was contaminated and has been relaunched; the numbers here are read from the
evaluation stream, which is monotone in step and internally consistent.

## Why this run existed

Every negative result on dressing is confounded with a 5,383-float point cloud: the
35-float privileged state is critic-only by design and no actor has ever read it
(`handoff`, 2026-09-19). The feasibility question — does "train dressing fast on one
workstation" have any hope — needs to know what the control problem costs when
perception is free. So: a state actor and a privileged critic, trained and evaluated on
the same 25 cells, nothing held out, the most generous setting there is.

## What the evaluations show

| vector step | task reward, final | forearm ratio, peak | **upper-arm ratio, peak** | grasp valid | return |
|---:|---:|---:|---:|---:|---:|
| 500 | -0.4866 | 0.0282 | **0.0000** | 0.64 | -268.8 |
| 1000 | -0.4615 | 0.0000 | **0.0000** | 1.00 | -253.2 |
| 1500 | -0.2250 | 0.0000 | **0.0000** | 1.00 | -157.8 |
| 2000 | -0.1523 | 0.0042 | **0.0000** | 1.00 | -96.6 |
| 2500 | -0.0858 | 0.0048 | **0.0000** | 0.84 | -86.5 |
| 3000 | -0.0295 | 0.0333 | **0.0000** | 1.00 | -58.5 |
| 3500 | -0.0419 | 0.0261 | **0.0000** | 1.00 | -58.1 |
| 4000 | -0.1883 | 0.0000 | **0.0000** | 1.00 | -134.4 |
| 4500 | -0.0800 | 0.0000 | **0.0000** | 1.00 | -88.4 |

Twenty-five episodes per row, one per cell. The reward improves by a factor of six, the
policy learns to stop tearing the grip off (0.64 to 1.00), and the upper-arm coverage is
**exactly zero at every evaluation on every cell**. Peak forearm progress never exceeds
0.033 of a forearm.

## Where the reward is flat

`dressing_reward.py::wang_progress`, the Wang RSS 2023 reference term:

```
opening not on the arm:  task = -|finger - opening_center|     upper bound exactly 0
opening on the forearm:  task = forearm_distance               starts at 0
opening on the upper arm: task = forearm_len + 5 * upperarm_distance
```

The approach term rises to zero as the opening reaches the fingertip. Threading starts
at zero, because forearm progress is measured back from the finger. The two meet at the
same value, so **the reward is flat across the threading event**: nothing in it
distinguishes an opening hovering at the fingertip from one that has just gone on, and
there is no gradient carrying a learner through.

The measured behaviour is that plateau exactly: the task reward converges to -0.03 to
-0.08 — an opening held three to eight centimetres from the fingertip — and stays.
This is the same plateau the counterfactual branch study found around a good action
(`2026-09-18-recovery-decisions.md`), located this time in the objective rather than in
the outcome.

## What it does and does not establish

It **does** establish that with free perception, a converged critic and a generous
budget, seventy-five thousand transitions of this objective produce zero upper-arm
coverage on twenty-five cells, and that the objective has no slope at the one event the
task is about.

It does **not** establish that the reward is the only obstacle, that a shaped reward
would learn, or that Wang's published result is wrong — that work may start the opening
already at the hand, where the plateau is never crossed by learning at all. The
relaunched clean run carries the same objective to 270,000 transitions and will say
whether the plateau is escaped later.

## Caveat on provenance

Two training processes were launched against the same run directory (a kill that
targeted the shell, not its child), so `eval_log.csv` and the checkpoints of
`state_upper_bound_s1` interleave two runs and must not be used. The table above is read
from the stdout evaluation stream, whose steps are monotone 500-4500 with no repeats.
The directory is kept as `state_upper_bound_s1_CONTAMINATED`; the clean rerun is
`state_ub_clean_s1`.
