# Why the same diagnostic says "learnable" on one task and not on dressing — 2026-09-19

Status: completed measurement. Two tasks, one solver, one machine, one protocol. No
policy was trained, no solver, reward or controller setting was changed.

This is the control that every negative result in this project has been missing. The
counterfactual branch diagnostic has only ever been run on dressing, where every method
has also failed, so its verdict could not be told apart from a diagnostic that always
says no. Running it unchanged on a task the same code base demonstrably learns settles
that, and the comparison turns out to say something sharper than the control was built
for.

Scripts: `scripts/cloth_drag_branches.py` (the control task's collection),
`scripts/consequence_to_noise.py` (the comparison). Artifacts:
`output/iaql/drag_branches_20260919/full/` and
`output/uipc_manip/consequence_to_noise_20260919.json`.

## The control task

The cloth-drag benchmark of the closed IAQL study: the same libuipc CUDA backend, the
same snapshot and restore, a three-dimensional action, and a SAC policy whose return
rises from -4 to +13 over 20,000 transitions. Reinforcement learning works there.

The protocol is the dressing one, with the task's own directions as the macro library
(toward the goal, away, two perpendicular to it, up, the policy, the policy at the
macros' magnitude): snapshot a state the policy visits, run a macro for eight
decisions, hand back to the same policy for thirty-two, repeat every macro three times
with identical commands. 24 states, 504 branches, 20,800 decisions, 7.2 minutes.

## Result 1: the simulator is irreproducible on both tasks, but they were not run alike

Repeating one macro against itself with bitwise identical commands from a restored
state never reproduces, on either task. On cloth drag the largest spread of a repeated
macro's final distance is 5.5e-5 m and **no** group of 168 is bitwise identical, so the
source — floating-point reductions whose order is not fixed — is present in both.

**The two tasks do not run the solver at the same settings, and this record originally
said they did.** The library defaults are a Newton velocity tolerance of 0.05 m/s and a
conjugate-gradient relative tolerance of 1e-3 (`src/core/core/scene_default_config.cpp`).
The dressing environment loosens both for speed, to 0.1 and 1e-2
(`python/uipc_manip/dressing_env.py`); the cloth-drag environment tightens the Newton
tolerance to 0.001 and leaves the linear tolerance at the library default
(`python/uipc_manip/iaql_env.py`). The control task therefore runs at a hundred times
the Newton tolerance and ten times the linear tolerance of the task it is being
compared with.

So amplification is not the whole story and may not be most of it: the task on which
reinforcement learning works is also the task that was run converged.
`2026-09-19-tolerance-sets-the-noise-floor.md` measures what that setting alone is
worth, and a branch run at matched tolerances is what decides between the two
explanations. The dressing predictability record, which measured the amplification
directly, was itself taken at the loose setting
(`2026-09-18-predictability-horizon.md`).

## Result 2: the separation is in the top of the ranking, not in the variance

With the *same* outcome statistic on both tasks — the worst progress over a branch's
last twelve decisions — the naive signal-to-noise reading does not separate them, and
saying so matters more than the headline:

| task | states | consequence | noise floor | eta² | F | margin / sd | margin < sd |
|---|---|---|---|---|---|---|---|
| cloth drag (RL succeeds) | 24 | +0.0000 | 0.0000 | 1.000 | 7.2e7 | **193.6** | 8 % |
| dressing tshirt_26 | 48 | +0.0137 | 0.0068 | 0.862 | 14.5 | **1.31** | 42 % |
| dressing tshirt_392 | 48 | +0.0067 | 0.0073 | 0.959 | 54.1 | **1.24** | 42 % |

`eta²` is the share of outcome variance explained by which macro was chosen, `F` the
same decomposition scaled by the within-macro noise, `margin / sd` the gap between the
best macro's mean and the second best's over the standard deviation of one macro's own
repeats.

Dressing's *consequence is larger* than the control task's, and its `eta²` is high:
telling a bad intervention from a good one is easy there. What is not easy is telling
the best from the next best, and that is the only comparison that improves a policy
which is already competent. On that comparison the two tasks are 150 times apart, and
at 42 % of dressing states the margin is below the noise of a single repeat, against
8 % on the control.

The ranking statistic says the same thing independently: agreement between independent
repeats on which macro won is 1.00 (top-1 and pairwise) on cloth drag and 0.47 on
dressing, where chance over seven macros is 0.14.

## What this rules out

* **"The diagnostic always says no."** It does not. On the control it reports 14 of 24
  states decisive, perfect rank agreement, and five different macros winning at
  different states — the state-dependence signature dressing lacks.
* **"Dressing has no decision structure."** It has plenty: `eta²` 0.86-0.96. The
  structure is not visible at the resolution that matters.
* **"The obstacle is the reward, the metric, or partial observability."** None of those
  enter here. The outcome is measured directly from the simulator, the same way on both
  tasks, with no policy in the loop except as the continuation.

It does **not** rule out that the whole difference is the solver tolerance, because the
two tasks were not run at the same one. That is now the leading explanation rather than
a caveat.

## What it does not yet establish

That the two tasks differ for any reason other than the tolerance they were run at.
The comparison above is confounded by it and cannot be read as a property of dressing
until a dressing branch run at the control's numerics is in hand.

Whether the seed itself is removable is a separate and still-open question. It comes
from floating-point reductions whose order is not fixed — two sites are located
(`atomicAdd` on the traversal counter in `stackless_bvh.inl`, the segmented and block
reductions in `spmv.cu`) — but tightening the tolerance, which needs no code change at
all, turns out to move the noise floor by two to three orders of magnitude on its own,
so the kernel edit is not the next thing to try.

## Reproduce

```bash
python scripts/cloth_drag_branches.py --checkpoint output/iaql/vec20k_s0_sac/online_beta_0.0.pt \
  --snapshot-steps 20,50,80 --window 8 --follow 32 --repeats 3 --slots 8 --out <dir>
python scripts/analyze_decision_branches.py <dir> --initial-key distance --success-threshold -0.01
python scripts/consequence_to_noise.py \
  --run "cloth drag=<dir>:sustained_coverage" \
  --run "dressing tshirt_26=output/uipc_manip/decision_branches_20260918/t26_14046,output/uipc_manip/decision_branches_early_20260919/t26_14046:sustained_coverage"
```
