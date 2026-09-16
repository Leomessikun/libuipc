# Actor-interface measurements, 2026-09-16

## Scope

These are native IPC **one-update probes**, not new learning curves or dressing
success measurements. Three existing `vec20k_s{0,1,2}_actor_trust` checkpoints,
two independent anchor batches per checkpoint, eight parallel slots per batch.
Every arm clones the same actor, critic and optimizer state. Saved Gaussian
noise reconstructs the actual sampled action. Eight-step discounted reward
rollouts restore the same simulator snapshot and reuse the same noise sequence.
There is no learned terminal value in the evaluation return. Policies are updated
once and then used throughout the rollout; this measures a policy change, not
only an isolated first-action perturbation.

The environment is the full-state, frictionless direct-picker `cloth_drag`
diagnostic, not visual cloth-body dressing. The network is the saved state MLP,
not the dressing dense/residual network. Discount .99, actor LR .0003, SiLU,
hidden width 128, no gradient clipping. The teacher retains the saved continuation
trust kappa=2: `g_R + exp(-2 d^2) g_C`, not an unweighted Bellman gradient.
The normalized action target radius is .03. These were fixed diagnostic choices,
not tuned on these results. Real IPC rollouts took 157s + 120s after world setup.

## Result 1: target fitting does not fix the update interface

Mean change in eight-step reward relative to the unchanged checkpoint:

| One update | Seed-0 checkpoint | Seed-1 checkpoint | Seed-2 checkpoint | Mean | Max action movement |
|---|---:|---:|---:|---:|---:|
| Fresh SAC | +.09222 | -.05895 | +.13505 | +.05611 | .69650 |
| SAC + gradient replacement, rho=.5 | +.13307 | -.03088 | +.23457 | +.11226 | .65623 |
| Raw IPC linear gradient, no SAC term | +.23193 | +.00522 | +.38389 | +.20701 | .66010 |
| Normalized/projected linear gradient | -.00436 | -.13720 | +.04836 | -.03107 | .16142 |
| Target regression, inherited Adam | -.00440 | -.13721 | +.04838 | -.03107 | .16142 |
| Target regression + actual displacement bound | -.00154 | -.03309 | +.00715 | -.00916 | .02994 |
| Random target direction | -.04057 | -.15974 | +.02995 | -.05679 | .16040 |
| Negative IPC target direction | -.05219 | -.17188 | +.01044 | -.07121 | .16368 |
| Reward-only target regression | -.00001 | -.13302 | +.04937 | -.02789 | .14794 |

Linear and squared-error target fitting produced **exactly identical anchor
actions in all six batches** (maximum difference 0). Their slight rollout
return differences are within simulator repeat variability: the largest
unchanged-policy repeat difference per slot was .0003144. The .03 target radius
allowed an actual .16142 movement, over five times larger. Bounding actual
movement shrank the failure but did not turn this branch into an improvement.

The raw and SAC-mix improvements are not movement-matched evidence that those
objectives are intrinsically better: their policy displacements were much larger.
Nor does the seed-1 one-update result diagnose why the old seed-2 training failed.

## Result 2: optimizer history and actual displacement are separate problems

Same checkpoint/anchor/noise construction; clear Adam state only for arms marked
fresh Adam. This is an ablation, not a proposal to reset Adam every training step.

| One update | Seed 0 | Seed 1 | Seed 2 | Mean reward change | Max movement |
|---|---:|---:|---:|---:|---:|
| Target, inherited Adam | -.00442 | -.13720 | +.04834 | -.03109 | .16141 |
| Target equals anchor (zero loss gradient), inherited Adam | -.02814 | -.15440 | +.02931 | -.05108 | .14292 |
| IPC target, fresh Adam | +.35720 | +.19710 | +.16634 | +.24021 | 1.06945 |
| IPC target, fresh Adam + bound | +.02328 | +.01237 | +.00533 | +.01366 | .02795 |
| Random target, fresh Adam | -.54905 | -.46607 | -.12166 | -.37893 | 1.05926 |
| Random target, fresh Adam + bound | -.01365 | -.00524 | -.00401 | -.00763 | .02909 |

The bounded IPC arm improved all six batch means; bounded random improved three
of six. Both share the same movement cap, but not exactly the same realized
movement. The mean cosine between actual movement and IPC teacher increased from
.0162 (inherited-Adam target) to .4523 (bounded fresh-Adam target). Clearing
history alone allows a very large step; it is not sufficient scale control.

A zero current gradient still moves the inherited-Adam policy because its first
moment is nonzero. This directly demonstrates contamination of the proposed
small correction by previous SAC/physics optimizer history. It does **not** prove
that this caused earlier learning-curve collapses. Shared network Jacobians,
action saturation and cross-sample coupling remain even with fresh moments.

The movement guard retries the *same* step at successively halved learning rates,
restoring both parameters and Adam state each time. If no trial passes, it restores
the original state. The bound applies only to sampled anchor actions; it is not
a global policy/KL or return-improvement guarantee. No return-based selection is
used to accept a step.

## Decision

Keep dense/residual SAC and the teacher-free `pretrain_wang joint` entry. Do not
promote normalized target regression with the shared inherited Adam state into
the training default. The promising next learning candidate must control both
optimizer-history interaction and actual action movement. An independent
correction optimizer is a candidate, not validated by these reset-optimizer
one-step probes; its accumulated history needs its own check. A matched hybrid
learning comparison must retain collection -> fresh step -> replay order and
match the number of SAC/fresh actor updates.

No new long training was started. These results justify investigating the actor
interface, but do not establish reliable learning, a faster dressing policy,
removal of all benefits of distillation, or a new RL algorithm. Differential
pretraining, response-Q, mechanical metrics and extra horizons remain parked.

## Follow-up implementation and smoke test

Commit `53c89d77` adds a transactional trust-radius option to fresh IPC actor
updates. With `--actor-step-radius r`, the update snapshots actor parameters and
optimizer state, retries an over-radius step with a halved learning rate, and
restores the snapshot if all retries fail. The default benchmark radius is .03;
setting it to zero disables this protection. The protection applies only to the
fresh IPC update and does not reset the ordinary SAC optimizer.

A native eight-slot, 256-transition smoke run with radius .03 completed in 12.8s:
24 fresh actor updates, 200 critic updates, no failed retries, and a measured
maximum fresh action step of .00212. This validates the runtime path; its eight
short evaluations are not a learning result. A dedicated unit test forces retry
and verifies the final step is at most .001.

## Artifacts and validation

- Reproducer: `python/uipc_manip/actor_interface_probe.py` (README gives commands).
- Raw results: `output/iaql/actor_interface_20260916/report.json` and
  `output/iaql/actor_momentum_20260916/report.json`; each directory preserves
  per-slot `rows.jsonl` and native `run.log`.
- [Tracked numerical summary](2026-09-16-actor-interface-summary.json) includes
  all per-checkpoint and per-batch means, movement and retry counts.
- 17 fresh-actor/probe tests passed, including first-step equivalence with Adam
  history, actual bound enforcement with optimizer restoration, and zero-gradient
  drift. Separately, all 17 teacher-free launcher/resume tests passed.
- These are three checkpoint identities, six batches, and 48 state/action anchors,
  not 48 independent training runs. Friction, held-out garments/bodies, visual
  observations and learning-time optimizer evolution are outside this experiment.
