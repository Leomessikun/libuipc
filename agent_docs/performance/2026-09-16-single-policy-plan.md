# Single-policy dressing and the IPC actor interface

## Training status inspected on 2026-09-16

The stopped `launch_budget.sh` had resumed `abl_both_s1` toward 400,000 transitions,
using `.claude/worktrees/ablation-01bf913e`, not the current fresh IPC code.
At shutdown the latest logged training count was 134,832. Latest completed
held-out evaluation: 132,216 transitions, mean final upper-arm ratio .07730,
success .08. Best ratio in this run's evaluation CSV: .41703 at 110,256,
success .28. Separate checkpoint reevaluations are not these same CSV rows.
This is a dense/residual SAC regional run, not an IPC-hybrid dressing result.

`abl_both_s2` completed 125,016 transitions: final ratio .23660, success 0;
its best ratio is .28835 at 115,200. `stab_both_s1` starts from a checkpoint
scoring .44155/.28 but reaches zero ratio/success by 14,400; this weights-only
restart is not equivalent to continuing the original replay. These results
establish instability, not a diagnosis of its cause.

The old full-state replay IPC arm with nominal rho=1, seed 0, completed 20,032
transitions: return 12.44544, 21/64 successes. Sidecar availability and locality
still dilute replacement: this is not a pure-IPC actor experiment. Seed 2's
losing rho=.5 run does not establish that full replacement is intrinsically bad.
No new long matched fresh-actor run was found; previous fresh results are smoke
checks, and response-pretraining results are offline pilots. The owner requested
a stop; both launcher and training child were terminated, preserving artifacts.

## Simplification implemented

`pretrain_wang joint --regions ...` runs one policy over the selected live
regional distribution, with no teacher checkpoint and distillation weight zero.
It reuses the existing sampler, regional replay, optimizer, evaluation and resume.
The protocol is marked `joint_dressing`, distinct from reproducing Wang's
regional-teacher/student recipe. Dense/residual defaults are retained. Tests
exercise multi-region training without teacher loading and checkpoint resume.
All 17 pretrain launcher tests pass. No new native training was launched here.

Twenty-seven regions do not require twenty-seven policies. However, removing
teachers removes a learning aid too; joint generalization and wall-clock savings
must be measured. Preserve regional held-out metrics to detect interference.
Start with a small set of regions and garments. Curriculum expansion inside one
run is future work: current resume intentionally preserves the saved distribution.
Do not expand by accidentally dropping replay and optimizer state.

## Corrections to the proposed sparse IPC scheme

A detached target `a*=project(a0+delta*g_hat)` bounds the target displacement,
not the policy update. At the anchor, squared-error regression has parameter
gradient `-2 delta J_theta^T g_hat` before projection effects; ordinary SGD
therefore still induces `2 eta delta J_theta J_theta^T g_hat` in action space.
Normalization is a testable scale-control ablation, not a proven cure for seed
failures. Near-zero/noisy gradients should not be promoted to a fixed large step.
Actual action/KL movement must be measured; a genuine constraint needs an
accept/reject or constrained update, including optimizer-state handling.

Order collection -> fresh correction -> replay SAC to retain the same-action
contract. Running replay actor updates first changes the anchor even when the
same Gaussian noise is reused. `U-1 SAC + 1 physics` preserves the number of
actor steps, but replaces one SAC step: it is not the unchanged full SAC budget.
Compare against `U-1 replay SAC + 1 fresh SAC` to isolate teacher information;
retain the original U replay-SAC reference too. Use separate names for update
frequency `f_phys`, gradient mixture `rho_mix`, target radius `delta`, and loss
weight. Their numerical values are not interchangeable.

Reward-only outperforming Bellman is evidence against the current continuation
*usage*, not proof that its direction is wrong: norm, variance, horizon mismatch
and critic error remain alternatives. Prioritize these measurements before a
new value network. A raw-action target and existing detached gradient correction
should be compared at matched actual action movement.

## Research positioning

SVG and SAPO already combine analytic dynamics gradients and learned value tails;
MPO and related policy-improvement methods separate local improvement from policy
fitting. A hybrid alone is not an established novelty claim. The potential
contribution is a measured, solver-consistent and computationally useful interface
for deformable-contact off-policy learning. The infrastructure can be useful
independently of generic algorithm novelty.

Sources: [Wang RSS 2023](https://www.roboticsproceedings.org/rss19/p008.pdf),
[SVG](https://arxiv.org/abs/1510.09142),
[SAPO](https://arxiv.org/abs/2412.12089),
[MPO](https://arxiv.org/abs/1806.06920).
